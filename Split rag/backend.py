# Backend module for RAG Chatbot.
# Contains the main RAG chatbot logic, file processing utilities, and integration with Ollama.

import os
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import torch
from ollama import Client
# from deep_translator import GoogleTranslator
# from langdetect import detect
from vectorize import VectorStore, DocumentProcessor, ingest_file, ingest_directory
import config
import asyncio
import json
from ragas.metrics import BleuScore, RougeScore, ChrfScore
from ragas.dataset_schema import SingleTurnSample
from bert_score import score
from sentence_transformers import CrossEncoder


# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, 'INFO'))
logger = logging.getLogger(__name__)

# Initialize global components
try:
    # Initialize the Ollama client
    ollama_client = Client()
    logger.info("Ollama client initialized!")
    
    # GPU diagnostics
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU device: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
    else:
        logger.info("Running on CPU")

    
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Initialize components
    vector_store = VectorStore()
    doc_processor = DocumentProcessor()
    
    # Auto-ingest documents from DATA_ROOT if it exists
    if config.DATA_ROOT.exists():
        logger.info(f"Ingesting documents from {config.DATA_ROOT} ...")
        ingest_directory(config.DATA_ROOT, vector_store, doc_processor)
    else:
        logger.info(f"Data root not found: {config.DATA_ROOT}")
        
except Exception as e:
    logger.error(f"Error during backend initialization: {e}")
    vector_store = None
    doc_processor = None
    ollama_client = None


class RAGChatbot:
    # RAG Chatbot that combines document retrieval with LLM generation.
    
    def __init__(self, vector_store, ollama_client, reranker_model):
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        self.reranker_model = reranker_model
        self.conversation_history = []
        # self.user_language = None  
        # self.translator_cache = {}  

    # def detect_language(self, text: str) -> str:
    #     """Detect the language of the input text."""
    #     try:
    #         return detect(text)
    #     except Exception as e:
    #         logger.error(f"Language detection error: {e}")
    #         return "en"  # Default to English

    # def get_translator(self, source: str, target: str) -> GoogleTranslator:
    #     """Get or create a cached translator instance."""
    #     key = f"{source}->{target}"
    #     if key not in self.translator_cache:
    #         self.translator_cache[key] = GoogleTranslator(source=source, target=target)
    #     return self.translator_cache[key]

    # def translate_to_english(self, text: str) -> str:
    #     """Translate text to English if it's not already in English."""
    #     try:
    #         lang = self.detect_language(text)
    #         if lang != "en":
    #             logger.info(f"Translating query from {lang} to English")
    #             translator = self.get_translator("auto", "en")
    #             return translator.translate(text)
    #         return text
    #     except Exception as e:
    #         logger.error(f"Translation error: {e}")
    #         return text

    # def translate_from_english(self, text: str, target_lang: str) -> str:
    #     """Translate text from English to target language."""
    #     try:
    #         if target_lang != "en":
    #             logger.info(f"Translating response from English to {target_lang}")
    #             translator = self.get_translator("en", target_lang)
    #             return translator.translate(text)
    #         return text
    #     except Exception as e:
    #         logger.error(f"Translation error: {e}")
    #         return text

    def get_relevant_context(self, query):
        # Retrieve relevant document chunks for the query.
        try:
            results = self.vector_store.search(query, top_k=5)  
            docs = results["documents"][0]
            return "\n\n".join(docs)
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return ""
        
    # def get_relevant_context(self, query):
    #     top_k = 5 
    #     confidence_score_threshold = 0.8 
    #     try:
    #         results = self.vector_store.search(query, top_k=top_k)
    #         retrieved_chunks = results["documents"][0]
            
    #         if not retrieved_chunks:
    #             print("No chunks retrieved from vector store.")
    #             return ""

    #         # Reranking using Cross-Encoder
    #         # Making pairs of (query, chunk) and scoring them with the Cross-Encoder
    #         pairs = [(query, chunk) for chunk in retrieved_chunks]
    #         scores = self.reranker_model.predict(pairs)
            
    #         # Add the score to the chunks and sort by score from high to low
    #         scored_chunks = sorted(
    #             zip(retrieved_chunks, scores), 
    #             key=lambda x: x[1], 
    #             reverse=True
    #         )

    #         # 3. Selection with confidence threshold
    #         final_context = []
    #         for chunk, score in scored_chunks:
    #             if score >= confidence_score_threshold:
    #                 final_context.append(chunk)

    #         if not final_context:
    #             # print(f"No chunks found above confidence treshold {confidence_score_threshold}, using top 5 instead")
    #             # final_context = [chunk for chunk, score in scored_chunks[:5]]
    #             return ""

    #         print(f"DEBUG: {len(final_context)} chunks selected after reranking.")
    #         return "\n---\n".join(final_context)
            
    #     except Exception as e:
    #         print(f"Retrieval/Reranking error: {e}")
    #         return ""


    def generate_response(self, query: str):
        # Generate a response using retrieved context and Ollama.
        
        # # Use cached language or detect once
        # if self.user_language is None:
        #     self.user_language = self.detect_language(query)
        #     logger.info(f"Detected query language: {self.user_language}")
        # original_lang = self.user_language
        
        # Search using original query (documents stored in original language)
        context = self.get_relevant_context(query)
        
        # Build conversation history
        history = "\n".join([
            f"Q: {h['question']}\nA: {h['answer']}" for h in self.conversation_history[-3:]
        ])

        # # Determine response language instruction
        # if original_lang == "nl":
        #     lang_instruction = "Always respond in Dutch (Nederlands)."
        # elif original_lang == "en":
        #     lang_instruction = "Always respond in English."
        # else:
        #     # For other languages, try to respond in that language
        #     lang_instruction = f"Always respond in the same language as the user's question."

        # Create prompt

#         prompt = f"""
# You are a helpful assistant answering based ONLY on the uploaded document context.
# If unsure, say so.
# Focus on providing accurate information from the context provided.

# Conversation history:
# {history}

# Relevant document context:
# {context}

# User question: {query}

# Answer:
# """

        prompt = f"""
        You are an intelligent assistent helping users find precise information from research documents.
        You must answer questions based ONLY using information proviced in the document context.
        If the answer is not clearly supported in this context, respond with:
        "Based on the available documents, i cannot find the answer"

        Conversation history: 
        {history}

        Relevant document context:
        {context}

        Task:
        Answer the following question as clearly and consisely as possible using only the document context above.
        when answering:
        - Use evidence directly from the context.
        - Do NOT speculate, infer missing details or use outside knowledge.
        - Be clear and neural, focus on accuracy.

        If you cannot find an answer, say so.

        user question:
        {query}
        """

        try:
            response = self.ollama_client.chat(
                model="qwen3:4b",
                messages=[{"role": "user", "content": prompt}],
                stream=False  
            )
            logger.debug(f"Ollama raw response: {response}")

            # Extract only clean message text
            answer = ""
            if hasattr(response, "message") and hasattr(response.message, "content"):
                answer = response.message.content
            elif isinstance(response, dict):
                message = response.get("message", {})
                if isinstance(message, dict):
                    answer = message.get("content", "")
                elif "content" in response:
                    answer = response["content"]
            else:
                answer = str(response)

            # Strip leading/trailing whitespace
            answer = answer.strip()

            # Remove <think>...</think> sections if present
            if "<think>" in answer:
                try:
                    answer = answer.split("</think>")[-1].strip()
                except Exception:
                    pass

            # # Skip language verification for English (performance optimization)
            # # Only verify/translate if user language is not English
            # if original_lang != "en":
            #     try:
            #         response_lang = self.detect_language(answer)
            #         if response_lang != original_lang:
            #             logger.info(f"LLM responded in {response_lang}, translating to {original_lang}")
            #             translator = self.get_translator("auto", original_lang)
            #             answer = translator.translate(answer)
            #     except Exception as e:
            #         logger.warning(f"Could not verify/fix response language: {e}")

            # Add to conversation history
            self.conversation_history.append({
                "question": query, 
                "answer": answer  
            })
            
            return answer

        except Exception as e:
            error_msg = f" Error from Ollama: {e}"
            # # Translate error message if needed (use cached translator)
            # if original_lang != "en":
            #     try:
            #         translator = self.get_translator("en", original_lang)
            #         error_msg = translator.translate(error_msg)
            #     except:
            #         pass
            return error_msg
            
    def clear_history(self):
        # Clear conversation history.
        self.conversation_history = []
        # self.user_language = None  # Reset language cache
        # self.translator_cache = {}  # Clear translator cache
        logger.info("Conversation history cleared")

class RAGEvaluator:
    def __init__(self, chatbot):
        self.chatbot = chatbot
        evaluation_questions_and_answers = json.loads(open("questions_answers.json", "r", encoding="utf-8").read())
        prepared_qa_nl = []
        prepared_qa_en = []
        for qa_pair in evaluation_questions_and_answers:
            if qa_pair["ground_truth"]["nl"].strip():
                prepared_qa_nl.append({
                    "question": qa_pair["question"]["nl"],
                    "ground_truth": qa_pair["ground_truth"]["nl"],
                    "project": qa_pair["project"],
                    "filename": qa_pair["filename"],
                    "specificity": qa_pair["specificity"]
                })
            if qa_pair["ground_truth"]["en"].strip():
                prepared_qa_en.append({
                    "question": qa_pair["question"]["en"],
                    "ground_truth": qa_pair["ground_truth"]["en"],
                    "project": qa_pair["project"],
                    "filename": qa_pair["filename"],
                    "specificity": qa_pair["specificity"]
                })
        self.prepared_qa_nl = prepared_qa_nl[:1]
        self.prepared_qa_en = prepared_qa_en[:1]
        print(f"loaded json succesfully: {len(self.prepared_qa_nl)} nl, and {len(self.prepared_qa_en)} en qa pairs")
    
    def store_relevant_documents(self, evaluation_data=[]):
        if evaluation_data:
            for qa_pair in evaluation_data:
                filename = qa_pair["filename"]
                directory_name = qa_pair["project"]
                try:
                    file_path = next((Path.cwd().parent / "Water management research papers").rglob(filename), None)
                except:
                    print("File not found. This could be due to the json. Start by verifying if the file is actually in the directory")
                    raise FileNotFoundError(f"File '{filename}' not found in project '{directory_name}'")
                
                filetype = filename.split('.')[-1]
                chunks = doc_processor.process_document(file_path, filetype)
                vector_store.add_documents(chunks, filename)
            
    def evaluate_retrieval(self, evaluation_data=[], top_k=5):
        if evaluation_data:
            hits = 0
            for qa_pair in evaluation_data:
                context = self.chatbot.get_relevant_context(qa_pair["question"])
                gt_words = [word.lower() for word in qa_pair["ground_truth"].split()[:5]]
                if any(word in context.lower() for word in gt_words):
                    hits += 1
            recall_at_k = hits / len(evaluation_data)
            print(f"Recall@{top_k}: {recall_at_k}")
        else:
            print("No evaluation data found.")
            return

    async def evaluate_generation(self, evaluation_data=[]):
        if evaluation_data:
            print('started evaluating')
            for qa_pair in evaluation_data:
                ground_truth = qa_pair["ground_truth"]
                generated_response = self.chatbot.generate_response(query=qa_pair["question"])

                print('before sample')
                sample = SingleTurnSample(
                    response=generated_response,
                    reference=ground_truth
                )
                bleu = await BleuScore().single_turn_ascore(sample)
                rouge = await RougeScore().single_turn_ascore(sample)
                chrf = await ChrfScore().single_turn_ascore(sample)

                precision, recall, F1 = score([generated_response], [ground_truth], lang="en", verbose=True)

                print(f"bleu score: {bleu}")
                print(f"rouge score: {rouge}")
                print(f"chrf score: {chrf}")
                print(f"precision: {precision}, recall: {recall}, F1: {F1}")

        else:
            print("No evaluation data found")

# Initialize chatbot
chatbot = RAGChatbot(vector_store, ollama_client, reranker_model) if vector_store and ollama_client else None
evaluator = RAGEvaluator(chatbot)


def resolve_target_folder(subfolder: str) -> Path:
    # Decide where to save uploaded files.
    subfolder = (subfolder or "").strip()
    
    if not subfolder:
        target = config.DATA_ROOT
    else:
        target = (config.DATA_ROOT / subfolder).resolve()
    
    # Make sure path is safe (stays inside our folder)
    if not str(target).startswith(str(config.DATA_ROOT)):
        raise ValueError("Target folder must be within the data root directory")
    
    target.mkdir(parents=True, exist_ok=True)
    return target


def resolve_document_path(relative_path: str) -> Path:
    # Turn document name into full path.
    if not relative_path:
        raise ValueError("No document selected")
    
    candidate = (config.DATA_ROOT / relative_path).resolve()
    
    if not str(candidate).startswith(str(config.DATA_ROOT)):
        raise ValueError("Document path must stay within the managed data directory")
    
    return candidate

def evaluate_model():
    print('storing chunks')
    # evaluator.store_relevant_documents(evaluator.prepared_qa_en)
    # evaluator.evaluate_retrieval(evaluator.prepared_qa_en)
    print('starting evaluation')
    asyncio.run(evaluator.evaluate_generation(evaluator.prepared_qa_en))
    # TODO: look into if the db should be cleared after evaluation? or finding a way to clear just the added info

def get_db_stats():
    # Get database statistics.
    if not vector_store:
        return "Vector store not initialized"
        
    stats = vector_store.get_collection_stats()
    if "error" in stats:
        return f"DB Error: {stats['error']}"
    return f"""
Vector DB Stats:
- Documents: {stats['total_documents']}
- Chunks: {stats['total_chunks']}
- Model: {stats['embedding_model']}
- Device: {stats['device']}
Documents: {stats['documents']}
"""


def upload_and_process_files(files, target_subfolder):
    # Handle file uploads from UI.
    if not files:
        return "No files uploaded!", get_db_stats()

    if not vector_store or not doc_processor:
        return "Backend not properly initialized!", get_db_stats()

    try:
        destination_dir = resolve_target_folder(target_subfolder)
    except ValueError as exc:
        return str(exc), get_db_stats()

    messages = []
    
    for file in files:
        try:
            temp_path = Path(file.name)
            suffix = temp_path.suffix.lower()
            
            if suffix not in config.SUPPORTED_EXTENSIONS:
                messages.append(f"Skipped {temp_path.name}: unsupported file type")
                continue

            # Copy file to destination folder
            original_name = getattr(file, "orig_name", temp_path.name)
            dest_path = destination_dir / Path(original_name).name
            shutil.copyfile(temp_path, dest_path)
            
            # Add to database
            ingested, msg = ingest_file(dest_path, config.DATA_ROOT, vector_store, doc_processor, force=True)
            
            if ingested:
                messages.append(msg)
            else:
                messages.append(f"No change: {dest_path.relative_to(config.DATA_ROOT)}")
            
        except Exception as exc:
            display_name = getattr(file, "orig_name", getattr(file, "name", "uploaded file"))
            messages.append(f"Failed {display_name}: {exc}")

    return "\n".join(messages), get_db_stats()


def reindex_all_documents():
    # Re-index everything from scratch.
    if not vector_store or not doc_processor:
        return "Backend not properly initialized!", get_db_stats()
        
    messages = ingest_directory(config.DATA_ROOT, vector_store, doc_processor, force=True)
    return "\n".join(messages), get_db_stats()


def refresh_documents():
    # Just update the UI without changing database.
    return get_db_stats()


def delete_document(document_id: str, remove_file: bool):
    # Delete a document from database and optionally from disk.
    if not vector_store:
        return "Vector store not initialized", get_db_stats()
        
    try:
        doc_path = resolve_document_path(document_id)
    except ValueError as exc:
        return str(exc), get_db_stats()

    # Remove from database
    removed_vectors = vector_store.delete_document(document_id)
    
    # Delete file from disk if requested
    file_message = ""
    if remove_file and doc_path.exists() and doc_path.is_file():
        try:
            doc_path.unlink()
            file_message = " and source file"
        except Exception as exc:
            file_message = f" but failed to delete file ({exc})"
    
    if removed_vectors:
        status = f"Removed {document_id} from vector DB{file_message}"
    else:
        status = f"Document {document_id} not found in vector DB"
    
    return status, get_db_stats()


def chat_response(message, history):
    # Handle chat messages from the UI.
    if not chatbot:
        return history, ""
        
    if not message.strip():
        return history, ""
        
    response = chatbot.generate_response(message)
    
    results = vector_store.search(message, top_k=1)
    filename = results['metadatas'][0][0]['relative_path'].split('\\')[-1]
    chunk = results['documents'][0][0].replace('\n', ' ')

    combined_response = (
        f"{response}\n\n"
        f"**File used**:\n {filename}\n"
        f"**Information:**\n: {chunk}"
    )

    history.append([message, combined_response])
    return history, ""


def clear_all_data():
    # Delete everything: database and chat history.
    if vector_store:
        vector_store.clear_collection()
    if chatbot:
        chatbot.clear_history()
    return "All data cleared!", [], get_db_stats()


def clear_chat_only():
    # Clear just the chat, keep documents.
    if chatbot:
        chatbot.clear_history()
    return [], ""


def list_documents():
    # Get list of all documents in the vector store.
    if not vector_store:
        return []
    return vector_store.list_documents()