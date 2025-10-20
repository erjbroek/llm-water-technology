from pathlib import Path
from typing import List, Dict, Any
import os
import uuid
import torch
import chromadb
import gradio as gr
from sentence_transformers import SentenceTransformer
from ollama import Client
import PyPDF2
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =================================================
# Setup Ollama & Torch
# =================================================
# Initialize the Ollama client used for forwarding prompts to a local LLM server.
# The Client object is expected to provide a `.chat(...)` method later in the code.
ollama_client = Client()
print("Ollama client initialized!")

# Quick diagnostic messages about whether a CUDA GPU is available. This helps
# us usunderstand if embeddings and other models will run on the GPU or CPU.
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
else:
    print("Running on CPU")

# =================================================
# Document Processor
# =================================================
class DocumentProcessor:
    """
    A small helper class that extracts text from uploaded documents and
    splits the extracted text into chunks suitable for embedding and storage.
    """

    def __init__(self):
        # Configure chunking: 1000 characters per chunk with 200 characters
        # overlap. Overlap helps preserve context across chunk boundaries when
        # retrieving. The `length_function` uses `len` (character length).
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file path.

        Returns a concatenated string of all pages. Returns an empty string on
        failure and prints a diagnostic message.
        """
        text = ""
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    # extract_text() may return None for some PDFs
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            print(f"Extracted {len(text)} chars from PDF: {file_path}")
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text

    def extract_text_from_txt(self, file_path: str) -> str:
        """Read and return the contents of a plain text file encoded as UTF-8."""
        text = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            print(f"Extracted {len(text)} chars from TXT: {file_path}")
        except Exception as e:
            print(f"Error reading TXT: {e}")
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Extract text from a .docx file and return it as a single string.
        Paragraphs are joined with newline characters. On error an empty
        string is returned and the exception is printed for debugging.
        """
        text = ""
        try:
            # Load the Word document using python-docx
            doc = Document(file_path)
            # Iterate over paragraphs and accumulate their text
            for para in doc.paragraphs:
                text += para.text + "\n"
            # Debug/logging: show how many characters were extracted
            print(f"Extracted {len(text)} chars from DOCX: {file_path}")
        except Exception as e:
            # Print the error and return empty string on failure
            print(f"Error reading DOCX: {e}")
        return text

    def process_document(self, file_path: str, file_type: str):
        """Dispatch to the correct extractor and split the resulting text.

        Raises ValueError if the file type is unsupported or if no text is
        extracted from the document.
        """
        if file_type == "pdf":
            text = self.extract_text_from_pdf(file_path)
        elif file_type == "txt":
            text = self.extract_text_from_txt(file_path)
        elif file_type == "docx":
            text = self.extract_text_from_docx(file_path)    
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        if not text.strip():
            raise ValueError("No text could be extracted from document")

        chunks = self.text_splitter.split_text(text)
        print(f"✂ Split into {len(chunks)} chunks. Example: {chunks[:2]}")
        return chunks

doc_processor = DocumentProcessor()

# =================================================
# Vector Store (ChromaDB + embeddings)
# =================================================
class VectorStore:
    """
    Simple vector store wrapper around ChromaDB plus a SentenceTransformer
    embedding model.

    Responsibilities:
    - Create a Chroma collection and manage adding/querying documents.
    - Produce embeddings for text chunks using SentenceTransformers.

    Notes:
    - The constructor deletes any existing collection with the same name so
      running the script multiple times starts with a clean DB by default.
    - SentenceTransformer is instantiated with the device set to cuda if
      available which speeds up embedding generation on GPUs.
    """

    def __init__(self):
        # Pick device automatically. SentenceTransformer accepts a `device`
        # argument (e.g. 'cpu' or 'cuda').
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2", device=self.device
        )

        # Initialize a local Chroma client and create a fresh collection for
        # storing documents and their embeddings.
        self.chroma_client = chromadb.Client()
        self.collection_name = "ai_ml_documents"
        try:
            # Attempt to delete existing collection of same name. If it does
            # not exist this will raise an error
            self.chroma_client.delete_collection(name=self.collection_name)
        except:
            pass
        self.collection = self.chroma_client.create_collection(name=self.collection_name)
        print(f"Vector store initialized on {self.device}")

    def add_documents(self, chunks, source_file: str):
                """
                Add a list of text chunks to the vector collection along with their
                embeddings and metadata.

                Args:
                        chunks: A list of strings (the text chunks) to embed and store.
                        source_file: The originating filename. This is stored in metadata
                                                 to trace which document each chunk came from.

                Implementation details:
                - We create unique IDs per chunk using the source filename, index and
                    a short uuid fragment to avoid collisions across uploads.
                - SentenceTransformers can return either numpy arrays or torch
                    tensors; we request non-tensor outputs and convert to lists for
                    Chroma's API which expects Python-native sequences.
                """
                embeddings = self.embedding_model.encode(chunks, convert_to_tensor=False)
                ids = [f"{source_file}{i}{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
                metadatas = [{"source": source_file, "chunk_id": i} for i in range(len(chunks))]
                # Chroma expects Python lists for embeddings, it calls `.tolist()` if
                # embeddings are numpy arrays.
                self.collection.add(
                        documents=chunks,
                        embeddings=embeddings.tolist(),
                        metadatas=metadatas,
                        ids=ids,
                )
                print(f"Added {len(chunks)} chunks from {source_file}")

    def search(self, query: str, top_k: int = 5):
        """
        Search the vector collection for the most relevant chunks to `query`.

        Returns Chroma's raw `query` result which contains the retrieved
        documents, metadatas, distances, etc. The caller expects the
        `documents` field to exist and to be indexable as shown in
        `RAGChatbot.get_relevant_context`.
        """
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        print(f"Search for '{query}' retrieved {len(results['documents'][0])} chunks")
        return results

    def get_collection_stats(self):
        try:
            count = self.collection.count()
            all_results = self.collection.get() if count > 0 else {"metadatas":[]}
            sources = {m.get("source","Unknown") for m in all_results["metadatas"]}
            return {
                "total_chunks": count,
                "total_documents": len(sources),
                "sources": list(sources),
                "embedding_model": "all-MiniLM-L6-v2",
                "device": self.device,
            }
        except Exception as e:
            return {"error": str(e)}

    def clear_collection(self):
        """
        Delete and recreate the collection to clear all stored vectors and
        metadata. This is intended for development/demo
        purposes only.
        """
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(name=self.collection_name)
            print("Vector collection cleared")
        except Exception as e:
            print(f"Error clearing collection: {e}")

vector_store = VectorStore()

# =================================================
# RAG Chatbot (retrieval + Ollama LLM)
# =================================================
class RAGChatbot:
    def __init__(self, vector_store, ollama_client):
        self.vector_store = vector_store
        self.ollama_client = ollama_client
        self.conversation_history = []

    def get_relevant_context(self, query):
        try:
            results = self.vector_store.search(query, top_k=5)
            docs = results["documents"][0]
            return "\n\n".join(docs)
        except Exception as e:
            print(f"Retrieval error: {e}")
            return ""

    def generate_response(self, query: str):
        context = self.get_relevant_context(query)
        history = "\n".join([
            f"Q: {h['question']}\nA: {h['answer']}" for h in self.conversation_history[-3:]
        ])

        prompt = f"""
You are a helpful assistant answering based ONLY on the uploaded document context.
If unsure, say so.

Conversation history:
{history}

Relevant document context:
{context}

User question: {query}
"""

        try:
            response = self.ollama_client.chat(
                model="qwen3:4b",
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            print("DEBUG Ollama raw response:", response)
    
            # Ollama's client can return different shapes depending on the
            # configuration; try to handle common ones robustly.
            if isinstance(response, dict):
                if "message" in response and isinstance(response["message"], dict):
                    # response['message'] is often a dict with a 'content' field.
                    answer = response["message"].get("content", str(response["message"]))
                elif "content" in response:
                    answer = response["content"]
                else:
                    answer = str(response)
            else:
                # Fallback to stringifying non-dict responses.
                answer = str(response)

            # Record the Q/A pair in conversation history for limited
            # conversational state; only the last 3 turns are included in the
            # prompt when generating a new response (see `history` above).
            self.conversation_history.append({"question": query, "answer": answer})

            return answer.strip()

        except Exception as e:
            return f"Error from Ollama: {e}"
            
    def clear_history(self):
        self.conversation_history = []
        print("Conversation history cleared")

chatbot = RAGChatbot(vector_store, ollama_client)

# =================================================
# Helper Functions (Gradio)
# =================================================
def get_db_stats():
    stats = vector_store.get_collection_stats()
    if "error" in stats:
        return f"DB Error: {stats['error']}"
    return f"""
Vector DB Stats:
- Documents: {stats['total_documents']}
- Chunks: {stats['total_chunks']}
- Model: {stats['embedding_model']}
- Device: {stats['device']}
Sources: {stats['sources']}
"""

def upload_and_process_files(files):
    """
    Handler wired to the Gradio file upload button. `files` is a list of
    uploaded file objects that have `name` and a temporary path accessible by
    the worker process. For each file we:

    - Determine the extension.
    - Extract and chunk text using `DocumentProcessor`.
    - Add chunks to the vector store with the original filename as the
      metadata `source` so it can be shown in DB stats.

    Returns a tuple (status_message, db_stats_text) which map to the two
    Gradio outputs bound in the UI.
    """
    if not files:
        return "No files uploaded!", get_db_stats()
    messages = []
    for file in files:
        # `file.name` is used both for display and as the `source` metadata.
        ext = file.name.split(".")[-1].lower()
        chunks = doc_processor.process_document(file.name, ext)
        vector_store.add_documents(chunks, file.name)
        messages.append(f"{file.name} ({len(chunks)} chunks)")
    return "\n".join(messages), get_db_stats()

def chat_response(message, history):
    """
    Gradio callback for sending messages to the chatbot. `history` is the
    chat UI state represented as a list of [user, assistant] rows. We return
    an updated history and an empty string to clear the text input box.
    """
    if not message.strip():
        return history, ""
    response = chatbot.generate_response(message)
    history.append([message, response])
    return history, ""

def clear_all_data():
    vector_store.clear_collection()
    chatbot.clear_history()
    return "All data cleared!", [], get_db_stats()

# =================================================
# Gradio Interface
# =================================================
with gr.Blocks(title="Document Q&A Assistant") as demo:
    # Top-level UI: a two-column layout with file upload + controls on the
    # left and the chat interface on the right.
    gr.Markdown("# Document-Based AI Q&A")

    with gr.Row():
        with gr.Column(scale=1):
            # File upload accepts PDF, and TXT. Gradio provides file
            # objects which include `name` and a temporary file path.
            file_upload = gr.Files(file_types=[".pdf", ".txt", ".docx"], file_count="multiple")
            upload_btn = gr.Button("Process Files", variant="primary")
            upload_status = gr.Textbox(label="File Status")
            db_stats = gr.Textbox(label="DB Stats", value=get_db_stats())
            clear_btn = gr.Button("Clear All Data")
            refresh_btn = gr.Button("Refresh Stats")

        with gr.Column(scale=2):
            # Chat UI on the right: conversation history and input box.
            chatbot_ui = gr.Chatbot(label="Chatbot", height=500)
            msg_box = gr.Textbox(label="Your Question")
            send_btn = gr.Button("Send")
            clear_chat_btn = gr.Button("Clear Chat")

    # Wire UI controls to the functions defined above.
    upload_btn.click(upload_and_process_files, file_upload, [upload_status, db_stats])
    send_btn.click(chat_response, [msg_box, chatbot_ui], [chatbot_ui, msg_box])
    msg_box.submit(chat_response, [msg_box, chatbot_ui], [chatbot_ui, msg_box])
    clear_btn.click(clear_all_data, outputs=[upload_status, chatbot_ui, db_stats])
    refresh_btn.click(get_db_stats, outputs=[db_stats])
    clear_chat_btn.click(lambda: ([], "Chat cleared!"), outputs=[chatbot_ui, upload_status])

if __name__ == "__main__":
    print("Launching Gradio...")
    demo.launch(share=True, debug=True)