# Vectorization code for Water Management Research Papers
# This code creates and manages a vector database for water management research documents.

import os
import logging
import uuid
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import torch
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

# Configure logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, 'INFO'))
logger = logging.getLogger(__name__)


class DocumentProcessor:
    # A helper class that extracts text from uploaded documents and
    # splits the extracted text into chunks suitable for embedding and storage.

    def __init__(self):
        # Configure chunking with overlap to preserve context across chunk boundaries
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            # chunk_size=1500,
            # chunk_overlap=300,

            # high chunk size 
            # chunk_size=2000,
            # chunk_overlap=400,

            # high overlap
            # chunk_size=1000,
            # chunk_overlap=400,

            
            # chunk_size=2000,
            # chunk_overlap=200,
        )

    def extract_text_from_pdf(self, file_path: Path) -> str:
        # Extract text from a PDF file path.
        text = ""
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            logger.info(f"Extracted {len(text)} chars from PDF: {file_path}")
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
        return text

    def extract_text_from_txt(self, file_path: Path) -> str:
        # Read and return the contents of a plain text file encoded as UTF-8.
        text = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            logger.info(f"Extracted {len(text)} chars from TXT: {file_path}")
        except Exception as e:
            logger.error(f"Error reading TXT: {e}")
        return text
    
    def extract_text_from_docx(self, file_path: Path) -> str:
        # Extract text from a .docx file and return it as a single string.
        text = ""
        try:
            doc = Document(str(file_path))
            for para in doc.paragraphs:
                text += para.text + "\n"
            logger.info(f"Extracted {len(text)} chars from DOCX: {file_path}")
        except Exception as e:
            logger.error(f"Error reading DOCX: {e}")
        return text

    def process_document(self, file_path: Path):
        # Dispatch to the correct extractor and split the resulting text.
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            text = self.extract_text_from_pdf(file_path)
        elif suffix == ".txt":
            text = self.extract_text_from_txt(file_path)
        elif suffix == ".docx":
            text = self.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        if not text.strip():
            raise ValueError("No text could be extracted from document")

        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks


class VectorStore:
    # Simple vector store wrapper around ChromaDB plus a SentenceTransformer
    # embedding model.

    def __init__(self):
        # Pick device automatically
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL, device=self.device
        )

        # Initialize a persistent Chroma client
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
        self.collection_name = config.COLLECTION_NAME
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )
        logger.info(f"Vector store initialized on {self.device}, storage: {config.CHROMA_PATH}")

    def _encode_chunks(self, chunks: List[str]):
        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)
        return embeddings.tolist()

    def compute_fingerprint(self, file_path: Path) -> Dict[str, Any]:
        # Get file info to check if it changed later
        stat = file_path.stat()
        return {
            "source_mtime": stat.st_mtime,
            "source_size": stat.st_size,
        }

    def build_base_metadata(self, file_path: Path, root_dir: Path) -> Dict[str, Any]:
        # Save info about where this file is located
        relative_path = file_path.relative_to(root_dir)
        
        # Get the folder name (empty string if file is in root directory)
        if relative_path.parent != Path('.'):
            folder_name = relative_path.parent.as_posix()
        else:
            folder_name = ""
        
        return {
            "source_path": str(file_path.resolve()),
            "relative_path": relative_path.as_posix(),
            "folder": folder_name,
            "filename": relative_path.name,
            "extension": file_path.suffix.lower().lstrip('.'),
        }

    def upsert_document(
        self,
        document_id: str,
        chunks: List[str],
        base_metadata: Dict[str, Any],
        fingerprint: Dict[str, Any],
        force: bool = False,
    ) -> Tuple[bool, str]:
        # Add or update a document in the database.
        
        if not chunks:
            return False, f"Skipping {document_id}: no content"

        # Check if document already exists in database
        where_filter = {"document_id": document_id}
        existing = self.collection.get(where=where_filter, include=["metadatas"], limit=1)
        already_exists = bool(existing.get("ids"))
        
        # Skip processing if file hasn't changed (unless force=True is specified)
        if already_exists and not force:
            old_meta = existing["metadatas"][0] or {}
            old_mtime = old_meta.get("source_mtime")
            old_size = old_meta.get("source_size")
            new_mtime = fingerprint.get("source_mtime")
            new_size = fingerprint.get("source_size")
            
            if old_mtime == new_mtime and old_size == new_size:
                return False, f"Unchanged: {document_id}"

        # Remove old version if updating existing document
        if already_exists:
            self.collection.delete(where=where_filter)

        # Turn text chunks into embeddings (vectors) using the embedding model
        embeddings = self._encode_chunks(chunks)
        
        # Create unique ID for each chunk within the document
        ids = []
        for i in range(len(chunks)):
            chunk_id = f"{document_id}::chunk::{i}-{uuid.uuid4().hex[:8]}"
            ids.append(chunk_id)
        
        # Attach metadata to each chunk (document info + chunk index)
        metadata_template = {**base_metadata, **fingerprint, "document_id": document_id}
        metadatas = []
        for i in range(len(chunks)):
            chunk_meta = {**metadata_template, "chunk_index": i}
            metadatas.append(chunk_meta)

        # Save everything to the vector database
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        return True, f"Indexed {len(chunks)} chunks from {document_id}"

    def delete_document(self, document_id: str) -> bool:
        # Remove all chunks for this document
        deleted = self.collection.delete(where={"document_id": document_id})
        return bool(deleted and deleted.get("ids"))

    def list_documents(self) -> List[str]:
        # Get list of all documents in database
        data = self.collection.get(include=["metadatas"])
        
        # Extract unique document IDs from metadata
        documents = set()
        for meta in data.get("metadatas", []):
            if meta and meta.get("document_id"):
                documents.add(meta.get("document_id"))
        
        # Return sorted list of document IDs
        return sorted(documents)

    def search(self, query: str, top_k: int = 5):
        # Search the vector collection for the most relevant chunks to query.
        if self.collection.count() == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        logger.info(f"Search for '{query}' retrieved {len(results['documents'][0])} chunks")
        return results

    def get_collection_stats(self):
        try:
            count = self.collection.count()
            all_results = self.collection.get(include=["metadatas"]) if count > 0 else {"metadatas": []}
            docs = {
                meta.get("document_id", meta.get("source", "Unknown"))
                for meta in all_results.get("metadatas", [])
                if meta
            }
            return {
                "total_chunks": count,
                "total_documents": len(docs),
                "documents": sorted(docs),
                "embedding_model": config.EMBEDDING_MODEL,
                "device": self.device,
            }
        except Exception as e:
            return {"error": str(e)}

    def clear_collection(self):
        # Delete and recreate the collection to clear all stored vectors and metadata.
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(name=self.collection_name)
            logger.info("Vector collection cleared")
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")


def iterate_supported_files(root_dir: Path) -> List[Path]:
    # Find all PDF, DOCX, and TXT files in the folder
    if not root_dir.exists():
        logger.warning(f"Data directory does not exist: {root_dir}")
        return []
    
    files = []
    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in config.SUPPORTED_EXTENSIONS:
            files.append(path)
    
    return files


def ingest_file(
    file_path: Path,
    root_dir: Path,
    vector_store: VectorStore,
    processor: DocumentProcessor,
    force: bool = False,
):
    # Take one file and add it to the database
    # Extract text and split into chunks
    chunks = processor.process_document(file_path)
    
    # Create a unique ID for this document
    document_id = file_path.relative_to(root_dir).as_posix()
    
    # Save information about this file
    metadata = vector_store.build_base_metadata(file_path, root_dir)
    fingerprint = vector_store.compute_fingerprint(file_path)
    
    # Add it to the vector database
    return vector_store.upsert_document(
        document_id=document_id,
        chunks=chunks,
        base_metadata=metadata,
        fingerprint=fingerprint,
        force=force,
    )


def ingest_directory(
    root_dir: Path,
    vector_store: VectorStore,
    processor: DocumentProcessor,
    force: bool = False,
):
    # Go through all files in the folder and add them to the database
    messages = []
    
    # Get all supported file types from the specified directory
    all_files = iterate_supported_files(root_dir)
    
    # Process each file one by one and add to vector database
    for file_path in all_files:
        try:
            ingested, msg = ingest_file(
                file_path, root_dir, vector_store, processor, force=force
            )
            logger.info(msg)
            if ingested:
                messages.append(msg)
        except Exception as exc:
            error_msg = f"Failed to ingest {file_path}: {exc}"
            logger.error(error_msg)
            messages.append(error_msg)
    
    # If nothing was processed, inform the user
    if not messages:
        messages.append("No new or updated documents found")
    
    return messages


def copy_and_ingest_from_source(
    source_dir: Path,
    target_dir: Path,
    vector_store: VectorStore,
    processor: DocumentProcessor,
    force: bool = False,
):
    # Copy documents from source directory to target directory and ingest them
    messages = []
    
    if not source_dir.exists():
        error_msg = f"Source directory does not exist: {source_dir}"
        logger.error(error_msg)
        return [error_msg]
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all supported files from source
    all_files = iterate_supported_files(source_dir)
    
    if not all_files:
        logger.info(f"No supported files found in {source_dir}")
        return ["No supported files found in source directory"]
    
    logger.info(f"Found {len(all_files)} files to process from {source_dir}")
    
    # Process each file
    for source_file in all_files:
        try:
            # Calculate relative path to preserve folder structure
            relative_path = source_file.relative_to(source_dir)
            target_file = target_dir / relative_path
            
            # Create target subdirectory if needed
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file if it doesn't exist or is different
            should_copy = True
            if target_file.exists():
                source_stat = source_file.stat()
                target_stat = target_file.stat()
                # Only copy if source is newer or different size
                if (source_stat.st_mtime <= target_stat.st_mtime and 
                    source_stat.st_size == target_stat.st_size and not force):
                    should_copy = False
            
            if should_copy:
                shutil.copy2(source_file, target_file)
                logger.info(f"Copied: {relative_path}")
                messages.append(f"Copied: {relative_path}")
            
            # Ingest the file into vector database
            ingested, msg = ingest_file(
                target_file, target_dir, vector_store, processor, force=force
            )
            
            if ingested:
                messages.append(f"Vectorized: {msg}")
            
        except Exception as exc:
            error_msg = f"Failed to process {source_file}: {exc}"
            logger.error(error_msg)
            messages.append(error_msg)
    
    return messages


if __name__ == "__main__":
    # Standalone script to vectorize documents from the copy folder into the main folder.
    # This preserves the folder structure and updates the vector database.
    import argparse
    
    parser = argparse.ArgumentParser(description="Vectorize documents from Water management research papers copy")
    parser.add_argument("--source", 
                       default="Water management research papers copy",
                       help="Source directory containing documents to vectorize")
    parser.add_argument("--target",
                       default="Water management research papers", 
                       help="Target directory where documents will be stored")
    parser.add_argument("--force", 
                       action="store_true",
                       help="Force reprocessing of all documents")
    
    args = parser.parse_args()
    
    # Setup paths - resolve to absolute paths
    script_dir = Path(__file__).parent
    source_dir = (script_dir / args.source).resolve()
    target_dir = (script_dir / args.target).resolve()
    
    logger.info("="*60)
    logger.info("Starting document vectorization process")
    logger.info(f"Source: {source_dir}")
    logger.info(f"Target: {target_dir}")
    logger.info(f"Force reprocess: {args.force}")
    logger.info("="*60)
    
    try:
        # Initialize components - vector store and document processor
        logger.info("Initializing vector store and document processor...")
        vector_store = VectorStore()
        processor = DocumentProcessor()
        
        # Copy and ingest documents from source to target directory
        logger.info("Processing documents...")
        messages = copy_and_ingest_from_source(
            source_dir=source_dir,
            target_dir=target_dir,
            vector_store=vector_store,
            processor=processor,
            force=args.force
        )
        
        # Display results from processing
        logger.info("="*60)
        logger.info("PROCESSING RESULTS:")
        logger.info("="*60)
        for message in messages:
            print(message)
        
        # Show final statistics about the vector database
        stats = vector_store.get_collection_stats()
        logger.info("="*60)
        logger.info("VECTOR DATABASE STATISTICS:")
        logger.info(f"Total documents: {stats.get('total_documents', 0)}")
        logger.info(f"Total chunks: {stats.get('total_chunks', 0)}")
        logger.info(f"Embedding model: {stats.get('embedding_model', 'Unknown')}")
        logger.info(f"Device: {stats.get('device', 'Unknown')}")
        logger.info("="*60)
        
        logger.info("Vectorization completed successfully!")
        
    except Exception as e:
        logger.error(f"Vectorization failed: {e}")
        raise