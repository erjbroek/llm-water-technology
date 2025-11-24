
# Configuration for the RAG chatbot project.
# Override values here to change settings in the codebase.


from pathlib import Path

# Model configuration
MODEL_NAME = "deepseek-r1:8b"
COLLECTION_NAME = "water_management_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Logging configuration
LOG_LEVEL = "WARNING"  # Only show warnings and errors 
# LOG_LEVEL = "DEBUG"  # Show everything including debug info

# Gradio configuration
GRADIO_SHARE = True
GRADIO_DEBUG = True

# File and directory configuration
DATA_ROOT = (Path(__file__).parent / "Water management research papers").resolve()
CHROMA_PATH = (Path(__file__).parent / "vector_store").resolve()

# Document processing configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

# DATA_ROOT.mkdir(parents=True, exist_ok=True)
# CHROMA_PATH.mkdir(parents=True, exist_ok=True)