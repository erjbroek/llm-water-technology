
## File Structure

- **`config.py`** - Configuration 
- **`vectorize.py`** - Vector store management and document processing
- **`backend.py`** - RAG chatbot logic and file processing utilities
- **`frontend.py`** - Gradio user interface
- **`main.py`** - Main entry point to run the application

## Key Features

- Document upload and processing (PDF, TXT, DOCX)
- Vector-based document retrieval using ChromaDB
- Integration with Ollama for LLM responses
- Gradio web interface for easy interaction
- Hierarchical folder structure preservation
- Document management (add, delete, reindex)

## Setup

1. Create a virtual enviornemnt:

python -m venv .venv

2. Activate the virtual environment:

.venv\Scripts\activate

3. Install the requirements:

pip install -r requirements.txt

4.  Run the code:

python main.py

5. Open the Gradio URL printed in the terminal and interact with the app.

## Architecture

### config.py
Contains all configuration constants including:
- Model names and settings
- File paths and directories
- Document processing parameters
- Gradio settings

### vectorize.py
Handles:
- Document text extraction (PDF, TXT, DOCX)
- Text chunking and splitting
- Vector embeddings generation
- ChromaDB vector store management
- Document fingerprinting for change detection

### backend.py
Provides:
- RAG chatbot implementation
- File upload and processing logic
- Document management operations
- Integration with Ollama for response generation
- Conversation history management

### frontend.py
Creates:
- Gradio web interface
- File upload controls
- Chat interface
- Document management UI
- Database statistics display

## Usage

1. Upload documents using the file upload interface
2. Documents are automatically processed and stored in the vector database
3. Ask questions in in the chat interface
4. The system automatically detects your language and responds appropriately:
   - English questions → English responses
   - Dutch questions → Dutch responses
5. The system retrieves relevant document chunks and generates responses using Ollama


