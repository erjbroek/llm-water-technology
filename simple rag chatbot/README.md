# Document-Based RAG Chatbot

# This repository implements a simple Retrieval-Augmented Generation (RAG) chatbot that answers questions based on uploaded documents. It uses:

- SentenceTransformer for embeddings (`all-MiniLM-L6-v2`)
- ChromaDB for vector storage
- Ollama LLM for generation
- Gradio for the frontend UI

## Quick start

1. Create a virtual enviornemnt:

python -m venv .venv

2. Activate the virtual environment:

.venv\Scripts\activate

3. Install the requirements:

pip install -r requirements.py

4.  Run the code:

python rag.py

5. Open the Gradio URL printed in the terminal and interact with the app.

