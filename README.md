# DocuMind AI 📚

A Multi-PDF Retrieval-Augmented Generation (RAG) Chatbot built using LangChain, ChromaDB, HuggingFace Embeddings, Gemini, and Streamlit.

## Features

- Multi-PDF Upload
- Semantic Search
- ChromaDB Vector Storage
- HuggingFace Embeddings
- Gemini 2.5 Flash Integration
- Source Attribution
- Chat History
- Streamlit UI

## Tech Stack

- Python
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Gemini API
- Streamlit

## Architecture

PDF
→ Text Extraction
→ Chunking
→ Embeddings
→ ChromaDB
→ Similarity Search
→ Gemini
→ Answer

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
