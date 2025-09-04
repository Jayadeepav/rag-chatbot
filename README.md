🚀 RAG Chatbot (Open Source, CPU-Friendly)

This project implements a Retrieval-Augmented Generation (RAG) chatbot, allowing you to upload documents, generate embeddings, and query them using a local LLM.

🔧 Tech Stack

Streamlit → Interactive UI for chatting and document upload

ChromaDB → Vector database to store and retrieve document embeddings

HuggingFace Embeddings → BAAI/bge-small-en-v1.5 for high-quality text embeddings

FAISS (alternative to ChromaDB, optional) → Fast similarity search and retrieval

Phi-2 GGUF (via llama.cpp) → Lightweight, local LLM for answering queries

PyPDF2 & Text loaders → For PDF and TXT document ingestion and preprocessing

📂 Features

Upload PDFs or text files

Automatic document ingestion and chunking

Generate embeddings using HuggingFace models

Store embeddings in ChromaDB / FAISS

Query the documents using a local Phi-2 model (no API needed)

CPU-friendly setup (runs locally without GPU)


Created with ❤️ by Jayadeepa V\
✉️deepa1283023@gmail.com\
Happy Coding!✌️
