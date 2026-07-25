# ResearchLens
# Research Lens

ResearchLens is an AI-powered research assistant that lets users upload and interact with their own documents.

## Current Features

- Upload multiple PDF, TXT, and CSV files
- Extract and clean document content
- Semantic text chunking
- BGE embeddings
- FAISS vector search
- Retrieval-Augmented Generation (RAG)
- Grounded answers using Groq Llama
- Source and page citations
- Multi-document question answering

## Tech Stack

- Python
- Gradio
- LangChain
- BAAI/bge-small-en-v1.5
- FAISS
- Groq / Llama
- PyMuPDF
- Sentence Transformers

## How It Works

Documents → Text Extraction → Chunking → Embeddings → FAISS → Semantic Retrieval → Groq LLM → Grounded Answer

## Supported Files

- PDF
- TXT
- CSV

## Status

ResearchLens is currently an MVP. The next phase focuses on cloud deployment, persistent document storage, authentication, and an improved research workspace.
