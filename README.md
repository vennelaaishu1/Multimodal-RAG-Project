# RAG Document Q&A System

## Project Description

This project is a Retrieval-Augmented Generation (RAG)
application that allows users to upload PDF documents
and ask questions about their content.

The system retrieves relevant information from the
uploaded documents and uses an LLM to generate answers.

## Architecture

PDF
↓
Document Loading
↓
Text Chunking
↓
Embeddings
↓
FAISS Vector Database
↓
Similarity Search
↓
Retrieved Context
↓
LLM
↓
Generated Answer

## Technologies

- Python
- LangChain
- Hugging Face
- Sentence Transformers
- FAISS
- Streamlit
- PyPDF

## Features

- PDF document upload
- Text extraction
- Document chunking
- Embedding generation
- Vector database
- Similarity search
- Context retrieval
- LLM-based answer generation

## How to Run

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py