# TurboVec RAG Pipeline

## Overview

This project is a fully local Retrieval-Augmented Generation (RAG) system designed for document-based question answering. It uses TurboVec for vector storage, LangGraph for orchestration, and large language models for response generation.

The system allows users to query their own documents efficiently without relying on external vector databases or cloud storage. All embeddings are generated locally, and the index is persisted on disk for reuse.

## Architecture

1. Documents are loaded from a local directory  
2. Text is split into chunks using LangChain text splitters  
3. Embeddings are generated using a local SentenceTransformer model  
4. Vectors are stored in a TurboVec index  
5. A retriever fetches the most relevant chunks  
6. LangGraph orchestrates retrieval and generation  
7. A language model generates the final answer based on retrieved context  

## Features

- Fully local vector database using TurboVec  
- No dependency on external vector database services  
- Persistent index storage for fast reloads  
- LangGraph-based modular pipeline  
- Support for custom document ingestion  
- Privacy-preserving design with local embeddings  
- Extensible architecture for additional nodes and workflows  

## Project Structure

turbovec-rag/
├── rag_pipeline.py
├── scripts/
│   └── build_index.py
├── data/
│   ├── sample_docs/
│   └── index.tv
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

## Installation

git clone https://github.com/RVSVarma/turbovec-rag.git
cd turbovec-rag

python -m venv .venv

Windows:
.venv\Scripts\activate

macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

## Configuration

Copy environment file:
copy .env.example .env

Add API keys if required:
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

## Adding Documents

Place .txt files inside:
data/sample_docs/

## Building the Index

python scripts/build_index.py --rebuild

This performs:
- Loading documents
- Chunking text
- Generating embeddings
- Building TurboVec index
- Saving index to disk

## Running the System

python rag_pipeline.py

Example:

Question: What are the house rules?

Answer: The house rules include noise restrictions after 10 PM and pet approval requirements.

## Configuration

- Chunk size: 500
- Chunk overlap: 50
- Top-k retrieval: 5
- Index path: data/index.tv
- Embeddings: SentenceTransformers

## Technology Stack

Python, LangChain, LangGraph, TurboVec, SentenceTransformers

## Future Improvements

- FastAPI backend
- Web UI
- Hybrid search
- Docker deployment
- PDF support

## License

MIT

## Author

Venkata Skandha Rajendra Varma
Master’s in Data Science and AI
