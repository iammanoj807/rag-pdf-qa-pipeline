"""
RAG Pipeline - Main Entry Point

This script runs the full RAG pipeline:
1. Load PDFs from the data directory
2. Split documents into chunks
3. Generate embeddings
4. Store in ChromaDB vector store
5. Ask questions using LLM
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

from pipeline.document_loader import process_all_pdfs
from pipeline.text_splitter import split_documents
from pipeline.embedding_manager import EmbeddingManager
from pipeline.vector_store import VectorStore
from pipeline.retriever import RAGRetriever
from pipeline.rag_chain import ask_question


def ingest_documents(data_dir: str = "./data"):
    """
    Run the ingestion pipeline: Load → Split → Embed → Store.

    Args:
        data_dir: Path to the directory containing PDF files

    Returns:
        Tuple of (vector_store, embedding_manager) for querying
    """
    print("=" * 60)
    print("📥  STEP 1: Loading PDF Documents")
    print("=" * 60)
    all_documents = process_all_pdfs(data_dir)

    print("\n" + "=" * 60)
    print("✂️  STEP 2: Splitting Documents into Chunks")
    print("=" * 60)
    chunks = split_documents(all_documents)

    print("\n" + "=" * 60)
    print("🧠  STEP 3: Generating Embeddings")
    print("=" * 60)
    embedding_manager = EmbeddingManager()
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    print("\n" + "=" * 60)
    print("💾  STEP 4: Storing in Vector Database")
    print("=" * 60)
    vector_store = VectorStore()
    vector_store.add_documents(chunks, embeddings)

    return vector_store, embedding_manager


def query_documents(vector_store, embedding_manager, question: str, top_k: int = 3):
    """
    Query the vector store with a question.

    Args:
        vector_store: Initialized VectorStore instance
        embedding_manager: Initialized EmbeddingManager instance
        question: The user's question
        top_k: Number of documents to retrieve
    """
    # Initialize LLM
    llm = init_chat_model(
        "google_genai:gemini-2.5-flash",
        temperature=0
    )

    # Create retriever
    retriever = RAGRetriever(vector_store, embedding_manager)

    # Ask the question
    ask_question(question, retriever, llm, top_k=top_k)


if __name__ == "__main__":
    # Step 1: Ingest all documents
    vector_store, embedding_manager = ingest_documents("./data")

    # Step 2: Ask questions
    print("\n\n" + "🔍" * 30)
    print("  RAG PIPELINE READY - ASKING QUESTIONS")
    print("🔍" * 30 + "\n")

    query_documents(
        vector_store,
        embedding_manager,
        "Is there anything mention about Absence of Work in First Call Contract Services?",
        top_k=3
    )
