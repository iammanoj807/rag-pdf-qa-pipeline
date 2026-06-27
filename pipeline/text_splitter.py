"""Text Splitter Module - Handles chunking documents and cleaning metadata."""

from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents: List of LangChain Document objects
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks

    Returns:
        List of chunked LangChain Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    if split_docs:
        print(f"\n Example chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}....")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs