"""RAG Chain Module - Handles LLM question-answering with retrieved context."""

from pipeline.retriever import RAGRetriever


def ask_question(query: str, retriever: RAGRetriever, llm, top_k: int = 5):
    """
    Ask a question using the RAG pipeline.

    Retrieves relevant document chunks, builds a context prompt,
    and sends it to the LLM for a final answer.

    Args:
        query: The user's question
        retriever: RAGRetriever instance for finding relevant documents
        llm: LangChain LLM instance (e.g., Gemini, Groq)
        top_k: Number of top documents to retrieve
    """
    # Step 1 — Retrieve chunks
    retrieved_docs = retriever.retrieve(query, top_k=top_k)

    if not retrieved_docs:
        print("No relevant documents found.")
        return

    # Step 2 — Build context with numbered sources
    context = ""
    for doc in retrieved_docs:
        context += f"""
[Source {doc['rank']}]
File: {doc['metadata'].get('source_file', 'Unknown')}
Page: {doc['metadata'].get('page', 'N/A')}
Score: {round(doc['similarity_score'], 2)}
Content: {doc['content']}
---"""

    # Step 3 — Build the prompt
    prompt = f"""You are a precise document assistant. 
    Use ONLY the context provided below to answer the question.
    If the answer is not found in the context, say "I cannot find this information in the provided documents."

    Format your response EXACTLY like this:

    📌 **Answer:**
    [Write a clear, complete answer using exact information from the documents. Use bullet points if there are multiple points.]

    📚 **Sources Used:**
    - 📄 File: <filename> | 📃 Page: <page number> | 🎯 Relevance Score: <score>

    ---

    Context:
    {context}

    Question: {query}

    Answer:"""

    # Step 4 — Send to LLM
    response = llm.invoke(prompt)

    # Step 5 — Print result
    print("\n" + "=" * 60)
    print(f"❓  Question: {query}")
    print("=" * 60)
    print(f"\n{response.content}")
    print("=" * 60)
