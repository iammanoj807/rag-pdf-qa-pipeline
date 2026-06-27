# 📄 RAG Pipeline — PDF Document Q&A System

A modular **Retrieval-Augmented Generation (RAG)** pipeline built with Python that lets you ask natural language questions about your PDF documents and get accurate, source-cited answers.

## ✨ Features

- 📥 **PDF Ingestion** — Automatically loads and processes all PDFs from a directory
- ✂️ **Smart Chunking** — Splits documents into optimized chunks using recursive character splitting
- 🧠 **Local Embeddings** — Generates vector embeddings using SentenceTransformer (`all-mpnet-base-v2`)
- 💾 **Persistent Storage** — Stores embeddings in ChromaDB with automatic persistence
- 🔍 **Semantic Search** — Retrieves the most relevant document chunks for any query
- 🤖 **LLM-Powered Answers** — Uses Google Gemini to generate precise, source-cited answers

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Retriever   │────▶│  Vector DB   │────▶│  Top-K Docs │
│  (Embedding) │     │  (ChromaDB)  │     │  (Relevant) │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
                                    ┌───────────────────┐
                                    │   LLM (Gemini)    │
                                    │   Context + Query  │
                                    └────────┬──────────┘
                                             │
                                             ▼
                                       Final Answer
                                    (with source citations)
```

## 📁 Project Structure

```
RAG/
├── pipeline/                    # Core pipeline modules
│   ├── __init__.py              # Package initializer
│   ├── document_loader.py       # PDF loading logic
│   ├── text_splitter.py         # Document chunking
│   ├── embedding_manager.py     # SentenceTransformer embeddings
│   ├── vector_store.py          # ChromaDB storage
│   ├── retriever.py             # Semantic search & retrieval
│   └── rag_chain.py             # LLM question-answering
├── data/                        # Your PDF files go here
│   └── pdf/                     # Place your PDFs in this folder
├── main.py                      # Entry point — run the full pipeline
├── requirements.txt             # Python dependencies
├── .env                         # API keys (not tracked by git)
└── README.md
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/rag-pdf-qa-pipeline.git
cd rag-pdf-qa-pipeline
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up API Keys

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

> Get your free Google AI API key from [Google AI Studio](https://aistudio.google.com/apikey)

### 5. Add Your PDFs

Place your PDF files inside the `data/pdf/` directory:

```bash
mkdir -p data/pdf
# Copy your PDFs into data/pdf/
```

### 6. Run the Pipeline

```bash
python main.py
```

## 📖 How It Works

| Step | Module | What It Does |
|------|--------|-------------|
| 1 | `document_loader.py` | Scans the `data/` folder and loads all PDFs using PyPDF |
| 2 | `text_splitter.py` | Splits pages into ~1000 character chunks with 200 char overlap |
| 3 | `embedding_manager.py` | Converts each chunk into a 768-dimension vector using SentenceTransformer |
| 4 | `vector_store.py` | Stores all vectors in ChromaDB (persisted to disk) |
| 5 | `retriever.py` | Takes a user query, embeds it, and finds the top-K most similar chunks |
| 6 | `rag_chain.py` | Sends the retrieved chunks + question to Gemini LLM for a final answer |

## 🛠️ Tech Stack

- **LangChain** — Framework for building LLM applications
- **ChromaDB** — Vector database for storing and querying embeddings
- **SentenceTransformers** — Local embedding model (`all-mpnet-base-v2`)
- **Google Gemini** — LLM for generating answers
- **PyPDF** — PDF document parsing

## 📝 Example Output

```
❓  Question: Is there anything about Holiday entitlement?
============================================================

📌 **Answer:**
Yes, agency workers are entitled to 5.6 weeks (28 days) of annual leave.
Holiday accrual is calculated at a rate of 12.07%...

📚 **Sources Used:**
- 📄 File: Employee Handbook.pdf | 📃 Page: 10 | 🎯 Relevance Score: 0.35
============================================================
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
