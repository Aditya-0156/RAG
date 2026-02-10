# RAG Knowledge Base

A Retrieval-Augmented Generation system that lets you upload documents and ask questions about them. Built with Python, LangChain, ChromaDB, and Google Gemini.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-latest-green)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-teal)
![ChromaDB](https://img.shields.io/badge/ChromaDB-latest-orange)
![Gemini](https://img.shields.io/badge/Google-Gemini-8E75B2)
![License](https://img.shields.io/badge/License-MIT-yellow)

## What it does

- **Ingest documents** &mdash; Load PDF, DOCX, TXT, and Markdown files, automatically chunk them for optimal retrieval
- **Semantic search** &mdash; Convert text to vector embeddings with Google's `gemini-embedding-001` model and store them in ChromaDB
- **Question answering** &mdash; Retrieve relevant chunks and generate grounded answers using Gemini
- **Web UI** &mdash; Glassmorphism-styled browser interface with drag-and-drop uploads
- **CLI** &mdash; Interactive terminal interface for quick usage

## Architecture

```
User Query
    |
    v
+------------------+     +------------------+     +------------------+
| Document Loader  | --> | Embedding Gen.   | --> |   Vector Store   |
| (PDF/TXT/DOCX/MD)|     | (Gemini Embed)   |     |   (ChromaDB)     |
+------------------+     +------------------+     +------------------+
                                                          |
                                                          v
                                                  +------------------+
                                                  |    Retriever     |
                                                  | (Similarity Search)|
                                                  +------------------+
                                                          |
                                                          v
                                                  +------------------+
                                                  |    RAG Chain     |
                                                  | (Gemini Flash)   |
                                                  +------------------+
                                                          |
                                                          v
                                                      Answer
```

## Project Structure

```
RAG/
├── main.py                          # Entry point (CLI + Web)
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py                    # Settings management (pydantic-settings)
│   ├── rag_chain.py                 # Core RAG orchestration
│   ├── ingestion/
│   │   └── document_loader.py       # Multi-format document loading & chunking
│   ├── embeddings/
│   │   └── embedding_generator.py   # Google Gemini embedding generation
│   ├── vectorstore/
│   │   └── chroma_store.py          # ChromaDB vector storage & search
│   ├── retrieval/
│   │   └── retriever.py             # Semantic retrieval & context formatting
│   └── web/
│       └── app.py                   # FastAPI backend + glassmorphism UI
├── data/
│   ├── raw/                         # Uploaded documents
│   └── chroma_db/                   # Persisted vector store
└── tests/
```

## Setup

### Prerequisites

- Python 3.12+
- A Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### Installation

```bash
# Clone the repo
git clone https://github.com/Aditya-0156/RAG.git
cd RAG

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

## Usage

### Web Interface

```bash
python main.py --mode web
```

Open `http://localhost:8000` in your browser. Drag and drop files to upload, then ask questions.

![Web UI Screenshot](https://via.placeholder.com/800x400?text=Add+screenshot+here)

### CLI

```bash
python main.py
```

```
> add /path/to/document.pdf
Added 12 document chunks to knowledge base.

> ask What are the key findings?
Answer:
Based on the document, the key findings include...

Sources: document.pdf
```

### Available CLI Commands

| Command | Description |
|---------|-------------|
| `add <path>` | Add a document or directory to the knowledge base |
| `ask <question>` | Ask a question about your documents |
| `stats` | Show knowledge base statistics |
| `clear` | Remove all documents |
| `help` | Show available commands |
| `quit` | Exit |

## Configuration

All settings are managed through environment variables (`.env`) with sensible defaults in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `GOOGLE_API_KEY` | *required* | Gemini API key |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `LLM_MODEL` | `gemini-flash-latest` | LLM for answer generation |
| `CHUNK_SIZE` | `1000` | Document chunk size (chars) |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |
| `TEMPERATURE` | `0.7` | LLM temperature |

## Tech Stack

- **Python 3.12** &mdash; Language
- **LangChain** &mdash; Document loading, text splitting
- **ChromaDB** &mdash; Vector database with persistence
- **Google Gemini** &mdash; Embeddings (`gemini-embedding-001`) and LLM (`gemini-flash-latest`)
- **FastAPI** &mdash; Web backend
- **Pydantic** &mdash; Configuration and validation

## Features to Add (Roadmap)

- [ ] User authentication and multi-user support
- [ ] Document versioning and history
- [ ] Support for more file formats (CSV, JSON, etc.)
- [ ] Advanced search filters
- [ ] Export conversation history
- [ ] Batch document processing

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

MIT
