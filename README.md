# Intelligent Document Search

A lightweight PDF search application built with Streamlit, SentenceTransformers, and FAISS. It extracts text from uploaded PDFs, splits it into searchable chunks, creates vector embeddings, and retrieves the most relevant passages using semantic similarity.

## Features

- Upload and index PDF documents
- Extract text and page metadata from PDFs
- Clean and chunk document text for better retrieval
- Generate embeddings with SentenceTransformers
- Store vectors in a FAISS index
- Perform semantic search over uploaded documents
- Soft-delete documents from the active index
- Preview PDFs in the app interface

## Project structure

```text
Intiligent_search/
├── README.md
├── app.py                    # Streamlit application entry point
├── requirements.txt          # Python dependencies
├── core/                     # Document processing and retrieval logic
│   ├── __init__.py
│   ├── chunker.py
│   ├── document_manager.py
│   ├── embedder.py
│   ├── extractor.py
│   ├── indexer.py
│   ├── preprocessor.py
│   └── search.py
├── utils/                    # Shared configuration and storage helpers
│   ├── __init__.py
│   ├── config.py
│   └── storage.py
├── data/                     # Application-generated runtime data
│   ├── uploads/
│   ├── legacy/               # Older/generated index snapshots kept for reference
│   │   ├── index.faiss
│   │   └── metadata.pkl
│   ├── faiss_index.bin
│   └── metadata.pkl
├── test.py                   # Simple index smoke test
├── __pycache__/              # Python cache (generated locally)
└── .venv/                    # Local virtual environment
```

## Setup

1. Create and activate a virtual environment:

```powershell
cd "d:\6th_sem_projects\Intiligent_search"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app:

```powershell
streamlit run app.py
```

## Usage

- Open the Streamlit app in your browser.
- Upload one or more PDF files from the sidebar.
- Wait for the documents to be processed and indexed.
- Enter a search query to retrieve the most relevant chunks.
- Use the document manager to review or remove indexed files.

## Runtime data

The application stores indexed vectors and metadata under the `data/` directory:

- `data/uploads/` — uploaded PDF files
- `data/faiss_index.bin` — FAISS vector index
- `data/metadata.pkl` — chunk metadata and document information

## Notes

- This project uses the `all-MiniLM-L6-v2` embedding model by default.
- The app is designed for local document search and is best suited for research, demos, and small-scale personal document indexing.
- Generated files like caches and local indexes may be recreated as needed.
