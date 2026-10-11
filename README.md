# Ask VTU Notes
A RAG (Retrieval-Augmented Generation) app that answers questions
from VTU notes PDFs, so students can ask instead of searching pages.

## How it works (current)
PDF -> extract text -> split into overlapping chunks -> embeddings
Question -> embed -> cosine similarity -> top 3 matching chunks

## Planned
Top chunks + question -> LLM -> written answer

## Tech stack
- Python
- pypdf (PDF reading)
- sentence-transformers, all-MiniLM-L6-v2 (embeddings)
- numpy (similarity search)

## How to run
1. Clone the repo
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install packages: `pip install -r requirements.txt`
5. Put a text-based PDF at `data/cn-notes.pdf` (filename is hardcoded for now)
6. Run: `python3 main.py` (first run downloads the embedding model)

## Status
- [x] Extract text from PDF
- [x] Chunking with overlap
- [x] Embeddings + similarity search
- [ ] LLM answers
- [ ] Multiple PDFs / OCR for scanned PDFs
- [ ] Streamlit UI
