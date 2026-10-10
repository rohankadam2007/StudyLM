# Ask VTU Notes
A RAG (Retrieval-Augmented Generation) app that answers questions
from VTU notes PDFs, so students can ask instead of searching pages.

## How it works
PDF -> extract text -> split into chunks -> embeddings -> vector store
Question -> find closest chunks -> LLM -> answer

## Tech stack
- Python
- pypdf (PDF reading)

## How to run
1. Clone the repo
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install packages: `pip install pypdf`
5. Put a PDF in the `data/` folder
6. Run: `python3 main.py`

## Status
- [x] Extract text from PDF
- [ ] Chunking
- [ ] Embeddings + vector store
- [ ] LLM answers
- [ ] Streamlit UI