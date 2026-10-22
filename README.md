<div align="center">

# 📚 StudyLM

### Ask questions from your own VTU notes, with exact page citations.

A high-precision RAG (Retrieval-Augmented Generation) workspace that answers from your PDFs, tells you the exact source file and page, and generates revision materials.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://studylm-dfwb3xgitkahyv3cbjvuv7.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Groq%20Cloud-F55036)
![Embeddings](https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-orange)
![NumPy](https://img.shields.io/badge/Vector%20Search-NumPy-013243?logo=numpy&logoColor=white)

</div>

---

## ✨ Why this project?

Searching through hundreds of pages of notes before exams is slow. This tool lets you ask a question in plain English and get an answer taken **only from your notes**, with a citation so you can verify it.

- 🌐 **Live Web App:** deployed 24/7 on Streamlit Cloud ([Try it live here](https://studylm-dfwb3xgitkahyv3cbjvuv7.streamlit.app/))
- 📍 **Traceable:** every answer shows its exact source file and page number
- 🛠️ **NotebookLM Studio:** one-click generators for Study Guides, 5-Question Quizzes, and Key Terms & Glossaries
- 🔍 **Reads Scanned Notes:** includes automatic OCR fallback for image-only or scanned PDFs
- 🧠 **Built from scratch:** no LangChain or LlamaIndex, so every step is visible and understandable

---

## 🏗️ How It Works

```
 PDFs ──► Extract text (pypdf + OCR fallback) ──► Clean ──► Chunk (overlapping)
                                                               │
                                                               ▼
                                                     Embed with MiniLM (normalized)
                                                               │
 Question ──► Embed ──► Dot product with all chunks ──► Top-k chunks
                                                               │
                                                               ▼
                                          Strict prompt ──► Groq LLM ──► Answer + Sources
```

| Step | What happens |
|---|---|
| **1. Ingest** | `pypdf` reads each page and cleans whitespace. If a scanned page is detected, `pytesseract` and `pdf2image` automatically run OCR to read the text. Pages are split into overlapping chunks that remember their page number |
| **2. Embed** | `sentence-transformers` converts each chunk into a normalized 384-d vector and caches it in `.cache/` for instant reloads |
| **3. Retrieve** | Because vectors are normalized, cosine similarity equals a dot product. A single NumPy multiplication (`embeddings @ query_embedding`) scores every chunk at once |
| **4. Generate** | The top chunks go into a strict prompt for `openai/gpt-oss-20b` running on Groq Cloud (`temperature=0.1`), which is told to answer only from the context |

---

## 🛠️ Tech Stack

| Purpose | Tool |
|---|---|
| Language | Python 3.10+ |
| Web interface | Streamlit |
| PDF reading & OCR | `pypdf`, `pytesseract`, `pdf2image` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector search | `numpy` |
| Cloud LLM | Groq Cloud API (`openai/gpt-oss-20b`) |
| Local LLM (Optional) | Ollama (`llama3.2:3b`) |
| Version control | Git & GitHub |

---

## 🚀 Getting Started

### Option 1: Live Web App (No Setup)
Use the live deployed app directly in your browser:  
👉 **[https://studylm-dfwb3xgitkahyv3cbjvuv7.streamlit.app/](https://studylm-dfwb3xgitkahyv3cbjvuv7.streamlit.app/)**

---

### Option 2: Run Locally

**1. Clone the project and install dependencies**
```bash
git clone [https://github.com/rohankadam2007/StudyLM.git](https://github.com/rohankadam2007/StudyLM.git)
cd StudyLM
pip install -r requirements.txt
```

**2. Install OCR system tools (optional, for scanned PDFs)**
```bash
# On macOS:
brew install tesseract poppler

# On Ubuntu/Debian:
sudo apt-get install tesseract-ocr poppler-utils
```

**3. Add your Groq API Key**

Create `.streamlit/secrets.toml` and add your key:
```toml
GROQ_API_KEY = "gsk_your_groq_api_key_here"
```

**4. Add your notes**

Put your PDF files in the `data/` folder (or upload them directly in the sidebar).

**5. Run**
```bash
streamlit run app.py
```

The first run processes and embeds your PDFs, then caches them in `.cache/`. After that, ask any question or use the Studio tools in the sidebar.

---

## 💡 Example Usage

```
Question: What is Computer Networking?

Answer: Based on the provided notes, a computer network is a system of interconnected computing devices that exchange data and share resources using communication protocols over physical or wireless technologies.

Sources: cn-notes.pdf (page 1, page 2)
```

---

## ⚠️ Limitations

- Answers are strictly restricted to your uploaded notes; if a topic is missing from the PDF, the model replies: *"I don't know based on these notes."*
- Complex diagrams, circuit schematics, and messy handwritten equations are not parsed well—always verify diagrams on the cited page.
- Search is exact brute-force dot product, which works great for semester notes (hundreds of pages) but is not meant for millions of documents.

---

## 🔮 Roadmap

- [x] Streamlit web interface
- [x] OCR support for scanned PDFs
- [x] Disk caching for embeddings (`.cache/`)
- [x] NotebookLM Studio tools (Study Guide, 5-Question Quiz, Glossary)
- [ ] Retrieval evaluation (top-3 hit rate on a test question set)
- [ ] "Not found in notes" threshold using similarity scores
- [ ] Export generated study guides and quizzes to PDF / Markdown

---

## 👨‍💻 Author

**Rohan G Kadam**, B.E. AI & ML student, VVIT Bengaluru  
GitHub: [@rohankadam2007](https://github.com/rohankadam2007)  
Live App: [studylm-dfwb3xgitkahyv3cbjvuv7.streamlit.app](https://studylm-dfwb3xgitkahyv3cbjvuv7.streamlit.app/)

---

<div align="center">
⭐ If you find this useful, consider giving it a star.
</div>