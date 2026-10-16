<div align="center">

# 📚 StudyLM

### Ask questions from your own VTU notes, 100% offline.

A local, privacy-first RAG (Retrieval-Augmented Generation) app that answers from your PDFs and tells you the exact source file and page.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/LLM-Llama%203.2%203B-black)
![Embeddings](https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-orange)
![NumPy](https://img.shields.io/badge/Vector%20Search-NumPy-013243?logo=numpy&logoColor=white)
![Offline](https://img.shields.io/badge/Runs-Fully%20Offline-brightgreen)

</div>

---

## ✨ Why this project?

Searching through hundreds of pages of notes before exams is slow. This tool lets you ask a question in plain English and get an answer taken **only from your notes**, with a citation so you can verify it.

- 🔒 **Private:** your notes never leave your computer
- 💸 **Free:** no API keys, no paid services
- 📍 **Traceable:** every answer shows its source file and page
- 🧠 **Built from scratch:** no LangChain or LlamaIndex, so every step is visible and understandable

---

## 🏗️ How It Works

```
 PDFs ──► Extract text ──► Clean ──► Chunk (overlapping)
                                        │
                                        ▼
                              Embed with MiniLM (normalized)
                                        │
 Question ──► Embed ──► Dot product with all chunks ──► Top-k chunks
                                                            │
                                                            ▼
                                     Strict prompt ──► Llama 3.2 3B ──► Answer + Sources
```

| Step | What happens |
|---|---|
| **1. Ingest** | `pypdf` reads each page, whitespace is cleaned, scanned PDFs are detected, and text is split into overlapping chunks that remember their page number |
| **2. Embed** | `sentence-transformers` converts each chunk into a normalized vector |
| **3. Retrieve** | Because vectors are normalized, cosine similarity equals a dot product. A single NumPy multiplication (`embeddings @ query_embedding`) scores every chunk at once |
| **4. Generate** | The top chunks go into a strict prompt for `llama3.2:3b` running on Ollama (`temperature=0`), which is told to answer only from the context |

---

## 🛠️ Tech Stack

| Purpose | Tool |
|---|---|
| Language | Python 3.10+ |
| PDF reading | `pypdf` |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector search | `numpy` |
| Local LLM | Ollama with `llama3.2:3b` |
| Version control | Git & GitHub |

---

## 🚀 Getting Started

**1. Install [Ollama](https://ollama.com/) and download the model**
```bash
ollama pull llama3.2:3b
```

**2. Clone the project and install dependencies**
```bash
git clone https://github.com/<rohankadam2007>/<StudyLM>.git
cd <StudyLM>
pip install -r requirements.txt
```

**3. Add your notes**

Put your PDF files in the `<sql-vtu-notes>/` folder.

**4. Run**
```bash
python <main>.py
```

The first run processes and embeds your PDFs, which takes a little while. After that, type your question and press Enter.

---

## 💡 Example Usage

```
Question: What is the difference between supervised and unsupervised learning?

Answer: <answer generated from your notes>

Sources: <sql_notes>.pdf (page <n>)
```

---

## ⚠️ Limitations

- A 3B model can still make mistakes, so check important answers against the cited page.
- Scanned (image-only) PDFs are detected but not read, since there is no OCR.
- Tables, diagrams, and equations are not handled well.
- Search is brute force, which works well for a few hundred pages but not for huge collections.

---

## 🔮 Roadmap

- [ ] Retrieval evaluation (top-3 hit rate on a test question set)
- [ ] "Not found in notes" threshold using similarity scores
- [ ] Compare chunk sizes using the evaluation
- [ ] Streamlit web interface
- [ ] OCR support for scanned PDFs

---

## 👨‍💻 Author

**Rohan**, B.E. AI & ML student, VVIT Bengaluru
GitHub: [@rohankadam2007](https://github.com/rohankadam2007)

---

<div align="center">
⭐ If you find this useful, consider giving it a star.
</div> 