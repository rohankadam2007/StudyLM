# 📚 Ask VTU Notes — Local, Privacy-First RAG Engine

A production-grade, offline Retrieval-Augmented Generation (RAG) system engineered to query academic course notes and syllabi. Built to provide zero-hallucination, grounded answers with exact source traceability without relying on paid cloud APIs.

---

## 🏗️ Architectural Overview

Unlike basic RAG implementations that wrap high-level black-box frameworks, this project is built from scratch with explicit control over data ingestion, vector math, and local LLM inference:

1. **Ingestion & Validation:** Parses PDF notes (`pypdf`), sanitizes whitespace, checks for scanned documents, and applies sliding-window chunking with overlap validation.
2. **Dense Semantic Retrieval:** Embeds document chunks using `sentence-transformers` (`all-MiniLM-L6-v2`) with normalized vectors.
3. **Optimized Similarity Math:** Replaces heavy cosine similarity functions with lightning-fast NumPy matrix multiplication (`embeddings @ query_embedding`).
4. **Local Grounded Generation:** Feeds retrieved context into a quantized `Llama 3.2:3b` model running locally via **Ollama**, enforced with strict system prompts (`temperature=0`) to eliminate hallucinations.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **PDF Extraction:** `pypdf`
- **Embeddings & NLP:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Vector Math:** `numpy`
- **Local LLM Inference:** `Ollama` (`Llama 3.2:3b`)
- **Version Control:** Git & GitHub

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure you have Python installed and [Ollama](https://ollama.com/) running locally with the model pulled:
```bash
ollama pull llama3.2:3b