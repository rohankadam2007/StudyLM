import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="StudyLM", page_icon="📓", layout="wide")
st.title("📓 StudyLM — Local NotebookLM for Course Notes")
st.caption("Offline RAG study assistant with page citations & studio tools powered by Llama 3.2.")

# 1. Load Embedding Model
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embed_model()

# 2. Sidebar Setup
pdf_files = sorted([f.name for f in DATA_DIR.glob("*.pdf")])

with st.sidebar:
    st.header("📂 Knowledge Base")
    if not pdf_files:
        st.error("No PDFs found in the 'data/' folder!")
        st.stop()
    
    selected_pdf = st.selectbox("Active Subject Notes:", options=pdf_files)
    st.markdown("---")
    st.header("🛠️ NotebookLM Studio")
    st.caption("Generate instant study materials from active notes.")
    study_guide_btn = st.button("📋 Generate Study Guide", use_container_width=True)
    quiz_btn = st.button("❓ Generate 5-Question Quiz", use_container_width=True)
    st.markdown("---")
    st.info("💡 Place text-based VTU PDFs inside `data/` to include them.")

# 3. Page-Aware Chunking (Preserves exact page numbers!)
@st.cache_resource
def process_pdf_with_pages(pdf_name):
    pdf_path = DATA_DIR / pdf_name
    reader = PdfReader(pdf_path)
    
    chunks_with_metadata = []
    chunk_size = 800
    overlap = 150
    total_extracted_chars = 0

    for page_idx, page in enumerate(reader.pages):
        raw_page_text = page.extract_text() or ""
        cleaned_page_text = " ".join(raw_page_text.split())
        total_extracted_chars += len(cleaned_page_text)
        
        if not cleaned_page_text:
            continue
            
        start = 0
        while start < len(cleaned_page_text):
            end = min(start + chunk_size, len(cleaned_page_text))
            chunk_slice = cleaned_page_text[start:end]
            
            chunks_with_metadata.append({
                "text": chunk_slice,
                "page": page_idx + 1,  # 1-indexed for humans
                "source": pdf_name
            })
            
            if end == len(cleaned_page_text):
                break
            start += chunk_size - overlap

    return chunks_with_metadata, total_extracted_chars

chunks_meta, total_chars = process_pdf_with_pages(selected_pdf)

# Graceful fallback if PDF is scanned or unreadable
if total_chars < 50:
    st.warning(
        f"⚠️ **`{selected_pdf}` appears to be a scanned image or empty PDF.**\n\n"
        f"Standard text extractors cannot read image-only notes. "
        f"Please select a typed, text-based PDF (like `sql_notes.pdf`) from the sidebar to continue."
    )
    st.stop()

# 4. Generate & Cache Embeddings
@st.cache_resource
def compute_metadata_embeddings(pdf_name, _chunks_meta):
    texts = [c["text"] for c in _chunks_meta]
    return model.encode(texts, normalize_embeddings=True)

embeddings = compute_metadata_embeddings(selected_pdf, chunks_meta)

# Helper function for Ollama calls
def query_llm(system_prompt, user_prompt):
    try:
        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0},
        )
        return response["message"]["content"]
    except Exception:
        return "⚠️ Unable to reach Ollama. Verify that the Ollama service is active (`ollama serve`)."

# 5. Studio Tools
if study_guide_btn:
    with st.spinner("Synthesizing Study Guide from notes..."):
        sample_context = "\n\n".join([f"[Page {c['page']}]: {c['text']}" for c in chunks_meta[:min(len(chunks_meta), 8)]])
        prompt = f"Context:\n{sample_context}\n\nTask: Create a structured study guide summarizing key definitions, core concepts, and high-yield topics."
        system_instruction = "You are an academic study assistant. Create concise, clear, and bulleted study guides strictly based on the provided context."
        guide = query_llm(system_instruction, prompt)
        
        st.subheader(f"📋 Study Guide: {selected_pdf}")
        st.markdown(guide)
        st.markdown("---")

if quiz_btn:
    with st.spinner("Formulating 5 practice exam questions..."):
        sample_context = "\n\n".join([f"[Page {c['page']}]: {c['text']}" for c in chunks_meta[:min(len(chunks_meta), 8)]])
        prompt = f"Context:\n{sample_context}\n\nTask: Generate 5 VTU exam-style practice questions with a detailed answer key at the bottom."
        system_instruction = "You are a university examiner. Generate accurate, challenging test questions and solutions strictly based on the provided material."
        quiz = query_llm(system_instruction, prompt)
        
        st.subheader(f"❓ Practice Quiz: {selected_pdf}")
        st.markdown(quiz)
        st.markdown("---")

# 6. Chat History Management
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📌 Source Citations & Grounding"):
                for src in msg["sources"]:
                    st.markdown(f"**📖 {src['source']} — Page {src['page']}** *(Relevance Score: `{src['score']:.4f}`)*")
                    st.caption(f"\"{src['text']}\"")
                    st.divider()

# 7. User Query & Retrieval
if prompt := st.chat_input(f"Ask a question about {selected_pdf}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages and running inference..."):
            query_vector = model.encode(prompt, normalize_embeddings=True)
            scores = embeddings @ query_vector
            top_k_indices = np.argsort(scores)[::-1][:3]
            
            retrieved_sources = [
                {
                    "text": chunks_meta[i]["text"],
                    "page": chunks_meta[i]["page"],
                    "source": chunks_meta[i]["source"],
                    "score": float(scores[i])
                }
                for i in top_k_indices
            ]

            # Context formatted with explicit page markers
            context_block = "\n\n".join([f"[Page {s['page']}]: {s['text']}" for s in retrieved_sources])
            user_message = f"Context:\n{context_block}\n\nQuestion: {prompt}"

            system_instruction = (
                "You answer questions using ONLY the context given by the user.\n"
                "Rules:\n"
                "- Use only facts and examples that are written in the context.\n"
                "- Do NOT invent tables, data, or examples.\n"
                "- If the answer is not in the context, say exactly: \"I don't know based on these notes.\"\n"
                "- Whenever possible, cite the page number like [Page X] in your answer.\n"
                "- Keep the answer concise and direct."
            )

            output_text = query_llm(system_instruction, user_message)
            st.markdown(output_text)
            
            with st.expander("📌 Source Citations & Grounding"):
                for src in retrieved_sources:
                    st.markdown(f"**📖 {src['source']} — Page {src['page']}** *(Relevance Score: `{src['score']:.4f}`)*")
                    st.caption(f"\"{src['text']}\"")
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": output_text,
        "sources": retrieved_sources
    })