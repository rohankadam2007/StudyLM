import os
import pickle
from pathlib import Path
import numpy as np
import ollama
import streamlit as st

# -----------------------------------------------------------------------------
# 1. Environment & Paths Configuration
# -----------------------------------------------------------------------------
# Explicitly support Homebrew binaries on macOS Apple Silicon
if "/opt/homebrew/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = f"/opt/homebrew/bin:{os.environ.get('PATH', '')}"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="StudyLM — Academic AI Workspace",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. UI Layout & Custom CSS
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .badge-container { display: flex; gap: 8px; margin-bottom: 1.2rem; flex-wrap: wrap; }
    .badge {
        font-size: 0.78rem; padding: 4px 10px; border-radius: 16px;
        background-color: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15); color: #ddd; font-weight: 500;
    }
    .stat-card {
        padding: 12px 16px; background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; text-align: center;
    }
    .stat-value { font-size: 1.3rem; font-weight: 700; color: #60a5fa; }
    .stat-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📓 StudyLM</div>', unsafe_allow_html=True)
st.markdown("""
<div class="badge-container">
    <span class="badge">🔒 100% Offline & Private</span>
    <span class="badge">⚡ Llama 3.2:3b</span>
    <span class="badge">📐 all-MiniLM-L6-v2 (384-d)</span>
    <span class="badge">📄 Page-Level Citations</span>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Sidebar Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📂 Sources & Knowledge Base")
    uploaded_file = st.file_uploader("Upload Notes (PDF)", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None:
        save_path = DATA_DIR / uploaded_file.name
        if not save_path.exists():
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Added `{uploaded_file.name}`!")
            st.rerun()

    pdf_files = sorted([f.name for f in DATA_DIR.glob("*.pdf")])
    if not pdf_files:
        st.warning("No PDFs detected in `data/`. Upload a document to start.")
        st.stop()

    selected_pdf = st.selectbox("Active Subject Notes:", options=pdf_files)
    st.markdown("---")

    st.markdown("### 🛠️ NotebookLM Studio")
    study_guide_btn = st.button("📋 Generate Study Guide", use_container_width=True)
    quiz_btn = st.button("❓ Generate 5-Question Quiz", use_container_width=True)
    glossary_btn = st.button("🏷️ Key Terms & Glossary", use_container_width=True)
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------------
# 4. Lazy-Loaded Embedding Model
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="⚡ Initializing embedding engine...")
def get_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------------------------------------------------
# 5. Document Processing & Persistent Disk Caching
# -----------------------------------------------------------------------------
def get_cache_paths(pdf_name):
    base_stem = Path(pdf_name).stem
    return CACHE_DIR / f"{base_stem}_chunks.pkl", CACHE_DIR / f"{base_stem}_embeddings.npy"

@st.cache_data(show_spinner=False)
def load_cached_index(pdf_name):
    chunks_file, emb_file = get_cache_paths(pdf_name)
    if chunks_file.exists() and emb_file.exists():
        with open(chunks_file, "rb") as f:
            data = pickle.load(f)
        embs = np.load(emb_file)
        return data["chunks"], data["total_chars"], data["total_pages"], embs
    return None

def build_pdf_index(pdf_name):
    from pypdf import PdfReader

    pdf_path = DATA_DIR / pdf_name
    reader = PdfReader(pdf_path)
    
    chunks = []
    chunk_size = 800
    overlap = 150
    total_pages = len(reader.pages)
    page_texts = []
    total_chars = 0

    # 1. Native text extraction pass
    for page in reader.pages:
        raw_text = page.extract_text() or ""
        cleaned = " ".join(raw_text.split())
        page_texts.append(cleaned)
        total_chars += len(cleaned)

    # 2. Scanned PDF fallback via OCR
    if total_chars < 50:
        st.info(f"🔍 `{pdf_name}` contains minimal digital text. Running OCR extraction...")
        try:
            import pytesseract
            from pdf2image import convert_from_path

            brew_tesseract = "/opt/homebrew/bin/tesseract"
            if os.path.exists(brew_tesseract):
                pytesseract.pytesseract.tesseract_cmd = brew_tesseract

            poppler_dir = "/opt/homebrew/bin" if os.path.exists("/opt/homebrew/bin/pdftoppm") else None

            images = convert_from_path(pdf_path, poppler_path=poppler_dir)
            page_texts = []
            total_chars = 0
            
            ocr_progress = st.progress(0, text="Reading scanned pages...")
            for idx, img in enumerate(images):
                ocr_str = pytesseract.image_to_string(img)
                cleaned = " ".join(ocr_str.split())
                page_texts.append(cleaned)
                total_chars += len(cleaned)
                percent = int(((idx + 1) / len(images)) * 100)
                ocr_progress.progress(percent, text=f"OCR processing page {idx + 1} of {len(images)}...")

            ocr_progress.empty()
            st.success("✅ OCR complete! Indexing document vectors...")

        except Exception as ocr_err:
            st.error(f"⚠️ OCR engine error: {ocr_err}")

    # 3. Page-aware chunking
    for page_idx, text in enumerate(page_texts):
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append({
                "text": text[start:end],
                "page": page_idx + 1,
                "source": pdf_name
            })
            if end == len(text):
                break
            start += chunk_size - overlap

    # 4. Vector encoding and disk caching
    if chunks:
        model = get_embed_model()
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True)
        
        chunks_file, emb_file = get_cache_paths(pdf_name)
        with open(chunks_file, "wb") as f:
            pickle.dump({
                "chunks": chunks,
                "total_chars": total_chars,
                "total_pages": total_pages
            }, f)
        np.save(emb_file, embeddings)
    else:
        embeddings = np.array([])

    return chunks, total_chars, total_pages, embeddings

# Check cache first; index only on initial run
cached = load_cached_index(selected_pdf)
if cached is not None:
    chunks_meta, total_chars, total_pages, embeddings = cached
else:
    chunks_meta, total_chars, total_pages, embeddings = build_pdf_index(selected_pdf)

# -----------------------------------------------------------------------------
# 6. Status Metrics Display
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{total_pages}</div><div class="stat-label">Pages</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{len(chunks_meta)}</div><div class="stat-label">Chunks</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{total_chars:,}</div><div class="stat-label">Characters</div></div>', unsafe_allow_html=True)
with col4:
    status_label = "Ready" if chunks_meta else "Unreadable"
    st.markdown(f'<div class="stat-card"><div class="stat-value">{status_label}</div><div class="stat-label">Status</div></div>', unsafe_allow_html=True)

st.write("")

if total_chars < 50:
    st.warning(f"⚠️ **`{selected_pdf}` has minimal text.** Verify OCR tools or upload a typed PDF.")
    st.stop()

# -----------------------------------------------------------------------------
# 7. LLM Query Interface
# -----------------------------------------------------------------------------
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
    except Exception as err:
        return f"⚠️ Unable to reach Ollama. Verify `ollama serve` is running. Error: {err}"

def run_studio_tool(action_name, task_prompt, system_inst):
    with st.spinner(f"Generating {action_name}..."):
        context = "\n\n".join([f"[Page {c['page']}]: {c['text']}" for c in chunks_meta[:min(len(chunks_meta), 8)]])
        res = query_llm(system_inst, f"Context:\n{context}\n\nTask: {task_prompt}")
        st.subheader(f"{action_name}: {selected_pdf}")
        st.markdown(res)
        st.divider()

if study_guide_btn:
    run_studio_tool("📋 Study Guide", "Create a structured study guide with core concepts and revision points.", "You are an academic study assistant.")
if quiz_btn:
    run_studio_tool("❓ Practice Quiz", "Create 5 exam-style questions with complete solutions at the bottom.", "You are a university exam creator.")
if glossary_btn:
    run_studio_tool("🏷️ Technical Glossary", "Extract key definitions and formulas alphabetically.", "You are a technical glossary generator.")

# -----------------------------------------------------------------------------
# 8. Grounded Retrieval Chat
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📌 Source Grounding & Citations"):
                for src in msg["sources"]:
                    conf = max(0.0, min(100.0, (src['score'] + 1) / 2 * 100))
                    st.markdown(f"**📖 {src['source']} — Page {src['page']}** • `{conf:.1f}% Relevance Match`")
                    st.caption(f"\"{src['text']}\"")

prompt = st.chat_input(f"Ask anything regarding {selected_pdf}...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. Fast Vector Search
        model = get_embed_model()
        q_vec = model.encode(prompt, normalize_embeddings=True)
        scores = embeddings @ q_vec
        # Using top 4 provides plenty of context for 10-mark answers without slowing down Llama
        top_k_indices = np.argsort(scores)[::-1][:4]
        
        retrieved_sources = [
            {
                "text": chunks_meta[i]["text"],
                "page": chunks_meta[i]["page"],
                "source": chunks_meta[i]["source"],
                "score": float(scores[i])
            }
            for i in top_k_indices
        ]

        context_block = "\n\n".join([f"[Page {s['page']}]: {s['text']}" for s in retrieved_sources])
        user_msg = f"Context:\n{context_block}\n\nQuestion: {prompt}"
        sys_msg = (
            "You are an academic university tutor preparing answers for exams.\n"
            "- Using the provided context chunks, synthesize a thorough, well-structured answer.\n"
            "- If the user asks for '5 marks' or '10 marks', format with clear headings, bullet points, and definitions.\n"
            "- Always cite pages as [Page X] when referencing facts.\n"
            "- Only state you don't know if the topic is completely unmentioned."
        )

        # 2. Real-Time Token Streaming Generator
        def stream_response():
            try:
                stream = ollama.chat(
                    model="llama3.2:3b",
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    stream=True,
                    options={"temperature": 0.2}
                )
                for chunk in stream:
                    yield chunk["message"]["content"]
            except Exception as e:
                yield f"⚠️ Ollama error: {e}"

        # 3. Stream text live to the screen
        out_text = st.write_stream(stream_response())

        # 4. Show Grounding / Citations right below
        with st.expander("📌 Source Grounding & Citations"):
            for src in retrieved_sources:
                conf = max(0.0, min(100.0, (src['score'] + 1) / 2 * 100))
                st.markdown(f"**📖 {src['source']} — Page {src['page']}** • `{conf:.1f}% Relevance Match`")
                st.caption(f"\"{src['text']}\"")

    st.session_state.messages.append({
        "role": "assistant",
        "content": out_text,
        "sources": retrieved_sources
    })