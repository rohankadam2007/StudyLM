import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama

# -----------------------------------------------------------------------------
# 1. Platform-Agnostic Directory Setup
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="StudyLM — Academic AI Workspace",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Modern UI Styles
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

# -----------------------------------------------------------------------------
# 3. Model Loader
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embed_model()

# -----------------------------------------------------------------------------
# 4. Sidebar: Knowledge Base & Upload Handling
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
# 5. Fault-Tolerant Text & OCR Extraction
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def process_pdf_safely(pdf_name):
    pdf_path = DATA_DIR / pdf_name
    reader = PdfReader(pdf_path)
    
    chunks_with_metadata = []
    chunk_size = 800
    overlap = 150
    total_pages = len(reader.pages)
    page_texts = []
    total_extracted_chars = 0

    # Primary pass: native digital text
    for page in reader.pages:
        raw_text = page.extract_text() or ""
        cleaned = " ".join(raw_text.split())
        page_texts.append(cleaned)
        total_extracted_chars += len(cleaned)

    # Fallback pass: OCR if content is minimal/scanned
    is_scanned = total_extracted_chars < 50
    ocr_attempted = False
    ocr_successful = False

    if is_scanned:
        ocr_attempted = True
        try:
            import pytesseract
            from pdf2image import convert_from_path
            
            images = convert_from_path(pdf_path)
            ocr_page_texts = []
            ocr_chars = 0
            for img in images:
                ocr_str = pytesseract.image_to_string(img)
                cleaned_ocr = " ".join(ocr_str.split())
                ocr_page_texts.append(cleaned_ocr)
                ocr_chars += len(cleaned_ocr)
            
            if ocr_chars >= 50:
                page_texts = ocr_page_texts
                total_extracted_chars = ocr_chars
                ocr_successful = True
        except Exception:
            # System lacks tesseract/poppler binary; handle gracefully without halting
            ocr_successful = False

    # Chunking pipeline
    for page_idx, cleaned_text in enumerate(page_texts):
        if not cleaned_text:
            continue
        start = 0
        while start < len(cleaned_text):
            end = min(start + chunk_size, len(cleaned_text))
            chunks_with_metadata.append({
                "text": cleaned_text[start:end],
                "page": page_idx + 1,
                "source": pdf_name
            })
            if end == len(cleaned_text):
                break
            start += chunk_size - overlap

    return chunks_with_metadata, total_extracted_chars, total_pages, is_scanned, ocr_attempted, ocr_successful

chunks_meta, total_chars, total_pages, is_scanned, ocr_attempted, ocr_successful = process_pdf_safely(selected_pdf)

# -----------------------------------------------------------------------------
# 6. Metrics & Header Display
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">📓 StudyLM</div>', unsafe_allow_html=True)
st.markdown("""
<div class="badge-container">
    <span class="badge">🔒 100% Offline & Private</span>
    <span class="badge">⚡ Llama 3.2:3b</span>
    <span class="badge">📐 all-MiniLM-L6-v2 (384-d)</span>
    <span class="badge">📄 Page-Level Citations</span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{total_pages}</div><div class="stat-label">Pages</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{len(chunks_meta)}</div><div class="stat-label">Chunks</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{total_chars:,}</div><div class="stat-label">Characters</div></div>', unsafe_allow_html=True)
with col4:
    status_text = "Ready" if total_chars >= 50 else "Unreadable"
    st.markdown(f'<div class="stat-card"><div class="stat-value">{status_text}</div><div class="stat-label">Status</div></div>', unsafe_allow_html=True)

st.write("")

# Diagnostic notice for unreadable documents
if total_chars < 50:
    st.warning(
        f"⚠️ **`{selected_pdf}` contains little to no extractable text.**\n\n"
        "- This occurs when notes are photocopies, scans, or handwritten images.\n"
        "- To enable optical character extraction on scanned notes, verify local OCR utilities: `brew install tesseract poppler`.\n"
        "- Otherwise, select a digital text-based PDF from the sidebar."
    )
    st.stop()

# -----------------------------------------------------------------------------
# 7. Embedding Indexing
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def compute_metadata_embeddings(pdf_name, _chunks_meta):
    texts = [c["text"] for c in _chunks_meta]
    return model.encode(texts, normalize_embeddings=True)

embeddings = compute_metadata_embeddings(selected_pdf, chunks_meta)

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
        return (
            "⚠️ **Unable to reach Ollama.**\n\n"
            "Ensure the local Ollama background process is running (`ollama serve` or open the Ollama app), "
            f"and verify `llama3.2:3b` is installed.\n\n*Error details: {err}*"
        )

# -----------------------------------------------------------------------------
# 8. Studio Execution Handlers
# -----------------------------------------------------------------------------
def run_studio_tool(action_name, custom_task, custom_system):
    with st.spinner(f"Generating {action_name}..."):
        sample_context = "\n\n".join([f"[Page {c['page']}]: {c['text']}" for c in chunks_meta[:min(len(chunks_meta), 8)]])
        prompt = f"Context:\n{sample_context}\n\nTask: {custom_task}"
        output = query_llm(custom_system, prompt)
        st.subheader(f"{action_name}: {selected_pdf}")
        st.markdown(output)
        st.divider()

if study_guide_btn:
    run_studio_tool(
        "📋 Study Guide",
        "Synthesize a clear study guide covering core topics, key principles, and revision takeaways.",
        "You are an academic study assistant. Provide concise, bulleted revision summaries strictly grounded in the provided notes."
    )

if quiz_btn:
    run_studio_tool(
        "❓ Practice Quiz",
        "Generate 5 exam-style practice questions with complete solutions and an answer key at the bottom.",
        "You are a university exam creator. Create structured, accurate test questions based strictly on the provided context."
    )

if glossary_btn:
    run_studio_tool(
        "🏷️ Technical Glossary",
        "Extract all key technical terminology, definitions, and equations into an alphabetical reference list.",
        "You are a technical editor. Extract concise definitions and formulas strictly as presented in the context."
    )

# -----------------------------------------------------------------------------
# 9. Conversational Interface & Citations
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
        with st.spinner("Retrieving relevant passages and synthesizing response..."):
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

            context_block = "\n\n".join([f"[Page {s['page']}]: {s['text']}" for s in retrieved_sources])
            user_message = f"Context:\n{context_block}\n\nQuestion: {prompt}"

            system_instruction = (
                "You answer questions using ONLY the context provided by the user.\n"
                "Rules:\n"
                "- Base answers strictly on facts and examples from the context.\n"
                "- Do NOT hallucinate or extrapolate.\n"
                "- If the context does not contain the answer, say exactly: \"I don't know based on these notes.\"\n"
                "- Reference the page number like [Page X] whenever quoting a concept.\n"
                "- Keep explanations direct, concise, and structured."
            )

            output_text = query_llm(system_instruction, user_message)
            st.markdown(output_text)
            
            with st.expander("📌 Source Grounding & Citations"):
                for src in retrieved_sources:
                    conf = max(0.0, min(100.0, (src['score'] + 1) / 2 * 100))
                    st.markdown(f"**📖 {src['source']} — Page {src['page']}** • `{conf:.1f}% Relevance Match`")
                    st.caption(f"\"{src['text']}\"")

    st.session_state.messages.append({
        "role": "assistant",
        "content": output_text,
        "sources": retrieved_sources
    })