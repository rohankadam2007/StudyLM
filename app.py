import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import ollama

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

st.set_page_config(page_title="Ask VTU Notes", page_icon="📚", layout="wide")
st.title("📚 Ask VTU Notes — Local AI Study Assistant")
st.caption("Query your VTU syllabus notes locally with sentence embeddings and Llama 3.2.")

# Cache the embedding model so it only loads once into memory
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embed_model()

# Scan data directory for available PDF notes
pdf_files = sorted([f.name for f in DATA_DIR.glob("*.pdf")])

with st.sidebar:
    st.header("📂 Knowledge Base")
    if not pdf_files:
        st.error("No PDFs found in the 'data/' folder!")
        selected_pdf = None
    else:
        selected_pdf = st.selectbox("Select Subject Notes:", options=pdf_files)
    st.markdown("---")
    st.info("💡 Drop additional VTU course PDFs into `data/` to include them.")

if not selected_pdf:
    st.stop()

# Extract and chunk PDF text
@st.cache_resource
def process_pdf(pdf_name):
    pdf_path = DATA_DIR / pdf_name
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += (page.extract_text() or "") + "\n"

    cleaned_text = " ".join(raw_text.split())

    # Sliding-window chunking
    chunk_size = 1000
    overlap = 200
    chunks = []
    start = 0
    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))
        chunks.append(cleaned_text[start:end])
        if end == len(cleaned_text):
            break
        start += chunk_size - overlap

    return cleaned_text, chunks

cleaned_text, chunks = process_pdf(selected_pdf)

if len(cleaned_text.strip()) < 100:
    st.error("This PDF contains minimal or scanned text. Please provide a text-based PDF.")
    st.stop()

# Generate and cache embeddings for the selected document chunks
@st.cache_resource
def compute_embeddings(pdf_name, chunk_tuple):
    return model.encode(list(chunk_tuple), normalize_embeddings=True)

embeddings = compute_embeddings(selected_pdf, tuple(chunks))

# Manage chat history state
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt input
if prompt := st.chat_input(f"Ask a question about {selected_pdf}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages and running inference..."):
            query_vector = model.encode(prompt, normalize_embeddings=True)
            scores = embeddings @ query_vector
            top_k_indices = np.argsort(scores)[::-1][:3]
            retrieved_chunks = [chunks[i] for i in top_k_indices]

            context_block = "\n\n".join(retrieved_chunks)
            user_message = f"Context:\n{context_block}\n\nQuestion: {prompt}"

            system_instruction = (
                "You answer questions using ONLY the context given by the user.\n"
                "Rules:\n"
                "- Use only facts and examples that are written in the context.\n"
                "- Do NOT invent tables, data, or examples.\n"
                "- If the answer is not in the context, say exactly: \"I don't know based on these notes.\"\n"
                "- Keep the answer short."
            )

            try:
                response = ollama.chat(
                    model="llama3.2:3b",
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_message},
                    ],
                    options={"temperature": 0},
                )
                output_text = response["message"]["content"]
            except Exception:
                output_text = "⚠️ Unable to reach Ollama. Verify that the Ollama service is active (`ollama serve`)."

            st.markdown(output_text)

    st.session_state.messages.append({"role": "assistant", "content": output_text})