from retrieval import embed_chunks, search
from pypdf import PdfReader
from pathlib import Path
from llm import build_prompt,ask_model

BASE_DIR = Path(__file__).parent

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text

#Chunking the text into smaller parts
def chunk_text(text, chunk_size=1000, overlap=200):
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    text = extract_text(BASE_DIR / "data" / "sql_notes.pdf")
    text = " ".join(text.split())

    if len(text.strip()) < 100:
        print("This PDF looks scanned, so I can't read the text. Please upload a text-based PDF.")
    else:
        chunks = chunk_text(text)
        print("Total chunks:", len(chunks))

        print("Embedding chunks... (takes a bit)")
        embeddings = embed_chunks(chunks)

        while True:
            question = input("\nAsk (or type quit): ")
            if question.strip().lower() == "quit":
                break

            results = search(question, chunks, embeddings, k=5)
            for chunk, score in results:
                print(f"\n--- score {score:.2f} ---")
                print(chunk)

            top_chunks = [chunk for chunk, score in results]
            prompt = build_prompt(question, top_chunks)
            try:
                answer = ask_model(prompt)
                print("\nAnswer:", answer)
            except Exception:
                 print("\nCould not reach the model. Is Ollama running?")