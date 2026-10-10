from pypdf import PdfReader
from pathlib import Path

BASE_DIR = Path(__file__).parent

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
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
    result = extract_text(BASE_DIR / "data" / "cn-notes.pdf")
    print("Characters:", len(result))

    if len(result.strip()) < 100:
        print("This PDF looks scanned, so I can't read the text. Please upload a text-based PDF.")
    else:
        chunks = chunk_text(result)
        print("Total chunks:", len(chunks))
        print(chunks[0][:300])
        if len(chunks) > 1:
            print("Overlap OK:", chunks[0][-200:] == chunks[1][:200])