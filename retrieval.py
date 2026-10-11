import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks):
    return model.encode(chunks, normalize_embeddings=True)


def search(query, chunks, embeddings, k=3):
    query_embedding = model.encode(query, normalize_embeddings=True)
    scores = embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i])) for i in top_indices]