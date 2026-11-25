import numpy as np
from retrieval import embed_chunks, search

CHUNKS = [
    "Python is a programming language",
    "The Eiffel Tower is in Paris, France",
    "Bananas are yellow fruits",
]


def test_embedding_shape():
    emb = embed_chunks(CHUNKS)
    assert emb.shape == (3, 384)


def test_embeddings_are_normalized():
    emb = embed_chunks(CHUNKS)
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)


def test_search_returns_k_results():
    emb = embed_chunks(CHUNKS)
    results = search("tower in France", CHUNKS, emb, k=2)
    assert len(results) == 2


def test_search_scores_sorted_high_to_low():
    emb = embed_chunks(CHUNKS)
    results = search("tower in France", CHUNKS, emb, k=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_finds_right_chunk():
    emb = embed_chunks(CHUNKS)
    top_chunk, _ = search("Which tower is in France?", CHUNKS, emb, k=1)[0]
    assert "Eiffel" in top_chunk


def test_search_k_larger_than_chunks():
    emb = embed_chunks(CHUNKS)
    results = search("fruit", CHUNKS, emb, k=10)
    assert len(results) == 3
