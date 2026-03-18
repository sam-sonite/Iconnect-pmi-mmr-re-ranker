from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def compute_pmi(query_tokens, doc_tokens, corpus_counts, total_tokens):
    score = 0.0
    for x in query_tokens:
        for y in doc_tokens:
            p_xy = corpus_counts[(x, y)] / total_tokens if (x, y) in corpus_counts else 1e-9
            p_x = corpus_counts[x] / total_tokens if x in corpus_counts else 1e-9
            p_y = corpus_counts[y] / total_tokens if y in corpus_counts else 1e-9
            score += math.log(p_xy / (p_x * p_y + 1e-9) + 1e-9)
    return score / (len(query_tokens) + 1e-9)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def pmi_mmr_rerank(
    query: str,
    docs: Sequence[str],
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    corpus_counts,
    total_tokens,
    lambda_param: float = 0.7,
    top_m: int = 5,
):
    del query_embedding
    query_tokens = query.lower().split()

    pmi_scores = []
    for doc in docs:
        doc_tokens = doc.lower().split()
        pmi_scores.append(compute_pmi(query_tokens, doc_tokens, corpus_counts, total_tokens))

    selected = []
    selected_indices = []

    first_idx = int(np.argmax(pmi_scores))
    selected.append(docs[first_idx])
    selected_indices.append(first_idx)

    while len(selected) < min(top_m, len(docs)):
        best_score = -1e9
        best_idx = -1

        for i in range(len(docs)):
            if i in selected_indices:
                continue

            relevance = pmi_scores[i]
            max_sim = max(
                [cosine_sim(embeddings[i], embeddings[j]) for j in selected_indices]
            ) if selected_indices else 0

            score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx == -1:
            break

        selected.append(docs[best_idx])
        selected_indices.append(best_idx)

    return selected


def faiss_retrieve(query_embedding: np.ndarray, doc_embeddings: np.ndarray, k: int = 10):
    try:
        import faiss

        dim = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(doc_embeddings.astype(np.float32))
        _, indices = index.search(np.array([query_embedding]).astype(np.float32), k)
        return indices[0]
    except ModuleNotFoundError:
        scores = np.dot(doc_embeddings.astype(np.float32), query_embedding.astype(np.float32))
        top_indices = np.argsort(scores)[::-1][:k]
        return top_indices


def retrieve_with_pmi_mmr(
    query: str,
    query_embedding: np.ndarray,
    documents: Sequence[str],
    embeddings: np.ndarray,
    corpus_counts,
    total_tokens,
    top_k: int = 10,
    top_m: int = 5,
):
    top_k_indices = faiss_retrieve(query_embedding, embeddings, top_k)
    top_k_docs = [documents[i] for i in top_k_indices]
    top_k_embeds = embeddings[top_k_indices]

    return pmi_mmr_rerank(
        query,
        top_k_docs,
        top_k_embeds,
        query_embedding,
        corpus_counts,
        total_tokens,
        top_m=top_m,
    )
