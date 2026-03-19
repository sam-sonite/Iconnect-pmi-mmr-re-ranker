# PMI MMR Reranker

A lightweight Python library for document retrieval using **PMI (Pointwise Mutual Information)** and **MMR (Maximal Marginal Relevance)** reranking.

---

## Installation

```bash
pip install .
```

---

## Basic Usage

```python
from collections import Counter
import numpy as np
from pmi_mmr_reranker import retrieve_with_pmi_mmr

documents = [
    "AI creates new professions",
    "Jobs in artificial intelligence",
    "Career opportunities in AI industry",
]

embeddings = np.random.rand(len(documents), 128).astype("float32")
query_embedding = np.random.rand(128).astype("float32")

corpus_counts = Counter({
    ("ai", "jobs"): 10,
    "ai": 50,
    "jobs": 40,
})

results = retrieve_with_pmi_mmr(
    query="profession creation",
    query_embedding=query_embedding,
    documents=documents,
    embeddings=embeddings,
    corpus_counts=corpus_counts,
    total_tokens=1000,
)

print(results)
```

---

## Features

- Combines semantic similarity with statistical relevance  
- Uses PMI for keyword importance  
- Applies MMR to reduce redundancy  
- Simple and easy to integrate  

---

## Exports

- `compute_pmi`  
- `cosine_sim`  
- `pmi_mmr_rerank`  
- `faiss_retrieve`  
- `retrieve_with_pmi_mmr`  

---

## Requirements

- Python 3.8+  
- NumPy  

---

## Notes

This project is experimental and not production-ready. Use at your own risk.

---

## License

MIT License
