"""Shared embedding-model identity (PLAN.md §4.5).

A single source of truth for which model embeds a chunk/query, so the
ingest pipeline (`ingest/index.py`, bulk-embeds the corpus once) and the RAG
subgraph (`rag/embeddings.py`, embeds one query per request) can never
silently drift onto different vector spaces. `get_embedding_model` is the
one factory both call — swapping in `multilingual-e5-base` (PLAN's
documented option if the demo audience wants Hungarian) is then a one-line
change here rather than a hunt across both call sites.

CPU/ONNX by design (§4.5): the API image stays free of PyTorch/CUDA.
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768


@lru_cache
def get_embedding_model() -> TextEmbedding:
    """Cached: loading the ONNX model is a one-off cost worth paying once."""
    return TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
