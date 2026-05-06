"""sentence-transformers wrapper for BAAI/bge-large-en-v1.5.

- **Lazy load** on first ``embed_*`` call (model weights are ~1.3 GB and
  loading takes 5–10 s on CPU; we don't want to pay that at server boot).
- **Thread-safe** singleton — only one process-wide instance.
- **L2-normalized** outputs so cosine similarity == inner product. This means
  FAISS ``IndexFlatIP`` gives correct cosine ranking with no extra work.
- **Query prefix** per the BGE model card: queries get
  ``"Represent this sentence for searching relevant passages: "`` prepended;
  documents do NOT. Mixing them up degrades retrieval quality.

First run downloads weights to ``~/.cache/huggingface/hub/`` (or
``HF_HOME``). After that, subsequent loads are offline.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BGEEmbedder:
    """Singleton wrapper around BAAI/bge-large-en-v1.5."""

    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    DIM = 1024
    DEVICE = "cpu"

    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> "SentenceTransformer":
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                logger.info(
                    "Loading embedding model %s on %s (first call may take 5–10 s)...",
                    self.MODEL_NAME,
                    self.DEVICE,
                )
                self._model = SentenceTransformer(self.MODEL_NAME, device=self.DEVICE)
                logger.info(
                    "Embedding model ready (dim=%d, device=%s)",
                    self.DIM,
                    self.DEVICE,
                )
        return self._model

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> "np.ndarray":
        """Embed documents (no query prefix). Returns shape (n, 1024), L2-normalized."""
        if not texts:
            import numpy as np
            return np.empty((0, self.DIM), dtype="float32")

        model = self._ensure_model()
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")

    def embed_query(self, text: str) -> "np.ndarray":
        """Embed a single query (with BGE query prefix). Returns shape (1024,), L2-normalized."""
        model = self._ensure_model()
        return model.encode(
            _BGE_QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# Process-wide singleton — share across requests.
_singleton = BGEEmbedder()


def get_embedder() -> BGEEmbedder:
    return _singleton
