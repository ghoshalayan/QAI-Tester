"""Per-project, per-namespace FAISS store with disk persistence.

Layout
------
    data/faiss/<project_id>/<namespace>.faiss

- ``project_id`` is the SQLite ``projects.id``.
- ``namespace`` is a string (e.g. ``"requirements"``, ``"tc_nodes"``,
  ``"brd_chunks"``) so different content types within a project don't share
  an ID space — domain primary keys can collide between tables.

Index type
----------
``IndexIDMap`` over ``IndexFlatIP``. Why:

- ``IndexFlatIP`` does exact (not approximate) inner-product search. Since the
  ``BGEEmbedder`` L2-normalizes its outputs, IP equals cosine similarity.
- ``IndexIDMap`` lets us add/search using our domain integer IDs (e.g.
  ``requirements.id``) directly — no extra position-to-id table needed.
- For tens-of-thousands of vectors per namespace this is plenty fast on CPU.
  Swap to HNSW only if a project genuinely outgrows it.

Concurrency
-----------
One ``threading.Lock`` per ``(project_id, namespace)``. FAISS reads are not
thread-safe, and we mutate the on-disk file too, so we serialize all access.
A registry of locks is itself protected by ``_dict_lock``.

Persistence
-----------
Saved to disk on every modification. Local single-user MVP — durability beats
throughput. If this becomes a hot path we'll add debouncing.
"""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import settings

if TYPE_CHECKING:
    import faiss
    import numpy as np

logger = logging.getLogger(__name__)

DIM = 1024  # bge-large-en-v1.5 output dimension


class FAISSStore:
    """Process-wide singleton holding loaded FAISS indices in memory."""

    def __init__(self) -> None:
        self._indices: dict[tuple[int, str], "faiss.IndexIDMap"] = {}
        self._locks: dict[tuple[int, str], threading.Lock] = {}
        self._dict_lock = threading.Lock()

    # ── path & lock helpers ────────────────────────────────────────

    def _path(self, project_id: int, namespace: str) -> Path:
        return settings.faiss_dir / str(project_id) / f"{namespace}.faiss"

    def _lock_for(self, project_id: int, namespace: str) -> threading.Lock:
        key = (project_id, namespace)
        with self._dict_lock:
            if key not in self._locks:
                self._locks[key] = threading.Lock()
            return self._locks[key]

    # ── load/save ─────────────────────────────────────────────────

    def _load_or_create(self, project_id: int, namespace: str) -> "faiss.IndexIDMap":
        import faiss

        key = (project_id, namespace)
        if key in self._indices:
            return self._indices[key]

        path = self._path(project_id, namespace)
        if path.exists():
            logger.info("FAISS load: %s", path)
            index = faiss.read_index(str(path))
        else:
            logger.info("FAISS create: %s", path)
            base = faiss.IndexFlatIP(DIM)
            index = faiss.IndexIDMap(base)

        self._indices[key] = index
        return index

    def _save(self, project_id: int, namespace: str) -> None:
        import faiss

        key = (project_id, namespace)
        if key not in self._indices:
            return
        path = self._path(project_id, namespace)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._indices[key], str(path))

    # ── private mutators (lock must already be held) ──────────────

    def _add_locked(
        self, project_id: int, namespace: str, ids: list[int], vectors: "np.ndarray",
    ) -> None:
        import numpy as np

        ids_arr = np.asarray(ids, dtype="int64")
        index = self._load_or_create(project_id, namespace)
        index.add_with_ids(vectors.astype("float32"), ids_arr)

    def _remove_locked(
        self, project_id: int, namespace: str, ids: list[int],
    ) -> int:
        import numpy as np

        if not ids:
            return 0
        index = self._load_or_create(project_id, namespace)
        ids_arr = np.asarray(ids, dtype="int64")
        try:
            return int(index.remove_ids(ids_arr))
        except Exception as e:
            logger.warning("FAISS remove failed (project=%s ns=%s): %s",
                           project_id, namespace, e)
            return 0

    # ── public API ────────────────────────────────────────────────

    def add(
        self,
        project_id: int,
        namespace: str,
        ids: list[int],
        vectors: "np.ndarray",
    ) -> None:
        """Append vectors. Does NOT remove existing entries with the same id —
        use :meth:`upsert` for replace-or-insert semantics."""
        if not ids:
            return
        if vectors.shape[0] != len(ids):
            raise ValueError("ids and vectors must have same length")
        if vectors.shape[1] != DIM:
            raise ValueError(f"vectors must have dim {DIM}, got {vectors.shape[1]}")

        with self._lock_for(project_id, namespace):
            self._add_locked(project_id, namespace, ids, vectors)
            self._save(project_id, namespace)

    def upsert(
        self,
        project_id: int,
        namespace: str,
        ids: list[int],
        vectors: "np.ndarray",
    ) -> None:
        """Remove any existing rows with these ids, then add the new vectors."""
        if not ids:
            return
        if vectors.shape[0] != len(ids):
            raise ValueError("ids and vectors must have same length")
        if vectors.shape[1] != DIM:
            raise ValueError(f"vectors must have dim {DIM}, got {vectors.shape[1]}")

        with self._lock_for(project_id, namespace):
            self._remove_locked(project_id, namespace, ids)
            self._add_locked(project_id, namespace, ids, vectors)
            self._save(project_id, namespace)

    def remove(
        self, project_id: int, namespace: str, ids: list[int],
    ) -> int:
        """Remove vectors by id. Returns the number actually removed."""
        with self._lock_for(project_id, namespace):
            n = self._remove_locked(project_id, namespace, ids)
            self._save(project_id, namespace)
            return n

    def search(
        self,
        project_id: int,
        namespace: str,
        query_vec: "np.ndarray",
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Top-k nearest neighbors. Returns ``[(domain_id, score), ...]`` desc by score."""
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.shape[1] != DIM:
            raise ValueError(f"query vec must have dim {DIM}, got {query_vec.shape[1]}")

        with self._lock_for(project_id, namespace):
            index = self._load_or_create(project_id, namespace)
            if index.ntotal == 0:
                return []
            scores, ids = index.search(
                query_vec.astype("float32"), min(k, index.ntotal),
            )

        out: list[tuple[int, float]] = []
        for score, id_ in zip(scores[0], ids[0]):
            if id_ == -1:
                continue
            out.append((int(id_), float(score)))
        return out

    def count(self, project_id: int, namespace: str) -> int:
        with self._lock_for(project_id, namespace):
            try:
                index = self._load_or_create(project_id, namespace)
                return int(index.ntotal)
            except Exception:
                return 0

    def reset(self, project_id: int, namespace: str | None = None) -> None:
        """Wipe one namespace, or all namespaces for a project when ``namespace`` is None."""
        if namespace is not None:
            with self._lock_for(project_id, namespace):
                self._indices.pop((project_id, namespace), None)
                p = self._path(project_id, namespace)
                if p.exists():
                    p.unlink()
            return

        # Wipe everything for the project
        keys = [k for k in list(self._indices.keys()) if k[0] == project_id]
        for key in keys:
            with self._lock_for(*key):
                self._indices.pop(key, None)
        project_dir = settings.faiss_dir / str(project_id)
        if project_dir.exists():
            shutil.rmtree(project_dir)


# Process-wide singleton
_singleton = FAISSStore()


def get_store() -> FAISSStore:
    return _singleton
