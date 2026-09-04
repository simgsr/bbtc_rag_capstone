"""ChromaDB vector store for sermon chunks and Bible verses.

``SermonVectorStore`` wraps a persistent ChromaDB client holding two
collections — ``sermon_collection`` (NG body chunks + LLM summaries) and
``bible_collection`` (per-verse text across translations) — and provides the
upsert + semantic-search methods used by the ingest pipeline and the agent's
vector/bible tools.

Embeddings are computed here (not by Chroma) so all documents and queries share
one backend, selected via ``EMBED_BACKEND`` (``st`` | ``mlx_bge`` | ``mlx_qwen``;
see the constants below). Init is LAZY — the model is only loaded on the first
upsert/search — so importing this module never forces a model load. Switching
backend changes the vector space and requires a wipe + re-ingest of BOTH
collections so stored and query vectors remain comparable.
"""
import chromadb
import os

# Load .env here (not just in app.py / src.llm) so EVERY entry point that touches
# the vector store — ingest.py, backfill scripts, the eval harness — resolves
# EMBED_BACKEND identically. Without this, a standalone script that imports
# SermonVectorStore without going through app/llm would fall back to the "st"
# default and silently mix embedding backends within a collection (same dim, so
# the dim guard below wouldn't catch it → degraded retrieval consistency).
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # dotenv optional; env may already be configured

# Embedding backend, selected via EMBED_BACKEND (default: current sentence-transformers path).
# NB: all vectors in a collection MUST come from one backend — switching requires a wipe +
# re-ingest (`ingest.py --wipe`, `bible_ingest.py --wipe`) so query and doc vectors match.
#   st        → sentence-transformers BAAI/bge-m3  (PyTorch MPS, fp32) — the original default
#   mlx_bge   → mlx-community/bge-m3-mlx-fp16       (MLX, Apple Silicon) — ~2x faster, 1024-dim
#   mlx_qwen  → mlx-community/Qwen3-Embedding-8B-4bit-DWQ (MLX) — higher quality, 4096-dim, heavy
_MLX_DEFAULT_MODELS = {
    "mlx_bge": "mlx-community/bge-m3-mlx-fp16",
    "mlx_qwen": "mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
}

# Distance metric for both collections. BGE-M3 (and the MLX variants) are trained
# for COSINE similarity; Chroma's default is squared-L2, which ranks unnormalised
# embeddings differently and degrades retrieval. The space is baked in at collection
# creation — a collection built before this default (on l2) is NOT silently upgraded
# by passing this metadata to get_or_create; migrate it in place, preserving the
# stored vectors, with `python scripts/migrate_chroma_cosine.py`.
_DISTANCE_SPACE = "cosine"


class _MLXEmbedder:
    """Adapter exposing a sentence-transformers-style `.encode()` over mlx_embeddings."""

    def __init__(self, repo: str, max_length: int = 1024):
        from mlx_embeddings.utils import load
        self._model, self._tok = load(repo)
        self._max_length = max_length

    def encode(self, texts: list[str]):
        import mlx.core as mx
        from mlx_embeddings.utils import generate
        out = generate(self._model, self._tok, list(texts), max_length=self._max_length)
        emb = getattr(out, "text_embeds", None)
        if emb is None:  # models that only expose hidden states → mean-pool
            emb = out.last_hidden_state.mean(axis=1)
        mx.eval(emb)
        return emb  # mx.array supports .tolist()


class SermonVectorStore:
    def __init__(self, persist_dir: str = "data/chroma_db", embeddings=None):
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embeddings = embeddings  # None → lazy-init on first use per EMBED_BACKEND
        _space = {"hnsw:space": _DISTANCE_SPACE}
        self._sermons = self._client.get_or_create_collection("sermon_collection", metadata=_space)
        self._bible = self._client.get_or_create_collection("bible_collection", metadata=_space)
        self._check_distance_space()

    def _check_distance_space(self):
        """Fail fast if an existing non-empty collection uses a distance space other
        than ``_DISTANCE_SPACE``. Chroma bakes ``hnsw:space`` in at creation time and
        silently ignores a different ``metadata`` passed to ``get_or_create``, so a
        collection built before the cosine switch stays on l2 — quietly degrading
        BGE-M3 retrieval. Cheap (reads config + count, no model load), so it runs in
        ``__init__``. Empty collections are skipped so a fresh DB never trips it.
        """
        for coll in (self._sermons, self._bible):
            try:
                if coll.count() == 0:
                    continue
                cfg = getattr(coll, "configuration_json", None) or {}
                space = (cfg.get("hnsw") or {}).get("space")
            except Exception:
                continue  # config unreadable on this Chroma build → don't block
            if space and space != _DISTANCE_SPACE:
                raise RuntimeError(
                    f"Collection '{coll.name}' uses distance space '{space}', but this "
                    f"build expects '{_DISTANCE_SPACE}' (BGE-M3 is trained for cosine). "
                    f"Migrate the existing vectors in place (no re-embedding needed):\n"
                    f"    python scripts/migrate_chroma_cosine.py"
                )

    def _ensure_embeddings(self):
        if self._embeddings is not None:
            return
        backend = os.getenv("EMBED_BACKEND", "st").lower()
        if backend in ("st", "hf", "sentence-transformers"):
            from sentence_transformers import SentenceTransformer
            print("🍎 Loading BGE-M3 embeddings (sentence-transformers / MPS)...", flush=True)
            self._embeddings = SentenceTransformer("BAAI/bge-m3")
        elif backend in _MLX_DEFAULT_MODELS:
            repo = os.getenv("MLX_EMBED_MODEL", _MLX_DEFAULT_MODELS[backend])
            max_len = int(os.getenv("MLX_EMBED_MAX_LEN", "1024"))
            print(f"🍎 Loading MLX embeddings: {repo} (max_length={max_len})...", flush=True)
            self._embeddings = _MLXEmbedder(repo, max_length=max_len)
        else:
            raise ValueError(
                f"Unknown EMBED_BACKEND={backend!r}. "
                f"Expected 'st', {', '.join(repr(k) for k in _MLX_DEFAULT_MODELS)}."
            )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_embeddings()
        if not getattr(self, "_dim_checked", False):
            self._check_vector_dim_alignment()
            self._dim_checked = True
        safe_texts = [t[:8000] for t in texts]
        return self._embeddings.encode(safe_texts).tolist()

    def _check_vector_dim_alignment(self):
        """Fail fast if the current embedder's dimensionality doesn't match what's
        already stored in a collection. This catches the footgun where EMBED_BACKEND
        was switched (e.g. ``st`` 1024-dim → ``mlx_qwen`` 4096-dim) without the
        required wipe + re-ingest — which would otherwise silently produce garbage
        similarity scores or a dimension-mismatch crash deep inside a query.

        Only checked against non-empty collections so a fresh DB never trips it.
        Runs once per process (guarded by ``self._dim_checked``).
        """
        # Skip the probe entirely if every collection is empty (e.g. a fresh
        # `ingest.py --wipe`) — otherwise we'd pay a full BGE-M3 forward pass just
        # to read a dim that's never compared against anything.
        if self._sermons.count() == 0 and self._bible.count() == 0:
            return
        # Prefer a static attribute when the backend exposes one (sentence-transformers
        # caches the dim; no forward pass). Fall back to a 1-string encode otherwise.
        dim = None
        getter = getattr(self._embeddings, "get_sentence_embedding_dimension", None)
        if callable(getter):
            try:
                dim = int(getter())
            except Exception:
                dim = None
        if dim is None:
            sample = self._embeddings.encode(["dim probe"])
            shape = getattr(sample, "shape", None)
            dim = shape[-1] if shape else len(sample[0])
        for coll in (self._sermons, self._bible):
            if coll.count() == 0:
                continue
            try:
                peek = coll.get(include=["embeddings"], limit=1)
                stored = peek.get("embeddings") or []
            except Exception:
                continue  # older Chroma may store embeddings elsewhere; skip silently
            if stored and len(stored[0]) != dim:
                raise RuntimeError(
                    f"Embedding-dimension mismatch on '{coll.name}': stored vectors are "
                    f"{len(stored[0])}-dim but the current EMBED_BACKEND produces {dim}-dim. "
                    f"You switched embedding backends without wiping. Re-ingest with: "
                    f"`ingest.py --wipe` and `python -m src.ingestion.bible.bible_ingest --wipe`."
                )

    _MAX_BATCH = 100

    def _upsert_in_batches(self, collection, chunks: list[str], metadatas: list[dict], ids: list[str]):
        for start in range(0, len(chunks), self._MAX_BATCH):
            end = start + self._MAX_BATCH
            batch = chunks[start:end]
            collection.upsert(
                documents=batch,
                embeddings=self._embed(batch),
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )

    def upsert_sermon_chunks(self, chunks: list[str], metadatas: list[dict], ids: list[str]):
        self._upsert_in_batches(self._sermons, chunks, metadatas, ids)

    def delete_sermon_chunks(self, sermon_id: str):
        """Delete all Chroma chunks for a sermon (matched by metadata ``sermon_id``).

        Used by force re-ingest to remove the previously-indexed version of a
        sermon whose ``sermon_id`` may have changed (e.g. vision recovered a
        better topic). No-op if the sermon has no chunks.
        """
        try:
            self._sermons.delete(where={"sermon_id": sermon_id})
        except Exception:
            pass  # nothing to delete / collection not ready

    def upsert_bible_chunks(self, chunks: list[str], metadatas: list[dict], ids: list[str]):
        self._upsert_in_batches(self._bible, chunks, metadatas, ids)

    def _search(self, collection, query: str, k: int, where: dict | None) -> list[dict]:
        n = collection.count()
        if n == 0:
            return []
        k_fetch = min(max(k * 3, 12), n)
        kwargs = {
            "n_results": k_fetch,
            "include": ["documents", "metadatas", "distances"],
            "query_embeddings": [self._embed([query])[0]],
        }
        if where:
            kwargs["where"] = where
        results = collection.query(**kwargs)
        candidates = [
            {"content": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
        ]
        return candidates[:k]

    def search_sermons(self, query: str, k: int = 4, where: dict | None = None) -> list[dict]:
        return self._search(self._sermons, query, k, where)

    def search_bible(self, query: str, k: int = 4, where: dict | None = None) -> list[dict]:
        return self._search(self._bible, query, k, where)

    def get_bible_versions(self, reference: str) -> list[dict]:
        results = self._bible.get(
            where={"reference": reference},
            include=["documents", "metadatas"]
        )
        return [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]

    def counts(self) -> dict:
        return {
            "sermon_collection": self._sermons.count(),
            "bible_collection": self._bible.count(),
        }
