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
        self._sermons = self._client.get_or_create_collection("sermon_collection")
        self._bible = self._client.get_or_create_collection("bible_collection")

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
        safe_texts = [t[:8000] for t in texts]
        return self._embeddings.encode(safe_texts).tolist()

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
