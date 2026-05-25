# src/storage/chroma_store.py
import chromadb
import os

class SermonVectorStore:
    def __init__(self, persist_dir: str = "data/chroma_db", embeddings=None):
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embeddings = embeddings  # None → lazy-init BGE-M3 on first use
        self._sermons = self._client.get_or_create_collection("sermon_collection")
        self._bible = self._client.get_or_create_collection("bible_collection")

    def _ensure_embeddings(self):
        if self._embeddings is not None:
            return
        from sentence_transformers import SentenceTransformer
        print("🍎 Loading BGE-M3 embeddings (MPS)...", flush=True)
        self._embeddings = SentenceTransformer("BAAI/bge-m3")

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
