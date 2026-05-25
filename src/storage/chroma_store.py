# src/storage/chroma_store.py
import chromadb
import os
import subprocess
import time

def _ensure_ollama(timeout: int = 20) -> bool:
    try:
        import requests
        requests.get("http://127.0.0.1:11434/", timeout=1)
        return True
    except Exception:
        print("🦙 Ollama not running — starting it now...")
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            print("❌ Ollama is not installed! Please install it from https://ollama.com")
            return False
        start = time.time()
        while time.time() - start < timeout:
            try:
                requests.get("http://127.0.0.1:11434/", timeout=1)
                print("🦙 Ollama is ready.")
                return True
            except Exception:
                time.sleep(1)
        print("⚠️  Ollama did not start within the timeout.")
        return False

class SermonVectorStore:
    def __init__(self, persist_dir: str = "data/chroma_db", embeddings=None):
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        if embeddings is not None:
            self._embeddings = embeddings
        else:
            # Defer embedding init — only connect to BGE-M3 on first actual query
            self._embeddings = None
        
        self._sermons = self._client.get_or_create_collection("sermon_collection")
        self._bible = self._client.get_or_create_collection("bible_collection")

    def _ensure_embeddings(self):
        if self._embeddings is not None:
            return
        _ensure_ollama()
        try:
            from langchain_ollama import OllamaEmbeddings
            self._embeddings = OllamaEmbeddings(model="BGE-M3", client_kwargs={"timeout": 60})
            self._embeddings.embed_query("test")
        except Exception as e:
            print(f"🦙 BGE-M3 not ready, attempting pull... ({e})")
            try:
                subprocess.run(["ollama", "pull", "bge-m3"], check=True)
                from langchain_ollama import OllamaEmbeddings
                self._embeddings = OllamaEmbeddings(model="BGE-M3", client_kwargs={"timeout": 60})
                self._embeddings.embed_query("test")
            except Exception as pull_err:
                raise RuntimeError(
                    f"BGE-M3 unavailable. Start Ollama and run: ollama pull bge-m3 ({pull_err})"
                )

    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if self._embeddings:
            safe_texts = [t[:8000] for t in texts]
            return self._embeddings.embed_documents(safe_texts)
        return None

    _MAX_BATCH = 100

    def _upsert_in_batches(self, collection, chunks: list[str], metadatas: list[dict], ids: list[str]):
        self._ensure_embeddings()
        for start in range(0, len(chunks), self._MAX_BATCH):
            end = start + self._MAX_BATCH
            batch_chunks = chunks[start:end]
            batch_embeddings = self._embed(batch_chunks)
            
            kwargs = {
                "documents": batch_chunks,
                "metadatas": metadatas[start:end],
                "ids": ids[start:end],
            }
            if batch_embeddings:
                kwargs["embeddings"] = batch_embeddings
            
            collection.upsert(**kwargs)

    def upsert_sermon_chunks(self, chunks: list[str], metadatas: list[dict], ids: list[str]):
        self._upsert_in_batches(self._sermons, chunks, metadatas, ids)

    def upsert_bible_chunks(self, chunks: list[str], metadatas: list[dict], ids: list[str]):
        self._upsert_in_batches(self._bible, chunks, metadatas, ids)

    def _search(self, collection, query: str, k: int, where: dict | None) -> list[dict]:
        n = collection.count()
        if n == 0:
            return []

        self._ensure_embeddings()
        k_fetch = min(max(k * 3, 12), n)
        kwargs = {
            "n_results": k_fetch,
            "include": ["documents", "metadatas", "distances"],
            "query_embeddings": [self._embeddings.embed_query(query)],
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
        """Returns all versions of a specific verse reference string (e.g. '1 John 1:9')."""
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
