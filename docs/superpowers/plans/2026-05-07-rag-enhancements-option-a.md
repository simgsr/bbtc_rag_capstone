# RAG Enhancements Option A — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement three independent, opt-in RAG enhancement techniques — Contextual Retrieval, Proposition-based Chunking, and RAPTOR — each in its own module and ChromaDB collection so they can be compared against the existing baseline.

**Architecture:** Each technique is an independent opt-in module activated by a CLI flag (`--contextual`, `--proposition`, `--raptor`). Contextual Retrieval and Proposition Chunking run at ingest time and write to separate ChromaDB collections (`contextual_collection`, `proposition_collection`). RAPTOR runs as a post-processing step after normal ingest and writes cluster summaries to `raptor_collection`. The existing `sermon_collection` and all existing code are unchanged. A `collection` parameter added to `search_sermons_tool` lets the agent compare retrieval approaches side by side.

**Tech Stack:** Python, ChromaDB, LangGraph, LangChain Ollama (BGE-M3), umap-learn, hdbscan, pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/ingestion/contextual_retrieval.py` | `add_context_to_chunk(chunk, full_document, llm) → str` |
| Create | `src/ingestion/proposition_chunker.py` | `extract_propositions(text, llm) → list[str]` |
| Create | `src/ingestion/raptor.py` | `build_raptor_tree(chunks, llm, embed_fn) → list[dict]` |
| Create | `tests/test_contextual_retrieval.py` | Unit tests for contextual_retrieval |
| Create | `tests/test_proposition_chunker.py` | Unit tests for proposition_chunker |
| Create | `tests/test_raptor.py` | Unit tests for raptor |
| Modify | `src/storage/chroma_store.py` | Add 3 new collections + their upsert/search/get methods |
| Modify | `ingest.py` | Add `--contextual`, `--proposition`, `--raptor` CLI flags |
| Modify | `src/tools/vector_tool.py` | Add `collection` parameter to `search_sermons_tool` |
| Modify | `requirements.txt` | Add `umap-learn` and `hdbscan` |

---

## Task 1: Feature Branch + Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout -b feature/rag-enhancements-option-a
```

- [ ] **Step 2: Add new dependencies to requirements.txt**

Open `requirements.txt` and append two lines at the end:

```
umap-learn
hdbscan
```

- [ ] **Step 3: Install the new dependencies**

```bash
pip install umap-learn hdbscan
```

Expected: Both packages install without errors. `umap-learn` brings in `numba` as a dependency (large download, ~200 MB); this is normal.

- [ ] **Step 4: Verify imports work**

```bash
python -c "import umap; import hdbscan; print('OK')"
```

Expected output: `OK`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "feat: add umap-learn and hdbscan for RAPTOR clustering"
```

---

## Task 2: Contextual Retrieval Module

**Files:**
- Create: `src/ingestion/contextual_retrieval.py`
- Create: `tests/test_contextual_retrieval.py`

**What it does:** Given a single chunk and the full document it came from, asks the LLM to write 1-2 sentences placing the chunk in context of the document, then prepends that context to the chunk. This makes the embedding carry document-level awareness.

- [ ] **Step 1: Write the failing test**

Create `tests/test_contextual_retrieval.py`:

```python
from unittest.mock import MagicMock
from src.ingestion.contextual_retrieval import add_context_to_chunk


def _make_llm(response_text: str):
    llm = MagicMock()
    response = MagicMock()
    response.content = response_text
    llm.invoke.return_value = response
    return llm


def test_prepends_context_to_chunk():
    llm = _make_llm("This chunk discusses Jesus's call to discipleship.")
    result = add_context_to_chunk("Take up your cross.", "Full document...", llm)
    assert result.startswith("This chunk discusses")
    assert "Take up your cross." in result


def test_context_and_chunk_separated_by_blank_line():
    llm = _make_llm("Context sentence here.")
    result = add_context_to_chunk("chunk text", "doc text", llm)
    assert "\n\nchunk text" in result


def test_fallback_returns_original_chunk_on_llm_error():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("LLM unavailable")
    result = add_context_to_chunk("chunk text", "doc text", llm)
    assert result == "chunk text"


def test_truncates_long_document_in_prompt():
    llm = _make_llm("context")
    long_doc = "x" * 10000
    add_context_to_chunk("chunk", long_doc, llm)
    prompt_sent = llm.invoke.call_args[0][0]
    # document portion in prompt must be capped
    assert len(prompt_sent) < 9000


def test_llm_called_exactly_once():
    llm = _make_llm("context")
    add_context_to_chunk("chunk", "doc", llm)
    llm.invoke.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they all fail**

```bash
pytest tests/test_contextual_retrieval.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.ingestion.contextual_retrieval'`

- [ ] **Step 3: Implement the module**

Create `src/ingestion/contextual_retrieval.py`:

```python
def add_context_to_chunk(chunk: str, full_document: str, llm) -> str:
    """Returns the chunk prefixed with a 1-2 sentence document-level context."""
    prompt = (
        "<document>\n"
        f"{full_document[:3000]}\n"
        "</document>\n\n"
        "Here is the chunk we want to situate within the whole document:\n"
        "<chunk>\n"
        f"{chunk}\n"
        "</chunk>\n\n"
        "Give a short succinct context (1-2 sentences) to situate this chunk "
        "within the overall document for better search retrieval. "
        "Answer only with the succinct context and nothing else."
    )
    try:
        response = llm.invoke(prompt)
        context = (response.content if hasattr(response, "content") else str(response)).strip()
        return f"{context}\n\n{chunk}"
    except Exception:
        return chunk
```

- [ ] **Step 4: Run tests to verify they all pass**

```bash
pytest tests/test_contextual_retrieval.py -v
```

Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/contextual_retrieval.py tests/test_contextual_retrieval.py
git commit -m "feat: add contextual_retrieval module for chunk context prepending"
```

---

## Task 3: Proposition Chunker Module

**Files:**
- Create: `src/ingestion/proposition_chunker.py`
- Create: `tests/test_proposition_chunker.py`

**What it does:** Instead of splitting by character window (800/150), asks the LLM to decompose the sermon body into atomic, self-contained propositions — one idea per chunk. Each proposition is fully understandable without surrounding context.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_proposition_chunker.py`:

```python
from unittest.mock import MagicMock
from src.ingestion.proposition_chunker import extract_propositions


def _make_llm(response_text: str):
    llm = MagicMock()
    response = MagicMock()
    response.content = response_text
    llm.invoke.return_value = response
    return llm


def test_returns_list_of_strings():
    llm = _make_llm("Jesus calls all believers to follow him.\nDiscipleship requires daily surrender.")
    result = extract_propositions("Some sermon text.", llm)
    assert isinstance(result, list)
    assert all(isinstance(p, str) for p in result)


def test_splits_on_newlines():
    llm = _make_llm("Proposition one.\nProposition two.\nProposition three.")
    result = extract_propositions("text", llm)
    assert len(result) == 3


def test_strips_empty_lines():
    llm = _make_llm("Prop one.\n\nProp two.\n\n")
    result = extract_propositions("text", llm)
    assert len(result) == 2


def test_fallback_returns_truncated_original_on_llm_error():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("LLM unavailable")
    result = extract_propositions("sermon text", llm)
    assert result == ["sermon text"]


def test_fallback_truncates_at_800_chars():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("LLM unavailable")
    long_text = "x" * 2000
    result = extract_propositions(long_text, llm)
    assert len(result[0]) == 800


def test_llm_called_once():
    llm = _make_llm("One proposition.")
    extract_propositions("some text", llm)
    llm.invoke.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_proposition_chunker.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.ingestion.proposition_chunker'`

- [ ] **Step 3: Implement the module**

Create `src/ingestion/proposition_chunker.py`:

```python
def extract_propositions(text: str, llm) -> list[str]:
    """Decompose sermon text into atomic, self-contained propositions.

    Each proposition expresses exactly one idea and is fully understandable
    without surrounding context. Falls back to a single truncated text on error.
    """
    prompt = (
        "Decompose the following sermon text into atomic, self-contained propositions. "
        "Each proposition should:\n"
        "- Express exactly one idea\n"
        "- Be fully understandable without the surrounding context\n"
        "- Include the subject and all necessary references\n\n"
        "Output one proposition per line. No numbering, no bullets, no blank lines.\n\n"
        f"Text:\n{text[:4000]}"
    )
    try:
        response = llm.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
        return lines if lines else [text[:800]]
    except Exception:
        return [text[:800]]
```

- [ ] **Step 4: Run tests to verify they all pass**

```bash
pytest tests/test_proposition_chunker.py -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/proposition_chunker.py tests/test_proposition_chunker.py
git commit -m "feat: add proposition_chunker module for atomic claim decomposition"
```

---

## Task 4: RAPTOR Core Module

**Files:**
- Create: `src/ingestion/raptor.py`
- Create: `tests/test_raptor.py`

**What it does:** Takes a flat list of chunks (content + embedding + metadata), reduces embedding dimensionality with UMAP, clusters with HDBSCAN, summarizes each cluster with the LLM, then re-embeds summaries for the next level — recursively up to `max_levels`. Returns a flat list of all cluster summaries across all levels.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_raptor.py`:

```python
import numpy as np
from unittest.mock import MagicMock, patch
from src.ingestion.raptor import (
    build_raptor_tree,
    _summarize_cluster,
    _cluster,
    _reduce_dims,
)


def _make_chunks(n: int) -> list[dict]:
    """Make n fake chunks with random 16-dim embeddings."""
    rng = np.random.default_rng(42)
    return [
        {
            "content": f"Sermon excerpt {i}. Jesus taught about discipleship.",
            "metadata": {"speaker": "Pastor A", "topic": f"Topic {i}", "year": 2024},
            "embedding": rng.random(16).tolist(),
        }
        for i in range(n)
    ]


def _make_llm(summary_text: str = "Cluster summary."):
    llm = MagicMock()
    response = MagicMock()
    response.content = summary_text
    llm.invoke.return_value = response
    return llm


def _make_embed_fn(dim: int = 16):
    rng = np.random.default_rng(0)
    def embed_fn(texts: list[str]) -> list[list[float]]:
        return rng.random((len(texts), dim)).tolist()
    return embed_fn


def test_build_raptor_tree_returns_list():
    chunks = _make_chunks(15)
    llm = _make_llm()
    result = build_raptor_tree(chunks, llm, _make_embed_fn())
    assert isinstance(result, list)


def test_build_raptor_tree_stops_when_too_few_chunks():
    # Only 5 chunks — below the 10-chunk threshold, no summaries should be created
    chunks = _make_chunks(5)
    llm = _make_llm()
    result = build_raptor_tree(chunks, llm, _make_embed_fn())
    assert result == []


def test_each_summary_has_required_keys():
    chunks = _make_chunks(20)
    llm = _make_llm("Summary of cluster.")
    results = build_raptor_tree(chunks, llm, _make_embed_fn(), max_levels=1)
    for item in results:
        assert "summary" in item
        assert "metadata" in item
        assert "embedding" in item


def test_summary_metadata_has_level():
    chunks = _make_chunks(20)
    llm = _make_llm("Summary.")
    results = build_raptor_tree(chunks, llm, _make_embed_fn(), max_levels=1)
    for item in results:
        assert item["metadata"]["level"] == 1


def test_summarize_cluster_returns_dict_on_success():
    llm = _make_llm("This cluster discusses grace.")
    texts = ["Grace is unmerited favor.", "We are saved by grace through faith."]
    metas = [{"speaker": "Pastor A", "topic": "Grace", "year": 2024}] * 2
    result = _summarize_cluster(texts, metas, cluster_id=0, level=1, llm=llm)
    assert result is not None
    assert "summary" in result
    assert result["summary"] == "This cluster discusses grace."


def test_summarize_cluster_returns_none_on_llm_error():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("LLM error")
    result = _summarize_cluster(["text"], [{}], cluster_id=0, level=1, llm=llm)
    assert result is None


def test_cluster_returns_array_of_labels():
    rng = np.random.default_rng(1)
    embeddings = rng.random((20, 2))
    labels = _cluster(embeddings)
    assert len(labels) == 20
    assert isinstance(labels[0], (int, np.integer))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_raptor.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.ingestion.raptor'`

- [ ] **Step 3: Implement the module**

Create `src/ingestion/raptor.py`:

```python
import numpy as np
from typing import Any, Callable


def _reduce_dims(embeddings: list[list[float]], n_components: int = 10) -> np.ndarray:
    """Reduce embedding dimensions with UMAP for clustering."""
    import umap
    arr = np.array(embeddings)
    n = min(n_components, arr.shape[0] - 2)
    reducer = umap.UMAP(n_components=max(n, 2), random_state=42, metric="cosine")
    return reducer.fit_transform(arr)


def _cluster(embeddings_2d: np.ndarray) -> np.ndarray:
    """Cluster embeddings with HDBSCAN. Returns label array (-1 means noise/unclustered)."""
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean")
    return clusterer.fit_predict(embeddings_2d)


def _summarize_cluster(
    texts: list[str],
    metadata_list: list[dict],
    cluster_id: int,
    level: int,
    llm: Any,
) -> dict | None:
    """Ask the LLM to summarize a cluster of related texts."""
    combined = "\n\n---\n\n".join(texts[:10])
    prompt = (
        "You are summarizing a cluster of related sermon excerpts. "
        "Write a comprehensive 3-5 sentence summary capturing the main themes, "
        "theological insights, and key points across all these excerpts.\n\n"
        f"Excerpts:\n{combined[:5000]}\n\n"
        "Summary:"
    )
    try:
        response = llm.invoke(prompt)
        summary = (response.content if hasattr(response, "content") else str(response)).strip()
        speakers = list({m.get("speaker", "") for m in metadata_list if m.get("speaker")})
        topics = list({m.get("topic", "") for m in metadata_list if m.get("topic")})
        years = sorted({m.get("year", 0) for m in metadata_list if m.get("year")})
        return {
            "summary": summary,
            "metadata": {
                "doc_type": "raptor_summary",
                "level": level,
                "cluster_id": cluster_id,
                "speakers": ", ".join(speakers[:5]),
                "topics": ", ".join(topics[:5]),
                "year_range": f"{min(years)}-{max(years)}" if years else "",
                "chunk_count": len(texts),
            },
        }
    except Exception:
        return None


def build_raptor_tree(
    all_chunks: list[dict],
    llm: Any,
    embed_fn: Callable[[list[str]], list[list[float]]],
    max_levels: int = 3,
) -> list[dict]:
    """Build RAPTOR tree from a flat list of {content, metadata, embedding} dicts.

    Returns a flat list of {summary, metadata, embedding} for every cluster
    summary created across all levels.
    """
    current_texts = [c["content"] for c in all_chunks]
    current_embeddings = [c["embedding"] for c in all_chunks]
    current_metas = [c["metadata"] for c in all_chunks]

    all_summaries: list[dict] = []

    for level in range(1, max_levels + 1):
        if len(current_texts) < 10:
            break

        embeddings_2d = _reduce_dims(current_embeddings)
        labels = _cluster(embeddings_2d)

        unique_labels = [lb for lb in set(labels.tolist()) if lb != -1]
        if not unique_labels:
            break

        level_summaries: list[dict] = []
        for cluster_id in unique_labels:
            indices = [i for i, lb in enumerate(labels.tolist()) if lb == cluster_id]
            result = _summarize_cluster(
                [current_texts[i] for i in indices],
                [current_metas[i] for i in indices],
                cluster_id,
                level,
                llm,
            )
            if result:
                level_summaries.append(result)

        if not level_summaries:
            break

        summary_texts = [s["summary"] for s in level_summaries]
        new_embeddings = embed_fn(summary_texts)
        for s, emb in zip(level_summaries, new_embeddings):
            s["embedding"] = emb

        all_summaries.extend(level_summaries)
        current_texts = summary_texts
        current_embeddings = new_embeddings
        current_metas = [s["metadata"] for s in level_summaries]

    return all_summaries
```

- [ ] **Step 4: Run tests to verify they all pass**

```bash
pytest tests/test_raptor.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/raptor.py tests/test_raptor.py
git commit -m "feat: add RAPTOR module for hierarchical cluster summarization"
```

---

## Task 5: Extend ChromaDB Store with New Collections

**Files:**
- Modify: `src/storage/chroma_store.py`

**What it does:** Adds three new ChromaDB collections (`contextual_collection`, `proposition_collection`, `raptor_collection`) plus their upsert, search, and data-fetch methods. The existing `sermon_collection` and all its methods remain unchanged.

- [ ] **Step 1: Read the current chroma_store.py**

Open `src/storage/chroma_store.py` and verify the current `__init__` ends with:

```python
self._sermons = self._client.get_or_create_collection("sermon_collection")
self._bible = self._client.get_or_create_collection("bible_collection")
self._reranker = Reranker()
```

- [ ] **Step 2: Add new collections to `__init__`**

After the `self._bible = ...` line (line 58), add:

```python
self._contextual = self._client.get_or_create_collection("contextual_collection")
self._proposition = self._client.get_or_create_collection("proposition_collection")
self._raptor = self._client.get_or_create_collection("raptor_collection")
```

- [ ] **Step 3: Add upsert, search, and fetch methods**

After the `search_bible` method (after line 123), add the following block:

```python
def upsert_contextual_chunks(self, chunks: list[str], metadatas: list[dict], ids: list[str]):
    self._upsert_in_batches(self._contextual, chunks, metadatas, ids)

def upsert_proposition_chunks(self, chunks: list[str], metadatas: list[dict], ids: list[str]):
    self._upsert_in_batches(self._proposition, chunks, metadatas, ids)

def upsert_raptor_chunks(self, chunks: list[str], metadatas: list[dict], ids: list[str], embeddings: list[list[float]]):
    """RAPTOR chunks come with pre-computed embeddings from the build step."""
    for start in range(0, len(chunks), self._MAX_BATCH):
        end = start + self._MAX_BATCH
        self._raptor.upsert(
            documents=chunks[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
            embeddings=embeddings[start:end],
        )

def search_contextual(self, query: str, k: int = 4, where: dict | None = None) -> list[dict]:
    return self._search(self._contextual, query, k, where)

def search_proposition(self, query: str, k: int = 4, where: dict | None = None) -> list[dict]:
    return self._search(self._proposition, query, k, where)

def search_raptor(self, query: str, k: int = 4, where: dict | None = None) -> list[dict]:
    return self._search(self._raptor, query, k, where)

def get_all_sermon_chunks_with_embeddings(self) -> list[dict]:
    """Fetch all sermon_collection chunks including their stored embeddings (for RAPTOR)."""
    n = self._sermons.count()
    if n == 0:
        return []
    results = self._sermons.get(include=["documents", "metadatas", "embeddings"])
    return [
        {"content": doc, "metadata": meta, "embedding": emb}
        for doc, meta, emb in zip(
            results["documents"], results["metadatas"], results["embeddings"]
        )
    ]
```

- [ ] **Step 4: Update the `counts` method to include new collections**

Replace the existing `counts` method:

```python
def counts(self) -> dict:
    return {
        "sermon_collection": self._sermons.count(),
        "bible_collection": self._bible.count(),
        "contextual_collection": self._contextual.count(),
        "proposition_collection": self._proposition.count(),
        "raptor_collection": self._raptor.count(),
    }
```

- [ ] **Step 5: Run the existing tests to confirm nothing broke**

```bash
pytest tests/test_vector_tool.py -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/storage/chroma_store.py
git commit -m "feat: add contextual, proposition, and raptor collections to chroma_store"
```

---

## Task 6: Wire Contextual Retrieval into the Ingest Pipeline

**Files:**
- Modify: `ingest.py`

**What it does:** When `--contextual` flag is passed, the pipeline runs each body chunk through `add_context_to_chunk()` before embedding and stores results in `contextual_collection` instead of (or in addition to) `sermon_collection`. The standard flow writes to `sermon_collection` as before; `--contextual` writes to `contextual_collection`.

- [ ] **Step 1: Add the import at the top of ingest.py**

After the line `from src.llm import get_llm` (line 25), add:

```python
from src.ingestion.contextual_retrieval import add_context_to_chunk
```

- [ ] **Step 2: Add contextual parameter to `process_group`**

Change the signature of `process_group` (line 86) from:

```python
def process_group(group, registry: SermonRegistry, vector_store: SermonVectorStore,
                  llm, splitter: RecursiveCharacterTextSplitter, incremental: bool, force: bool = False):
```

to:

```python
def process_group(group, registry: SermonRegistry, vector_store: SermonVectorStore,
                  llm, splitter: RecursiveCharacterTextSplitter, incremental: bool,
                  force: bool = False, contextual: bool = False, proposition: bool = False):
```

- [ ] **Step 3: Apply contextual retrieval and route to correct collection**

Find the `# Body chunks` block in `process_group` (around line 200):

```python
    # Body chunks
    if ng_body:
        chunks = splitter.split_text(ng_body) or [ng_body[:800]]
        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            metas.append({**chunk_meta, "doc_type": "body"})
            ids.append(f"{sermon_id}_body_{i}")
```

Replace it with:

```python
    # Body chunks
    if ng_body:
        if proposition:
            from src.ingestion.proposition_chunker import extract_propositions
            raw_chunks = extract_propositions(ng_body, llm)
        else:
            raw_chunks = splitter.split_text(ng_body) or [ng_body[:800]]

        for i, chunk in enumerate(raw_chunks):
            if contextual:
                chunk = add_context_to_chunk(chunk, ng_body, llm)
            docs.append(chunk)
            metas.append({**chunk_meta, "doc_type": "contextual_body" if contextual else "body"})
            ids.append(f"{sermon_id}_body_{i}")
```

- [ ] **Step 4: Route to the correct collection based on flags**

Find the final upsert block (around line 213):

```python
    if docs:
        vector_store.upsert_sermon_chunks(docs, metas, ids)
```

Replace it with:

```python
    if docs:
        if contextual:
            vector_store.upsert_contextual_chunks(docs, metas, ids)
        elif proposition:
            vector_store.upsert_proposition_chunks(docs, metas, ids)
        else:
            vector_store.upsert_sermon_chunks(docs, metas, ids)
```

- [ ] **Step 5: Add `--contextual` and `--proposition` CLI flags to `run_pipeline` and `__main__`**

Change `run_pipeline` signature (line 219) from:

```python
def run_pipeline(wipe: bool = False, year: int | None = None, incremental: bool = True, force: bool = False):
```

to:

```python
def run_pipeline(wipe: bool = False, year: int | None = None, incremental: bool = True,
                 force: bool = False, contextual: bool = False, proposition: bool = False):
```

In the `for group in groups:` loop (around line 268), change the `process_group` call to:

```python
            process_group(group, registry, vector_store, llm, splitter, incremental,
                         force, contextual=contextual, proposition=proposition)
```

In the `if __name__ == "__main__":` block (around line 285), add two arguments:

```python
    parser.add_argument("--contextual", action="store_true",
                        help="Use contextual retrieval (LLM context per chunk, writes to contextual_collection)")
    parser.add_argument("--proposition", action="store_true",
                        help="Use proposition chunking (LLM atomic claims, writes to proposition_collection)")
```

And pass them to `run_pipeline`:

```python
    run_pipeline(wipe=args.wipe, year=args.year, incremental=not args.wipe,
                 force=args.force, contextual=args.contextual, proposition=args.proposition)
```

- [ ] **Step 6: Verify the help text shows the new flags**

```bash
python ingest.py --help
```

Expected output includes:
```
  --contextual    Use contextual retrieval ...
  --proposition   Use proposition chunking ...
```

- [ ] **Step 7: Commit**

```bash
git add ingest.py
git commit -m "feat: wire contextual retrieval and proposition chunking into ingest pipeline"
```

---

## Task 7: Wire RAPTOR Build into the Ingest Pipeline

**Files:**
- Modify: `ingest.py`

**What it does:** When `--raptor` flag is passed, after the standard ingest completes, fetches all chunks from `sermon_collection`, builds the RAPTOR tree, and upserts the resulting cluster summaries into `raptor_collection`.

- [ ] **Step 1: Add imports at the top of ingest.py**

After the existing import from `contextual_retrieval`, add:

```python
from src.ingestion.raptor import build_raptor_tree
```

- [ ] **Step 2: Add a `_build_raptor` helper function in ingest.py**

After the `run_pipeline` function definition (just before `if __name__ == "__main__":`), add:

```python
def _build_raptor(vector_store: SermonVectorStore, llm):
    print("\n🌳 Building RAPTOR tree from sermon_collection...")
    all_chunks = vector_store.get_all_sermon_chunks_with_embeddings()
    if len(all_chunks) < 10:
        print(f"  ⚠️  Only {len(all_chunks)} chunks found — need at least 10 for RAPTOR. Skipping.")
        return

    print(f"  📊 {len(all_chunks)} chunks → clustering...")
    summaries = build_raptor_tree(
        all_chunks,
        llm,
        embed_fn=lambda texts: vector_store._embed(texts),
    )

    if not summaries:
        print("  ⚠️  RAPTOR produced no summaries (not enough clusters). Skipping.")
        return

    docs = [s["summary"] for s in summaries]
    metas = [s["metadata"] for s in summaries]
    embeddings = [s["embedding"] for s in summaries]
    ids = [f"raptor_L{s['metadata']['level']}_C{s['metadata']['cluster_id']}" for s in summaries]

    vector_store.upsert_raptor_chunks(docs, metas, ids, embeddings)
    print(f"  ✅ RAPTOR: {len(summaries)} cluster summaries stored in raptor_collection")
```

- [ ] **Step 3: Add `raptor` parameter to `run_pipeline` and call `_build_raptor` at the end**

Change `run_pipeline` signature to also accept `raptor`:

```python
def run_pipeline(wipe: bool = False, year: int | None = None, incremental: bool = True,
                 force: bool = False, contextual: bool = False, proposition: bool = False,
                 raptor: bool = False):
```

At the end of `run_pipeline`, just before the final print statement, add:

```python
    if raptor:
        _build_raptor(vector_store, llm)
```

- [ ] **Step 4: Add `--raptor` CLI flag to the argparse block**

```python
    parser.add_argument("--raptor", action="store_true",
                        help="Build RAPTOR tree after ingest (writes cluster summaries to raptor_collection)")
```

And update the `run_pipeline` call:

```python
    run_pipeline(wipe=args.wipe, year=args.year, incremental=not args.wipe,
                 force=args.force, contextual=args.contextual,
                 proposition=args.proposition, raptor=args.raptor)
```

- [ ] **Step 5: Also support standalone RAPTOR build (skip ingest, just build tree)**

To allow `python ingest.py --raptor` without re-processing all files, add a guard at the top of `run_pipeline` after the vector_store and llm setup:

After `llm = get_llm()` and `splitter = ...`, add:

```python
    # If raptor-only mode: skip ingest and go straight to tree building
    if raptor and not wipe and not year and not force:
        staging_exists = os.path.isdir(STAGING_DIR)
        chroma_has_data = vector_store.counts()["sermon_collection"] > 0
        if chroma_has_data:
            _build_raptor(vector_store, llm)
            return
```

- [ ] **Step 6: Verify help text**

```bash
python ingest.py --help
```

Expected output includes `--raptor`.

- [ ] **Step 7: Commit**

```bash
git add ingest.py
git commit -m "feat: wire RAPTOR tree building into ingest pipeline with --raptor flag"
```

---

## Task 8: Update Vector Tool for Collection Selection

**Files:**
- Modify: `src/tools/vector_tool.py`

**What it does:** Adds a `collection` parameter to `search_sermons_tool` so the LangGraph agent can choose which collection to search (standard, contextual, proposition, or raptor). Default is `"standard"` to preserve existing behavior.

- [ ] **Step 1: Read the current vector_tool.py**

Open `src/tools/vector_tool.py` and note that `search_sermons_tool` currently calls `vector_store.search_sermons(query, k=max(k, 5), where=where)`.

- [ ] **Step 2: Update the tool signature and docstring**

Replace the current `search_sermons_tool` function definition:

```python
    @tool
    def search_sermons_tool(
        query: str,
        year: int | None = None,
        speaker: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        k: int = 5,
    ) -> str:
        """Searches sermon text and summaries using semantic similarity.
        Use for 'What did the pastor say about X?' or 'Find sermons about Y'.
        Args:
          query: The search phrase (short, concept-focused terms work best).
          year: Exact year filter (integer e.g. 2024).
          speaker: Partial speaker name filter (e.g. 'Chua').
          min_year: Earliest year inclusive (e.g. 2024 for 'last 2 years' when current year is 2025).
          max_year: Latest year inclusive.
          k: Number of results to return (default 5, use 8-10 for broad topic queries).
        Returns excerpts with topic, speaker, date, and key verse."""
```

With:

```python
    @tool
    def search_sermons_tool(
        query: str,
        year: int | None = None,
        speaker: str | None = None,
        min_year: int | None = None,
        max_year: int | None = None,
        k: int = 5,
        collection: str = "standard",
    ) -> str:
        """Searches sermon text and summaries using semantic similarity.
        Use for 'What did the pastor say about X?' or 'Find sermons about Y'.
        Args:
          query: The search phrase (short, concept-focused terms work best).
          year: Exact year filter (integer e.g. 2024).
          speaker: Partial speaker name filter (e.g. 'Chua').
          min_year: Earliest year inclusive (e.g. 2024 for 'last 2 years' when current year is 2025).
          max_year: Latest year inclusive.
          k: Number of results to return (default 5, use 8-10 for broad topic queries).
          collection: Which collection to search: 'standard' (default), 'contextual',
                      'proposition', or 'raptor'. Use 'raptor' for broad thematic queries.
        Returns excerpts with topic, speaker, date, and key verse."""
```

- [ ] **Step 3: Route to the correct search method based on `collection`**

Find the line:

```python
        results = vector_store.search_sermons(query, k=max(k, 5), where=where)
```

Replace it with:

```python
        _search_map = {
            "standard": vector_store.search_sermons,
            "contextual": vector_store.search_contextual,
            "proposition": vector_store.search_proposition,
            "raptor": vector_store.search_raptor,
        }
        search_fn = _search_map.get(collection, vector_store.search_sermons)
        results = search_fn(query, k=max(k, 5), where=where)
```

- [ ] **Step 4: Write a test for collection routing**

Add to `tests/test_vector_tool.py` at the end:

```python
def test_tool_routes_to_contextual_collection():
    store = MagicMock()
    store.search_sermons.return_value = []
    store.search_contextual.return_value = _sample_results()
    store.search_proposition.return_value = []
    store.search_raptor.return_value = []
    tool = make_vector_tool(store)
    result = tool.invoke({"query": "grace", "collection": "contextual"})
    store.search_contextual.assert_called_once()
    store.search_sermons.assert_not_called()
    assert "Pastor John" in result


def test_tool_defaults_to_standard_collection():
    store = MagicMock()
    store.search_sermons.return_value = _sample_results()
    tool = make_vector_tool(store)
    tool.invoke({"query": "grace"})
    store.search_sermons.assert_called_once()
```

- [ ] **Step 5: Run all vector tool tests**

```bash
pytest tests/test_vector_tool.py -v
```

Expected: All tests pass (including the 2 new ones).

- [ ] **Step 6: Commit**

```bash
git add src/tools/vector_tool.py tests/test_vector_tool.py
git commit -m "feat: add collection parameter to search_sermons_tool for comparing RAG techniques"
```

---

## Task 9: Final Smoke Test + PR

**Files:**
- None (validation only)

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests pass. Note any unexpected failures and investigate before continuing.

- [ ] **Step 2: Verify help text shows all three new flags**

```bash
python ingest.py --help
```

Expected output includes `--contextual`, `--proposition`, and `--raptor`.

- [ ] **Step 3: Verify ChromaDB wiring (no data needed)**

```bash
python -c "
from src.storage.chroma_store import SermonVectorStore
vs = SermonVectorStore()
counts = vs.counts()
print('Collections:', list(counts.keys()))
assert 'contextual_collection' in counts
assert 'proposition_collection' in counts
assert 'raptor_collection' in counts
print('OK')
"
```

Expected output:
```
Collections: ['sermon_collection', 'bible_collection', 'contextual_collection', 'proposition_collection', 'raptor_collection']
OK
```

- [ ] **Step 4: Commit any final cleanup and open PR**

```bash
git push -u origin feature/rag-enhancements-option-a
gh pr create \
  --title "feat: RAG enhancements Option A (Contextual Retrieval, Proposition Chunking, RAPTOR)" \
  --body "Adds three independent opt-in RAG techniques. Each technique writes to its own ChromaDB collection. New CLI flags: --contextual, --proposition, --raptor. The search_sermons_tool gains a collection parameter to compare approaches at query time. Existing behavior unchanged."
```

---

## Usage Reference (after implementation)

```bash
# Standard ingest (unchanged baseline)
python ingest.py

# Ingest with contextual retrieval (1 extra LLM call per chunk)
python ingest.py --contextual

# Ingest with proposition chunking (1 extra LLM call per document)
python ingest.py --proposition

# Build RAPTOR tree from existing sermon_collection (no re-ingest needed)
python ingest.py --raptor

# Full rebuild + RAPTOR in one pass
python ingest.py --wipe --raptor
```

```python
# In agent: compare contextual vs standard retrieval
search_sermons_tool(query="faith and doubt", collection="contextual")
search_sermons_tool(query="faith and doubt", collection="standard")

# Use RAPTOR for broad thematic questions
search_sermons_tool(query="what are the recurring themes across all sermons on discipleship", collection="raptor")
```
