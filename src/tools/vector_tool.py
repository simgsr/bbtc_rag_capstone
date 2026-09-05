"""Agent tool: semantic search over sermon content (``sermon_collection``).

``make_vector_tool(vector_store)`` returns ``search_sermons_tool`` — BGE-M3
similarity search over NG body chunks and LLM summaries, used for content
questions ("what did the pastor say about X?"). It supports optional metadata
filters (exact/`min`/`max` year, partial speaker) translated to Chroma ``where``
clauses, and de-duplicates hits so multiple excerpts from one sermon are merged
under a single header. Returns formatted excerpts, not raw rows.
"""
from langchain_core.tools import tool
from src.storage.chroma_store import SermonVectorStore


def search_sermons_with_filters(
    vector_store: SermonVectorStore,
    query: str,
    k: int = 5,
    conditions: list | None = None,
    speaker: str | None = None,
) -> list[dict]:
    """The shared retrieval path behind ``search_sermons_tool`` AND the eval
    harness (``evals/run_eval.py``): build the Chroma ``where`` clause, apply the
    speaker oversample + case-insensitive substring post-filter, and return the
    top-``k`` results. Kept in one place so the eval measures exactly what the
    agent tool returns — they can't drift apart again.

    ``conditions`` is a list of Chroma conditions for the year/min_year/max_year
    filters (e.g. ``{"year": {"$eq": 2024}}``). ``speaker`` is intentionally NOT
    pushed into ``where``: Chroma metadata filters are exact-match only, but stored
    speakers carry titles ("SP Chua Seng Lee"), so an exact match on "Chua" would
    return nothing. Instead we oversample then post-filter, preserving the tool's
    "partial speaker name" contract.
    """
    where: dict | None = None
    if conditions:
        if len(conditions) == 1:
            where = conditions[0]
        else:
            where = {"$and": conditions}

    fetch_k = max(k, 5)
    if speaker:
        # Oversampling by a fixed factor risks under-filling `k` for a prolific
        # speaker on a rare topic — the 4× window may not contain enough of
        # *their* sermons (e.g. SP Daniel Foo has 110; a niche query could have
        # none of his in the top 20). Fetch the whole collection instead, then
        # post-filter by speaker and keep the top-`k` by distance. The year/min/
        # max `where` clause still applies inside Chroma, so this is bounded to
        # the year-matched subset, and the query is embedded once regardless of
        # `n_results`, so the cost is just an in-memory Chroma scan — negligible
        # for this corpus (~2k chunks).
        fetch_k = max(fetch_k, vector_store.counts()["sermon_collection"])
    results = vector_store.search_sermons(query, k=fetch_k, where=where)
    if speaker and results:
        needle = speaker.lower()
        results = [r for r in results
                   if needle in ((r.get("metadata") or {}).get("speaker") or "").lower()]
    return results[:max(k, 5)]


def make_vector_tool(vector_store: SermonVectorStore):

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

        conditions = []
        if year is not None:
            conditions.append({"year": {"$eq": year}})
        if min_year is not None:
            conditions.append({"year": {"$gte": min_year}})
        if max_year is not None:
            conditions.append({"year": {"$lte": max_year}})
        # Filter handling (year bounds → Chroma `where`; speaker → oversample +
        # substring post-filter) lives in search_sermons_with_filters, shared with
        # the eval harness so the eval measures exactly what this tool returns.
        results = search_sermons_with_filters(
            vector_store, query, k=k, conditions=conditions, speaker=speaker
        )
        if not results:
            return "No relevant sermon content found."

        # Group by sermon; keep body/summary excerpts separate from doc_type="metadata"
        # title chunks. The metadata chunk ("Topic | Theme | Speaker | Key verse |
        # Date") is indexed so topical queries can *find* a sermon, but its content is
        # a restatement of the header — don't surface it as an excerpt when real
        # body/summary excerpts exist. For textless sermons whose only chunk is
        # metadata, fall back to it so they still show something.
        sermons: dict[tuple, dict] = {}
        for res in results:
            m = res.get("metadata") or {}
            key = (
                m.get("topic") or "Unknown Topic",
                m.get("speaker") or "Unknown",
                m.get("date") or "",
                m.get("key_verse") or ""
            )
            bucket = sermons.setdefault(key, {"excerpts": [], "meta": None})
            if m.get("doc_type") == "metadata":
                if bucket["meta"] is None:
                    bucket["meta"] = res["content"]
            elif res["content"] not in bucket["excerpts"]:
                bucket["excerpts"].append(res["content"])

        parts = []
        for key, bucket in sermons.items():
            contents = bucket["excerpts"] or ([bucket["meta"]] if bucket["meta"] else [])
            topic, speaker, date, key_verse = key
            header = f"[{topic} | {speaker} | {date} | {key_verse}]"
            merged_content = "\n\n... [another excerpt from same sermon] ...\n\n".join(contents)
            parts.append(f"{header}\n{merged_content}")

        return "\n\n---\n\n".join(parts)

    return search_sermons_tool
