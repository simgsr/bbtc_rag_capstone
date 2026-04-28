from langchain_core.tools import tool
from src.storage.chroma_store import SermonVectorStore


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
        if speaker:
            conditions.append({"speaker": {"$eq": speaker}})

        where: dict | None = None
        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        results = vector_store.search_sermons(query, k=max(k, 5), where=where)
        if not results:
            return "No relevant sermon content found."

        parts = []
        for res in results:
            m = res.get("metadata") or {}
            header = (
                f"[{m.get('topic') or 'Unknown Topic'} | {m.get('speaker') or 'Unknown'} "
                f"| {m.get('date') or ''} | {m.get('key_verse') or ''}]"
            )
            parts.append(f"{header}\n{res['content']}")

        return "\n\n---\n\n".join(parts)

    return search_sermons_tool
