import re
from langchain_core.tools import tool
from src.storage.chroma_store import SermonVectorStore
from src.storage.normalize_book import normalize_book


def _normalize_ref(reference: str) -> str | None:
    """Parse and canonicalize a verse reference string."""
    m = re.match(
        r'^([\w\s]+?)\s+(\d+)(?::(\d+)(?:-(\d+))?)?$',
        reference.strip(),
        re.IGNORECASE,
    )
    if not m:
        return None
    canonical = normalize_book(m.group(1))
    if not canonical:
        return None
    ch = m.group(2)
    vs, ve = m.group(3), m.group(4)
    if vs:
        return f"{canonical} {ch}:{vs}-{ve}" if ve else f"{canonical} {ch}:{vs}"
    return f"{canonical} {ch}"


def make_bible_tool(vector_store: SermonVectorStore):

    @tool
    def get_bible_versions_tool(reference: str) -> str:
        """Retrieve all available Bible translations of a specific verse.
        Use for Translation Audit requests or comparing Bible versions.
        Args:
          reference: Verse reference in 'Book Chapter:Verse' format (e.g. '1 John 1:9', 'John 3:16').
        Returns all stored translations (NIV, ESV, ASV, etc.) of that verse."""
        canonical_ref = _normalize_ref(reference)
        if not canonical_ref:
            return (
                f"Could not parse verse reference '{reference}'. "
                "Use format 'Book Chapter:Verse' (e.g. '1 John 1:9')."
            )
        results = vector_store.get_bible_versions(canonical_ref)
        if not results:
            return f"No Bible translations found for '{canonical_ref}'."
        parts = [
            f"**{r['metadata']['version']}**: {r['content']}"
            for r in results
        ]
        return f"Bible translations of {canonical_ref}:\n\n" + "\n\n".join(parts)

    @tool
    def search_bible_tool(query: str, k: int = 5) -> str:
        """Semantically search the Bible archive for passages matching a topic or concept.
        Use when looking for Bible passages about a theme (e.g. 'forgiveness', 'walking by faith').
        Args:
          query: Search phrase (3-6 words work best).
          k: Number of results (default 5).
        Returns matching verse text with references and version."""
        results = vector_store.search_bible(query, k=k)
        if not results:
            return "No matching Bible passages found."
        parts = []
        for r in results:
            meta = r.get("metadata") or {}
            parts.append(
                f"**{meta.get('reference', '')}** ({meta.get('version', '')}): {r['content']}"
            )
        return "\n\n".join(parts)

    return get_bible_versions_tool, search_bible_tool
