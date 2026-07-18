"""Agent tools: Bible lookup + search over ``bible_collection``.

``make_bible_tool(vector_store)`` returns two tools:

  * ``get_bible_versions_tool(reference)`` — returns every stored translation
    (KJV, ASV, YLT, BBE, ChiUn, NIV, ESV) of one verse, for translation-audit /
    version comparison. The reference is canonicalised via ``normalize_book`` so
    ``1 john 1:9`` and ``1 John 1:9`` resolve to the same stored key.
  * ``search_bible_tool(query, k, version)`` — BGE-M3 semantic search for passages
    about a theme ("forgiveness", "walking by faith"); optional case-insensitive
    ``version`` filter (oversample + post-filter, since Chroma ``$eq`` is
    case-sensitive and ``ChiUn`` is mixed-case).

Verse-reference parsing lives in ``_normalize_ref`` (``Book Chapter:Verse[-Verse]``).
"""
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
        Returns all stored translations of that verse. The archive holds 7 versions:
        KJV, ASV, YLT, BBE (Basic English), ChiUn (Chinese Union — Chinese text),
        NIV, ESV. Note ChiUn returns Chinese text; pass an English reference."""
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
    def search_bible_tool(query: str, k: int = 5, version: str | None = None) -> str:
        """Semantically search the Bible archive for passages matching a topic or concept.
        Use when looking for Bible passages about a theme (e.g. 'forgiveness', 'walking by faith').
        Args:
          query: Search phrase (3-6 words work best).
          k: Number of results (default 5).
          version: Optional translation filter (one of KJV, ASV, YLT, BBE, ChiUn, NIV, ESV).
            Set to 'ChiUn' for Mandarin queries; for English topic queries prefer leaving unset
            but be aware results may include Chinese (ChiUn) verses — filter to an English version
            (e.g. 'NIV') if Chinese hits are unwanted. Case-insensitive ('niv' matches 'NIV').
        Returns matching verse text with references and version."""
        # Chroma metadata `$eq` is case-sensitive and stored version tags are
        # mixed-case (KJV, NIV, ... but "ChiUn"), so a lowercase/variants request
        # would silently yield zero hits. Oversample ~8x (one slot per version +
        # headroom) and post-filter by case-insensitive equality, mirroring the
        # speaker filter in vector_tool.
        fetch_k = max(k, 5)
        if version:
            fetch_k = max(fetch_k * 8, 40)
        results = vector_store.search_bible(query, k=fetch_k)
        if version and results:
            want = version.strip().lower()
            results = [r for r in results
                       if ((r.get("metadata") or {}).get("version") or "").strip().lower() == want]
        results = results[:max(k, 5)]
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
