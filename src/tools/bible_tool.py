from langchain.tools import tool
import re

def make_bible_tool(vector_store):
    @tool
    def compare_bible_versions(reference: str):
        """
        Retrieves and compares different Bible translations for a specific reference.
        Input should be a standard reference like 'John 3:16' or 'Genesis 1:1'.
        """
        # 1. Normalize reference to ref_id
        # Simple normalization: John 3:16 -> JOH_003_016
        match = re.search(r"([1-3]?\s?[A-Za-z]+)\s*(\d+)[:](\d+)", reference)
        if not match:
            return f"Could not parse reference: {reference}. Please use format 'Book Chapter:Verse'."
        
        book, chap, verse = match.groups()
        ref_id = f"{book[:3].upper()}_{int(chap):03}_{int(verse):03}"
        
        # 2. Get versions from store
        versions = vector_store.get_bible_versions(ref_id)
        
        if not versions:
            return f"No versions found for {reference} in the database."
        
        # 3. Format result
        output = [f"### Comparison for {reference}\n"]
        for v in versions:
            meta = v['metadata']
            output.append(f"**{meta['version']}**: {v['content']}")
            if 'verse_meaning' in meta:
                output.append(f"*Meaning*: {meta['verse_meaning']}\n")
        
        return "\n".join(output)

    return compare_bible_versions
