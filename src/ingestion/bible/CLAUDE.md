# Bible Ingestion

The archive holds **7 translations**: KJV, ASV, YLT, BBE (Basic English), ChiUn (Chinese Union — Chinese text) from Scrollmapper JSON (public domain, auto-downloaded); NIV and ESV from local EPUB files you supply (copyrighted — not in the repo). Place them at `data/bibles/NIV.epub` and `data/bibles/ESV The Holy Bible.epub`.

```bash
python -m src.ingestion.bible.bible_ingest            # all 7 (NIV/ESV skipped if EPUBs absent)
python -m src.ingestion.bible.bible_ingest --wipe     # wipe + re-ingest bible_collection
python -m src.ingestion.bible.bible_ingest --versions KJV WEB NIV
```

- Each translation is stored with `status='indexed'` and `_is_indexed()` skips already-indexed versions, so a translation is ingested once. Missing EPUBs simply never enter the source list (they're discovered by scanning `data/bibles/*.epub`) — drop the file in and it gets picked up on the next run; nothing writes a `'skipped'` status.
- `ChiUn` is Chinese. `search_bible_tool` takes an optional `version` filter; for English topic queries pass an English version (e.g. `NIV`) to avoid Chinese verses surfacing. `get_bible_versions_tool` returns all 7 versions of a verse — only include ChiUn when Chinese is wanted.
