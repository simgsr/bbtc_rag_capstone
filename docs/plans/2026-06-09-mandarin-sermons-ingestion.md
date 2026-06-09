# Mandarin Sermons Ingestion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the BBTC ingest pipeline to scrape, parse, and index the **Mandarin sermon archive** (2019–present), so that Mandarin sermons appear in SQL counts, verses-per-book aggregations, and semantic search alongside the English archive. Verse aggregations (most-preached / never-preached books) must include Mandarin-sourced verses.

**Architecture:** The Mandarin archive is structurally simpler than English — it is **PS-only** (slides PDFs; no Notes/Guide files) and listed as **direct files** on the year archive page (no per-sermon detail pages from 2021 onward). All existing classify→group→extract→summarize→embed stages are reused; new logic is added at three points: (1) scraper CLI invokes Mandarin year archives, (2) extractor parses Chinese book names + Chinese date prefixes from filenames, (3) the ingest summarizer translates Chinese slide text to English before summarising. Embeddings already use BGE-M3 (multilingual) — no embedding-layer changes needed.

**Tech Stack:** Python 3.14, cloudscraper, BeautifulSoup4, PyMuPDF, sqlite3, ChromaDB, BGE-M3 (sentence-transformers, MPS), MLX-LM (Qwen3-4B-4bit).

---

## Decisions confirmed by user (2026-06-09)

| # | Question | Decision |
|---|---|---|
| 1 | LLM summarization strategy for Mandarin | **Translate-then-summarize in English** (option 1c). Maximises searchability for English-language queries; LLM emits a single English summary stored in the same `summary` column as English sermons. |
| 2 | Speaker field for Mandarin | **Leave `speaker` NULL.** Mandarin filenames almost never encode a speaker. Don't fabricate a placeholder. |
| 3 | Rollout scope | **Phased.** Phase 1 + Phase 4 single-year smoke test first (proves scrape + ingest end-to-end with minimal extraction), then Phases 2–3 for the long tail. |
| 4 | Pre-Q4-2019 audio-only sermons | **Skip silently.** Those entries are MP3 audio with no PDF/PPTX artefact. The scraper's existing extension allowlist (`_RESOURCE_EXTENSIONS`) already filters them out — no special handling needed. |

---

## Findings — Mandarin archive structure (verified 2026-06-09)

URL pattern: `https://www.bbtc.com.sg/mw-sermons-{year}/`

| Year | Sermon-detail pages | Direct PDFs | Notes |
|---|---|---|---|
| 2019 | 30 (mostly `.mp3`) | 34 (Q4 only) | Pre-Q4 was audio-only; Q4 onwards posts slide PDFs |
| 2021 | 0 | 53 | Pure direct-file listing |
| 2023 | 0 | 54 | Same |
| 2024 | 0 | 53 | Same |
| 2026 | 0 (YTD) | 23 | Same |

Estimated total: ~310 Mandarin sermons across 2019–2026.

Filename patterns observed (Chinese characters preserved):

```
20241229-西面称颂救主_compressed.pdf
190602-属神的子民.pdf
20211226-耶稣的成长-路加2-39-52_compressed.pdf
MW2-Joshua-10-B_compressed.pdf
属灵家的坚垒-24May_compressed.pdf
20260531-不要怕他们_compressed.pdf
```

Key characteristics:
- Date prefix is usually `YYYYMMDD-` or `YYMMDD-` (some files have date at the end, e.g. `…-24May`).
- Verse refs may use Chinese book names (`路加2-39-52` = Luke 2:39-52) or English (`Joshua-10`); some files have no verse hint at all.
- No speaker info in filename.
- `_compressed` is a common suffix indicating an export from PowerPoint with image compression.
- Many slide PDFs will be image-only (text extraction will yield nothing) — same limitation as English PS files; OCR is out of scope.

---

## File Map

| Action  | File                                            | Responsibility |
|---------|-------------------------------------------------|----------------|
| Modify  | `src/scraper/bbtc_scraper.py`                   | CLI accepts `lang` arg; `--all` loops both languages |
| Modify  | `Makefile`                                      | `scrape` target covers Mandarin too |
| Modify  | `src/storage/normalize_book.py`                 | Add Chinese book-name aliases to `BOOK_MAP` |
| Create  | `src/ingestion/cn_filename_parser.py`           | `parse_date_from_cn_filename()`, `parse_verses_from_cn_filename()`, `derive_topic_from_cn_filename()` |
| Modify  | `src/ingestion/ps_extractor.py`                 | Wire in Chinese verse parser as a fallback when English regex returns no matches |
| Modify  | `src/ingestion/filename_parser.py`              | Add `YYYYMMDD-` / `YYMMDD-` patterns as date fallback |
| Modify  | `ingest.py`                                     | For Mandarin PS-only groups: derive topic + date from filename; route summarizer through translate-then-summarize prompt |
| Modify  | `src/llm.py` (or new helper)                    | `summarize_mandarin_sermon(text)` — single LLM call that translates + summarizes in English |
| Create  | `tests/test_cn_filename_parser.py`              | Tests for Chinese date/verse/topic extraction |
| Modify  | `tests/test_normalize_book.py`                  | Tests for Chinese book-name normalization |
| Modify  | `CLAUDE.md`                                     | Document Mandarin pipeline + caveats |

---

## Phase 1 — Scraper CLI for Mandarin (smoke test)

**Goal:** Prove the existing `scrape_year(year, lang="Mandarin")` plumbing works end-to-end. No extraction changes yet.

**Files:** `src/scraper/bbtc_scraper.py`, `Makefile`

- [ ] **Step 1.1:** Add `lang` arg to `bbtc_scraper.py.__main__`. Support `python bbtc_scraper.py 2024 Mandarin` and `python bbtc_scraper.py --all` (loops over both English + Mandarin).
- [ ] **Step 1.2:** Update `Makefile` `scrape` target so `make scrape YEAR=2024` covers both languages, or add a `scrape-mandarin` target.
- [ ] **Step 1.3:** Smoke test: `python src/scraper/bbtc_scraper.py 2024 Mandarin`. Expect ~53 PDFs in `data/staging/` with `Mandarin_2024_` prefix.
- [ ] **Step 1.4:** Verify `classify_file` returns `"ps"` for every Mandarin filename (sanity — none should match NG or handout regex).

---

## Phase 2 — Chinese-aware extractors (the bulk of the work)

### 2a. Chinese book-name aliases

**Files:** `src/storage/normalize_book.py`, `tests/test_normalize_book.py`

- [ ] **Step 2a.1:** Build a curated Chinese→canonical table covering all 66 books. Three forms per book:
  - Full form: `创世记` → Genesis, `路加福音` → Luke, `约翰福音` → John, `罗马书` → Romans, `启示录` → Revelation
  - Abbreviated: `路加` → Luke, `约翰` → John, `罗马` → Romans
  - Numbered: `约翰一书` / `约翰壹书` → 1 John, `提摩太前书` → 1 Timothy, `撒母耳记上` → 1 Samuel
- [ ] **Step 2a.2:** Merge into existing `BOOK_MAP` keyed lower-case English aliases — keys can be Chinese (no `.lower()` issues since Chinese chars have no case).
- [ ] **Step 2a.3:** Tests: round-trip every Chinese alias to its canonical English name.

Reference table seed (to be expanded):

```
创世记/Genesis  出埃及记/Exodus  利未记/Leviticus  民数记/Numbers  申命记/Deuteronomy
约书亚记/Joshua  士师记/Judges  路得记/Ruth
撒母耳记上/1 Samuel  撒母耳记下/2 Samuel  列王纪上/1 Kings  列王纪下/2 Kings
诗篇/Psalms  箴言/Proverbs  传道书/Ecclesiastes  雅歌/Song of Songs
以赛亚书/Isaiah  耶利米书/Jeremiah  以西结书/Ezekiel  但以理书/Daniel
马太福音/Matthew  马可福音/Mark  路加福音/Luke  约翰福音/John
使徒行传/Acts  罗马书/Romans  哥林多前书/1 Corinthians  哥林多后书/2 Corinthians
加拉太书/Galatians  以弗所书/Ephesians  腓立比书/Philippians  歌罗西书/Colossians
提摩太前书/1 Timothy  提摩太后书/2 Timothy  希伯来书/Hebrews  雅各书/James
约翰一书/1 John  约翰二书/2 John  约翰三书/3 John  启示录/Revelation
```

### 2b. Chinese filename parser

**Files:** `src/ingestion/cn_filename_parser.py` (new), `tests/test_cn_filename_parser.py` (new)

- [ ] **Step 2b.1:** `parse_date_from_cn_filename(filename) -> str | None`. Patterns:
  - `(\d{8})` → `YYYY-MM-DD` (e.g. `20241229` → `2024-12-29`)
  - `(\d{6})` → `20YY-MM-DD` (e.g. `190602` → `2019-06-02`)
  - Trailing-date forms: `-(\d{1,2})(Jan|Feb|…|Dec)` → resolve year from `Mandarin_{year}_` prefix
- [ ] **Step 2b.2:** `parse_verses_from_cn_filename(filename) -> list[dict]`. Regex shapes:
  - `({ChineseBookPattern})\s*(\d+)[:章]\s*(\d+)(?:[-–~](\d+))?`
  - `({ChineseBookPattern})(\d+)-(\d+)-(\d+)` (matches `路加2-39-52`)
  - `({ChineseBookPattern})(\d+)章(\d+)节` (verbose Chinese form)
  - Use `normalize_book` to canonicalize the matched Chinese book to English
- [ ] **Step 2b.3:** `derive_topic_from_cn_filename(filename) -> str`. Strip: `Mandarin_{year}_`, leading `YYYYMMDD-`/`YYMMDD-`, trailing `_compressed`, file extension. Result: clean Chinese topic title.
- [ ] **Step 2b.4:** Tests covering each filename shape observed in the findings table above.

### 2c. Wire into extractors

**Files:** `src/ingestion/ps_extractor.py`, `src/ingestion/filename_parser.py`

- [ ] **Step 2c.1:** In `parse_verses_from_filename`: if English regex returns 0 verses AND filename contains any Chinese codepoint (`一-鿿`), call `parse_verses_from_cn_filename` and use those results.
- [ ] **Step 2c.2:** In `filename_parser` date fallback: add `YYYYMMDD-` and `YYMMDD-` patterns near the top of the regex chain.

---

## Phase 3 — Ingest pipeline adaptation

**Files:** `ingest.py`, `src/llm.py`

- [ ] **Step 3.1:** In `process_group`, when `_detect_language(filename) == "Mandarin"` and `ng_file is None`:
  - Set `topic = derive_topic_from_cn_filename(ps_file)`
  - Set `date = parse_date_from_cn_filename(ps_file) or fallback`
  - Set `speaker = None`, `theme = None`
- [ ] **Step 3.2:** Add `summarize_mandarin_sermon(text: str) -> str` to `src/llm.py`. Single prompt that asks the LLM to: (a) translate the input Chinese text to English, then (b) produce a 2-3 paragraph English summary in the same style as existing English sermons. Reuse `get_ingest_llm()` (MLX Qwen3 handles Chinese). Apply same `enable_thinking=False` setting.
- [ ] **Step 3.3:** Route Mandarin groups through `summarize_mandarin_sermon` instead of the English summarizer. Store the English summary in `sermons.summary`. This is the input to the LLM verse-extraction path too — so any verses the LLM finds in the translation get inserted via the existing path.
- [ ] **Step 3.4:** Confirm `language="Mandarin"` is set on the row (already handled by `_detect_language` from `Mandarin_` prefix — verify with a unit test).

---

## Phase 4 — Smoke test (insert between Phases 1 and 2)

Per user decision #3, run this *immediately after Phase 1* to validate end-to-end before investing in Chinese extractors.

- [ ] **Step 4.1:** Scrape one year: `python src/scraper/bbtc_scraper.py 2024 Mandarin`
- [ ] **Step 4.2:** Incremental ingest: `python ingest.py --year 2024`. Expect ~53 Mandarin rows with empty topic/speaker/theme but populated `summary` (from raw extracted PDF text, no Chinese-aware parsing yet — most will be sparse if slides are image-only).
- [ ] **Step 4.3:** Verify in SQL:
  ```sql
  SELECT language, COUNT(*) FROM sermons WHERE year=2024 GROUP BY language;
  -- expect: English=~50, Mandarin=~53
  ```
- [ ] **Step 4.4:** Confirm the Gradio app's "Recent sermons" table shows Mandarin entries (they'll look bare without Phase 2).

After this smoke test succeeds, proceed to Phases 2 + 3 for proper Chinese extraction.

---

## Phase 5 — Full backfill + verification

- [ ] **Step 5.1:** Scrape all years: extend the `--all` loop to cover Mandarin from 2019 to current. Skip 2015–2018 (no Mandarin archive exists).
- [ ] **Step 5.2:** Run full incremental ingest. Monitor for failures (image-only PDFs are expected to log `text_quality=failed`).
- [ ] **Step 5.3:** Verify aggregations:
  ```sql
  -- Mandarin sermon count by year
  SELECT year, COUNT(*) FROM sermons WHERE language='Mandarin' GROUP BY year ORDER BY year;

  -- Verses contributed by Mandarin sermons
  SELECT COUNT(*) FROM verses v JOIN sermons s USING(sermon_id) WHERE s.language='Mandarin';

  -- "Books never preached" must reflect Mandarin contributions
  SELECT bb.book_name FROM bible_books bb
  WHERE bb.book_name NOT IN (
    SELECT DISTINCT COALESCE(ba.canonical, v.book) FROM verses v
    LEFT JOIN book_aliases ba ON LOWER(TRIM(v.book))=ba.alias
    WHERE v.book IS NOT NULL AND v.book!='')
  ORDER BY bb.book_order;
  ```
- [ ] **Step 5.4:** Re-run the agent's "Books never preached in BBTC sermons" quick-query through the Gradio UI and verify the list shrinks (Mandarin sermons should fill some gaps).

---

## Phase 6 — Docs + UX touch-ups

- [ ] **Step 6.1:** Update `CLAUDE.md`:
  - Add Mandarin to the "Running the application" section: `python src/scraper/bbtc_scraper.py 2024 Mandarin`
  - Document the PS-only Mandarin convention + the translate-then-summarize strategy
  - Note known limitations (no speaker, many image-only PDFs)
- [ ] **Step 6.2:** Update the system prompt in `app.py` to mention that `language` column is `'English'` or `'Mandarin'`, and that Mandarin sermons have empty `speaker`. (The prompt already says this for English — adjust caveats.)
- [ ] **Step 6.3:** Add `language='Mandarin'` filter examples to the SQL prompt patterns: "Mandarin sermons by year", "Top verses from Mandarin sermons", etc.

---

## Risks + open considerations

1. **Image-only PDFs:** A large fraction of Mandarin slide exports are likely image-only PowerPoint compressions (`_compressed.pdf`). Text extraction yields nothing, so summaries will be sparse. Verse extraction falls back to filename regex — which is why Phase 2b matters. OCR is out of scope; document this as a known limitation.
2. **Chinese book regex ambiguity:** Some short forms (`约翰` alone) can match part of a longer phrase. Anchor regex to word boundaries / non-Chinese-character context, or require a digit immediately after.
3. **Date parsing for year=2019 dual-archive:** Some 2019 Mandarin entries are MP3s linked from sermon-detail pages (`pages` count = 30). The scraper's extension filter already drops MP3s — verify it doesn't accidentally crawl the sermon page and write empty manifests.
4. **LLM translation quality:** Qwen3-4B-4bit handles Chinese, but a 4-bit model may translate idioms loosely. If summary quality is poor, consider escalating Mandarin ingestion to `groq` or `gemini` via the `INGEST_PROVIDER` env var (per `CLAUDE.md`). Test on 3-5 samples first.
5. **Bilingual chunk-level search:** BGE-M3 is multilingual, so an English query may still retrieve Chinese chunks if semantics align. This is mostly a feature, but watch for confusing cross-language retrievals in the early UI testing.
6. **Verse-extraction LLM path:** The existing pipeline runs the LLM on NG+PS body text to extract verse refs. For Mandarin, the LLM is operating on the English translation (per decision 1c) — verify it still emits canonical English `verse_ref` strings (e.g. `"Luke 9:23"`) that flow through `insert_verse` → `normalize_book` cleanly.

---

## Effort estimate

- Phase 1 (CLI + smoke test): ~30 min
- Phase 2 (Chinese extractors + tests): ~3-4 hours (book aliases table is the slow part)
- Phase 3 (ingest + LLM wiring): ~1-2 hours
- Phase 4 (smoke test execution): ~15 min runtime
- Phase 5 (full backfill): ~30-60 min runtime depending on LLM throughput on ~310 sermons
- Phase 6 (docs): ~30 min

**Total: ~6-9 hours of focused work, two evening sessions.**
