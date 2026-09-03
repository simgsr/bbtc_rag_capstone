# RAG Evaluation Harness

Retrieval + groundedness eval harness driven by `evals/golden_set.json`:

```bash
python -m evals.run_eval --retrieval     # fast, embedding-only: recall@k + filter precision
python -m evals.run_eval --groundedness  # slow, invokes live agent via app.respond
python -m evals.run_eval                 # both; writes evals/eval_report.json; exits non-zero if any pass-rate < 0.7
```

- **Retrieval** items drive `SermonVectorStore.search_sermons` (replicating the tool's filter handling) and measure recall@k against `must_find`/`must_find_any` sermon_ids, soft topic-precision@k, and hard filter-precision (speaker/year). Baseline: ~0.86 pass rate, ~0.86 avg recall.
- **Groundedness** items invoke the full ReAct agent and check each answer for expected facts present, forbidden phrases absent (e.g. "based on my knowledge", "typically"), that a tool was actually used, and that negative queries declare "no records" rather than fabricate. Baseline: ~0.86–1.0 pass rate (gnd-05 top-speakers is a known brittle case — the model occasionally omits a verbatim name despite using SQL).
- Golden-set facts/sermon_ids were verified against `data/sermons.db`; re-verify if the archive is re-ingested.
