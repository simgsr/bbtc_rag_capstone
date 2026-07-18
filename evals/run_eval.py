"""RAG evaluation harness for the BBTC sermon archive.

Two evaluation modes, run from a single golden set (``evals/golden_set.json``):

* **Retrieval** — drives ``SermonVectorStore.search_sermons`` directly (the same
  call the agent's ``search_sermons_tool`` makes) and measures:
    - recall@k against ``must_find`` / ``must_find_any`` sermon_ids
    - topic-precision@k (% of top-k whose topic contains ``topic_keyword``)
    - filter-precision@k (% of top-k satisfying a metadata filter, e.g. speaker/year)

* **Groundedness** — drives the full ReAct agent via ``app.respond`` and checks
  each answer for:
    - expected facts present (case-insensitive substring)
    - forbidden phrases absent (e.g. "based on my knowledge", "typically")
    - a tool was actually used (``must_use_tool``)
    - negative queries declare "no records" rather than fabricating

Usage:
    python -m evals.run_eval                 # run both
    python -m evals.run_eval --retrieval     # retrieval only (fast, no LLM)
    python -m evals.run_eval --groundedness  # agent groundedness only (slow)
    python -m evals.run_eval --selection "qwen3.6:35b [local · fast · default]"

The retrieval pass needs only the embedding model; the groundedness pass needs
Ollama (or a cloud key) since it invokes the live agent.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

GOLDEN = Path(__file__).with_name("golden_set.json")

# Phrases that indicate the model fabricated instead of consulting the archive.
_NO_RECORDS_MARKERS = (
    "no records", "no matching", "contains no", "does not contain",
    "doesn't contain", "could not find", "no sermons", "not in the archive",
    "no results", "nothing in the archive",
)


def _load_golden() -> dict:
    with GOLDEN.open() as f:
        return json.load(f)


# ── Retrieval ────────────────────────────────────────────────────────────────

def _build_where(item: dict) -> dict | None:
    """Build the Chroma `where` clause EXCLUDING speaker (see vector_tool.py):
    Chroma metadata filters are exact-match only, but speakers carry titles
    ("SP Chua Seng Lee"), so a partial speaker filter is applied via post-filter
    oversampling instead of a `where` clause.
    """
    conds = []
    if item.get("year") is not None:
        conds.append({"year": {"$eq": item["year"]}})
    if item.get("min_year") is not None:
        conds.append({"year": {"$gte": item["min_year"]}})
    if item.get("max_year") is not None:
        conds.append({"year": {"$lte": item["max_year"]}})
    if len(conds) == 1:
        return conds[0]
    if len(conds) > 1:
        return {"$and": conds}
    return None


def _retrieved_sermon_ids(results: list[dict]) -> list[str]:
    ids = []
    for r in results:
        sid = (r.get("metadata") or {}).get("sermon_id")
        if sid:
            ids.append(sid)
    return ids


def _search_like_the_tool(vs, item: dict, k: int) -> list[dict]:
    """Replicate search_sermons_tool's filter handling so the eval tests what the
    agent actually sees: year/min_year/max_year as a Chroma `where`, speaker as
    oversample + case-insensitive substring post-filter."""
    where = _build_where(item)
    fetch_k = max(k, 5)
    speaker = item.get("speaker")
    if speaker:
        fetch_k = max(fetch_k * 4, 20)
    results = vs.search_sermons(item["query"], k=fetch_k, where=where)
    if speaker and results:
        needle = speaker.lower()
        results = [r for r in results
                   if needle in ((r.get("metadata") or {}).get("speaker") or "").lower()]
    return results[:max(k, 5)]


def run_retrieval(verbose: bool = False) -> dict:
    from src.storage.chroma_store import SermonVectorStore
    vs = SermonVectorStore()
    golden = _load_golden()
    items = golden["retrieval"]
    rows = []
    for item in items:
        k = item["k"]
        results = _search_like_the_tool(vs, item, k)
        retrieved = _retrieved_sermon_ids(results)
        retrieved_set = set(retrieved)

        # recall@k (HARD gate when must_find / must_find_any present)
        must = item.get("must_find") or []
        must_any = item.get("must_find_any") or []
        recall_gate_ok = True
        if must:
            hit = len(must_set := set(must) & retrieved_set)
            recall = hit / len(must)
            recall_detail = f"{hit}/{len(must)}"
            recall_gate_ok = hit == len(must)
        elif must_any:
            hit = len(set(must_any) & retrieved_set)
            recall = 1.0 if hit else 0.0
            recall_detail = f"{'OK' if hit else 'MISS'} (1-of-any)"
            recall_gate_ok = bool(hit)
        else:
            recall = 1.0
            recall_detail = "n/a"

        # topic-precision@k (SOFT metric — reported, not a hard gate). Substring match
        # on topic undercounts semantic relevance ("The Call to Pray" is relevant to a
        # "prayer" query but lacks the exact word), so we treat it as a quality signal,
        # not pass/fail. The relevant set also includes must_find/must_find_any ids.
        kw = (item.get("topic_keyword") or "").lower()
        if kw or must or must_any:
            rel_ids = set(must) | set(must_any)
            relevant = sum(
                1 for r in results
                if ((r.get("metadata") or {}).get("sermon_id") in rel_ids)
                or (kw and kw in ((r.get("metadata") or {}).get("topic") or "").lower())
            )
            precision = relevant / max(len(results), 1)
            prec_detail = f"{relevant}/{len(results)}"
        else:
            precision = 1.0
            prec_detail = "n/a"

        # filter-precision@k (HARD gate when a filter is specified — all results must match)
        filt = item.get("filter_check")
        if filt == "speaker_contains":
            val = str(item["filter_value"]).lower()
            ok = sum(1 for r in results if val in ((r.get("metadata") or {}).get("speaker") or "").lower())
            filter_prec = ok / max(len(results), 1)
            filter_detail = f"{ok}/{len(results)}"
            filter_gate_ok = ok == len(results) and len(results) > 0
        elif filt == "year_equals":
            val = item["filter_value"]
            ok = sum(1 for r in results if (r.get("metadata") or {}).get("year") == val)
            filter_prec = ok / max(len(results), 1)
            filter_detail = f"{ok}/{len(results)}"
            filter_gate_ok = ok == len(results) and len(results) > 0
        else:
            filter_prec = 1.0
            filter_detail = "n/a"
            filter_gate_ok = True

        passed = recall_gate_ok and filter_gate_ok
        rows.append({
            "id": item["id"], "query": item["query"], "k": k,
            "recall": recall, "recall_detail": recall_detail,
            "precision": precision, "prec_detail": prec_detail,
            "filter_prec": filter_prec, "filter_detail": filter_detail,
            "retrieved": retrieved, "passed": passed,
        })
        if verbose:
            print(f"  [{'PASS' if passed else 'FAIL'}] {item['id']}: recall={recall_detail} "
                  f"prec={prec_detail} filter={filter_detail}")

    n = len(rows)
    avg_recall = sum(r["recall"] for r in rows) / n if n else 0.0
    avg_prec = sum(r["precision"] for r in rows) / n if n else 0.0
    pass_rate = sum(1 for r in rows if r["passed"]) / n if n else 0.0
    return {
        "mode": "retrieval",
        "items": n,
        "avg_recall": round(avg_recall, 3),
        "avg_precision": round(avg_prec, 3),
        "pass_rate": round(pass_rate, 3),
        "rows": rows,
    }


# ── Groundedness ─────────────────────────────────────────────────────────────

def _check_groundedness(answer: str, tools_used: list, item: dict) -> dict:
    ans = (answer or "").lower()
    checks = []

    for fact in item.get("expected_facts", []):
        ok = fact.lower() in ans
        checks.append({"check": f"expected_fact:{fact}", "ok": ok})

    for phrase in item.get("forbidden_phrases", []):
        ok = phrase.lower() not in ans
        checks.append({"check": f"forbidden_phrase_absent:{phrase}", "ok": ok})

    if item.get("must_use_tool"):
        ok = len(tools_used) > 0
        checks.append({"check": "used_a_tool", "ok": ok})

    if item.get("expect_no_records"):
        ok = any(m in ans for m in _NO_RECORDS_MARKERS)
        checks.append({"check": "declares_no_records", "ok": ok})

    passed = all(c["ok"] for c in checks)
    return {"passed": passed, "checks": checks}


def run_groundedness(selection: str, verbose: bool = False) -> dict:
    # Import app lazily so --retrieval doesn't spin up the agent/Ollama.
    from app import respond
    golden = _load_golden()
    items = golden["groundedness"]
    rows = []
    for item in items:
        t0 = time.time()
        try:
            answer, tools_used, _tok, elapsed = respond(item["question"], [], selection)
        except Exception as e:
            rows.append({"id": item["id"], "question": item["question"], "passed": False,
                         "elapsed": round(time.time() - t0, 1),
                         "checks": [{"check": "agent_no_exception", "ok": False}],
                         "answer": f"<ERROR: {e}>", "tools_used": []})
            if verbose:
                print(f"  [ERROR] {item['id']}: {e}")
            continue
        result = _check_groundedness(answer, tools_used, item)
        rows.append({
            "id": item["id"], "question": item["question"], "passed": result["passed"],
            "elapsed": round(elapsed, 1), "checks": result["checks"],
            "tools_used": tools_used,
            "answer": (answer[:300] + "…") if len(answer) > 300 else answer,
        })
        if verbose:
            tag = "PASS" if result["passed"] else "FAIL"
            failed = [c["check"] for c in result["checks"] if not c["ok"]]
            print(f"  [{tag}] {item['id']} ({round(elapsed,1)}s, tools={tools_used})"
                  + (f" — failed: {failed}" if failed else ""))

    n = len(rows)
    pass_rate = sum(1 for r in rows if r["passed"]) / n if n else 0.0
    fact_pass = 0
    fact_tot = 0
    for r in rows:
        for c in r["checks"]:
            if c["check"].startswith("expected_fact:"):
                fact_tot += 1
                if c["ok"]:
                    fact_pass += 1
    return {
        "mode": "groundedness",
        "selection": selection,
        "items": n,
        "pass_rate": round(pass_rate, 3),
        "fact_recall": round(fact_pass / fact_tot, 3) if fact_tot else 1.0,
        "rows": rows,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def _print_summary(result: dict):
    print(f"\n{'='*60}\n{result['mode'].upper()} SUMMARY  ({result['items']} items)\n{'='*60}")
    if result["mode"] == "retrieval":
        print(f"  avg recall@k      : {result['avg_recall']}")
        print(f"  avg precision@k   : {result['avg_precision']}")
        print(f"  item pass rate    : {result['pass_rate']}")
        print(f"\n  {'ID':<28} {'recall':<14} {'prec':<10} {'filter':<10} {'ok'}")
        for r in result["rows"]:
            print(f"  {r['id']:<28} {r['recall_detail']:<14} {r['prec_detail']:<10} "
                  f"{r['filter_detail']:<10} {'✓' if r['passed'] else '✗'}")
    else:
        print(f"  selection         : {result.get('selection')}")
        print(f"  item pass rate    : {result['pass_rate']}")
        print(f"  expected-fact recall: {result['fact_recall']}")
        print(f"\n  {'ID':<22} {'time':<7} {'tools':<28} {'ok'}")
        for r in result["rows"]:
            tools = ",".join(r["tools_used"]) or "-"
            print(f"  {r['id']:<22} {r['elapsed']:<7} {tools:<28} {'✓' if r['passed'] else '✗'}")
            if not r["passed"]:
                failed = [c["check"] for c in r["checks"] if not c["ok"]]
                print(f"      ↳ failed checks: {failed}")


def main():
    p = argparse.ArgumentParser(description="BBTC RAG eval harness")
    p.add_argument("--retrieval", action="store_true", help="run retrieval eval only")
    p.add_argument("--groundedness", action="store_true", help="run groundedness eval only")
    p.add_argument("--selection", default=None, help="inference engine label for groundedness")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Default: run both. --retrieval → retrieval only; --groundedness → groundedness
    # only; both flags → both. (The earlier `not args.X` form made `--retrieval
    # --groundedness` run neither and exit 0 with an empty report.)
    run_ret = args.retrieval or not args.groundedness
    run_gnd = args.groundedness or not args.retrieval

    # Default selection = the app's default engine. Only resolve it (which imports
    # app.py and pre-warms the agent) when groundedness actually runs, so
    # `--retrieval` doesn't load the Ollama/agent stack.
    selection = args.selection
    if selection is None and run_gnd:
        try:
            from app import _DEFAULT_SELECTION
            selection = _DEFAULT_SELECTION
        except Exception:
            selection = None

    results = []
    if run_ret:
        print("▶ Running retrieval eval …")
        r = run_retrieval(verbose=args.verbose)
        results.append(r)
        _print_summary(r)
    if run_gnd:
        print("\n▶ Running groundedness eval (invokes live agent — this is slow) …")
        r = run_groundedness(selection, verbose=args.verbose)
        results.append(r)
        _print_summary(r)

    # Persist a JSON report next to the golden set.
    out = Path(__file__).with_name(f"eval_report.json")
    with out.open("w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\n📄 Full report written to {out}")

    # Exit non-zero if any pass-rate is below a 0.7 bar, so CI can gate on it.
    bar = 0.7
    failed_modes = [r["mode"] for r in results if r.get("pass_rate", 1.0) < bar]
    if failed_modes:
        print(f"⚠️  pass rate below {bar} for: {failed_modes}")
        sys.exit(1)


if __name__ == "__main__":
    main()