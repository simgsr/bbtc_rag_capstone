# Documentation

Design notes and implementation plans for the BBTC Sermon Intelligence project.

For the **authoritative reference** on architecture, data flow, and operational
quirks, see [`CLAUDE.md`](../CLAUDE.md) in the repo root. For setup and usage, see
the top-level [`README.md`](../README.md). For maintainer/contributor workflow,
see [`CONTRIBUTING.md`](../CONTRIBUTING.md).

## Layout

| Path | What it is |
|---|---|
| [`plans/`](plans/) | **Active / recent** implementation plans |
| [`archive/`](archive/) | Historical design + plan docs, kept for provenance (the "why we built it this way" trail). Not maintained — treat as a snapshot in time |
| `archive/design/` | Design write-ups (RAG redesign, UI redesign, Bible-book normalization, cleanup) |
| `archive/plans/` | Step-by-step plans that produced the current codebase |
| `archive/superpowers/` | Earlier RAG-enhancement and UI experiments |

## Reading order for a new maintainer

1. Root [`README.md`](../README.md) — what it does, how to run it.
2. [`CLAUDE.md`](../CLAUDE.md) — architecture, components, schema, notable quirks.
3. [`CONTRIBUTING.md`](../CONTRIBUTING.md) — dev setup, tests, common operations.
4. `archive/design/` — background on major design decisions, if you need the "why".
