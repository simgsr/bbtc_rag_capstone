"""BBTC Sermon Intelligence — Gradio chat UI + LangGraph ReAct agent.

Application entry point (run with ``python app.py``; serves on 127.0.0.1:7860).
It wires together:

  * A LangGraph ReAct agent (``create_react_agent``) backed by a runtime-
    selectable chat LLM — MLX (Apple Silicon), Ollama, Groq, or Gemini — chosen
    via the "Inference Engine" dropdown. LLM construction lives in ``src/llm.py``.
  * Five agent tools: SQL over ``data/sermons.db`` (``make_sql_tool``), semantic
    sermon search + Bible search / translation lookup over ChromaDB
    (``make_vector_tool`` / ``make_bible_tool``), and Plotly charts
    (``make_viz_tool``).
  * A Gradio Blocks UI (chat, engine picker, quick-query pills, live archive
    stats) defined at the bottom of this file (``with gr.Blocks() as demo``).

Caching: agents/LLMs are cached per provider in ``_agent_cache`` / ``_llm_cache``,
except MLX which is rebuilt per call (see ``respond`` and CLAUDE.md → "Notable
Quirks" for the httpx "client closed" workaround). The Ollama backend can spawn
and reap a local ``ollama serve`` (``_ensure_ollama`` / ``_shutdown_ollama``);
the MLX ``mlx_lm.server`` is managed in ``src/llm.py``.

See CLAUDE.md → "Architecture" for the end-to-end data flow.
"""
import warnings
from langgraph.warnings import LangGraphDeprecatedSinceV10
warnings.filterwarnings("ignore", category=LangGraphDeprecatedSinceV10)
import gradio as gr
import os
import subprocess
import time
import urllib.request
from dotenv import load_dotenv
from src.storage.chroma_store import SermonVectorStore
from src.llm import get_llm, get_chat_llm, GROQ_MODEL, GEMINI_MODEL, OLLAMA_CHAT_MODEL, MLX_CHAT_MODEL
from src.ui_helpers import extract_chart_path, fetch_archive_stats, render_stats_bar
from src.storage.sqlite_store import SermonRegistry
from src.tools.sql_tool import make_sql_tool
from src.tools.vector_tool import make_vector_tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from src.tools.viz_tool import make_viz_tool
from src.tools.bible_tool import make_bible_tool
import plotly.io as pio
from langgraph.prebuilt import create_react_agent

load_dotenv()

# Ensure Plotly uses light template for consistency
pio.templates.default = "plotly_white"

# ─────────────────────────────────────────────────────────────────────────────
# Ollama subprocess lifecycle
# Spawn/reap a local `ollama serve` ONLY if we started it (never touches a
# pre-existing system daemon). Cleanup mirrors the MLX handling in src/llm.py.
# ─────────────────────────────────────────────────────────────────────────────
_ollama_proc = None  # only set if we spawn `ollama serve` ourselves


def _shutdown_ollama() -> None:
    """Terminate ollama serve if (and only if) this process started it."""
    global _ollama_proc
    if _ollama_proc is None or _ollama_proc.poll() is not None:
        return
    print("🦙 Shutting down ollama serve ...", flush=True)
    _ollama_proc.terminate()
    try:
        _ollama_proc.wait(timeout=10)
    except Exception:
        _ollama_proc.kill()
    _ollama_proc = None


def _register_ollama_cleanup() -> None:
    """Register atexit + signal handlers (no-op if we didn't spawn ollama). Must run on main thread."""
    import atexit, signal
    atexit.register(_shutdown_ollama)
    sigs = [signal.SIGTERM, signal.SIGINT]
    if hasattr(signal, "SIGHUP"):
        sigs.append(signal.SIGHUP)
    for sig in sigs:
        try:
            prev = signal.getsignal(sig)
            def _handler(signum, frame, _prev=prev):
                _shutdown_ollama()
                if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                    _prev(signum, frame)
                else:
                    raise SystemExit(0)
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass


_register_ollama_cleanup()


def _ensure_ollama(timeout: int = 20) -> bool:
    global _ollama_proc

    def _is_up() -> bool:
        try:
            urllib.request.urlopen("http://127.0.0.1:11434", timeout=2)
            return True
        except Exception:
            return False

    if _is_up():
        return True  # already running externally — leave it alone

    print("🦙 Ollama not running — starting it now...")
    _ollama_proc = subprocess.Popen(
        ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1)
        if _is_up():
            print("🦙 Ollama is ready.")
            return True

    print("⚠️  Ollama did not start within the timeout.")
    return False
_ollama_up = _ensure_ollama()

try:
    registry = SermonRegistry()
    vector_store = SermonVectorStore()

    sql_tool = make_sql_tool(registry.db_path)
    vector_tool = make_vector_tool(vector_store)
    viz_tool = make_viz_tool(registry)
    get_bible_versions_tool, search_bible_tool = make_bible_tool(vector_store)

    from datetime import date as _date
    _today = _date.today().isoformat()
    _cur_year = _date.today().year

    SYSTEM_PROMPT = (
        f"You are the BBTC Sermon Intelligence Assistant for Bethesda Bedok-Tampines Church.\n"
        f"Today is {_today}. The archive covers 2015–{_cur_year}. "
        f"When a user says 'last N years', compute min_year = {_cur_year} - N + 1.\n\n"

        "## CRITICAL RULES — obey before doing anything else\n"
        "1. ALWAYS call at least one tool before composing any answer. No exceptions.\n"
        "2. Every fact, name, number, verse, or date in your reply MUST originate from a tool result "
        "in this conversation. Never draw on training-data knowledge.\n"
        "3. If tools return no matching data, say: 'The archive contains no records matching that query.' "
        "Do not invent, estimate, or extrapolate.\n"
        "4. Attribute every claim to its source: 'The database shows…', 'The search returned…', "
        "'According to the archive…'. Never say 'Based on my knowledge' or 'Typically…'.\n"
        "5. For Semantic Analysis / trends / thematic shifts: MUST call search_sermons_tool and synthesize "
        "actual sermon content — do NOT merely list counts or book names.\n\n"

        "## Tool selection\n"
        "- Counts, lists, dates, speakers, verses, statistics: sql_query_tool\n"
        "- Sermon content, doctrine, theology, 'what was preached about X': "
        "sql_query_tool THEN search_sermons_tool\n"
        "- Trends, thematic analysis, shifts in emphasis: "
        "sql_query_tool (themes by year) THEN search_sermons_tool (semantic synthesis)\n"
        "- A speaker's focus or evolution across years: "
        "sql_query_tool THEN search_sermons_tool with speaker filter\n"
        "- Chart / graph / plot: viz_tool\n"
        "- Compare Bible translations of a specific verse: get_bible_versions_tool\n"
        "- Find Bible passages on a topic or theme: search_bible_tool\n\n"

        "## sql_query_tool\n"
        "Schema:\n"
        "  sermons(sermon_id TEXT PK, date TEXT 'YYYY-MM-DD', year INTEGER,\n"
        "          language TEXT, speaker TEXT, topic TEXT, theme TEXT,\n"
        "          summary TEXT, key_verse TEXT, status TEXT)\n"
        "  verses(id INTEGER PK, sermon_id TEXT FK, verse_ref TEXT, book TEXT,\n"
        "         chapter INTEGER, verse_start INTEGER, verse_end INTEGER, is_key_verse INTEGER)\n"
        "  book_aliases(alias TEXT, canonical TEXT)\n\n"
        "Important notes:\n"
        "  - language is 'English' or 'Mandarin'. Add WHERE language='English' to exclude Mandarin sermons.\n"
        "  - Speaker names carry titles: 'Ps [Name]', 'SP [Name]', 'Elder [Name]'.\n"
        "    Examples: 'Ps Jeffrey Aw', 'SP Chua Seng Lee', 'Elder Lok Vi Ming'.\n"
        "    Use LIKE with wildcards for partial matches: WHERE speaker LIKE '%Aw%'\n"
        "  - Join verses to book_aliases for canonical book names:\n"
        "    LEFT JOIN book_aliases ba ON LOWER(TRIM(v.book))=ba.alias\n\n"
        "Ready-to-use queries:\n\n"
        "Sermon count by year:\n"
        "  SELECT year, COUNT(*) n FROM sermons WHERE year IS NOT NULL GROUP BY year ORDER BY year\n\n"
        "Themes by year:\n"
        "  SELECT year, COUNT(*) sermon_count, GROUP_CONCAT(DISTINCT theme) themes\n"
        "  FROM sermons WHERE year>=2015 AND year IS NOT NULL GROUP BY year ORDER BY year\n\n"
        "Top 5 verses per year:\n"
        "  SELECT year, verse_ref, cnt FROM (\n"
        "    SELECT s.year, v.verse_ref, COUNT(*) AS cnt,\n"
        "    ROW_NUMBER() OVER (PARTITION BY s.year ORDER BY COUNT(*) DESC) AS rn\n"
        "    FROM verses v JOIN sermons s ON v.sermon_id=s.sermon_id\n"
        "    GROUP BY s.year, v.verse_ref) WHERE rn<=5 ORDER BY year DESC, rn\n\n"
        "Most preached Bible books:\n"
        "  SELECT ba.canonical, COUNT(*) n FROM verses v\n"
        "  LEFT JOIN book_aliases ba ON LOWER(TRIM(v.book))=ba.alias\n"
        "  WHERE v.book IS NOT NULL GROUP BY ba.canonical ORDER BY n DESC LIMIT 15\n\n"
        "Books never preached (use this query verbatim — do not simplify):\n"
        "  SELECT bb.book_name, bb.testament FROM bible_books bb\n"
        "  WHERE bb.book_name NOT IN (\n"
        "    SELECT DISTINCT COALESCE(ba.canonical,v.book) FROM verses v\n"
        "    LEFT JOIN book_aliases ba ON LOWER(TRIM(v.book))=ba.alias\n"
        "    WHERE v.book IS NOT NULL AND v.book!='')\n"
        "  ORDER BY bb.book_order\n\n"
        "A speaker's most-used verses:\n"
        "  SELECT v.verse_ref, COUNT(*) n FROM verses v\n"
        "  JOIN sermons s USING(sermon_id) WHERE s.speaker LIKE '%Name%'\n"
        "  GROUP BY v.verse_ref ORDER BY n DESC LIMIT 10\n\n"
        "Keyword / doctrine search:\n"
        "  SELECT topic, speaker, date, summary FROM sermons\n"
        "  WHERE lower(topic) LIKE '%keyword%' OR lower(summary) LIKE '%keyword%'\n"
        "  ORDER BY date DESC LIMIT 10\n\n"
        "Recent sermons:\n"
        "  SELECT date, speaker, topic, key_verse FROM sermons\n"
        "  WHERE year IS NOT NULL ORDER BY date DESC LIMIT 20\n\n"

        "## search_sermons_tool\n"
        "Semantic search over sermon body text and LLM-generated summaries.\n"
        "  - query: 3-5 word concept phrase (e.g. 'cost of discipleship', 'Holy Spirit empowerment')\n"
        "  - k: 5 (default) for specific topics; 8-10 for broad thematic queries\n"
        "  - min_year / max_year: optional integer year range filters\n"
        "  - speaker: optional partial speaker name filter (e.g. 'Aw')\n"
        "Try 2-3 query variants if the first call returns few or no results.\n\n"

        "## get_bible_versions_tool\n"
        "Returns all stored translations (KJV, ASV, YLT, NIV, ESV) of a specific verse.\n"
        "  - Pass the canonical reference exactly: 'Book Chapter:Verse' (e.g. 'John 3:16', '1 John 1:9')\n"
        "  - Use for Translation Audit requests or any question comparing Bible versions.\n\n"

        "## search_bible_tool\n"
        "Semantic search across the full Bible archive "
        "(KJV, ASV, YLT, NIV, ESV — approx. 102,000 verse-chunks).\n"
        "  - query: 3-6 word concept phrase (e.g. 'forgiveness of sins', 'walking by faith')\n"
        "  - k: number of results (default 5)\n"
        "Returns verse text with reference and translation label.\n\n"

        "## viz_tool\n"
        "Only for explicit chart / graph / plot requests.\n"
        "  - chart_name: sermons_per_speaker | sermons_per_year | verses_per_book | sermons_scatter\n"
        "  - top_n (integer, default 15): controls the number of bars in ranked charts "
        "(sermons_per_speaker, verses_per_book)\n"
        "Return the file path from the tool exactly as-is in your response.\n\n"

        "## Response format\n"
        "- Structure answers with markdown headers (##) and bullet lists where helpful.\n"
        "- Bold key terms, speaker names, and verse references.\n"
        "- Cite every claim: include speaker name, date, or verse reference.\n"
        "- For lists of sermons, use a numbered list or table.\n"
        "- For thematic analysis, write in paragraphs with direct evidence from search results.\n"
        "- If a record has missing data (e.g. no date), note that explicitly rather than omitting it.\n"
        "- Keep responses focused; do not pad with generic observations.\n\n"

        "## Formatting\n"
        "Use plain text and Unicode only. Do NOT use LaTeX notation "
        "(e.g. $\\rightarrow$, $\\cdot$, $\\to$, $\\times$). "
        "Use Unicode instead: → · × ≥ ≤ ≈\n"
    )

    _agent_cache: dict = {}
    _llm_cache: dict = {}  # strong refs to LLMs — prevents GC from closing their httpx clients

    def _build_agent(provider: str, model: str = None):
        _llm = get_chat_llm(provider=provider, temperature=0.1, model=model)
        _llm_cache[provider] = _llm  # strong ref so httpx client isn't GC-closed
        return create_react_agent(
            _llm,
            tools=[sql_tool, vector_tool, viz_tool, get_bible_versions_tool, search_bible_tool],
            prompt=SystemMessage(content=SYSTEM_PROMPT),
        )

    def get_agent(provider: str = "ollama_local", model: str = None):
        # MLX path uses mlx_lm.server's OpenAI-compat layer, which can leave the
        # httpx client in a closed state after some streamed responses. Rebuild
        # per call — the model stays loaded in mlx_lm.server so the cost is millis
        # (and it lets the server swap weights when a different MLX model is picked).
        if provider == "mlx":
            return _build_agent(provider, model)
        if provider not in _agent_cache:
            _agent_cache[provider] = _build_agent(provider, model)
        return _agent_cache[provider]

    # Pre-warm MLX agent at startup (spawns mlx_lm.server + loads model)
    get_agent("mlx")
    _init_ok = True

except Exception as e:
    print(f"⚠️ Initialization warning: {e}")
    _init_ok = False
    registry = None
    vector_store = None
    get_agent = None

_stats_bar_html = (
    render_stats_bar(fetch_archive_stats(registry.db_path))
    if registry is not None
    else render_stats_bar(None)
)

_ollama_status = "online" if _ollama_up else "offline"

# ── Inference Engine options ─────────────────────────────────────────────────
# Selectable MLX chat models (the configured default first, then other known repos).
# Only one MLX model is served at a time; switching restarts mlx_lm.server (see src/llm.py).
_KNOWN_MLX_MODELS = [
    "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
    "mlx-community/Qwen3-Next-80B-A3B-Instruct-8bit",
]
MLX_CHAT_MODELS = list(dict.fromkeys([MLX_CHAT_MODEL, *_KNOWN_MLX_MODELS]))

# Dropdown choices as (label, value) — value encodes provider (+ MLX model repo).
_ENGINE_CHOICES = (
    [(f"{m.split('/')[-1]}  ·  MLX", f"mlx::{m}") for m in MLX_CHAT_MODELS]
    + [
        (f"{OLLAMA_CHAT_MODEL}  ·  Ollama", "ollama_local"),
        ("Groq  ·  Cloud", "groq"),
        ("Gemini 3 Flash  ·  Cloud", "gemini"),
    ]
)
_DEFAULT_SELECTION = f"mlx::{MLX_CHAT_MODEL}"


def _parse_selection(selection: str):
    """Split a dropdown value into (provider, model). model is None for non-MLX providers."""
    if selection and selection.startswith("mlx::"):
        return "mlx", selection.split("::", 1)[1]
    return selection or "ollama_local", None


def _inference_badge_html(selection: str) -> str:
    provider, model = _parse_selection(selection)
    if provider == "groq":
        has_key = bool(os.getenv("GROQ_API_KEY"))
        status = "online" if has_key else "offline"
        label = f"groq · {GROQ_MODEL}" if has_key else "groq · no key"
    elif provider == "gemini":
        has_key = bool(os.getenv("GOOGLE_API_KEY"))
        status = "online" if has_key else "offline"
        label = f"gemini · {GEMINI_MODEL}" if has_key else "gemini · no key"
    elif provider == "mlx":
        status = "online"
        label = f"{(model or MLX_CHAT_MODEL).split('/')[-1]} · mlx"
    else:  # ollama_local
        status = _ollama_status
        label = f"{OLLAMA_CHAT_MODEL} · local"
    return (
        "<div style='display:flex;justify-content:space-between;align-items:center;margin-top:8px;'>"
        f"<span style='color:var(--c-text-2);font-family:\"Source Code Pro\",monospace;font-size:0.72rem;'>inference</span>"
        f"<span class='status-badge status-{status}'>{label}</span>"
        "</div>"
    )


_TOOL_LABELS = {
    "sql_query_tool": "SQL",
    "search_sermons_tool": "Vector Search",
    "viz_tool": "Visualization",
    "get_bible_versions_tool": "Bible Versions",
    "search_bible_tool": "Bible Search",
}

_PROVIDER_DISPLAY = {
    "ollama_local": OLLAMA_CHAT_MODEL,
    "groq": GROQ_MODEL,
    "gemini": GEMINI_MODEL,
    "mlx": MLX_CHAT_MODEL.split("/")[-1],
}


def _build_meta_footer(tools_used: list, token_info: dict, provider: str = "", elapsed: float = 0, model: str = None) -> str:
    left = ""
    if tools_used:
        labels = [_TOOL_LABELS.get(t, t) for t in tools_used]
        left = " · ".join(labels)

    right_parts = []
    if provider:
        if provider == "mlx" and model:
            model_name = model.split("/")[-1]
        else:
            model_name = _PROVIDER_DISPLAY.get(provider, provider)
        right_parts.append(f"LangGraph ReAct · {model_name}")
    if elapsed:
        right_parts.append(f"{elapsed:.1f}s")
    total = token_info.get("total") or (token_info.get("input", 0) + token_info.get("output", 0))
    if total:
        right_parts.append(f"{total:,} tok")
    right = " · ".join(right_parts)

    if not left and not right:
        return ""

    return (
        "<div style='margin-top:10px;padding-top:8px;"
        "border-top:1px solid var(--c-border-light);"
        "display:flex;justify-content:space-between;align-items:center;"
        "font-size:0.68rem;font-family:monospace;color:var(--c-text-3);letter-spacing:0.4px;'>"
        f"<span>{left}</span>"
        f"<span>{right}</span>"
        "</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chat handler — the Gradio callback that runs one turn through the agent.
# Parses the engine selection, builds/loads the agent, passes the last 3
# exchanges of history, invokes the ReAct agent, and retries on the MLX
# "client has been closed" error (see CLAUDE.md → "Notable Quirks").
# ─────────────────────────────────────────────────────────────────────────────
def respond(message, history, selection="ollama_local"):
    if not _init_ok or get_agent is None:
        return "⚠️ Agent not initialized. Check that Ollama is running.", [], {}, 0.0

    provider, model = _parse_selection(selection)

    try:
        agent = get_agent(provider, model)
    except Exception as e:
        return f"⚠️ Could not load {provider} agent: {e}", [], {}, 0.0

    truncated_history = history[-6:] if len(history) > 6 else history
    messages = []
    for turn in truncated_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            content = turn["content"]
            if isinstance(content, list):
                text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
                content = " ".join(text_parts)
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=message))

    try:
        t0 = time.time()
        # httpx raises "Cannot send a request, as the client has been closed."
        # when the underlying client was closed externally (mlx_lm.server stream
        # teardown, transient connection drop, etc.). Evict any cached state for
        # this provider and rebuild — retry up to 3 times.
        result = None
        last_err = None
        for attempt in range(3):
            try:
                result = agent.invoke({"messages": messages})
                break
            except Exception as inner:
                last_err = inner
                if "client has been closed" not in str(inner):
                    raise
                _agent_cache.pop(provider, None)
                _llm_cache.pop(provider, None)
                agent = get_agent(provider, model)
        if result is None:
            raise last_err
        elapsed = time.time() - t0

        final_msg = result["messages"][-1]
        final = final_msg.content
        if isinstance(final, list):
            text_parts = [block.get("text", "") for block in final if isinstance(block, dict) and block.get("type") == "text"]
            final = "\n".join(text_parts)
        elif not isinstance(final, str):
            final = str(final)

        # If the LLM dropped the chart path from its response, recover it from ToolMessage
        if "/tmp/bbtc_chart_" not in final:
            import re
            for msg in result["messages"]:
                match = re.search(r'/tmp/bbtc_chart_[a-f0-9]+\.(png|json)', str(msg.content))
                if match:
                    final = final.rstrip() + "\n" + match.group(0)
                    break

        # Extract tool names from ToolMessages
        tools_used = []
        for msg in result["messages"]:
            if isinstance(msg, ToolMessage) and getattr(msg, "name", None) and msg.name not in tools_used:
                tools_used.append(msg.name)

        # Extract token counts (LangChain usage_metadata takes priority)
        token_info: dict = {}
        usage = getattr(final_msg, "usage_metadata", None)
        if usage:
            token_info = {
                "input": usage.get("input_tokens", 0),
                "output": usage.get("output_tokens", 0),
                "total": usage.get("total_tokens", 0),
            }
        else:
            meta = getattr(final_msg, "response_metadata", {}) or {}
            raw = meta.get("token_usage") or meta.get("usage") or {}
            if raw:
                inp = raw.get("prompt_tokens") or raw.get("input_tokens") or raw.get("prompt_eval_count", 0)
                out = raw.get("completion_tokens") or raw.get("output_tokens") or raw.get("eval_count", 0)
                token_info = {"input": inp, "output": out, "total": raw.get("total_tokens") or (inp + out)}

        return final, tools_used, token_info, elapsed
    except Exception as e:
        return f"⚠️ An error occurred while processing your request: {e}", [], {}, 0.0


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

:root {
  --c-bg:           #f3f4f6;
  --c-surface:      #ffffff;
  --c-border:       #d7dae0;
  --c-border-light: #e6e8ec;
  --c-text:         #0f172a;
  --c-text-2:       #3f4757;
  --c-text-3:       #667085;
  --c-user-bg:      #f4f5f7;
  --c-user-border:  #d7dae0;
  --c-user-text:    #0f172a;
  --c-asst-bg:      #ffffff;
  --c-asst-border:  #e6e8ec;
  --c-asst-text:    #0f172a;
  --c-input-text:   #0f172a;
  --c-input-ph:     #98a2b3;
  --c-card-bg:      #ffffff;
  --c-card-border:  #e0e0e0;
  --c-card-sel-bg:  #ede9fe;
  --c-card-hov-bg:  #f5f3ff;
  --c-card-hov-bdr: #a78bfa;
  /* brand accent (cohesive indigo) */
  --c-accent:       #4f46e5;
  --c-accent-hov:   #4338ca;
  --c-accent-active:#3730a3;
  --c-accent-soft:  #eef2ff;
  /* elevation */
  --shadow-sm: 0 1px 2px rgba(16,24,40,0.04), 0 1px 3px rgba(16,24,40,0.07);
  --shadow-md: 0 4px 14px rgba(16,24,40,0.08);
  --shadow-focus: 0 0 0 3px rgba(79,70,229,0.18);
  --c-p1-bg:#ede9fe; --c-p1-bdr:#a78bfa; --c-p1-txt:#3b0764;
  --c-p2-bg:#dcfce7; --c-p2-bdr:#4ade80; --c-p2-txt:#14532d;
  --c-p3-bg:#fef9c3; --c-p3-bdr:#facc15; --c-p3-txt:#713f12;
  --c-p4-bg:#fee2e2; --c-p4-bdr:#f87171; --c-p4-txt:#7f1d1d;
  --c-p5-bg:#e0f2fe; --c-p5-bdr:#38bdf8; --c-p5-txt:#0c4a6e;
}

footer { visibility: hidden }
* { box-sizing: border-box; }
body { background: var(--c-bg) !important; }

.gradio-container {
    background: transparent !important;
    color: var(--c-text) !important;
    font-family: 'Source Code Pro', monospace !important;
    max-width: 1440px !important;
    font-size: 15px !important;
}

.dark, [data-theme="dark"] { color-scheme: light !important; }
.gradio-container, .main, #root { background: var(--c-bg) !important; color: var(--c-text) !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--c-border-light); }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.18); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.32); }

/* Header */
#title-container {
    padding: 16px 22px;
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: 8px;
    box-shadow: var(--shadow-sm);
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 12px;
}
#title-container img { height: 36px; opacity: 0.85; }
#title-text h1 {
    font-family: 'Source Code Pro', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0 0 2px 0;
    color: var(--c-text);
    letter-spacing: -0.3px;
    line-height: 1.2;
}
#title-text p {
    color: var(--c-text-3);
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0;
    font-family: 'Source Code Pro', monospace;
}

/* Stats bar */
.stats-bar {
    display: flex;
    gap: 0;
    background: var(--c-surface);
    border: 1px solid var(--c-border);
    border-radius: 8px;
    box-shadow: var(--shadow-sm);
    padding: 10px 18px;
    margin-bottom: 12px;
    color: var(--c-text-2);
    font-size: 0.78rem;
    letter-spacing: 0.3px;
    font-family: 'Source Code Pro', monospace;
}

/* Chat area */
.chatbot-container {
    border-radius: 8px !important;
    border: 1px solid var(--c-border) !important;
    background: var(--c-surface) !important;
    box-shadow: var(--shadow-sm) !important;
    overflow: hidden !important;
}

/* Messages */
.message-user {
    background: var(--c-user-bg) !important;
    border: 1px solid var(--c-user-border) !important;
    border-radius: 4px !important;
    padding: 9px 14px !important;
    color: var(--c-user-text) !important;
    font-size: 0.84rem !important;
    font-family: 'Source Code Pro', monospace !important;
    line-height: 1.6;
}
.message-assistant {
    background: var(--c-asst-bg) !important;
    border: 1px solid var(--c-asst-border) !important;
    border-radius: 4px !important;
    padding: 9px 14px !important;
    color: var(--c-asst-text) !important;
    font-size: 0.84rem !important;
    font-family: 'Source Code Pro', monospace !important;
    line-height: 1.65;
}

/* Plotly chart */
.gradio-plot {
    border-radius: 8px !important;
    overflow: hidden !important;
    border: 1px solid var(--c-border) !important;
    box-shadow: var(--shadow-sm) !important;
    margin-top: 8px !important;
    background: var(--c-surface) !important;
}

/* Input row — unified box */
#input-row {
    margin-top: 10px !important;
    border: 1px solid var(--c-border) !important;
    border-radius: 8px !important;
    background: var(--c-surface) !important;
    box-shadow: var(--shadow-sm) !important;
    overflow: hidden !important;
    gap: 0 !important;
    align-items: stretch !important;
    transition: border-color 0.12s ease, box-shadow 0.12s ease !important;
}
#input-row:focus-within { border-color: var(--c-accent) !important; box-shadow: var(--shadow-focus) !important; }
#input-row > div { padding: 0 !important; margin: 0 !important; border: none !important; background: transparent !important; }
#input-row textarea {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.84rem !important;
    color: var(--c-input-text) !important;
    background: transparent !important;
    border: none !important;
    outline: none !important;
    padding: 10px 14px !important;
    resize: none !important;
}
#input-row textarea::placeholder { color: var(--c-input-ph) !important; }

/* Submit button */
.btn-primary {
    background: var(--c-accent) !important;
    border: none !important;
    border-left: 1px solid var(--c-accent) !important;
    border-radius: 0 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    font-size: 0.7rem !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-family: 'Source Code Pro', monospace !important;
    box-shadow: none !important;
    min-width: 72px !important;
    transition: background 0.12s ease !important;
}
.btn-primary:hover { background: var(--c-accent-hov) !important; transform: none !important; filter: none !important; box-shadow: none !important; }
.btn-primary:active { background: var(--c-accent-active) !important; }
.btn-primary:disabled { background: #b6b9c2 !important; cursor: not-allowed !important; }

/* Secondary button (Clear) */
button.secondary {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    color: var(--c-text-2) !important;
    border-radius: 3px !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}
button.secondary:hover {
    background: #fee2e2 !important;
    border-color: #f87171 !important;
    color: #7f1d1d !important;
}

/* Sidebar */
.sidebar {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    padding: 20px !important;
    border-radius: 8px !important;
    box-shadow: var(--shadow-sm) !important;
}
.sidebar h3 {
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: var(--c-text-3) !important;
    margin-bottom: 12px !important;
}
.sidebar p, .sidebar li,
.sidebar .prose p, .sidebar .prose li,
.sidebar .prose strong, .sidebar .prose em,
.sidebar .markdown p, .sidebar .markdown li,
.sidebar .markdown strong, .sidebar .markdown em {
    font-size: 0.8rem !important;
    color: var(--c-text) !important;
    line-height: 1.6 !important;
    font-family: 'Source Code Pro', monospace !important;
}

/* Status badges */
.status-badge {
    padding: 2px 9px;
    border-radius: 3px;
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Source Code Pro', monospace;
    border: 1px solid;
}
.status-online  { background: #dcfce7; color: #166534; border-color: #4ade80; }
.status-offline { background: #fee2e2; color: #7f1d1d; border-color: #f87171; }

/* Inference Engine dropdown */
#model-dropdown, #model-dropdown .wrap, #model-dropdown .wrap-inner { background: transparent !important; }
#model-dropdown input {
    background: var(--c-card-bg) !important;
    border: 1px solid var(--c-card-border) !important;
    border-radius: 6px !important;
    color: var(--c-text) !important;
    font-family: 'Source Code Pro', monospace !important;
    font-size: 0.78rem !important;
    padding: 9px 12px !important;
    cursor: pointer !important;
    transition: border-color 0.12s ease, box-shadow 0.12s ease !important;
}
#model-dropdown input:focus {
    border-color: var(--c-accent) !important;
    box-shadow: var(--shadow-focus) !important;
    outline: none !important;
}
#model-dropdown ul.options {
    background: var(--c-surface) !important;
    border: 1px solid var(--c-border) !important;
    border-radius: 6px !important;
    box-shadow: var(--shadow-md) !important;
    font-family: 'Source Code Pro', monospace !important;
    z-index: 120 !important;
    padding: 4px !important;
}
#model-dropdown ul.options li.item {
    color: var(--c-text) !important;
    font-size: 0.76rem !important;
    padding: 8px 10px !important;
    border-radius: 4px !important;
}
#model-dropdown ul.options li.item:hover,
#model-dropdown ul.options li.item.selected,
#model-dropdown ul.options li.active {
    background: var(--c-accent-soft) !important;
    color: var(--c-accent-active) !important;
}

/* Quick query pills */
#quick-query-pills .examples-holder,
#quick-query-pills table,
#quick-query-pills tbody,
#quick-query-pills tr {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
    border: none !important;
    padding: 4px 0 !important;
    margin: 0 !important;
    background: transparent !important;
}
#quick-query-pills td { border: none !important; padding: 0 !important; }
#quick-query-pills .label-wrap { display: none !important; }
#quick-query-pills .example {
    border-radius: 2px !important;
    padding: 5px 13px !important;
    font-size: 0.73rem !important;
    font-family: 'Source Code Pro', monospace !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    transition: all 0.12s ease !important;
    line-height: 1.4 !important;
}
#quick-query-pills .example:hover {
    filter: brightness(0.92) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.12) !important;
}
#quick-query-pills .example:nth-child(1),
#quick-query-pills .example:nth-child(2),
#quick-query-pills .example:nth-child(3) {
    background: var(--c-p1-bg) !important; border: 1px solid var(--c-p1-bdr) !important; color: var(--c-p1-txt) !important;
}
#quick-query-pills .example:nth-child(4),
#quick-query-pills .example:nth-child(5),
#quick-query-pills .example:nth-child(6) {
    background: var(--c-p2-bg) !important; border: 1px solid var(--c-p2-bdr) !important; color: var(--c-p2-txt) !important;
}
#quick-query-pills .example:nth-child(7),
#quick-query-pills .example:nth-child(8) {
    background: var(--c-p3-bg) !important; border: 1px solid var(--c-p3-bdr) !important; color: var(--c-p3-txt) !important;
}
#quick-query-pills .example:nth-child(9) {
    background: var(--c-p4-bg) !important; border: 1px solid var(--c-p4-bdr) !important; color: var(--c-p4-txt) !important;
}
#quick-query-pills .example:nth-child(10) {
    background: var(--c-p5-bg) !important; border: 1px solid var(--c-p5-bdr) !important; color: var(--c-p5-txt) !important;
}
"""

_QUICK_QUERY_LABELS = [
    "📊 Scripture Coverage",
    "📖 Gap Analysis",
    "📈 Ministry Shifts",
    "✝️ BBTC Theology",
    "✝️ BBTC Theology",
    "🧭 Theological Themes",
    "📜 Bible Versions",
    "📖 Bible Passages",
    "👤 Speaker Specific",
    "📝 Sermons Specific",
]

_QUICK_QUERY_FULL = [
    ["Scripture Coverage: Generate a frequency heatmap of the most frequently preached Bible books."],
    ["Gap Analysis: List all Bible books that have never been preached in BBTC sermons."],
    ["Ministry Shifts: Identify shifts in ministry emphasis within BBTC over the last 5 years."],
    ["BBTC Theology: Do BBTC believe in once saved always saved?."],
    ["BBTC Theology: Explain the biblical sequence of End Times events based on BBTC teachings."],
    ["Theological Themes: Identify the consistent theological themes in BBTC's vision statements and pulpit series between 2015 and 2026"],
    ["Bible Translation: List all Bible translations of 1 John 1:9 in the bible arhives."],
    ["Find Bible passages about forgiveness and grace using the Bible archive."],
    ["Speaker Specific: Chart sermon count by speaker from 2015 to now,"],
    ["Sermons Specific: Summarize the key message and scripture shared in last week's sermon."],
]

_force_light_js = """
function() {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI layout — header + stats bar, chat column, and the sidebar (engine
# picker, quick-query pills). Event wiring is at the bottom of the block.
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="BBTC Sermon Intelligence") as demo:
    with gr.Row(elem_id="header"):
        with gr.Column(scale=4):
            gr.HTML("""
                <div id='title-container'>
                    <img src='https://www.bbtc.com.sg/wp-content/uploads/2021/04/BBTC-Logo-Header.png' alt='BBTC Logo'>
                    <div id='title-text'>
                        <h1>Sermon Intelligence</h1>
                        <p>Bethesda Bedok-Tampines Church &nbsp;·&nbsp; Agentic RAG</p>
                    </div>
                </div>
            """)

    gr.HTML(_stats_bar_html)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                min_height=550,
                show_label=False,
                elem_classes="chatbot-container",
                avatar_images=(None, "https://www.bbtc.com.sg/wp-content/uploads/2021/04/BBTC-Logo-Header.png")
            )
            with gr.Row(elem_id="input-row"):
                msg = gr.Textbox(
                    placeholder="Describe the data you need or ask a theological question...",
                    container=False,
                    lines=1,
                    max_lines=4,
                    scale=7,
                )
                submit = gr.Button("Send", variant="primary", scale=1, min_width=80, elem_classes="btn-primary")

            gr.Examples(
                examples=_QUICK_QUERY_FULL,
                inputs=msg,
                example_labels=_QUICK_QUERY_LABELS,
                label=None,
                elem_id="quick-query-pills",
            )

        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown("### System Health")

            vec_status = "online" if vector_store else "offline"
            gr.HTML(f"""
                <div style='display: flex; flex-direction: column; gap: 12px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='color: var(--c-text-2); font-size: 0.8rem; font-family: "Source Code Pro", monospace;'>Vector Store</span>
                        <span class='status-badge status-{vec_status}'>{vec_status}</span>
                    </div>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <span style='color: var(--c-text-2); font-size: 0.8rem; font-family: "Source Code Pro", monospace;'>SQL Registry</span>
                        <span class='status-badge status-online'>active</span>
                    </div>
                </div>
            """)
            inference_status = gr.HTML(_inference_badge_html(_DEFAULT_SELECTION))

            gr.Markdown("---")
            gr.Markdown("### Inference Engine")
            provider_dropdown = gr.Dropdown(
                choices=_ENGINE_CHOICES,
                value=_DEFAULT_SELECTION,
                show_label=False,
                interactive=True,
                filterable=False,
                container=False,
                elem_id="model-dropdown",
            )
            gr.HTML(
                "<p style='margin:6px 2px 0;font-size:0.68rem;line-height:1.45;"
                "color:var(--c-text-3);font-family:\"Source Code Pro\",monospace;'>"
                "Switching to a larger MLX model reloads its weights — the first response "
                "after a switch can take a few minutes.</p>"
            )
            provider_state = gr.State(_DEFAULT_SELECTION)

            gr.Markdown("---")
            gr.Markdown("### Capabilities")
            gr.Markdown(
                "- **Semantic Search**: Retrieval of sermon content across a decade of archives.\n"
                "- **SQL Analytics**: High-precision metadata querying for statistics and counts.\n"
                "- **Dynamic Viz**: Real-time generation of interactive Plotly visualizations.\n"
                "- **Bible Context**: Multi-version Bible referencing and cross-comparison."
            )

            gr.Markdown("---")
            clear = gr.Button("Clear Chat", variant="secondary")

    def _on_provider_change(selection):
        # `selection` is the dropdown value (e.g. "mlx::<repo>", "groq"). Store it as-is.
        return selection, _inference_badge_html(selection)

    provider_dropdown.change(
        _on_provider_change,
        inputs=provider_dropdown,
        outputs=[provider_state, inference_status],
    )

    def user_msg(user_message, history: list):
        if history is None:
            history = []
        return "", history + [{"role": "user", "content": user_message}]

    def bot_msg(history: list, selection: str):
        if not history or history[-1]["role"] != "user":
            return history

        user_message = history[-1]["content"]
        chat_history = history[:-1]
        bot_message, tools_used, token_info, elapsed = respond(user_message, chat_history, selection)

        provider, model = _parse_selection(selection)
        text, chart_path = extract_chart_path(bot_message)
        meta_footer = _build_meta_footer(tools_used, token_info, provider, elapsed, model)

        content = []
        if text:
            content.append({"type": "text", "text": text + meta_footer})

        if chart_path:
            if chart_path.endswith('.json'):
                try:
                    import plotly.io as pio
                    fig = pio.read_json(chart_path)
                    content.append(gr.Plot(fig))
                except Exception as e:
                    content.append({"type": "text", "text": f"\n⚠️ Error loading interactive chart: {e}"})
            else:
                content.append({"type": "image", "image": {"path": chart_path}})

        if not content:
            content = bot_message

        history.append({"role": "assistant", "content": content})
        return history

    disable_submit = lambda: gr.update(value="Processing…", interactive=False)
    enable_submit = lambda: gr.update(value="Send", interactive=True)

    msg.submit(user_msg, [msg, chatbot], [msg, chatbot], queue=True).then(
        disable_submit, None, submit
    ).then(
        bot_msg, [chatbot, provider_state], chatbot
    ).then(
        enable_submit, None, submit
    )
    submit.click(user_msg, [msg, chatbot], [msg, chatbot], queue=True).then(
        disable_submit, None, submit
    ).then(
        bot_msg, [chatbot, provider_state], chatbot
    ).then(
        enable_submit, None, submit
    )
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 0)) or None
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        theme=gr.themes.Default(),
        css=custom_css,
        js=_force_light_js,
        allowed_paths=["/tmp"]
    )
