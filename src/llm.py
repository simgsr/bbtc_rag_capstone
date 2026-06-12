import os
from typing import Any, List
from dotenv import load_dotenv
from pydantic import PrivateAttr
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-3-flash-preview"

def _auto_detect_ollama_model(env_key: str) -> str:
    val = os.getenv(env_key)
    if val:
        return val
    try:
        import urllib.request, json
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3) as r:
            models = json.loads(r.read()).get("models", [])
        if models:
            name = models[0]["name"]
            print(f"ℹ️  {env_key} not set — auto-selected: {name}")
            return name
    except Exception:
        pass
    raise RuntimeError(
        f"{env_key} is not set in .env and no Ollama models were found. "
        "Pull a model first: ollama pull <model>"
    )

OLLAMA_CHAT_MODEL = _auto_detect_ollama_model("OLLAMA_CHAT_MODEL")
OLLAMA_INGEST_MODEL = _auto_detect_ollama_model("OLLAMA_INGEST_MODEL")
MLX_INGEST_MODEL = os.getenv("MLX_INGEST_MODEL", "mlx-community/Qwen3-4B-4bit")
MLX_CHAT_MODEL = os.getenv("MLX_CHAT_MODEL", "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit")
MLX_SERVER_HOST = os.getenv("MLX_SERVER_HOST", "127.0.0.1")
MLX_SERVER_PORT = int(os.getenv("MLX_SERVER_PORT", "8081"))
INGEST_PROVIDER = os.getenv("INGEST_PROVIDER", "ollama_local")

_mlx_server_proc = None


def _shutdown_mlx_server() -> None:
    """Terminate the mlx_lm.server subprocess if this process started it."""
    global _mlx_server_proc
    if _mlx_server_proc is None or _mlx_server_proc.poll() is not None:
        return
    print("🍎 Shutting down mlx_lm.server ...", flush=True)
    _mlx_server_proc.terminate()
    try:
        _mlx_server_proc.wait(timeout=10)
    except Exception:
        _mlx_server_proc.kill()
    _mlx_server_proc = None


def _register_mlx_cleanup() -> None:
    """Register atexit + signal handlers (no-op if mlx_lm.server was never started). Must run on main thread."""
    import atexit, signal
    atexit.register(_shutdown_mlx_server)
    sigs = [signal.SIGTERM, signal.SIGINT]
    if hasattr(signal, "SIGHUP"):  # not present on Windows
        sigs.append(signal.SIGHUP)
    for sig in sigs:
        try:
            prev = signal.getsignal(sig)
            def _handler(signum, frame, _prev=prev):
                _shutdown_mlx_server()
                if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                    _prev(signum, frame)
                else:
                    raise SystemExit(0)
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass  # only main thread can install signal handlers


_register_mlx_cleanup()


def _ensure_mlx_server(model: str, host: str = MLX_SERVER_HOST, port: int = MLX_SERVER_PORT) -> str:
    """Start mlx_lm.server on `port` if not already running. Returns OpenAI base_url."""
    global _mlx_server_proc
    import subprocess, sys, time, urllib.request
    base_url = f"http://{host}:{port}/v1"

    def _ping() -> bool:
        try:
            urllib.request.urlopen(f"{base_url}/models", timeout=2)
            return True
        except Exception:
            return False

    if _ping():
        return base_url

    if _mlx_server_proc is not None and _mlx_server_proc.poll() is not None:
        _mlx_server_proc = None

    cache_slots = int(os.getenv("MLX_PROMPT_CACHE_SLOTS", "4"))
    cache_bytes = int(os.getenv("MLX_PROMPT_CACHE_BYTES", "8000000000"))  # 8 GB KV cache
    print(f"🍎 Starting mlx_lm.server on {host}:{port} (model: {model}, prompt-cache: {cache_slots} slots / {cache_bytes//10**9} GB) ...", flush=True)
    _mlx_server_proc = subprocess.Popen(
        [sys.executable, "-m", "mlx_lm", "server",
         "--model", model, "--host", host, "--port", str(port),
         "--prompt-cache-size", str(cache_slots),
         "--prompt-cache-bytes", str(cache_bytes),
         "--log-level", "INFO"],
        stdout=None, stderr=None,  # route to parent terminal for debugging
    )

    deadline = time.time() + 180
    while time.time() < deadline:
        if _ping():
            print("🍎 mlx_lm.server ready", flush=True)
            return base_url
        if _mlx_server_proc.poll() is not None:
            raise RuntimeError("mlx_lm.server exited before becoming ready")
        time.sleep(1)
    raise TimeoutError("mlx_lm.server did not become ready within 180s")


class MLXChatModel(BaseChatModel):
    """LangChain-compatible chat model backed by mlx-lm (Apple Silicon only)."""
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 512
    _mlx_model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError("mlx-lm is not installed. Run: pip install mlx-lm")
        print(f"🍎 Loading MLX model: {self.model_name} ...", flush=True)
        self._mlx_model, self._tokenizer = load(self.model_name)

    MLX_TIMEOUT: int = 90  # seconds before declaring a hung generation

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        import re
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        if hasattr(self._tokenizer, "apply_chat_template"):
            role_map = {"human": "user", "ai": "assistant", "system": "system"}
            msg_dicts = [{"role": role_map.get(m.type, "user"), "content": m.content} for m in messages]
            try:
                # Disable Qwen3 thinking mode — saves tokens for structured ingest tasks
                prompt = self._tokenizer.apply_chat_template(
                    msg_dicts, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            except TypeError:
                prompt = self._tokenizer.apply_chat_template(
                    msg_dicts, tokenize=False, add_generation_prompt=True
                )
        else:
            prompt = "\n".join(m.content for m in messages)
        sampler = make_sampler(temp=self.temperature)

        def _run():
            return generate(
                self._mlx_model, self._tokenizer,
                prompt=prompt, max_tokens=self.max_tokens,
                sampler=sampler, verbose=False,
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run)
            try:
                response = future.result(timeout=self.MLX_TIMEOUT)
            except FuturesTimeout:
                raise TimeoutError(f"MLX generate timed out after {self.MLX_TIMEOUT}s")

        # Strip residual <think>...</think> blocks (e.g. if model ignores enable_thinking)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=response))])

    @property
    def _llm_type(self) -> str:
        return "mlx"


def get_ingest_llm():
    """Returns the ingest LLM selected by INGEST_PROVIDER env var (default: ollama_local)."""
    if INGEST_PROVIDER == "mlx":
        return get_llm(provider="mlx", model=MLX_INGEST_MODEL)
    return get_llm(provider=INGEST_PROVIDER, model=OLLAMA_INGEST_MODEL)


def get_chat_llm(provider: str = "ollama_local", temperature: float = 0.1):
    """Returns the chat-agent LLM. For provider='mlx', spins up mlx_lm.server and connects via ChatOpenAI (which supports tool calling)."""
    if provider == "mlx":
        from langchain_openai import ChatOpenAI
        base_url = _ensure_mlx_server(MLX_CHAT_MODEL)
        return ChatOpenAI(
            model=MLX_CHAT_MODEL,
            temperature=temperature,
            base_url=base_url,
            api_key="not-needed",
        )
    return get_llm(provider=provider, temperature=temperature)


def get_llm(provider="ollama_local", temperature=0, model=None):
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        return ChatGroq(model=model or GROQ_MODEL, temperature=temperature, api_key=api_key)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env")
        return ChatGoogleGenerativeAI(
            model=model or GEMINI_MODEL, temperature=temperature, google_api_key=api_key
        )

    if provider == "mlx":
        return MLXChatModel(model_name=model or MLX_INGEST_MODEL, temperature=float(temperature))

    from langchain_ollama import ChatOllama
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "32768"))
    return ChatOllama(
        model=model or OLLAMA_CHAT_MODEL,
        temperature=temperature,
        timeout=600,
        num_ctx=num_ctx,
    )
