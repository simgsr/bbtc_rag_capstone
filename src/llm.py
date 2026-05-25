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
INGEST_PROVIDER = os.getenv("INGEST_PROVIDER", "ollama_local")


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
    return ChatOllama(model=model or OLLAMA_CHAT_MODEL, temperature=temperature, timeout=120)
