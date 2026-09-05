import json
from unittest.mock import patch, MagicMock

from langchain_ollama import ChatOllama


def test_get_llm_returns_chat_ollama():
    with patch("langchain_ollama.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm
        get_llm()
        mock_cls.assert_called_once()


def test_get_llm_passes_temperature():
    with patch("langchain_ollama.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm
        get_llm(temperature=0.5)
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("temperature") == 0.5


def test_get_llm_ollama_local_model():
    with patch("langchain_ollama.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm, OLLAMA_CHAT_MODEL
        get_llm(provider="ollama_local")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("model") == OLLAMA_CHAT_MODEL


# ── MLX server adoption: _ensure_mlx_server must never blindly adopt an
#    already-running server — it verifies the model the /v1/models endpoint
#    reports it actually serves. ────────────────────────────────────────────

class _FakeResp:
    """Stand-in for urllib's urlopen response: context manager whose .read()
    returns the JSON body."""

    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _models_resp(*model_ids: str) -> _FakeResp:
    payload = {"object": "list", "data": [{"id": mid} for mid in model_ids]}
    return _FakeResp(json.dumps(payload).encode())


def _base_url():
    from src import llm
    return f"http://{llm.MLX_SERVER_HOST}:{llm.MLX_SERVER_PORT}/v1"


def test_ensure_mlx_server_reuses_matching_running_server():
    from src import llm
    with patch("urllib.request.urlopen", return_value=_models_resp("want-model")), \
         patch.object(llm, "_mlx_server_proc", None), \
         patch.object(llm, "_mlx_server_model", None), \
         patch("subprocess.Popen") as popen:
        base = llm._ensure_mlx_server("want-model")
        assert base == _base_url()
        popen.assert_not_called()  # no server spawned — the running one is reused


def test_ensure_mlx_server_refuses_foreign_server_serving_wrong_model():
    """A server we didn't spawn that's serving a different model must be refused
    loudly, not adopted silently (wrong-model completions from wrong weights)."""
    import pytest
    from src import llm
    with patch("urllib.request.urlopen", return_value=_models_resp("other-model")), \
         patch.object(llm, "_mlx_server_proc", None), \
         patch.object(llm, "_mlx_server_model", None), \
         patch("subprocess.Popen") as popen:
        with pytest.raises(RuntimeError, match="already serving 'other-model'"):
            llm._ensure_mlx_server("want-model")
        popen.assert_not_called()


def test_ensure_mlx_server_restarts_own_server_when_serving_wrong_model():
    """Our own tracked server serving the wrong model is shut down and restarted
    with the requested model (verified against /models, not our bookkeeping)."""
    from src import llm
    responses = [
        _models_resp("old-model"),   # adoption check: our proc is serving the wrong one
        _models_resp("old-model"),   # not ready yet — still the old server
        _models_resp("want-model"),  # wait loop: restarted server now reports the right model
    ]
    with patch("urllib.request.urlopen", side_effect=responses), \
         patch.object(llm, "_mlx_server_proc", MagicMock()), \
         patch.object(llm, "_mlx_server_model", "old-model"), \
         patch.object(llm, "_shutdown_mlx_server") as shutdown, \
         patch("subprocess.Popen", return_value=MagicMock(poll=lambda: None)) as popen, \
         patch("time.sleep"):
        base = llm._ensure_mlx_server("want-model")
        assert base == _base_url()
        shutdown.assert_called_once()
        popen.assert_called_once()


