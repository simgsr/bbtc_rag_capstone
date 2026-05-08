from unittest.mock import patch, MagicMock
from langchain_ollama import ChatOllama


def test_get_llm_returns_chat_ollama():
    with patch("src.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm
        get_llm()
        mock_cls.assert_called_once()


def test_get_llm_passes_temperature():
    with patch("src.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm
        get_llm(temperature=0.5)
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("temperature") == 0.5


def test_get_llm_ollama_local_model():
    with patch("src.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm, OLLAMA_LOCAL_MODEL
        get_llm(provider="ollama_local")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("model") == OLLAMA_LOCAL_MODEL


def test_get_llm_ollama_deepseek_model():
    with patch("src.llm.ChatOllama") as mock_cls:
        mock_cls.return_value = MagicMock(spec=ChatOllama)
        from src.llm import get_llm, OLLAMA_DEEPSEEK_MODEL
        get_llm(provider="ollama_deepseek")
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs.get("model") == OLLAMA_DEEPSEEK_MODEL
