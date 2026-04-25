# tests/test_reranker.py
from unittest.mock import patch, MagicMock
from src.storage.reranker import Reranker


def _candidates():
    return [
        {"content": "Jesus is Lord", "metadata": {}, "distance": 0.9},
        {"content": "Grace is unmerited favour", "metadata": {}, "distance": 0.8},
        {"content": "Faith without works is dead", "metadata": {}, "distance": 0.7},
    ]


def test_reranker_returns_top_k():
    with patch("src.storage.reranker.CrossEncoder") as mock_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9, 0.3]
        mock_cls.return_value = mock_model

        r = Reranker()
        results = r.rerank("grace", _candidates(), top_k=2)

    assert len(results) == 2


def test_reranker_orders_by_score_descending():
    with patch("src.storage.reranker.CrossEncoder") as mock_cls:
        mock_model = MagicMock()
        # second candidate should score highest
        mock_model.predict.return_value = [0.2, 0.95, 0.4]
        mock_cls.return_value = mock_model

        r = Reranker()
        results = r.rerank("grace", _candidates(), top_k=3)

    assert results[0]["content"] == "Grace is unmerited favour"


def test_reranker_handles_empty_candidates():
    with patch("src.storage.reranker.CrossEncoder"):
        r = Reranker()
        assert r.rerank("grace", [], top_k=5) == []


def test_reranker_top_k_larger_than_candidates():
    with patch("src.storage.reranker.CrossEncoder") as mock_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.9]
        mock_cls.return_value = mock_model

        r = Reranker()
        results = r.rerank("grace", _candidates()[:2], top_k=10)

    assert len(results) == 2
