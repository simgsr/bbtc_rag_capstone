from sentence_transformers import CrossEncoder

_MODEL_NAME = "models/bge-reranker-v2-m3"


class Reranker:
    def __init__(self, model_name: str = _MODEL_NAME):
        self._model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates:
            return []
        pairs = [[query, c["content"]] for c in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [c for _, c in ranked[:top_k]]
