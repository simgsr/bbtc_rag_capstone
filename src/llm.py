from langchain_ollama import ChatOllama
import os

GROQ_MODEL = "openai/gpt-oss-20b"


def get_llm(provider="ollama", temperature=0, ollama_model="gemma4:latest"):
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        return ChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=api_key)
    return ChatOllama(model=ollama_model, temperature=temperature)
