from langchain_ollama import ChatOllama
import os

GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-3-flash-preview"
OLLAMA_LOCAL_MODEL = "macdev/gpt-oss20b-large-ctx"
OLLAMA_DEEPSEEK_MODEL = "deepseek-v4-flash:cloud"


def get_llm(provider="ollama_local", temperature=0):
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        return ChatGroq(model=GROQ_MODEL, temperature=temperature, api_key=api_key)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env")
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL, temperature=temperature, google_api_key=api_key
        )

    model = OLLAMA_DEEPSEEK_MODEL if provider == "ollama_deepseek" else OLLAMA_LOCAL_MODEL
    return ChatOllama(model=model, temperature=temperature)
