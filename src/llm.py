from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-3-flash-preview"

# Configurable via .env — see .env.example for device-capacity guidance
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma4:latest")
OLLAMA_INGEST_MODEL = os.getenv("OLLAMA_INGEST_MODEL", "gemma4:latest")

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

    # For Ollama, we choose the model based on whether it's for ingestion or chat
    if model is None:
        # If no specific model is requested, we default based on common patterns
        # (Though usually the caller should specify)
        model = OLLAMA_INGEST_MODEL
        
    return ChatOllama(model=model, temperature=temperature)
