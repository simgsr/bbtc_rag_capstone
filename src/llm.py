from langchain_ollama import ChatOllama
import os

GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-2.5-pro"


# def get_llm(provider="ollama", temperature=0, ollama_model="gemma4:latest"):
def get_llm(provider="ollama", temperature=0, ollama_model="macdev/gpt-oss20b-large-ctx"):
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
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature, google_api_key=api_key)

    return ChatOllama(model=ollama_model, temperature=temperature)
