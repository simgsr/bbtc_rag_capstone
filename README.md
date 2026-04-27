# Sermon Intelligence RAG Pipeline

Sermon Intelligence is a hybrid Agentic RAG (Retrieval-Augmented Generation) pipeline for the Bethesda Bedok-Tampines Church (BBTC) sermon archive. It combines SQL-based metadata querying with vector-based semantic search to provide comprehensive insights into years of sermon history.

## Features
- **Agentic Search**: Powered by LangGraph and Groq/Gemini, the assistant intelligently routes queries between SQL and Vector stores.
- **Hybrid Storage**: Uses SQLite for structured metadata (speakers, dates, series) and ChromaDB for semantic content search.
- **Visualization Tool**: Built-in capability to generate charts (sermons per year, top speakers, etc.) using Matplotlib.
- **Automated Ingestion**: Integrated Dagster pipeline for weekly scraping and indexing of new sermons.

## Setup

### Local Development
1. **Clone the repository.**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure environment**:
   Copy `.env.example` to `.env` and fill in your `GROQ_API_KEY` and `GEMINI_API_KEY`.
4. **Run the app**:
   ```bash
   python app.py
   ```

### Data Pipeline
To run the full ingestion pipeline (scrape + extract + vectorize):
```bash
dagster asset materialize --select sermon_ingestion_summary -m dagster_pipeline
```

## Deployment
The included `Dockerfile` is configured to run the Gradio interface on port `7860`, making it ready for deployment on Hugging Face Spaces or Render.

### Persistent Storage
Ensure the `data/` directory is mounted to a persistent volume to preserve the SQLite database and ChromaDB vector store across restarts.
