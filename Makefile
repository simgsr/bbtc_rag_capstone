.PHONY: install scrape ingest run setup dagster test clean

# Virtual environment settings
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
YEAR ?= $(shell date +%Y)

# Setup entire project (one-click install) — scrapes all years 2015–present then ingests
setup: install
	@echo "📥 Scraping all sermon years (2015–present)..."
	$(PYTHON) src/scraper/bbtc_scraper.py --all
	@echo "🧠 Running full ingestion (wipe + rebuild)..."
	$(PYTHON) ingest.py --wipe
	@echo "✅ Setup complete! You can now run the app with 'make run'"

# Install dependencies and setup environment
install:
	@echo "📦 Setting up virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "📦 Installing dependencies..."
	$(PIP) install -r requirements.txt
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env from .env.example..."; \
		cp .env.example .env; \
	fi
	@echo "✅ Install complete."

# Scrape a single year (default: current year). Use YEAR=2024 to override.
scrape:
	@echo "📥 Scraping sermons for year $(YEAR)..."
	$(PYTHON) src/scraper/bbtc_scraper.py $(YEAR)

# Ingest sermons into SQLite and ChromaDB
ingest:
	@echo "🧠 Ingesting sermons..."
	$(PYTHON) ingest.py

# Run the Gradio Chat UI
run:
	@echo "🚀 Starting Gradio UI..."
	$(PYTHON) app.py

# Run Dagster for weekly scheduling
dagster:
	@echo "⏱️ Starting Dagster scheduler..."
	DAGSTER_HOME=$(PWD)/.dagster $(VENV_DIR)/bin/dagster dev -m dagster_pipeline

# Run tests
test:
	@echo "🧪 Running tests..."
	$(PYTHON) -m pytest tests/ -v

# Clean up environment and data
clean:
	@echo "🧹 Cleaning up data and environment..."
	rm -rf $(VENV_DIR)
	rm -rf data/chroma_db
	rm -rf data/sermons.db
	rm -rf data/staging
	@echo "✅ Cleanup complete."
