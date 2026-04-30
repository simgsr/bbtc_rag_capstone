.PHONY: install scrape ingest run setup test clean

# Virtual environment settings
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
YEAR ?= $(shell date +%Y)

# Setup entire project (one-click install)
setup: install scrape ingest
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

# Scrape the sermons for the current year (or specify YEAR=2024 make scrape)
scrape:
	@echo "📥 Scraping sermons for year $(YEAR)..."
	$(PYTHON) src/scraper/bbtc_scraper.py $(YEAR)

# Ingest sermons into SQLite and ChromaDB
ingest:
	@echo "🧠 Ingesting sermons into database..."
	$(PYTHON) ingest.py --wipe

# Run the Gradio Chat UI
run:
	@echo "🚀 Starting Gradio UI..."
	$(PYTHON) app.py

# Run Dagster for weekly scheduling
dagster:
	@echo "⏱️ Starting Dagster scheduler..."
	DAGSTER_HOME=$$(mktemp -d) $(VENV_DIR)/bin/dagster dev -m dagster_pipeline

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
