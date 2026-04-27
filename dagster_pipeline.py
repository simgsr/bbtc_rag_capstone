import os
from datetime import datetime
from dagster import asset, Definitions, ScheduleDefinition, AssetSelection, define_asset_job, AssetExecutionContext, MetadataValue
from src.scraper.bbtc_scraper import BBTCScraper
from src.storage.sqlite_store import SermonRegistry
from src.storage.chroma_store import SermonVectorStore
from src.ingestion.metadata_extractor import MetadataExtractor
from src.ingestion.bible.bible_ingest import ingest_bible

# Initialize components
registry = SermonRegistry()
vector_store = SermonVectorStore()
extractor = MetadataExtractor()
scraper = BBTCScraper(registry=registry)

@asset
def bible_ingestion(context: AssetExecutionContext):
    """
    Asset that checks data/bibles/ for new EPUB files and indexes them.
    """
    bible_dir = "data/bibles"
    if not os.path.exists(bible_dir):
        os.makedirs(bible_dir)
        context.log.info(f"📁 Created {bible_dir} directory.")
        return {"indexed_count": 0}

    # Files to version mapping (could be improved to auto-detect version from filename)
    available_files = {
        "NIV.epub": "NIV",
        "ESV The Holy Bible.epub": "ESV",
        "Bible - American Standard Version.epub": "ASV"
    }
    
    # Check what's already indexed in SQLite
    indexed_versions = {b['version_id'] for b in registry.get_indexed_bibles()}
    context.log.info(f"🔍 Found {len(indexed_versions)} already indexed Bible versions.")

    new_versions = []
    for filename, version_id in available_files.items():
        if version_id not in indexed_versions:
            if os.path.exists(os.path.join(bible_dir, filename)):
                new_versions.append((version_id, filename))

    if not new_versions:
        context.log.info("✨ No new Bible versions found to ingest.")
        return {"indexed_count": 0}

    context.log.info(f"🆕 Found {len(new_versions)} new Bible versions to ingest: {[v[0] for v in new_versions]}")
    
    count = 0
    for version_id, filename in new_versions:
        success = ingest_bible(version_id, filename, logger=context.log.info)
        if success:
            registry.mark_bible_indexed(version_id, filename)
            count += 1
            context.log.info(f"✅ Finished ingesting {version_id}")

    context.add_output_metadata(
        metadata={
            "newly_indexed": MetadataValue.int(count),
            "total_versions": MetadataValue.int(len(registry.get_indexed_bibles()))
        }
    )

    return {"indexed_count": count}

@asset
def sermon_ingestion_summary(context: AssetExecutionContext):
    """
    Weekly asset that scrapes new sermons for the current year,
    extracts metadata, and updates the vector store.
    """
    current_year = datetime.now().year
    years_to_scrape = range(2015, current_year + 1)
    
    context.log.info(f"🚀 Starting ingestion for years: {list(years_to_scrape)}")
    
    for year in years_to_scrape:
        context.log.info(f"📅 Scrapping year {year}...")
        # 1. Scrape and download (this updates SQLite and saves .txt files)
        scraper.scrape_year(year)
    
    # 2. Group files and process as sermon units
    from src.ingestion.filename_parser import parse_cell_guide_filename
    from src.ingestion.sermon_grouper import group_sermon_files
    from src.ingestion.speaker_from_filename import speaker_from_filename
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    sermons = registry.get_all_sermons()
    pending = [s for s in sermons if s['status'] in ('extracted', 'processed')]
    context.log.info(f"📋 {len(pending)} sermon files pending indexing.")

    # Index pending files by filename for fast lookup
    by_filename = {s['filename']: s for s in pending}
    pending_filenames = list(by_filename.keys())

    sermon_groups = group_sermon_files(pending_filenames)
    context.log.info(f"📦 {len(sermon_groups)} sermon groups formed.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    processed_count = 0

    def _load_text(sermon: dict) -> str | None:
        txt_name = os.path.splitext(sermon['filename'])[0] + ".txt"
        txt_path = os.path.join("data/sermons", txt_name)
        if not os.path.exists(txt_path):
            context.log.warning(f"⚠️ Text file not found: {txt_name}")
            return None
        with open(txt_path, encoding="utf-8") as f:
            return f.read()

    def _vectorise(sermon_id: str, text: str, meta: dict):
        chunks = splitter.split_text(text) or [text[:500]]
        ids = [f"{sermon_id}_{i}" for i in range(len(chunks))]
        vector_store.upsert_sermon_chunks(chunks, [meta] * len(chunks), ids)

    for group in sermon_groups:
        # ── Cell-guide-led group ──────────────────────────────────────────────
        if group.cell_guide:
            cg_sermon = by_filename[group.cell_guide]
            sermon_id = cg_sermon['sermon_id']

            # Metadata from filename (reliable — no LLM needed)
            parsed = parse_cell_guide_filename(group.cell_guide)
            speaker = parsed.get("speaker") or speaker_from_filename(group.cell_guide)
            date    = parsed.get("date") or cg_sermon.get("date")
            topic   = parsed.get("topic")

            context.log.info(f"📖 [{sermon_id}] {topic or group.cell_guide} — {speaker}")

            update_record = {
                "sermon_id":     sermon_id,
                "filename":      cg_sermon['filename'],
                "url":           cg_sermon['url'],
                "speaker":       speaker,
                "date":          date,
                "topic":         topic,
                "series":        cg_sermon.get("series"),
                "bible_book":    cg_sermon.get("bible_book"),
                "primary_verse": cg_sermon.get("primary_verse"),
                "status":        "indexed",
            }
            registry.insert_sermon(update_record)

            # Vectorise cell guide content
            cg_text = _load_text(cg_sermon)
            if cg_text:
                _vectorise(sermon_id, cg_text, update_record)

            # Vectorise paired slide content under the same sermon_id
            for slide_filename in group.slides:
                slide_sermon = by_filename.get(slide_filename)
                if not slide_sermon:
                    continue
                context.log.info(f"  📎 Adding slides: {slide_filename}")
                slide_text = _load_text(slide_sermon)
                if slide_text:
                    _vectorise(sermon_id, slide_text, update_record)
                # Mark slide as indexed (absorbed into the cell guide group)
                registry.mark_processed(slide_sermon['url'], status="indexed")

            # LLM intelligence (summary + verses) from combined text
            combined = "\n\n".join(filter(None, [cg_text, *(
                _load_text(by_filename[s]) for s in group.slides if s in by_filename
            )]))
            if combined:
                intel = extractor.extract(combined[:2000])
                registry.insert_intelligence({
                    "sermon_id":     sermon_id,
                    "speaker":       speaker,
                    "primary_verse": intel.get("primary_verse"),
                    "verses_used":   intel.get("verses_used"),
                    "summary":       intel.get("summary"),
                })

            processed_count += 1

        # ── Standalone slides (no matching cell guide) ────────────────────────
        else:
            for slide_filename in group.slides:
                slide_sermon = by_filename.get(slide_filename)
                if not slide_sermon:
                    continue
                sermon_id = slide_sermon['sermon_id']
                slide_text = _load_text(slide_sermon)
                if not slide_text:
                    continue

                context.log.info(f"📄 Standalone: {slide_filename}")
                meta = extractor.extract(slide_text[:2000])
                speaker = (
                    speaker_from_filename(slide_filename)
                    or meta.get("speaker")
                )
                update_record = {
                    "sermon_id":     sermon_id,
                    "filename":      slide_sermon['filename'],
                    "url":           slide_sermon['url'],
                    "speaker":       speaker,
                    "date":          meta.get("date") or slide_sermon.get("date"),
                    "series":        meta.get("series"),
                    "bible_book":    meta.get("bible_book"),
                    "primary_verse": meta.get("primary_verse"),
                    "status":        "indexed",
                }
                registry.insert_sermon(update_record)
                _vectorise(sermon_id, slide_text, update_record)
                registry.insert_intelligence({
                    "sermon_id":     sermon_id,
                    "speaker":       speaker,
                    "primary_verse": meta.get("primary_verse"),
                    "verses_used":   meta.get("verses_used"),
                    "summary":       meta.get("summary"),
                })
                processed_count += 1
    
    # Add final metadata for the UI
    context.add_output_metadata(
        metadata={
            "year": current_year,
            "newly_indexed": MetadataValue.int(processed_count),
            "total_in_db": MetadataValue.int(len(sermons)),
            "last_run": MetadataValue.text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        }
    )
    
    return {
        "year": current_year,
        "newly_indexed": processed_count,
        "total_in_db": len(sermons)
    }

# Job definition
ingestion_job = define_asset_job("sermon_ingestion_job", selection=AssetSelection.all())

# Weekly schedule (Sunday at midnight)
sermon_weekly_schedule = ScheduleDefinition(
    job=ingestion_job,
    cron_schedule="0 0 * * 0", 
)

defs = Definitions(
    assets=[bible_ingestion, sermon_ingestion_summary],
    schedules=[sermon_weekly_schedule],
    jobs=[ingestion_job],
)
