"""
Dagster pipeline — thin wrapper around ingest.py.
Weekly schedule: Saturday at 22:00 (so new weekend files are ready).

UI:  DAGSTER_HOME=$(mktemp -d) dagster dev -m dagster_pipeline
Run: dagster asset materialize --select sermon_ingestion -m dagster_pipeline
"""

from datetime import datetime
from dagster import (
    asset, Definitions, ScheduleDefinition, AssetSelection,
    define_asset_job, AssetExecutionContext, MetadataValue, in_process_executor,
    Config,
)
from ingest import run_pipeline
from src.scraper.bbtc_scraper import BBTCScraper


class ScraperConfig(Config):
    year: int | None = None
    all_years: bool = False  # set True for initial backfill (2015–present)


@asset
def sermon_scraping(context: AssetExecutionContext, config: ScraperConfig):
    """Scrape the BBTC website for new sermons."""
    scraper = BBTCScraper()
    now = datetime.now()

    if config.year:
        targets = [(config.year, None)]
    elif config.all_years:
        targets = [(y, None) for y in range(2015, now.year + 1)]
    else:
        # Default: current month + previous month so late-posted sermons
        # from the prior month aren't missed on month rollover. If we're in
        # January, the previous month is December of the prior year.
        prev_year = now.year if now.month > 1 else now.year - 1
        prev_month = now.month - 1 if now.month > 1 else 12
        targets = [(prev_year, prev_month), (now.year, now.month)]

    for year, month_filter in targets:
        context.log.info(f"Scraping year {year} (month_filter={month_filter})...")
        scraper.scrape_year(year, month_filter=month_filter)

    context.log.info("Scraping complete.")
    return MetadataValue.text("done")


@asset(deps=[sermon_scraping])
def sermon_ingestion(context: AssetExecutionContext):
    """Weekly incremental ingestion of new BBTC sermons."""
    context.log.info("Starting incremental sermon ingestion...")
    run_pipeline(wipe=False, year=None, incremental=True)
    context.log.info("Ingestion complete.")
    return MetadataValue.text("done")


from src.ingestion.bible.bible_ingest import ingest_bible

@asset
def bible_ingestion(context: AssetExecutionContext):
    """Check for new EPUBs in data/bibles/ and ingest them if unindexed."""
    context.log.info("Checking for new Bible translations...")
    def logger(msg):
        context.log.info(msg)
    ingest_bible(wipe=False, logger=logger)
    context.log.info("Bible ingestion check complete.")
    return MetadataValue.text("done")

ingestion_job = define_asset_job(
    "full_ingestion_job",
    selection=AssetSelection.all(),
    executor_def=in_process_executor,
)

sermon_weekly_schedule = ScheduleDefinition(
    job=ingestion_job,
    #cron_schedule="0 22 * * 6",  # Saturday 22:00
    cron_schedule="53 1 * * 2",  # Tuesday 01:30
)

defs = Definitions(
    assets=[sermon_scraping, sermon_ingestion, bible_ingestion],
    schedules=[sermon_weekly_schedule],
    jobs=[ingestion_job],
    executor=in_process_executor,
)
