import os
from src.scraper.bbtc_scraper import BBTCScraper

scraper = BBTCScraper()
failed_files = [
    "data/staging/English_2026_43-JOHN-4V43-54-WHEN-YOU-NEED-A-MIRACLE-20260214-V2-compressed.pdf",
    "data/staging/English_2026_43-JOHN-9V1-38-WHEN-ALL-YOU-SEE-IS-DARKNESS-20260307-V2.0-compressed.pdf",
    "data/staging/English_2026_21-THE-CALL-TO-KNOW-GODS-WORD-20260117-1_compressed.pdf",
    "data/staging/English_2026_When-Something-is-Missing-8-Feb-compressed.pdf"
]

for f in failed_files:
    print(f"Testing {f}...")
    text, quality = scraper._extract_text_from_file(f)
    print(f"Result: quality={quality}, text_len={len(text)}")
    if quality == "failed":
        print("❌ Extraction failed.")
    else:
        print("✅ Extraction succeeded!")
