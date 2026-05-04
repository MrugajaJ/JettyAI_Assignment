"""
main.py
-------
Orchestrates the full web-scraping + enrichment + trust-scoring pipeline.

Pipeline per article
--------------------
  1.  Scrape raw data        (scraper/)
  2.  Chunk content          (utils/chunking.py)
  3.  Detect language        (utils/lang_detect.py)
  4.  Generate topic tags    (utils/tagging.py)
  5.  Fetch domain authority (scoring/trust_score.get_domain_authority)
  6.  Calculate trust score  (scoring/trust_score.calculate_trust_score)
  7.  Assemble final schema  (strip internal _ keys)

Outputs
-------
  output/blogs.json        — list of blog records
  output/youtube.json      — list of YouTube records
  output/pubmed.json       — list of PubMed records
  output/scraped_data.json — all records combined
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 output so box-drawing chars work on Windows cp1252 consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Scrapers
# ---------------------------------------------------------------------------
from scraper.blog_scraper    import scrape_blog_posts
from scraper.youtube_scraper import scrape_youtube_videos
from scraper.pubmed_scraper  import scrape_pubmed_articles

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
from utils.chunking    import chunk_content
from utils.lang_detect import detect_language
from utils.tagging     import generate_tags

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
from scoring.trust_score import (
    calculate_trust_score,
    get_domain_authority,
    score_breakdown,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# ── INPUT SOURCES ──────────────────────────────────────────────────────────
# Edit these lists to scrape different articles.
# ---------------------------------------------------------------------------

BLOG_URLS: list[str] = [
    "https://blog.google/technology/ai/bard-google-ai-search-updates/",
    "https://blog.google/technology/ai/google-gemini-ai/",
    "https://research.google/blog/scaling-vision-transformers-to-22-billion-parameters/",
]

YOUTUBE_URLS: list[str] = [
    "https://www.youtube.com/watch?v=aircAruvnKk",   # 3Blue1Brown — Neural networks
    "https://youtu.be/kCc8FmEb1nY",                  # Karpathy — GPT from scratch
]

PUBMED_SOURCES: list[str] = [
    "https://pubmed.ncbi.nlm.nih.gov/38505432/",
]

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Schema keys that are purely internal (populated by scrapers, consumed here)
# ---------------------------------------------------------------------------
_INTERNAL_KEYS = {"_raw_text", "_title", "_description", "_video_id",
                  "_pmid", "_journal"}


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _enrich_record(raw: dict) -> dict:
    """
    Run the full enrichment pipeline on a single raw scraper record.

    Steps
    -----
    1. Chunk the raw text                  → content_chunks
    2. (Re-)detect language from raw text  → language  (overrides scraper value
                                             only when scraper left it None)
    3. Generate topic tags                 → topic_tags
    4. Fetch domain authority for URL      → float (passed to trust scorer)
    5. Calculate trust score               → trust_score
    6. Assemble final public schema dict   (internal _ keys stripped)

    Returns
    -------
    dict conforming to the shared JSON schema.
    """
    url       = raw.get("source_url", "")
    raw_text  = raw.get("_raw_text", "")

    # ── Step 1: chunk ────────────────────────────────────────────────────────
    chunks = chunk_content(raw_text, max_words=150)
    logger.info("  chunks: %d", len(chunks))

    # ── Step 2: language (prefer scraper value; fallback to detection) ────────
    language = raw.get("language") or detect_language(raw_text) or "unknown"

    # ── Step 3: tags ─────────────────────────────────────────────────────────
    tags = generate_tags(raw_text, max_tags=6)
    logger.info("  tags: %s", tags)

    # ── Step 4: domain authority ──────────────────────────────────────────────
    da_score = get_domain_authority(url)
    logger.info("  domain authority: %.2f", da_score)

    # ── Step 5: trust score ───────────────────────────────────────────────────
    # Temporarily inject enriched fields so calculate_trust_score can use them
    enriched_source = {**raw, "language": language}
    trust = calculate_trust_score(enriched_source, domain_authority=da_score)
    logger.info("  trust score: %.2f", trust)

    # ── Step 6: assemble final public record ──────────────────────────────────
    final: dict = {
        "source_url":           url,
        "source_type":          raw.get("source_type"),
        "author":               raw.get("author"),
        "published_date":       raw.get("published_date"),
        "language":             language,
        "region":               raw.get("region"),
        "topic_tags":           tags,
        "trust_score":          trust,
        "content_chunks":       chunks,
    }
    # Pass through YouTube-specific field when present
    if "transcript_available" in raw:
        final["transcript_available"] = raw["transcript_available"]
    return final


def _save_json(data: list[dict], path: Path) -> None:
    """Write a list of dicts to a JSON file with pretty-printing."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, default=str)
    logger.info("Saved %d record(s) → %s", len(data), path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> list[dict]:
    """
    Execute the full pipeline for all configured sources.

    Returns
    -------
    list[dict]
        All enriched records (blogs + YouTube + PubMed).
    """
    all_records: list[dict] = []

    # ════════════════════════════════════════════════════════════════════════
    # BLOGS
    # ════════════════════════════════════════════════════════════════════════
    logger.info("━━━━━━━━━━━━━━  BLOGS  ━━━━━━━━━━━━━━")
    raw_blogs = scrape_blog_posts(BLOG_URLS)

    blog_records: list[dict] = []
    for raw in raw_blogs:
        logger.info("Enriching: %s", raw.get("source_url", "?"))
        blog_records.append(_enrich_record(raw))

    _save_json(blog_records, OUTPUT_DIR / "blogs.json")
    all_records.extend(blog_records)

    # ════════════════════════════════════════════════════════════════════════
    # YOUTUBE
    # ════════════════════════════════════════════════════════════════════════
    logger.info("━━━━━━━━━━━━━━  YOUTUBE  ━━━━━━━━━━━━━━")
    raw_videos = scrape_youtube_videos(YOUTUBE_URLS)

    youtube_records: list[dict] = []
    for raw in raw_videos:
        logger.info("Enriching: %s", raw.get("source_url", "?"))
        youtube_records.append(_enrich_record(raw))

    _save_json(youtube_records, OUTPUT_DIR / "youtube.json")
    all_records.extend(youtube_records)

    # ════════════════════════════════════════════════════════════════════════
    # PUBMED
    # ════════════════════════════════════════════════════════════════════════
    logger.info("━━━━━━━━━━━━━━  PUBMED  ━━━━━━━━━━━━━━")
    raw_pubmed = scrape_pubmed_articles(PUBMED_SOURCES)

    pubmed_records: list[dict] = []
    for raw in raw_pubmed:
        logger.info("Enriching: %s", raw.get("source_url", "?"))
        pubmed_records.append(_enrich_record(raw))

    _save_json(pubmed_records, OUTPUT_DIR / "pubmed.json")
    all_records.extend(pubmed_records)

    # ════════════════════════════════════════════════════════════════════════
    # COMBINED
    # ════════════════════════════════════════════════════════════════════════
    _save_json(all_records, OUTPUT_DIR / "scraped_data.json")

    return all_records


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(records: list[dict]) -> None:
    """Print a formatted console summary table of all processed records."""
    col_url   = 58
    col_type  = 9
    col_lang  = 6
    col_tags  = 30
    col_score = 7
    total_w   = col_url + col_type + col_lang + col_tags + col_score + 12

    divider = "─" * total_w

    header = (
        f"{'Source URL':<{col_url}}  "
        f"{'Type':<{col_type}}  "
        f"{'Lang':<{col_lang}}  "
        f"{'Top Tags':<{col_tags}}  "
        f"{'Score':>{col_score}}"
    )

    print(f"\n{'═' * total_w}")
    print(f"  PIPELINE SUMMARY  —  {len(records)} article(s) processed")
    print(f"{'═' * total_w}")
    print(header)
    print(divider)

    for rec in records:
        url        = str(rec.get("source_url", ""))
        src_type   = str(rec.get("source_type", ""))
        lang       = str(rec.get("language") or "?")
        tags       = ", ".join(rec.get("topic_tags") or [])
        trust      = rec.get("trust_score")
        trust_str  = f"{trust:.2f}" if isinstance(trust, float) else "N/A"

        # Truncate long values to fit the table
        url_disp  = url[:col_url]  if len(url)  > col_url  else url
        tags_disp = tags[:col_tags] if len(tags) > col_tags else tags

        print(
            f"{url_disp:<{col_url}}  "
            f"{src_type:<{col_type}}  "
            f"{lang:<{col_lang}}  "
            f"{tags_disp:<{col_tags}}  "
            f"{trust_str:>{col_score}}"
        )

    print(divider)

    # Aggregate stats
    scores = [r["trust_score"] for r in records if isinstance(r.get("trust_score"), float)]
    if scores:
        avg   = sum(scores) / len(scores)
        high  = max(scores)
        low   = min(scores)
        print(f"  avg trust={avg:.2f}  |  high={high:.2f}  |  low={low:.2f}")

    print(f"{'═' * total_w}\n")
    print(f"  Output files written to: {OUTPUT_DIR.resolve()}")
    print(f"    • blogs.json        ({len([r for r in records if r.get('source_type')=='blog'])} records)")
    print(f"    • youtube.json      ({len([r for r in records if r.get('source_type')=='youtube'])} records)")
    print(f"    • pubmed.json       ({len([r for r in records if r.get('source_type')=='pubmed'])} records)")
    print(f"    • scraped_data.json ({len(records)} records combined)")
    print(f"{'═' * total_w}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Pipeline started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    try:
        results = run_pipeline()
        _print_summary(results)
        logger.info("Pipeline complete. %d record(s) written.", len(results))
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unhandled pipeline error: %s", exc)
        sys.exit(1)
