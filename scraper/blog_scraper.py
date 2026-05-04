"""
blog_scraper.py
---------------
Scrapes a list of blog post URLs and extracts structured article data.

Primary extraction: newspaper3k
Fallback extraction: BeautifulSoup4 (targets <article>, <main>, or largest <div>)

Returns a list of dicts matching the shared content schema.
"""

import logging
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString

# newspaper3k import guard — it may not be installed in all envs
try:
    from newspaper import Article, ArticleException
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    logging.warning("newspaper3k is not installed; falling back to BeautifulSoup4 for all URLs.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tags whose content we always strip before extracting body text
_NOISE_TAGS = {"nav", "footer", "header", "aside", "script", "style", "noscript",
               "form", "button", "iframe", "ins", "figure"}

# Common ad/nav wrapper class/id substrings to remove
_NOISE_KEYWORDS = {"nav", "menu", "sidebar", "footer", "header", "advertisement",
                   "banner", "cookie", "popup", "modal", "breadcrumb", "related",
                   "subscribe", "newsletter", "social", "share", "comment", "ads",
                   "promo", "widget", "toc"}

REQUEST_TIMEOUT = 15  # seconds
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Cache-Control": "max-age=0",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_empty_record(url: str) -> dict:
    """Return the schema skeleton for a single article."""
    return {
        "source_url": url,
        "source_type": "blog",
        "author": None,
        "published_date": None,
        "language": None,
        "region": None,
        "topic_tags": [],        # filled later
        "trust_score": None,     # filled later
        "content_chunks": [],    # filled later
        # Internal full text — consumed by downstream chunker, not part of final schema
        "_raw_text": "",
        "_title": "",
    }


def _fetch_html(url: str) -> Optional[str]:
    """Download raw HTML with httpx; returns None on failure."""
    try:
        resp = httpx.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT,
                         follow_redirects=True)
        resp.raise_for_status()
        return resp.text
    except httpx.HTTPError as exc:
        logger.error("HTTP error fetching %s: %s", url, exc)
        return None


def _is_noise_element(tag) -> bool:
    """Return True if a BeautifulSoup tag should be treated as noise."""
    if tag.name in _NOISE_TAGS:
        return True
    # Some BS4 tags (e.g. pseudo-elements) can have attrs=None
    if not tag.attrs:
        return False
    # Check class and id attributes for known noise keywords
    for attr in ("class", "id"):
        values = tag.get(attr, [])
        if isinstance(values, str):
            values = [values]
        for val in values:
            if any(kw in val.lower() for kw in _NOISE_KEYWORDS):
                return True
    return False


def _strip_noise(soup: BeautifulSoup) -> None:
    """Destructively remove all noise elements from a soup tree."""
    for tag in soup.find_all(True):
        if _is_noise_element(tag):
            tag.decompose()


def _get_text(tag) -> str:
    """Extract and clean text from a BS4 tag."""
    return " ".join(tag.get_text(separator=" ").split())


def _bs4_extract(html: str) -> dict:
    """
    BeautifulSoup4 fallback extractor.

    Priority:
      1. <article> tag
      2. <main> tag
      3. Largest <div> block by text length
    """
    soup = BeautifulSoup(html, "html.parser")

    # ---- author (best-effort meta tags) -----------------------------------
    author = None
    for meta in soup.find_all("meta"):
        prop = meta.get("property", "") or meta.get("name", "")
        if "author" in prop.lower():
            author = meta.get("content")
            break
    if not author:
        # JSON-LD schema.org author
        import json, re
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
                if isinstance(data, dict):
                    a = data.get("author")
                    if isinstance(a, dict):
                        author = a.get("name")
                    elif isinstance(a, str):
                        author = a
                    if author:
                        break
            except (json.JSONDecodeError, TypeError):
                pass

    # ---- published date ---------------------------------------------------
    published_date = None
    for meta in soup.find_all("meta"):
        prop = (meta.get("property", "") or meta.get("name", "") or
                meta.get("itemprop", "")).lower()
        if any(k in prop for k in ("published_time", "date", "pubdate")):
            raw = meta.get("content", "")
            if raw:
                published_date = raw[:10]  # keep YYYY-MM-DD portion
                break
    if not published_date:
        time_tag = soup.find("time")
        if time_tag:
            published_date = (time_tag.get("datetime") or time_tag.get_text()).strip()
            if published_date:
                published_date = published_date[:10]

    # ---- title ------------------------------------------------------------
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title:
        title = og_title.get("content", "")
    if not title and soup.title:
        title = soup.title.get_text().strip()

    # ---- strip noise before body extraction --------------------------------
    _strip_noise(soup)

    # ---- body text --------------------------------------------------------
    body_text = ""
    article_tag = soup.find("article")
    if article_tag:
        body_text = _get_text(article_tag)
    else:
        main_tag = soup.find("main")
        if main_tag:
            body_text = _get_text(main_tag)
        else:
            # Largest <div> heuristic
            best_div = max(
                soup.find_all("div"),
                key=lambda d: len(d.get_text()),
                default=None,
            )
            if best_div:
                body_text = _get_text(best_div)

    return {
        "title": title or None,
        "author": author or None,
        "published_date": published_date or None,
        "text": body_text,
    }


def _newspaper_extract(url: str) -> Optional[dict]:
    """
    Primary extractor using newspaper3k.
    Returns a dict with keys: title, author, published_date, text.
    Returns None if extraction fails.
    """
    if not NEWSPAPER_AVAILABLE:
        return None
    try:
        article = Article(url)
        article.download()
        article.parse()

        authors = article.authors
        author = ", ".join(authors) if authors else None

        published_date = None
        if article.publish_date:
            if isinstance(article.publish_date, datetime):
                published_date = article.publish_date.strftime("%Y-%m-%d")
            else:
                published_date = str(article.publish_date)[:10]

        return {
            "title": article.title or None,
            "author": author,
            "published_date": published_date,
            "text": article.text or "",
        }
    except Exception as exc:  # ArticleException or network errors
        logger.warning("newspaper3k failed for %s: %s", url, exc)
        return None


def _detect_language(text: str) -> Optional[str]:
    """Detect language using langdetect; returns ISO 639-1 code or None."""
    try:
        from langdetect import detect, LangDetectException
        if text and len(text.strip()) > 20:
            return detect(text)
    except Exception:
        pass
    return None


def _infer_region(url: str) -> Optional[str]:
    """
    Very lightweight region hint from the URL's TLD.
    Returns a country/region string or None.
    """
    TLD_MAP = {
        ".uk": "GB", ".co.uk": "GB", ".in": "IN", ".co.in": "IN",
        ".au": "AU", ".ca": "CA", ".de": "DE", ".fr": "FR",
        ".jp": "JP", ".cn": "CN", ".br": "BR", ".mx": "MX",
    }
    try:
        hostname = urlparse(url).hostname or ""
        for tld, region in TLD_MAP.items():
            if hostname.endswith(tld):
                return region
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_blog_posts(urls: list[str]) -> list[dict]:
    """
    Scrape a list of blog post URLs and return structured article records.

    Parameters
    ----------
    urls : list[str]
        Blog post URLs to scrape (designed for 3, works for any count).

    Returns
    -------
    list[dict]
        One dict per URL conforming to the shared content schema.
        topic_tags, trust_score, and content_chunks are intentionally empty.
    """
    results = []

    for url in urls:
        logger.info("Scraping: %s", url)
        record = _build_empty_record(url)

        # --- Step 1: try newspaper3k ----------------------------------------
        extracted = _newspaper_extract(url)

        # --- Step 2: fall back to BeautifulSoup4 if needed ------------------
        if not extracted or not extracted.get("text"):
            logger.info("  → Falling back to BeautifulSoup4 for %s", url)
            html = _fetch_html(url)
            if html:
                extracted = _bs4_extract(html)
            else:
                logger.error("  → Could not fetch HTML for %s; skipping.", url)
                results.append(record)
                continue

        # --- Populate record -------------------------------------------------
        record["author"] = extracted.get("author")          # None if missing
        record["published_date"] = extracted.get("published_date")  # None if missing
        record["_title"] = extracted.get("title") or ""
        record["_raw_text"] = extracted.get("text") or ""

        # Language detection on the extracted body text
        record["language"] = _detect_language(record["_raw_text"])

        # Lightweight region inference from URL
        record["region"] = _infer_region(url)

        results.append(record)
        logger.info(
            "  ✓ Done | author=%s | date=%s | lang=%s | chars=%d",
            record["author"],
            record["published_date"],
            record["language"],
            len(record["_raw_text"]),
        )

    return results


# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_URLS = [
        "https://towardsdatascience.com/a-friendly-introduction-to-graph-neural-networks-98949571b2f5",
        "https://openai.com/research/gpt-4",
        "https://www.bbc.com/future/article/20240101-the-science-of-sleep",
    ]

    articles = scrape_blog_posts(TEST_URLS)
    for art in articles:
        print("\n" + "=" * 60)
        print(f"URL    : {art['source_url']}")
        print(f"Title  : {art['_title']}")
        print(f"Author : {art['author']}")
        print(f"Date   : {art['published_date']}")
        print(f"Lang   : {art['language']}")
        print(f"Region : {art['region']}")
        print(f"Chars  : {len(art['_raw_text'])}")
        print(f"Schema keys: {[k for k in art if not k.startswith('_')]}")
