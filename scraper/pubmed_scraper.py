"""
pubmed_scraper.py
-----------------
Scrapes a PubMed article URL or raw PMID and extracts structured data.

API used : NCBI E-utilities (no API key required)
  - esearch.fcgi  — resolve a PubMed URL / article title query → PMID
  - efetch.fcgi   — fetch full article XML (abstract, authors, journal …)

XML parsed with : Python stdlib  xml.etree.ElementTree  (no extra deps)

Returns a dict matching the shared content schema (same as blog / youtube scrapers).
source_type = 'pubmed'
"""

import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Optional
from urllib.parse import urlparse, parse_qs

import httpx

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_ESEARCH_URL = f"{_EUTILS_BASE}/esearch.fcgi"
_EFETCH_URL  = f"{_EUTILS_BASE}/efetch.fcgi"

# NCBI asks for a short delay between requests to be a good citizen
_NCBI_POLITE_DELAY = 0.4   # seconds

REQUEST_TIMEOUT = 20        # seconds

# E-utilities prefers a valid tool name + email in the query string
_NCBI_TOOL  = "pubmed_scraper"
_NCBI_EMAIL = "research@example.com"   # replace with a real address if you have one

_COMMON_PARAMS = {
    "tool":  _NCBI_TOOL,
    "email": _NCBI_EMAIL,
}

DEFAULT_HEADERS = {
    "User-Agent": f"PubMedScraper/1.0 (tool={_NCBI_TOOL}; mailto={_NCBI_EMAIL})"
}

# Regex that matches a bare PMID (all digits, 1-8 chars)
_PMID_RE = re.compile(r"^\d{1,8}$")


# ---------------------------------------------------------------------------
# Helpers — Input normalisation
# ---------------------------------------------------------------------------

def _parse_input(raw: str) -> Optional[str]:
    """
    Accept any of:
      - A bare PMID string              e.g.  "38505432"
      - A PubMed article URL            e.g.  "https://pubmed.ncbi.nlm.nih.gov/38505432/"
      - A PubMed Central URL            e.g.  "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1234/"
      - A DOI string                    e.g.  "10.1001/jama.2024.1234"
      - A DOI URL                       e.g.  "https://doi.org/10.1001/jama.2024.1234"

    Returns the PMID string directly if it can be extracted from the URL,
    otherwise returns the best search term to pass to esearch.
    """
    raw = raw.strip()

    # 1 — already a bare PMID
    if _PMID_RE.match(raw):
        logger.info("Input recognised as bare PMID: %s", raw)
        return raw

    parsed = urlparse(raw)
    hostname = (parsed.hostname or "").lower()
    path     = parsed.path.rstrip("/")

    # 2 — pubmed.ncbi.nlm.nih.gov/PMID  or  /pubmed/PMID
    if "pubmed.ncbi.nlm.nih.gov" in hostname:
        parts = [p for p in path.split("/") if p]
        if parts and _PMID_RE.match(parts[-1]):
            logger.info("Extracted PMID %s from PubMed URL", parts[-1])
            return parts[-1]

    if "ncbi.nlm.nih.gov" in hostname and "/pubmed/" in path:
        parts = [p for p in path.split("/") if p]
        candidate = parts[-1]
        if _PMID_RE.match(candidate):
            logger.info("Extracted PMID %s from NCBI URL", candidate)
            return candidate

    # 3 — DOI (URL or bare string)
    doi_match = re.search(r"(10\.\d{4,9}/[^\s\"'<>]+)", raw)
    if doi_match:
        doi = doi_match.group(1)
        logger.info("Detected DOI: %s — will query esearch", doi)
        return doi   # returned as search term for esearch

    # 4 — anything else: treat as raw search term
    logger.info("Could not extract PMID directly; using as esearch term: %s", raw)
    return raw


# ---------------------------------------------------------------------------
# Helpers — NCBI E-utilities HTTP calls
# ---------------------------------------------------------------------------

def _ncbi_get(url: str, params: dict) -> Optional[str]:
    """
    Perform a GET request against an NCBI E-utilities endpoint.
    Returns the response text, or None on failure.
    Applies the polite delay before every request.
    """
    time.sleep(_NCBI_POLITE_DELAY)
    try:
        resp = httpx.get(
            url,
            params={**_COMMON_PARAMS, **params},
            headers=DEFAULT_HEADERS,
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text
    except httpx.HTTPStatusError as exc:
        logger.error("HTTP %s from NCBI: %s — URL: %s", exc.response.status_code, exc, url)
    except httpx.HTTPError as exc:
        logger.error("Network error contacting NCBI: %s", exc)
    return None


def _resolve_to_pmid(search_term: str) -> Optional[str]:
    """
    Use esearch to turn a DOI, title query, or other term into a PMID.
    Returns the first PMID hit, or None if no results.
    """
    logger.info("  → Running esearch for: %s", search_term)
    xml_text = _ncbi_get(_ESEARCH_URL, {
        "db":     "pubmed",
        "term":   search_term,
        "retmax": "1",
        "retmode": "xml",
    })
    if not xml_text:
        return None
    try:
        root = ET.fromstring(xml_text)
        id_el = root.find(".//Id")
        if id_el is not None and id_el.text:
            pmid = id_el.text.strip()
            logger.info("  → esearch resolved to PMID: %s", pmid)
            return pmid
        logger.warning("  ⚠ esearch returned no results for: %s", search_term)
    except ET.ParseError as exc:
        logger.error("Failed to parse esearch XML: %s", exc)
    return None


def _fetch_pubmed_xml(pmid: str) -> Optional[str]:
    """
    Call efetch to retrieve PubMed article XML for the given PMID.
    """
    logger.info("  → Fetching efetch XML for PMID: %s", pmid)
    return _ncbi_get(_EFETCH_URL, {
        "db":      "pubmed",
        "id":      pmid,
        "rettype": "abstract",
        "retmode": "xml",
    })


# ---------------------------------------------------------------------------
# Helpers — XML parsing
# ---------------------------------------------------------------------------

def _text(element: Optional[ET.Element], default: str = "") -> str:
    """Safely extract .text from an ElementTree element."""
    if element is not None and element.text:
        return element.text.strip()
    return default


def _iter_text(element: ET.Element) -> str:
    """
    Recursively collect all text content under an element (including tails).
    Useful for abstract text which may contain mixed-content XML tags like
    <b>, <i>, <sup> inserted by PubMed.
    """
    parts = []
    for node in element.iter():
        if node.text:
            parts.append(node.text.strip())
        if node.tail:
            parts.append(node.tail.strip())
    return " ".join(p for p in parts if p)


def _parse_article_xml(xml_text: str) -> Optional[dict]:
    """
    Parse the PubMed efetch XML response and return a flat dict:
        title, authors (comma-joined str), journal, year, abstract
    Returns None if the XML cannot be parsed or is empty.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error("Failed to parse efetch XML: %s", exc)
        return None

    article = root.find(".//PubmedArticle")
    if article is None:
        logger.error("No <PubmedArticle> element found in efetch response.")
        return None

    medline = article.find("MedlineCitation")
    if medline is None:
        logger.error("No <MedlineCitation> in article XML.")
        return None

    art_node = medline.find("Article")
    if art_node is None:
        logger.error("No <Article> in MedlineCitation.")
        return None

    # ---- Title ---------------------------------------------------------------
    title_el = art_node.find("ArticleTitle")
    title = _iter_text(title_el) if title_el is not None else None

    # ---- Authors -------------------------------------------------------------
    author_list_el = art_node.find("AuthorList")
    authors = []
    if author_list_el is not None:
        for author_el in author_list_el.findall("Author"):
            last  = _text(author_el.find("LastName"))
            first = _text(author_el.find("ForeName")) or _text(author_el.find("Initials"))
            # CollectiveName for group authors (e.g. "COVID-19 Task Force")
            collective = _text(author_el.find("CollectiveName"))
            if collective:
                authors.append(collective)
            elif last:
                authors.append(f"{last} {first}".strip())

    author_str = ", ".join(authors) if authors else None

    # ---- Journal -------------------------------------------------------------
    journal_el = art_node.find("Journal")
    journal = None
    if journal_el is not None:
        # Prefer the full journal title; fall back to ISO abbreviation
        journal = (
            _text(journal_el.find("Title"))
            or _text(journal_el.find("ISOAbbreviation"))
        )

    # ---- Publication year ----------------------------------------------------
    pub_year = None
    # Try Journal > JournalIssue > PubDate > Year first
    pub_date_el = None
    if journal_el is not None:
        pub_date_el = journal_el.find("JournalIssue/PubDate")
    if pub_date_el is not None:
        year_str = _text(pub_date_el.find("Year"))
        if year_str:
            pub_year = year_str
        else:
            # Some records use MedlineDate: "2024 Jan-Feb"
            medline_date = _text(pub_date_el.find("MedlineDate"))
            year_match = re.search(r"\b(19|20)\d{2}\b", medline_date)
            if year_match:
                pub_year = year_match.group(0)

    # ---- Abstract ------------------------------------------------------------
    abstract_el = art_node.find("Abstract")
    abstract_text = ""
    if abstract_el is not None:
        # Abstract may have multiple <AbstractText Label="BACKGROUND"> sections
        parts = []
        for abs_text_el in abstract_el.findall("AbstractText"):
            label = abs_text_el.get("Label", "")
            content = _iter_text(abs_text_el)
            if label:
                parts.append(f"{label}: {content}")
            else:
                parts.append(content)
        abstract_text = " ".join(parts)

    # Return None only if we got absolutely nothing useful
    if not title and not abstract_text:
        logger.warning("Parsed XML but found neither title nor abstract.")
        return None

    return {
        "title":    title,
        "author":   author_str,    # None if no authors found
        "journal":  journal,
        "year":     pub_year,      # string "YYYY" or None
        "abstract": abstract_text,
    }


# ---------------------------------------------------------------------------
# Helpers — Language detection
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> Optional[str]:
    """Detect ISO 639-1 language code; returns None on failure."""
    try:
        from langdetect import detect
        if text and len(text.strip()) > 30:
            return detect(text)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------

def _build_empty_record(source_url: str) -> dict:
    """Return the schema skeleton for a single PubMed article."""
    return {
        "source_url":     source_url,
        "source_type":    "pubmed",
        "author":         None,
        "published_date": None,   # stored as "YYYY" for PubMed
        "language":       None,
        "region":         None,   # PubMed is global; left None
        "topic_tags":     [],     # filled later
        "trust_score":    None,   # filled later
        "content_chunks": [],     # filled later
        # Internal fields
        "_pmid":     None,
        "_title":    "",
        "_journal":  "",
        "_raw_text": "",          # abstract text
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_pubmed_article(source: str) -> dict:
    """
    Scrape a single PubMed article from a URL, PMID, or DOI.

    Parameters
    ----------
    source : str
        PubMed URL (e.g. https://pubmed.ncbi.nlm.nih.gov/38505432/),
        bare PMID  (e.g. "38505432"),
        or DOI     (e.g. "10.1056/NEJMoa2204991").

    Returns
    -------
    dict
        A record conforming to the shared content schema.
        topic_tags, trust_score, and content_chunks are intentionally empty.
    """
    record = _build_empty_record(source)

    # ---- Step 1: resolve input to a PMID ------------------------------------
    candidate = _parse_input(source)
    if candidate is None:
        logger.error("Could not interpret input: %s", source)
        return record

    # If candidate looks like a PMID already, use it directly
    if _PMID_RE.match(candidate):
        pmid = candidate
    else:
        # Need esearch to convert DOI / search term → PMID
        pmid = _resolve_to_pmid(candidate)
        if not pmid:
            logger.error("esearch could not resolve a PMID for: %s", source)
            return record

    record["_pmid"] = pmid

    # ---- Step 2: efetch XML -------------------------------------------------
    xml_text = _fetch_pubmed_xml(pmid)
    if not xml_text:
        logger.error("efetch returned no content for PMID: %s", pmid)
        return record

    # ---- Step 3: parse XML --------------------------------------------------
    parsed = _parse_article_xml(xml_text)
    if not parsed:
        logger.error("XML parsing yielded no data for PMID: %s", pmid)
        return record

    # ---- Step 4: populate record --------------------------------------------
    record["author"]         = parsed.get("author")       # None if no authors
    record["published_date"] = parsed.get("year")          # "YYYY" or None
    record["_title"]         = parsed.get("title") or ""
    record["_journal"]       = parsed.get("journal") or ""
    record["_raw_text"]      = parsed.get("abstract") or ""

    # Language detection on abstract text
    record["language"] = _detect_language(record["_raw_text"])

    logger.info(
        "  ✓ Done | PMID=%s | author=%s | year=%s | lang=%s | abstract_chars=%d",
        pmid,
        record["author"],
        record["published_date"],
        record["language"],
        len(record["_raw_text"]),
    )
    return record


def scrape_pubmed_articles(sources: list[str]) -> list[dict]:
    """
    Convenience wrapper to scrape multiple PubMed sources.

    Parameters
    ----------
    sources : list[str]
        List of PubMed URLs, PMIDs, or DOIs.

    Returns
    -------
    list[dict]
        One record per source.
    """
    results = []
    for src in sources:
        logger.info("Processing PubMed source: %s", src)
        results.append(scrape_pubmed_article(src))
    return results


# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_SOURCES = [
        # Full PubMed URL
        "https://pubmed.ncbi.nlm.nih.gov/38505432/",
        # Bare PMID
        "33728050",
        # DOI string
        "10.1056/NEJMoa2301165",
    ]

    articles = scrape_pubmed_articles(TEST_SOURCES)
    for art in articles:
        print("\n" + "=" * 60)
        print(f"Source       : {art['source_url']}")
        print(f"PMID         : {art['_pmid']}")
        print(f"Title        : {art['_title']}")
        print(f"Authors      : {art['author']}")
        print(f"Journal      : {art['_journal']}")
        print(f"Year         : {art['published_date']}")
        print(f"Language     : {art['language']}")
        print(f"Abstract     : {art['_raw_text'][:120]}{'...' if len(art['_raw_text']) > 120 else ''}")
        print(f"Schema keys  : {[k for k in art if not k.startswith('_')]}")
