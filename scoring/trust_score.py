"""
scoring/trust_score.py
----------------------
Computes a composite trust score (0.0 – 1.0) for a scraped content record.

The score is a weighted sum of five independent sub-scores:

  Component                  Weight   Range
  ─────────────────────────  ──────   ─────
  author_score               0.25     0–1
  citation_score             0.20     0–1
  domain_authority_score     0.25     0–1
  recency_score              0.20     0–1
  medical_disclaimer_score   0.10     0–1
  ─────────────────────────  ──────
  trust_score (final)        1.00     0–1  (rounded to 2 d.p.)

Public API
----------
get_domain_authority(url: str) -> float
calculate_trust_score(source: dict, domain_authority: float | None = None) -> float
score_breakdown(source: dict, domain_authority: float | None = None) -> dict
apply_abuse_penalties(score: float, source: dict, domain_authority: float | None = None) -> float
"""

import logging
import os
import re
from datetime import date, datetime
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weights  (must sum to 1.0)
# ---------------------------------------------------------------------------

_W_AUTHOR      = 0.25
_W_CITATION    = 0.20
_W_DOMAIN_AUTH = 0.25
_W_RECENCY     = 0.20
_W_DISCLAIMER  = 0.10

assert abs((_W_AUTHOR + _W_CITATION + _W_DOMAIN_AUTH + _W_RECENCY + _W_DISCLAIMER) - 1.0) < 1e-9, \
    "Trust score weights must sum to exactly 1.0"

# ---------------------------------------------------------------------------
# Author score — known institutional / organisational names
# ---------------------------------------------------------------------------

# Exact or partial substring matches (case-insensitive).
# A longer list makes the heuristic more precise; extend as needed.
_INSTITUTIONAL_NAMES: frozenset[str] = frozenset({
    # Health / science bodies
    "nih", "cdc", "who", "fda", "ema", "ncbi", "pubmed",
    "nature", "science", "lancet", "jama", "bmj", "nejm",
    "new england journal", "british medical journal",
    "national institutes of health", "centers for disease control",
    "world health organization", "european medicines agency",
    # Universities / research bodies (common names)
    "university", "institute", "college", "hospital", "foundation",
    "association", "academy", "society", "consortium", "center",
    "department", "laboratory", "lab", "research group",
    # News / publishers
    "reuters", "associated press", "bbc", "guardian", "times",
    "washington post", "new york times", "forbes", "economist",
    "springer", "elsevier", "wiley", "oxford", "cambridge",
})

# ---------------------------------------------------------------------------
# Medical disclaimer phrases
# ---------------------------------------------------------------------------

_DISCLAIMER_PHRASES: tuple[str, ...] = (
    "consult a doctor",
    "consult your doctor",
    "consult a physician",
    "seek medical advice",
    "medical advice",
    "healthcare professional",
    "health care professional",
    "consult a healthcare",
    "speak to a doctor",
    "talk to your doctor",
    "not a substitute for",
    "not medical advice",
    "for informational purposes only",
    "always consult",
)

# Source types where the disclaimer check is meaningful
_MEDICAL_SOURCE_TYPES: frozenset[str] = frozenset({"pubmed", "blog"})


# ---------------------------------------------------------------------------
# Sub-score: author
# ---------------------------------------------------------------------------

def _author_score(source: dict) -> float:
    """
    0.30  — author is null / missing
    0.60  — author is present but appears to be a single personal name
    1.00  — author matches an institutional / organisational name
    """
    author: Optional[str] = source.get("author")

    if not author or not str(author).strip():
        logger.debug("author_score: no author → 0.30")
        return 0.30

    author_lower = author.strip().lower()

    # Check against the institutional name list (substring match)
    if any(inst in author_lower for inst in _INSTITUTIONAL_NAMES):
        logger.debug("author_score: institutional match ('%s') → 1.00", author)
        return 1.00

    # Heuristic: a single personal name usually contains 1–3 comma/space-separated tokens
    # and doesn't contain institutional keywords.
    logger.debug("author_score: personal name ('%s') → 0.60", author)
    return 0.60


# ---------------------------------------------------------------------------
# Sub-score: citation (source-type proxy)
# ---------------------------------------------------------------------------

_CITATION_BY_SOURCE: dict[str, float] = {
    "pubmed":  1.0,   # peer-reviewed; citation data exists via API (future)
    "youtube": 0.5,   # user-generated; no formal citation system
    "blog":    0.3,   # self-published; lowest baseline trust
}
_CITATION_DEFAULT = 0.3   # unknown source types


def _citation_score(source: dict) -> float:
    """
    Proxy citation score based on source type.
    Real citation counts require a separate API call (Semantic Scholar, CrossRef, etc.).
    """
    source_type = (source.get("source_type") or "").lower()
    score = _CITATION_BY_SOURCE.get(source_type, _CITATION_DEFAULT)
    logger.debug("citation_score: source_type='%s' → %.2f", source_type, score)
    return score


# ---------------------------------------------------------------------------
# Sub-score: domain authority
# ---------------------------------------------------------------------------

_DOMAIN_AUTH_DEFAULT = 0.4   # used when Open PageRank score is unavailable
_DOMAIN_AUTH_MAX     = 10.0  # Open PageRank scale: 0–10


def _domain_authority_score(domain_authority: Optional[float]) -> float:
    """
    Normalise an Open PageRank score (0–10) to a 0–1 value.
    If None or out of range, return the default 0.40.
    """
    if domain_authority is None:
        logger.debug("domain_authority_score: no score provided → %.2f (default)", _DOMAIN_AUTH_DEFAULT)
        return _DOMAIN_AUTH_DEFAULT

    try:
        da = float(domain_authority)
    except (TypeError, ValueError):
        logger.warning("domain_authority_score: invalid value '%s' → default", domain_authority)
        return _DOMAIN_AUTH_DEFAULT

    if not (0.0 <= da <= _DOMAIN_AUTH_MAX):
        logger.warning(
            "domain_authority_score: value %.2f out of [0, %.0f] range → clamping",
            da, _DOMAIN_AUTH_MAX,
        )
        da = max(0.0, min(da, _DOMAIN_AUTH_MAX))

    score = da / _DOMAIN_AUTH_MAX
    logger.debug("domain_authority_score: DA=%.2f → %.2f", da, score)
    return score


# ---------------------------------------------------------------------------
# Sub-score: recency
# ---------------------------------------------------------------------------

def _parse_year(date_str: str) -> Optional[int]:
    """
    Extract a 4-digit year from a date string.
    Accepts: 'YYYY', 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM:SSZ', etc.
    Returns None if no year can be parsed.
    """
    if not date_str:
        return None
    match = re.search(r"\b(19|20)\d{2}\b", str(date_str))
    return int(match.group(0)) if match else None


def _recency_score(source: dict) -> float:
    """
    1.0  — published within the last 1 year
    0.7  — 1–3 years ago
    0.4  — 3–5 years ago
    0.2  — date is null / unparseable  (raised from 0.1 — total absence of
           date info is less penalising than provably outdated content)
    0.1  — older than 5 years
    """
    raw_date: Optional[str] = source.get("published_date")
    year = _parse_year(raw_date) if raw_date else None

    if year is None:
        logger.debug("recency_score: no parseable date → 0.20")
        return 0.20

    current_year = date.today().year
    age_years = current_year - year

    if age_years <= 1:
        score = 1.0
    elif age_years <= 3:
        score = 0.7
    elif age_years <= 5:
        score = 0.4
    else:
        score = 0.1

    logger.debug("recency_score: year=%d, age=%d years → %.2f", year, age_years, score)
    return score


# ---------------------------------------------------------------------------
# Sub-score: medical disclaimer
# ---------------------------------------------------------------------------

def _medical_disclaimer_score(source: dict) -> float:
    """
    For medical source types (pubmed, blog):
      1.0  — at least one disclaimer phrase is present in the content
      0.4  — no disclaimer found (medical content without a caveat is riskier)

    For non-medical source types (youtube, etc.):
      0.8  — neutral / not applicable (no penalty, no bonus)
    """
    source_type = (source.get("source_type") or "").lower()

    if source_type not in _MEDICAL_SOURCE_TYPES:
        logger.debug("medical_disclaimer_score: non-medical source_type='%s' → 0.80", source_type)
        return 0.80

    # Combine all available text fields for the search
    text_fields = [
        source.get("_raw_text", ""),
        source.get("_description", ""),
        source.get("_title", ""),
    ]
    combined = " ".join(f for f in text_fields if f).lower()

    for phrase in _DISCLAIMER_PHRASES:
        if phrase in combined:
            logger.debug(
                "medical_disclaimer_score: found '%s' in content → 1.00", phrase
            )
            return 1.00

    logger.debug("medical_disclaimer_score: no disclaimer phrase found → 0.40")
    return 0.40


# ---------------------------------------------------------------------------
# Domain authority lookup — Open PageRank API
# ---------------------------------------------------------------------------

_OPR_API_URL     = "https://openpagerank.com/api/v1.0/getPageRank"
_OPR_DEFAULT     = 5.0   # neutral mid-range fallback on any error
_OPR_TIMEOUT     = 10    # seconds


def get_domain_authority(url: str) -> float:
    """
    Query the Open PageRank API and return the page_rank_decimal score (0–10)
    for the domain extracted from `url`.

    Parameters
    ----------
    url : str
        Any URL whose hostname will be looked up (e.g. 'https://pubmed.ncbi.nlm.nih.gov/...').

    Returns
    -------
    float
        Open PageRank score in [0.0, 10.0].
        Returns 5.0 (neutral default) on any of:
          - OPR_API_KEY env var is missing or empty
          - Network / HTTP error
          - Unexpected JSON structure
          - Domain not found in the API response

    Environment
    -----------
    OPR_API_KEY : str
        Your Open PageRank API key (free tier available at https://openpagerank.com).
    """
    # --- Extract hostname ---------------------------------------------------
    try:
        parsed  = urlparse(url)
        domain  = parsed.hostname or ""
        # Strip leading 'www.' so 'www.ncbi.nlm.nih.gov' → 'ncbi.nlm.nih.gov'
        if domain.startswith("www."):
            domain = domain[4:]
        if not domain:
            logger.warning("get_domain_authority: could not extract domain from '%s' → %.1f", url, _OPR_DEFAULT)
            return _OPR_DEFAULT
    except Exception as exc:
        logger.warning("get_domain_authority: URL parse error for '%s': %s → %.1f", url, exc, _OPR_DEFAULT)
        return _OPR_DEFAULT

    # --- Read API key -------------------------------------------------------
    api_key = os.environ.get("OPR_API_KEY", "").strip()
    if not api_key:
        logger.warning(
            "OPR_API_KEY environment variable is not set. "
            "Returning neutral default %.1f for domain '%s'.",
            _OPR_DEFAULT, domain,
        )
        return _OPR_DEFAULT

    # --- Call Open PageRank API ---------------------------------------------
    try:
        response = httpx.get(
            _OPR_API_URL,
            params={"domains[]": domain},
            headers={"API-OPR": api_key},
            timeout=_OPR_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    except httpx.HTTPStatusError as exc:
        logger.warning(
            "get_domain_authority: HTTP %d from OPR for '%s' → %.1f. Detail: %s",
            exc.response.status_code, domain, _OPR_DEFAULT, exc,
        )
        return _OPR_DEFAULT
    except httpx.HTTPError as exc:
        logger.warning(
            "get_domain_authority: network error for '%s' → %.1f. Detail: %s",
            domain, _OPR_DEFAULT, exc,
        )
        return _OPR_DEFAULT
    except Exception as exc:
        logger.warning(
            "get_domain_authority: unexpected error for '%s' → %.1f. Detail: %s",
            domain, _OPR_DEFAULT, exc,
        )
        return _OPR_DEFAULT

    # --- Parse response -----------------------------------------------------
    # OPR response shape:
    # {"status": 200, "response": [{"page_rank_decimal": 7.23, "domain": "...", ...}]}
    try:
        results = data.get("response", [])
        if not results:
            logger.warning(
                "get_domain_authority: empty 'response' array for '%s' → %.1f",
                domain, _OPR_DEFAULT,
            )
            return _OPR_DEFAULT

        pr_value = results[0].get("page_rank_decimal")
        if pr_value is None:
            logger.warning(
                "get_domain_authority: 'page_rank_decimal' missing for '%s' → %.1f",
                domain, _OPR_DEFAULT,
            )
            return _OPR_DEFAULT

        score = float(pr_value)
        logger.info("get_domain_authority: domain='%s' → OPR score=%.2f", domain, score)
        return score

    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning(
            "get_domain_authority: JSON parse error for '%s' → %.1f. Detail: %s",
            domain, _OPR_DEFAULT, exc,
        )
        return _OPR_DEFAULT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_trust_score(
    source: dict,
    domain_authority: Optional[float] = None,
) -> float:
    """
    Compute a composite trust score for a scraped content record.

    Parameters
    ----------
    source : dict
        A content record as returned by any of the scrapers.
        Expected keys: source_type, author, published_date, _raw_text, etc.
    domain_authority : float | None, optional
        Open PageRank score (0–10) for the source domain.
        Pass None to use the default of 0.40.

    Returns
    -------
    float
        Trust score in [0.0, 1.0], rounded to 2 decimal places.
    """
    a  = _author_score(source)
    c  = _citation_score(source)
    d  = _domain_authority_score(domain_authority)
    r  = _recency_score(source)
    m  = _medical_disclaimer_score(source)

    raw_score = (
        _W_AUTHOR      * a +
        _W_CITATION    * c +
        _W_DOMAIN_AUTH * d +
        _W_RECENCY     * r +
        _W_DISCLAIMER  * m
    )

    base_trust = round(min(max(raw_score, 0.0), 1.0), 2)

    logger.info(
        "trust_score | url=%-55s | author=%.2f | citation=%.2f | "
        "domain=%.2f | recency=%.2f | disclaimer=%.2f | BASE=%.2f",
        str(source.get("source_url", "?"))[:55],
        a, c, d, r, m, base_trust,
    )

    # Apply abuse-prevention penalty layer
    trust = apply_abuse_penalties(base_trust, source, domain_authority)
    return trust


def score_breakdown(
    source: dict,
    domain_authority: Optional[float] = None,
) -> dict:
    """
    Return a detailed breakdown of every sub-score alongside the final trust score.
    Useful for debugging and audit trails.

    Returns
    -------
    dict with keys:
        author_score, citation_score, domain_authority_score,
        recency_score, medical_disclaimer_score, trust_score
    """
    a = _author_score(source)
    c = _citation_score(source)
    d = _domain_authority_score(domain_authority)
    r = _recency_score(source)
    m = _medical_disclaimer_score(source)

    raw_score = (
        _W_AUTHOR      * a +
        _W_CITATION    * c +
        _W_DOMAIN_AUTH * d +
        _W_RECENCY     * r +
        _W_DISCLAIMER  * m
    )

    base_trust = round(min(max(raw_score, 0.0), 1.0), 2)
    final_trust = apply_abuse_penalties(base_trust, source, domain_authority)

    return {
        "author_score":             round(a, 4),
        "citation_score":           round(c, 4),
        "domain_authority_score":   round(d, 4),
        "recency_score":            round(r, 4),
        "medical_disclaimer_score": round(m, 4),
        "base_trust_score":         base_trust,
        "trust_score":              final_trust,   # after abuse penalties
    }


# ---------------------------------------------------------------------------
# Abuse prevention — post-score penalty layer
# ---------------------------------------------------------------------------

# Generic/bot-like author names that indicate low-quality or spammy content
_FAKE_AUTHOR_NAMES: frozenset[str] = frozenset({
    "admin", "user", "editor", "webmaster", "author", "staff",
    "moderator", "mod", "anonymous", "anon", "guest", "contributor",
    "writer", "blogger", "team", "support",
})

# Topic tags that flag health-sensitive content requiring a disclaimer
_HEALTH_TAGS: frozenset[str] = frozenset({
    "health", "medical", "medicine", "treatment", "disease", "drug",
    "symptom", "diagnosis", "therapy", "prescription", "clinical",
    "patient", "vaccine", "virus", "infection", "surgery",
})

# Minimum domain authority below which SEO spam penalty is applied
_SPAM_DA_THRESHOLD = 2.0


def apply_abuse_penalties(
    score: float,
    source: dict,
    domain_authority: Optional[float] = None,
) -> float:
    """
    Apply post-scoring abuse-prevention penalties to an existing trust score.

    This is a *deduction layer* that runs after calculate_trust_score.  Each
    rule is evaluated independently; multiple penalties can stack.

    Penalty rules
    -------------
    1. Fake/generic author     − 0.15   if author matches a known generic name
                                        (e.g. 'admin', 'editor', 'webmaster')
    2. SEO spam domain         − 0.20   if domain authority < 2.0
    3. Missing medical disc.   − 0.20   if topic_tags contain a health keyword
                                        AND source_type == 'blog'
                                        AND no disclaimer phrase found in text
    4. Outdated content        − 0.15   if published_date is more than 5 years
                                        before the current year

    Parameters
    ----------
    score : float
        Base trust score already computed by calculate_trust_score.
    source : dict
        The full content record (same dict passed to calculate_trust_score).
    domain_authority : float | None, optional
        Open PageRank score (0–10).  Required for rule 2; if None, rule 2
        is skipped.

    Returns
    -------
    float
        Adjusted score clamped to [0.0, 1.0], rounded to 2 decimal places.
    """
    adjusted = score
    penalties: list[str] = []   # human-readable log of applied penalties

    # ---- Rule 1: fake / generic author -------------------------------------
    author = (source.get("author") or "").strip().lower()
    # Check each whitespace-separated token individually so 'admin team' matches
    author_tokens = set(author.split())
    if author_tokens & _FAKE_AUTHOR_NAMES:
        adjusted -= 0.15
        penalties.append("fake_author(−0.15)")

    # ---- Rule 2: SEO spam domain -------------------------------------------
    if domain_authority is not None:
        try:
            da = float(domain_authority)
            if da < _SPAM_DA_THRESHOLD:
                adjusted -= 0.20
                penalties.append(f"spam_domain(DA={da:.1f},−0.20)")
        except (TypeError, ValueError):
            pass   # invalid DA value — skip rule

    # ---- Rule 3: missing medical disclaimer on health blog ------------------
    source_type = (source.get("source_type") or "").lower()
    if source_type == "blog":
        tags_lower = {t.lower() for t in (source.get("topic_tags") or [])}
        if tags_lower & _HEALTH_TAGS:
            # Check if any disclaimer phrase is present in the text
            text_fields = [
                source.get("_raw_text", ""),
                source.get("_description", ""),
                source.get("_title", ""),
            ]
            combined = " ".join(f for f in text_fields if f).lower()
            has_disclaimer = any(phrase in combined for phrase in _DISCLAIMER_PHRASES)
            if not has_disclaimer:
                adjusted -= 0.20
                penalties.append("missing_medical_disclaimer(−0.20)")

    # ---- Rule 4: outdated content (> 5 years) -------------------------------
    raw_date = source.get("published_date")
    year = _parse_year(raw_date) if raw_date else None
    if year is not None:
        age = date.today().year - year
        if age > 5:
            adjusted -= 0.15
            penalties.append(f"outdated({year},age={age}y,−0.15)")

    # ---- Clamp and round ----------------------------------------------------
    final = round(min(max(adjusted, 0.0), 1.0), 2)

    if penalties:
        logger.info(
            "abuse_penalties | url=%-50s | penalties=%s | %.2f → %.2f",
            str(source.get("source_url", "?"))[:50],
            ", ".join(penalties),
            score, final,
        )
    else:
        logger.debug(
            "abuse_penalties | url=%-50s | no penalties applied | score=%.2f",
            str(source.get("source_url", "?"))[:50], final,
        )

    return final


# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_RECORDS = [
        {
            "label": "PubMed — institutional author, recent, has disclaimer",
            "record": {
                "source_url":     "https://pubmed.ncbi.nlm.nih.gov/38505432/",
                "source_type":    "pubmed",
                "author":         "National Institutes of Health",
                "published_date": "2024",
                "_raw_text":      (
                    "This study presents findings on vaccine efficacy. "
                    "Always consult a healthcare professional before making medical decisions."
                ),
            },
            "domain_authority": 9.2,
        },
        {
            "label": "YouTube — personal author, 2 years old, no disclaimer",
            "record": {
                "source_url":     "https://www.youtube.com/watch?v=abc123",
                "source_type":    "youtube",
                "author":         "Jane Smith",
                "published_date": "2022-11-15",
                "_raw_text":      "Today we explore neural networks and their applications.",
            },
            "domain_authority": 8.5,
        },
        {
            "label": "Blog — no author, old date, no disclaimer",
            "record": {
                "source_url":     "https://some-health-blog.com/article",
                "source_type":    "blog",
                "author":         None,
                "published_date": "2018-03-01",
                "_raw_text":      "Eating well is important. Exercise every day.",
            },
            "domain_authority": None,
        },
        {
            "label": "Blog — CDC as author, very recent, has disclaimer",
            "record": {
                "source_url":     "https://cdc.gov/flu/prevention",
                "source_type":    "blog",
                "author":         "CDC",
                "published_date": "2025-01-10",
                "_raw_text":      (
                    "Flu vaccines are recommended annually. "
                    "This content is for informational purposes only. "
                    "Seek medical advice from a healthcare professional."
                ),
            },
            "domain_authority": 8.8,
        },
    ]

    print(f"\n{'Label':<55} {'Score':>6}  Breakdown")
    print("─" * 120)
    for item in TEST_RECORDS:
        breakdown = score_breakdown(item["record"], item.get("domain_authority"))
        print(
            f"{item['label']:<55} {breakdown['trust_score']:>6.2f}  "
            f"author={breakdown['author_score']:.2f}  "
            f"citation={breakdown['citation_score']:.2f}  "
            f"domain={breakdown['domain_authority_score']:.2f}  "
            f"recency={breakdown['recency_score']:.2f}  "
            f"disclaimer={breakdown['medical_disclaimer_score']:.2f}"
        )
