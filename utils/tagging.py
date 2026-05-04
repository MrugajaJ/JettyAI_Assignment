"""
utils/tagging.py
----------------
Extracts keyword topic tags from article text.

Primary method  : YAKE  (Yet Another Keyword Extractor)
Fallback method : Word-frequency count with a stop-word filter
                  (used when YAKE is unavailable or text is too short)

Public API
----------
generate_tags(text: str, max_tags: int = 6) -> list[str]
"""

import logging
import re
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum character length for YAKE to produce reliable output
_MIN_CHARS_YAKE = 80

# YAKE config
_YAKE_LANGUAGE            = "en"
_YAKE_MAX_NGRAM_SIZE      = 2
_YAKE_DEDUP_THRESHOLD     = 0.9
_YAKE_DEDUP_ALGO          = "seqm"       # sequence matcher — best for 2-gram dedup
_YAKE_WINDOW_SIZE         = 2

# ---------------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------------

# Compact but comprehensive English stop-word set used for both post-processing
# YAKE output and the frequency-fallback method.
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "as", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "few", "more", "most", "other", "some", "such", "than", "too", "very",
    "just", "its", "it", "this", "that", "these", "those", "their", "they",
    "them", "there", "then", "when", "where", "which", "who", "whom",
    "what", "how", "all", "any", "about", "above", "after", "also",
    "between", "into", "through", "during", "before", "same", "our",
    "we", "he", "she", "i", "you", "me", "him", "her", "us", "my",
    "your", "his", "our", "their", "one", "two", "new", "use", "used",
    "using", "based", "study", "studies", "results", "result", "show",
    "shown", "paper", "work", "however", "thus", "therefore", "although",
    "while", "since", "among", "within", "across", "without", "only",
    "here", "well", "whether", "per", "up", "out", "over", "under",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_keyword(kw: str) -> str:
    """
    Lowercase a keyword and strip leading/trailing punctuation and whitespace.
    Returns an empty string if the result is a pure stop word (single token).
    """
    kw = kw.lower().strip()
    # Remove non-alphanumeric characters from both ends (hyphens inside are OK)
    kw = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", kw)
    return kw


def _is_valid_tag(kw: str) -> bool:
    """Return True if `kw` is a non-empty, non-stop-word tag."""
    if not kw:
        return False
    tokens = kw.split()
    # Single-token tags must not be a stop word
    if len(tokens) == 1 and tokens[0] in _STOP_WORDS:
        return False
    # All tokens in a multi-gram must not ALL be stop words
    if all(t in _STOP_WORDS for t in tokens):
        return False
    return True


def _deduplicate(tags: list[str]) -> list[str]:
    """Remove duplicates while preserving insertion order."""
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            result.append(tag)
    return result


# ---------------------------------------------------------------------------
# Primary method — YAKE
# ---------------------------------------------------------------------------

def _yake_extract(text: str, max_tags: int) -> Optional[list[str]]:
    """
    Extract keywords using YAKE.
    Returns a list of cleaned keyword strings, or None if YAKE is unavailable
    or extraction fails.

    YAKE returns (keyword, score) tuples where a *lower* score means more
    relevant — so we sort ascending and take the top `max_tags`.
    """
    try:
        import yake  # lazy import so the module loads even without yake installed

        extractor = yake.KeywordExtractor(
            lan=_YAKE_LANGUAGE,
            n=_YAKE_MAX_NGRAM_SIZE,
            dedupLim=_YAKE_DEDUP_THRESHOLD,
            dedupFunc=_YAKE_DEDUP_ALGO,
            windowsSize=_YAKE_WINDOW_SIZE,
            top=max_tags * 3,   # over-fetch then filter, to account for stop-word removal
        )

        raw_keywords = extractor.extract_keywords(text)  # list of (kw, score)

        # Sort by score ascending (lower = better relevance in YAKE)
        raw_keywords.sort(key=lambda x: x[1])

        tags: list[str] = []
        for kw, _score in raw_keywords:
            cleaned = _clean_keyword(kw)
            if _is_valid_tag(cleaned):
                tags.append(cleaned)
            if len(tags) >= max_tags:
                break

        return _deduplicate(tags) if tags else None

    except ImportError:
        logger.warning(
            "yake is not installed. Run `pip install yake`. "
            "Falling back to word-frequency method."
        )
        return None

    except Exception as exc:
        logger.warning("YAKE extraction failed: %s — falling back to frequency method.", exc)
        return None


# ---------------------------------------------------------------------------
# Fallback method — word frequency
# ---------------------------------------------------------------------------

def _frequency_extract(text: str, max_tags: int) -> list[str]:
    """
    Simple word-frequency fallback.

    Steps:
      1. Lowercase and tokenise on non-alphanumeric boundaries.
      2. Remove stop words and tokens shorter than 3 characters.
      3. Count frequencies and return the `max_tags` most common tokens.
    """
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    filtered = [t for t in tokens if t not in _STOP_WORDS]

    if not filtered:
        return []

    most_common = Counter(filtered).most_common(max_tags)
    return [word for word, _count in most_common]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_tags(text: str, max_tags: int = 6) -> list[str]:
    """
    Extract up to `max_tags` keyword topic tags from `text`.

    Parameters
    ----------
    text : str
        The article body, abstract, or transcript to tag.
    max_tags : int, optional
        Maximum number of tags to return (default 6).

    Returns
    -------
    list[str]
        Lowercased, deduplicated keyword strings.
        Returns [] for empty input or if no valid tags can be extracted.

    Strategy
    --------
    1. YAKE keyword extractor (primary) — best for longer, well-formed text.
    2. Word-frequency counter (fallback) — used when YAKE is unavailable or
       the text is shorter than 80 characters.
    """
    # --- Guard: empty / whitespace ------------------------------------------
    if not text or not text.strip():
        return []

    cleaned_text = text.strip()

    # --- Choose extraction method -------------------------------------------
    if len(cleaned_text) >= _MIN_CHARS_YAKE:
        tags = _yake_extract(cleaned_text, max_tags)
        if tags:
            logger.debug("generate_tags: YAKE produced %d tags", len(tags))
            return tags
        # YAKE returned None or empty — fall through to frequency method
        logger.debug("generate_tags: YAKE returned no tags; using frequency fallback")

    else:
        logger.debug(
            "generate_tags: text too short for YAKE (%d < %d chars); "
            "using frequency fallback",
            len(cleaned_text), _MIN_CHARS_YAKE,
        )

    # --- Frequency fallback -------------------------------------------------
    tags = _frequency_extract(cleaned_text, max_tags)
    logger.debug("generate_tags: frequency fallback produced %d tags", len(tags))
    return tags


# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLES = {
        "ML article": (
            "Machine learning is a subfield of artificial intelligence that gives computers "
            "the ability to learn without being explicitly programmed. Deep learning uses "
            "neural networks with many layers to model complex patterns in data. "
            "Convolutional neural networks are widely used for image classification tasks. "
            "Transfer learning allows models trained on large datasets to be fine-tuned for "
            "specific downstream tasks with limited labelled data."
        ),
        "Medical abstract": (
            "Background: Type 2 diabetes mellitus is a chronic metabolic disorder characterised "
            "by insulin resistance and impaired beta-cell function. Methods: A randomised "
            "controlled trial was conducted in 320 adult patients. Results: HbA1c levels were "
            "significantly reduced in the intervention group receiving metformin combined with "
            "lifestyle modification compared to the control group receiving metformin alone."
        ),
        "Short text": "AI is cool.",
        "Empty text": "",
    }

    for label, text in SAMPLES.items():
        tags = generate_tags(text, max_tags=6)
        print(f"\n[{label}]")
        print(f"  Tags ({len(tags)}): {tags}")
