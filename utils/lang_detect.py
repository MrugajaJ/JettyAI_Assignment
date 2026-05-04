"""
utils/lang_detect.py
--------------------
Thin, safe wrapper around the `langdetect` library for ISO 639-1
language code detection.

Public API
----------
detect_language(text: str) -> str
    Returns an ISO 639-1 code (e.g. 'en', 'fr', 'de') or 'unknown'
    if detection fails or the input is too short to be reliable.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum character count needed for a reliable langdetect result.
# The library's own docs suggest at least ~20 characters.
_MIN_CHARS = 20

# langdetect uses a probabilistic model; by default it is non-deterministic.
# Setting a fixed seed makes results reproducible across runs.
_LANGDETECT_SEED = 42

# Sentinel returned on any failure
UNKNOWN = "unknown"

# ---------------------------------------------------------------------------
# Optional: seed langdetect for reproducibility
# ---------------------------------------------------------------------------
try:
    from langdetect import DetectorFactory
    DetectorFactory.seed = _LANGDETECT_SEED
except ImportError:
    pass   # handled inside detect_language below


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Detect the natural language of `text` and return its ISO 639-1 code.

    Parameters
    ----------
    text : str
        The text whose language should be detected.  Can be a sentence,
        paragraph, or longer document — longer is more reliable.

    Returns
    -------
    str
        ISO 639-1 language code  (e.g. 'en', 'fr', 'de', 'zh-cn')
        or  'unknown'  in any of these cases:
          - text is None, empty, or purely whitespace
          - text is shorter than 20 characters
          - langdetect raises LangDetectException (indeterminate result)
          - langdetect is not installed
    """
    # --- Guard: None / empty / whitespace -----------------------------------
    if not text or not text.strip():
        logger.debug("detect_language: empty input → 'unknown'")
        return UNKNOWN

    cleaned = text.strip()

    # --- Guard: too short for reliable detection ----------------------------
    if len(cleaned) < _MIN_CHARS:
        logger.debug(
            "detect_language: text too short (%d chars < %d) → 'unknown'",
            len(cleaned), _MIN_CHARS,
        )
        return UNKNOWN

    # --- Detection ----------------------------------------------------------
    try:
        from langdetect import detect, LangDetectException

        lang_code = detect(cleaned)

        if not lang_code.startswith("en"):
            logger.warning(
                "detect_language: non-English content detected (lang='%s') "
                "for text[:60]=%r — content will still be processed.",
                lang_code, cleaned[:60],
            )
        else:
            logger.debug("detect_language: detected '%s' for text[:60]=%r", lang_code, cleaned[:60])

        return lang_code

    except LangDetectException as exc:
        # Raised when langdetect cannot determine a language with confidence
        logger.warning(
            "LangDetectException for text[:60]=%r — returning 'unknown'. Reason: %s",
            cleaned[:60], exc,
        )
        return UNKNOWN

    except ImportError:
        logger.error(
            "langdetect is not installed. "
            "Run `pip install langdetect` and retry."
        )
        return UNKNOWN

    except Exception as exc:
        # Catch-all: never let a detection failure crash the pipeline
        logger.warning(
            "Unexpected error in detect_language: %s — returning 'unknown'", exc
        )
        return UNKNOWN


# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TEST_CASES = [
        # (input_text, expected_language_approx)
        ("The quick brown fox jumps over the lazy dog.", "en"),
        ("La vie est belle et le soleil brille aujourd'hui.", "fr"),
        ("Die Wissenschaft ist der Schlüssel zur Zukunft.", "de"),
        ("El aprendizaje automático está transformando la industria.", "es"),
        ("机器学习正在改变世界。", "zh-cn"),
        ("",               "unknown"),          # empty
        ("Hi",             "unknown"),          # too short
        ("   \n\n  ",      "unknown"),          # whitespace only
    ]

    print(f"{'Input':<55} {'Expected':<10} {'Got':<10} {'OK?'}")
    print("-" * 90)
    for text, expected in TEST_CASES:
        result = detect_language(text)
        ok = "✓" if result == expected else "✗"
        display = repr(text[:50])
        print(f"{display:<55} {expected:<10} {result:<10} {ok}")
