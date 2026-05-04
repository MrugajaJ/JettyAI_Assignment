"""
youtube_scraper.py
------------------
Scrapes a list of YouTube video URLs and extracts structured data.

Metadata  : YouTube Data API v3  (google-api-python-client)
            API key read from env var  YOUTUBE_API_KEY
Transcript: youtube-transcript-api  (English preferred; auto-generated fallback)

Returns a list of dicts matching the shared content schema (same as blog_scraper).
source_type is set to 'youtube'.
"""

import logging
import os
import re
from typing import Optional
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from googleapiclient.discovery import build as yt_build
    from googleapiclient.errors import HttpError as YTHttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    logging.warning(
        "google-api-python-client is not installed; YouTube metadata will be unavailable."
    )

try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )
    TRANSCRIPT_API_AVAILABLE = True
except ImportError:
    TRANSCRIPT_API_AVAILABLE = False
    logging.warning(
        "youtube-transcript-api is not installed; transcripts will be unavailable."
    )

try:
    from langdetect import detect as _langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Preference order for transcript language codes
_TRANSCRIPT_LANG_PREFERENCE = ["en", "en-US", "en-GB", "en-AU", "en-CA"]

# YouTube API service name / version
_YT_API_SERVICE = "youtube"
_YT_API_VERSION = "v3"


# ---------------------------------------------------------------------------
# Helpers — URL parsing
# ---------------------------------------------------------------------------

def _extract_video_id(url: str) -> Optional[str]:
    """
    Extract the YouTube video ID from any common URL format.

    Supports:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/embed/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - https://m.youtube.com/watch?v=VIDEO_ID
    """
    url = url.strip()

    # youtu.be short links
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid if vid else None

    # Standard / embed / shorts / mobile
    if parsed.hostname in (
        "www.youtube.com", "youtube.com", "m.youtube.com",
        "www.youtube-nocookie.com",
    ):
        # /watch?v=
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]

        # /embed/VIDEO_ID  or  /shorts/VIDEO_ID
        path_parts = parsed.path.lstrip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] in ("embed", "shorts", "v"):
            return path_parts[1]

    # Last resort — regex
    match = re.search(
        r"(?:v=|youtu\.be/|embed/|shorts/)([a-zA-Z0-9_-]{11})", url
    )
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Helpers — schema
# ---------------------------------------------------------------------------

def _build_empty_record(url: str, video_id: Optional[str]) -> dict:
    """Return the schema skeleton for a single YouTube video."""
    return {
        "source_url": url,
        "source_type": "youtube",
        "author": None,              # channel title
        "published_date": None,      # ISO date YYYY-MM-DD
        "language": None,
        "region": None,
        "topic_tags": [],            # filled later
        "trust_score": None,         # filled later
        "content_chunks": [],        # filled later
        "transcript_available": False,  # updated after fetch attempt
        # Internal fields (prefixed with _) consumed downstream
        "_video_id": video_id,
        "_title": "",
        "_description": "",
        "_raw_text": "",             # concatenated transcript segments
    }


# ---------------------------------------------------------------------------
# Helpers — YouTube Data API
# ---------------------------------------------------------------------------

def _get_api_key() -> Optional[str]:
    """Read the YouTube API key from the environment."""
    key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not key:
        logger.error(
            "YOUTUBE_API_KEY environment variable is not set or is empty. "
            "Video metadata will not be fetched."
        )
        return None
    return key


def _build_yt_client():
    """Build and return an authenticated YouTube API client, or None on failure."""
    if not GOOGLE_API_AVAILABLE:
        return None
    api_key = _get_api_key()
    if not api_key:
        return None
    try:
        return yt_build(_YT_API_SERVICE, _YT_API_VERSION, developerKey=api_key)
    except Exception as exc:
        logger.error("Failed to build YouTube API client: %s", exc)
        return None


def _fetch_video_metadata(client, video_id: str) -> Optional[dict]:
    """
    Call the YouTube Data API videos.list endpoint for a single video.

    Returns a simplified dict:
        {title, channel_title, published_at (YYYY-MM-DD), description}
    or None on failure.
    """
    if client is None:
        return None
    try:
        response = (
            client.videos()
            .list(part="snippet", id=video_id)
            .execute()
        )
        items = response.get("items", [])
        if not items:
            logger.warning("No API results for video ID: %s", video_id)
            return None

        snippet = items[0].get("snippet", {})

        # published_at comes as RFC 3339 — keep only the date part
        raw_date = snippet.get("publishedAt", "")
        published_date = raw_date[:10] if raw_date else None  # YYYY-MM-DD

        return {
            "title": snippet.get("title") or None,
            "channel_title": snippet.get("channelTitle") or None,
            "published_date": published_date,
            "description": snippet.get("description") or "",
        }

    except YTHttpError as exc:
        logger.error("YouTube API HTTP error for video %s: %s", video_id, exc)
        return None
    except Exception as exc:
        logger.error("Unexpected error fetching metadata for %s: %s", video_id, exc)
        return None


# ---------------------------------------------------------------------------
# Helpers — Transcript
# ---------------------------------------------------------------------------

def _fetch_transcript(video_id: str) -> Optional[str]:
    """
    Fetch the English transcript for a YouTube video and return it as a
    single concatenated string.

    Priority:
      1. Manually created English transcript (en / en-US / en-GB …)
      2. Auto-generated English transcript

    Returns None if no English transcript is available or if the API is
    not installed, logging an appropriate warning in both cases.
    """
    if not TRANSCRIPT_API_AVAILABLE:
        logger.warning(
            "youtube-transcript-api not available; transcript set to null for %s.",
            video_id,
        )
        return None

    try:
        # v1.x requires instantiation; v0.x used class methods
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        transcript = None

        # 1 — Try manually created English transcripts first
        for lang_code in _TRANSCRIPT_LANG_PREFERENCE:
            try:
                transcript = transcript_list.find_manually_created_transcript(
                    [lang_code]
                )
                logger.info("  → Found manual transcript (%s) for %s", lang_code, video_id)
                break
            except NoTranscriptFound:
                continue

        # 2 — Fall back to auto-generated English
        if transcript is None:
            for lang_code in _TRANSCRIPT_LANG_PREFERENCE:
                try:
                    transcript = transcript_list.find_generated_transcript(
                        [lang_code]
                    )
                    logger.info(
                        "  → Found auto-generated transcript (%s) for %s",
                        lang_code, video_id,
                    )
                    break
                except NoTranscriptFound:
                    continue

        if transcript is None:
            logger.warning(
                "No English transcript found for video %s; setting transcript to null.",
                video_id,
            )
            return None

        # Fetch and concatenate all segment texts
        # Handle both v0.x (dict segments) and v1.x (object segments)
        segments = transcript.fetch()
        parts = []
        for seg in segments:
            if isinstance(seg, dict):
                parts.append(seg.get("text", ""))
            else:
                parts.append(getattr(seg, "text", ""))
        full_text = " ".join(parts)
        # Collapse excessive whitespace
        full_text = " ".join(full_text.split())
        return full_text if full_text else None

    except TranscriptsDisabled:
        logger.warning(
            "Transcripts are disabled for video %s; setting transcript to null.",
            video_id,
        )
        return None
    except VideoUnavailable:
        logger.warning(
            "Video %s is unavailable; setting transcript to null.", video_id
        )
        return None
    except Exception as exc:
        logger.warning(
            "Unexpected error fetching transcript for %s: %s — setting to null.",
            video_id, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Helpers — Language detection
# ---------------------------------------------------------------------------

def _detect_language(text: str) -> Optional[str]:
    """Detect ISO 639-1 language code from text; returns None on failure."""
    if not LANGDETECT_AVAILABLE:
        return None
    try:
        if text and len(text.strip()) > 30:
            return _langdetect(text)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrape_youtube_videos(urls: list[str]) -> list[dict]:
    """
    Scrape a list of YouTube video URLs and return structured records.

    Parameters
    ----------
    urls : list[str]
        YouTube video URLs (designed for 2, works for any count).

    Returns
    -------
    list[dict]
        One dict per URL conforming to the shared content schema.
        topic_tags, trust_score, and content_chunks are intentionally empty.
        _raw_text contains the full concatenated transcript (or empty string).
    """
    # Build the API client once and reuse across all videos
    yt_client = _build_yt_client()

    results = []

    for url in urls:
        logger.info("Processing YouTube URL: %s", url)

        # ---- Step 1: extract video ID ----------------------------------------
        video_id = _extract_video_id(url)
        if not video_id:
            logger.error("  ✗ Could not extract video ID from URL: %s", url)
            results.append(_build_empty_record(url, None))
            continue

        logger.info("  → Video ID: %s", video_id)
        record = _build_empty_record(url, video_id)

        # ---- Step 2: fetch metadata via YouTube Data API ---------------------
        metadata = _fetch_video_metadata(yt_client, video_id)
        if metadata:
            record["author"] = metadata.get("channel_title")   # None if missing
            record["published_date"] = metadata.get("published_date")  # None if missing
            record["_title"] = metadata.get("title") or ""
            record["_description"] = metadata.get("description") or ""
        else:
            logger.warning(
                "  ⚠ Metadata unavailable for %s; author and date set to null.", video_id
            )
            # author and published_date remain None — graceful degradation

        # ---- Step 3: fetch transcript ----------------------------------------
        transcript_text = _fetch_transcript(video_id)
        has_transcript = transcript_text is not None and len(transcript_text) > 0
        record["transcript_available"] = has_transcript

        if has_transcript:
            record["_raw_text"] = transcript_text
        else:
            # Fallback: use description as raw text so chunking still runs
            description = record.get("_description", "")
            record["_raw_text"] = description
            if description:
                logger.info(
                    "  ℹ Transcript unavailable for %s; "
                    "will chunk video description (%d chars) instead.",
                    video_id, len(description),
                )
            else:
                logger.warning(
                    "  ⚠ Transcript and description both unavailable for %s; "
                    "_raw_text will be empty.", video_id,
                )

        # ---- Step 4: detect language from transcript (preferred) or description
        detection_source = transcript_text or record["_description"]
        record["language"] = _detect_language(detection_source)

        # ---- Step 5: region — YouTube is global; leave None unless metadata
        #              exposes a default language that maps to a region
        record["region"] = None

        results.append(record)
        logger.info(
            "  ✓ Done | channel=%s | date=%s | lang=%s | transcript_chars=%d",
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
    # Requires YOUTUBE_API_KEY to be set in the environment
    # Example: set YOUTUBE_API_KEY=YOUR_KEY  (Windows)
    #          export YOUTUBE_API_KEY=YOUR_KEY  (Unix)

    TEST_URLS = [
        "https://www.youtube.com/watch?v=aircAruvnKk",   # 3Blue1Brown — Neural networks
        "https://youtu.be/kCc8FmEb1nY",                  # Andrej Karpathy — GPT from scratch
    ]

    videos = scrape_youtube_videos(TEST_URLS)
    for vid in videos:
        print("\n" + "=" * 60)
        print(f"URL          : {vid['source_url']}")
        print(f"Video ID     : {vid['_video_id']}")
        print(f"Title        : {vid['_title']}")
        print(f"Channel      : {vid['author']}")
        print(f"Published    : {vid['published_date']}")
        print(f"Language     : {vid['language']}")
        print(f"Transcript   : {'✓ ' + str(len(vid['_raw_text'])) + ' chars' if vid['_raw_text'] else '✗ unavailable'}")
        print(f"Schema keys  : {[k for k in vid if not k.startswith('_')]}")
