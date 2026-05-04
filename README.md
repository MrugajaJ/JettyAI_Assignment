# Web Scraping & Trust Scoring System

A modular Python pipeline that scrapes articles from **blogs**, **YouTube videos**, and **PubMed research papers**, enriches each piece of content with language detection, keyword tagging, and content chunking, then assigns a composite **trust score** using a multi-factor weighted formula.

---

## Folder Structure

```
Jetty_Assignment/
├── main.py                    # Pipeline orchestrator
├── requirements.txt
│
├── scraper/
│   ├── __init__.py
│   ├── blog_scraper.py        # newspaper3k + BeautifulSoup4 fallback
│   ├── youtube_scraper.py     # YouTube Data API v3 + transcript-api
│   └── pubmed_scraper.py      # NCBI E-utilities (esearch + efetch)
│
├── scoring/
│   ├── __init__.py
│   └── trust_score.py         # 5-factor scorer + abuse penalties + OPR lookup
│
├── utils/
│   ├── __init__.py
│   ├── chunking.py            # Paragraph → sentence → word chunker
│   ├── lang_detect.py         # langdetect wrapper
│   └── tagging.py             # YAKE keyword extractor + frequency fallback
│
└── output/
    ├── __init__.py
    ├── blogs.json
    ├── youtube.json
    ├── pubmed.json
    └── scraped_data.json      # All records combined
```

---

## Installation

```bash
# 1. Clone / download the project
cd Jetty_Assignment

# 2. (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install all dependencies
pip install -r requirements.txt
```

### `requirements.txt` contents

```
requests
beautifulsoup4
newspaper3k
youtube-transcript-api
google-api-python-client
langdetect
keybert
yake
httpx
```

---

## Configuration

Two environment variables must be set before running. You can now use the provided `.env` file in the root directory for easier configuration.

1.  Open the `.env` file in the root directory.
2.  Replace the placeholders with your actual API keys:

```env
YOUTUBE_API_KEY=your_youtube_api_key_here
OPR_API_KEY=your_opr_api_key_here
```

| Variable | Required | Purpose |
|---|---|---|
| `YOUTUBE_API_KEY` | Yes (for YT metadata) | YouTube Data API v3 key — get one free at [console.cloud.google.com](https://console.cloud.google.com) |
| `OPR_API_KEY` | Yes (for domain trust) | Open PageRank API key — get one free at [openpagerank.com](https://openpagerank.com) |

The pipeline degrades gracefully when these are absent (metadata / DA scores will use safe defaults).

---

## How to Run

```bash
python main.py
```

This single command runs the full pipeline: scrape → chunk → tag → score → save.  
Results are written to the `output/` directory and a summary table is printed to the console.

---

## Tools & Libraries Used

| Library | Version* | Purpose |
|---|---|---|
| `newspaper3k` | ≥ 0.2.8 | Primary blog article extractor (title, author, date, body) |
| `beautifulsoup4` | ≥ 4.12 | Fallback HTML parser; targets `<article>`, `<main>`, largest `<div>` |
| `httpx` | ≥ 0.27 | Async-capable HTTP client for all outbound requests |
| `google-api-python-client` | ≥ 2.0 | YouTube Data API v3 — fetches video snippet metadata |
| `youtube-transcript-api` | ≥ 0.6 | Downloads YouTube transcripts (manual + auto-generated) |
| `langdetect` | ≥ 1.0.9 | ISO 639-1 language detection from article text |
| `yake` | ≥ 0.4.8 | Unsupervised keyword extraction (primary tagging method) |
| `keybert` | ≥ 0.8 | Semantic keyword extraction (available for future use) |
| `xml.etree.ElementTree` | stdlib | Parses NCBI efetch XML responses — no extra install needed |

*Pin exact versions in production with `pip freeze > requirements.txt`*

---

## Scraping Approach

### Blogs
Blog posts are scraped using **newspaper3k** as the primary extractor, which handles article parsing, author detection, and date extraction out of the box. If newspaper3k returns empty body text or throws an exception, the pipeline falls back to **BeautifulSoup4**, which searches for content in this priority order: `<article>` → `<main>` → largest `<div>` by character count. Noise elements (`<nav>`, `<footer>`, ads, sidebars) are stripped using a tag-name and keyword blocklist before text extraction.

### YouTube
YouTube video metadata (channel title, publish date, description) is fetched via the **YouTube Data API v3** using an API key stored in `YOUTUBE_API_KEY`. Transcripts are retrieved via **youtube-transcript-api**, preferring manually created English transcripts and falling back to auto-generated captions. If no transcript is available, `transcript_available` is set to `false` and the video **description** is used as the text source for chunking and tagging.

### PubMed
PubMed articles are accessed through the **NCBI E-utilities API** (no API key required). The pipeline first calls `esearch.fcgi` to resolve any URL, DOI, or search term to a PMID, then calls `efetch.fcgi` with `retmode=xml` to retrieve the full article record. The XML response is parsed with Python's **stdlib `xml.etree.ElementTree`** — extracting title, all authors, journal name, publication year, and abstract text (including multi-section structured abstracts).

---

## Trust Score Design

### Formula

```
trust_score = (0.25 × author_score)
            + (0.20 × citation_score)
            + (0.25 × domain_authority_score)
            + (0.20 × recency_score)
            + (0.10 × medical_disclaimer_score)
```

Then `apply_abuse_penalties()` applies post-hoc deductions.

### Factor Descriptions

| Factor | Weight | Logic |
|---|---|---|
| **author_score** | 0.25 | `1.0` institutional name · `0.60` personal name · `0.30` null |
| **citation_score** | 0.20 | `1.0` PubMed · `0.50` YouTube · `0.30` blog (source-type proxy) |
| **domain_authority_score** | 0.25 | Open PageRank score ÷ 10; default `0.40` when unavailable |
| **recency_score** | 0.20 | `1.0` ≤1yr · `0.70` 1–3yr · `0.40` 3–5yr · `0.10` >5yr · `0.20` null |
| **medical_disclaimer_score** | 0.10 | `1.0` disclaimer present · `0.40` missing on medical blog · `0.80` N/A |

### Abuse Penalties (applied after base score)

| Rule | Deduction | Trigger |
|---|---|---|
| Fake/generic author | −0.15 | Author token matches: `admin`, `editor`, `webmaster`, etc. |
| SEO spam domain | −0.20 | Domain Authority < 2.0 |
| Missing medical disclaimer | −0.20 | Health tag on blog with no disclaimer phrase |
| Outdated content | −0.15 | Published more than 5 years ago |

Final score is clamped to **[0.0, 1.0]** and rounded to 2 decimal places.

---

## Edge Case Handling

- **Missing author** — set to `null`; `author_score` defaults to `0.30`
- **Missing published date** — set to `null`; `recency_score` defaults to `0.20`
- **YouTube transcript unavailable** — `transcript_available: false`; description used for chunking/tagging
- **PubMed multiple authors** — joined as a single comma-separated string
- **Non-English content** — detected language stored in `language` field; `WARNING` logged; content still processed fully
- **Very long articles (> 2000 words)** — `chunk_content` pre-splits into 2000-word batches before paragraph/sentence splitting
- **newspaper3k failure** — falls back to BeautifulSoup4 automatically
- **Missing API keys** — pipeline logs a warning and uses safe defaults (`domain_authority = 5.0`, transcript skipped)
- **NCBI rate limits** — 400 ms polite delay between every E-utilities request

---

## Limitations

1. **No real citation counts** — `citation_score` is a static proxy based on source type. Actual citation data requires a separate API call (e.g. Semantic Scholar, CrossRef).
2. **English-only transcript preference** — the transcript fetcher only looks for English tracks; non-English video transcripts are discarded in favor of the description fallback.
3. **YAKE language assumption** — YAKE is initialised with `language='en'`. For non-English articles it will still run but produce lower-quality tags. A language-aware initialisation would require detecting language first.
4. **Domain authority caching** — `get_domain_authority()` makes a live HTTP request per article. In bulk runs this is slow and may exhaust the free OPR API quota. A local cache or batch endpoint would be more efficient.
5. **Static institutional name list** — `author_score` uses a hardcoded set of known institution keywords. New or obscure institutions will be scored as personal authors.
6. **Paywalled content** — newspaper3k and BeautifulSoup4 cannot access paywalled articles; they will return empty body text and fall through to schema defaults.

---

## Example Output

```json
{
  "source_url": "https://pubmed.ncbi.nlm.nih.gov/38505432/",
  "source_type": "pubmed",
  "author": "Doe J, Smith A, Patel R",
  "published_date": "2024",
  "language": "en",
  "region": null,
  "topic_tags": ["vaccine efficacy", "clinical trial", "immunogenicity", "rna vaccine", "adverse events", "dose response"],
  "trust_score": 0.81,
  "content_chunks": [
    "Background: This randomised controlled trial evaluated the immunogenicity of a novel mRNA vaccine in adults aged 18–65.",
    "Methods: 640 participants were enrolled across four sites. Primary endpoint was seroconversion rate at day 28.",
    "Results: Seroconversion was achieved in 94.3% of the intervention group versus 12.1% of placebo. Adverse events were mild and transient.",
    "Conclusions: The vaccine demonstrated strong immunogenic response with an acceptable safety profile, supporting advancement to Phase III trials."
  ]
}
```
