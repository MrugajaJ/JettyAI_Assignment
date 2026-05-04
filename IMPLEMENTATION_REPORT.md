# Data Scraping & Trust Scoring: Implementation Report

---

## 1. Scraping Strategy

The pipeline targets three distinct source types — blogs, YouTube videos, and PubMed research articles — each requiring a different extraction strategy.

**Blog scraping** uses **newspaper3k** as the primary extractor. Newspaper3k is purpose-built for article parsing: it resolves redirects, strips boilerplate, and heuristically identifies title, author, publication date, and body text from arbitrary CMS layouts. When newspaper3k fails — due to JavaScript-rendered pages, paywalls, or malformed HTML — the pipeline falls back to **BeautifulSoup4**. The fallback targets semantic container elements in priority order: `<article>` → `<main>` → the largest `<div>` block by character count. Before extraction, a blocklist of tag names (`nav`, `footer`, `script`, `aside`) and CSS class/id substrings (`"ads"`, `"sidebar"`, `"cookie"`, `"newsletter"`) is used to strip navigational chrome and advertising content, ensuring only editorial body text is captured. **httpx** is used for all HTTP requests due to its support for redirects, configurable timeouts, and a modern async-compatible interface.

**YouTube scraping** is split into two independent API calls. The **YouTube Data API v3** (via `google-api-python-client`) provides structured metadata — channel title, publication date, and description — which are more reliable than anything that could be scraped from the rendered HTML page. Transcript text is fetched separately using **youtube-transcript-api**, which exposes both manually created and auto-generated caption tracks. The extractor prefers manually created English transcripts (`en`, `en-US`, `en-GB`) and falls back to auto-generated captions. If no English transcript exists at all, the field `transcript_available` is set to `false` and the video description is substituted as the text corpus for downstream processing.

**PubMed scraping** is performed entirely through the **NCBI E-utilities REST API**, which is free and does not require an API key. The pipeline first calls `esearch.fcgi` to normalise any input form — full URL, bare PMID, DOI string, or search term — into a canonical PMID. It then calls `efetch.fcgi` with `db=pubmed&rettype=abstract&retmode=xml` to retrieve the structured XML record. Parsing is done with Python's **stdlib `xml.etree.ElementTree`** — avoiding an external dependency — and handles multi-section structured abstracts (e.g. `BACKGROUND`, `METHODS`, `RESULTS`), mixed-content XML tags inside abstract text, collective/group author names, and the `MedlineDate` format used when a full calendar date is unavailable.

---

## 2. Topic Tagging Method

**Why YAKE over TF-IDF:** TF-IDF requires a reference corpus to compute inverse document frequency; without a pre-built corpus of domain-specific documents, its rankings are unreliable for single-document tagging. YAKE (Yet Another Keyword Extractor) is an *unsupervised* statistical method that works on a single document alone, using features such as term frequency, co-occurrence patterns, positional bias (terms near the beginning of a document tend to be more salient), and sentence-level spread. This makes it immediately deployable without any training data.

**N-gram size tuning:** The extractor is configured with `max_ngram_size=2`, meaning it considers both unigrams (`"vaccine"`) and bigrams (`"vaccine efficacy"`). Bigrams are important for preserving domain-specific compound concepts that lose meaning when split. A size of 3 was tested but produced too many fragmented trigrams from abstract text; 2 gave the best signal-to-noise ratio. The `deduplication_threshold=0.9` parameter with the `seqm` (sequence matcher) algorithm suppresses near-duplicate phrases such as `"clinical trial"` and `"clinical trials"` appearing as separate tags.

**Stop-word filtering:** After YAKE scores keywords, a post-processing step applies a curated stop-word set to remove two classes of noise. The first class is standard English function words (`"the"`, `"with"`, `"however"`). The second is domain-noise words that are statistically frequent in scientific and blog text but carry no topical signal — words like `"study"`, `"results"`, `"paper"`, `"show"`, and `"based"`. Removing the second class is what distinguishes a useful tag list (`["mRNA vaccine", "immunogenicity", "dose response"]`) from a noisy one (`["study shows", "results based", "paper presents"]`). If YAKE is unavailable or the text is too short (< 80 characters), the pipeline falls back to a simple word-frequency count applying the same stop-word set, ensuring the tagging step never fails silently.

---

## 3. Trust Score Algorithm

The trust score is a **weighted linear combination** of five independent sub-scores, each normalised to [0, 1]:

```
trust_score = 0.25 × author_score
            + 0.20 × citation_score
            + 0.25 × domain_authority_score
            + 0.20 × recency_score
            + 0.10 × medical_disclaimer_score
```

**author_score (weight 0.25):** Captures source credibility at the author level. An institutional or organisational author (matched against a curated list including `"NIH"`, `"Nature"`, `"university"`, `"hospital"`) receives `1.0` because institutions carry editorial accountability. A named personal author receives `0.60` — credible but unverified. A null author receives `0.30`, reflecting significant uncertainty rather than a zero to avoid over-penalising legitimate anonymous publications. The 0.25 weight reflects that author identity is the single most important trust signal available at scrape time.

**citation_score (weight 0.20):** Currently a source-type proxy: PubMed articles score `1.0` (peer-reviewed), YouTube `0.50` (user-generated but structured), and blogs `0.30` (self-published). Real citation counts would require a secondary API call to Semantic Scholar or CrossRef; the proxy is documented as a placeholder.

**domain_authority_score (weight 0.25):** The Open PageRank score (0–10) is divided by 10 to normalise it to [0, 1]. OPR measures the quality and quantity of inbound links to a domain — a proxy for real-world credibility accumulated over time. When the OPR API is unavailable, the score defaults to `0.40` (a conservative mid-low value that neither rewards nor catastrophically penalises an untested domain). The 0.25 weight matches author\_score because domain reputation is equally strong a signal.

**recency_score (weight 0.20):** Content older than 5 years in fast-moving fields (medicine, AI, policy) can be actively misleading. Scores are bucketed: ≤1 year → `1.0`, 1–3 years → `0.70`, 3–5 years → `0.40`, >5 years → `0.10`, missing date → `0.20`. The missing-date default of `0.20` is slightly higher than the >5-year bucket because the absence of a date is not the same as proven staleness.

**medical_disclaimer_score (weight 0.10):** Applied only to medical source types (PubMed, health blogs). A disclaimer phrase (`"consult a doctor"`, `"for informational purposes only"`) signals editorial responsibility. Its weight is intentionally low (0.10) because most legitimate medical sources include disclaimers and the factor would otherwise compress score variance.

**Anti-gaming properties:** Because the formula uses five orthogonal signals, a bad actor cannot game the score by optimising a single dimension. A spam blog could, for instance, add a medical disclaimer and claim an institutional author, but it cannot manufacture a high domain authority, recent publication date, or peer-review status simultaneously. The abuse penalty layer adds a second defence: generic author names, low domain authority (< 2.0), missing health disclaimers, and provably outdated content each carry explicit deductions of 0.15–0.20 that are applied *after* the base score is computed.

---

## 4. Edge Case Handling

**Missing metadata:** Every schema field has an explicitly defined fallback. `author` and `published_date` default to `null` (Python `None`) rather than empty strings, so downstream consumers can reliably distinguish "data was not found" from "the field is genuinely empty." The scoring functions treat `null` author as `author_score = 0.30` and `null` date as `recency_score = 0.20` — both penalising but not catastrophic values, reflecting uncertainty rather than confirmed absence of quality.

**Non-English content:** The `detect_language()` wrapper returns an ISO 639-1 code regardless of language. When a non-English language is detected, a `WARNING`-level log message is emitted — `"non-English content detected (lang='de')"` — and processing continues identically. The language code is stored in the `language` field of the output schema, allowing consumers to filter or weight records accordingly without the pipeline itself discarding content.

**Transcript failures:** YouTube transcript unavailability is handled through a multi-tier fallback and an explicit schema flag. `TranscriptsDisabled`, `NoTranscriptFound`, `VideoUnavailable`, and unexpected exceptions are each caught individually with specific log messages. When any failure occurs, `transcript_available` is set to `false` and the video's textual description is substituted as `_raw_text`, so that chunking, tagging, and language detection still have material to operate on rather than receiving an empty string.

---

## 5. Abuse Prevention Logic

The `apply_abuse_penalties()` function is a **post-hoc deduction layer** that runs after the base trust score is computed. It is intentionally separated from the main formula so that its rules can be audited, adjusted, or disabled independently.

**Fake/generic author (−0.15):** A blocklist of generic author tokens — `"admin"`, `"editor"`, `"webmaster"`, `"anonymous"`, `"blogger"`, `"staff"` — is checked against each whitespace-separated token in the author string. Token-level matching (rather than whole-string) catches compound values like `"admin team"` or `"site editor"`. This pattern is strongly correlated with low-quality content farms that auto-publish articles without editorial ownership.

**SEO spam domain (−0.20):** Domains with an Open PageRank score below 2.0 are treated as potential spam domains. A domain authority this low indicates that the domain has almost no legitimate inbound links — a hallmark of newly registered spam sites, link farms, or scraped content mirrors. The penalty of 0.20 is the largest single deduction, reflecting that domain reputation is a strong trust signal. The rule is only applied when a DA value is available; it is skipped silently when `domain_authority=None` to avoid penalising legitimate content from domains not yet in the OPR index.

**Missing medical disclaimer on health content (−0.20):** Health-related topic tags (`"health"`, `"medical"`, `"disease"`, `"drug"`, `"treatment"`) combined with `source_type == "blog"` and no detected disclaimer phrase constitutes a high-risk pattern. Consumer health blogs that make medical claims without responsible caveats pose genuine public harm risk. The combined text of `_raw_text`, `_title`, and `_description` is searched for any phrase from the disclaimer list; the penalty is applied only if all three conditions are simultaneously true, avoiding false positives on general wellness content.

**Outdated content (−0.15):** While the base `recency_score` already penalises old content, the abuse layer adds an additional deduction for content that is more than 5 years old. This double penalty is intentional: in fast-moving domains, outdated content is not merely less relevant but potentially actively misleading — outdated medical dosages, superseded AI benchmarks, or obsolete legal guidance. The penalty is only applied when a parseable year is available, so it does not compound the existing `null`-date penalty.
