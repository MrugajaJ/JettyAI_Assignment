"""
utils/chunking.py
-----------------
Splits long article text into overlapping-free, semantically coherent chunks
suitable for downstream embedding or trust scoring.

Chunking strategy (in order):
  1. Split on double newlines (paragraph boundaries).
  2. If a paragraph still exceeds `max_words`, split further on sentence
     boundaries using a regex that matches  .  !  ?  followed by whitespace.
  3. If a sentence fragment still exceeds `max_words`, hard-split on word count
     as a last resort so no chunk ever escapes the limit.

Edge cases handled:
  - Empty / whitespace-only input  →  []
  - Text shorter than 50 words     →  [text]  (single-item list, no split)
"""

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Threshold below which we never bother splitting
_SHORT_TEXT_WORD_THRESHOLD = 50

# Texts longer than this are pre-split into fixed-word sections before
# paragraph/sentence processing. Prevents O(n²) behaviour on very long strings
# and ensures no single processing pass ever exceeds memory limits.
_LARGE_TEXT_BATCH_WORDS = 2000

# Sentence-boundary regex: split AFTER  .  !  ?  when followed by whitespace
# or end-of-string. Keeps the punctuation attached to the preceding sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    """Return the number of whitespace-separated tokens in `text`."""
    return len(text.split())


def _split_into_sentences(paragraph: str) -> list[str]:
    """
    Split a paragraph into sentences using punctuation boundaries.
    Returns at least one element (the original paragraph) even if no
    sentence boundary is found.
    """
    sentences = _SENTENCE_SPLIT_RE.split(paragraph.strip())
    return [s.strip() for s in sentences if s.strip()]


def _hard_split(text: str, max_words: int) -> list[str]:
    """
    Last-resort word-count split when neither paragraph nor sentence
    boundaries keep the chunk within `max_words`.

    Slices the token list into windows of exactly `max_words` words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        if chunk:
            chunks.append(chunk)
    return chunks


def _split_paragraph(paragraph: str, max_words: int) -> list[str]:
    """
    Try to keep `paragraph` as one chunk.
    If it exceeds `max_words`, split by sentences, then hard-split any
    sentence that is still too long.
    """
    if _word_count(paragraph) <= max_words:
        return [paragraph]

    # Split into sentences and regroup into chunks ≤ max_words
    sentences = _split_into_sentences(paragraph)
    chunks: list[str] = []
    current_words: list[str] = []

    for sentence in sentences:
        sentence_wc = _word_count(sentence)

        # If a single sentence already exceeds the limit, hard-split it
        if sentence_wc > max_words:
            # Flush what we have so far
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
            chunks.extend(_hard_split(sentence, max_words))
            continue

        # Would adding this sentence overflow the current chunk?
        if current_words and len(current_words) + sentence_wc > max_words:
            chunks.append(" ".join(current_words))
            current_words = sentence.split()
        else:
            current_words.extend(sentence.split())

    # Flush any remaining words
    if current_words:
        chunks.append(" ".join(current_words))

    return chunks



def _chunk_single_pass(text: str, max_words: int) -> list[str]:
    """
    Core paragraph → sentence → word chunking logic for a single text block.
    Called by chunk_content for both normal-length and batched sections.
    Assumes `text` is already stripped and non-empty.
    """
    # Split on paragraph boundaries (double newlines)
    raw_paragraphs = re.split(r"\n{2,}", text)

    # Collapse internal newlines / extra whitespace inside each paragraph
    paragraphs = []
    for para in raw_paragraphs:
        cleaned = " ".join(para.split())
        if cleaned:
            paragraphs.append(cleaned)

    # Split oversized paragraphs by sentence, then by word count
    chunks: list[str] = []
    for para in paragraphs:
        chunks.extend(_split_paragraph(para, max_words))

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_content(text: str, max_words: int = 150) -> list[str]:
    """
    Split `text` into a list of non-empty string chunks, each containing
    at most `max_words` words.

    Parameters
    ----------
    text : str
        The raw article / transcript text to chunk.
    max_words : int, optional
        Maximum number of words per chunk (default 150).

    Returns
    -------
    list[str]
        Ordered list of text chunks.
        - Returns []       for empty / whitespace-only input.
        - Returns [text]   if the full text is shorter than 50 words.
        - For texts > 2000 words, processes in 2000-word batches so no
          single pass ever operates on an unbounded string.
        - Otherwise splits on paragraphs → sentences → words as needed.
    """
    # --- Edge case 1: empty input --------------------------------------------
    if not text or not text.strip():
        return []

    # Normalise: collapse excessive blank lines, strip leading/trailing space
    text = text.strip()

    # --- Edge case 2: very short text ----------------------------------------
    if _word_count(text) < _SHORT_TEXT_WORD_THRESHOLD:
        return [text]

    # --- Edge case 3: very long text — pre-split into batches ----------------
    # This prevents any single regex or string operation from running over a
    # 50,000-word transcript in one shot.
    total_words = _word_count(text)
    if total_words > _LARGE_TEXT_BATCH_WORDS:
        words = text.split()
        batches = [
            " ".join(words[i : i + _LARGE_TEXT_BATCH_WORDS])
            for i in range(0, len(words), _LARGE_TEXT_BATCH_WORDS)
        ]
        final_chunks: list[str] = []
        for batch in batches:
            final_chunks.extend(_chunk_single_pass(batch, max_words))
        return [c for c in final_chunks if c.strip()]

    # --- Normal path ---------------------------------------------------------
    return _chunk_single_pass(text, max_words)


# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the
    natural intelligence displayed by animals including humans.

    AI research has been defined as the field of study of intelligent agents, which refers to
    any system that perceives its environment and takes actions that maximize its chance of
    achieving its goals. The term artificial intelligence had previously been used to describe
    machines that mimic and display human cognitive skills associated with the human mind,
    such as learning and problem-solving.

    Machine learning (ML) is a type of AI that allows software applications to become more
    accurate at predicting outcomes without being explicitly programmed to do so. Machine
    learning algorithms use historical data as input to predict new output values. This
    approach is increasingly becoming important to data scientists. Supervised learning,
    unsupervised learning, and reinforcement learning are the most common types.

    Deep learning is part of a broader family of machine learning methods based on artificial
    neural networks with representation learning. Learning can be supervised, semi-supervised,
    or unsupervised. Deep-learning architectures such as deep neural networks, recurrent neural
    networks, convolutional neural networks, and transformers have been applied to fields
    including computer vision, speech recognition, natural language processing, and more.
    """

    chunks = chunk_content(SAMPLE, max_words=50)
    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks, 1):
        wc = _word_count(chunk)
        print(f"[Chunk {i:02d} | {wc} words]\n{chunk}\n")

    # Edge case tests
    print("--- Edge cases ---")
    print("Empty input:", chunk_content(""))
    print("Whitespace: ", chunk_content("   \n\n  "))
    print("Short text: ", chunk_content("This is a very short text."))
