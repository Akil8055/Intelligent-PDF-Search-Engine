"""
chunker.py
----------
Splits cleaned page-level text into overlapping chunks for embedding.

Uses a sliding window over word tokens so that:
  - Each chunk stays within the model's context window.
  - Overlap between consecutive chunks preserves semantic continuity
    across chunk boundaries (avoids cutting mid-thought).

Each output chunk records:
  - "text"   : the chunk's content.
  - "page"   : the originating page number (1-based).
  - "source" : the originating document/filename (optional metadata).
"""

from typing import List, Dict, Optional


def chunk_text(
    data: List[Dict],
    chunk_size: int = 200,
    overlap: int = 40,
    source: Optional[str] = None,
) -> List[Dict]:
    """
    Split a list of page dicts into overlapping word-level chunks.

    Strategy:
        For each page, tokenise the text into words and apply a sliding window
        of size `chunk_size` words, advancing by (chunk_size - overlap) words
        each step. This guarantees that every chunk has `overlap` words in
        common with its neighbours, preserving contextual continuity.

    Args:
        data (List[Dict]): List of page dicts, each with:
            - "text" (str)  : cleaned page text.
            - "page" (int)  : 1-based page number.
        chunk_size (int): Maximum number of words per chunk. Default: 200.
        overlap (int): Number of words shared between consecutive chunks.
                       Must be less than chunk_size. Default: 40.
        source (str, optional): Document identifier (e.g. filename) to embed
                                in each chunk's metadata. Default: None.

    Returns:
        List[Dict]: List of chunk dicts, each containing:
            - "text"   (str)       : The chunk's word content.
            - "page"   (int)       : Source page number.
            - "source" (str|None)  : Document identifier passed via `source`.

    Raises:
        ValueError: If overlap >= chunk_size (would create infinite loops).
        TypeError : If data items are missing required keys.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})."
        )

    chunks: List[Dict] = []
    step = chunk_size - overlap  # how many words to advance each iteration

    for entry in data:
        # Validate required keys
        if "text" not in entry or "page" not in entry:
            raise TypeError(
                f"Each page dict must contain 'text' and 'page' keys. Got: {list(entry.keys())}"
            )

        page_num: int = entry["page"]
        words: List[str] = entry["text"].split()

        # Edge case: page is shorter than one full chunk → emit as single chunk
        if len(words) <= chunk_size:
            if words:  # skip completely empty pages
                chunks.append({
                    "text": " ".join(words),
                    "page": page_num,
                    "source": source,
                })
            continue

        # Sliding window over word list
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]

            chunks.append({
                "text": " ".join(chunk_words),
                "page": page_num,
                "source": source,
            })

            # If we've already reached the last word, stop
            if end == len(words):
                break

            start += step

    return chunks


def chunk_pages(
    data: List[Dict],
    chunk_size: int = 200,
    overlap: int = 40,
    source: Optional[str] = None,
) -> List[Dict]:
    """
    Convenience alias for chunk_text() — provided for semantic clarity
    when calling from the main pipeline.

    See chunk_text() for full documentation.
    """
    return chunk_text(data, chunk_size=chunk_size, overlap=overlap, source=source)


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate two pages of extracted text
    sample_data = [
        {
            "text": " ".join([f"word{i}" for i in range(350)]),  # 350 words
            "page": 1,
        },
        {
            "text": " ".join([f"term{i}" for i in range(80)]),   # 80 words
            "page": 2,
        },
    ]

    results = chunk_text(sample_data, chunk_size=200, overlap=40, source="sample.pdf")

    print(f"Total chunks produced: {len(results)}")
    for idx, chunk in enumerate(results, 1):
        word_count = len(chunk["text"].split())
        print(f"  Chunk {idx:02d} | Page {chunk['page']} | Words: {word_count} | Source: {chunk['source']}")
