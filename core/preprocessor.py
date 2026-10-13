"""
preprocessor.py
---------------
Handles text cleaning and normalization for the search pipeline.

Intentionally does NOT remove stopwords — preserving them ensures
that semantic similarity models (e.g. Sentence Transformers) receive
full, grammatically intact sentences for better embedding quality.
"""

import re
import unicodedata
from typing import List, Dict


def clean_text(text: str) -> str:
    """
    Clean and normalize a raw text string.

    Operations performed (in order):
        1. Unicode normalization (NFKC) — converts ligatures, full-width chars, etc.
        2. Remove null bytes and non-printable control characters.
        3. Replace hyphenated line-breaks (soft-wrapped words) with joined words.
        4. Collapse multiple consecutive newlines into a single newline.
        5. Replace tabs and other whitespace variants with a single space.
        6. Strip leading/trailing whitespace from each line.
        7. Collapse multiple consecutive spaces into one.
        8. Strip the final string.

    Note:
        Stopwords are intentionally preserved for semantic embedding quality.

    Args:
        text (str): Raw text string to clean.

    Returns:
        str: Cleaned, normalized text string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    # 1. Normalize unicode (NFKC handles ligatures like ﬁ → fi, fullwidth chars, etc.)
    text = unicodedata.normalize("NFKC", text)

    # 2. Remove null bytes and non-printable control chars (except \n and space)
    #    Keeps standard printable ASCII + extended Latin characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 3. Rejoin soft-hyphenated line breaks (e.g. "impor-\ntant" → "important")
    text = re.sub(r"-\n", "", text)

    # 4. Collapse multiple consecutive blank lines into a single newline
    text = re.sub(r"\n{2,}", "\n", text)

    # 5. Replace tabs and exotic whitespace with a single space
    text = re.sub(r"[ \t\r\f\v]+", " ", text)

    # 6. Strip leading/trailing whitespace from every individual line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(line for line in lines if line)  # drop empty lines

    # 7. Collapse any leftover multiple spaces within a line
    text = re.sub(r" {2,}", " ", text)

    # 8. Final strip
    return text.strip()


def clean_pages(pages_data: List[Dict]) -> List[Dict]:
    """
    Apply clean_text() to a list of page dicts produced by extractor.py.

    Args:
        pages_data (List[Dict]): Each dict must have at least a "text" key.

    Returns:
        List[Dict]: Same structure with "text" values cleaned in-place (new list).
    """
    cleaned = []
    for entry in pages_data:
        cleaned_entry = dict(entry)  # shallow copy to avoid mutating the original
        cleaned_entry["text"] = clean_text(entry["text"])
        cleaned.append(cleaned_entry)
    return cleaned


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "This   is  a  test.\n\n\nSome  extra   spaces.\n"
        "A hyphen-\nated word.\tTabbed content.\n"
    )
    print("=== Original ===")
    print(repr(sample))
    print("\n=== Cleaned ===")
    print(repr(clean_text(sample)))
