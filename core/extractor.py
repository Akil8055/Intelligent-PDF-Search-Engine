"""
extractor.py
------------
Handles PDF text extraction using PyMuPDF (fitz).

Each page is extracted independently and returned as a structured dict,
enabling page-level tracking throughout the search pipeline.
"""

import fitz  # PyMuPDF
import os
from typing import List, Dict


def extract_text_with_pages(pdf_path: str) -> List[Dict]:
    """
    Extract text from a PDF file, page by page.

    Args:
        pdf_path (str): Absolute or relative path to the PDF file.

    Returns:
        List[Dict]: A list of dicts, each with:
            - "text" (str): Extracted text from the page.
            - "page" (int): 1-based page number.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the file is not a valid PDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.lower().endswith(".pdf"):
        raise ValueError(f"Expected a .pdf file, got: {pdf_path}")

    pages_data: List[Dict] = []

    # Open the PDF document
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Failed to open PDF '{pdf_path}': {e}")

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]

            # Extract plain text from the page
            text = page.get_text("text")

            # Only include pages that contain actual text content
            if text and text.strip():
                pages_data.append({
                    "text": text,
                    "page": page_index + 1  # Convert to 1-based page numbering
                })
    finally:
        doc.close()

    return pages_data


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    results = extract_text_with_pages(path)
    print(f"Extracted {len(results)} page(s) from '{path}'")
    for r in results[:2]:  # Preview first 2 pages
        preview = r["text"][:200].replace("\n", " ")
        print(f"  Page {r['page']}: {preview}...")
