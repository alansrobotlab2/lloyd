#!/usr/bin/env python3
"""
Extract text from OCR PDF (Pioneer AI) using PyMuPDF (fitz).

Handles both text-based and OCR-embedded PDFs.
Saves extracted text to a .txt file with the same base name.

Usage:
    python pioneer_ocr.py <input.pdf> [output.txt]

Example:
    python pioneer_ocr.py PioneerAI.pdf
"""

import sys
import os
import fitz  # PyMuPDF


def extract_text_from_ocr_pdf(pdf_path: str, output_path: str = None) -> str:
    """
    Extract all text from a PDF, including OCR-embedded text layers.

    Args:
        pdf_path: Path to the input PDF file.
        output_path: Optional path for the output .txt file.
                     If None, defaults to <basename>.txt.

    Returns:
        The extracted text as a single string.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_path is None:
        base = os.path.splitext(pdf_path)[0]
        output_path = f"{base}.txt"

    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Primary text extraction
        text = page.get_text("text")

        # If page has no text, try OCR mode
        if not text.strip():
            print(f"  Page {page_num + 1}: No text layer found, attempting OCR...")
            # OCR extraction via pixmap + text extraction
            pix = page.get_pixmap(dpi=200)
            # PyMuPDF has built-in OCR via page.get_text("dict") with blocks
            # But for true OCR from images, you'd need an OCR engine.
            # For OCR-embedded PDFs (text in a hidden layer), try:
            text = page.get_text("xhtml")
            # Strip HTML tags if we got XHTML
            if "<" in text:
                import re
                text = re.sub(r"<[^>]+>", "", text)

        if text.strip():
            pages_text.append(f"--- Page {page_num + 1} ---\n{text.strip()}\n")
        else:
            pages_text.append(f"--- Page {page_num + 1} ---\n[No extractable text found]\n")

    doc.close()

    full_text = "\n\n".join(pages_text)

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Extracted {len(pages_text)} pages -> {output_path}")
    return full_text


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.pdf> [output.txt]")
        print()
        print("Extracts text from OCR PDFs using PyMuPDF (fitz).")
        print("If output.txt is not specified, defaults to <input>.txt")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        text = extract_text_from_ocr_pdf(pdf_path, output_path)
        print("\n--- Extracted Text Preview (first 500 chars) ---")
        print(text[:500])
        if len(text) > 500:
            print(f"... ({len(text)} total characters)")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()