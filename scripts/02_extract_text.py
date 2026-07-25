"""
scripts/02_extract_text.py

Extracts raw text from every PDF in data/raw_pdfs/, saving one .txt
file per paper into data/raw_texts/.

Can be run directly, or imported:
    from scripts.extract_text import extract_text_from_pdf, extract_all
"""

from __future__ import annotations

import os

import fitz  # PyMuPDF

PDF_FOLDER_PATH = "../../data/raw_pdfs"
TEXT_FOLDER_PATH = "../../data/raw_texts"


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract and concatenate text from every page of a single PDF.

    Shared by both the batch script (paths) and the live upload
    pipeline (in-memory bytes) -- see extract_text_from_bytes below
    for the upload variant.
    """
    doc = fitz.open(pdf_path)
    full_paper_text = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(full_paper_text)


def extract_text_from_bytes(pdf_bytes: bytes) -> tuple[str, int]:
    """Same extraction, but for in-memory PDF bytes (user uploads).

    Returns (full_text, page_count).
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    full_paper_text = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(full_paper_text), page_count


def extract_all(pdf_folder: str = PDF_FOLDER_PATH, text_folder: str = TEXT_FOLDER_PATH) -> None:
    """Batch-process every PDF in pdf_folder into a matching .txt file."""
    os.makedirs(text_folder, exist_ok=True)

    for file_name in os.listdir(pdf_folder):
        if not file_name.endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_folder, file_name)
        text_file_name = file_name.replace(".pdf", ".txt")
        text_file_path = os.path.join(text_folder, text_file_name)

        if os.path.exists(text_file_path):
            print(f"Skipping: {file_name} (already extracted)")
            continue

        try:
            raw_text = extract_text_from_pdf(pdf_path)
            with open(text_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(raw_text)
            print(f"Successfully processed and saved: {text_file_name}")
        except Exception as e:
            print(f"Severe error extracting text from {file_name}: {e}")


if __name__ == "__main__":
    extract_all()