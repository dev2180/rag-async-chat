"""
MODULE: app/ingestion/pdf_loader.py

Extracts text from PDF files.
"""

from pathlib import Path
from pypdf import PdfReader


def load_pdf_text(file_path: Path) -> str:
    reader = PdfReader(str(file_path))
    text_parts = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)

    full_text = "\n".join(text_parts)

    # Basic cleaning
    full_text = full_text.replace("\n\n", "\n")
    full_text = full_text.replace("  ", " ")

    return full_text.strip()
