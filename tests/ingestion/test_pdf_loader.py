from pathlib import Path
from reportlab.pdfgen import canvas
from app.ingestion.pdf_loader import load_pdf_text


def test_pdf_loading(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    c = canvas.Canvas(str(pdf_path))
    c.drawString(100, 750, "Hello PDF")
    c.save()

    text = load_pdf_text(pdf_path)

    assert "Hello PDF" in text
