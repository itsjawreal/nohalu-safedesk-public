"""Unit tests for document ingestion (PDF/DOCX/CSV/MD/TXT).

Run: python tests/test_ingest.py
Requires: pypdf, python-docx, reportlab (test-only fixture).
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import ingest


def test_csv():
    data = b"topik,nilai\nTunjangan makan siang,Rp200.000\n"
    out = ingest.parse_file("x.csv", data)
    assert out["source_type"] == "csv"
    assert "Tunjangan makan siang" in out["text"] and "Rp200.000" in out["text"]
    print("  ok csv")


def test_markdown():
    out = ingest.parse_file("x.md", b"# Kebijakan\nCuti 12 hari.\n")
    assert out["source_type"] == "markdown"
    assert "Cuti 12 hari" in out["text"]
    print("  ok markdown")


def test_txt():
    out = ingest.parse_file("x.txt", b"Kebijakan parkir: gratis.")
    assert out["source_type"] == "text"
    assert "parkir" in out["text"]
    print("  ok txt")


def test_docx():
    import docx
    buf = io.BytesIO()
    d = docx.Document()
    d.add_paragraph("Kebijakan cuti tahunan 12 hari.")
    d.save(buf)
    out = ingest.parse_file("x.docx", buf.getvalue())
    assert out["source_type"] == "docx"
    assert "12 hari" in out["text"]
    print("  ok docx")


def test_pdf():
    try:
        from reportlab.pdfgen import canvas
    except ImportError:
        print("  skip pdf (reportlab not installed)")
        return
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, "Tunjangan makan siang Rp200.000")
    c.save()
    out = ingest.parse_file("x.pdf", buf.getvalue())
    assert out["source_type"] == "pdf"
    assert "200.000" in out["text"]
    print("  ok pdf")


def test_unsupported():
    try:
        ingest.parse_file("x.xlsx", b"PK\x03\x04")
    except ingest.IngestError as e:
        assert "tidak didukung" in str(e).lower()
        print("  ok unsupported")
        return
    raise AssertionError("xlsx should be rejected")


def test_empty():
    try:
        ingest.parse_file("x.txt", b"")
    except ingest.IngestError:
        print("  ok empty")
        return
    raise AssertionError("empty file should be rejected")


def test_blank_pdf():
    try:
        from pypdf import PdfWriter
    except ImportError:
        print("  skip blank_pdf")
        return
    buf = io.BytesIO()
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    w.write(buf)
    try:
        ingest.parse_file("blank.pdf", buf.getvalue())
    except ingest.IngestError as e:
        assert "tidak ada teks" in str(e).lower()
        print("  ok blank_pdf")
        return
    raise AssertionError("blank pdf should be rejected")


def _run_all():
    test_csv()
    test_markdown()
    test_txt()
    test_docx()
    test_pdf()
    test_unsupported()
    test_empty()
    test_blank_pdf()
    print("\nALL INGEST TESTS PASS")


if __name__ == "__main__":
    _run_all()
