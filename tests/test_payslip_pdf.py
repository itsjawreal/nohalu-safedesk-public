"""P1 — PDF slip gaji: generator tanpa dependensi + backfill seed.

Dua hal yang dijaga:

1. `payslip_pdf.build` menghasilkan PDF yang benar-benar bisa dibaca ulang
   (bukan hanya diawali `%PDF`). Escaping `(`/`)`/`\\` adalah bagian yang paling
   mudah rusak — nama seperti "Budi (Santoso)" akan membuat struktur PDF tidak
   valid kalau tidak di-escape.
2. `seed_personal(db_path=...)` mengisi `file_ref` untuk slip lama yang belum
   punya berkas, dan **idempotent** (putaran kedua tidak membuat apa pun).
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "pdf.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

import io  # noqa: E402

import db  # noqa: E402
import files as files_mod  # noqa: E402
import payslip_pdf  # noqa: E402
from pypdf import PdfReader  # noqa: E402


def _text(pdf_bytes):
    return PdfReader(io.BytesIO(pdf_bytes)).pages[0].extract_text()


def test_build_returns_readable_single_page():
    pdf = payslip_pdf.build({
        "user_name": "Sari Dewi", "user_id": "u_sari", "period": "2026-09",
        "gross": 19500000, "deductions": 2500000, "net": 17000000})
    assert pdf.startswith(b"%PDF-1.4")
    assert pdf.rstrip().endswith(b"%%EOF")
    text = _text(pdf)
    assert "Sari Dewi" in text
    assert "2026-09" in text
    assert "Rp17.000.000" in text


def test_special_characters_are_escaped():
    """Nama dengan tanda kurung/backslash tidak boleh merusak struktur PDF."""
    pdf = payslip_pdf.build({
        "user_name": "Budi (Santoso) \\ O'Brien", "user_id": "u_budi",
        "period": "2026-01", "gross": 100, "deductions": 0, "net": 100})
    assert "Budi (Santoso)" in _text(pdf)


def test_missing_fields_do_not_crash():
    pdf = payslip_pdf.build({"period": "2026-01"})
    text = _text(pdf)
    assert "2026-01" in text
    assert "Rp0" in text


def test_seed_backfills_missing_files_and_is_idempotent():
    conn = db.connect(os.environ["NOHALU_DB_PATH"])
    db.init_db(conn)
    db.seed_if_empty(conn)
    db.ensure_seed_users(conn)
    from seed import seed_personal

    first = seed_personal(conn, db_path=os.environ["NOHALU_DB_PATH"])
    assert first["payslips"] >= 1
    assert first["payslip_files"] == first["payslips"]

    missing = conn.execute(
        "SELECT COUNT(*) AS n FROM payslips WHERE file_ref IS NULL OR file_ref = ''"
    ).fetchone()["n"]
    assert missing == 0, "setiap slip seed harus punya berkas"

    second = seed_personal(conn, db_path=os.environ["NOHALU_DB_PATH"])
    assert second["payslip_files"] == 0, "putaran kedua tidak membuat berkas baru"
    assert second["payslips"] == 0


def test_backfilled_file_resolves_on_disk():
    conn = db.connect(os.environ["NOHALU_DB_PATH"])
    row = conn.execute(
        "SELECT file_ref FROM payslips WHERE file_ref IS NOT NULL LIMIT 1").fetchone()
    assert row and row["file_ref"]
    path = files_mod.resolve(os.environ["NOHALU_DB_PATH"], row["file_ref"])
    assert path and os.path.isfile(path)
    assert files_mod.content_type(row["file_ref"]) == "application/pdf"
