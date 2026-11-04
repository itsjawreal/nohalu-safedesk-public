"""Regresi: hasil edit admin tidak boleh hilang saat server restart.

Bug produksi: `ensure_seed_docs()` membandingkan isi dokumen dengan seed dan
menimpanya bila berbeda. Karena edit admin membuat isi != seed, setiap restart
mengembalikan SEMUA dokumen ke versi seed ("updated 61 seed doc(s)"), sehingga
perubahan admin hilang tanpa jejak.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_LLM_ENABLED"] = "0"

import db  # noqa: E402


def _fresh_conn():
    path = os.path.join(tempfile.mkdtemp(), "persist.db")
    conn = db.connect(path)
    db.init_db(conn)
    db.seed_if_empty(conn)
    return conn


def test_admin_edit_survives_reseed():
    conn = _fresh_conn()
    doc_id = "doc_glossary"
    before = conn.execute("SELECT content FROM documents WHERE id=?", (doc_id,)).fetchone()[0]

    conn.execute("UPDATE documents SET content=?, local_edit=1 WHERE id=?", ("EDITED", doc_id))
    conn.commit()

    db.ensure_seed_docs(conn)  # simulasi restart

    after = conn.execute("SELECT content, local_edit FROM documents WHERE id=?", (doc_id,)).fetchone()
    assert after["content"] == "EDITED", "edit admin ditimpa seed saat restart"
    assert after["local_edit"] == 1
    assert before != "EDITED"


def test_untouched_seed_doc_still_syncs():
    """Dokumen yang belum pernah disunting tetap mengikuti perbaikan seed."""
    conn = _fresh_conn()
    doc_id = "doc_hr_transport_2026"
    conn.execute("UPDATE documents SET content='versi lama' WHERE id=?", (doc_id,))
    conn.commit()

    db.ensure_seed_docs(conn)

    row = conn.execute("SELECT content, local_edit FROM documents WHERE id=?", (doc_id,)).fetchone()
    assert row["local_edit"] == 0
    assert row["content"] != "versi lama", "seed sync berhenti bekerja untuk dokumen biasa"


def test_new_seed_doc_still_inserted():
    conn = _fresh_conn()
    conn.execute("DELETE FROM documents WHERE id='doc_glossary'")
    conn.commit()
    db.ensure_seed_docs(conn)
    assert conn.execute(
        "SELECT 1 FROM documents WHERE id='doc_glossary'").fetchone() is not None
