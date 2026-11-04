"""Regresi: daftar pertanyaan lanjutan harus SELALU ada dan aman RBAC.

Bug: `suggestions` hanya diisi oleh LLM. Saat LLM mati/gagal/timeout atau
jawaban berupa refusal, backend mengembalikan `[]` sehingga UI jatuh ke daftar
statis global yang memuat topik terlarang untuk role tersebut (mis. "Apa status
Proyek Kenari?" untuk employee). Sekarang ada fallback deterministik yang
dibangun dari dokumen yang lolos filter izin.
"""
import os
import sys
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_LLM_ENABLED"] = "0"

import db
from engine import answer_question, _fallback_suggestions, filter_allowed_docs, get_user


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    db.seed_if_empty(conn)
    return conn


def test_suggestions_always_present_offline():
    """LLM dimatikan: setiap jawaban tetap punya >=3 saran."""
    conn = make_conn()
    for uid in ("u_budi", "u_andi", "u_sari", "u_rina", "u_dewi"):
        res = answer_question(conn, uid, "Bagaimana cara mengajukan cuti?")
        assert len(res["suggestions"]) >= 3, (uid, res["suggestions"])
        assert len(res["suggestions"]) <= 5, (uid, res["suggestions"])
    print("  ok suggestions_always_present_offline")


def test_suggestions_present_on_refusal():
    """Pertanyaan yang ditolak (no_access) tetap memberi saran yang sah."""
    conn = make_conn()
    res = answer_question(conn, "u_budi", "Siapa PIC untuk klien Kenari?")
    assert res["refusal_reason"] == "no_access", res["refusal_reason"]
    assert len(res["suggestions"]) >= 3, res["suggestions"]
    joined = " ".join(res["suggestions"]).lower()
    # "Kenari — Profil Mitra" boleh (dokumen publik untuk employee);
    # yang dilarang adalah dokumen internal "Proyek Kenari".
    assert "proyek kenari" not in joined, "saran tidak boleh menunjuk dokumen terlarang"
    print("  ok suggestions_present_on_refusal")


def test_suggestions_respect_rbac_scope():
    """Semua saran employee harus berasal dari dokumen yang boleh ia akses."""
    conn = make_conn()
    budi = get_user(conn, "u_budi")
    allowed = filter_allowed_docs(conn, budi)
    titles = {(d.get("title") or "").lower() for d in allowed}
    assert "proyek kenari — dokumen internal" not in titles
    sug = _fallback_suggestions(allowed, "Apa itu Nusantara OS?", n=5)
    assert len(sug) == 5, sug
    for q in sug:
        low = q.lower()
        assert "proyek kenari" not in low, q
        assert "struktur gaji" not in low, q
    print("  ok suggestions_respect_rbac_scope")


def test_suggestions_unique_and_not_echo_question():
    conn = make_conn()
    rina = get_user(conn, "u_rina")
    allowed = filter_allowed_docs(conn, rina)
    q = "Apa isi Glosarium Istilah Internal?"
    sug = _fallback_suggestions(allowed, q, n=5)
    assert len({s.lower() for s in sug}) == len(sug), "saran duplikat"
    assert q.lower() not in {s.lower() for s in sug}, "saran mengulang pertanyaan"
    print("  ok suggestions_unique_and_not_echo_question")


def test_fallback_suggestion_does_not_double_pengumuman():
    """Judul pengumuman sudah berawalan "Pengumuman", template juga.

    Terukur di :8890: saran keluar sebagai "Apa isi pengumuman Pengumuman
    Finance — Batas Waktu Reimbursement Oktober?".
    """
    allowed = [{"id": "doc_x", "title": "Pengumuman HR — Town Hall November 2026",
                "source_type": "hr_announcement"}]
    sug = _fallback_suggestions(allowed, "halo", n=1)
    assert sug == ["Apa isi pengumuman HR — Town Hall November 2026?"], sug
    # Judul tanpa awalan tetap utuh.
    allowed = [{"id": "doc_y", "title": "Kebijakan Cuti",
                "source_type": "hr_announcement"}]
    sug = _fallback_suggestions(allowed, "halo", n=1)
    assert sug == ["Apa isi pengumuman Kebijakan Cuti?"], sug
    print("  ok fallback_suggestion_does_not_double_pengumuman")


if __name__ == "__main__":
    test_suggestions_always_present_offline()
    test_suggestions_present_on_refusal()
    test_suggestions_respect_rbac_scope()
    test_suggestions_unique_and_not_echo_question()
    test_fallback_suggestion_does_not_double_pengumuman()
    print("SUGGESTIONS TESTS PASS")
