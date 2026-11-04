"""Regresi: kegagalan LLM harus tercatat sebagai kode, bukan ditelan jadi None.

Bug yang dijaga file ini:
1. `_llm_rephrase` / `_llm_open_answer` dulu melakukan
   `result["llm_error"] = None` di blok `except`, sehingga kegagalan hilang dan
   badge UI "Jawaban aman ditampilkan" (js/chat/view.js) tidak pernah muncul.
2. Pesan exception mentah tidak boleh bocor ke klien (bisa memuat URL provider,
   body error, atau potongan prompt). Yang diekspos hanya kode `<stage>:<kode>`.

Tidak ada panggilan jaringan di sini: `llm.generate` selalu di-monkeypatch.
"""
import os
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_LLM_ENABLED"] = "0"

import db
import engine
from engine import answer_question


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    db.seed_if_empty(conn)
    return conn


GROUNDED_Q = "Berapa tunjangan transport bulanan?"


def _enable(monkeypatch):
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")


def test_error_code_mapping_never_leaks_raw_message():
    code = engine._llm_error_code(
        RuntimeError("LLM HTTP 500: upstream stack trace /v1/chat/completions"), "rephrase")
    assert code == "rephrase:upstream_error"
    assert "http" not in code.lower() and "/v1" not in code

    assert engine._llm_error_code(
        RuntimeError("llm leaked restricted entity"), "rephrase") == "rephrase:restricted_entity"
    assert engine._llm_error_code(
        RuntimeError("llm invented numbers"), "open") == "open:invented_numbers"
    assert engine._llm_error_code(
        RuntimeError("validation rejected"), "open") == "open:validation_rejected"
    assert engine._llm_error_code(
        RuntimeError("LLM network error: timed out"), "open") == "open:upstream_error"
    # exception tak dikenal tetap jadi kode generik, bukan pesan asli
    assert engine._llm_error_code(RuntimeError("koneksi putus di 10.0.0.7"), "open") == \
        "open:unexpected_error"
    assert engine._llm_error_code(None, "pick") == "pick:unexpected_error"


def test_rephrase_failure_is_recorded(monkeypatch):
    """Gagal parafrase -> draft deterministik dipakai DAN alasannya tercatat."""
    conn = make_conn()
    _enable(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("LLM HTTP 503: service unavailable")

    monkeypatch.setattr(engine.llm, "generate", boom)
    r = answer_question(conn, "u_budi", GROUNDED_Q, use_llm=True)

    assert r["engine_used"] == "deterministic", r["engine_used"]
    assert r["llm_error"] == "rephrase:upstream_error", r["llm_error"]
    assert "750.000" in r["answer"], "draft terverifikasi harus tetap dikirim"
    assert r["suggestions"], "saran tetap wajib ada meski LLM gagal"


def test_rephrase_success_clears_error(monkeypatch):
    """Percobaan yang akhirnya berhasil tidak boleh menyisakan kode error."""
    conn = make_conn()
    _enable(monkeypatch)
    draft = answer_question(conn, "u_budi", GROUNDED_Q, use_llm=False)["answer"]

    calls = []

    def fake_generate(system, user, max_tokens=None, reasoning=None):
        calls.append(reasoning)
        # parafrase: kembalikan draft apa adanya (lolos semua gerbang validasi)
        return {"text": draft, "ms": 12, "model": "mercury-3", "usage": {}}

    monkeypatch.setattr(engine.llm, "generate", fake_generate)
    r = answer_question(conn, "u_budi", GROUNDED_Q, use_llm=True)

    assert r["engine_used"] == "llm", r["llm_error"]
    assert r["llm_error"] is None
    assert engine.REPHRASE_REASONING in calls, calls


def test_rephrase_uses_configured_reasoning(monkeypatch):
    """Level reasoning parafrase tidak lagi hard-coded 'high'."""
    conn = make_conn()
    _enable(monkeypatch)
    draft = answer_question(conn, "u_budi", GROUNDED_Q, use_llm=False)["answer"]
    seen = []

    def fake_generate(system, user, max_tokens=None, reasoning=None):
        seen.append(reasoning)
        return {"text": draft, "ms": 5, "model": "mercury-3", "usage": {}}

    monkeypatch.setattr(engine.llm, "generate", fake_generate)
    answer_question(conn, "u_budi", GROUNDED_Q, use_llm=True)

    assert seen[0] == engine.REPHRASE_REASONING
    assert seen[0] != "high"


def test_open_answer_validation_rejection_is_recorded(monkeypatch):
    """Jawaban LLM yang ditolak validasi harus meninggalkan kode, bukan None."""
    conn = make_conn()
    _enable(monkeypatch)
    user = engine.get_user(conn, "u_budi")
    allowed = engine.filter_allowed_docs(conn, user)
    result = {
        "answer": "", "citations": [], "conflict_detected": False, "conflict_note": "",
        "permission_decision": "allow", "refusal_reason": None, "pii_masked": False,
        "prompt_injection_detected": False, "selected_sources": [], "retrieved_sources": [],
    }

    def fake_generate(system, user_prompt, max_tokens=None, reasoning=None):
        if "DAFTAR DOKUMEN" in user_prompt:
            return {"text": allowed[0]["id"], "ms": 1, "model": "mercury-3", "usage": {}}
        # email karangan -> _ungrounded_entities menolak
        return {"text": "Hubungi hr@evil.example.com untuk detailnya.",
                "ms": 1, "model": "mercury-3", "usage": {}}

    monkeypatch.setattr(engine.llm, "generate", fake_generate)
    ok = engine._llm_open_answer(allowed, user, "Siapa PIC urusan yang tidak jelas?",
                                 conn, result)

    # Jawaban LLM ditolak; engine memulihkan dengan kutipan verbatim dokumen
    # (grounded by construction) sehingga hasilnya tetap True — tapi kode
    # kegagalan WAJIB terekam dan email karangan tidak boleh lolos.
    assert result["llm_error"] == "open:validation_rejected", result["llm_error"]
    assert "evil.example.com" not in result["answer"]
    assert result["engine_used"] == "deterministic"
    assert ok is True, "kutipan grounded harus dipakai sebagai pemulihan"


def test_leak_gate_ignores_common_phrase_bigrams():
    """Bigram frasa umum bukan bukti kebocoran; bigram entitas tetap diblok.

    Regresi: ("tidak","berlaku") dan ("berlaku","sejak") hanya muncul di
    dokumen restricted, sehingga parafrase sah tentang tunjangan transport
    ditolak "llm leaked restricted entity" (17/331 jawaban historis).
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    allowed = engine.filter_allowed_docs(conn, user)

    benign = ("Tunjangan transport bulanan Rp750.000, berlaku sejak pengumuman HR, "
              "ketentuan lama sudah tidak berlaku.")
    assert engine._restricted_bigram_hits(conn, allowed, benign) == set()
    assert engine._leaks_restricted(conn, allowed, benign) is False

    # entitas/angka yang benar-benar eksklusif dokumen restricted tetap terdeteksi
    leaky = "Alokasi khusus proyek Delta adalah Rp28.000.000."
    hits = engine._restricted_bigram_hits(conn, allowed, leaky)
    assert {("alokasi", "khusus"), ("project", "delta")} <= hits, hits
    assert engine._leaks_restricted(conn, allowed, leaky) is True

    # Nominal gaji yang tidak ada di sumber terpilih sudah dijaga lapis lain
    # (`_amounts_in`), jadi bigram-nya cukup terdeteksi sebagian.
    band = "Grade 3 memiliki kisaran Rp12.000.000 sampai Rp18.000.000."
    assert engine._restricted_bigram_hits(conn, allowed, band), "bigram band gaji harus terdeteksi"
    assert engine._amounts_in(band) - engine._amounts_in("") == {"12000000", "18000000"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
