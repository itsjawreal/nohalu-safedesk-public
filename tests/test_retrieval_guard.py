"""Regresi gerbang relevansi retrieval (F1–F5).

Konteks bug: pertanyaan di luar korpus ("Berapa harga bitcoin hari ini?")
dijawab dengan dokumen cuti hanya karena kata umum "hari" kebetulan cocok,
dan jawaban LLM "Saya tidak memiliki informasi…" yang panjang tidak dikenali
sebagai fallback sehingga `refusal_reason` tetap None + sitasi tidak relevan.

Yang dijaga file ini:
1. Pertanyaan di luar korpus -> refusal `no_data`, tanpa sitasi.
2. Pertanyaan sah (dalam korpus) TIDAK ikut ditolak (anti-regresi).
3. `_is_fallback` mengenali fallback panjang (tanpa batas 160 char).
4. `canon` memetakan akhiran "-nya" dan sinonim baru.
"""
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

# suite deterministik: jangan pernah memanggil LLM jaringan
os.environ["NOHALU_LLM_ENABLED"] = "0"

import db
import engine
from engine import answer_question, NEUTRAL_REFUSAL


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    db.seed_if_empty(conn)
    return conn


OUT_OF_CORPUS = [
    "Berapa harga bitcoin hari ini?",
    "Berapa kurs dollar hari ini?",
    "Siapa presiden Indonesia?",
    "Bagaimana cara membuat rendang?",
    "Apa resep nasi goreng?",
    "Apa itu quantum computing?",
    "Siapa juara piala dunia 2022?",
    # Judul "Alur Kerja …" memuat kata umum "kerja" (df=29/70). Dulu kecocokan
    # judul semacam itu dianggap bukti, sehingga pertanyaan luar domain ini
    # dijawab prosedur reimbursement.
    "Jelaskan cara kerja blockchain",
]


def test_out_of_corpus_refused():
    """Pertanyaan di luar korpus: refusal netral, tanpa sitasi palsu."""
    conn = make_conn()
    for q in OUT_OF_CORPUS:
        r = answer_question(conn, "u_budi", q)
        assert r["refusal_reason"] == "no_data", (q, r["refusal_reason"], r["selected_sources"])
        assert r["answer"] == NEUTRAL_REFUSAL, (q, r["answer"][:120])
        assert r["citations"] == [], (q, r["citations"])
        assert r["selected_sources"] == [], (q, r["selected_sources"])
        # saran tetap wajib ada walau refusal
        assert len(r["suggestions"]) >= 3, (q, r["suggestions"])


def test_title_only_match_does_not_dump_unrelated_document():
    """Judul cocok lewat kata umum saja tidak boleh membuang isi dokumen.

    Terukur: "susun jadwal olahraga saya" menyempit ke dokumen jadwal gaji
    (kata "jadwal", df=2) dan jadwal evaluasi kinerja. Tidak satu pun memuat
    "olahraga", jadi `best` kosong — dan cabang lama menumpahkan SELURUH isi
    dokumen gaji ke layar. User bertanya jadwal olahraga, dibayar tanggal gaji.
    """
    conn = make_conn()
    for q in ("susun jadwal olahraga saya",
              "buatkan jadwal makan siang"):
        r = answer_question(conn, "u_budi", q)
        assert r["refusal_reason"] == "no_data", (q, r["refusal_reason"], r["selected_sources"])
        assert r["selected_sources"] == [], (q, r["selected_sources"])
        assert r["citations"] == [], (q, r["citations"])
        assert "gaji" not in r["answer"].lower(), (q, r["answer"][:160])


def test_in_corpus_still_answered():
    """Anti-regresi: pertanyaan sah tidak boleh ikut ditolak gerbang baru."""
    conn = make_conn()
    cases = [
        ("Apa itu Nusantara OS?", "doc_it_nusantara_os"),
        ("Kapan gaji dibayarkan?", "doc_fin_payroll_schedule"),
        ("Bagaimana kebijakan WFH?", "doc_hr_working_hours"),
        ("Apa visi dan misi perusahaan?", "doc_company_mission_values"),
        ("Apa saja layanan utama perusahaan?", "doc_portfolio_services"),
        ("Bagaimana kebijakan parkir?", "doc_offices_facilities"),
        ("Apa standar SLA operasional?", "doc_ops_sla"),
        ("Berapa cuti tahunan?", "doc_leave_policy"),
        ("Bagaimana alur pengajuan reimbursement?", "doc_workflow_approval"),
        ("Siapa PIC IT kalau laptop rusak?", "doc_contacts_it_finance"),
        ("Apa itu Kenari?", "doc_kenari_partner"),
        ("Berapa lama cuti melahirkan?", "doc_hr_maternity_leave"),
        ("Bagaimana kebijakan cuti sakit?", "doc_hr_sick_emergency_leave"),
        ("Apa saja tunjangan BPJS dan THR?", "doc_hr_benefits_bpjs_thr"),
        ("Bagaimana prosedur resign?", "doc_hr_resignation"),
        ("Apa kebijakan lembur?", "doc_hr_overtime"),
        ("Bagaimana cara melihat slip gaji?", "doc_payslip_howto"),
        ("Bagaimana cara reservasi ruang rapat?", "doc_ops_meeting_rooms"),
        ("Di mana kantor cabang Bandung?", "doc_offices_facilities"),
        ("Apa kanal komunikasi internal?", "doc_company_communication"),
    ]
    for q, want in cases:
        r = answer_question(conn, "u_budi", q)
        assert r["refusal_reason"] is None, (q, r["refusal_reason"])
        assert want in r["selected_sources"], (q, want, r["selected_sources"])


def test_company_subtopic_not_blanket_profile():
    """Kata "perusahaan" tidak boleh membuat semua pertanyaan dijawab profil."""
    conn = make_conn()
    cases = [
        ("Apa visi dan misi perusahaan?", "doc_company_mission_values"),
        ("Siapa klien perusahaan?", "doc_company_clients"),
        ("Apa budaya perusahaan?", "doc_culture_events"),
        ("Siapa saja pendiri perusahaan?", "doc_company_founders"),
        ("Apa penghargaan yang diterima perusahaan?", "doc_company_awards"),
        ("Bagaimana program keberlanjutan perusahaan?", "doc_company_sustainability"),
        ("Apa lini bisnis perusahaan?", "doc_company_business_lines"),
        ("Apa itu kode etik perusahaan?", "doc_code_of_conduct"),
    ]
    for q, want in cases:
        r = answer_question(conn, "u_budi", q)
        assert r["refusal_reason"] is None, (q, r["refusal_reason"])
        assert want in r["selected_sources"], (q, want, r["selected_sources"])


def test_generic_title_word_does_not_beat_distinctive_token():
    """Kata umum di judul ("program") tidak boleh mengalahkan kata pembeda.

    Regresi: "program keberlanjutan" dulu dijawab "Program Kesehatan &
    Wellness" karena "program" cocok di kedua judul, sementara
    "keberlanjutan" (df=1) diabaikan.
    """
    conn = make_conn()
    cases = [
        ("Bagaimana program keberlanjutan perusahaan?", "doc_company_sustainability"),
        ("Apa program CSR perusahaan?", "doc_company_sustainability"),
        ("Bagaimana komitmen lingkungan perusahaan?", "doc_company_sustainability"),
        ("Apa program kesehatan dan wellness?", "doc_hr_wellness"),
    ]
    for q, want in cases:
        r = answer_question(conn, "u_budi", q)
        assert r["refusal_reason"] is None, (q, r["refusal_reason"])
        assert r["selected_sources"][:1] == [want], (q, want, r["selected_sources"])


def test_is_fallback_long_text():
    """Fallback panjang (dulu >160 char) harus tetap terdeteksi."""
    long_refusal = (
        "Saya tidak memiliki informasi mengenai harga bitcoin hari ini. "
        "Data yang tersedia dalam konteks tidak mencakup harga pasar, kurs, "
        "atau instrumen keuangan apa pun, sehingga saya tidak bisa menjawab."
    )
    assert engine._is_fallback(long_refusal) is True
    assert engine._is_fallback("Maaf, saya tidak punya data tersebut.") is True
    assert engine._is_fallback("Belum ada informasi tentang itu.") is True
    # jawaban sah tidak boleh dianggap fallback
    assert engine._is_fallback("Cuti tahunan 12 hari, diajukan minimal 3 hari kerja.") is False
    assert engine._is_fallback("Gaji dibayarkan setiap tanggal 27 bulan berjalan.") is False
    assert engine._is_fallback("") is False


def test_fallback_answer_normalized_to_refusal(monkeypatch):
    """Bila LLM tetap mengembalikan fallback, hasil akhir = refusal jujur."""
    conn = make_conn()

    def fake_open(allowed, user, question, conn_, result):
        result["answer"] = (
            "Saya tidak memiliki informasi mengenai hal tersebut. Konteks yang "
            "tersedia hanya membahas kebijakan internal perusahaan."
        )
        result["citations"] = [{"id": "doc_leave_policy", "title": "Kebijakan Cuti Tahunan"}]
        result["selected_sources"] = ["doc_leave_policy"]
        result["engine_used"] = "llm"
        return True

    monkeypatch.setattr(engine, "_llm_open_answer", fake_open)
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    try:
        r = answer_question(conn, "u_budi", "Berapa harga bitcoin hari ini?")
    finally:
        os.environ["NOHALU_LLM_ENABLED"] = "0"

    assert r["refusal_reason"] == "no_data", r["refusal_reason"]
    assert r["answer"] == NEUTRAL_REFUSAL, r["answer"][:120]
    assert r["citations"] == [], r["citations"]
    assert r["selected_sources"] == [], r["selected_sources"]


def test_canon_suffix_and_synonyms():
    """Akhiran -nya & sinonim baru dipetakan ke token yang sama."""
    assert engine.canon("tanggung jawabnya") == "tanggung jawab"
    assert engine.canon("cutinya") == "cuti"
    assert engine.canon("resign") == "pengunduran"
    assert engine.canon("benefit") == "hak"
    assert engine.canon("sustainability") == "keberlanjutan"
    assert engine.canon("CSR") == "keberlanjutan"
    # kata pendek tidak boleh dipotong sembarangan
    assert engine.canon("hanya") == "hanya"
    assert engine.canon("punya") == "punya"


def test_weak_grounding_logged(capsys):
    """Gerbang relevansi menulis jejak diagnosa (bukan jawaban palsu)."""
    conn = make_conn()
    answer_question(conn, "u_budi", "Berapa harga bitcoin hari ini?")
    err = capsys.readouterr().err
    assert "weak grounding" in err, err[:300]


def test_project_pic_is_not_denied():
    """PIC proyek harus dijawab dari dokumen proyek, bukan dicap `no_access`.

    "Siapa PIC proyek Alpha?" memuat "pic" yang dipetakan ke sinonim
    "contact". Dulu router memilih topik kontak lebih dulu, dokumen kontak HR
    tidak boleh diakses karyawan, dan engine mengembalikan `no_access` —
    sehingga jawaban LLM yang sah dan bersitasi ikut dicap `deny`.
    """
    conn = make_conn()
    r = answer_question(conn, "u_budi", "Siapa PIC proyek Alpha?")
    assert r["refusal_reason"] is None, r["refusal_reason"]
    assert r["permission_decision"] == "allow", r["permission_decision"]
    assert "doc_project_alpha" in r["selected_sources"], r["selected_sources"]
    assert "Andi Pratama" in r["answer"], r["answer"][:200]


if __name__ == "__main__":
    test_out_of_corpus_refused()
    test_in_corpus_still_answered()
    test_project_pic_is_not_denied()
    test_company_subtopic_not_blanket_profile()
    test_generic_title_word_does_not_beat_distinctive_token()
    test_is_fallback_long_text()
    test_canon_suffix_and_synonyms()
    print("ALL RETRIEVAL GUARD TESTS PASS")
