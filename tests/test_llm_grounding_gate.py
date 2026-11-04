"""Regresi gerbang jawaban LLM: klaim tanpa bukti & klasifikasi penolakan akses.

Dua cacat nyata yang diukur di server lokal (`:8890`, LLM aktif):

A. Penolakan akses tidak terklasifikasi
   "Berapa gaji seluruh karyawan?" dan "Apa isi dokumen internal Proyek Kenari?"
   dijawab model dengan kalimat penolakan yang wajar, tetapi `refusal_reason`
   tetap None dan `permission_decision` tetap "allow". Audit mencatat seolah
   permintaan itu dilayani, padahal datanya memang tidak boleh diakses.

B. Jawaban dari ingatan parametrik
   "Siapa presiden Indonesia?" dijawab "Prabowo Subianto…", "Siapa juara piala
   dunia 2022?" dijawab "Argentina", tanpa satu pun tool yang mengembalikan
   data. Fakta itu benar di dunia nyata, tetapi tidak ada di korpus perusahaan
   dan tidak bisa diverifikasi — asisten internal tidak boleh menyajikannya
   seolah berasal dari dokumen.

Tes ini memakai LLM palsu (tanpa jaringan) supaya jalur agent benar-benar
dijalankan. Empat kasus pertama GAGAL bila gerbangnya dimatikan.
"""
import os
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_LLM_ENABLED"] = "0"

import agent
import db
import engine


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    db.seed_if_empty(conn)
    db.ensure_seed_users(conn)
    return conn


def _llm_says(text, tool_calls=None):
    def fake_chat(messages, tools=None, **kw):
        return {"text": text, "tool_calls": tool_calls or [], "finish_reason": "stop",
                "ms": 1, "usage": {}, "model": "mercury-3"}
    return fake_chat


def _llm_search_then(text, query="kebijakan"):
    """Model memanggil search_policies dulu, lalu menjawab `text`."""
    state = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return {"text": "", "finish_reason": "tool_calls", "ms": 1, "usage": {},
                    "model": "mercury-3",
                    "tool_calls": [{"id": "c1", "name": "search_policies",
                                    "arguments": {"query": query},
                                    "raw_arguments": '{"query":"%s"}' % query}]}
        return {"text": text, "tool_calls": [], "finish_reason": "stop",
                "ms": 1, "usage": {}, "model": "mercury-3"}
    return fake_chat


# ---------------------------------------------------------------------------
# B — klaim tanpa bukti
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,answer", [
    ("Siapa presiden Indonesia?",
     "Presiden Indonesia saat ini adalah **Prabowo Subianto**, dilantik 20 Oktober 2024."),
    ("Siapa juara piala dunia 2022?",
     "Juara Piala Dunia 2022 adalah **Argentina**, mengalahkan Prancis lewat adu penalti."),
    ("Apa itu quantum computing?",
     "Quantum computing memakai superposisi dan entanglement untuk menghitung."),
])
def test_paramemoric_answer_without_evidence_is_rejected(monkeypatch, question, answer):
    """Jawaban tanpa tool & tanpa dokumen tidak boleh lolos sebagai jawaban LLM."""
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", _llm_says(answer))

    r = agent.run(conn, user, question, None)

    assert r["engine_used"] == "deterministic", r["engine_used"]
    # Dua jalur sama-sama benar dan sama-sama jujur: gerbang luar-domain
    # menolak sebelum model dipanggil (`out_of_domain`), atau validator menolak
    # jawaban model yang mengarang (`ungrounded_claim`). Yang WAJIB sama:
    # hasilnya penolakan netral tanpa sitasi.
    assert r["llm_error"] in ("agent:ungrounded_claim", "agent:out_of_domain"), r["llm_error"]
    assert r["refusal_reason"] == "no_data", r["refusal_reason"]
    assert r["answer"] == engine.NEUTRAL_REFUSAL, r["answer"][:120]
    assert r["citations"] == []


def test_empty_search_result_is_not_evidence(monkeypatch):
    """Memanggil `search_policies` yang kosong BUKAN bukti.

    Model yang mencari, tidak menemukan apa pun, lalu menjawab dari ingatan
    umumnya tidak boleh dianggap grounded hanya karena ada jejak tool.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", _llm_search_then(
        "Presiden Indonesia adalah Prabowo Subianto.", query="presiden indonesia"))

    r = agent.run(conn, user, "Siapa presiden Indonesia?", None)

    assert r["llm_error"] == "agent:ungrounded_claim", r["llm_error"]
    assert "Prabowo" not in r["answer"]


def test_grounded_answer_still_passes(monkeypatch):
    """Anti-regresi: jawaban yang benar-benar membaca dokumen tidak ditolak."""
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", _llm_search_then(
        "Tunjangan transport bulanan Rp750.000.", query="tunjangan transport"))

    r = agent.run(conn, user, "Berapa tunjangan transport bulanan?", None)

    assert r["engine_used"] == "llm", r["llm_error"]
    assert r["llm_error"] is None, r["llm_error"]
    assert "750.000" in r["answer"]
    assert r["citations"], "jawaban sah harus tetap punya sitasi"


def test_personal_data_answer_still_passes(monkeypatch):
    """Data kepegawaian (tool personal) adalah bukti sah, bukan klaim tanpa dasar."""
    conn = make_conn()
    user = engine.get_user(conn, "u_maya")
    state = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return {"text": "", "finish_reason": "tool_calls", "ms": 1, "usage": {},
                    "model": "mercury-3",
                    "tool_calls": [{"id": "c1", "name": "get_leave_balance",
                                    "arguments": {}, "raw_arguments": "{}"}]}
        return {"text": "Sisa cuti Anda 12 hari.", "tool_calls": [],
                "finish_reason": "stop", "ms": 1, "usage": {}, "model": "mercury-3"}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Berapa sisa cuti saya?", None)

    assert r["engine_used"] == "llm", r["llm_error"]
    assert r["llm_error"] is None
    assert "12 hari" in r["answer"]


def test_greeting_is_not_gated(monkeypatch):
    """Sapaan murni tidak butuh sumber dan tidak boleh jadi refusal."""
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", _llm_says("Halo Budi, ada yang bisa saya bantu?"))

    r = agent.run(conn, user, "halo", None)

    assert r["llm_error"] is None, r["llm_error"]
    assert r["engine_used"] == "llm"
    assert "Halo" in r["answer"]


# ---------------------------------------------------------------------------
# A — klasifikasi penolakan akses
# ---------------------------------------------------------------------------
def test_full_access_refusal_is_classified_no_access(monkeypatch):
    """Penolakan penuh atas data restricted -> no_access + permission deny."""
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", _llm_says(
        "Data gaji seluruh karyawan tidak tersedia untuk akun Anda. "
        "Slip gaji hanya dapat diakses oleh pemilik akun masing-masing."))

    r = agent.run(conn, user, "Berapa gaji seluruh karyawan?", None)

    assert r["refusal_reason"] == "no_access", r["refusal_reason"]
    assert r["permission_decision"] == "deny", r["permission_decision"]
    assert r["citations"] == []


def test_tool_denial_is_classified_no_access(monkeypatch):
    """Penolakan dari tool (bukan dari teks model) juga harus terklasifikasi."""
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    state = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return {"text": "", "finish_reason": "tool_calls", "ms": 1, "usage": {},
                    "model": "mercury-3",
                    "tool_calls": [{"id": "c1", "name": "get_team_attendance",
                                    "arguments": {}, "raw_arguments": "{}"}]}
        return {"text": "Rekap kehadiran tim tidak tersedia untuk akun Anda.",
                "tool_calls": [], "finish_reason": "stop", "ms": 1,
                "usage": {}, "model": "mercury-3"}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Tampilkan rekap kehadiran seluruh tim", None)

    assert r["refusal_reason"] == "no_access", r["refusal_reason"]
    assert r["permission_decision"] == "deny"
    assert any(not t.get("ok") for t in r["agent_trace"]), r["agent_trace"]


def test_partial_answer_keeps_citations_and_is_not_no_access(monkeypatch):
    """Jawaban parsial: sebut data restricted tidak ada, tapi kutip sumber sah.

    Ini yang paling mudah rusak: mengisi `refusal_reason` akan MENGHAPUS
    sitasi sah, sehingga user kehilangan satu-satunya informasi yang boleh ia
    baca. Kolom audit `permission_decision` tetap harus "deny".

    Pertanyaannya sengaja BUKAN nama proyek terbatas: untuk nama proyek,
    penolakan wajib simetris dengan proyek yang tidak ada (lihat
    `test_restricted_project_is_denied_symmetrically`), sehingga tidak ada
    jawaban parsial yang sah di jalur itu.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", _llm_search_then(
        "Struktur gaji per grade tidak tersedia untuk akun Anda.\n\n"
        "Yang bisa saya sampaikan dari kebijakan tunjangan: tunjangan "
        "transport dibayarkan bulanan.",
        query="tunjangan transport"))

    r = agent.run(conn, user, "Berapa kisaran gaji karyawan?", None)

    assert r["permission_decision"] == "deny", r["permission_decision"]
    assert r["refusal_reason"] is None, r["refusal_reason"]
    assert r["citations"], "sitasi sah tidak boleh dihapus oleh klasifikasi akses"


def test_personal_salary_question_is_not_denied(monkeypatch):
    """Pertanyaan gaji MILIK SENDIRI bukan penolakan akses.

    Engine deterministik menolak topik "gaji" untuk karyawan (struktur gaji
    restricted), tetapi "berapa gaji saya bulan ini" dijawab sah oleh tool
    slip gaji. Tanpa pengecualian intent personal, audit salah mencatat deny.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    state = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return {"text": "", "finish_reason": "tool_calls", "ms": 1, "usage": {},
                    "model": "mercury-3",
                    "tool_calls": [{"id": "c1", "name": "get_payslips",
                                    "arguments": {}, "raw_arguments": "{}"}]}
        return {"text": "Slip terbaru Anda periode September 2026.",
                "tool_calls": [], "finish_reason": "stop", "ms": 1,
                "usage": {}, "model": "mercury-3"}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Berapa gaji saya bulan ini?", None)

    assert r["permission_decision"] == "allow", r["permission_decision"]
    assert r["refusal_reason"] is None, r["refusal_reason"]


# ---------------------------------------------------------------------------
# C — penolakan yang ditulis model TANPA memanggil tool ber-scope
# ---------------------------------------------------------------------------
_REFUSAL_TEXT = (
    "Sisa cuti {name} tidak dapat saya tampilkan untuk akun Anda. Akses saldo "
    "cuti hanya mencakup bawahan langsung; data itu hanya dapat diakses HR/admin."
)


def _llm_refuses(name):
    """Model menolak dengan kalimat benar, tanpa memanggil tool apa pun."""
    text = _REFUSAL_TEXT.format(name=name)

    def fake_chat(messages, tools=None, **kw):
        return {"text": text, "tool_calls": [], "finish_reason": "stop",
                "ms": 1, "usage": {}, "model": "mercury-3"}
    return fake_chat


def test_refusal_without_tool_is_still_classified_deny(monkeypatch):
    """Terukur di :8890: `u_andi` tanya sisa cuti Rina → model menolak benar,
    tetapi audit mencatat `allow` karena tidak ada tool yang menolak.

    Dua sinyal lama tidak menyala: `access_denied` kosong (model hanya memanggil
    `get_org_chart`/`get_directory`) dan engine menganggap "sisa cuti" intent
    personal yang selalu boleh. Probe izin deterministik menutup celah itu.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_andi")
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", _llm_refuses("Rina"))

    r = agent.run(conn, user, "Berapa sisa cuti Rina?", None)

    assert r["permission_decision"] == "deny", r["permission_decision"]
    assert r["refusal_reason"] == "no_access", r["refusal_reason"]
    assert r["citations"] == []


def test_employee_asking_coworker_leave_is_denied(monkeypatch):
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", _llm_refuses("Maya"))

    r = agent.run(conn, user, "Berapa sisa cuti Maya?", None)

    assert r["permission_decision"] == "deny", r["permission_decision"]
    assert r["refusal_reason"] == "no_access", r["refusal_reason"]


def test_team_lead_asking_direct_report_leave_is_allowed(monkeypatch):
    """Anti-regresi: bawahan langsung BUKAN penolakan.

    Probe harus benar-benar menjalankan `get_team_leave_balance` dan membaca
    hasilnya, bukan sekadar menyala untuk setiap nama karyawan lain.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_andi")
    state = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return {"text": "", "finish_reason": "tool_calls", "ms": 1, "usage": {},
                    "model": "mercury-3",
                    "tool_calls": [{"id": "c1", "name": "get_team_leave_balance",
                                    "arguments": {"employee": "Maya"},
                                    "raw_arguments": '{"employee":"Maya"}'}]}
        return {"text": "Sisa cuti Maya tahun 2026 adalah 12 hari.",
                "tool_calls": [], "finish_reason": "stop", "ms": 1,
                "usage": {}, "model": "mercury-3"}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Berapa sisa cuti Maya?", None)

    assert r["permission_decision"] == "allow", r["permission_decision"]
    assert r["refusal_reason"] is None, r["refusal_reason"]


def test_policy_question_about_leave_is_not_denied(monkeypatch):
    """Pertanyaan kebijakan cuti tidak boleh diprobe sebagai akses orang lain."""
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", _llm_search_then(
        "Cuti tahunan 12 hari kerja per tahun; pengajuan minimal 3 hari kerja "
        "sebelum tanggal mulai.", query="kebijakan cuti"))

    r = agent.run(conn, user, "Bagaimana aturan carry-over cuti?", None)

    assert r["permission_decision"] == "allow", r["permission_decision"]
    assert r["refusal_reason"] is None, r["refusal_reason"]


def test_refusal_reason_filled_for_melihat_phrasing(monkeypatch):
    """Frasa "tidak bisa melihat" juga penolakan penuh.

    Terukur di :8890: model menulis "Saya tidak bisa melihat sisa cuti Maya…";
    `permission_decision` sudah deny, tetapi `refusal_reason` tetap null
    sehingga UI menampilkannya sebagai jawaban biasa.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    text = ("Saya tidak bisa melihat sisa cuti Maya Anggraini dari akun Anda. "
            "Sebagai karyawan, akses ke saldo cuti orang lain hanya untuk "
            "HR/admin atau atasan langsung. Silakan hubungi tim HR.")
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", _llm_says(text))

    r = agent.run(conn, user, "Berapa sisa cuti Maya?", None)

    assert r["permission_decision"] == "deny", r["permission_decision"]
    assert r["refusal_reason"] == "no_access", r["refusal_reason"]
    assert r["citations"] == []


def test_refusal_reason_filled_for_memeriksa_phrasing(monkeypatch):
    """Frasa "tidak bisa memeriksa" juga penolakan penuh.

    Terukur di :8890: "Apakah Maya absen hari ini?" dijawab "Saya tidak bisa
    memeriksa kehadiran Maya…", tool-nya memang ditolak, tetapi
    `refusal_reason` tetap null karena verbanya tidak ada di daftar.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    state = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return {"text": "", "finish_reason": "tool_calls", "ms": 1, "usage": {},
                    "model": "mercury-3",
                    "tool_calls": [{"id": "c1", "name": "get_team_attendance",
                                    "arguments": {}, "raw_arguments": "{}"}]}
        return {"text": ("Saya tidak bisa memeriksa kehadiran Maya hari ini. "
                         "Rekap kehadiran orang lain hanya untuk team lead, HR, "
                         "atau admin."),
                "tool_calls": [], "finish_reason": "stop", "ms": 1,
                "usage": {}, "model": "mercury-3"}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Apakah Maya absen hari ini?", None)

    assert r["permission_decision"] == "deny", r["permission_decision"]
    assert r["refusal_reason"] == "no_access", r["refusal_reason"]


def test_long_multiparagraph_refusal_is_total_when_no_data(monkeypatch):
    """Penolakan panjang tanpa data sah tetap penolakan penuh.

    `_looks_like_refusal` menganggap jawaban >400 karakter berparagraf sebagai
    parsial (karena blok kedua biasanya data sah). Untuk jawaban yang hanya
    memuat metadata organisasi, tidak ada informasi yang bisa hilang.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    text = (
        "Saya tidak bisa melihat data absensi Maya hari ini. Akun Anda "
        "(employee) tidak memiliki akses ke rekap kehadiran atau jadwal cuti "
        "karyawan lain — fitur itu hanya tersedia untuk team lead, HR, atau "
        "admin.\n\n"
        "Maya Anggraini (Backend Engineer) bukan bawahan langsung Anda, jadi "
        "datanya juga tidak tercakup dalam cakupan tim Anda. Informasi ini "
        "bisa Anda tanyakan langsung ke atasan tim atau ke HR."
    )
    state = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return {"text": "", "finish_reason": "tool_calls", "ms": 1, "usage": {},
                    "model": "mercury-3",
                    "tool_calls": [{"id": "c1", "name": "get_directory",
                                    "arguments": {}, "raw_arguments": "{}"}]}
        return {"text": text, "tool_calls": [], "finish_reason": "stop",
                "ms": 1, "usage": {}, "model": "mercury-3"}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Apakah Maya absen hari ini?", None)

    assert r["permission_decision"] == "deny", r["permission_decision"]
    assert r["refusal_reason"] == "no_access", r["refusal_reason"]


def test_partial_answer_with_melihat_phrasing_keeps_citations(monkeypatch):
    """Anti-regresi: frasa baru tidak boleh menghapus sitasi jawaban parsial."""
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", _llm_search_then(
        "Rincian gaji per grade tidak bisa saya lihat dari akun Anda.\n\n"
        "Dari kebijakan tunjangan: tunjangan transport dibayarkan bulanan.",
        query="tunjangan transport"))

    r = agent.run(conn, user, "Berapa kisaran gaji karyawan?", None)

    assert r["permission_decision"] == "deny", r["permission_decision"]
    assert r["refusal_reason"] is None, r["refusal_reason"]
    assert r["citations"], "sitasi sah tidak boleh dihapus"


def test_policy_question_naming_someone_is_not_denied(monkeypatch):
    """Pertanyaan PROSEDUR yang menyebut nama bukan permintaan data orang lain.

    "Apa aturan cuti untuk Maya?" dijawab dokumen kebijakan. Tanpa pengecualian
    ini, karyawan yang bertanya prosedur dicatat sebagai penolakan akses.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", _llm_search_then(
        "Pengajuan cuti minimal 3 hari kerja sebelum tanggal mulai dan "
        "disetujui Team Lead lalu HR.", query="aturan cuti"))

    r = agent.run(conn, user, "Apa aturan cuti untuk Maya?", None)

    assert r["permission_decision"] == "allow", r["permission_decision"]
    assert r["refusal_reason"] is None, r["refusal_reason"]


def test_probe_target_ignores_own_name_and_ambiguous_first_names():
    conn = make_conn()
    user = engine.get_user(conn, "u_maya")

    assert agent._probe_target(conn, user, "Berapa sisa cuti saya?") is None
    # "Dewi" hanya milik satu karyawan -> dikenali; "Dimas" juga unik.
    assert agent._probe_target(conn, user, "Sisa cuti Dewi?")[0] == "u_dewi"
    # Nama penanya sendiri tidak pernah menjadi target.
    assert agent._probe_target(conn, user, "Sisa cuti Maya?") is None


def test_speculative_capability_denial_does_not_taint_valid_answer(monkeypatch):
    """Denial dari tool yang tidak relevan tidak boleh menandai jawaban sah.

    Terukur di :8890: "Siapa yang mengelola backup?" — model memanggil
    `list_employees` (ditolak, status kepegawaian khusus HR/admin), lalu
    menjawab benar dari dokumen "Kebijakan Backup & Pemulihan Data". Audit
    mencatat `deny` sehingga UI memperlakukan jawaban bersitasi sebagai
    penolakan akses.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    state = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return {"text": "", "finish_reason": "tool_calls", "ms": 1, "usage": {},
                    "model": "mercury-3",
                    "tool_calls": [{"id": "c1", "name": "list_employees",
                                    "arguments": {}, "raw_arguments": "{}"}]}
        if state["n"] == 2:
            return {"text": "", "finish_reason": "tool_calls", "ms": 1, "usage": {},
                    "model": "mercury-3",
                    "tool_calls": [{"id": "c2", "name": "get_document",
                                    "arguments": {"doc_id": "doc_it_backup"},
                                    "raw_arguments": '{"doc_id":"doc_it_backup"}'}]}
        return {"text": ("Berdasarkan Kebijakan Backup & Pemulihan Data, "
                         "pengelolaan backup dilakukan oleh tim IT."),
                "tool_calls": [], "finish_reason": "stop", "ms": 1,
                "usage": {}, "model": "mercury-3"}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Siapa yang mengelola backup?", None)

    assert r["permission_decision"] == "allow", r["permission_decision"]
    assert r["refusal_reason"] is None, r["refusal_reason"]
    assert r["citations"], "sitasi dokumen yang sah harus dipertahankan"


def test_denial_classification_does_not_depend_on_model_wording():
    """`refusal_reason` tidak boleh bergantung diksi model.

    Terukur di :8890: model menulis "Data gaji karyawan lain tidak tercakup
    dalam akses akun Anda" — penolakan yang benar, tetapi tidak ada frasa di
    daftar sehingga `permission_decision=deny` sementara `refusal_reason`
    tetap null. UI lalu menampilkannya sebagai jawaban biasa.

    Keputusannya sekarang struktural: `denied` sudah diputuskan server, tidak
    ada sitasi, tidak ada draft, dan tidak ada tool data selain metadata
    organisasi -> penolakan penuh, apa pun kalimatnya.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")

    class _Ctx:
        evidence = {}
        prepared = None
        tool_data = ["get_team"]

    # Kalimat yang TIDAK ada di daftar frasa.
    assert agent._refusal_is_total(
        "Data gaji rekan kerja tidak tercakup dalam peran akun Anda.", _Ctx())

    # Penolakan panjang (mis. penjelasan + arahan "hubungi HR") tetap penolakan
    # penuh selama tidak ada data yang benar-benar disampaikan: arahan bukan
    # data, dan user tidak mendapat angka gaji yang ia minta.
    long_refusal = ("Saya tidak bisa melihat data gaji karyawan lain. " + "x " * 250
                    + "\n\nSebagai gantinya, hubungi HR untuk klarifikasi.")
    assert len(long_refusal) > 400
    assert agent._refusal_is_total(long_refusal, _Ctx())

    # Anti-regresi: ada sitasi / draft / data selain metadata -> bukan penolakan.
    class _Cited(_Ctx):
        evidence = {"doc_payslip_howto": {"content": "slip"}}

    assert not agent._refusal_is_total("tidak bisa melihat", _Cited())

    # Tool yang benar-benar mengembalikan DATA (bukan metadata organisasi)
    # berarti ada informasi sah di jawaban -> bukan penolakan penuh.
    class _Data(_Ctx):
        tool_data = ["get_team", "get_leave_balance"]

    assert not agent._refusal_is_total("tidak bisa melihat", _Data())

    # Draft yang menunggu konfirmasi juga bukan penolakan.
    class _Draft(_Ctx):
        prepared = {"summary": "Cuti 3-5 Nov"}

    assert not agent._refusal_is_total("tidak bisa melihat", _Draft())

    print("  ok denial_classification_does_not_depend_on_model_wording")


def test_relevant_capability_denial_still_denies(monkeypatch):
    """Anti-regresi: denial dari tool yang MEMANG topiknya tetap dihitung."""
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    state = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return {"text": "", "finish_reason": "tool_calls", "ms": 1, "usage": {},
                    "model": "mercury-3",
                    "tool_calls": [{"id": "c1", "name": "list_employees",
                                    "arguments": {}, "raw_arguments": "{}"}]}
        return {"text": "Daftar karyawan berstatus probation tidak tersedia untuk akun Anda.",
                "tool_calls": [], "finish_reason": "stop", "ms": 1,
                "usage": {}, "model": "mercury-3"}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Siapa saja karyawan yang masih berstatus probation?", None)

    assert r["permission_decision"] == "deny", r["permission_decision"]
    assert r["refusal_reason"] == "no_access", r["refusal_reason"]

def test_long_multiparagraph_refusal_survives_grounding_gate(monkeypatch):
    """Penolakan panjang berparagraf tidak boleh dibuang sebagai karangan.

    Terukur di :8890 untuk `u_budi` + "Slip gaji Maya?": model menolak dengan
    dua paragraf (paragraf kedua hanya ajakan), lalu `ungrounded_claim`
    membuangnya dan user menerima "tidak memiliki informasi". Itu menyiratkan
    datanya tidak ada, padahal datanya ada dan hanya perannya yang kurang.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    text = (
        "Maaf, saya tidak bisa menampilkan slip gaji Maya. Sebagai karyawan "
        "dengan peran Content Producer, akun Anda hanya memiliki akses ke "
        "slip gaji milik Anda sendiri. Data gaji karyawan lain tidak tersedia "
        "untuk peran ini.\n\n"
        "Kalau Anda butuh, saya bisa bantu menampilkan slip gaji Anda sendiri "
        "atau menjawab pertanyaan lain seputar cuti dan absensi."
    )
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", _llm_says(text))

    r = agent.run(conn, user, "Slip gaji Maya?", None)

    assert "tidak memiliki informasi" not in r["answer"].lower(), r["answer"][:160]
    assert "slip gaji" in r["answer"].lower(), r["answer"][:160]
    assert r["permission_decision"] == "deny", r["permission_decision"]
    assert r["refusal_reason"] == "no_access", r["refusal_reason"]


def test_long_refusal_exception_does_not_open_parametric_claims(monkeypatch):
    """Anti-regresi: pengecualian penolakan tidak boleh meloloskan karangan.

    Pembedanya adalah posisi: klaim parametrik tidak diawali frasa penolakan,
    jadi `_starts_with_refusal` tidak menyala dan gerbang tetap menolak.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    text = ("Presiden Indonesia saat ini adalah Prabowo Subianto. Beliau "
            "dilantik pada 20 Oktober 2024 setelah memenangkan pemilu.\n\n"
            "Kabinetnya dilantik pada 21 Oktober 2024 dengan 48 menteri.")
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", _llm_says(text))

    r = agent.run(conn, user, "Siapa presiden Indonesia?", None)

    assert r["llm_error"] == "agent:ungrounded_claim", r["llm_error"]
    assert "Prabowo" not in r["answer"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# Challenge A Uji 1 — penolakan simetris untuk proyek terbatas
# ---------------------------------------------------------------------------
# Aturan resmi: respons untuk proyek yang ADA-TETAPI-TERBATAS harus sama
# seperti respons untuk proyek yang TIDAK ADA. Kalau berbeda, perbedaannya
# sendiri mengonfirmasi bahwa proyek terbatas itu nyata — dan itu pelanggaran,
# sekalipun tidak satu pun detail proyeknya disebut.
_SYM_KEYS = ("answer", "citations", "permission_decision", "refusal_reason",
             "selected_sources", "conflict_detected", "pii_masked",
             "prompt_injection_detected", "suggestions")


def _snap(conn, uid, question):
    r = engine.answer_question(conn, uid, question)
    return {k: r.get(k) for k in _SYM_KEYS}


@pytest.mark.parametrize("uid", ["u_budi", "u_sari", "u_dewi"])
def test_restricted_project_is_denied_symmetrically(uid):
    """Kenari (ada, terbatas) == Zoro/Merpati (tidak ada), seluruh field."""
    conn = make_conn()
    restricted = _snap(conn, uid, "Apa status Proyek Kenari?")
    assert restricted["refusal_reason"] == "no_access", restricted["refusal_reason"]
    for fake in ("Apa status Proyek Zoro?", "Apa status Proyek Merpati?"):
        other = _snap(conn, uid, fake)
        diff = [k for k in _SYM_KEYS if restricted[k] != other.get(k)]
        assert not diff, f"{uid}: {diff} berbeda antara Kenari dan {fake}"


@pytest.mark.parametrize("uid", ["u_budi", "u_sari"])
def test_restricted_project_answer_never_mentions_the_name(uid):
    """Penolakan tidak boleh menyebut nama proyek — menyebutnya = mengonfirmasi."""
    conn = make_conn()
    r = engine.answer_question(conn, uid, "Apa status Proyek Kenari?")
    assert "Kenari" not in r["answer"], r["answer"]
    assert r["citations"] == [], r["citations"]


def test_restricted_project_denied_before_llm_runs(monkeypatch):
    """Gerbang harus menutup jalur LLM juga, bukan hanya engine deterministik.

    Tanpa ini model menyusun kalimatnya sendiri ("tidak tersedia di akun
    Anda…") dan justru mengonfirmasi keberadaan proyek terbatas.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    calls = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        calls["n"] += 1
        return {"text": "Status Proyek Kenari tidak tersedia di akun Anda.",
                "tool_calls": [], "finish_reason": "stop", "ms": 1,
                "usage": {}, "model": "mercury-3"}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Apa status Proyek Kenari?", None)

    assert calls["n"] == 0, "LLM tidak boleh dipanggil untuk proyek terbatas"
    assert r["refusal_reason"] == "no_access", r["refusal_reason"]
    assert "Kenari" not in r["answer"], r["answer"]


def test_public_partner_profile_is_still_answerable():
    """Gerbang proyek tidak boleh menutup profil mitra publik."""
    conn = make_conn()
    r = engine.answer_question(conn, "u_budi", "Apa itu Kenari?")
    assert r["refusal_reason"] is None, r["refusal_reason"]
    assert "mitra" in r["answer"].lower(), r["answer"]


@pytest.mark.parametrize("uid", ["u_andi", "u_rina"])
def test_project_owner_still_gets_the_answer(uid):
    """Yang berhak tetap mendapat jawaban lengkap — gerbang bukan blokir global."""
    conn = make_conn()
    r = engine.answer_question(conn, uid, "Apa status Proyek Kenari?")
    assert r["refusal_reason"] is None, r["refusal_reason"]
    assert "850" in r["answer"], r["answer"]
