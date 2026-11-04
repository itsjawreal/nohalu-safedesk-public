"""Sprint 1 — data personal: cuti, absensi, slip, org chart.

Semua tes offline (LLM dimatikan), DB in-memory baru.
"""
import os
import sys
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_LLM_ENABLED"] = "0"

import db
import personal
import portal
from engine import get_user, answer_question


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    db.seed_if_empty(conn)
    db.ensure_seed_users(conn)
    from seed import seed_personal
    seed_personal(conn)
    return conn


def _u(conn, uid):
    return get_user(conn, uid)


# ---------------------------------------------------------------------------
# Profil & org chart
# ---------------------------------------------------------------------------
def test_profile_has_manager_and_department():
    conn = make_conn()
    p = personal.profile(conn, _u(conn, "u_budi"))
    assert p["department"] == "Creative & Production"
    assert p["job_title"] == "Content Producer"
    assert p["manager"]["id"] == "u_andi"
    assert p["manager"]["name"] == "Andi Pratama"
    assert p["employee_no"] and p["email"]
    print("  ok profile_has_manager_and_department")


def test_org_chart_chain_and_reports():
    conn = make_conn()
    chart = personal.org_chart(conn, _u(conn, "u_budi"))
    assert chart["me"]["id"] == "u_budi"
    assert [c["id"] for c in chart["chain"]] == ["u_andi", "u_rina"]
    assert chart["reports"] == []
    andi = personal.org_chart(conn, _u(conn, "u_andi"))
    assert {r["id"] for r in andi["reports"]} >= {"u_budi", "u_fajar", "u_maya", "u_yusuf"}
    print("  ok org_chart_chain_and_reports")


def test_team_peers_and_reports():
    conn = make_conn()
    t = personal.team(conn, _u(conn, "u_budi"))
    assert t["manager"]["id"] == "u_andi"
    assert {p["id"] for p in t["peers"]} >= {"u_fajar", "u_maya", "u_yusuf"}
    assert t["reports"] == []
    print("  ok team_peers_and_reports")


def test_org_chart_does_not_leak_salary():
    conn = make_conn()
    blob = str(personal.org_chart(conn, _u(conn, "u_rina")))
    assert "Grade" not in blob and "6.000.000" not in blob
    assert "gaji" not in blob.lower()
    print("  ok org_chart_does_not_leak_salary")


# ---------------------------------------------------------------------------
# Saldo cuti
# ---------------------------------------------------------------------------
def _leave_payload(start, end, **kw):
    d = {"type": "cuti", "start_date": start, "end_date": end}
    d.update(kw)
    return d


def test_leave_balance_starts_from_entitlement():
    conn = make_conn()
    lb = personal.leave_balance(conn, _u(conn, "u_budi"))
    assert lb["entitlement"] == 12
    assert lb["used"] == 0 and lb["pending"] == 0
    assert lb["remaining"] == 12
    print("  ok leave_balance_starts_from_entitlement")


def test_pending_leave_reserves_balance():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    data, code = portal.create_request(conn, budi, _leave_payload("2027-03-01", "2027-03-03"))
    assert code == 201, data
    lb = personal.leave_balance(conn, budi, 2027)
    assert lb["pending"] == 3, lb
    assert lb["remaining"] == 9
    print("  ok pending_leave_reserves_balance")


def test_approved_leave_consumes_balance():
    conn = make_conn()
    budi, andi, sari = _u(conn, "u_budi"), _u(conn, "u_andi"), _u(conn, "u_sari")
    data, _ = portal.create_request(conn, budi, _leave_payload("2027-03-10", "2027-03-12"))
    assert portal.decide_request(conn, andi, data["id"], "approve", "ok")[1] == 200
    assert portal.decide_request(conn, sari, data["id"], "approve", "ok")[1] == 200
    lb = personal.leave_balance(conn, budi, 2027)
    assert lb["used"] == 3 and lb["pending"] == 0
    assert lb["remaining"] == 9
    print("  ok approved_leave_consumes_balance")


def test_rejected_and_cancelled_leave_release_balance():
    conn = make_conn()
    budi, andi = _u(conn, "u_budi"), _u(conn, "u_andi")
    a, _ = portal.create_request(conn, budi, _leave_payload("2027-04-01", "2027-04-02"))
    portal.decide_request(conn, andi, a["id"], "reject", "bentrok proyek")
    b, _ = portal.create_request(conn, budi, _leave_payload("2027-04-05", "2027-04-06"))
    portal.cancel_request(conn, budi, b["id"])
    lb = personal.leave_balance(conn, budi, 2027)
    assert lb["used"] == 0 and lb["pending"] == 0
    assert lb["remaining"] == 12
    print("  ok rejected_and_cancelled_leave_release_balance")


def test_carry_over_added_to_balance():
    conn = make_conn()
    lb = personal.leave_balance(conn, _u(conn, "u_sari"))
    assert lb["carry_over"] == 2
    assert lb["remaining"] == 14
    print("  ok carry_over_added_to_balance")


# ---------------------------------------------------------------------------
# Absensi
# ---------------------------------------------------------------------------
def test_clock_in_then_out():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    data, code = personal.clock(conn, budi, "in")
    assert code in (200, 201), (data, code)
    assert data["clock_in"]
    out, code2 = personal.clock(conn, budi, "out")
    assert code2 == 200 and out["clock_out"]
    print("  ok clock_in_then_out")


def test_clock_is_idempotent():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    first, _ = personal.clock(conn, budi, "in")
    again, code = personal.clock(conn, budi, "in")
    assert code == 200 and again["idempotent"] is True
    assert again["clock_in"] == first["clock_in"]
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM attendance WHERE user_id='u_budi' AND date=?",
        (personal.today().isoformat(),)).fetchone()["n"]
    assert n == 1, "clock-in ganda tidak boleh membuat baris kedua untuk hari yang sama"
    print("  ok clock_is_idempotent")


def test_clock_out_without_in_is_conflict():
    conn = make_conn()
    data, code = personal.clock(conn, _u(conn, "u_dimas"), "out")
    assert code == 409 and data["error"] == "not_clocked_in"
    print("  ok clock_out_without_in_is_conflict")


def test_attendance_summary_month_filter():
    conn = make_conn()
    a = personal.attendance_summary(conn, _u(conn, "u_budi"))
    assert a["month"].count("-") == 1
    assert a["work_start"] == "09:00"
    old = personal.attendance_summary(conn, _u(conn, "u_budi"), "2020-01")
    assert old["days"] == []
    print("  ok attendance_summary_month_filter")


# ---------------------------------------------------------------------------
# Slip gaji
# ---------------------------------------------------------------------------
def test_payslip_only_own():
    conn = make_conn()
    budi = personal.payslips(conn, _u(conn, "u_budi"))
    sari = personal.payslips(conn, _u(conn, "u_sari"))
    assert budi["count"] >= 1
    assert all(p["net"] == 7500000 - 850000 for p in budi["payslips"])
    assert {p["id"] for p in budi["payslips"]}.isdisjoint({p["id"] for p in sari["payslips"]})
    assert budi["payslips"][0]["net"] != sari["payslips"][0]["net"]
    print("  ok payslip_only_own")


def test_every_seed_user_has_payslip():
    """Setiap karyawan aktif punya slip: demo tidak boleh menjawab "belum ada".

    Bug nyata: hanya 4 dari 12 user punya slip, sehingga akun demo paling
    sering dipakai (Maya) selalu mendapat "belum ada slip gaji di sistem".
    """
    from seed import SEED_USERS
    conn = make_conn()
    for u in SEED_USERS:
        got = personal.payslips(conn, _u(conn, u["id"]))
        assert got["count"] >= 1, f"{u['id']} tidak punya slip"
        for p in got["payslips"]:
            assert p["gross"] > p["deductions"] > 0
            assert p["net"] == p["gross"] - p["deductions"]
    print("  ok every_seed_user_has_payslip")


def test_seed_payslip_matches_salary_grade():
    """Gross harus masuk rentang grade di `doc_salary_structure` (6–28 jt)."""
    conn = make_conn()
    for u in ("u_budi", "u_maya", "u_andi", "u_sari", "u_rina"):
        for p in personal.payslips(conn, _u(conn, u))["payslips"]:
            assert 6_000_000 <= p["gross"] <= 28_000_000, (u, p)
    print("  ok seed_payslip_matches_salary_grade")


def test_publish_payslip_requires_hr_or_admin():
    conn = make_conn()
    payload = {"user_id": "u_budi", "period": "2026-10", "gross": 7500000,
               "deductions": 850000}
    data, code = personal.publish_payslip(conn, _u(conn, "u_budi"), payload)
    assert code == 403 and data["error"] == "not_authorized"
    data, code = personal.publish_payslip(conn, _u(conn, "u_sari"), payload)
    assert code == 201, data
    got = personal.payslips(conn, _u(conn, "u_budi"))
    assert any(p["period"] == "2026-10" and p["net"] == 6650000 for p in got["payslips"])
    # notifikasi terkirim ke karyawan
    n = conn.execute("SELECT COUNT(*) AS n FROM notifications WHERE user_id='u_budi'"
                     " AND message LIKE '%2026-10%'").fetchone()["n"]
    assert n == 1
    print("  ok publish_payslip_requires_hr_or_admin")


def test_publish_payslip_validates_period():
    conn = make_conn()
    sari = _u(conn, "u_sari")
    data, code = personal.publish_payslip(conn, sari, {"user_id": "u_budi", "period": "2026-13"})
    assert code == 422 and data["error"] == "invalid_period"
    data, code = personal.publish_payslip(conn, sari, {"user_id": "nope", "period": "2026-10"})
    assert code == 404
    print("  ok publish_payslip_validates_period")


# ---------------------------------------------------------------------------
# Intent personal di chat
# ---------------------------------------------------------------------------
PERSONAL_QUESTIONS = [
    ("Berapa sisa cuti saya?", "leave_balance"),
    ("How many annual leave days do I have left?", "leave_balance"),
    ("Siapa atasan saya?", "manager"),
    ("Who is my manager?", "manager"),
    ("Siapa anggota tim saya?", "team"),
    ("Di mana slip gaji saya?", "payslip"),
    ("Bagaimana absensi saya bulan ini?", "attendance"),
    ("Jam berapa saya masuk besok?", "work_schedule"),
    ("What time do I start work tomorrow?", "work_schedule"),
]

# Pertanyaan KEBIJAKAN/PROSEDUR tidak boleh dibajak menjadi data personal.
POLICY_QUESTIONS = [
    "Bagaimana kebijakan absensi?",
    "Bagaimana cara melihat slip gaji?",
    "Apa prosedur pengajuan cuti?",
    "How do I request leave?",
    "Bagaimana alur approval reimbursement?",
    "Apa aturan lembur?",
]


# Pertanyaan nominal gaji orang lain BUKAN intent `team`/`manager`: struktur
# organisasi tidak boleh dipakai untuk menjawab pertanyaan gaji.
SALARY_NOT_PERSONAL = [
    "Berapa gaji anggota tim saya?",
    "Berapa gaji tim saya?",
    "Berapa salary anggota tim saya?",
    "Berapa penghasilan bawahan saya?",
]


def test_salary_question_is_not_answered_by_org_structure():
    for q in SALARY_NOT_PERSONAL:
        got = personal.detect_personal_intent(q)
        assert got is None, f"pertanyaan gaji dibajak jadi {got}: {q}"
    # Slip gaji milik sendiri tetap personal.
    assert personal.detect_personal_intent("Di mana slip gaji saya?") == "payslip"
    assert personal.detect_personal_intent("Berapa gaji saya bulan ini?") == "payslip"
    print("  ok salary_question_is_not_answered_by_org_structure")


def test_personal_intent_detection():
    for q, want in PERSONAL_QUESTIONS:
        got = personal.detect_personal_intent(q)
        assert got == want, (q, got, want)
    print("  ok personal_intent_detection")


def test_policy_question_not_hijacked_by_personal_intent():
    for q in POLICY_QUESTIONS:
        got = personal.detect_personal_intent(q)
        assert got is None, f"pertanyaan prosedural dibajak jadi {got}: {q}"
    print("  ok policy_question_not_hijacked_by_personal_intent")


def test_policy_answers_come_from_documents():
    conn = make_conn()
    cases = [
        ("Bagaimana kebijakan absensi?", "doc_attendance_policy", "clock-in"),
        ("Bagaimana cara melihat slip gaji?", "doc_payslip_howto", "Slip gaji"),
        ("Apa prosedur pengajuan cuti?", "doc_workflow_leave", "cuti"),
    ]
    for q, doc, needle in cases:
        r = answer_question(conn, "u_budi", q, use_llm=False)
        assert r["engine_used"] == "deterministic", (q, r["engine_used"])
        assert doc in r["selected_sources"], (q, r["selected_sources"])
        assert needle.lower() in r["answer"].lower(), (q, r["answer"][:120])
    print("  ok policy_answers_come_from_documents")


def test_work_schedule_answer_uses_profile_and_hours():
    conn = make_conn()
    r = answer_question(conn, "u_budi", "Jam berapa saya masuk besok?", use_llm=False)
    assert r["engine_used"] == "personal"
    assert r["personal_intent"] == "work_schedule"
    assert personal.WORK_START in r["answer"]
    assert "Jakarta" in r["answer"]
    assert len(r["suggestions"]) >= 3
    print("  ok work_schedule_answer_uses_profile_and_hours")


def test_personal_answers_use_data_not_policy():
    conn = make_conn()
    r = answer_question(conn, "u_budi", "Berapa sisa cuti saya?", use_llm=False)
    assert r["engine_used"] == "personal"
    assert "12" in r["answer"] and "sisa" in r["answer"].lower()
    assert "Cuti tahunan 12 hari" not in r["answer"]
    assert len(r["suggestions"]) >= 3

    r = answer_question(conn, "u_budi", "Siapa atasan saya?", use_llm=False)
    assert "Andi Pratama" in r["answer"]
    assert r["engine_used"] == "personal"

    r = answer_question(conn, "u_budi", "Di mana slip gaji saya?", use_llm=False)
    assert "2026-09" in r["answer"]
    print("  ok personal_answers_use_data_not_policy")


def test_personal_limitations_not_zero_documents():
    """Jawaban personal tidak boleh menyebut '0 dokumen' (menyesatkan)."""
    conn = make_conn()
    r = answer_question(conn, "u_budi", "Berapa sisa cuti saya?", use_llm=False)
    assert "0 dokumen" not in r["limitations"], r["limitations"]
    assert "data kepegawaian" in r["limitations"].lower(), r["limitations"]
    print("  ok personal_limitations_not_zero_documents")


def test_personal_answer_does_not_leak_other_users():
    """Jawaban slip hanya memuat angka milik penanya.

    Dulu uji ini bergantung pada "u_dimas tidak punya slip". Setelah seed
    dilengkapi (semua karyawan punya slip), ketergantungan itu justru
    menyembunyikan isolasi yang sebenarnya: yang harus dibuktikan adalah
    nominal Budi tidak muncul di jawaban Dimas.
    """
    conn = make_conn()
    r = answer_question(conn, "u_dimas", "Di mana slip gaji saya?", use_llm=False)
    assert "6.650.000" not in r["answer"], r["answer"]      # net Budi
    assert "7.500.000" not in r["answer"], r["answer"]      # gross Budi
    assert "6.380.000" in r["answer"], r["answer"]         # net Dimas sendiri
    print("  ok personal_answer_does_not_leak_other_users")


def test_non_personal_question_still_document_based():
    conn = make_conn()
    r = answer_question(conn, "u_budi", "Bagaimana alur pengajuan reimbursement?", use_llm=False)
    assert r["engine_used"] == "deterministic"
    assert "formulir" in r["answer"]
    print("  ok non_personal_question_still_document_based")


def test_personal_data_question_about_other_person_is_refused():
    """Pertanyaan data personal orang lain tidak dijawab dengan data penanya.

    Dulu "Berapa sisa cuti Maya?" dijawab dengan saldo cuti Budi: angka benar
    untuk subjek yang salah, tanpa penolakan. Itu kelas bug paling berbahaya
    di jalur personal karena user tidak punya cara tahu jawabannya bukan
    tentang Maya.
    """
    conn = make_conn()
    for q, label in (("Berapa sisa cuti Maya?", "cuti"),
                     ("Absensi Maya bulan ini?", "absensi"),
                     ("Slip gaji Maya?", "slip gaji")):
        r = answer_question(conn, "u_budi", q, use_llm=False)
        assert r["refusal_reason"] == "no_access", (q, r["refusal_reason"])
        assert "Sisa cuti Anda" not in r["answer"], (q, r["answer"][:160])
        assert "6.650.000" not in r["answer"], (q, r["answer"][:160])
        assert "karyawan lain" in r["answer"], (q, r["answer"][:160])
    # data sendiri tetap jalan
    r = answer_question(conn, "u_budi", "Berapa sisa cuti saya?", use_llm=False)
    assert r["engine_used"] == "personal", r["engine_used"]
    assert r["refusal_reason"] is None, r["refusal_reason"]
    assert "Sisa cuti Anda" in r["answer"], r["answer"][:160]
    print("  ok personal_data_question_about_other_person_is_refused")


def test_other_person_target_matches_full_and_unique_first_name():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    hit = personal.other_person_target(conn, budi, "Berapa sisa cuti Maya?")
    assert hit and hit[0] == "u_maya", hit
    hit = personal.other_person_target(conn, budi, "Sisa cuti Maya Lestari?")
    assert hit and hit[0] == "u_maya", hit
    # penanya sendiri tidak pernah jadi target
    assert personal.other_person_target(conn, budi, "Sisa cuti Budi?") is None
    # tanpa nama orang: bukan target
    assert personal.other_person_target(conn, budi, "Sisa cuti saya?") is None
    print("  ok other_person_target_matches_full_and_unique_first_name")


def _run_all():
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nALL PERSONAL TESTS PASS")


if __name__ == "__main__":
    _run_all()
