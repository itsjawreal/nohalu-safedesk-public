"""Unit tests for portal.py — dashboard, directory, requests, approval, notifikasi.

Run: python tests/test_portal.py   (atau pytest)
Semua offline: LLM dimatikan, DB in-memory.
"""
import json
import os
import sys
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_LLM_ENABLED"] = "0"

import db
import portal
from engine import get_user, filter_allowed_docs, effective_roles


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    db.seed_if_empty(conn)
    db.ensure_seed_users(conn)
    return conn


def _u(conn, uid):
    return get_user(conn, uid)


# ---------------------------------------------------------------------------
# Role / ACL
# ---------------------------------------------------------------------------
def test_finance_role_exists_and_inherits():
    conn = make_conn()
    dewi = _u(conn, "u_dewi")
    assert dewi is not None, "user finance (u_dewi) harus ter-seed"
    assert dewi["role"] == "finance"
    roles = effective_roles("finance")
    assert "employee" in roles and "team_lead" in roles
    # finance harus bisa baca SOP/approval matrix
    docs = {d["id"] for d in filter_allowed_docs(conn, dewi)}
    assert "doc_approval_matrix" in docs
    assert "doc_workflow_approval" in docs
    print("  ok finance_role_exists_and_inherits")


def test_finance_cannot_read_salary_structure():
    """Struktur gaji tetap hr/admin; finance tidak boleh mewarisi itu."""
    conn = make_conn()
    dewi = _u(conn, "u_dewi")
    docs = {d["id"] for d in filter_allowed_docs(conn, dewi)}
    assert "doc_salary_structure" not in docs, "finance tidak boleh lihat struktur gaji"
    print("  ok finance_cannot_read_salary_structure")


def test_employee_still_blocked():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    docs = {d["id"] for d in filter_allowed_docs(conn, budi)}
    assert "doc_project_kenari" not in docs
    assert "doc_hr_contacts" not in docs
    print("  ok employee_still_blocked")


# ---------------------------------------------------------------------------
# Approval routing
# ---------------------------------------------------------------------------
def test_approver_roles_threshold():
    """Rantai reimbursement mengikuti doc_approval_matrix, bukan selalu Finance.

    "Reimbursement sampai Rp1.000.000 disetujui Team Lead. Di atas
    Rp1.000.000 disetujui Finance Manager." Sebelumnya kode selalu menambah
    tahap finance, sehingga agent menjawab "tidak perlu Finance" sementara
    pengajuannya tetap duduk di meja Finance.
    """
    assert portal._approver_roles("cuti", None) == ["team_lead", "hr"]
    assert portal._approver_roles("reimbursement", "500000") == ["team_lead"]
    assert portal._approver_roles("reimbursement", "1000000") == ["team_lead"]
    assert portal._approver_roles("reimbursement", "1000001") == ["team_lead", "finance"]
    assert portal._approver_roles("reimbursement", "2000000") == ["team_lead", "finance"]
    print("  ok approver_roles_threshold")


def test_parse_amount():
    assert portal.parse_amount("2.000.000") == 2000000
    assert portal.parse_amount("Rp 750.000") == 750000
    assert portal.parse_amount(1500000) == 1500000
    assert portal.parse_amount(None) == 0
    assert portal.parse_amount("") == 0
    print("  ok parse_amount")


def test_finance_manager_flag():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    small, _ = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "500000", "note": "kecil"})
    big, _ = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "2000000", "note": "besar"})
    rows = {r["id"]: r for r in
            (portal._request_row(x) for x in conn.execute("SELECT * FROM requests").fetchall())}
    assert rows[small["id"]]["finance_manager_required"] is False
    assert rows[big["id"]]["finance_manager_required"] is True
    print("  ok finance_manager_flag")


# ---------------------------------------------------------------------------
# Full approval flow
# ---------------------------------------------------------------------------
def test_reimbursement_two_stage_flow():
    """Nominal di atas ambang: Team Lead lalu Finance Manager."""
    conn = make_conn()
    budi, andi, dewi = _u(conn, "u_budi"), _u(conn, "u_andi"), _u(conn, "u_dewi")
    data, code = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "1750000", "note": "e2e"})
    assert code == 201 and data["stages"] == ["team_lead", "finance"]
    rid = data["id"]

    # tahap 1: team lead
    res, code = portal.decide_request(conn, andi, rid, "approve", "tl ok")
    assert code == 200 and res["status"] == "pending" and res["stage_idx"] == 1

    # tahap 2: finance
    res, code = portal.decide_request(conn, dewi, rid, "approve", "fin ok")
    assert code == 200 and res["status"] == "approved"

    row = portal._request_row(
        conn.execute("SELECT * FROM requests WHERE id = ?", (rid,)).fetchone())
    assert row["status"] == "approved"
    assert len(row["history"]) == 2
    assert row["history"][0]["role"] == "team_lead"
    assert row["history"][1]["role"] == "finance"
    assert row["current_role"] is None
    print("  ok reimbursement_two_stage_flow")


def test_cannot_decide_own_or_wrong_role():
    conn = make_conn()
    budi, andi, sari = _u(conn, "u_budi"), _u(conn, "u_andi"), _u(conn, "u_sari")
    data, _ = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "100000", "note": "x"})
    rid = data["id"]
    res, code = portal.decide_request(conn, budi, rid, "approve", None)
    assert code == 403 and res["error"] == "cannot_decide_own"
    res, code = portal.decide_request(conn, sari, rid, "approve", None)
    assert code == 403 and res["error"] == "not_authorized"
    print("  ok cannot_decide_own_or_wrong_role")


def test_reject_stops_flow():
    conn = make_conn()
    budi, andi = _u(conn, "u_budi"), _u(conn, "u_andi")
    data, _ = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "100000", "note": "x"})
    rid = data["id"]
    res, code = portal.decide_request(conn, andi, rid, "reject", "tidak lengkap")
    assert code == 200 and res["status"] == "rejected"
    row = portal._request_row(
        conn.execute("SELECT * FROM requests WHERE id = ?", (rid,)).fetchone())
    assert row["status"] == "rejected"
    assert row["decision_note"] == "tidak lengkap"
    print("  ok reject_stops_flow")


def test_cuti_flow_requires_dates():
    conn = make_conn()
    budi, andi, sari = _u(conn, "u_budi"), _u(conn, "u_andi"), _u(conn, "u_sari")
    res, code = portal.create_request(conn, budi, {"type": "cuti"})
    # 422 (unprocessable) lebih tepat daripada 400 untuk field wajib yang kosong.
    assert code == 422, "cuti tanpa tanggal harus ditolak"
    assert res["error"] == "dates_required"

    data, code = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": "2026-12-10", "end_date": "2026-12-12"})
    assert code == 201 and data["stages"] == ["team_lead", "hr"]
    rid = data["id"]
    _, code = portal.decide_request(conn, andi, rid, "approve", "ok")
    assert code == 200
    res, code = portal.decide_request(conn, sari, rid, "approve", "hr ok")
    assert code == 200 and res["status"] == "approved"
    print("  ok cuti_flow_requires_dates")


# ---------------------------------------------------------------------------
# Notifikasi
# ---------------------------------------------------------------------------
def test_notifications_on_create_and_decide():
    conn = make_conn()
    budi, andi, dewi = _u(conn, "u_budi"), _u(conn, "u_andi"), _u(conn, "u_dewi")
    data, _ = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "1750000", "note": "n"})
    rid = data["id"]

    n_tl = portal.notifications(conn, andi)
    assert n_tl["unread"] >= 1, "team lead harus dapat notifikasi pengajuan baru"

    portal.decide_request(conn, andi, rid, "approve", "ok")
    n_fin = portal.notifications(conn, dewi)
    assert n_fin["unread"] >= 1, "finance harus dapat notifikasi tahap berikutnya"

    portal.decide_request(conn, dewi, rid, "approve", "ok")
    n_budi = portal.notifications(conn, budi)
    assert any("disetujui" in i["message"] for i in n_budi["items"])
    print("  ok notifications_on_create_and_decide")


def test_notification_mark_read():
    conn = make_conn()
    budi, andi = _u(conn, "u_budi"), _u(conn, "u_andi")
    portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "100000", "note": "x"})
    n = portal.notifications(conn, andi)
    nid = n["items"][0]["id"]
    res = portal.mark_notification_read(conn, andi, nid)
    assert res.get("ok") is True or res.get("id") == nid
    after = portal.notifications(conn, andi)
    assert after["unread"] == n["unread"] - 1
    print("  ok notification_mark_read")


# ---------------------------------------------------------------------------
# Listing / RBAC portal
# ---------------------------------------------------------------------------
def test_list_requests_visibility():
    conn = make_conn()
    budi, andi, rina, dewi = (_u(conn, "u_budi"), _u(conn, "u_andi"),
                              _u(conn, "u_rina"), _u(conn, "u_dewi"))
    portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "250000", "note": "x"})

    b = portal.list_requests(conn, budi)
    assert len(b["mine"]) == 1 and b["approval"] == [] and b["all"] == []

    a = portal.list_requests(conn, andi)
    assert len(a["approval"]) == 1, "team lead melihat yang menunggu dia"

    r = portal.list_requests(conn, rina)
    assert len(r["all"]) == 1, "admin melihat semua"

    d = portal.list_requests(conn, dewi)
    assert d["approval"] == [], "finance belum giliran (masih di team lead)"
    print("  ok list_requests_visibility")


def test_dashboard_shapes():
    conn = make_conn()
    for uid in ("u_budi", "u_sari", "u_andi", "u_rina", "u_dewi"):
        u = _u(conn, uid)
        d = portal.dashboard(conn, u)
        assert d["user"]["id"] == uid
        assert "workspaces" in d and "requests" in d
        assert d["available_documents"] >= 0
    # admin/hr punya org stats
    assert "org" in portal.dashboard(conn, _u(conn, "u_rina"))
    assert "org" in portal.dashboard(conn, _u(conn, "u_sari"))
    assert "org" not in portal.dashboard(conn, _u(conn, "u_budi"))
    print("  ok dashboard_shapes")


def test_dashboard_approval_upcoming_not_hidden():
    """HR/finance harus melihat pengajuan yang SEDANG menuju mejanya.

    Regresi nyata: `approval_waiting` hanya berisi tahap berjalan, sehingga HR
    melihat "0 antrean" padahal sembilan pengajuan sedang di tahap Team Lead
    dan akan sampai ke HR. Angka nol itu benar secara harfiah dan menyesatkan
    secara operasional.
    """
    conn = make_conn()
    budi, sari, dewi, andi = (_u(conn, "u_budi"), _u(conn, "u_sari"),
                              _u(conn, "u_dewi"), _u(conn, "u_andi"))
    cuti, code = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": "2027-06-01", "end_date": "2027-06-02"})
    assert code == 201
    reimb, code = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "1200000"})
    assert code == 201

    d_hr = portal.dashboard(conn, sari)
    assert d_hr["approval_waiting"] == [], "belum giliran HR"
    ups = {u["id"]: u for u in d_hr["approval_upcoming"]}
    assert cuti["id"] in ups, "cuti harus terlihat sebagai antrean mendatang HR"
    assert ups[cuti["id"]]["waiting_on"] == "team_lead"
    assert ups[cuti["id"]]["position"] == 1

    d_fin = portal.dashboard(conn, dewi)
    fups = {u["id"]: u for u in d_fin["approval_upcoming"]}
    assert reimb["id"] in fups and fups[reimb["id"]]["waiting_on"] == "team_lead"

    # Team lead memutuskan tahap 1: pengajuan pindah ke meja HR/finance.
    assert portal.decide_request(conn, andi, cuti["id"], "approve", "ok")[1] == 200
    assert portal.decide_request(conn, andi, reimb["id"], "approve", "ok")[1] == 200
    d_hr = portal.dashboard(conn, sari)
    assert [w["id"] for w in d_hr["approval_waiting"]] == [cuti["id"]]
    assert all(u["id"] != cuti["id"] for u in d_hr["approval_upcoming"])
    print("  ok dashboard_approval_upcoming_not_hidden")


def test_dashboard_team_summary_for_lead():
    """Team lead mendapat ringkasan bawahannya; karyawan biasa tidak."""
    conn = make_conn()
    andi, budi = _u(conn, "u_andi"), _u(conn, "u_budi")
    d = portal.dashboard(conn, andi)
    assert "team" in d, "team lead punya bawahan langsung"
    t = d["team"]
    assert t["count"] == len(t["members"]) and t["count"] >= 1
    ids = {m["id"] for m in t["members"]}
    assert "u_budi" in ids
    assert "team" not in portal.dashboard(conn, budi)
    # Ringkasan finance hanya untuk peran yang memang memutus reimbursement.
    assert "finance" in portal.dashboard(conn, _u(conn, "u_dewi"))
    assert "finance" not in portal.dashboard(conn, _u(conn, "u_sari"))
    print("  ok dashboard_team_summary_for_lead")


def test_payroll_salary_rbac():
    """Struktur gaji hanya hr/admin. Finance butuh approval nominal, bukan band gaji."""
    conn = make_conn()
    assert portal.payroll(conn, _u(conn, "u_budi"))["salary_structure"] is None
    assert portal.payroll(conn, _u(conn, "u_sari"))["salary_structure"]
    assert portal.payroll(conn, _u(conn, "u_rina"))["salary_structure"]
    assert portal.payroll(conn, _u(conn, "u_dewi"))["salary_structure"] is None
    # jadwal pembayaran tetap terlihat semua role (info umum)
    for uid in ("u_budi", "u_sari", "u_andi", "u_rina", "u_dewi"):
        assert portal.payroll(conn, _u(conn, uid))["schedule"]
    print("  ok payroll_salary_rbac")


def test_directory_hides_contact_for_employee():
    conn = make_conn()
    emp = portal.directory(conn, _u(conn, "u_budi"))
    assert emp["contact_visible"] is False
    assert emp["contact_note"] is None
    hr = portal.directory(conn, _u(conn, "u_sari"))
    assert hr["contact_visible"] is True
    print("  ok directory_hides_contact_for_employee")


def test_repair_overrouted_finance_fixes_legacy_rows():
    """Baris lama yang masih membawa tahap Finance harus dibereskan.

    Sebelum `_approver_roles` mengikuti doc_approval_matrix, setiap
    reimbursement dibuat dengan rantai `["team_lead", "finance"]`. Baris itu
    tetap pending dan menahan pengaju menunggu approver yang tidak dibutuhkan.
    """
    conn = make_conn()
    budi, andi, dewi = _u(conn, "u_budi"), _u(conn, "u_andi"), _u(conn, "u_dewi")
    data, _ = portal.create_request(conn, budi, {"type": "reimbursement", "amount": "2000000"})
    rid = data["id"]
    # Tiru bentuk baris lama: rantai panjang, masih di tahap Team Lead.
    conn.execute(
        "UPDATE requests SET stages=?, stage_idx=0, status='pending' WHERE id=?",
        (json.dumps(["team_lead", "finance"]), rid))
    conn.commit()
    assert portal.repair_overrouted_finance(conn) == []
    assert portal.repair_overrouted_finance(conn) == [], "nominal di atas ambang tidak disentuh"

    small, _ = portal.create_request(conn, budi, {"type": "reimbursement", "amount": "120000"})
    conn.execute(
        "UPDATE requests SET stages=?, stage_idx=0, status='pending' WHERE id=?",
        (json.dumps(["team_lead", "finance"]), small["id"]))
    conn.commit()
    fixed = portal.repair_overrouted_finance(conn)
    assert fixed == [small["id"]], fixed
    row = portal._request_row(
        conn.execute("SELECT * FROM requests WHERE id=?", (small["id"],)).fetchone())
    assert row["stages"] == ["team_lead"], "tahap Finance harus hilang"
    assert row["status"] == "pending", "masih menunggu Team Lead"
    assert row["history"][-1]["action"] == "stage_removed"
    # Idempotent: putaran kedua tidak mengubah apa pun lagi.
    assert portal.repair_overrouted_finance(conn) == []

    # Bila pengajuan sudah duduk DI tahap Finance, melepas tahap itu
    # menyelesaikannya — bukan menyisakan pengajuan yang tak bisa diputuskan.
    other, _ = portal.create_request(conn, budi, {"type": "reimbursement", "amount": "90000"})
    conn.execute(
        "UPDATE requests SET stages=?, stage_idx=1, status='pending' WHERE id=?",
        (json.dumps(["team_lead", "finance"]), other["id"]))
    conn.commit()
    assert portal.repair_overrouted_finance(conn) == [other["id"]]
    row = portal._request_row(
        conn.execute("SELECT * FROM requests WHERE id=?", (other["id"],)).fetchone())
    assert row["status"] == "approved" and row["current_role"] is None
    print("  ok repair_overrouted_finance_fixes_legacy_rows")


def test_ensure_seed_users_idempotent():
    conn = make_conn()
    before = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    db.ensure_seed_users(conn)   # tidak boleh menduplikasi
    db.ensure_seed_users(conn)
    after = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]
    assert before == after
    print("  ok ensure_seed_users_idempotent")


def _run_all():
    test_finance_role_exists_and_inherits()
    test_finance_cannot_read_salary_structure()
    test_employee_still_blocked()
    test_approver_roles_threshold()
    test_parse_amount()
    test_finance_manager_flag()
    test_reimbursement_two_stage_flow()
    test_cannot_decide_own_or_wrong_role()
    test_reject_stops_flow()
    test_cuti_flow_requires_dates()
    test_notifications_on_create_and_decide()
    test_notification_mark_read()
    test_list_requests_visibility()
    test_dashboard_shapes()
    test_dashboard_approval_upcoming_not_hidden()
    test_dashboard_team_summary_for_lead()
    test_payroll_salary_rbac()
    test_directory_hides_contact_for_employee()
    test_ensure_seed_users_idempotent()
    test_repair_overrouted_finance_fixes_legacy_rows()
    print("\nALL PORTAL TESTS PASS")


if __name__ == "__main__":
    _run_all()
