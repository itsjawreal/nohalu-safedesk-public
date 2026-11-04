"""Sprint 0 — self-approval tidak boleh membuat pengajuan macet.

Kasus nyata sebelum fix:
  A. HR ajukan cuti      -> TL approve -> HR approve(self) 403 -> STUCK
  B. TL ajukan cuti      -> TL approve(self) 403              -> macet di tahap 1
  C. Finance reimburse   -> TL approve -> Finance self 403    -> STUCK
"""
import os
import sys
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_LLM_ENABLED"] = "0"

import db
import portal
from engine import get_user


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    db.seed_if_empty(conn)
    db.ensure_seed_users(conn)
    return conn


def _u(conn, uid):
    return get_user(conn, uid)


def _row(conn, rid):
    return portal._request_row(
        conn.execute("SELECT * FROM requests WHERE id = ?", (rid,)).fetchone())


# ---------------------------------------------------------------------------
# A. HR mengajukan cuti (tahap 2 = HR = dirinya)
# ---------------------------------------------------------------------------
def test_hr_leave_skips_own_hr_stage():
    conn = make_conn()
    sari, andi = _u(conn, "u_sari"), _u(conn, "u_andi")
    data, code = portal.create_request(conn, sari, {
        "type": "cuti", "start_date": "2027-01-05", "end_date": "2027-01-06"})
    assert code == 201
    assert data["skipped_stages"] == ["hr"], data
    assert data["stages"] == ["team_lead"]

    res, code = portal.decide_request(conn, andi, data["id"], "approve", "ok")
    assert code == 200 and res["status"] == "approved", (code, res)
    row = _row(conn, data["id"])
    assert row["status"] == "approved"
    assert any(h["action"] == "skipped_self" for h in row["history"])
    print("  ok hr_leave_skips_own_hr_stage")


# ---------------------------------------------------------------------------
# B. Team lead mengajukan cuti (tahap 1 = TL = dirinya)
# ---------------------------------------------------------------------------
def test_team_lead_leave_skips_own_tl_stage():
    conn = make_conn()
    andi, sari = _u(conn, "u_andi"), _u(conn, "u_sari")
    data, code = portal.create_request(conn, andi, {
        "type": "cuti", "start_date": "2027-02-05", "end_date": "2027-02-06"})
    assert code == 201
    assert data["skipped_stages"] == ["team_lead"]
    assert data["stages"] == ["hr"]

    res, code = portal.decide_request(conn, sari, data["id"], "approve", "hr ok")
    assert code == 200 and res["status"] == "approved"
    print("  ok team_lead_leave_skips_own_tl_stage")


# ---------------------------------------------------------------------------
# C. Finance mengajukan reimbursement (tahap 2 = finance = dirinya)
# ---------------------------------------------------------------------------
def test_finance_reimbursement_skips_own_finance_stage():
    conn = make_conn()
    dewi, andi = _u(conn, "u_dewi"), _u(conn, "u_andi")
    data, code = portal.create_request(conn, dewi, {
        "type": "reimbursement", "amount": "2000000"})
    assert code == 201
    assert data["skipped_stages"] == ["finance"]
    assert data["stages"] == ["team_lead"]

    res, code = portal.decide_request(conn, andi, data["id"], "approve", "ok")
    assert code == 200 and res["status"] == "approved"
    print("  ok finance_reimbursement_skips_own_finance_stage")


# ---------------------------------------------------------------------------
# Admin mengajukan: tahap 1 TL (orang lain) tetap ada, tahap 2 finance (orang lain)
# ---------------------------------------------------------------------------
def test_admin_request_keeps_both_stages():
    conn = make_conn()
    rina, andi, dewi = _u(conn, "u_rina"), _u(conn, "u_andi"), _u(conn, "u_dewi")
    data, code = portal.create_request(conn, rina, {
        "type": "reimbursement", "amount": "2000000"})
    assert code == 201
    assert data["skipped_stages"] == []
    assert data["stages"] == ["team_lead", "finance"]
    assert portal.decide_request(conn, andi, data["id"], "approve", "ok")[1] == 200
    res, code = portal.decide_request(conn, dewi, data["id"], "approve", "ok")
    assert code == 200 and res["status"] == "approved"
    print("  ok admin_request_keeps_both_stages")


# ---------------------------------------------------------------------------
# Karyawan biasa: rantai normal tidak berubah
# ---------------------------------------------------------------------------
def test_employee_chain_unchanged():
    conn = make_conn()
    budi, andi, sari = _u(conn, "u_budi"), _u(conn, "u_andi"), _u(conn, "u_sari")
    data, code = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": "2027-05-04", "end_date": "2027-05-05"})
    assert code == 201 and data["skipped_stages"] == []
    assert data["stages"] == ["team_lead", "hr"]
    assert portal.decide_request(conn, andi, data["id"], "approve", "ok")[1] == 200
    res, code = portal.decide_request(conn, sari, data["id"], "approve", "ok")
    assert code == 200 and res["status"] == "approved"
    print("  ok employee_chain_unchanged")


# ---------------------------------------------------------------------------
# Bila ada rekan se-role, pemisahan tugas tetap (tidak dilewati)
# ---------------------------------------------------------------------------
def test_second_team_lead_prevents_skip():
    conn = make_conn()
    conn.execute("INSERT INTO users (id,name,role,allowed_workspaces) VALUES (?,?,?,?)",
                 ("u_tl2", "Team Lead Dua", "team_lead", '["ws_onboarding"]'))
    conn.commit()
    andi = _u(conn, "u_andi")
    data, code = portal.create_request(conn, andi, {
        "type": "cuti", "start_date": "2027-06-01", "end_date": "2027-06-02"})
    assert code == 201
    assert data["skipped_stages"] == [], "ada TL lain, tahap tidak boleh dilewati"
    assert data["stages"] == ["team_lead", "hr"]
    # TL sendiri tetap tidak boleh menyetujui
    res, code = portal.decide_request(conn, andi, data["id"], "approve", "x")
    assert code == 403 and res["error"] == "cannot_decide_own"
    # rekan TL boleh
    res, code = portal.decide_request(conn, _u(conn, "u_tl2"), data["id"], "approve", "ok")
    assert code == 200 and res["status"] == "pending"
    print("  ok second_team_lead_prevents_skip")


# ---------------------------------------------------------------------------
# repair_self_stuck: data lama yang sudah terlanjur macet
# ---------------------------------------------------------------------------
def test_repair_self_stuck_fixes_legacy_rows():
    conn = make_conn()
    # Simulasi baris lama: HR ajukan cuti, tahap hr masih menunggu dirinya.
    conn.execute(
        """INSERT INTO requests (user_id,type,title,amount,start_date,end_date,note,
           status,approver_roles,stages,stage_idx,history,created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        ("u_sari", "cuti", "Cuti HR lama", None, "2027-07-01", "2027-07-02", None,
         "pending", '["team_lead","hr"]', '["team_lead","hr"]', 1, '[]',
         "2026-10-30T00:00:00Z"))
    conn.commit()
    fixed = portal.repair_self_stuck(conn)
    assert fixed, "harus ada pengajuan yang diperbaiki"
    row = conn.execute("SELECT * FROM requests").fetchone()
    assert row["status"] == "approved"
    assert row["self_approved"] == 1
    print("  ok repair_self_stuck_fixes_legacy_rows")


def test_repair_is_idempotent():
    conn = make_conn()
    conn.execute(
        """INSERT INTO requests (user_id,type,title,status,approver_roles,stages,
           stage_idx,history,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
        ("u_sari", "cuti", "x", "pending", '["team_lead","hr"]',
         '["team_lead","hr"]', 1, '[]', "2026-10-30T00:00:00Z"))
    conn.commit()
    first = portal.repair_self_stuck(conn)
    second = portal.repair_self_stuck(conn)
    assert first and second == []
    print("  ok repair_is_idempotent")


def test_request_row_exposes_skipped_stages():
    """UI butuh tahu tahap mana yang dilewati (transparansi governance)."""
    conn = make_conn()
    sari = _u(conn, "u_sari")
    data, _ = portal.create_request(conn, sari, {
        "type": "cuti", "start_date": "2027-08-05", "end_date": "2027-08-06"})
    row = _row(conn, data["id"])
    assert row["skipped_stages"] == ["hr"]
    assert row["stages"] == ["team_lead"]
    print("  ok request_row_exposes_skipped_stages")


def test_self_approval_notifies_admin_watcher():
    """Tahap yang dilewati harus terlihat oleh admin (bukan diam-diam)."""
    conn = make_conn()
    sari, rina = _u(conn, "u_sari"), _u(conn, "u_rina")
    portal.create_request(conn, sari, {
        "type": "cuti", "start_date": "2027-09-05", "end_date": "2027-09-06"})
    notes = portal.notifications(conn, rina)
    assert any("melewati tahap" in i["message"] for i in notes["items"]), notes
    print("  ok self_approval_notifies_admin_watcher")


def test_all_stages_self_approve_at_creation(monkeypatch=None):
    """Bila SEMUA tahap hanya bisa diisi pengaju, pengajuan langsung approved."""
    import portal as portal_mod
    conn = make_conn()
    andi = _u(conn, "u_andi")
    orig = portal_mod._approver_roles
    portal_mod._approver_roles = lambda t, a: ["team_lead"]
    try:
        data, code = portal_mod.create_request(conn, andi, {
            "type": "reimbursement", "amount": "250000"})
    finally:
        portal_mod._approver_roles = orig
    assert code == 201
    assert data["status"] == "approved"
    assert data["self_approved"] is True
    assert data["stages"] == []
    row = _row(conn, data["id"])
    assert row["status"] == "approved" and row["self_approved"] is True
    assert row["decided_by"] == "u_andi"
    print("  ok all_stages_self_approve_at_creation")


def test_resolve_stages_helper():
    conn = make_conn()
    andi = _u(conn, "u_andi")
    kept, skipped = portal._resolve_stages(conn, andi, ["team_lead", "hr"])
    assert kept == ["hr"] and skipped == ["team_lead"]
    kept, skipped = portal._resolve_stages(conn, _u(conn, "u_budi"),
                                           ["team_lead", "hr"])
    assert kept == ["team_lead", "hr"] and skipped == []
    print("  ok resolve_stages_helper")


def test_legacy_skip_persists_even_when_decide_forbidden():
    """Perbaikan tahap tidak boleh hilang saat approver yang salah memanggil decide."""
    conn = make_conn()
    conn.execute(
        """INSERT INTO requests (user_id,type,title,status,approver_roles,stages,
           stage_idx,history,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
        ("u_sari", "cuti", "Cuti HR lama", "pending", '["team_lead","hr"]',
         '["team_lead","hr"]', 1, '[]', "2026-10-30T00:00:00Z"))
    conn.commit()
    rid = conn.execute("SELECT id FROM requests").fetchone()["id"]
    # Budi (employee) tidak berhak dan tidak boleh memicu perbaikan lewat efek samping.
    res, code = portal.decide_request(conn, _u(conn, "u_budi"), rid, "approve", "x")
    assert code == 403, res
    row = conn.execute("SELECT * FROM requests WHERE id=?", (rid,)).fetchone()
    assert row["status"] == "pending", "pengguna tak berwenang tidak boleh mengubah status"

    # Pemilik (HR) yang memicu: tahap self-approval dibereskan dan langsung approved.
    res, code = portal.decide_request(conn, _u(conn, "u_sari"), rid, "approve", "x")
    assert code == 200 and res["status"] == "approved", (code, res)
    row = conn.execute("SELECT * FROM requests WHERE id=?", (rid,)).fetchone()
    assert row["self_approved"] == 1
    print("  ok legacy_skip_persists_even_when_decide_forbidden")
