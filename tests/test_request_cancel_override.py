"""Sprint 0 — cancel (pengaju) & override (admin/HR) + audit event."""
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


def _audit_events(conn):
    return [r["event_type"] for r in
            conn.execute("SELECT event_type FROM audit_log").fetchall()]


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------
def test_owner_can_cancel_pending():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    data, _ = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "150000"})
    res, code = portal.cancel_request(conn, budi, data["id"])
    assert code == 200 and res["status"] == "cancelled"
    row = portal._request_row(
        conn.execute("SELECT * FROM requests WHERE id=?", (data["id"],)).fetchone())
    assert row["status"] == "cancelled"
    assert row["cancelled_at"]
    assert row["history"][-1]["action"] == "cancelled"
    print("  ok owner_can_cancel_pending")


def test_cancel_then_decide_rejected():
    conn = make_conn()
    budi, andi = _u(conn, "u_budi"), _u(conn, "u_andi")
    data, _ = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "150000"})
    portal.cancel_request(conn, budi, data["id"])
    res, code = portal.decide_request(conn, andi, data["id"], "approve", "ok")
    assert code == 400 and res["error"] == "already_decided"
    print("  ok cancel_then_decide_rejected")


def test_cancel_only_owner_and_only_pending():
    conn = make_conn()
    budi, andi = _u(conn, "u_budi"), _u(conn, "u_andi")
    data, _ = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "2000000"})
    res, code = portal.cancel_request(conn, andi, data["id"])
    assert code == 404, "pengguna lain tidak boleh tahu pengajuan itu ada"
    portal.decide_request(conn, andi, data["id"], "approve", "ok")
    # lolos ke finance, masih pending -> tetap bisa dibatalkan pemilik
    res, code = portal.cancel_request(conn, budi, data["id"])
    assert code == 200
    # sudah cancelled -> tidak bisa dua kali
    res, code = portal.cancel_request(conn, budi, data["id"])
    assert code == 400 and res["error"] == "not_cancellable"
    print("  ok cancel_only_owner_and_only_pending")


def test_cancel_unknown_id():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    res, code = portal.cancel_request(conn, budi, 9999)
    assert code == 404
    print("  ok cancel_unknown_id")


def test_cancel_notifies_current_approver():
    conn = make_conn()
    budi, andi = _u(conn, "u_budi"), _u(conn, "u_andi")
    data, _ = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "150000"})
    before = portal.notifications(conn, andi)["unread"]
    portal.cancel_request(conn, budi, data["id"])
    after = portal.notifications(conn, andi)
    assert after["unread"] == before + 1
    assert any("dibatalkan" in i["message"] for i in after["items"])
    print("  ok cancel_notifies_current_approver")


# ---------------------------------------------------------------------------
# Override
# ---------------------------------------------------------------------------
def test_override_requires_note_and_role():
    conn = make_conn()
    budi, andi, rina = _u(conn, "u_budi"), _u(conn, "u_andi"), _u(conn, "u_rina")
    data, _ = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "150000"})
    res, code = portal.override_request(conn, andi, data["id"], "approve", "x" * 20)
    assert code == 403 and res["error"] == "not_authorized"
    res, code = portal.override_request(conn, rina, data["id"], "approve", "pendek")
    assert code == 422 and res["error"] == "note_required"
    res, code = portal.override_request(conn, rina, data["id"], "approve",
                                        "Approver tidak tersedia, disetujui manual.")
    assert code == 200 and res["status"] == "approved"
    print("  ok override_requires_note_and_role")


def test_override_records_audit_and_history():
    conn = make_conn()
    budi, sari = _u(conn, "u_budi"), _u(conn, "u_sari")
    data, _ = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "150000"})
    portal.override_request(conn, sari, data["id"], "reject",
                            "Ditolak karena bukti tidak lengkap.")
    row = portal._request_row(
        conn.execute("SELECT * FROM requests WHERE id=?", (data["id"],)).fetchone())
    assert row["status"] == "rejected"
    assert row["overridden_by"] == "u_sari"
    assert row["history"][-1]["action"] == "override_reject"
    assert "approval_override" in _audit_events(conn)
    print("  ok override_records_audit_and_history")


def test_override_after_decided_rejected():
    conn = make_conn()
    budi, andi, rina = _u(conn, "u_budi"), _u(conn, "u_andi"), _u(conn, "u_rina")
    data, _ = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "150000"})
    portal.decide_request(conn, andi, data["id"], "reject", "tidak lengkap")
    res, code = portal.override_request(conn, rina, data["id"], "approve",
                                        "Membatalkan penolakan sebelumnya.")
    assert code == 400 and res["error"] == "already_decided"
    print("  ok override_after_decided_rejected")


# ---------------------------------------------------------------------------
# Audit event untuk alur pengajuan
# ---------------------------------------------------------------------------
def test_request_events_logged():
    conn = make_conn()
    budi, andi = _u(conn, "u_budi"), _u(conn, "u_andi")
    data, _ = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "150000"})
    portal.decide_request(conn, andi, data["id"], "approve", "ok")
    events = _audit_events(conn)
    assert "request_create" in events
    assert "request_decide" in events
    print("  ok request_events_logged")
