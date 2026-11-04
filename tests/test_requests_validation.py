"""Sprint 0 — validasi input pengajuan (tanggal, nominal, catatan, idempotency).

Run: python tests/test_requests_validation.py   (atau pytest)
Offline: LLM mati, DB in-memory.
"""
import datetime
import os
import sys
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_LLM_ENABLED"] = "0"

import agent_tools
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


# ---------------------------------------------------------------------------
# Nominal
# ---------------------------------------------------------------------------
def test_parse_amount_strict_accepts_common_formats():
    assert portal.parse_amount_strict("750000") == (750000, None)
    assert portal.parse_amount_strict("750.000") == (750000, None)
    assert portal.parse_amount_strict("Rp 2.000.000") == (2000000, None)
    assert portal.parse_amount_strict("IDR 1,500,000") == (1500000, None)
    assert portal.parse_amount_strict(1250000) == (1250000, None)
    print("  ok parse_amount_strict_accepts_common_formats")


def test_parse_amount_strict_rejects_garbage():
    for bad in ("-500000", "abc", "", "0", "-1", "1.5", "1000.50", None, True):
        val, err = portal.parse_amount_strict(bad)
        assert val is None and err, f"{bad!r} harus ditolak"
    print("  ok parse_amount_strict_rejects_garbage")


def test_reimbursement_rejects_invalid_amount():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    for bad in ("-500000", "abc", "0", ""):
        res, code = portal.create_request(conn, budi, {"type": "reimbursement", "amount": bad})
        assert code == 422, f"{bad!r} seharusnya 422, dapat {code}"
        assert res["error"] in ("amount_invalid", "amount_required")
    print("  ok reimbursement_rejects_invalid_amount")


def test_reimbursement_rejects_over_limit():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    over = portal.MAX_REIMBURSEMENT + 1
    res, code = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": str(over)})
    assert code == 422 and res["error"] == "amount_too_large"
    print("  ok reimbursement_rejects_over_limit")


def test_amount_value_stored_as_integer():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    data, code = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "Rp 1.250.000"})
    assert code == 201
    row = conn.execute("SELECT amount, amount_value FROM requests WHERE id=?",
                       (data["id"],)).fetchone()
    assert row["amount_value"] == 1250000
    assert isinstance(row["amount_value"], int)
    print("  ok amount_value_stored_as_integer")


# ---------------------------------------------------------------------------
# Tanggal cuti
# ---------------------------------------------------------------------------
def test_cuti_rejects_end_before_start():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    res, code = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": "2026-12-31", "end_date": "2026-12-01"})
    assert code == 422 and res["error"] == "end_before_start"
    print("  ok cuti_rejects_end_before_start")


def test_cuti_rejects_non_iso_dates():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    res, code = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": "abc", "end_date": "xyz"})
    assert code == 422 and res["error"] == "invalid_start_date"
    res, code = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": "2026-12-01", "end_date": "2026-02-30"})
    assert code == 422 and res["error"] == "invalid_end_date"
    print("  ok cuti_rejects_non_iso_dates")


def test_cuti_rejects_past_start():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    res, code = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": "2020-01-01", "end_date": "2020-01-02"})
    assert code == 422 and res["error"] == "start_in_past"
    print("  ok cuti_rejects_past_start")


def test_cuti_rejects_too_many_days():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    res, code = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": "2027-01-01", "end_date": "2028-02-05"})
    assert code == 422 and res["error"] == "leave_too_long"
    print("  ok cuti_rejects_too_many_days")


def test_cuti_rejects_overlap():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    first, code = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": "2027-03-10", "end_date": "2027-03-12"})
    assert code == 201
    res, code = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": "2027-03-11", "end_date": "2027-03-13"})
    assert code == 409 and res["error"] == "leave_overlap"
    assert res["fields"]["conflict_id"] == first["id"]
    # non-overlap tetap boleh
    res, code = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": "2027-03-13", "end_date": "2027-03-14"})
    assert code == 201, res
    print("  ok cuti_rejects_overlap")


def test_cuti_overlap_ignores_cancelled():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    first, _ = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": "2027-04-10", "end_date": "2027-04-12"})
    portal.cancel_request(conn, budi, first["id"])
    res, code = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": "2027-04-10", "end_date": "2027-04-12"})
    assert code == 201, res
    print("  ok cuti_overlap_ignores_cancelled")


# ---------------------------------------------------------------------------
# Batas panjang & tipe
# ---------------------------------------------------------------------------
def test_note_and_title_limits():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    res, code = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "10000", "note": "x" * (portal.MAX_NOTE_LEN + 1)})
    assert code == 422 and res["error"] == "note_too_long"
    res, code = portal.create_request(conn, budi, {
        "type": "reimbursement", "amount": "10000",
        "title": "y" * (portal.MAX_TITLE_LEN + 1)})
    assert code == 422 and res["error"] == "title_too_long"
    print("  ok note_and_title_limits")


def test_invalid_type():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    res, code = portal.create_request(conn, budi, {"type": "pinjaman"})
    assert code == 400 and res["error"] == "invalid_type"
    print("  ok invalid_type")


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
def test_client_ref_idempotent():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    payload = {"type": "reimbursement", "amount": "100000",
               "client_ref": "abcd1234-ref"}
    first, c1 = portal.create_request(conn, budi, payload)
    second, c2 = portal.create_request(conn, budi, payload)
    assert c1 == 201 and c2 == 200, (c1, c2)
    assert first["id"] == second["id"]
    assert second.get("idempotent") is True
    n = conn.execute("SELECT COUNT(*) AS n FROM requests").fetchone()["n"]
    assert n == 1
    print("  ok client_ref_idempotent")


def test_client_ref_invalid_format():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    for bad in ("short", "has space 1234", "x" * 65):
        res, code = portal.create_request(conn, budi, {
            "type": "reimbursement", "amount": "10000", "client_ref": bad})
        assert code == 422 and res["error"] == "invalid_client_ref", bad
    print("  ok client_ref_invalid_format")


def test_client_ref_scoped_per_user():
    conn = make_conn()
    budi, dimas = _u(conn, "u_budi"), _u(conn, "u_dimas")
    payload = {"type": "reimbursement", "amount": "10000", "client_ref": "same-ref-1"}
    a, _ = portal.create_request(conn, budi, payload)
    b, code = portal.create_request(conn, dimas, payload)
    assert code == 201 and a["id"] != b["id"]
    print("  ok client_ref_scoped_per_user")


# ---------------------------------------------------------------------------
# Tanggal fleksibel (B): user menulis "03/11/2026", "3 November 2026", "besok"
# ---------------------------------------------------------------------------
def test_parse_flexible_date_formats():
    import datetime as _dt
    d = _dt.date
    assert portal.parse_flexible_date("2026-11-03")[0] == d(2026, 11, 3)
    assert portal.parse_flexible_date("03/11/2026")[0] == d(2026, 11, 3)
    assert portal.parse_flexible_date("3-11-2026")[0] == d(2026, 11, 3)
    assert portal.parse_flexible_date("3.11.2026")[0] == d(2026, 11, 3)
    assert portal.parse_flexible_date("2026/11/03")[0] == d(2026, 11, 3)
    assert portal.parse_flexible_date("3 November 2026")[0] == d(2026, 11, 3)
    assert portal.parse_flexible_date("3 nov 2026")[0] == d(2026, 11, 3)
    # dd/mm dan mm/dd sama-sama valid -> dipakai dd/mm, ditandai ambigu
    got, code, assumed = portal.parse_flexible_date("10/03/2026")
    assert got == d(2026, 3, 10) and code == "ambiguous" and assumed == "dd/mm"
    # angka pertama > 12 -> jelas dd/mm
    got, code, _ = portal.parse_flexible_date("25/03/2026")
    assert got == d(2026, 3, 25) and code is None
    # angka kedua > 12 -> jelas mm/dd
    got, code, _ = portal.parse_flexible_date("03/25/2026")
    assert got == d(2026, 3, 25) and code is None
    print("  ok parse_flexible_date_formats")


def test_parse_flexible_date_relative_and_invalid():
    import datetime as _dt
    base = _dt.date(2026, 10, 31)
    assert portal.parse_flexible_date("besok", today=base)[0] == _dt.date(2026, 11, 1)
    assert portal.parse_flexible_date("lusa", today=base)[0] == _dt.date(2026, 11, 2)
    assert portal.parse_flexible_date("hari ini", today=base)[0] == base
    for bad in ("abc", "", None, "2026-02-30", "31/02/2026", "13/13/2026"):
        got, code, _ = portal.parse_flexible_date(bad)
        assert got is None and code == "invalid", bad
    print("  ok parse_flexible_date_relative_and_invalid")


def test_create_request_accepts_flexible_dates_and_stores_iso():
    """Tanggal bebas → ISO, diuji relatif terhadap hari ini.

    Tanggal mati ("3 November 2026") membuat tes ini gagal sendiri begitu jam
    sistem melewatinya: produk menolak cuti di masa lalu — perilaku yang benar.
    """
    conn = make_conn()
    budi = _u(conn, "u_budi")
    start = portal._today() + datetime.timedelta(days=30)
    end = start + datetime.timedelta(days=2)
    start_txt = f"{start.day} {agent_tools._MONTH_ID[start.month - 1]} {start.year}"
    end_txt = f"{end.day:02d}/{end.month:02d}/{end.year}"
    res, code = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": start_txt, "end_date": end_txt})
    assert code == 201, res
    row = conn.execute("SELECT start_date,end_date FROM requests WHERE id=?",
                       (res["id"],)).fetchone()
    # Disimpan SELALU ISO, walau input bebas.
    assert row["start_date"] == start.isoformat()
    assert row["end_date"] == end.isoformat()
    print("  ok create_request_accepts_flexible_dates_and_stores_iso")


def test_create_request_rejects_unreadable_date_with_hint():
    conn = make_conn()
    budi = _u(conn, "u_budi")
    res, code = portal.create_request(conn, budi, {
        "type": "cuti", "start_date": "kemarin lusa", "end_date": "2026-11-05"})
    assert code == 422 and res["error"] == "invalid_start_date"
    assert "tidak bisa dibaca" in res["message"]
    print("  ok create_request_rejects_unreadable_date_with_hint")
