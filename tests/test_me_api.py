"""Sprint 1 — kontrak HTTP endpoint personal (/api/me/*, /api/org, /api/team).

Menguji lewat FastAPI TestClient: status code, isolasi data antar user, dan
bentuk payload yang dipakai UI "Saya".
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

_TMP = tempfile.mkdtemp()
os.environ.setdefault("NOHALU_DB_PATH", os.path.join(_TMP, "me.db"))
os.environ["NOHALU_LLM_ENABLED"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402

client = TestClient(appmod.app)


def _h(uid):
    from authutil import login_headers
    return login_headers(client, uid)


def test_me_profile_returns_manager():
    r = client.get("/api/me/profile", headers=_h("u_budi"))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["id"] == "u_budi"
    assert d["department"] == "Creative & Production"
    assert d["manager"]["name"] == "Andi Pratama"
    print("  ok me_profile_returns_manager")


def test_me_leave_balance_endpoint():
    r = client.get("/api/me/leave-balance", headers=_h("u_budi"))
    assert r.status_code == 200, r.text
    d = r.json()
    for k in ("year", "entitlement", "used", "pending", "remaining", "requests"):
        assert k in d, (k, d)
    assert d["remaining"] == d["entitlement"] + d["carry_over"] - d["used"] - d["pending"]
    # tahun eksplisit
    r2 = client.get("/api/me/leave-balance?year=2027", headers=_h("u_budi"))
    assert r2.status_code == 200 and r2.json()["year"] == 2027
    print("  ok me_leave_balance_endpoint")


def test_me_payslips_isolated():
    """Isolasi diuji lewat ID & nominal, bukan lewat "user tanpa slip".

    Setelah seed dilengkapi, setiap karyawan punya slip; asersi lama
    (`dimas.count == 0`) justru akan lulus hanya karena datanya kosong dan
    tidak membuktikan apa pun tentang kebocoran.
    """
    budi = client.get("/api/me/payslips", headers=_h("u_budi")).json()
    dimas = client.get("/api/me/payslips", headers=_h("u_dimas")).json()
    assert budi["count"] >= 1 and dimas["count"] >= 1
    budi_ids = {p["id"] for p in budi["payslips"]}
    dimas_ids = {p["id"] for p in dimas["payslips"]}
    assert budi_ids.isdisjoint(dimas_ids), "slip antar user tidak boleh tercampur"
    assert {p["net"] for p in budi["payslips"]}.isdisjoint(
        {p["net"] for p in dimas["payslips"]})
    print("  ok me_payslips_isolated")


def test_me_attendance_and_clock():
    r = client.get("/api/me/attendance", headers=_h("u_budi"))
    assert r.status_code == 200, r.text
    d = r.json()
    assert "summary" in d and "work_start" in d
    r = client.post("/api/me/attendance/clock", headers=_h("u_budi"), json={"action": "in"})
    assert r.status_code in (200, 201), r.text
    body = r.json()
    assert body["clock_in"]
    r2 = client.post("/api/me/attendance/clock", headers=_h("u_budi"), json={"action": "in"})
    assert r2.status_code == 200 and r2.json()["idempotent"] is True
    # clock_out yang belum ada harus null, bukan string kosong
    assert r2.json()["clock_out"] != "", r2.json()
    fresh = client.post("/api/me/attendance/clock", headers=_h("u_nadia"),
                        json={"action": "in"})
    assert fresh.status_code in (200, 201), fresh.text
    assert fresh.json()["clock_out"] is None, fresh.json()
    r3 = client.post("/api/me/attendance/clock", headers=_h("u_budi"), json={"action": "bogus"})
    assert r3.status_code == 400 and r3.json()["detail"]["code"] == "invalid_action"
    print("  ok me_attendance_and_clock")


def test_org_chart_and_team_endpoints():
    r = client.get("/api/org/chart", headers=_h("u_budi"))
    assert r.status_code == 200, r.text
    d = r.json()
    assert [c["id"] for c in d["chain"]] == ["u_andi", "u_rina"]
    r = client.get("/api/team", headers=_h("u_andi"))
    assert r.status_code == 200
    assert {p["id"] for p in r.json()["reports"]} >= {"u_budi", "u_maya"}
    print("  ok org_chart_and_team_endpoints")


def test_publish_payslip_http_permissions():
    payload = {"user_id": "u_budi", "period": "2026-11", "gross": 7500000,
               "deductions": 850000}
    r = client.post("/api/payslips", headers=_h("u_budi"), json=payload)
    assert r.status_code == 403, r.text
    r = client.post("/api/payslips", headers=_h("u_sari"), json=payload)
    assert r.status_code == 201, r.text
    assert r.json()["net"] == 6650000
    print("  ok publish_payslip_http_permissions")


def test_me_endpoints_require_auth():
    for path in ("/api/me/profile", "/api/me/leave-balance", "/api/me/payslips",
                 "/api/me/attendance", "/api/org/chart", "/api/team"):
        assert client.get(path).status_code == 401, path
    print("  ok me_endpoints_require_auth")


def test_chat_personal_intent_over_http():
    r = client.post("/api/chat", headers=_h("u_budi"),
                    json={"question": "Berapa sisa cuti saya?"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["engine_used"] == "personal"
    assert "sisa" in d["answer"].lower()
    assert len(d["suggestions"]) >= 3
    print("  ok chat_personal_intent_over_http")


def test_leave_request_blocked_when_balance_insufficient():
    # sisa 12 hari; ajukan 3 hari berulang sampai melewati saldo
    ok = 0
    for i in range(5):
        day = 1 + i * 4
        r = client.post("/api/requests", headers=_h("u_nadia"), json={
            "type": "cuti",
            "start_date": f"2027-05-{day:02d}",
            "end_date": f"2027-05-{day + 2:02d}",
        })
        if r.status_code == 201:
            ok += 1
        else:
            assert r.status_code == 422, r.text
            assert r.json()["detail"]["code"] == "insufficient_leave_balance"
            break
    assert ok == 4, f"harus berhenti setelah 4x3 hari (12 hari), dapat {ok}"
    print("  ok leave_request_blocked_when_balance_insufficient")
