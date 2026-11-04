"""P0/P1 — penjagaan jalur tulis admin (audit 2026-11-01).

Empat celah nyata yang ditemukan saat audit ulang plan, semuanya di jalur
yang bisa dipanggil langsung (bukan hanya lewat UI):

  D1 `PUT /api/users/{id}/permissions` menulis kolom tanpa validasi/audit/
     proteksi admin terakhir → satu request bisa mengunci konsol Kelola.
  D2 karyawan yang dinonaktifkan tetap bisa login dan memakai token lamanya.
  D3 workspace karangan (`ws_palsu`) diterima → user kehilangan akses dokumen.
  D4 penerbitan slip gaji (data finansial pribadi) tidak masuk audit.

Tes ini mengunci perilaku SETELAH perbaikan; masing-masing gagal pada kode
sebelumnya.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "admin_guards.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402
from authutil import login_headers  # noqa: E402

client = TestClient(appmod.app)
H_ADMIN = login_headers(client, "u_rina")


def _admin_id():
    users = client.get("/api/users?limit=200", headers=H_ADMIN).json()["users"]
    return [u["id"] for u in users if u["role"] == "admin"]


def _audit_types(limit=200):
    body = client.get("/api/audit?limit=%d" % limit, headers=H_ADMIN).json()
    return [row.get("event_type") for row in body["audit"]]


# ---------------------------------------------------------------------------
# D1 — PUT /permissions harus memakai aturan yang sama dengan PATCH
# ---------------------------------------------------------------------------
def test_put_permissions_rejects_unknown_role():
    r = client.put("/api/users/u_maya/permissions",
                   json={"role": "superadmin"}, headers=H_ADMIN)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["code"] == "invalid_role"
    # pembanding: jalur PATCH memang sudah menolak
    assert client.patch("/api/users/u_maya", json={"role": "superadmin"},
                        headers=H_ADMIN).status_code == 422
    # role lama tidak berubah
    maya = next(u for u in client.get("/api/users?q=maya", headers=H_ADMIN).json()["users"]
                if u["id"] == "u_maya")
    assert maya["role"] != "superadmin"


def test_put_permissions_requires_a_field():
    """Payload kosong dulu menjawab 200 "permissions_updated" tanpa menulis apa pun."""
    r = client.put("/api/users/u_maya/permissions", json={}, headers=H_ADMIN)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "no_fields"


def test_put_permissions_cannot_demote_last_admin():
    """Celah paling berbahaya: seluruh konsol Kelola/Audit/Cadangan terkunci.

    Tanpa perbaikan, request ini 200 dan tidak ada admin tersisa — tidak ada
    jalan pulih lewat UI karena semua endpoint admin ikut 403.
    """
    admins = _admin_id()
    assert admins == ["u_rina"], "asumsi seed: satu admin"
    r = client.put("/api/users/u_rina/permissions",
                   json={"role": "employee"}, headers=H_ADMIN)
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["code"] == "last_admin"
    assert _admin_id() == ["u_rina"], "admin terakhir hilang"
    assert client.get("/api/users", headers=H_ADMIN).status_code == 200


def test_put_permissions_is_audited():
    before = _audit_types().count("user_permissions")
    r = client.put("/api/users/u_fajar/permissions",
                   json={"allowed_workspaces": ["ws_onboarding"]}, headers=H_ADMIN)
    assert r.status_code == 200, r.text
    assert _audit_types().count("user_permissions") == before + 1


# ---------------------------------------------------------------------------
# D2 — akun nonaktif tidak boleh punya akses
# ---------------------------------------------------------------------------
def test_deactivated_user_cannot_login_or_reuse_token():
    created = client.post("/api/users", json={
        "id": "u_guard_off", "name": "Guard Off",
        "allowed_workspaces": ["ws_onboarding"]}, headers=H_ADMIN)
    assert created.status_code == 200, created.text
    h_old = login_headers(client, "u_guard_off")

    r = client.delete("/api/users/u_guard_off", headers=H_ADMIN)
    assert r.status_code == 200, r.text
    assert r.json()["sessions_revoked"] >= 1, "sesi tidak dicabut saat nonaktif"

    assert client.get("/api/me", headers=h_old).status_code == 401
    assert client.post("/api/chat", json={"question": "halo"},
                       headers=h_old).status_code == 401
    # login ulang juga ditolak
    assert client.post("/api/auth/login",
                       json={"user_id": "u_guard_off"}).status_code == 401


def test_deactivation_revokes_every_session():
    client.post("/api/users", json={
        "id": "u_guard_two", "name": "Guard Two",
        "allowed_workspaces": ["ws_onboarding"]}, headers=H_ADMIN)
    a = login_headers(client, "u_guard_two")
    b = login_headers(client, "u_guard_two")
    assert client.get("/api/me", headers=a).status_code == 200
    assert client.get("/api/me", headers=b).status_code == 200
    client.delete("/api/users/u_guard_two", headers=H_ADMIN)
    assert client.get("/api/me", headers=a).status_code == 401
    assert client.get("/api/me", headers=b).status_code == 401


def test_reactivated_user_can_login_again():
    client.post("/api/users", json={
        "id": "u_guard_back", "name": "Guard Back",
        "allowed_workspaces": ["ws_onboarding"]}, headers=H_ADMIN)
    client.delete("/api/users/u_guard_back", headers=H_ADMIN)
    assert client.post("/api/auth/login",
                       json={"user_id": "u_guard_back"}).status_code == 401
    r = client.patch("/api/users/u_guard_back",
                     json={"employment_status": "active"}, headers=H_ADMIN)
    assert r.status_code == 200, r.text
    assert client.post("/api/auth/login",
                       json={"user_id": "u_guard_back"}).status_code == 200


# ---------------------------------------------------------------------------
# D3 — workspace harus benar-benar ada
# ---------------------------------------------------------------------------
def test_unknown_workspace_is_rejected_on_patch():
    r = client.patch("/api/users/u_yusuf",
                     json={"allowed_workspaces": ["ws_palsu"]}, headers=H_ADMIN)
    assert r.status_code == 422, r.text
    assert r.json()["detail"]["code"] == "unknown_workspace"
    yusuf = next(u for u in client.get("/api/users?q=yusuf", headers=H_ADMIN).json()["users"]
                 if u["id"] == "u_yusuf")
    assert "ws_palsu" not in yusuf["allowed_workspaces"]


def test_unknown_workspace_is_rejected_on_put():
    r = client.put("/api/users/u_yusuf/permissions",
                   json={"allowed_workspaces": ["ws_onboarding", "ws_palsu"]},
                   headers=H_ADMIN)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "unknown_workspace"


def test_known_workspaces_still_accepted():
    r = client.put("/api/users/u_intan/permissions",
                   json={"allowed_workspaces": ["ws_onboarding", "ws_project_alpha"]},
                   headers=H_ADMIN)
    assert r.status_code == 200, r.text
    intan = next(u for u in client.get("/api/users?q=intan", headers=H_ADMIN).json()["users"]
                 if u["id"] == "u_intan")
    assert set(intan["allowed_workspaces"]) == {"ws_onboarding", "ws_project_alpha"}


# ---------------------------------------------------------------------------
# D4 — penerbitan slip gaji harus masuk audit
# ---------------------------------------------------------------------------
PAY_USER = "u_guard_pay"


def _ensure_pay_user():
    """Karyawan khusus tes ini.

    Modul tes berbagi satu instance `db` (path dibaca saat import pertama),
    jadi menulis notifikasi/slip untuk user seed akan mengotori tes lain yang
    menghitung badge global. User khusus + pembersihan di akhir menjaga suite
    tetap independen.
    """
    if not any(u["id"] == PAY_USER
               for u in client.get("/api/users?limit=200", headers=H_ADMIN).json()["users"]):
        client.post("/api/users", json={
            "id": PAY_USER, "name": "Guard Pay",
            "allowed_workspaces": ["ws_onboarding"]}, headers=H_ADMIN)
    return PAY_USER


def _cleanup_pay(period):
    appmod.CONN.execute("DELETE FROM payslips WHERE user_id=? AND period=?",
                        (PAY_USER, period))
    appmod.CONN.execute("DELETE FROM notifications WHERE user_id=?", (PAY_USER,))
    appmod.CONN.commit()


def test_payslip_publish_is_audited():
    _ensure_pay_user()
    before = _audit_types().count("payslip_publish")
    try:
        r = client.post("/api/payslips", json={
            "user_id": PAY_USER, "period": "2026-12",
            "gross": 9_000_000, "deductions": 1_000_000}, headers=H_ADMIN)
        assert r.status_code in (200, 201), r.text
        assert _audit_types().count("payslip_publish") == before + 1
    finally:
        _cleanup_pay("2026-12")


def test_payslip_publish_rejected_is_not_audited():
    """Percobaan yang ditolak tidak boleh tercatat sebagai penerbitan."""
    before = _audit_types().count("payslip_publish")
    h_emp = login_headers(client, "u_budi")
    r = client.post("/api/payslips", json={
        "user_id": "u_budi", "period": "2026-12",
        "gross": 1, "deductions": 0}, headers=h_emp)
    assert r.status_code == 403
    assert _audit_types().count("payslip_publish") == before
