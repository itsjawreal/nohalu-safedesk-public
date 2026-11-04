"""Sprint 4.1/4.2 — CRUD karyawan + impor CSV (admin).

Sebelum ini endpoint hanya `PUT /api/users/{id}/permissions`; menambah
karyawan atau memperbaiki atasan harus lewat SQL langsung. Tes ini mengunci
aturan yang mudah hilang saat refactor: validasi peran, siklus atasan, admin
terakhir, dan dry-run impor yang tidak boleh menulis apa pun.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "users_api.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402
from authutil import login_headers  # noqa: E402

client = TestClient(appmod.app)
H_ADMIN = login_headers(client, "u_rina")
H_HR = login_headers(client, "u_sari")
H_EMP = login_headers(client, "u_budi")


def _list(limit=200):
    r = client.get("/api/users?limit=%d" % limit, headers=H_ADMIN)
    assert r.status_code == 200, r.text
    return r.json()


def _find(uid):
    return next((u for u in _list()["users"] if u["id"] == uid), None)


def test_list_users_has_profile_and_paging():
    body = _list()
    assert body["total"] >= 12
    assert "limit" in body and "offset" in body
    budi = next(u for u in body["users"] if u["id"] == "u_budi")
    assert budi["department"], "profil departemen harus ikut terkirim"
    assert budi["manager_id"] == "u_andi"
    page = client.get("/api/users?limit=3&offset=0", headers=H_ADMIN).json()
    assert len(page["users"]) == 3
    assert page["has_more"] is True


def test_only_admin_can_create_user():
    r = client.post("/api/users", json={"name": "X"}, headers=H_HR)
    assert r.status_code == 403, r.text


def test_create_user_and_read_back():
    r = client.post("/api/users", json={
        "name": "Raka Test", "role": "employee",
        "department": "QA", "job_title": "QA Engineer",
        "manager_id": "u_andi", "email": "raka.test@nusantarakreatif.id",
        "allowed_workspaces": ["ws_onboarding"],
    }, headers=H_ADMIN)
    assert r.status_code == 200, r.text
    uid = r.json()["id"]
    assert uid.startswith("u_")
    row = _find(uid)
    assert row["role"] == "employee"
    assert row["job_title"] == "QA Engineer"
    assert row["allowed_workspaces"] == ["ws_onboarding"]


def test_create_rejects_unknown_role_and_bad_email():
    r = client.post("/api/users", json={"name": "A", "role": "superuser"}, headers=H_ADMIN)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "invalid_role"

    r = client.post("/api/users", json={"name": "B", "email": "bukan-email"}, headers=H_ADMIN)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "invalid_email"


def test_create_rejects_duplicate_id_and_email():
    payload = {"id": "u_dup_test", "name": "Dup Test", "email": "dup@example.com"}
    assert client.post("/api/users", json=payload, headers=H_ADMIN).status_code == 200
    again = client.post("/api/users", json=payload, headers=H_ADMIN)
    assert again.status_code == 409
    assert again.json()["detail"]["code"] == "user_exists"
    other = client.post("/api/users", json={
        "name": "Dup Email", "email": "dup@example.com"}, headers=H_ADMIN)
    assert other.status_code == 409
    assert other.json()["detail"]["code"] == "email_taken"


def test_patch_updates_profile_and_reports_changed_fields():
    r = client.patch("/api/users/u_budi", json={
        "job_title": "Senior Content Producer", "location": "Bandung"},
        headers=H_ADMIN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body["changed_fields"]) == {"job_title", "location"}
    assert _find("u_budi")["job_title"] == "Senior Content Producer"

    same = client.patch("/api/users/u_budi", json={
        "job_title": "Senior Content Producer"}, headers=H_ADMIN)
    assert same.status_code == 200
    assert same.json()["changed_fields"] == []


def test_manager_cycle_is_rejected():
    # u_andi atasan u_budi. Menjadikan u_budi atasan u_andi = lingkaran.
    r = client.patch("/api/users/u_andi", json={"manager_id": "u_budi"},
                     headers=H_ADMIN)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "manager_cycle"

    r = client.patch("/api/users/u_andi", json={"manager_id": "u_andi"},
                     headers=H_ADMIN)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "manager_cycle"


def test_unknown_manager_is_rejected():
    r = client.patch("/api/users/u_budi", json={"manager_id": "u_hantu"},
                     headers=H_ADMIN)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "manager_not_found"


def test_last_admin_cannot_be_demoted_or_deactivated():
    # Hanya u_rina yang admin di seed.
    r = client.patch("/api/users/u_rina", json={"role": "employee"}, headers=H_ADMIN)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "last_admin"

    r = client.delete("/api/users/u_rina", headers=H_ADMIN)
    assert r.status_code == 409


def test_second_admin_allows_demoting_the_first():
    """Admin terakhir dilindungi, tetapi begitu ada admin kedua penurunan sah.

    Dilakukan oleh admin KEDUA, bukan oleh u_rina sendiri: menurunkan diri
    sendiri membuat token yang sedang dipakai langsung kehilangan izin, dan
    tes berikutnya akan gagal karena 403 — bukan karena aturannya salah.
    """
    created = client.post("/api/users", json={
        "id": "u_admin2", "name": "Admin Dua", "role": "admin",
        "allowed_workspaces": ["ws_onboarding"]}, headers=H_ADMIN)
    assert created.status_code == 200, created.text
    h2 = login_headers(client, "u_admin2")
    try:
        r = client.patch("/api/users/u_rina", json={"role": "employee"}, headers=h2)
        assert r.status_code == 200, r.text
        # Dibaca dengan token admin yang masih aktif (u_rina sudah bukan admin).
        listed = client.get("/api/users?limit=200", headers=h2).json()["users"]
        assert next(u for u in listed if u["id"] == "u_rina")["role"] == "employee"
    finally:
        # Kembalikan lewat admin kedua (u_rina masih employee saat ini).
        client.patch("/api/users/u_rina", json={"role": "admin"}, headers=h2)
        client.patch("/api/users/u_admin2", json={"role": "employee"}, headers=H_ADMIN)


def test_delete_is_soft_and_cannot_target_self():
    r = client.delete("/api/users/u_rina", headers=H_ADMIN)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] in ("cannot_deactivate_self", "last_admin")

    assert client.post("/api/users", json={
        "id": "u_temp_del", "name": "Temp Del"}, headers=H_ADMIN).status_code == 200
    r = client.delete("/api/users/u_temp_del", headers=H_ADMIN)
    assert r.status_code == 200, r.text
    assert _find("u_temp_del")["employment_status"] == "inactive"


CSV_OK = """nama,email,role,departemen,jabatan,atasan,workspace
Indra CSV,indra.csv@nusantarakreatif.id,employee,QA,QA Engineer,u_andi,ws_onboarding
Lestari CSV,lestari.csv@nusantarakreatif.id,team_lead,Design,Design Lead,u_rina,ws_onboarding;ws_project_alpha
"""

CSV_BAD = """nama,email,role
Baris Salah,bukan-email,employee
Baris Role,ok@example.com,superuser
,tanpa-nama@example.com,employee
"""


def test_import_dry_run_writes_nothing():
    before = _list()["total"]
    r = client.post("/api/users/import", json={"csv": CSV_OK, "dry_run": True},
                    headers=H_ADMIN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dry_run"] is True
    assert body["created"] == 2 and body["failed"] == 0
    assert _list()["total"] == before, "dry-run tidak boleh menulis"


def test_import_apply_creates_rows_and_parses_semicolon_workspaces():
    r = client.post("/api/users/import", json={"csv": CSV_OK, "dry_run": False},
                    headers=H_ADMIN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created"] == 2
    indra = next(u for u in _list()["users"] if u["name"] == "Indra CSV")
    assert indra["department"] == "QA"
    lestari = next(u for u in _list()["users"] if u["name"] == "Lestari CSV")
    assert set(lestari["allowed_workspaces"]) == {"ws_onboarding", "ws_project_alpha"}


def test_import_reports_per_row_errors_and_continues():
    r = client.post("/api/users/import", json={"csv": CSV_BAD, "dry_run": True},
                    headers=H_ADMIN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["failed"] == 3
    codes = [row["errors"][0]["code"] for row in body["rows"]]
    assert "invalid_email" in codes
    assert "invalid_role" in codes
    assert "name_required" in codes
    # laporan per baris menunjuk nomor baris file (header = baris 1)
    assert body["rows"][0]["line"] == 2


def test_import_by_structured_rows():
    r = client.post("/api/users/import", json={
        "rows": [{"nama": "Struk Baris", "role": "employee",
                  "workspace": ["ws_onboarding"]}],
        "dry_run": True}, headers=H_ADMIN)
    assert r.status_code == 200, r.text
    assert r.json()["created"] == 1


def test_import_empty_csv_is_rejected():
    r = client.post("/api/users/import", json={"csv": "", "dry_run": True},
                    headers=H_ADMIN)
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "csv_empty"


def test_employee_cannot_import():
    r = client.post("/api/users/import", json={"csv": CSV_OK}, headers=H_EMP)
    assert r.status_code == 403


# ---- Ekspor CSV (round-trip dengan impor) ----

def _csv_rows(text):
    """Parse CSV hasil ekspor (pemisah `;`, ada BOM) jadi list dict."""
    import csv
    import io

    text = text.lstrip("\ufeff")
    return list(csv.DictReader(io.StringIO(text), delimiter=";"))


def test_export_csv_has_import_compatible_header():
    r = client.get("/api/users/export.csv", headers=H_ADMIN)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    rows = _csv_rows(r.text)
    assert rows, "ekspor tidak boleh kosong"
    # Kolom yang diekspor harus bisa dibaca balik oleh impor: kalau header
    # bergeser, berkas hasil ekspor jadi tidak berguna untuk HR.
    expected = {"id", "nama", "email", "role", "departemen", "jabatan",
                "atasan", "tanggal_masuk", "lokasi", "nomor_karyawan",
                "status", "workspace"}
    assert expected <= set(rows[0].keys())


def test_export_csv_respects_filters():
    all_rows = _csv_rows(client.get("/api/users/export.csv", headers=H_ADMIN).text)
    hr_rows = _csv_rows(client.get("/api/users/export.csv?role=hr",
                                   headers=H_ADMIN).text)
    assert hr_rows, "harus ada minimal satu HR"
    assert all(r["role"] == "hr" for r in hr_rows)
    assert len(hr_rows) < len(all_rows)


def test_export_csv_round_trips_through_import_dry_run():
    """Berkas ekspor harus lolos pratinjau impor tanpa error baris."""
    text = client.get("/api/users/export.csv", headers=H_ADMIN).text
    r = client.post("/api/users/import",
                    json={"csv": text.lstrip("\ufeff"), "dry_run": True},
                    headers=H_ADMIN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["failed"] == 0, body["rows"]
    assert body["updated"] == body["total"], "semua baris harus terbaca sebagai update"


def test_export_csv_is_admin_only():
    for h in (H_HR, H_EMP):
        r = client.get("/api/users/export.csv", headers=h)
        assert r.status_code == 403, r.text
