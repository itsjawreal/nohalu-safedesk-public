"""Sprint 4 — CRUD workspace (sebelumnya hanya GET).

Workspace adalah unit izin: dokumen punya `workspace_ids`, karyawan punya
`allowed_workspaces`. Tanpa endpoint tulis, membuat unit organisasi baru
berarti SQL langsung ke DB produksi — padahal `users_admin` sudah menolak
workspace yang tidak dikenal.

Kontrak yang dijaga:
* hanya admin (bukan HR) yang boleh menulis;
* id dibuat dari nama bila kosong, dan tidak bisa dipakai dua kali;
* nama unik tanpa membedakan huruf besar/kecil;
* workspace yang masih dipakai tidak bisa dihapus (409) — kalau tidak,
  dokumen di dalamnya jadi tak terjangkau siapa pun;
* setiap perubahan meninggalkan jejak audit.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "ws_admin.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402
from authutil import login_headers  # noqa: E402

client = TestClient(appmod.app)
H_ADMIN = login_headers(client, "u_rina")
H_HR = login_headers(client, "u_sari")


def _cleanup(ws_id):
    appmod.CONN.execute("DELETE FROM workspaces WHERE id = ?", (ws_id,))
    appmod.CONN.execute(
        "DELETE FROM audit_log WHERE user_id = 'u_rina' AND question LIKE ?",
        (f"%workspace {ws_id}%",))
    appmod.CONN.commit()


def test_admin_creates_workspace_with_generated_slug():
    r = client.post("/api/workspaces", json={"name": "Tim Data & Analytics"},
                    headers=H_ADMIN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == "ws_tim_data_analytics", body
    assert body["name"] == "Tim Data & Analytics"
    try:
        listed = client.get("/api/workspaces", headers=H_ADMIN).json()
        ids = {w["id"] for w in listed["workspaces"]}
        assert body["id"] in ids
        row = next(w for w in listed["workspaces"] if w["id"] == body["id"])
        assert row["documents"] == 0 and row["users"] == 0
    finally:
        _cleanup(body["id"])


def test_explicit_id_is_accepted_and_validated():
    r = client.post("/api/workspaces", json={"id": "ws_data_lab", "name": "Data Lab"},
                    headers=H_ADMIN)
    assert r.status_code == 200, r.text
    try:
        bad = client.post("/api/workspaces",
                          json={"id": "WS Salah!", "name": "X"}, headers=H_ADMIN)
        assert bad.status_code == 422
        assert bad.json()["detail"]["code"] == "invalid_id"
    finally:
        _cleanup("ws_data_lab")


def test_duplicate_name_and_id_are_rejected():
    first = client.post("/api/workspaces", json={"name": "Tim Kreatif"},
                        headers=H_ADMIN)
    assert first.status_code == 200, first.text
    ws_id = first.json()["id"]
    try:
        dup_name = client.post("/api/workspaces", json={"name": "tim kreatif"},
                               headers=H_ADMIN)
        assert dup_name.status_code == 409
        assert dup_name.json()["detail"]["code"] == "name_taken"
        dup_id = client.post("/api/workspaces",
                             json={"id": ws_id, "name": "Nama Lain"}, headers=H_ADMIN)
        assert dup_id.status_code == 409
        assert dup_id.json()["detail"]["code"] == "id_taken"
    finally:
        _cleanup(ws_id)


def test_rename_changes_name_only():
    created = client.post("/api/workspaces", json={"name": "Tim Sementara"},
                          headers=H_ADMIN).json()
    try:
        r = client.patch(f"/api/workspaces/{created['id']}",
                         json={"name": "Tim Tetap"}, headers=H_ADMIN)
        assert r.status_code == 200, r.text
        assert r.json() == {"id": created["id"], "name": "Tim Tetap",
                            "status": "renamed"}
        listed = client.get("/api/workspaces", headers=H_ADMIN).json()["workspaces"]
        assert next(w for w in listed if w["id"] == created["id"])["name"] == "Tim Tetap"
    finally:
        _cleanup(created["id"])


def test_delete_refuses_workspace_still_in_use():
    """`ws_onboarding` dipakai user + dokumen seed → tidak boleh terhapus."""
    r = client.delete("/api/workspaces/ws_onboarding", headers=H_ADMIN)
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "workspace_in_use"
    assert detail["fields"]["users"] >= 1
    assert detail["fields"]["documents"] >= 1


def test_delete_empty_workspace():
    created = client.post("/api/workspaces", json={"name": "Tim Sekali Pakai"},
                          headers=H_ADMIN).json()
    r = client.delete(f"/api/workspaces/{created['id']}", headers=H_ADMIN)
    assert r.status_code == 200, r.text
    ids = {w["id"] for w in client.get("/api/workspaces", headers=H_ADMIN).json()["workspaces"]}
    assert created["id"] not in ids
    # Hapus dua kali → 404, bukan 500.
    assert client.delete(f"/api/workspaces/{created['id']}",
                         headers=H_ADMIN).status_code == 404


def test_hr_cannot_write_workspaces():
    assert client.post("/api/workspaces", json={"name": "X"},
                       headers=H_HR).status_code == 403
    assert client.patch("/api/workspaces/ws_onboarding", json={"name": "X"},
                        headers=H_HR).status_code == 403
    assert client.delete("/api/workspaces/ws_onboarding",
                         headers=H_HR).status_code == 403


def test_changes_are_audited():
    created = client.post("/api/workspaces", json={"name": "Tim Audit"},
                          headers=H_ADMIN).json()
    try:
        rows = appmod.CONN.execute(
            "SELECT event_type FROM audit_log WHERE question LIKE ?",
            (f"%workspace {created['id']}%",)).fetchall()
        kinds = [r["event_type"] for r in rows]
        assert "workspace_create" in kinds, kinds
    finally:
        _cleanup(created["id"])
