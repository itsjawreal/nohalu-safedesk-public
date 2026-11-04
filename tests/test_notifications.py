"""Sprint 3.1/3.2/3.5 — kategori notifikasi, preferensi, read-all, dedupe.

Kontrak yang dijaga:
1. Setiap notifikasi punya `category` yang valid; default lama = `system`.
2. `GET /api/notifications?category=` memfilter tanpa mengubah badge global.
3. `POST /api/notifications/read-all` menandai semua (atau satu kategori).
4. Preferensi mematikan kategori non-system, tetapi `system` selalu lolos.
5. `notify_once` tidak pernah menggandakan notifikasi (idempotensi job).
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "notif_api.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402
import portal  # noqa: E402

client = TestClient(appmod.app)
from authutil import login_headers  # noqa: E402
H = login_headers(client, "u_budi")
H_RINA = login_headers(client, "u_rina")


def _clear(uid="u_budi"):
    appmod.CONN.execute("DELETE FROM notifications WHERE user_id = ?", (uid,))
    appmod.CONN.execute("DELETE FROM notification_prefs WHERE user_id = ?", (uid,))
    appmod.CONN.execute("DELETE FROM notification_log WHERE user_id = ?", (uid,))
    appmod.CONN.commit()


def _seed(uid="u_budi"):
    portal._notify(appmod.CONN, [uid], "approval satu", "/app/persetujuan",
                   category="approval")
    portal._notify(appmod.CONN, [uid], "slip satu", "/app/slip-gaji",
                   category="payslip")
    portal._notify(appmod.CONN, [uid], "sistem satu", None, category="system")
    appmod.CONN.commit()


def test_notification_has_category_and_filter():
    _clear()
    _seed()
    all_items = client.get("/api/notifications", headers=H).json()
    assert all_items["unread"] == 3
    assert {i["category"] for i in all_items["items"]} == {"approval", "payslip", "system"}
    only = client.get("/api/notifications?category=payslip", headers=H).json()
    assert [i["message"] for i in only["items"]] == ["slip satu"]
    # Badge global tidak ikut mengecil karena filter hanya untuk panel.
    assert only["unread"] == 3
    assert only["unread_by_category"]["payslip"] == 1
    assert only["has_more"] is False
    _clear()


def test_invalid_category_rejected():
    r = client.get("/api/notifications?category=ngawur", headers=H)
    assert r.status_code == 422


def test_read_all_and_read_all_per_category():
    _clear()
    _seed()
    r = client.post("/api/notifications/read-all", headers=H,
                    json={"category": "approval"})
    assert r.status_code == 200 and r.json()["updated"] == 1
    left = client.get("/api/notifications", headers=H).json()
    assert left["unread"] == 2
    assert left["unread_by_category"]["approval"] == 0

    r = client.post("/api/notifications/read-all", headers=H, json={})
    assert r.json()["updated"] == 2
    assert client.get("/api/notifications", headers=H).json()["unread"] == 0
    _clear()


def test_prefs_block_non_system_but_never_system():
    _clear()
    r = client.put("/api/notifications/prefs", headers=H,
                   json={"category": "payslip", "enabled": False})
    assert r.status_code == 200
    _seed()
    msgs = [i["message"] for i in client.get("/api/notifications", headers=H).json()["items"]]
    assert "slip satu" not in msgs
    assert "sistem satu" in msgs

    # `system` terkunci: tidak boleh dimatikan.
    r = client.put("/api/notifications/prefs", headers=H,
                   json={"category": "system", "enabled": False})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "category_locked"

    # Kategori tak dikenal ditolak.
    r = client.put("/api/notifications/prefs", headers=H,
                   json={"category": "promo", "enabled": False})
    assert r.status_code == 422
    _clear()


def test_notify_once_is_idempotent():
    _clear()
    key = "stale-approval:999:hr:2026-10-30"
    first = portal.notify_once(appmod.CONN, ["u_budi"], "ingat", "/app/persetujuan",
                               category="approval", dedupe_key=key)
    second = portal.notify_once(appmod.CONN, ["u_budi"], "ingat", "/app/persetujuan",
                                category="approval", dedupe_key=key)
    appmod.CONN.commit()
    assert (first, second) == (1, 0)
    rows = appmod.CONN.execute(
        "SELECT COUNT(*) AS n FROM notifications WHERE user_id='u_budi'").fetchone()
    assert rows["n"] == 1
    _clear()


def test_legacy_rows_without_category_are_system():
    """Baris lama (hasil migrasi ALTER TABLE) harus dianggap `system`."""
    _clear()
    appmod.CONN.execute(
        "INSERT INTO notifications (user_id, message, link, created_at) "
        "VALUES ('u_budi','warisan',NULL,'2026-01-01T00:00:00Z')")
    appmod.CONN.commit()
    items = client.get("/api/notifications", headers=H).json()["items"]
    assert items[0]["category"] == "system"
    _clear()


def test_prefs_boolean_coercion_from_strings():
    """UI bisa mengirim "false"/"0"; jangan dianggap truthy."""
    _clear()
    for raw in ("false", "0", 0, False):
        r = client.put("/api/notifications/prefs", headers=H,
                       json={"category": "approval", "enabled": raw})
        assert r.status_code == 200, r.text
        cat = next(c for c in r.json()["categories"] if c["category"] == "approval")
        assert cat["enabled"] is False, raw
    r = client.put("/api/notifications/prefs", headers=H,
                   json={"category": "approval", "enabled": "yes"})
    assert next(c for c in r.json()["categories"]
                if c["category"] == "approval")["enabled"] is True
    _clear()


def test_read_all_category_returns_real_global_unread():
    """Menandai satu kategori tidak boleh melaporkan unread global = 0."""
    _clear()
    _seed()
    r = client.post("/api/notifications/read-all", headers=H,
                    json={"category": "approval"})
    body = r.json()
    assert body["updated"] == 1
    assert body["unread"] == 2, body
    _clear()


def test_mark_read_returns_remaining_unread():
    _clear()
    _seed()
    nid = client.get("/api/notifications", headers=H).json()["items"][0]["id"]
    body = client.post(f"/api/notifications/{nid}/read", headers=H).json()
    assert body["ok"] is True and body["unread"] == 2, body
    _clear()
