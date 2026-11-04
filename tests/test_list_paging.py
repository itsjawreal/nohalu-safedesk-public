"""P0 — paging seragam untuk daftar yang masih dipotong diam-diam.

Tiga endpoint sebelumnya menerima `limit` tapi tidak punya `offset`, atau
memakai LIMIT mati:

* `GET /api/chat/sessions` — `LIMIT 100` hardcode, tanpa rate-limit. User
  dengan >100 percakapan kehilangan sisanya dan tidak ada cara mengambilnya.
* `GET /api/announcements` — `limit` saja; `unread` dihitung dari halaman,
  jadi badge lonceng salah begitu pengumuman > limit.
* `GET /api/notifications` — `limit` saja; notifikasi ke-51 dst. tidak pernah
  bisa dibaca dari UI.

Kontrak yang dijaga di sini: `limit`/`offset` benar-benar memotong, `total`
adalah jumlah penuh (bukan panjang halaman), `has_more` konsisten, dan dua
halaman berurutan tidak tumpang tindih.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "list_paging.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402
import portal  # noqa: E402
from authutil import login_headers  # noqa: E402

client = TestClient(appmod.app)
H_BUDI = login_headers(client, "u_budi")
H_ADMIN = login_headers(client, "u_rina")

# User khusus supaya data uji tidak menggeser hitungan test lain yang memakai
# DB bersama (lihat catatan di HANDOFF: satu instance `db` untuk semua modul).
UID = "u_budi"


def _cleanup():
    conn = appmod.CONN
    conn.execute("DELETE FROM notifications WHERE user_id = ?", (UID,))
    conn.execute("DELETE FROM notification_log WHERE user_id = ?", (UID,))
    conn.execute(
        """DELETE FROM chat_messages WHERE session_id IN
           (SELECT id FROM chat_sessions WHERE user_id = ?)""", (UID,))
    conn.execute("DELETE FROM chat_sessions WHERE user_id = ?", (UID,))
    conn.commit()


def _make_sessions(n):
    conn = appmod.CONN
    ids = []
    for i in range(n):
        sid = f"s_page_{i:03d}"
        conn.execute(
            "INSERT INTO chat_sessions (id, user_id, created_at, updated_at, status)"
            " VALUES (?,?,?,?, 'active')",
            (sid, UID, f"2026-01-01T00:{i % 60:02d}:00Z",
             f"2026-01-01T00:{i % 60:02d}:00Z"))
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, text, citations, created_at)"
            " VALUES (?, 'user', ?, '[]', '2026-01-01T00:00:00Z')", (sid, f"halo {i}"))
        ids.append(sid)
    conn.commit()
    return ids


def test_chat_sessions_pages_and_reports_total():
    _cleanup()
    _make_sessions(7)
    first = client.get("/api/chat/sessions?limit=3&offset=0", headers=H_BUDI).json()
    assert first["count"] == 3
    assert first["total"] == 7, first
    assert first["has_more"] is True
    second = client.get("/api/chat/sessions?limit=3&offset=3", headers=H_BUDI).json()
    assert second["count"] == 3
    third = client.get("/api/chat/sessions?limit=3&offset=6", headers=H_BUDI).json()
    assert third["count"] == 1
    assert third["has_more"] is False
    seen = [s["id"] for s in first["sessions"] + second["sessions"] + third["sessions"]]
    assert len(seen) == 7 and len(set(seen)) == 7, "halaman tidak boleh tumpang tindih"
    _cleanup()


def test_chat_sessions_skips_empty_sessions_before_limiting():
    """Sesi kosong difilter di SQL, bukan setelah LIMIT.

    Kalau difilter setelah LIMIT, satu halaman bisa berisi 10 sesi kosong dan
    tampil kosong semua padahal ada sesi berisi di halaman berikutnya.
    """
    _cleanup()
    conn = appmod.CONN
    for i in range(5):
        conn.execute(
            "INSERT INTO chat_sessions (id, user_id, created_at, updated_at, status)"
            " VALUES (?,?,?,?, 'active')",
            (f"s_empty_{i}", UID, f"2026-02-01T00:00:0{i}Z",
             f"2026-02-01T00:00:0{i}Z"))
    _make_sessions(2)
    conn.commit()
    body = client.get("/api/chat/sessions?limit=2", headers=H_BUDI).json()
    assert body["total"] == 2, body
    assert len(body["sessions"]) == 2
    assert body["has_more"] is False
    _cleanup()


def test_announcements_offset_and_unread_is_global():
    """`unread` = seluruh pengumuman belum dibaca, bukan hanya halaman ini."""
    full = client.get("/api/announcements?limit=100", headers=H_BUDI).json()
    total = full["total"]
    assert total >= 1, full
    page = client.get("/api/announcements?limit=1&offset=0", headers=H_BUDI).json()
    assert page["count"] == 1
    assert page["total"] == total
    assert page["has_more"] is (total > 1)
    if total > 1:
        second = client.get("/api/announcements?limit=1&offset=1", headers=H_BUDI).json()
        assert second["items"][0]["id"] != page["items"][0]["id"]


def test_notifications_offset_reaches_items_past_first_page():
    _cleanup()
    for i in range(5):
        portal._notify(appmod.CONN, [UID], f"notif {i}", None, category="system")
    appmod.CONN.commit()
    p1 = client.get("/api/notifications?limit=2&offset=0", headers=H_BUDI).json()
    p2 = client.get("/api/notifications?limit=2&offset=2", headers=H_BUDI).json()
    p3 = client.get("/api/notifications?limit=2&offset=4", headers=H_BUDI).json()
    assert (p1["count"], p2["count"], p3["count"]) == (2, 2, 1)
    assert p1["total"] == 5
    assert p3["has_more"] is False
    ids = [i["id"] for i in p1["items"] + p2["items"] + p3["items"]]
    assert len(set(ids)) == 5
    _cleanup()


def test_notifications_offset_beyond_end_is_empty_not_error():
    _cleanup()
    body = client.get("/api/notifications?limit=10&offset=999", headers=H_BUDI).json()
    assert body["items"] == []
    assert body["has_more"] is False
    _cleanup()
