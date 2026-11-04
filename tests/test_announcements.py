"""Sprint 3.3 — pengumuman HR: RBAC, status baca, 404 untuk yang tak berhak."""
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "ann_api.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402

client = TestClient(appmod.app)
DOC = "doc_hr_transport_2026"


def _h(uid):
    r = client.post("/api/auth/login", json={"user_id": uid})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


def _clear_reads(uid):
    appmod.CONN.execute("DELETE FROM announcement_reads WHERE user_id = ?", (uid,))
    appmod.CONN.commit()


def test_employee_sees_announcement_with_excerpt():
    _clear_reads("u_budi")
    r = client.get("/api/announcements", headers=_h("u_budi"))
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    item = next(i for i in body["items"] if i["id"] == DOC)
    assert item["is_read"] is False
    assert "tunjangan transport" in item["excerpt"].lower()
    assert "content" not in item, "list tidak boleh mengirim isi penuh dokumen"
    # Garis waktu di UI butuh "hari ini" dari server supaya penanda tanggalnya
    # tidak bergeser mengikuti zona waktu perangkat.
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", body["today"])


def test_mark_read_is_idempotent_and_personal():
    _clear_reads("u_budi")
    h = _h("u_budi")
    before = client.get("/api/announcements", headers=h).json()
    # Seed memuat lebih dari satu pengumuman; yang diuji adalah efek satu
    # dokumen, bukan "semua pengumuman jadi terbaca".
    assert before["count"] > 1, "uji ini butuh lebih dari satu pengumuman"

    assert client.post(f"/api/announcements/{DOC}/read", headers=h).status_code == 200
    assert client.post(f"/api/announcements/{DOC}/read", headers=h).status_code == 200
    body = client.get("/api/announcements", headers=h).json()
    assert next(i for i in body["items"] if i["id"] == DOC)["is_read"] is True
    assert body["unread"] == before["unread"] - 1, "hanya satu pengumuman yang terbaca"
    others = [i for i in body["items"] if i["id"] != DOC]
    assert all(i["is_read"] is False for i in others)

    # Status baca milik user lain tidak ikut berubah.
    _clear_reads("u_sari")
    other = client.get("/api/announcements", headers=_h("u_sari")).json()
    assert next(i for i in other["items"] if i["id"] == DOC)["is_read"] is False


def test_announcement_rbac_respects_workspace_roles():
    """Dokumen announcement restricted tidak bocor ke role yang tidak berhak."""
    appmod.CONN.execute(
        """INSERT OR REPLACE INTO documents
           (id,title,source_type,effective_date,workspace_ids,access_roles,content,
            owner,confidentiality,status,version,content_hash)
           VALUES ('doc_ann_secret','Rahasia Direksi','hr_announcement','2026-10-01',
                   ?,?, 'hanya direksi','HR','restricted','indexed','1',NULL)""",
        ('["ws_finance"]', '["admin"]'),
    )
    appmod.CONN.commit()
    try:
        ids_emp = {i["id"] for i in client.get("/api/announcements",
                                              headers=_h("u_budi")).json()["items"]}
        assert "doc_ann_secret" not in ids_emp
        r = client.post("/api/announcements/doc_ann_secret/read", headers=_h("u_budi"))
        assert r.status_code == 404
        ids_admin = {i["id"] for i in client.get("/api/announcements",
                                                headers=_h("u_rina")).json()["items"]}
        assert "doc_ann_secret" in ids_admin
    finally:
        appmod.CONN.execute("DELETE FROM documents WHERE id='doc_ann_secret'")
        appmod.CONN.commit()


def test_unknown_announcement_404():
    r = client.post("/api/announcements/doc_tidak_ada/read", headers=_h("u_budi"))
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "announcement_not_found"


def test_mark_all_announcements_read_only_touches_allowed_docs():
    """`read-all` tidak boleh menulis baris baca untuk dokumen di luar RBAC."""
    appmod.CONN.execute(
        """INSERT OR REPLACE INTO documents
           (id,title,source_type,effective_date,workspace_ids,access_roles,content,
            owner,confidentiality,status,version,content_hash)
           VALUES ('doc_ann_secret2','Rahasia 2','hr_announcement','2026-10-02',
                   '["ws_finance"]','["admin"]','rahasia','HR','restricted',
                   'indexed','1',NULL)"""
    )
    appmod.CONN.commit()
    _clear_reads("u_budi")
    try:
        r = client.post("/api/announcements/read-all", headers=_h("u_budi"))
        assert r.status_code == 200
        body = r.json()
        assert body["updated"] >= 1
        assert body["unread"] == 0, body
        rows = {x["doc_id"] for x in appmod.CONN.execute(
            "SELECT doc_id FROM announcement_reads WHERE user_id='u_budi'").fetchall()}
        assert "doc_ann_secret2" not in rows, "dokumen restricted tidak boleh ditandai"
        assert client.get("/api/announcements",
                          headers=_h("u_budi")).json()["unread"] == 0
        # Badge lonceng ikut bersih (kategori `announcement`).
        notif = client.get("/api/notifications", headers=_h("u_budi")).json()
        assert notif["unread_by_category"]["announcement"] == 0, notif
        # User lain tidak terpengaruh.
        _clear_reads("u_sari")
        assert client.get("/api/announcements",
                          headers=_h("u_sari")).json()["unread"] >= 1
    finally:
        appmod.CONN.execute("DELETE FROM documents WHERE id='doc_ann_secret2'")
        appmod.CONN.execute("DELETE FROM announcement_reads WHERE user_id='u_budi'")
        appmod.CONN.commit()


def test_mark_one_announcement_read_clears_its_notification_only():
    """Membuka satu pengumuman membersihkan notifikasinya, bukan yang lain."""
    _clear_reads("u_budi")
    appmod.CONN.execute(
        """INSERT OR REPLACE INTO documents
           (id,title,source_type,effective_date,workspace_ids,access_roles,content,
            owner,confidentiality,status,version,content_hash)
           VALUES ('doc_ann_a','Pengumuman A','hr_announcement','2026-10-05',
                   '["ws_onboarding"]','["employee","hr","admin"]','isi A','HR','internal',
                   'indexed','1',NULL)""")
    appmod.CONN.execute(
        """INSERT OR REPLACE INTO documents
           (id,title,source_type,effective_date,workspace_ids,access_roles,content,
            owner,confidentiality,status,version,content_hash)
           VALUES ('doc_ann_b','Pengumuman B','hr_announcement','2026-10-06',
                   '["ws_onboarding"]','["employee","hr","admin"]','isi B','HR','internal',
                   'indexed','1',NULL)""")
    appmod.CONN.execute("DELETE FROM notifications WHERE user_id='u_budi'")
    appmod.CONN.commit()
    try:
        import portal
        portal._notify(appmod.CONN, ["u_budi"], "Pengumuman A baru",
                       "/app/pengumuman", category="announcement", ref="doc_ann_a")
        portal._notify(appmod.CONN, ["u_budi"], "Pengumuman B baru",
                       "/app/pengumuman", category="announcement", ref="doc_ann_b")
        appmod.CONN.commit()
        r = client.post("/api/announcements/doc_ann_a/read", headers=_h("u_budi"))
        assert r.status_code == 200
        assert r.json()["unread"] == 1, r.json()
        rows = {x["ref"]: x["is_read"] for x in appmod.CONN.execute(
            "SELECT ref, is_read FROM notifications WHERE user_id='u_budi' "
            "AND category='announcement'").fetchall()}
        assert rows["doc_ann_a"] == 1
        assert rows["doc_ann_b"] == 0, "pengumuman lain tidak boleh ikut terbaca"
    finally:
        appmod.CONN.execute("DELETE FROM documents WHERE id IN ('doc_ann_a','doc_ann_b')")
        appmod.CONN.execute("DELETE FROM notifications WHERE user_id='u_budi'")
        appmod.CONN.execute("DELETE FROM announcement_reads WHERE user_id='u_budi'")
        appmod.CONN.commit()
