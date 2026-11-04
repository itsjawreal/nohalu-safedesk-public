"""P0 hardening — sesi bertanda tangan, revokasi, dan jejak audit.

Sebelum ini, token = `user_id` plaintext: `Bearer u_rina` langsung menjadi
admin tanpa login. Tes ini menjaga agar celah itu tidak kembali.
"""
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "auth_api.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402
import auth  # noqa: E402
import portal  # noqa: E402

client = TestClient(appmod.app)


def _login(uid="u_budi"):
    r = client.post("/api/auth/login", json={"user_id": uid})
    assert r.status_code == 200, r.text
    return r.json()


def _h(token):
    return {"Authorization": "Bearer " + token}


def test_plain_user_id_token_is_rejected():
    """Inti perbaikan: id mentah bukan lagi token yang sah."""
    r = client.get("/api/me", headers=_h("u_rina"))
    assert r.status_code == 401, r.text
    assert r.json()["detail"] in ("invalid_token", "session_revoked")


def test_login_returns_signed_token_with_ttl():
    body = _login()
    tok = body["token"]
    assert tok != "u_budi"
    assert tok.startswith("v1.u_budi.")
    assert body["ttl_seconds"] > 0
    assert body["expires_at"].endswith("Z")
    r = client.get("/api/me", headers=_h(tok))
    assert r.status_code == 200
    assert r.json()["id"] == "u_budi"


def test_two_logins_get_distinct_tokens():
    """Tanpa jti, dua login pada detik yang sama menghasilkan token identik
    dan INSERT ke `sessions` gagal karena UNIQUE(token_hash)."""
    a = _login("u_maya")["token"]
    b = _login("u_maya")["token"]
    assert a != b


def test_logout_revokes_token():
    tok = _login("u_andi")["token"]
    assert client.get("/api/me", headers=_h(tok)).status_code == 200
    r = client.post("/api/auth/logout", headers=_h(tok))
    assert r.status_code == 200
    r2 = client.get("/api/me", headers=_h(tok))
    assert r2.status_code == 401
    assert r2.json()["detail"] == "session_revoked"


def test_logout_is_idempotent_and_safe_without_token():
    tok = _login("u_nadia")["token"]
    assert client.post("/api/auth/logout", headers=_h(tok)).status_code == 200
    assert client.post("/api/auth/logout", headers=_h(tok)).status_code == 200
    assert client.post("/api/auth/logout").status_code == 200


def test_logout_all_revokes_every_session_of_user():
    t1 = _login("u_fajar")["token"]
    t2 = _login("u_fajar")["token"]
    r = client.post("/api/auth/logout-all", headers=_h(t1))
    assert r.status_code == 200
    assert r.json()["revoked"] >= 2
    assert client.get("/api/me", headers=_h(t1)).status_code == 401
    assert client.get("/api/me", headers=_h(t2)).status_code == 401


def test_tampered_signature_is_rejected():
    tok = _login("u_budi")["token"]
    parts = tok.split(".")
    parts[4] = "A" * len(parts[4])  # tanda tangan dirusak
    r = client.get("/api/me", headers=_h(".".join(parts)))
    assert r.status_code == 401
    assert r.json()["detail"] == "invalid_token"


def test_expired_token_is_rejected(monkeypatch):
    tok = auth.issue_token(appmod.DEFAULT_DB_PATH, "u_budi", ttl_hours_=-1)["token"]
    with appmod._DB_LOCK:
        appmod.CONN.execute(
            "INSERT INTO sessions (token_hash,user_id,issued_at,expires_at) VALUES (?,?,?,?)",
            (auth.token_hash(tok), "u_budi", "2020-01-01T00:00:00Z",
             "2020-01-01T01:00:00Z"))
        appmod.CONN.commit()
    r = client.get("/api/me", headers=_h(tok))
    assert r.status_code == 401
    assert r.json()["detail"] == "token_expired"


def test_missing_bearer_prefix_is_rejected():
    tok = _login()["token"]
    r = client.get("/api/me", headers={"Authorization": tok})
    assert r.status_code == 401
    assert r.json()["detail"] == "invalid_authorization"


def test_admin_reading_foreign_session_is_audited():
    """Admin boleh membaca sesi orang lain, tetapi harus meninggalkan jejak."""
    budi_tok = _login("u_budi")["token"]
    sid = "sess_audit_probe"
    with appmod._DB_LOCK:
        appmod.CONN.execute(
            "INSERT OR REPLACE INTO chat_sessions (id,user_id,created_at,status)"
            " VALUES (?,?,?,?)", (sid, "u_budi", "2026-01-01T00:00:00Z", "active"))
        appmod.CONN.execute(
            "INSERT INTO chat_messages (session_id,role,text,created_at) VALUES (?,?,?,?)",
            (sid, "user", "halo", "2026-01-01T00:00:01Z"))
        appmod.CONN.execute("DELETE FROM audit_log WHERE event_type='chat_session_read'")
        appmod.CONN.commit()

    admin_tok = _login("u_rina")["token"]
    r = client.get("/api/chat/" + sid, headers=_h(admin_tok))
    assert r.status_code == 200, r.text

    with appmod._DB_LOCK:
        row = appmod.CONN.execute(
            "SELECT user_id, details FROM audit_log WHERE event_type='chat_session_read'"
            " ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None, "admin membaca sesi orang lain tanpa jejak audit"
    assert row["user_id"] == "u_rina"
    assert sid in (row["details"] or "")


def test_owner_reading_own_session_is_not_audited():
    budi_tok = _login("u_budi")["token"]
    sid = "sess_own_probe"
    with appmod._DB_LOCK:
        appmod.CONN.execute(
            "INSERT OR REPLACE INTO chat_sessions (id,user_id,created_at,status)"
            " VALUES (?,?,?,?)", (sid, "u_budi", "2026-01-01T00:00:00Z", "active"))
        appmod.CONN.execute("DELETE FROM audit_log WHERE event_type='chat_session_read'")
        appmod.CONN.commit()
    assert client.get("/api/chat/" + sid, headers=_h(budi_tok)).status_code == 200
    with appmod._DB_LOCK:
        n = appmod.CONN.execute(
            "SELECT COUNT(*) c FROM audit_log WHERE event_type='chat_session_read'"
        ).fetchone()["c"]
    assert n == 0


def test_upload_rejects_oversized_file(monkeypatch):
    monkeypatch.setattr(appmod, "MAX_UPLOAD_BYTES", 1024)
    admin_tok = _login("u_rina")["token"]
    big = b"x" * 4096
    r = client.post("/api/documents/upload",
                    files={"file": ("besar.txt", big, "text/plain")},
                    data={"title": "Besar"},
                    headers=_h(admin_tok))
    assert r.status_code == 413, r.text
    assert r.json()["detail"] == "file_too_large"


def test_upload_accepts_small_file(monkeypatch):
    monkeypatch.setattr(appmod, "MAX_UPLOAD_BYTES", 1024 * 1024)
    admin_tok = _login("u_rina")["token"]
    r = client.post("/api/documents/upload",
                    files={"file": ("kecil.txt", b"isi dokumen kecil", "text/plain")},
                    data={"title": "Kecil", "workspace_ids": "ws_onboarding"},
                    headers=_h(admin_tok))
    assert r.status_code == 200, r.text
    assert r.json()["id"].startswith("doc_")
