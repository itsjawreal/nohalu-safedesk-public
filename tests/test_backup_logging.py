"""P2 — cadangan harian + retensi, log terstruktur, dan health DB.

Sebelum ini tidak ada job cadangan sama sekali (crontab kosong di server) dan
log hanya `print()` bebas yang tidak bisa diagregasi.
"""
import json
import logging
import os
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import backup  # noqa: E402
import logging_setup  # noqa: E402


def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t(a INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()


def test_backup_creates_file_and_prunes(tmp_path):
    db = str(tmp_path / "app.db")
    _make_db(db)
    names = []
    for day in range(1, 10):
        res = backup.run(db, keep=7, stamp=f"202601{day:02d}-0300")
        names.append(res["name"])
        assert res["bytes"] > 0
        assert os.path.exists(res["path"])
    kept = backup.list_backups(db)
    assert len(kept) == 7, kept
    # Yang tersisa harus yang terbaru.
    assert kept[0]["name"] == names[-1]
    assert names[0] not in [k["name"] for k in kept]


def test_backup_is_readable_snapshot(tmp_path):
    db = str(tmp_path / "app.db")
    _make_db(db)
    res = backup.run(db, keep=7, stamp="20260101-0300")
    copy = sqlite3.connect(res["path"])
    assert copy.execute("SELECT a FROM t").fetchone()[0] == 1
    copy.close()


def test_backup_same_minute_does_not_overwrite(tmp_path):
    db = str(tmp_path / "app.db")
    _make_db(db)
    a = backup.run(db, keep=7, stamp="20260101-0300")
    b = backup.run(db, keep=7, stamp="20260101-0300")
    assert a["name"] != b["name"]
    assert len(backup.list_backups(db)) == 2


def test_prune_never_deletes_foreign_files(tmp_path):
    db = str(tmp_path / "app.db")
    _make_db(db)
    folder = backup.backup_dir(db)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "catatan.txt"), "w") as fh:
        fh.write("bukan cadangan")
    for day in range(1, 4):
        backup.run(db, keep=1, stamp=f"202601{day:02d}-0300")
    assert os.path.exists(os.path.join(folder, "catatan.txt"))


def test_json_log_line_is_parseable(capsys):
    import importlib
    importlib.reload(logging_setup)  # paksa handler baru (modul idempoten per proses)
    log = logging_setup.get_logger("test")
    logging_setup.event(log, "unit", "halo", level=logging.WARNING, n=3, db="x")
    raw = capsys.readouterr().out.strip().splitlines()[-1]
    body = json.loads(raw)
    assert body["event"] == "unit"
    assert body["level"] == "warning"
    assert body["msg"] == "halo"
    assert body["n"] == 3 and body["db"] == "x"
    assert body["ts"].endswith("Z")


def test_integrity_check_reports_ok(tmp_path):
    db = str(tmp_path / "app.db")
    _make_db(db)
    conn = sqlite3.connect(db)
    out = logging_setup.integrity_check(conn)
    conn.close()
    assert out["ok"] is True
    assert out["detail"] == "ok"
    assert out["foreign_key_violations"] == 0


def test_integrity_check_never_raises_on_broken_connection():
    class Broken:
        def execute(self, *a, **k):
            raise sqlite3.DatabaseError("disk image is malformed")
    out = logging_setup.integrity_check(Broken())
    assert out["ok"] is False
    assert "malformed" in out["detail"]


def test_backup_endpoints_admin_only(tmp_path, monkeypatch):
    """Endpoint cadangan: admin boleh, HR ditolak, dan hasilnya tercatat."""
    os.environ["NOHALU_DB_PATH"] = str(tmp_path / "api.db")
    os.environ["NOHALU_LLM_ENABLED"] = "0"
    os.environ["NOHALU_REMINDERS"] = "0"
    os.environ["NOHALU_BACKUP_ENABLED"] = "0"
    from fastapi.testclient import TestClient
    import app as appmod
    from authutil import login_headers

    client = TestClient(appmod.app)
    h_admin = login_headers(client, "u_rina")
    h_hr = login_headers(client, "u_sari")

    r = client.post("/api/admin/backups", headers=h_admin)
    assert r.status_code == 200, r.text
    body = r.json()
    # Path absolut server tidak boleh bocor ke klien; klien hanya butuh nama.
    assert "path" not in body, body
    assert backup.NAME_RE.match(body["name"]), body["name"]
    assert body["kept"] >= 1

    r = client.get("/api/admin/backups", headers=h_admin)
    assert r.status_code == 200
    listed = r.json()
    assert "dir" not in listed, listed
    assert any(b["name"] == body["name"] for b in listed["backups"])

    assert client.post("/api/admin/backups", headers=h_hr).status_code == 403
    assert client.get("/api/admin/backups", headers=h_hr).status_code == 403


def test_health_reports_backup_and_db(tmp_path):
    os.environ["NOHALU_DB_PATH"] = str(tmp_path / "h.db")
    os.environ["NOHALU_LLM_ENABLED"] = "0"
    os.environ["NOHALU_REMINDERS"] = "0"
    os.environ["NOHALU_BACKUP_ENABLED"] = "0"
    from fastapi.testclient import TestClient
    import app as appmod

    body = TestClient(appmod.app).get("/api/health").json()
    assert body["db"] == "ok"
    assert body["backup"]["keep"] == 7
    assert "count" in body["backup"]
