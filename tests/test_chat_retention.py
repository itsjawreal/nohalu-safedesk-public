"""P1 — retensi riwayat chat + pembersihan sesi kosong.

Riwayat chat memuat pertanyaan pribadi karyawan (gaji, cuti, keluhan). Tanpa
job retensi, tabel tumbuh tanpa batas dan data itu tersimpan selamanya.
Sesi kosong (dibuat lalu ditinggalkan) juga sampah: UI menyembunyikannya,
jadi barisnya hanya memperlambat `GET /api/chat/sessions`.

Kontrak:
* sesi lebih tua dari `CHAT_RETENTION_DAYS` dihapus beserta pesannya,
* sesi baru / masih dalam masa retensi TIDAK tersentuh,
* sesi kosong dihapus hanya setelah masa tenggang (bukan seketika — user
  mungkin masih mengetik di tab yang baru dibuka),
* job idempotent: putaran kedua tidak menghapus apa pun lagi.
"""
import datetime as dt
import os
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "retention.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

import db  # noqa: E402
import reminders  # noqa: E402


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    db.seed_if_empty(conn)
    db.ensure_seed_users(conn)
    return conn


def _iso(delta_days=0, delta_hours=0):
    when = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
        days=delta_days, hours=delta_hours)
    return when.isoformat().replace("+00:00", "Z")


def _session(conn, sid, created, with_message=True):
    conn.execute(
        "INSERT INTO chat_sessions (id, user_id, created_at, updated_at, status)"
        " VALUES (?,?,?,?, 'active')", (sid, "u_budi", created, created))
    if with_message:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, text, citations, created_at)"
            " VALUES (?, 'user', 'halo', '[]', ?)", (sid, created))
    conn.commit()


def _ids(conn):
    return {r["id"] for r in conn.execute("SELECT id FROM chat_sessions").fetchall()}


def test_purges_sessions_older_than_retention_with_their_messages():
    conn = make_conn()
    old = _iso(delta_days=reminders.CHAT_RETENTION_DAYS + 30)
    _session(conn, "s_old", old)
    assert reminders.job_purge_chat_sessions(conn) == 1
    assert "s_old" not in _ids(conn)
    left = conn.execute(
        "SELECT COUNT(*) AS n FROM chat_messages WHERE session_id = 's_old'"
    ).fetchone()["n"]
    assert left == 0, "pesan milik sesi yang dihapus harus ikut terhapus"


def test_keeps_recent_sessions():
    conn = make_conn()
    _session(conn, "s_new", _iso(delta_days=1))
    assert reminders.job_purge_chat_sessions(conn) == 0
    assert "s_new" in _ids(conn)


def test_purges_only_old_empty_sessions_not_fresh_ones():
    """Sesi kosong baru bisa saja tab yang sedang dibuka user — jangan dibuang."""
    conn = make_conn()
    _session(conn, "s_empty_old", _iso(delta_hours=reminders.EMPTY_SESSION_GRACE_HOURS + 2),
             with_message=False)
    _session(conn, "s_empty_new", _iso(delta_hours=1), with_message=False)
    assert reminders.job_purge_chat_sessions(conn) == 1
    remaining = _ids(conn)
    assert "s_empty_old" not in remaining
    assert "s_empty_new" in remaining


def test_job_is_idempotent():
    conn = make_conn()
    _session(conn, "s_old", _iso(delta_days=reminders.CHAT_RETENTION_DAYS + 1))
    assert reminders.job_purge_chat_sessions(conn) == 1
    assert reminders.job_purge_chat_sessions(conn) == 0


def test_job_registered_in_run_once():
    assert "job_purge_chat_sessions" in reminders.JOBS
    conn = make_conn()
    out = reminders.run_once(conn)
    assert out["job_purge_chat_sessions"] == 0


def test_purges_in_chunks_so_sqlite_variable_limit_is_not_hit(monkeypatch):
    """Ribuan sesi kedaluwarsa sekaligus tidak boleh membuat job gagal.

    SQLite membatasi jumlah variabel per statement (999 di build lama). Satu
    `IN (?,?,...)` untuk semua sesi akan melempar `too many SQL variables`,
    dan job yang sama akan gagal lagi setiap 15 menit — retensi berhenti
    diam-diam. Ukuran chunk diperkecil di sini supaya batas itu benar-benar
    terlewati tanpa membuat tesnya lambat.
    """
    monkeypatch.setattr(reminders, "PURGE_CHUNK", 3)
    conn = make_conn()
    old = _iso(delta_days=reminders.CHAT_RETENTION_DAYS + 5)
    total = 10
    for i in range(total):
        _session(conn, f"s_chunk_{i:02d}", old)

    assert reminders.job_purge_chat_sessions(conn) == total
    assert _ids(conn) == set()
    left = conn.execute("SELECT COUNT(*) AS n FROM chat_messages").fetchone()["n"]
    assert left == 0
