"""Regresi: seeding idempotent tidak boleh meninggalkan write-lock WAL.

Bug yang pernah terjadi: `seed_personal()` dan `ensure_seed_docs()` hanya
memanggil `commit()` bila ada baris baru. Saat server dijalankan ulang di DB
yang sudah ter-seed, `INSERT OR IGNORE` tetap membuka transaksi tulis tetapi
tidak pernah di-commit — koneksi startup menahan write-lock WAL selamanya dan
SEMUA request tulis berikutnya gagal `database is locked` (chat 500).
"""
import os
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import db  # noqa: E402
from seed import seed_personal  # noqa: E402


def _fresh_db():
    path = os.path.join(tempfile.mkdtemp(prefix="safedesk-lock-"), "lock.db")
    conn = db.connect(path)
    db.seed_if_empty(conn)
    db.ensure_seed_docs(conn)
    db.ensure_seed_users(conn)
    seed_personal(conn)
    return path, conn


def test_seeding_does_not_hold_write_lock():
    path, conn = _fresh_db()
    # jalankan ulang seluruh seeding (kondisi restart server di DB ter-seed)
    db.seed_if_empty(conn)
    db.ensure_seed_docs(conn)
    db.ensure_seed_users(conn)
    seed_personal(conn)

    other = sqlite3.connect(path, timeout=1.0)
    other.execute("PRAGMA busy_timeout = 1000")
    try:
        other.execute(
            "INSERT INTO notifications (user_id, message, created_at) VALUES (?,?,?)",
            ("u_budi", "uji write-lock", "2026-10-30T00:00:00Z"))
        other.commit()
    finally:
        other.close()
        conn.close()
    print("  ok seeding_does_not_hold_write_lock")


if __name__ == "__main__":
    test_seeding_does_not_hold_write_lock()
