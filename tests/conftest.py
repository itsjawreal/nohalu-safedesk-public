"""Konfigurasi test global.

`db.DEFAULT_DB_PATH` dibaca saat modul `db` pertama kali di-import, jadi env
harus dipasang SEBELUM modul test mana pun meng-import `db`/`app`. Tanpa ini,
urutan koleksi pytest menentukan apakah test menyentuh DB dev yang sedang
dipakai server lokal (dan gagal dengan `database is locked` saat ALTER TABLE).

Modul test yang butuh DB sendiri tetap boleh menimpa NOHALU_DB_PATH; nilai di
sini hanya default aman.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(ROOT, "backend") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "backend"))

_TMP = tempfile.mkdtemp(prefix="safedesk-test-")
os.environ.setdefault("NOHALU_DB_PATH", os.path.join(_TMP, "default.db"))
os.environ.setdefault("NOHALU_LLM_ENABLED", "0")
# Tes sekarang login sungguhan (token bertanda tangan) alih-alih memakai
# `Bearer <user_id>`. Satu suite menjalankan puluhan login dari IP TestClient
# yang sama, jadi kuota login produksi (20/menit) akan menghabiskan diri
# sendiri. `test_login_ratelimit.py` menguji modul ratelimit secara langsung,
# sehingga menaikkan batas di sini tidak melemahkan tes tersebut.
os.environ.setdefault("NOHALU_LOGIN_RATE_LIMIT", "100000")
