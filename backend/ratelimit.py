"""Rate-limit in-memory + resolusi IP klien di belakang reverse proxy.

Dipisah dari app.py supaya bisa diuji tanpa membuka DB.
"""
import datetime
import os
import threading

from fastapi import HTTPException

CHAT_RATE_LIMIT = int(os.environ.get("NOHALU_CHAT_RATE_LIMIT", "30"))
LOGIN_RATE_LIMIT = int(os.environ.get("NOHALU_LOGIN_RATE_LIMIT", "20"))
# Pengajuan & approval: cegah spam / double-click.
REQUEST_RATE_LIMIT = int(os.environ.get("NOHALU_REQUEST_RATE_LIMIT", "10"))
DECIDE_RATE_LIMIT = int(os.environ.get("NOHALU_DECIDE_RATE_LIMIT", "30"))
READ_RATE_LIMIT = int(os.environ.get("NOHALU_READ_RATE_LIMIT", "120"))
# Endpoint publik tanpa sesi (landing page). Bucket per IP, lebih longgar dari
# READ karena satu kunjungan halaman bisa memicu beberapa permintaan.
PUBLIC_RATE_LIMIT = int(os.environ.get("NOHALU_PUBLIC_RATE_LIMIT", "60"))
RATE_WINDOW = 60

_RATE = {}
_RATE_LOCK = threading.Lock()
# Ambang pemangkasan: bucket yang sudah lewat jendela tidak pernah dipakai lagi
# (permintaan berikutnya membuat entri baru), tetapi dulu tetap tersimpan
# selamanya. Satu bucket per IP+kategori+user berarti proses panjang dengan
# banyak klien unik (proxy Vercel, bot, percobaan login) menumbuhkan dict ini
# tanpa batas. Dipangkas saat ukurannya melewati ambang, bukan setiap panggilan,
# supaya biayanya tetap O(1) amortized.
_PRUNE_THRESHOLD = int(os.environ.get("NOHALU_RATE_PRUNE_THRESHOLD", "2048"))


def _prune_locked(now, window):
    """Buang bucket kedaluwarsa. Dipanggil dengan `_RATE_LOCK` sudah dipegang."""
    dead = [k for k, rec in _RATE.items() if (now - rec[0]) > window]
    for k in dead:
        _RATE.pop(k, None)
    return len(dead)


def reset() -> None:
    """Kosongkan semua bucket (dipakai tes)."""
    with _RATE_LOCK:
        _RATE.clear()


def rate_limit(key, limit, window=RATE_WINDOW):
    now = datetime.datetime.now(datetime.timezone.utc).timestamp()
    with _RATE_LOCK:
        rec = _RATE.get(key)
        if not rec or (now - rec[0]) > window:
            if len(_RATE) >= _PRUNE_THRESHOLD:
                _prune_locked(now, window)
            _RATE[key] = [now, 1]
            return
        rec[1] += 1
        if rec[1] > limit:
            raise HTTPException(status_code=429, detail="rate_limited")


def client_ip(request) -> str:
    """IP klien sebenarnya di belakang reverse proxy (Vercel → InstaBox).

    Tanpa ini, `request.client.host` selalu IP egress proxy yang dipakai bersama
    semua pengunjung, sehingga satu bucket rate-limit menampung semua orang dan
    login acak-acakan kena 429.

    Hanya `X-Forwarded-For` yang dipercaya: diuji di produksi, Vercel MENIMPA
    header ini dengan IP klien, sedangkan `X-Real-IP` diteruskan apa adanya dari
    klien (bisa dipalsukan) sehingga tidak dipakai sebagai kunci bucket.
    """
    if request is None:
        return "unknown"
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        first = fwd.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"
