"""Regresi rate-limit login: bucket per (IP, user), bukan per IP proxy bersama.

Bug produksi: semua request masuk lewat reverse proxy, sehingga
`request.client.host` selalu IP egress yang sama. Satu bucket dipakai bersama
semua pengunjung -> login acak-acakan dapat 429 walau baru sekali klik.

Modul diuji langsung dari `ratelimit.py` (tanpa membuka DB aplikasi).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import ratelimit  # noqa: E402


class _Client:
    def __init__(self, host):
        self.host = host


class _Req:
    def __init__(self, headers=None, host="10.0.0.1"):
        self.headers = headers or {}
        self.client = _Client(host)


def test_client_ip_prefers_xff_first_hop():
    req = _Req({"x-forwarded-for": "203.0.113.9, 10.0.0.1", "x-real-ip": "10.0.0.1"})
    assert ratelimit.client_ip(req) == "203.0.113.9"


def test_client_ip_ignores_spoofable_x_real_ip():
    """X-Real-IP diteruskan apa adanya oleh proxy -> tidak boleh jadi kunci bucket."""
    req = _Req({"x-real-ip": "9.9.9.9"}, host="198.51.100.4")
    assert ratelimit.client_ip(req) == "198.51.100.4"


def test_client_ip_falls_back_to_socket():
    assert ratelimit.client_ip(_Req({}, host="198.51.100.4")) == "198.51.100.4"


def test_client_ip_handles_missing_request():
    assert ratelimit.client_ip(None) == "unknown"


def test_rate_limit_isolated_per_user_behind_same_ip():
    """Banyak login user berbeda dari IP proxy yang sama tidak saling mengunci."""
    ratelimit.reset()
    ip = "10.1.2.3"
    for i in range(25):
        ratelimit.rate_limit(f"login:{ip}:u_user{i}", ratelimit.LOGIN_RATE_LIMIT)
    # satu user yang sama tetap terkena limit
    hit = False
    for _ in range(ratelimit.LOGIN_RATE_LIMIT + 1):
        try:
            ratelimit.rate_limit(f"login:{ip}:u_spam", ratelimit.LOGIN_RATE_LIMIT)
        except Exception as e:
            hit = getattr(e, "status_code", None) == 429
    assert hit, "user yang sama harus tetap dibatasi"


def test_repeated_login_same_user_still_limited():
    ratelimit.reset()
    key = "login:10.9.9.9:u_dewi"
    for _ in range(ratelimit.LOGIN_RATE_LIMIT):
        ratelimit.rate_limit(key, ratelimit.LOGIN_RATE_LIMIT)
    try:
        ratelimit.rate_limit(key, ratelimit.LOGIN_RATE_LIMIT)
    except Exception as e:
        assert getattr(e, "status_code", None) == 429
    else:
        raise AssertionError("login berulang harus kena 429")


def test_buckets_do_not_share_across_different_ips():
    ratelimit.reset()
    for _ in range(ratelimit.LOGIN_RATE_LIMIT + 5):
        try:
            ratelimit.rate_limit("login:1.1.1.1:u_a", ratelimit.LOGIN_RATE_LIMIT)
        except Exception:
            pass
    # IP lain untuk user yang sama tidak terpengaruh
    ratelimit.rate_limit("login:2.2.2.2:u_a", ratelimit.LOGIN_RATE_LIMIT)


def test_expired_buckets_are_pruned(monkeypatch):
    """Dict `_RATE` tidak boleh tumbuh selamanya.

    Bucket kedaluwarsa tidak pernah dipakai lagi (permintaan berikutnya membuat
    entri baru), tetapi dulu tetap tersimpan sampai proses mati. Satu bucket per
    IP+kategori+user berarti proses panjang dengan banyak klien unik menahan
    memori tanpa batas.
    """
    ratelimit.reset()
    monkeypatch.setattr(ratelimit, "_PRUNE_THRESHOLD", 10)
    monkeypatch.setattr(ratelimit, "RATE_WINDOW", 60)

    # Isi 10 bucket lalu mundurkan waktunya agar semuanya kedaluwarsa.
    for i in range(10):
        ratelimit.rate_limit("old:%d" % i, 5)
    for rec in ratelimit._RATE.values():
        rec[0] -= 3600

    # Panggilan berikutnya melewati ambang -> bucket mati dibuang.
    ratelimit.rate_limit("new:1", 5)
    keys = set(ratelimit._RATE)
    assert keys == {"new:1"}, keys
    ratelimit.reset()
