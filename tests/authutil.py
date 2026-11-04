"""Helper tes: dapatkan header Authorization yang SAH (token bertanda tangan).

Sejak P0 hardening, `Bearer <user_id>` tidak diterima lagi — token harus
ditandatangani server dan tercatat di tabel `sessions`. Tes yang dulu memakai
id mentah sekarang login sekali lalu memakai token hasilnya.

Cache ditempel pada objek klien (bukan `id(client)`): id bisa dipakai ulang
setelah klien lama di-GC, dan itu membuat tes lintas modul memakai token milik
klien/DB lain sehingga gagal 401 secara acak.
"""
_ATTR = "_safedesk_auth_cache"


def _cache(client):
    cache = getattr(client, _ATTR, None)
    if cache is None:
        cache = {}
        setattr(client, _ATTR, cache)
    return cache


def login_headers(client, uid):
    cache = _cache(client)
    cached = cache.get(uid)
    if cached:
        return dict(cached)
    r = client.post("/api/auth/login", json={"user_id": uid})
    assert r.status_code == 200, f"login {uid} gagal: {r.status_code} {r.text[:200]}"
    h = {"Authorization": "Bearer " + r.json()["token"]}
    cache[uid] = h
    return dict(h)


def token_for(client, uid):
    """Token mentah (dipakai tes yang membangun request HTTP manual)."""
    return login_headers(client, uid)["Authorization"].split(" ", 1)[1]


def clear(client=None):
    if client is None:
        return
    try:
        delattr(client, _ATTR)
    except AttributeError:
        pass
