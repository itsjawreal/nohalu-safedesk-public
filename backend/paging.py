"""paging.py — pagination bersama untuk endpoint daftar.

Sebelum ini `?limit=&offset=` diterima di URL tapi tidak pernah dibaca: endpoint
mengembalikan seluruh isi tabel (audit 500 baris / 103 KB, 70 dokumen, 69
referensi). Modul ini menyatukan cara membaca parameter dan memotong hasil,
supaya setiap daftar punya kontrak yang sama:

    {"items": [...], "count": <jumlah di halaman ini>,
     "total": <jumlah setelah filter>, "limit": n, "offset": n,
     "has_more": bool}

`count` tetap dipertahankan di respons spesifik (mis. `documents`, `audit`)
supaya klien lama tidak rusak.
"""
import math

DEFAULT_LIMIT = 50
MAX_LIMIT = 200


def window(limit, offset, default=DEFAULT_LIMIT, cap=MAX_LIMIT):
    """Normalisasi `limit`/`offset` dari query string.

    Nilai aneh (negatif, bukan angka, 0) diperlakukan sebagai default, bukan
    error: pagination tidak boleh membuat halaman gagal tampil.
    """
    try:
        lim = int(limit)
    except (TypeError, ValueError):
        lim = default
    try:
        off = int(offset)
    except (TypeError, ValueError):
        off = 0
    if lim <= 0:
        lim = default
    lim = min(lim, cap)
    off = max(0, off)
    return lim, off


def slice_page(items, limit, offset, default=DEFAULT_LIMIT, cap=MAX_LIMIT):
    """Potong daftar + metadata halaman. `total` = jumlah sebelum dipotong."""
    lim, off = window(limit, offset, default, cap)
    total = len(items)
    page = items[off:off + lim]
    return {
        "items": page,
        "count": len(page),
        "total": total,
        "limit": lim,
        "offset": off,
        "has_more": off + len(page) < total,
        "pages": max(1, math.ceil(total / lim)) if lim else 1,
    }


def spread(payload, key, page):
    """Tempel metadata halaman ke respons, sambil mempertahankan nama kunci asli.

    Endpoint lama mengembalikan `{"documents": [...]}`; klien tidak perlu diubah,
    jadi kunci aslinya diisi halaman dan metadata ditambahkan di sampingnya.
    """
    out = {key: page["items"]}
    out.update({k: v for k, v in page.items() if k != "items"})
    out.update(payload)
    return out
