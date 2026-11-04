"""Landing publik — GET /api/public/stats (audit 2026-11-01).

Konteks: halaman landing dulu menulis angka sendiri ("64 dokumen", "4 peran",
"0 jawaban tanpa sumber") dan ketiganya melenceng dari DB. Endpoint ini
membuat angka halaman berasal dari satu sumber yang sama dengan aplikasi.

Yang dikunci tes ini:
  * bentuk respons (7 kunci) stabil untuk klien halaman
  * tidak ada data pribadi / isi dokumen yang ikut terkirim
  * angka cocok dengan hitungan DB langsung, bukan angka karangan
  * `accessible` memakai jalur ACL yang sama dengan chat
  * header cache ada (halaman tidak menghantam DB tiap muat)
  * rate limit bekerja per IP
  * DB bermasalah -> 503, bukan angka palsu
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "public_stats.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"
# Batas rate-limit TIDAK diatur lewat env: modul tes lain sudah mengimpor
# `app` lebih dulu saat suite berjalan penuh, jadi nilai env di sini tidak
# terbaca. Tes yang butuh batas kecil menambal konstanta modul (lihat
# `test_rate_limited`).

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402
import portal  # noqa: E402
import ratelimit  # noqa: E402
from authutil import login_headers  # noqa: E402

client = TestClient(appmod.app)

KEYS = {"documents", "roles", "workspaces", "audit_rows", "accessible", "generated_at"}


def _fresh():
    """Kosongkan bucket rate-limit: setiap tes memanggil endpoint beberapa kali."""
    ratelimit.reset()


def _get():
    return client.get("/api/public/stats")


# --------------------------------------------------------------- bentuk respons
def test_shape():
    _fresh()
    r = _get()
    assert r.status_code == 200, r.text[:200]
    body = r.json()
    assert set(body) == KEYS
    assert isinstance(body["accessible"], dict)
    assert body["generated_at"].endswith("Z")


def test_no_personal_or_document_fields():
    """Nilai apa pun yang bisa mengidentifikasi orang/dokumen tidak boleh ada."""
    _fresh()
    body = _get().json()
    flat = repr(body)
    for forbidden in ("u_rina", "u_budi", "Rina", "Budi", "@nusantarakreatif",
                      "Kenari", "doc_hr", "content", "email"):
        assert forbidden not in flat, f"respons memuat '{forbidden}'"


def test_numbers_match_db():
    _fresh()
    body = _get().json()
    conn = appmod.CONN
    docs = conn.execute(
        "SELECT COUNT(*) FROM documents WHERE deleted_at IS NULL").fetchone()[0]
    roles = conn.execute(
        "SELECT COUNT(DISTINCT role) FROM users WHERE employment_status='active'"
    ).fetchone()[0]
    ws = conn.execute("SELECT COUNT(*) FROM workspaces").fetchone()[0]
    audit = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    assert body["documents"] == docs
    assert body["roles"] == roles
    assert body["workspaces"] == ws
    assert body["audit_rows"] == audit


def test_accessible_matches_acl_path():
    """Angka per peran harus sama dengan jalur ACL yang dipakai chat."""
    _fresh()
    body = _get().json()
    expected = portal.public_stats(appmod.CONN)["accessible"]
    assert body["accessible"] == expected
    # Peran bawaan seed wajib ada — halaman memakai tiga di antaranya.
    for role in ("employee", "hr", "team_lead"):
        assert role in body["accessible"]
    # Tidak ada peran yang bisa mengakses lebih banyak dari total dokumen.
    for role, n in body["accessible"].items():
        assert 0 <= n <= body["documents"], f"{role}={n} > total"


def test_cache_header():
    _fresh()
    r = _get()
    assert r.headers.get("cache-control") == "public, max-age=300"


def test_rate_limited(monkeypatch):
    """Batas kecil ditambal di konstanta modul, bukan lewat env.

    Urutan koleksi pytest menentukan modul mana yang mengimpor `app` lebih
    dulu; kalau batas diset lewat env, tes ini lulus sendiri tetapi gagal saat
    suite berjalan penuh (batas sudah terlanjur 60).
    """
    monkeypatch.setattr(appmod, "PUBLIC_RATE_LIMIT", 3)
    _fresh()
    codes = [_get().status_code for _ in range(4)]
    assert codes[:3] == [200, 200, 200], codes
    assert codes[3] == 429, codes


def test_db_failure_returns_503(monkeypatch):
    _fresh()
    def boom(conn):
        raise RuntimeError("db down")
    monkeypatch.setattr(portal, "public_stats", boom)
    r = _get()
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "stats_unavailable"


def test_endpoint_is_public():
    """Tanpa header Authorization pun tetap 200 — halaman memanggilnya sebelum login.

    Dijalankan lewat klien TANPA header apa pun; kalau route ini tidak sengaja
    dibungkus `Depends(parse_token)`, tes gagal 401.
    """
    _fresh()
    r = client.get("/api/public/stats")
    assert r.status_code == 200
    assert r.json()["documents"] > 0


def test_unknown_auth_still_ok():
    """Token ngawur tidak boleh mengubah hasil: endpoint tidak membaca identitas."""
    _fresh()
    r = client.get("/api/public/stats", headers={"Authorization": "Bearer palsu"})
    assert r.status_code == 200
    assert r.json()["documents"] == _get().json()["documents"]
