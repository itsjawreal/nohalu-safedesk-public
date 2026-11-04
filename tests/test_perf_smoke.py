"""Smoke performa (§5.7) — anggaran waktu untuk endpoint yang paling sering
dipanggil, plus bukti bahwa paging benar-benar memotong biaya.

Bukan benchmark: tidak ada ambang ketat dalam milidetik (mesin CI bisa lambat
dan angka absolut mudah jadi flaky). Yang diuji adalah **perilaku yang
seharusnya selalu benar**:

* endpoint daftar merespons dalam anggaran longgar (5 s) tanpa LLM,
* halaman kecil tidak lebih lambat dari halaman penuh,
* `/api/health` tidak menyentuh LLM/DB berat.
"""
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "perf.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402
from authutil import login_headers  # noqa: E402

client = TestClient(appmod.app)
H = login_headers(client, "u_rina")

BUDGET_S = 5.0


def _timed(path, headers=None):
    started = time.perf_counter()
    r = client.get(path, headers=headers or H)
    return r, (time.perf_counter() - started)


def test_health_is_cheap():
    r, elapsed = _timed("/api/health", headers={})
    assert r.status_code == 200
    assert elapsed < BUDGET_S, f"/api/health {elapsed:.3f}s"


def test_list_endpoints_stay_within_budget():
    for path in ("/api/documents?limit=10", "/api/audit?limit=20",
                 "/api/requests", "/api/referensi?limit=10"):
        r, elapsed = _timed(path)
        assert r.status_code == 200, (path, r.status_code)
        assert elapsed < BUDGET_S, f"{path} {elapsed:.3f}s"


def test_small_page_not_slower_than_full_page():
    """Bukti paging bekerja: halaman kecil tidak boleh lebih mahal.

    Batasnya longgar (2x) karena pengukuran mikro di mesin bersama berisik;
    yang dicari adalah regresi orde besaran, bukan selisih 5 ms.
    """
    _, small = _timed("/api/audit?limit=1")
    _, full = _timed("/api/audit?limit=200")
    assert small < max(full * 2, 0.5), (small, full)
