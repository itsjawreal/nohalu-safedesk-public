"""KPI §11 — metrik yang dulu berkolom "tidak terukur".

Plan §11 mencantumkan p95 latensi chat, laju 5xx, waktu approve, rasio
grounded, dan refusal tanpa angka apa pun. Modul `metrics.py` mengisi itu:

* runtime (in-memory) diisi middleware untuk SETIAP request `/api/*`,
* turunan DB dihitung saat `/api/admin/kpi` dipanggil.

Tes ini mengunci kontrak bentuk + aritmetika, bukan angka produksi.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "kpi.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402
import metrics  # noqa: E402
from authutil import login_headers  # noqa: E402

client = TestClient(appmod.app)
H_RINA = login_headers(client, "u_rina")
H_BUDI = login_headers(client, "u_budi")


def test_p95_uses_nearest_rank_not_mean():
    metrics.reset()
    for ms in range(1, 101):
        metrics.record("/api/x", ms, 200)
    snap = metrics.snapshot()
    assert snap["paths"]["/api/x"]["p95_ms"] == 95, snap
    assert snap["requests"] == 100


def test_error_rate_counts_only_5xx():
    metrics.reset()
    for status in (200, 200, 404, 500):
        metrics.record("/api/y", 5, status)
    snap = metrics.snapshot()
    assert snap["errors_5xx"] == 1
    assert snap["error_rate"] == 0.25
    # 4xx bukan error server: 404 di atas tidak boleh ikut dihitung.
    assert snap["paths"]["/api/y"]["errors_5xx"] == 1


def test_chat_sessions_are_grouped_under_one_key():
    """`/api/chat/<id>` tidak boleh jadi ratusan baris terpisah di laporan."""
    metrics.reset()
    metrics.record("/api/chat/abc", 10, 200)
    metrics.record("/api/chat/def", 20, 200)
    snap = metrics.snapshot()
    assert "/api/chat/{session_id}" in snap["paths"], snap["paths"]
    assert snap["chat_p95_ms"] is not None


def test_middleware_records_real_requests():
    metrics.reset()
    client.get("/api/health")  # dikecualikan dari log, jadi tidak dihitung
    client.get("/api/me/leave", headers=H_BUDI)
    snap = metrics.snapshot()
    assert snap["requests"] >= 1, snap


def test_kpi_endpoint_requires_hr_and_returns_shape():
    assert client.get("/api/admin/kpi", headers=H_BUDI).status_code == 403
    r = client.get("/api/admin/kpi", headers=H_RINA)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"runtime", "data", "llm_usage"}
    for key in ("requests", "errors_5xx", "error_rate", "paths"):
        assert key in body["runtime"], key
    for key in ("audit_rows", "grounded_rate", "refusal_rate",
                "approve_avg_hours", "multi_turn_rate"):
        assert key in body["data"], key
    # Pemakaian token harus terlihat: tanpa angka ini, pertumbuhan payload
    # hanya bisa dirasakan sebagai keluhan, bukan diukur.
    for key in ("calls", "prompt_tokens", "completion_tokens",
                "max_prompt_tokens", "max_prompt_chars", "truncated"):
        assert key in body["llm_usage"], key


def test_db_snapshot_computes_grounded_and_approve_latency():
    """Hitung delta, bukan mengosongkan tabel.

    Semua modul tes berbagi satu instance `db` (path dibaca saat import
    pertama), jadi `DELETE FROM audit_log` di sini akan merusak tes lain yang
    berjalan sesudahnya. Baris uji memakai penanda unik lalu dihapus lagi.
    """
    conn = appmod.CONN
    before = metrics.db_snapshot(conn)
    conn.execute(
        """INSERT INTO audit_log (timestamp, user_id, question, answer,
           citations, refusal_reason) VALUES
           ('2026-01-01T00:00:00Z','u_kpi_probe','kpi-q1','a1','["doc_x"]',NULL),
           ('2026-01-01T00:00:01Z','u_kpi_probe','kpi-q2','a2','[]','no_access')""")
    conn.execute(
        """INSERT INTO requests (user_id,type,title,status,approver_roles,
           stages,stage_idx,history,created_at,decided_at)
           VALUES ('u_kpi_probe','cuti','KPI probe','approved','[]','[]',0,'[]',
                   '2026-01-01T00:00:00Z','2026-01-01T12:00:00Z')""")
    conn.commit()
    try:
        after = metrics.db_snapshot(conn)
        assert after["audit_rows"] == before["audit_rows"] + 2
        assert after["approve_samples"] == before["approve_samples"] + 1
        # Hitungan absolut, BUKAN arah rasio. Arah rasio bergantung isi DB
        # dari modul tes lain (semua modul berbagi satu instance `db`): kalau
        # DB sudah penuh refusal, menambah satu baris refusal justru
        # MENURUNKAN rasio. Asersi arah rasio adalah flake yang menunggu
        # urutan tes berubah.
        assert after["grounded_rows"] == before["grounded_rows"] + 1
        assert after["refused_rows"] == before["refused_rows"] + 1
        assert after["approve_avg_hours"] is not None
    finally:
        conn.execute("DELETE FROM audit_log WHERE user_id = 'u_kpi_probe'")
        conn.execute("DELETE FROM requests WHERE user_id = 'u_kpi_probe'")
        conn.commit()


def test_window_drops_samples_older_than_one_hour():
    """p95 harus berarti "jendela terakhir", bukan "seumur proses".

    Tanpa batas waktu, satu lonjakan lambat pagi hari tetap menaikkan p95
    sampai proses di-restart — angka itu berhenti menggambarkan kondisi
    sekarang, dan operator jadi mengejar regresi yang sudah hilang.
    """
    import time as _time
    metrics.reset()
    # Sampel "lama": dimundurkan melewati jendela, langsung ke deque internal.
    old = _time.time() - (metrics.WINDOW_SECONDS + 60)
    with metrics._LOCK:
        for _ in range(50):
            metrics._SAMPLES.append((old, "/api/old", 9000, 200))
    for _ in range(5):
        metrics.record("/api/now", 10, 200)

    snap = metrics.snapshot()
    assert "/api/old" not in snap["paths"], snap["paths"]
    assert snap["requests"] == 5, snap
    assert snap["paths"]["/api/now"]["p95_ms"] == 10


def test_covered_seconds_never_exceeds_window():
    """`covered_seconds` menjelaskan arti angka, bukan menyesatkan."""
    metrics.reset()
    metrics.record("/api/x", 1, 200)
    snap = metrics.snapshot()
    assert 0 <= snap["covered_seconds"] <= snap["window_seconds"]
    assert snap["samples"] == snap["requests"]


def test_id_paths_grouped_so_report_lists_endpoints_not_ids():
    """Laporan p95 harus berisi endpoint, bukan satu baris per dokumen."""
    metrics.reset()
    for doc in ("doc_17", "doc_42", "doc_99"):
        metrics.record("/api/documents/" + doc, 5, 200)
    metrics.record("/api/documents/versions", 7, 200)
    snap = metrics.snapshot()
    assert "/api/documents/{id}" in snap["paths"], snap["paths"]
    assert "/api/documents/versions" in snap["paths"], snap["paths"]
    assert snap["paths"]["/api/documents/{id}"]["count"] == 3


def test_short_uptime_is_not_reported_as_sample_cap():
    """Proses baru hidup 5 detik bukan "batas sampel tercapai".

    Dua sebab jendela pendek harus dibedakan: uptime (wajar, tidak perlu
    tindakan) vs deque penuh (volume tinggi, p95 berhenti mencakup 1 jam).
    UI memakai `sample_capped` untuk memutuskan apakah menampilkan peringatan;
    kalau salah, setiap restart memunculkan peringatan palsu.
    """
    metrics.reset()
    metrics.record("/api/x", 1, 200)
    snap = metrics.snapshot()
    assert snap["sample_capped"] is False, snap
    assert snap["uptime_seconds"] >= 0
    assert snap["covered_seconds"] <= snap["uptime_seconds"]


def test_sse_stream_not_counted_as_latency():
    """Stream yang ditahan terbuka bukan latensi.

    `/api/notifications/stream` hidup selama tab browser terbuka. Kalau
    durasinya masuk p95, satu tab yang dibiarkan semalam menghasilkan
    "endpoint 8 jam" dan menenggelamkan regresi latensi yang sebenarnya.
    """
    metrics.reset()
    metrics.record("/api/notifications/stream", 8 * 3600 * 1000, 200)
    metrics.record("/api/me", 12, 200)
    snap = metrics.snapshot()
    assert "/api/notifications/stream" not in snap["paths"], snap["paths"]
    assert snap["requests"] == 1
