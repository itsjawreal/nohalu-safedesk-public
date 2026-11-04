"""HOTFIX H — DB tidak boleh terkunci selama panggilan LLM.

Bug produksi (P0-2): `POST /api/chat` menahan `_DB_LOCK` + transaksi tulis
selama `answer_question()` (LLM, timeout 180 s). Akibatnya satu pertanyaan
lambat membekukan seluruh aplikasi; abort klien meninggalkan
`backend/data/safedesk.db-journal` yang mengunci DB sampai server restart.

Tes ini memastikan:
- endpoint lain tetap cepat saat chat lambat (lock tidak ditahan saat I/O),
- pesan user sudah ter-commit sebelum LLM dipanggil (tidak hilang saat abort),
- WAL + busy_timeout aktif.
"""
import os
import sqlite3
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

_TMP = tempfile.mkdtemp()
os.environ.setdefault("NOHALU_DB_PATH", os.path.join(_TMP, "concurrency.db"))
os.environ["NOHALU_LLM_ENABLED"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import agent as agentmod  # noqa: E402
import app as appmod  # noqa: E402
import db as dbmod  # noqa: E402

# `db.DEFAULT_DB_PATH` sudah dibaca saat modul pertama di-import (mungkin oleh
# conftest), jadi pakai path yang benar-benar dipakai app — bukan env di sini.
DB_PATH = appmod.CONN.db_path
client = TestClient(appmod.app)


def _h(uid):
    from authutil import login_headers
    return login_headers(client, uid)


def test_wal_and_busy_timeout_enabled():
    conn = appmod.CONN._conn()
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000
    print("  ok wal_and_busy_timeout_enabled")


def test_second_connection_can_write_while_server_idle():
    """Penulis lain tidak langsung `database is locked` (busy_timeout bekerja)."""
    other = sqlite3.connect(DB_PATH, timeout=5)
    other.execute("PRAGMA busy_timeout = 5000")
    other.execute(
        "INSERT INTO notifications (user_id, message, created_at) VALUES (?,?,?)",
        ("u_budi", "uji konkurensi", "2026-10-30T00:00:00Z"))
    other.commit()
    other.close()
    print("  ok second_connection_can_write_while_server_idle")


def test_slow_llm_does_not_block_other_endpoints(monkeypatch):
    """Selama chat menunggu LLM, endpoint lain harus tetap < 1,5 s."""
    gate = threading.Event()

    def slow_answer(*a, **k):
        gate.wait(timeout=10)
        return {"answer": "jawaban lambat", "citations": [], "audit_id": 1,
                "conflict_detected": False, "conflict_note": "",
                "permission_decision": "allow", "refusal_reason": None,
                "pii_masked": False, "prompt_injection_detected": False,
                "selected_sources": [], "engine_used": "deterministic",
                "llm_model": None, "llm_ms": None, "llm_error": None,
                "suggestions": ["Apa kebijakan cuti?"]}

    monkeypatch.setattr(agentmod, "run", slow_answer)

    result = {}

    def run_chat():
        result["r"] = client.post("/api/chat", headers=_h("u_budi"),
                                  json={"question": "pertanyaan lambat"})

    t = threading.Thread(target=run_chat, daemon=True)
    t.start()
    time.sleep(0.4)  # biarkan chat masuk ke panggilan LLM

    t0 = time.time()
    r = client.get("/api/notifications", headers=_h("u_budi"))
    elapsed = time.time() - t0
    assert r.status_code == 200, r.text
    assert elapsed < 1.5, f"endpoint lain tertahan {elapsed:.2f}s selama LLM"

    gate.set()
    t.join(timeout=10)
    assert result["r"].status_code == 200, result["r"].text
    print(f"  ok slow_llm_does_not_block_other_endpoints ({elapsed:.2f}s)")


def test_user_message_committed_before_llm(monkeypatch):
    """Pesan user harus sudah ada di DB saat LLM masih berjalan (tahan abort)."""
    seen = {}

    def inspect_then_fail(*a, **k):
        # Panggilan ini terjadi SETELAH commit pesan user.
        row = sqlite3.connect(DB_PATH, timeout=5).execute(
            "SELECT COUNT(*) FROM chat_messages WHERE role='user' AND text=?",
            ("pesan yang harus tersimpan",)).fetchone()
        seen["count"] = row[0]
        raise RuntimeError("llm mati di tengah")

    monkeypatch.setattr(agentmod, "run", inspect_then_fail)
    r = client.post("/api/chat", headers=_h("u_budi"),
                    json={"question": "pesan yang harus tersimpan"})
    assert r.status_code == 500
    assert seen.get("count") == 1, "pesan user belum ter-commit sebelum LLM"
    print("  ok user_message_committed_before_llm")


def test_no_journal_file_left_after_error():
    """Tidak ada file `-journal`/transaksi menggantung setelah request gagal."""
    client.post("/api/chat", headers=_h("u_budi"), json={"question": "cek jurnal"})
    assert not os.path.exists(DB_PATH + "-journal"), "file -journal tertinggal"
    print("  ok no_journal_file_left_after_error")


def test_connection_is_thread_local():
    assert isinstance(appmod.CONN, dbmod.ThreadLocalConnection)
    box = {}

    def worker():
        box["conn"] = appmod.CONN._conn()

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert box["conn"] is not appmod.CONN._conn(), "koneksi dipakai bersama antar-thread"
    print("  ok connection_is_thread_local")


def test_middleware_discards_pending_transaction_after_crash():
    """Middleware membuang transaksi sisa ketika handler async gagal.

    Ini jalur yang dipakai endpoint `async` (unggah berkas, SSE): semuanya
    berjalan di thread event loop, jadi koneksi yang menahan transaksi sisa
    adalah koneksi milik thread itu — bukan threadpool. Tanpa
    `CONN.discard_pending()` di blok `finally`, transaksi terbuka itu ikut
    ter-commit oleh request berikutnya.

    Sengaja memanggil middleware langsung (bukan lewat TestClient) supaya
    penempatan thread-nya deterministik: `asyncio.run` menjalankan app palsu
    dan middleware di thread yang sama dengan `CONN`.
    """
    import asyncio

    marker = "uji-middleware-bocor"

    async def crashing_app(scope, receive, send):
        appmod.CONN.execute(
            "INSERT INTO notifications (user_id, message, created_at) VALUES (?,?,?)",
            ("u_budi", marker, "2026-10-30T00:00:00Z"))
        raise RuntimeError("gagal setelah menulis, sebelum commit")

    mw = appmod.RequestLogMiddleware(crashing_app)
    scope = {"type": "http", "method": "POST", "path": "/api/uji",
             "headers": [], "query_string": b""}

    async def noop_receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def noop_send(message):
        return None

    try:
        asyncio.run(mw(scope, noop_receive, noop_send))
    except RuntimeError:
        pass  # exception diteruskan ke handler FastAPI, seperti aslinya

    conn = appmod.CONN._conn()
    assert not conn.in_transaction, "transaksi sisa dibiarkan terbuka di event loop"
    left = sqlite3.connect(DB_PATH, timeout=5).execute(
        "SELECT COUNT(*) FROM notifications WHERE message=?", (marker,)
    ).fetchone()[0]
    assert left == 0, "tulisan parsial dari handler gagal jadi durable"
    print("  ok middleware_discards_pending_transaction_after_crash")


def test_thread_local_connection_discards_pending_on_scope_change():
    """`_conn()` membuang transaksi sisa saat thread dipakai request lain."""
    path = DB_PATH
    conn = dbmod.connect_threadlocal(path)
    dbmod.begin_request_scope("req-1")
    conn.execute(
        "INSERT INTO notifications (user_id, message, created_at) VALUES (?,?,?)",
        ("u_budi", "sisa transaksi", "2026-10-30T00:00:00Z"))
    assert conn._conn().in_transaction, "penulisan harus membuka transaksi"

    # Request berikutnya: scope berubah, transaksi sisa harus dibuang.
    dbmod.begin_request_scope("req-2")
    conn._conn()
    assert not conn._conn().in_transaction, "transaksi sisa tidak dibuang"

    other = sqlite3.connect(path, timeout=5).execute(
        "SELECT COUNT(*) FROM notifications WHERE message='sisa transaksi'"
    ).fetchone()[0]
    assert other == 0, "baris dari transaksi yang dibuang malah durable"
    print("  ok thread_local_connection_discards_pending_on_scope_change")


def test_db_locked_rolls_back_on_exception_without_middleware():
    """Lapis ketiga: `db_locked` me-rollback walau middleware tak terlibat.

    Middleware hanya menutup jalur HTTP. Fungsi portal dipanggil juga dari
    agent (tool call) dan job latar; kalau salah satunya menulis lalu gagal,
    transaksinya harus tetap dibuang. Tes ini memanggil route yang dibungkus
    `db_locked` langsung, tanpa request HTTP.
    """
    marker = "uji-db-locked-bocor"

    def boom(*a, **k):
        appmod.CONN.execute(
            "INSERT INTO notifications (user_id, message, created_at) VALUES (?,?,?)",
            ("u_budi", marker, "2026-10-30T00:00:00Z"))
        raise RuntimeError("gagal di dalam db_locked")

    wrapped = appmod.db_locked(boom)
    try:
        wrapped()
    except RuntimeError:
        pass

    assert not appmod.CONN._conn().in_transaction, \
        "db_locked membiarkan transaksi terbuka setelah exception"
    left = sqlite3.connect(DB_PATH, timeout=5).execute(
        "SELECT COUNT(*) FROM notifications WHERE message=?", (marker,)
    ).fetchone()[0]
    assert left == 0, "tulisan parsial dari db_locked jadi durable"
    print("  ok db_locked_rolls_back_on_exception_without_middleware")
