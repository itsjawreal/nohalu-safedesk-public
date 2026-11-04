"""Sprint 3.4 — SSE notifikasi: header, event `unread`, dan tidak menahan DB lock."""
import os
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "sse_api.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"
# Percepat stream supaya test tidak menunggu 55 detik.
os.environ["NOHALU_SSE_MAX_SECONDS"] = "2"
os.environ["NOHALU_SSE_TICK"] = "0.05"
os.environ["NOHALU_SSE_PING"] = "0.2"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402
import portal  # noqa: E402

client = TestClient(appmod.app)
from authutil import login_headers, token_for  # noqa: E402
H = login_headers(client, "u_budi")


def _clear():
    appmod.CONN.execute("DELETE FROM notifications WHERE user_id='u_budi'")
    appmod.CONN.execute("DELETE FROM notification_prefs WHERE user_id='u_budi'")
    appmod.CONN.commit()


def test_stream_headers_and_unread_event():
    _clear()
    portal._notify(appmod.CONN, ["u_budi"], "halo", None, category="system")
    appmod.CONN.commit()
    with client.stream("GET", "/api/notifications/stream", headers=H) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        assert "no-cache" in r.headers["cache-control"]
        assert r.headers.get("x-accel-buffering") == "no"
        seen = []
        for line in r.iter_lines():
            if line:
                seen.append(line)
            if len(seen) >= 3:
                break
    joined = "\n".join(seen)
    assert "event: unread" in joined
    assert '"unread": 1' in joined
    _clear()


def test_stream_does_not_hold_db_lock():
    """Request lain harus tetap jalan selama stream terbuka (lock tidak ditahan)."""
    _clear()
    result = {}

    def other_request():
        t0 = time.monotonic()
        r = client.get("/api/notifications", headers=H)
        result["status"] = r.status_code
        result["ms"] = (time.monotonic() - t0) * 1000

    with client.stream("GET", "/api/notifications/stream", headers=H) as r:
        it = r.iter_lines()
        first = next(it)  # stream sudah aktif & generator berjalan
        assert first == "event: unread"
        t = threading.Thread(target=other_request)
        t.start()
        t.join(timeout=3)
        assert not t.is_alive(), "GET /api/notifications terblokir oleh stream"
        assert result.get("status") == 200
        assert result.get("ms", 9999) < 2000
    _clear()


def test_stream_limit_per_user_takeover():
    """Kuota SSE dibatasi, tapi koneksi baru MENGGANTIKAN yang tertua.

    Regresi produksi: dulu kuota penuh dijawab 429. Karena proxy menahan
    koneksi upstream setelah tab ditutup, user sah terkunci 429 sampai TTL
    habis. Sekarang stream tertua di-evict, jadi stream baru selalu dapat slot
    dan jumlah stream hidup tetap <= SSE_MAX_PER_USER.
    """
    import app as appmod2
    old = appmod2.SSE_MAX_PER_USER
    appmod2.SSE_MAX_PER_USER = 1
    appmod2._SSE_ACTIVE.clear()
    try:
        with client.stream("GET", "/api/notifications/stream", headers=H) as r:
            assert r.status_code == 200
            it = r.iter_lines()
            assert next(it) == "event: unread"
            assert len(appmod2._SSE_ACTIVE["u_budi"]) == 1
            # Stream kedua: bukan 429, tapi takeover (slot lama di-evict).
            # Generator WAJIB dipegang variabelnya; `next(r.iter_lines())` inline
            # membuat generator langsung di-GC → stream ditutup → slot dilepas.
            with client.stream("GET", "/api/notifications/stream", headers=H) as r2:
                assert r2.status_code == 200
                it2 = r2.iter_lines()
                assert next(it2) == "event: unread"
                assert len(appmod2._SSE_ACTIVE["u_budi"]) == 1, \
                    "kuota tidak boleh terlampaui setelah takeover"
    finally:
        appmod2.SSE_MAX_PER_USER = old
        appmod2._SSE_ACTIVE.clear()
    # Setelah semua ditutup, slot kembali tersedia.
    with client.stream("GET", "/api/notifications/stream", headers=H) as r:
        assert r.status_code == 200
        assert next(r.iter_lines()) == "event: unread"


def test_release_frees_the_right_slot():
    """Menutup satu stream tidak boleh membebaskan slot stream lain yang hidup.

    Versi lama memakai `pop(0)` saat release: menutup stream **terbaru** justru
    membuang slot stream **tertua** yang masih terbuka, sehingga kuota per-user
    bisa terlampaui. Sekarang release memakai token, jadi hanya slot milik
    stream yang berakhir yang dilepas.

    Catatan: generator `iter_lines()` harus dipegang variabelnya. Kalau dibuat
    inline (`next(r.iter_lines())`), generator langsung di-GC dan stream ikut
    ditutup — test jadi mengukur hal yang salah.
    """
    import app as appmod2
    old = appmod2.SSE_MAX_PER_USER
    appmod2.SSE_MAX_PER_USER = 2
    appmod2._SSE_ACTIVE.clear()
    try:
        with client.stream("GET", "/api/notifications/stream", headers=H) as a:
            assert a.status_code == 200
            it_a = a.iter_lines()
            assert next(it_a) == "event: unread"
            slot_a = appmod2._SSE_ACTIVE["u_budi"][0]
            with client.stream("GET", "/api/notifications/stream", headers=H) as b:
                assert b.status_code == 200
                it_b = b.iter_lines()
                assert next(it_b) == "event: unread"
                assert len(appmod2._SSE_ACTIVE["u_budi"]) == 2
            # B (terbaru) ditutup; A masih hidup → slot A harus bertahan.
            assert appmod2._SSE_ACTIVE["u_budi"] == [slot_a], \
                "slot stream A (masih hidup) tidak boleh terhapus saat B ditutup"
            del it_b
        # A ditutup → kuota kosong.
        del it_a
        assert not appmod2._SSE_ACTIVE.get("u_budi")
    finally:
        appmod2.SSE_MAX_PER_USER = old
        appmod2._SSE_ACTIVE.clear()


def test_takeover_evicts_oldest_slot():
    """Kuota penuh → slot TERTUA yang di-evict, bukan yang terbaru."""
    import app as appmod2
    old = appmod2.SSE_MAX_PER_USER
    appmod2.SSE_MAX_PER_USER = 2
    appmod2._SSE_ACTIVE.clear()
    try:
        with client.stream("GET", "/api/notifications/stream", headers=H) as a:
            it_a = a.iter_lines()
            assert next(it_a) == "event: unread"
            slot_a = appmod2._SSE_ACTIVE["u_budi"][0]
            with client.stream("GET", "/api/notifications/stream", headers=H) as b:
                it_b = b.iter_lines()
                assert next(it_b) == "event: unread"
                slot_b = appmod2._SSE_ACTIVE["u_budi"][1]
                with client.stream("GET", "/api/notifications/stream", headers=H) as c:
                    it_c = c.iter_lines()
                    assert next(it_c) == "event: unread"
                    live = appmod2._SSE_ACTIVE["u_budi"]
                    assert len(live) == 2, "kuota tidak boleh terlampaui"
                    assert slot_a not in live, "slot tertua harus di-evict"
                    assert slot_b in live, "slot yang lebih baru harus bertahan"
            del it_c
        del it_b
        del it_a
    finally:
        appmod2.SSE_MAX_PER_USER = old
        appmod2._SSE_ACTIVE.clear()


def test_no_429_for_same_user_after_many_opens():
    """Membuka stream berkali-kali tidak boleh menghasilkan 429 (bug produksi)."""
    import app as appmod2
    old = appmod2.SSE_MAX_PER_USER
    appmod2.SSE_MAX_PER_USER = 2
    appmod2._SSE_ACTIVE.clear()
    try:
        codes = []
        for _ in range(6):
            with client.stream("GET", "/api/notifications/stream", headers=H) as r:
                codes.append(r.status_code)
                it = r.iter_lines()
                assert next(it) == "event: unread"
                assert len(appmod2._SSE_ACTIVE["u_budi"]) <= 2
        assert codes == [200] * 6, f"ada 429 tak terduga: {codes}"
    finally:
        appmod2.SSE_MAX_PER_USER = old
        appmod2._SSE_ACTIVE.clear()


def test_slot_released_fast_when_client_disconnects():
    """Klien tutup koneksi → slot kuota harus lepas dalam hitungan tick, bukan 55 s.

    Regresi produksi: generator hanya menyentuh socket saat ada event atau ping
    (25 s), jadi `finally` baru jalan setelah ~54 s. Akibatnya user sah kena 429
    berulang setelah menutup beberapa tab. Sekarang tiap tick ada heartbeat +
    `request.is_disconnected()`, jadi release harus < 2 s pada tick 0.05 s.

    Test memakai uvicorn nyata (bukan TestClient) karena butuh socket yang
    benar-benar ditutup sepihak oleh klien.
    """
    import socket as _socket
    import threading as _threading
    import time as _time

    import uvicorn
    import app as appmod2

    appmod2._SSE_ACTIVE.clear()

    sock = _socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(appmod2.app, host="127.0.0.1", port=port,
                            log_level="error", lifespan="off")
    server = uvicorn.Server(config)
    t = _threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = _time.time() + 15
    while not server.started and _time.time() < deadline:
        _time.sleep(0.05)
    assert server.started, "uvicorn test server gagal start"

    try:
        c = _socket.create_connection(("127.0.0.1", port), timeout=5)
        c.sendall(
            b"GET /api/notifications/stream HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            + ("Authorization: Bearer " + token_for(client, "u_budi") + "\r\n").encode()
            + b"Accept: text/event-stream\r\n\r\n")
        c.settimeout(5)
        head = c.recv(4096)
        assert b"200" in head.split(b"\r\n", 1)[0], head[:200]
        assert appmod2._SSE_ACTIVE.get("u_budi"), "slot harus terpakai saat stream hidup"

        # Tutup sepihak: server harus mendeteksi dan melepas slot dengan cepat.
        c.close()
        t0 = _time.time()
        while _time.time() - t0 < 5:
            if not appmod2._SSE_ACTIVE.get("u_budi"):
                break
            _time.sleep(0.05)
        elapsed = _time.time() - t0
        assert not appmod2._SSE_ACTIVE.get("u_budi"), \
            f"slot SSE masih tertahan {elapsed:.1f}s setelah klien putus"
        assert elapsed < 2.0, f"release terlalu lambat: {elapsed:.2f}s"
    finally:
        server.should_exit = True
        t.join(timeout=10)
        appmod2._SSE_ACTIVE.clear()
