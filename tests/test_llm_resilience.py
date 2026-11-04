"""Regresi lapisan LLM: argumen tool rusak, jawaban terpotong, retry, aksi terlapor.

Empat bug nyata yang dijaga file ini (semua direproduksi sebelum diperbaiki):

1. `llm.chat()` mengubah argumen tool yang JSON-nya rusak menjadi `{}` dan
   pemanggil tetap MENJALANKAN tool itu. Karena hampir semua tool punya default
   untuk semua parameternya, `clock_attendance` mencatat absensi dan
   `mark_notifications_read` menandai semua notifikasi terbaca — aksi nyata
   dengan parameter yang tidak pernah diminta model.
2. `finish_reason="length"` diabaikan, sehingga "Cuti tahunan Anda adalah 12
   hari dan dapat" dikirim sebagai jawaban final yang seolah lengkap.
3. `_post()` tidak punya retry: satu 429/503 sesaat langsung menjatuhkan seluruh
   jawaban ke fallback deterministik.
4. Aksi yang SUDAH berjalan hilang dari jawaban ketika provider gagal di step
   berikutnya: user menerima kutipan kebijakan, `agent_trace` kosong, dan tidak
   diberi tahu bahwa datanya sudah berubah.
"""
import json
import os
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_LLM_ENABLED"] = "0"

import db
import engine
import agent
import llm


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    db.seed_if_empty(conn)
    db.ensure_seed_users(conn)
    return conn


def _body(tool_args, finish="tool_calls", text=""):
    return {"choices": [{
        "message": {"content": text, "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "clock_attendance", "arguments": tool_args}}]},
        "finish_reason": finish}], "model": "mercury-3", "usage": {}}


# ---------------------------------------------------------------------------
# 1. Argumen tool rusak
# ---------------------------------------------------------------------------
def test_broken_tool_arguments_are_flagged_not_silently_emptied(monkeypatch):
    monkeypatch.setattr(llm, "_post", lambda p, timeout=None: (_body('{"action": "in'), 5))
    out = llm.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    call = out["tool_calls"][0]
    assert call["arguments"] == {}
    assert call["arguments_error"] == "invalid_json", call
    # raw_arguments tetap apa adanya supaya bisa di-echo balik ke provider
    assert call["raw_arguments"] == '{"action": "in'


def test_valid_tool_arguments_have_no_error_flag(monkeypatch):
    monkeypatch.setattr(llm, "_post",
                        lambda p, timeout=None: (_body('{"action": "in"}'), 5))
    out = llm.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert out["tool_calls"][0]["arguments"] == {"action": "in"}
    assert out["tool_calls"][0]["arguments_error"] is None


def test_non_object_arguments_are_flagged(monkeypatch):
    monkeypatch.setattr(llm, "_post", lambda p, timeout=None: (_body('["in"]'), 5))
    out = llm.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert out["tool_calls"][0]["arguments_error"] == "not_an_object"


def test_agent_refuses_to_execute_tool_with_broken_arguments(monkeypatch):
    """Aksi tulis TIDAK boleh berjalan bila argumennya tidak bisa dibaca."""
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    calls = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"text": "", "finish_reason": "tool_calls", "ms": 3, "usage": {},
                    "model": "mercury-3", "truncated": False,
                    "tool_calls": [{
                        "id": "c1", "name": "clock_attendance",
                        "arguments": {}, "raw_arguments": '{"action": "in',
                        "arguments_error": "invalid_json"}]}
        return {"text": "Saya belum mencatat absensi karena argumennya tidak terbaca.",
                "tool_calls": [], "finish_reason": "stop", "ms": 2, "usage": {},
                "model": "mercury-3", "truncated": False}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Clock-in sekarang", None)

    rows = conn.execute(
        "SELECT clock_in FROM attendance WHERE user_id='u_budi'").fetchall()
    assert not [x for x in rows if x["clock_in"]], "absensi tidak boleh tercatat"
    bad = [t for t in r["agent_trace"] if t.get("error") == "bad_arguments"]
    assert bad, r["agent_trace"]
    assert bad[0]["tool"] == "clock_attendance"


def test_forced_round_does_not_execute_broken_arguments(monkeypatch):
    """Putaran paksa (`_force_action`) tidak boleh jadi pintu belakang bug #1.

    Jalur ini dulu memanggil `execute()` mentah, sehingga JSON argumen yang
    terpotong menjadi `{}` tetap menjalankan tool — persis lubang yang sudah
    ditutup di loop utama. Gerbangnya kini dipakai bersama.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    calls = {"n": 0}
    ran = []

    def fake_chat(messages, tools=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"text": "Baik, saya siapkan draft cuti Anda sekarang.",
                    "tool_calls": [], "finish_reason": "stop", "ms": 3,
                    "usage": {}, "model": "mercury-3", "truncated": False}
        return {"text": "", "finish_reason": "tool_calls", "ms": 3, "usage": {},
                "model": "mercury-3", "truncated": False,
                "tool_calls": [{"id": "c1", "name": "prepare_leave_request",
                                "arguments": {},
                                "raw_arguments": '{"start_date": "20',
                                "arguments_error": "invalid_json"}]}

    def spy_execute(ctx, name, args):
        ran.append((name, args))
        return {"ok": True}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    monkeypatch.setattr(agent, "execute", spy_execute)
    r = agent.run(conn, user, "Tolong buatkan draft cuti 20-22 Desember 2026", None)

    assert calls["n"] >= 2, "putaran paksa harus dicoba"
    assert not ran, ran
    assert r["llm_error"] == "agent:action_not_executed", r
    assert "belum membuat draft" in r["answer"].lower(), r["answer"]


# ---------------------------------------------------------------------------
# 2. Jawaban terpotong (finish_reason=length)
# ---------------------------------------------------------------------------
def test_truncated_flag_is_exposed(monkeypatch):
    body = {"choices": [{"message": {"content": "Cuti tahunan Anda adalah 12 hari dan dapat"},
                         "finish_reason": "length"}],
            "model": "mercury-3", "usage": {}}
    monkeypatch.setattr(llm, "_post", lambda p, timeout=None: (body, 5))
    out = llm.chat([{"role": "user", "content": "x"}])
    assert out["truncated"] is True
    assert out["finish_reason"] == "length"


def test_agent_continues_truncated_answer(monkeypatch):
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    calls = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"text": "Cuti tahunan Anda adalah 12 hari dan dapat",
                    "tool_calls": [], "finish_reason": "length", "ms": 3,
                    "usage": {}, "model": "mercury-3", "truncated": True}
        return {"text": "diambil maksimal 12 hari per tahun.",
                "tool_calls": [], "finish_reason": "stop", "ms": 2,
                "usage": {}, "model": "mercury-3", "truncated": False}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Halo", None)
    assert calls["n"] == 2, "model harus diminta melanjutkan sekali"
    assert "diambil maksimal" in r["answer"], r["answer"]


def test_agent_marks_answer_incomplete_when_continuation_fails(monkeypatch):
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    calls = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"text": "Cuti tahunan Anda adalah 12 hari dan dapat",
                    "tool_calls": [], "finish_reason": "length", "ms": 3,
                    "usage": {}, "model": "mercury-3", "truncated": True}
        raise RuntimeError("LLM HTTP 503: down")

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Halo", None)
    assert "terpotong" in r["answer"].lower(), r["answer"]


# ---------------------------------------------------------------------------
# 3. Retry terbatas pada 429/5xx/timeout
# ---------------------------------------------------------------------------
def _http_error(code):
    import urllib.error
    import io
    return urllib.error.HTTPError(
        "https://provider/v1/chat/completions", code, "err", {},
        io.BytesIO(b'{"error":"x"}'))


def test_retry_on_429_then_success(monkeypatch):
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setenv("NOHALU_LLM_ATTEMPTS", "3")
    monkeypatch.setenv("NOHALU_LLM_BACKOFF", "0")
    monkeypatch.setattr(llm, "api_key", lambda: "test-key")
    state = {"n": 0}

    def fake_urlopen(req, timeout=None):
        state["n"] += 1
        if state["n"] == 1:
            raise _http_error(429)
        return type("R", (), {"read": lambda self: json.dumps(
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
             "model": "mercury-3", "usage": {}}).encode(),
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: False})()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    out = llm.generate("s", "u")
    assert out["text"] == "ok"
    assert state["n"] == 2


def test_no_retry_on_400(monkeypatch):
    monkeypatch.setenv("NOHALU_LLM_ATTEMPTS", "3")
    monkeypatch.setenv("NOHALU_LLM_BACKOFF", "0")
    monkeypatch.setattr(llm, "api_key", lambda: "test-key")
    state = {"n": 0}

    def fake_urlopen(req, timeout=None):
        state["n"] += 1
        raise _http_error(400)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError) as exc:
        llm.generate("s", "u")
    assert "LLM HTTP 400" in str(exc.value)
    assert state["n"] == 1, "4xx non-429 tidak boleh diulang"


def test_gives_up_after_attempt_budget(monkeypatch):
    monkeypatch.setenv("NOHALU_LLM_ATTEMPTS", "3")
    monkeypatch.setenv("NOHALU_LLM_BACKOFF", "0")
    monkeypatch.setattr(llm, "api_key", lambda: "test-key")
    state = {"n": 0}

    def fake_urlopen(req, timeout=None):
        state["n"] += 1
        raise _http_error(503)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError):
        llm.generate("s", "u")
    assert state["n"] == 3


# ---------------------------------------------------------------------------
# 4. Aksi yang sudah berjalan tetap dilaporkan
# ---------------------------------------------------------------------------
def test_executed_action_is_reported_when_llm_dies_afterwards(monkeypatch):
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    calls = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"text": "", "finish_reason": "tool_calls", "ms": 3, "usage": {},
                    "model": "mercury-3", "truncated": False,
                    "tool_calls": [{
                        "id": "c1", "name": "clock_attendance",
                        "arguments": {"action": "in"},
                        "raw_arguments": '{"action": "in"}',
                        "arguments_error": None}]}
        raise RuntimeError("LLM HTTP 503: down")

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Clock-in sekarang", None)

    row = conn.execute(
        "SELECT clock_in FROM attendance WHERE user_id='u_budi'").fetchone()
    assert row and row["clock_in"], "aksi memang harus berjalan"
    assert "clock_attendance" in (r.get("executed") or []), r
    assert "sudah berubah" in r["answer"].lower(), r["answer"]
    assert r["llm_error"] == "agent:upstream_error"
    # Jangan sampai user diberi kutipan kebijakan yang menyiratkan tidak ada aksi
    assert "jam kerja normal" not in r["answer"].lower()


def test_no_executed_report_when_only_read_tools_ran(monkeypatch):
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    calls = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"text": "", "finish_reason": "tool_calls", "ms": 3, "usage": {},
                    "model": "mercury-3", "truncated": False,
                    "tool_calls": [{
                        "id": "c1", "name": "get_leave_balance", "arguments": {},
                        "raw_arguments": "{}", "arguments_error": None}]}
        raise RuntimeError("LLM HTTP 503: down")

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Berapa sisa cuti saya?", None)
    assert not r.get("executed"), r
    assert r["llm_error"] == "agent:upstream_error"


# ---------------------------------------------------------------------------
# 5. Status publik tidak membocorkan URL provider
# ---------------------------------------------------------------------------
def test_public_status_hides_provider_url():
    assert "base_url" not in llm.status(detail=False)
    assert "base_url" in llm.status(detail=True)


# ---------------------------------------------------------------------------
# 6. Circuit breaker: provider mati tidak boleh menahan tiap request
# ---------------------------------------------------------------------------
def test_circuit_opens_after_repeated_failures(monkeypatch):
    llm.reset_circuit()
    monkeypatch.setenv("NOHALU_LLM_ATTEMPTS", "1")
    monkeypatch.setenv("NOHALU_LLM_BACKOFF", "0")
    monkeypatch.setenv("NOHALU_LLM_CB_THRESHOLD", "2")
    monkeypatch.setenv("NOHALU_LLM_CB_COOLDOWN", "30")
    monkeypatch.setattr(llm, "api_key", lambda: "k")

    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        raise _http_error(503)

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            llm.generate("s", "u")
    assert calls["n"] == 2

    # Panggilan ketiga harus gagal cepat tanpa menyentuh jaringan
    with pytest.raises(RuntimeError) as exc:
        llm.generate("s", "u")
    assert "circuit open" in str(exc.value)
    assert calls["n"] == 2, "provider tidak boleh dihubungi saat pemutus terbuka"
    assert engine._llm_error_code(exc.value, "agent") == "agent:circuit_open"
    llm.reset_circuit()


def test_circuit_resets_after_success(monkeypatch):
    llm.reset_circuit()
    monkeypatch.setenv("NOHALU_LLM_ATTEMPTS", "1")
    monkeypatch.setenv("NOHALU_LLM_BACKOFF", "0")
    monkeypatch.setenv("NOHALU_LLM_CB_THRESHOLD", "5")
    monkeypatch.setattr(llm, "api_key", lambda: "k")
    state = {"n": 0}

    def fake_urlopen(req, timeout=None):
        state["n"] += 1
        if state["n"] == 1:
            raise _http_error(503)
        return type("R", (), {"read": lambda self: json.dumps(
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
             "model": "mercury-3", "usage": {}}).encode(),
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: False})()

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError):
        llm.generate("s", "u")
    assert llm.generate("s", "u")["text"] == "ok"
    assert llm._CB_STATE["failures"] == 0
    llm.reset_circuit()


# ---------------------------------------------------------------------------
# 7. Jawaban terpotong di jalur deterministik (rephrase)
# ---------------------------------------------------------------------------
def test_rephrase_rejects_truncated_text(monkeypatch):
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(engine.llm, "generate", lambda *a, **k: {
        "text": "Tunjangan transport Anda adalah Rp750.000 dan dapat",
        "truncated": True, "finish_reason": "length", "ms": 3,
        "usage": {}, "model": "mercury-3"})
    r = engine.answer_question(conn, user["id"], "Berapa tunjangan transport bulanan?")
    assert r["llm_error"] == "rephrase:truncated_text", r
    assert "750.000" in r["answer"], r["answer"]
    # Kalimat terpotong tidak boleh menjadi jawaban final
    assert "dan dapat" not in r["answer"], r["answer"]


# ---------------------------------------------------------------------------
# 7. 4xx non-retryable tidak boleh memarkir LLM untuk semua user
# ---------------------------------------------------------------------------
def test_client_error_does_not_open_circuit(monkeypatch):
    """Empat 400 content_filter tidak boleh mematikan LLM global.

    Terbukti sebelum perbaikan: 4x HTTP 400 berurutan -> `_cb_is_open()` True,
    sehingga semua user lain menerima fallback deterministik selama cooldown
    padahal provider sehat. 400/401/413 adalah sifat SATU request, bukan
    kesehatan provider.
    """
    llm.reset_circuit()
    monkeypatch.setenv("NOHALU_LLM_ATTEMPTS", "1")
    monkeypatch.setenv("NOHALU_LLM_BACKOFF", "0")
    monkeypatch.setenv("NOHALU_LLM_CB_THRESHOLD", "2")
    monkeypatch.setattr(llm, "api_key", lambda: "k")
    monkeypatch.setattr(llm.urllib.request, "urlopen",
                        lambda req, timeout=None: (_ for _ in ()).throw(_http_error(400)))
    for _ in range(5):
        with pytest.raises(RuntimeError):
            llm.generate("s", "u")
    assert not llm._cb_is_open(), "400 tidak boleh membuka circuit breaker"


def test_server_error_still_opens_circuit(monkeypatch):
    """Kebalikannya: 503 tetap dihitung sebagai provider tidak sehat."""
    llm.reset_circuit()
    monkeypatch.setenv("NOHALU_LLM_ATTEMPTS", "1")
    monkeypatch.setenv("NOHALU_LLM_BACKOFF", "0")
    monkeypatch.setenv("NOHALU_LLM_CB_THRESHOLD", "2")
    monkeypatch.setattr(llm, "api_key", lambda: "k")
    monkeypatch.setattr(llm.urllib.request, "urlopen",
                        lambda req, timeout=None: (_ for _ in ()).throw(_http_error(503)))
    for _ in range(2):
        with pytest.raises(RuntimeError):
            llm.generate("s", "u")
    assert llm._cb_is_open()


# ---------------------------------------------------------------------------
# 8. Anggaran konteks per giliran
# ---------------------------------------------------------------------------
def test_agent_stops_adding_tools_when_context_budget_exhausted(monkeypatch):
    """Hasil tool berikutnya ditolak saat anggaran konteks habis.

    Loop agent mengirim ulang seluruh riwayat tiap langkah, jadi tanpa pagar
    satu giliran bisa menembus jendela konteks model. Model harus tetap
    menjawab dari data yang sudah ada.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    calls = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        calls["n"] += 1
        if calls["n"] <= 3:
            return {"text": "", "finish_reason": "tool_calls", "ms": 2, "usage": {},
                    "model": "mercury-3", "truncated": False,
                    "tool_calls": [{
                        "id": f"c{calls['n']}", "name": "search_policies",
                        "arguments": {"query": f"cuti {calls['n']}"},
                        "raw_arguments": "{}", "arguments_error": None}]}
        return {"text": "Cuti tahunan 12 hari.", "tool_calls": [],
                "finish_reason": "stop", "ms": 2, "usage": {},
                "model": "mercury-3", "truncated": False}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    monkeypatch.setattr(agent, "MAX_CONTEXT_CHARS", 1)  # paksa habis segera
    r = agent.run(conn, user, "Berapa cuti tahunan?", None)
    blocked = [t for t in r["agent_trace"] if t.get("error") == "tool_error"]
    assert blocked, r["agent_trace"]
    assert r["answer"], "model tetap harus menjawab dari data yang ada"


# ---------------------------------------------------------------------------
# 9. Anggaran waktu per giliran
# ---------------------------------------------------------------------------
def test_turn_time_budget_stops_further_steps(monkeypatch):
    """Giliran yang melewati anggaran waktu tidak boleh memanggil model lagi."""
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    calls = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        calls["n"] += 1
        return {"text": "", "finish_reason": "tool_calls", "ms": 2, "usage": {},
                "model": "mercury-3", "truncated": False,
                "tool_calls": [{"id": f"c{calls['n']}", "name": "get_leave_balance",
                                "arguments": {}, "raw_arguments": "{}",
                                "arguments_error": None}]}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    monkeypatch.setattr(agent, "MAX_TURN_SECONDS", 0.0)
    r = agent.run(conn, user, "Berapa sisa cuti saya?", None)
    assert calls["n"] <= 1, calls
    assert r["llm_error"] == "agent:time_budget", r
    assert r["answer"], "user tetap harus menerima jawaban dari data yang ada"


def test_retry_after_header_is_honoured(monkeypatch):
    """429 dengan `Retry-After` harus menunggu sesuai header, bukan backoff buta."""
    import io
    import urllib.error
    llm.reset_circuit()
    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setenv("NOHALU_LLM_ATTEMPTS", "2")
    monkeypatch.setenv("NOHALU_LLM_BACKOFF", "5")
    monkeypatch.setattr(llm, "api_key", lambda: "k")
    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))

    def fake_urlopen(req, timeout=None):
        h = {"Retry-After": "0.4"}
        raise urllib.error.HTTPError("https://p/v1", 429, "e", h,
                                     io.BytesIO(b'{"error":"rate"}'))

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError):
        llm.generate("s", "u")
    assert slept, "harus ada jeda sebelum percobaan kedua"
    assert abs(slept[0] - 0.4) < 0.01, slept


# ---------------------------------------------------------------------------
# 10. Semua jalur final menghormati `truncated`
# ---------------------------------------------------------------------------
def test_step_limit_summary_marks_truncated_answer(monkeypatch):
    """Ringkasan step-limit tidak boleh mengirim kalimat yang berhenti di tengah."""
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    calls = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        calls["n"] += 1
        if tools:
            return {"text": "", "finish_reason": "tool_calls", "ms": 1, "usage": {},
                    "model": "mercury-3", "truncated": False,
                    "tool_calls": [{"id": f"c{calls['n']}",
                                    "name": "get_leave_balance", "arguments": {},
                                    "raw_arguments": "{}", "arguments_error": None}]}
        # Ringkasan final terpotong dan pelanjutan mengembalikan teks identik
        # (perilaku provider yang terukur) -> tidak boleh jadi kalimat dobel.
        return {"text": "Cuti tahunan Anda adalah 12 hari dan dapat",
                "finish_reason": "length", "ms": 1, "usage": {},
                "model": "mercury-3", "truncated": True}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Berapa sisa cuti saya?", None)
    assert r["llm_error"] == "agent:step_limit"
    assert "terpotong" in r["answer"].lower() or "belum lengkap" in r["answer"].lower(), r["answer"]
    assert r["answer"].count("12 hari dan dapat") == 1, r["answer"]


# ---------------------------------------------------------------------------
# 11. Cache panggilan identik tidak boleh mengubah hasil gagal jadi "sukses"
# ---------------------------------------------------------------------------
def test_cache_hit_preserves_failure_status(monkeypatch):
    """Panggilan identik yang gagal harus tetap tercatat gagal saat cache hit."""
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    calls = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        calls["n"] += 1
        if calls["n"] <= 2:
            return {"text": "", "finish_reason": "tool_calls", "ms": 1, "usage": {},
                    "model": "mercury-3", "truncated": False,
                    "tool_calls": [{"id": f"c{calls['n']}",
                                    "name": "prepare_leave_request",
                                    "arguments": {"start_date": "2000-01-01",
                                                  "end_date": "2000-01-03"},
                                    "raw_arguments": "{}", "arguments_error": None}]}
        return {"text": "Tanggal itu sudah lewat.", "tool_calls": [],
                "finish_reason": "stop", "ms": 1, "usage": {},
                "model": "mercury-3", "truncated": False}

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agent.llm, "chat", fake_chat)
    r = agent.run(conn, user, "Tolong buatkan draft cuti 1-3 Januari 2000", None)
    hits = [t for t in r["agent_trace"] if t.get("cached")]
    assert hits, r["agent_trace"]
    assert all(t["ok"] is False for t in hits), hits
