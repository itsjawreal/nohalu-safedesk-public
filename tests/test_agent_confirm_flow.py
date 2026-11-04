"""Alur konfirmasi lewat HTTP penuh: chat -> prepared -> confirm -> chat.

Kelas bug yang ditutup di sini (bug #28 di produksi): pesan user yang memuat
kata "setujui" diloloskan `_looks_like_approval`, lalu model memanggil
`prepare_*` DAN `submit_prepared` pada giliran yang sama. Pengajuan benar-benar
diputuskan padahal user belum pernah melihat ringkasan atau kartu konfirmasi.

Seluruh tes lama berjalan di level unit (`agent.run` dengan koneksi SQLite
buatan). Tidak ada satu pun yang menempuh jalur HTTP `POST /api/chat` ->
baris `chat_messages` -> `POST /api/agent/confirm`, sehingga regresi pada
penyimpanan `prepared`, penandaan `prepared_done`, atau gerbang satu-giliran
tidak tertangkap. Tes ini menutup celah itu.
"""
import datetime as dt
import json
import os
import re
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

_TMP = tempfile.mkdtemp()
os.environ["NOHALU_DB_PATH"] = os.path.join(_TMP, "confirm_flow.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import agent as agentmod  # noqa: E402
import app as appmod  # noqa: E402

client = TestClient(appmod.app)


def _h(uid):
    r = client.post("/api/auth/login", json={"user_id": uid})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


# Tanggal sengaja jauh di depan dan unik per tes. Seluruh modul tes berbagi
# satu DB aplikasi (modul pertama yang meng-import `app` menentukan path), jadi
# cuti bertanggal `today+N` di sini bisa beririsan dengan tanggal tetap yang
# dipakai tes lain — dan `POST /api/requests` lalu 409 leave_overlap.
_YEAR = 2030


def _future(days):
    """Tanggal ISO jauh di masa depan — cuti bertanggal lampau ditolak portal."""
    return (dt.date(_YEAR, 1, 1) + dt.timedelta(days=days)).isoformat()


# Semua sesi modul ini memakai awalan ini supaya pembersihan tidak menyentuh
# sesi milik tes lain yang memakai DB yang sama.
SID_PREFIX = "sess-flow-"


@pytest.fixture(autouse=True)
def _isolate():
    """Bersihkan jejak modul ini dari DB bersama sebelum & sesudah tiap tes.

    Yang dihapus hanya baris bertanda modul ini: pengajuan bertahun `_YEAR`,
    draft & pesan pada sesi berawalan `SID_PREFIX`. Menghapus lebih luas akan
    merusak tes lain yang berbagi DB aplikasi.
    """
    def wipe():
        appmod.CONN.execute(
            "DELETE FROM requests WHERE start_date LIKE ? OR end_date LIKE ?",
            (f"{_YEAR}-%", f"{_YEAR}-%"))
        appmod.CONN.execute(
            "DELETE FROM agent_prepared WHERE session_id LIKE ?",
            (SID_PREFIX + "%",))
        appmod.CONN.execute(
            "DELETE FROM chat_messages WHERE session_id LIKE ?",
            (SID_PREFIX + "%",))
        appmod.CONN.execute(
            "DELETE FROM chat_sessions WHERE id LIKE ?", (SID_PREFIX + "%",))
        appmod.CONN.commit()
    wipe()
    yield
    wipe()


def _tool_msg(name, args, cid="c1"):
    return {"id": cid, "name": name, "arguments": args,
            "raw_arguments": json.dumps(args)}


def _reply(text="", calls=None, ms=1):
    return {"text": text, "tool_calls": calls or [], "finish_reason":
            "tool_calls" if calls else "stop", "ms": ms, "usage": {},
            "model": "mercury-3"}


def _pending_requests(uid):
    row = appmod.CONN.execute(
        "SELECT COUNT(*) AS n FROM requests WHERE user_id=?", (uid,)).fetchone()
    return row["n"]


def _token_from_tool_messages(messages):
    """Ambil confirm_token dari hasil tool `prepare_*` (jalur model nyata)."""
    for m in reversed(messages):
        if m.get("role") != "tool":
            continue
        try:
            data = json.loads(m.get("content") or "{}")
        except ValueError:
            continue
        if data.get("confirm_token"):
            return data["confirm_token"]
    return None


def _token_from_system_prompt(messages):
    """Token draft yang dipulihkan `ToolContext` muncul di system prompt."""
    for m in messages:
        if m.get("role") != "system":
            continue
        hit = re.search(r"confirm_token:\s*(cfm_[A-Za-z0-9_\-]+)", m.get("content") or "")
        if hit:
            return hit.group(1)
    return None


# ---------------------------------------------------------------------------
# 1. Satu giliran tidak boleh membuat draft lalu langsung menjalankannya
# ---------------------------------------------------------------------------
def test_chat_cannot_prepare_and_submit_in_one_turn(monkeypatch):
    """`POST /api/chat` tidak boleh mengubah data walau model memaksa.

    Model dibuat meniru bug #28: panggil `prepare_leave_request`, lalu langsung
    `submit_prepared` pada giliran yang sama. Gerbang di loop agent harus
    menolak submit; DB harus tetap tanpa pengajuan.
    """
    uid = "u_budi"
    start, end = _future(20), _future(22)
    before = _pending_requests(uid)
    state = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return _reply(calls=[_tool_msg(
                "prepare_leave_request", {"start_date": start, "end_date": end})])
        if state["n"] == 2:
            token = _token_from_tool_messages(messages)
            assert token, "token draft tidak sampai ke model"
            # Inilah yang dulu terjadi: submit di giliran yang sama.
            return _reply(calls=[_tool_msg(
                "submit_prepared", {"confirm_token": token}, cid="c2")])
        return _reply(text=(
            f"Draft cuti {start} s.d. {end} siap. Tekan tombol "
            "\"Setujui & kirim\" untuk menjalankannya."))

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agentmod.llm, "chat", fake_chat)

    r = client.post("/api/chat", headers=_h(uid), json={
        "question": f"Setujui pengajuan cuti saya {start} s.d. {end}",
        "session_id": SID_PREFIX + "one-turn"})
    assert r.status_code == 200, r.text
    body = r.json()

    # Diperiksa lebih dulu: inilah kerusakan yang dilihat user — pengajuan
    # lahir tanpa pernah melihat ringkasan atau menekan tombol apa pun.
    assert _pending_requests(uid) == before, "pengajuan dibuat tanpa konfirmasi user"
    assert body["prepared"], "kartu konfirmasi tidak dikirim ke UI"
    assert body["prepared"]["action"] == "create_request"
    assert body["prepared"]["token"].startswith("cfm_")

    submit = [t for t in body["agent_trace"] if t["tool"] == "submit_prepared"]
    assert submit, body["agent_trace"]
    assert submit[0]["ok"] is False, "submit satu-giliran lolos gerbang"
    assert submit[0]["error"] == "tool_error"

    row = appmod.CONN.execute(
        "SELECT used FROM agent_prepared WHERE token=?",
        (body["prepared"]["token"],)).fetchone()
    assert row is not None and row["used"] == 0, "token ikut terpakai padahal ditolak"


# ---------------------------------------------------------------------------
# 2. Konfirmasi lewat pesan terpisah (jalur LLM) benar-benar menjalankan
# ---------------------------------------------------------------------------
def test_next_turn_confirmation_executes_prepared(monkeypatch):
    """`chat` -> `chat("ya, kirim")` -> pengajuan benar-benar tersimpan."""
    uid = "u_budi"
    start, end = _future(30), _future(31)
    before = _pending_requests(uid)
    state = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return _reply(calls=[_tool_msg(
                "prepare_leave_request", {"start_date": start, "end_date": end})])
        if state["n"] == 2:
            return _reply(text=(
                f"Draft cuti {start} s.d. {end} siap. Balas \"ya\" untuk mengirim."))
        # Giliran kedua (sesi sama): token dipulihkan dari system prompt.
        token = _token_from_system_prompt(messages)
        assert token, "draft dari giliran sebelumnya tidak dipulihkan"
        return _reply(calls=[_tool_msg(
            "submit_prepared", {"confirm_token": token})])

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agentmod.llm, "chat", fake_chat)

    first = client.post("/api/chat", headers=_h(uid), json={
        "question": f"Ajukan cuti {start} s.d. {end}",
        "session_id": SID_PREFIX + "next-turn"}).json()
    sid = first["session_id"]
    assert first["prepared"] and first["prepared"]["token"]
    assert _pending_requests(uid) == before

    state2 = {"n": 0}

    def fake_confirm(messages, tools=None, **kw):
        state2["n"] += 1
        token = _token_from_system_prompt(messages)
        assert token
        if state2["n"] == 1:
            return _reply(calls=[_tool_msg(
                "submit_prepared", {"confirm_token": token})])
        return _reply(text=f"Pengajuan cuti {start} s.d. {end} sudah tersimpan.")

    monkeypatch.setattr(agentmod.llm, "chat", fake_confirm)
    second = client.post("/api/chat", headers=_h(uid),
                         json={"question": "ya, kirim", "session_id": sid})
    assert second.status_code == 200, second.text
    out = second.json()
    assert [t["tool"] for t in out["agent_trace"]] == ["submit_prepared"], out
    assert out["agent_trace"][0]["ok"] is True, out["agent_trace"]
    assert _pending_requests(uid) == before + 1, "pengajuan tidak tersimpan"
    assert out["prepared"] is None, "kartu masih aktif padahal sudah dijalankan"


# ---------------------------------------------------------------------------
# 3. Tombol "Setujui & kirim" (tanpa LLM) menempuh jalur yang sama
# ---------------------------------------------------------------------------
def test_confirm_endpoint_after_chat_card(monkeypatch):
    """Kartu dari `/api/chat` bisa dijalankan lewat `/api/agent/confirm`.

    Yang dijaga: token yang tersimpan di `chat_messages.prepared` adalah token
    yang sama dengan yang diterima endpoint, `prepared_done` ditandai, dan
    pesan hasilnya masuk riwayat lengkap dengan saran lanjutan.
    """
    uid = "u_budi"
    start, end = _future(40), _future(41)
    before = _pending_requests(uid)

    state = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return _reply(calls=[_tool_msg(
                "prepare_leave_request", {"start_date": start, "end_date": end})])
        return _reply(text=f"Draft cuti {start} s.d. {end} siap dikirim.")

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agentmod.llm, "chat", fake_chat)

    h = _h(uid)
    first = client.post("/api/chat", headers=h, json={
        "question": f"Ajukan cuti {start} s.d. {end}",
        "session_id": SID_PREFIX + "confirm"}).json()
    assert first["prepared"], first
    sid, token = first["session_id"], first["prepared"]["token"]

    # Kartu tersimpan di riwayat, belum dijalankan.
    hist = client.get("/api/chat/" + sid, headers=h).json()
    card = [m for m in hist["messages"] if m["prepared"]]
    assert card and card[0]["prepared"]["token"] == token
    assert card[0]["prepared_done"] is False
    assert _pending_requests(uid) == before

    r = client.post("/api/agent/confirm", headers=h,
                    json={"token": token, "session_id": sid})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True and r.json()["request_id"]
    assert _pending_requests(uid) == before + 1

    hist = client.get("/api/chat/" + sid, headers=h).json()
    card = [m for m in hist["messages"] if m["prepared"]][0]
    assert card["prepared_done"] is True, "kartu selesai masih tampak aktif"
    last = hist["messages"][-1]
    assert last["role"] == "assistant" and "tersimpan" in last["text"]
    assert last["suggestions"], "saran lanjutan hilang setelah konfirmasi"

    # Sekali pakai: klik kedua tidak menggandakan pengajuan.
    again = client.post("/api/agent/confirm", headers=h,
                        json={"token": token, "session_id": sid})
    assert again.status_code == 400
    assert _pending_requests(uid) == before + 1


# ---------------------------------------------------------------------------
# 4. Penolakan eksplisit menggugurkan kartu tanpa memanggil LLM
# ---------------------------------------------------------------------------
def test_rejection_turn_discards_card_and_keeps_data(monkeypatch):
    """`chat("ga jadi")` setelah kartu: token gugur, pengajuan tidak dibuat."""
    uid = "u_budi"
    start, end = _future(50), _future(51)
    before = _pending_requests(uid)

    state = {"n": 0}

    def fake_chat(messages, tools=None, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return _reply(calls=[_tool_msg(
                "prepare_leave_request", {"start_date": start, "end_date": end})])
        return _reply(text=f"Draft cuti {start} s.d. {end} siap dikirim.")

    monkeypatch.setenv("NOHALU_LLM_ENABLED", "1")
    monkeypatch.setattr(agentmod.llm, "chat", fake_chat)

    h = _h(uid)
    first = client.post("/api/chat", headers=h, json={
        "question": f"Ajukan cuti {start} s.d. {end}",
        "session_id": SID_PREFIX + "confirm"}).json()
    assert first["prepared"], first
    sid, token = first["session_id"], first["prepared"]["token"]

    def boom(*a, **kw):
        raise AssertionError("penolakan draft tidak boleh memanggil LLM")

    monkeypatch.setattr(agentmod.llm, "chat", boom)
    second = client.post("/api/chat", headers=h,
                         json={"question": "ga jadi deh", "session_id": sid})
    assert second.status_code == 200, second.text
    out = second.json()
    assert out["prepared"] is None, "kartu masih dikirim setelah ditolak"
    assert _pending_requests(uid) == before

    row = appmod.CONN.execute(
        "SELECT used FROM agent_prepared WHERE token=?", (token,)).fetchone()
    assert row["used"] == 1, "token tidak digugurkan"

    # Tombol lama yang masih terkirim ke klien tidak boleh berfungsi.
    stale = client.post("/api/agent/confirm", headers=h,
                        json={"token": token, "session_id": sid})
    assert stale.status_code == 400
    assert _pending_requests(uid) == before
