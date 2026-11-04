"""HOTFIX H — isolasi sesi chat & riwayat.

Bug produksi (P0-1): `POST /api/chat` menerima `session_id` dari klien tanpa
memeriksa pemiliknya. User mana pun yang tahu/menebak id sesi bisa menyisipkan
pesan ke riwayat orang lain dan mencemari audit.

Juga meregresi route ordering: `/api/chat/sessions` dulu tertangkap
`/api/chat/{session_id}` sehingga selalu 404 session_not_found.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

_TMP = tempfile.mkdtemp()
os.environ["NOHALU_DB_PATH"] = os.path.join(_TMP, "chat_owner.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import agent as agentmod  # noqa: E402
import app as appmod  # noqa: E402

client = TestClient(appmod.app)


def _h(uid):
    r = client.post("/api/auth/login", json={"user_id": uid})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


H_BUDI = _h("u_budi")
H_RINA = _h("u_rina")
H_ANDI = _h("u_andi")


def _chat(headers, question, session_id=None):
    body = {"question": question}
    if session_id:
        body["session_id"] = session_id
    return client.post("/api/chat", headers=headers, json=body)


def test_sessions_list_route_not_shadowed():
    r = client.get("/api/chat/sessions", headers=H_BUDI)
    assert r.status_code == 200, r.text
    assert "sessions" in r.json()
    print("  ok sessions_list_route_not_shadowed")


def test_cannot_write_to_other_users_session():
    r = _chat(H_BUDI, "Berapa cuti tahunan?")
    assert r.status_code == 200, r.text
    sid = r.json()["session_id"]

    r2 = _chat(H_RINA, "sisip pesan", session_id=sid)
    assert r2.status_code == 403, r2.text
    assert r2.json()["detail"] == "session_forbidden"

    hist = client.get("/api/chat/" + sid, headers=H_BUDI).json()
    texts = [m["text"] for m in hist["messages"]]
    assert "sisip pesan" not in texts
    assert len([m for m in hist["messages"] if m["role"] == "user"]) == 1
    print("  ok cannot_write_to_other_users_session")


def test_sessions_list_only_own_and_has_title():
    r = _chat(H_ANDI, "Bagaimana alur reimbursement?")
    sid = r.json()["session_id"]
    data = client.get("/api/chat/sessions", headers=H_ANDI).json()
    mine = [s for s in data["sessions"] if s["id"] == sid]
    assert mine, data
    assert mine[0]["title"].startswith("Bagaimana alur")
    assert mine[0]["messages"] >= 2
    # sesi orang lain tidak boleh muncul
    ids = {s["id"] for s in data["sessions"]}
    assert all(not s.startswith("sess-budi") for s in ids)
    print("  ok sessions_list_only_own_and_has_title")


def test_rename_and_delete_owner_only():
    sid = _chat(H_BUDI, "Apa visi perusahaan?").json()["session_id"]
    r = client.patch("/api/chat/sessions/" + sid, headers=H_BUDI,
                     json={"title": "Visi perusahaan"})
    assert r.status_code == 200 and r.json()["title"] == "Visi perusahaan"

    r = client.patch("/api/chat/sessions/" + sid, headers=H_RINA,
                     json={"title": "dibajak"})
    assert r.status_code == 404

    r = client.delete("/api/chat/sessions/" + sid, headers=H_RINA)
    assert r.status_code == 404
    assert client.get("/api/chat/" + sid, headers=H_BUDI).status_code == 200

    r = client.delete("/api/chat/sessions/" + sid, headers=H_BUDI)
    assert r.status_code == 200
    assert client.get("/api/chat/" + sid, headers=H_BUDI).status_code == 404
    print("  ok rename_and_delete_owner_only")


def test_rename_validates_title():
    sid = _chat(H_BUDI, "Apa kebijakan WFH?").json()["session_id"]
    r = client.patch("/api/chat/sessions/" + sid, headers=H_BUDI, json={"title": "  "})
    assert r.status_code == 422
    r = client.patch("/api/chat/sessions/" + sid, headers=H_BUDI,
                     json={"title": "x" * 200})
    assert r.status_code == 422
    print("  ok rename_validates_title")


def test_question_length_validated():
    r = _chat(H_BUDI, "x" * 9000)
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "question_too_long"
    print("  ok question_length_validated")


def test_llm_failure_marks_session_interrupted(monkeypatch):
    """Kalau jawaban gagal total, sesi tetap punya pesan penjelas (bukan kosong)."""
    sid = _chat(H_BUDI, "pemicu gagal").json()["session_id"]

    def boom(*a, **k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(agentmod, "run", boom)
    r = _chat(H_BUDI, "pertanyaan kedua", session_id=sid)
    assert r.status_code == 500
    hist = client.get("/api/chat/" + sid, headers=H_BUDI).json()
    last = hist["messages"][-1]
    assert last["role"] == "assistant"
    assert last["interrupted"] is True
    assert "dihentikan" in last["text"].lower()
    print("  ok llm_failure_marks_session_interrupted")


def test_agent_confirm_endpoint_runs_prepared_draft():
    """Tombol "Setujui & kirim" menjalankan draft tanpa pesan user baru."""
    import db as dbmod
    import agent_tools as tools
    conn = appmod.CONN
    user = appmod.get_user(conn, "u_maya")
    sid = "sess-confirm-1"
    ctx = tools.ToolContext(conn, user, sid)
    prep = tools.tool_prepare_leave_request(
        ctx, start_date="2026-11-20", end_date="2026-11-22", title="Keluarga")
    token = prep["confirm_token"]

    h = _h("u_maya")
    r = client.post("/api/agent/confirm", headers=h,
                    json={"token": token, "session_id": sid})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True and body["request_id"]

    # token sekali pakai
    again = client.post("/api/agent/confirm", headers=h,
                        json={"token": token, "session_id": sid})
    assert again.status_code == 400

    # hasil tercatat di percakapan
    hist = client.get("/api/chat/" + sid, headers=h).json()
    assert any("Pengajuan" in m["text"] for m in hist["messages"])


def test_agent_confirm_rejects_foreign_token():
    import agent_tools as tools
    conn = appmod.CONN
    owner = appmod.get_user(conn, "u_maya")
    ctx = tools.ToolContext(conn, owner, "sess-confirm-2")
    prep = tools.tool_prepare_leave_request(
        ctx, start_date="2026-11-27", end_date="2026-11-28", title="X")
    r = client.post("/api/agent/confirm", headers=H_BUDI,
                    json={"token": prep["confirm_token"], "session_id": "sess-confirm-2"})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Riwayat yang dibuka ulang: kartu draft, saran lanjutan, metadata
# ---------------------------------------------------------------------------
def test_history_keeps_suggestions_and_marks_executed_draft():
    """Riwayat harus lengkap setelah dibuka ulang.

    Tiga keluhan nyata yang diregresi di sini:
      a) jawaban terakhir tidak punya "Pertanyaan lanjutan" (saran dulu hanya
         hidup di respons HTTP, tidak pernah disimpan),
      b) kartu draft yang SUDAH dijalankan muncul lagi dengan tombol aktif,
      c) pesan hasil konfirmasi tidak membawa saran.
    """
    import agent_tools as tools
    conn = appmod.CONN
    sid = "sess-hist-meta"
    ctx = tools.ToolContext(conn, appmod.get_user(conn, "u_maya"), sid)
    prep = tools.tool_prepare_leave_request(
        ctx, start_date="2026-12-14", end_date="2026-12-15", title="Keluarga")
    token = prep["confirm_token"]

    # Pesan assistant yang membawa kartu draft (seperti ditulis /api/chat).
    import json as _json
    conn.execute(
        "INSERT INTO chat_messages (session_id, role, text, citations, created_at,"
        " prepared, suggestions) VALUES (?,?,?,?,?,?,?)",
        (sid, "assistant", "Draft cuti siap.", "[]", "2026-10-31T00:00:00Z",
         _json.dumps({"token": token, "action": "create_request",
                      "summary": prep["summary"]}), None))
    conn.commit()

    h = _h("u_maya")
    r = client.post("/api/agent/confirm", headers=h,
                    json={"token": token, "session_id": sid})
    assert r.status_code == 200, r.text

    hist = client.get("/api/chat/" + sid, headers=h).json()
    card = [m for m in hist["messages"] if m["prepared"]]
    assert card, "kartu draft hilang dari riwayat"
    assert card[0]["prepared_done"] is True, "kartu draft yang dijalankan masih aktif"

    last = hist["messages"][-1]
    assert last["role"] == "assistant"
    assert last["suggestions"], "jawaban terakhir tanpa saran lanjutan"

    # Dibuka dua kali: hasilnya sama (tidak menumpuk / berubah).
    again = client.get("/api/chat/" + sid, headers=h).json()
    assert again["messages"][-1]["suggestions"] == last["suggestions"]


def test_history_backfills_suggestions_for_old_rows():
    """Baris lama (kolom `suggestions` NULL) tetap mendapat saran saat dibaca."""
    import json as _json
    conn = appmod.CONN
    sid = "sess-hist-old"
    conn.execute(
        "INSERT INTO chat_sessions (id, user_id, created_at, status)"
        " VALUES (?,?,?,?)", (sid, "u_maya", "2026-10-01T00:00:00Z", "active"))
    conn.execute(
        "INSERT INTO chat_messages (session_id, role, text, citations, created_at, trace)"
        " VALUES (?,?,?,?,?,?)",
        (sid, "assistant", "Sisa cuti Anda 12 hari.", "[]",
         "2026-10-01T00:00:00Z",
         _json.dumps([{"tool": "get_leave_balance", "args": {}, "ok": True}])))
    conn.commit()

    h = _h("u_maya")
    hist = client.get("/api/chat/" + sid, headers=h).json()
    assert hist["messages"][-1]["suggestions"], "saran tidak diisi untuk baris lama"


def test_backfill_marks_old_executed_draft_as_done():
    """Kartu draft lama yang tokennya sudah dipakai tidak boleh tampil aktif."""
    import json as _json
    import db as dbmod
    conn = appmod.CONN
    sid = "sess-hist-backfill"
    conn.execute(
        "INSERT INTO chat_sessions (id, user_id, created_at, status)"
        " VALUES (?,?,?,?)", (sid, "u_maya", "2026-10-01T00:00:00Z", "active"))
    conn.execute(
        "INSERT INTO agent_prepared (token,user_id,session_id,action,payload,summary,"
        "created_at,expires_at,used) VALUES (?,?,?,?,?,?,?,?,1)",
        ("cfm_old_done", "u_maya", sid, "create_request", "{}", "Draft lama",
         "2026-10-01T00:00:00", "2026-10-02T00:00:00"))
    conn.execute(
        "INSERT INTO chat_messages (session_id, role, text, citations, created_at,"
        " prepared) VALUES (?,?,?,?,?,?)",
        (sid, "assistant", "Draft siap.", "[]", "2026-10-01T00:00:00Z",
         _json.dumps({"token": "cfm_old_done", "action": "create_request",
                      "summary": "Draft lama"})))
    conn.execute(
        "INSERT INTO chat_messages (session_id, role, text, citations, created_at)"
        " VALUES (?,?,?,?,?)",
        (sid, "assistant", "Pengajuan #99 tersimpan dengan status pending.",
         "[]", "2026-10-01T00:01:00Z"))
    conn.commit()

    out = dbmod.backfill_chat_meta(conn)
    assert out["prepared_done"] >= 1
    row = conn.execute(
        "SELECT prepared_done FROM chat_messages WHERE session_id=? AND prepared IS NOT NULL",
        (sid,)).fetchone()
    assert row["prepared_done"] == 1


# --------------------------------------------------------------------------
# T1 — backfill judul sesi lama
# --------------------------------------------------------------------------
def test_backfill_fills_missing_session_title():
    """Sesi lama tanpa judul mendapat judul dari pesan user PERTAMA.

    Di prod, 156 dari 181 sesi (86%) berjudul kosong karena dibuat sebelum
    `_touch_session(title_hint)` ada. Tanpa backfill, riwayat hanya menampilkan
    "Percakapan baru" dan tidak bisa dibedakan satu sama lain.
    """
    import db as dbmod
    conn = appmod.CONN
    sid = "sess-title-backfill"
    conn.execute(
        "INSERT INTO chat_sessions (id, user_id, created_at, status)"
        " VALUES (?,?,?,?)", (sid, "u_maya", "2026-10-01T00:00:00Z", "active"))
    conn.execute(
        "INSERT INTO chat_messages (session_id, role, text, citations, created_at)"
        " VALUES (?,?,?,?,?)",
        (sid, "user", "Bagaimana cara klaim reimbursement?", "[]",
         "2026-10-01T00:00:00Z"))
    conn.execute(
        "INSERT INTO chat_messages (session_id, role, text, citations, created_at)"
        " VALUES (?,?,?,?,?)",
        (sid, "assistant", "Kirim bukti ke Finance.", "[]", "2026-10-01T00:01:00Z"))
    conn.commit()

    out = dbmod.backfill_chat_meta(conn)
    assert out["titles"] >= 1
    row = conn.execute("SELECT title FROM chat_sessions WHERE id=?", (sid,)).fetchone()
    assert row["title"] == "Bagaimana cara klaim reimbursement?"

    # Idempotent: judul yang sudah ada (termasuk hasil rename user) tidak ditimpa.
    conn.execute("UPDATE chat_sessions SET title='Judul pilihan user' WHERE id=?", (sid,))
    conn.commit()
    dbmod.backfill_chat_meta(conn)
    row = conn.execute("SELECT title FROM chat_sessions WHERE id=?", (sid,)).fetchone()
    assert row["title"] == "Judul pilihan user"


def test_backfill_title_truncated_to_120():
    """Judul dipotong 120 karakter, sama seperti `_touch_session(title_hint)`."""
    import db as dbmod
    conn = appmod.CONN
    sid = "sess-title-long"
    conn.execute(
        "INSERT INTO chat_sessions (id, user_id, created_at, status)"
        " VALUES (?,?,?,?)", (sid, "u_maya", "2026-10-01T00:00:00Z", "active"))
    conn.execute(
        "INSERT INTO chat_messages (session_id, role, text, citations, created_at)"
        " VALUES (?,?,?,?,?)",
        (sid, "user", "x" * 500, "[]", "2026-10-01T00:00:00Z"))
    conn.commit()
    dbmod.backfill_chat_meta(conn)
    row = conn.execute("SELECT title FROM chat_sessions WHERE id=?", (sid,)).fetchone()
    assert len(row["title"]) == 120


def test_backfill_skips_empty_session_without_messages():
    """Sesi tanpa pesan tidak dipaksa punya judul (tidak ada sumber teks)."""
    import db as dbmod
    conn = appmod.CONN
    sid = "sess-title-empty"
    conn.execute(
        "INSERT INTO chat_sessions (id, user_id, created_at, status)"
        " VALUES (?,?,?,?)", (sid, "u_maya", "2026-10-01T00:00:00Z", "active"))
    conn.commit()
    dbmod.backfill_chat_meta(conn)
    row = conn.execute("SELECT title FROM chat_sessions WHERE id=?", (sid,)).fetchone()
    assert not row["title"]


# --------------------------------------------------------------------------
# T3 — validasi session_id dari klien
# --------------------------------------------------------------------------
def test_rejects_path_traversal_session_id():
    r = _chat(H_BUDI, "tes", session_id="../../etc/passwd")
    assert r.status_code == 400, r.text
    assert r.json()["detail"]["code"] == "invalid_session_id"
    row = appmod.CONN.execute(
        "SELECT id FROM chat_sessions WHERE id LIKE '%/%'").fetchone()
    assert row is None, "session_id tidak valid tetap tersimpan"


def test_rejects_oversized_session_id():
    r = _chat(H_BUDI, "tes", session_id="X" * 5000)
    assert r.status_code == 400, r.text
    assert r.json()["detail"]["code"] == "invalid_session_id"
    assert appmod.CONN.execute(
        "SELECT id FROM chat_sessions WHERE length(id) > 64").fetchone() is None


def test_rejects_quoted_session_id():
    """Artefak lama di prod: id berbentuk "{'detail': 'Method Not Allowed'}"."""
    r = _chat(H_BUDI, "tes", session_id="{'detail': 'Method Not Allowed'}")
    assert r.status_code == 400, r.text
    assert r.json()["detail"]["code"] == "invalid_session_id"


def test_accepts_uuid_and_short_session_id():
    """Format sah tetap lolos: UUID v4 (dipakai UI) dan id pendek (dipakai test)."""
    for sid in ("3f2b8c1e-9a4d-4e7f-8b21-0c5d6e7f8a90", "sess-ok-1"):
        r = _chat(H_BUDI, "tes", session_id=sid)
        assert r.status_code == 200, r.text
        assert r.json()["session_id"] == sid


def test_rejects_invalid_session_id_on_confirm():
    """Endpoint konfirmasi draft memakai aturan yang sama."""
    h = _h("u_maya")
    r = client.post("/api/agent/confirm", headers=h,
                    json={"token": "cfm_apa_saja", "session_id": "../../x"})
    assert r.status_code == 400, r.text
    assert r.json()["detail"]["code"] == "invalid_session_id"


def test_confirm_without_session_id_infers_from_token():
    """Kartu draft yang dijalankan tanpa `session_id` dari klien tetap tuntas.

    Bug produksi: UI mengirim `/api/agent/confirm` tanpa `session_id` ketika
    state sesi belum termuat. Akibatnya `_mark_prepared_done` tidak jalan dan
    pesan "Pengajuan #N tersimpan" tidak tercatat — draft yang SUDAH jadi
    pengajuan tetap tampil dengan tombol aktif saat riwayat dibuka ulang.
    Token draft sendiri sudah menyimpan `session_id` asal, jadi server harus
    memakainya sebagai sumber kebenaran.
    """
    import agent_tools as tools
    conn = appmod.CONN
    sid = "sess-confirm-nosid"
    h = _h("u_maya")
    ctx = tools.ToolContext(conn, appmod.get_user(conn, "u_maya"), sid)
    prep = tools.tool_prepare_leave_request(
        ctx, start_date="2026-12-21", end_date="2026-12-22", title="Natal")
    token = prep["confirm_token"]

    import json as _json
    conn.execute(
        "INSERT INTO chat_messages (session_id, role, text, citations, created_at,"
        " prepared) VALUES (?,?,?,?,?,?)",
        (sid, "assistant", "Draft cuti siap.", "[]", "2026-10-31T00:00:00Z",
         _json.dumps({"token": token, "action": "create_request",
                      "summary": prep["summary"]})))
    conn.commit()

    # Tanpa session_id — persis seperti UI saat state sesi kosong.
    r = client.post("/api/agent/confirm", headers=h, json={"token": token})
    assert r.status_code == 200, r.text

    row = conn.execute(
        "SELECT prepared_done FROM chat_messages"
        " WHERE session_id=? AND prepared IS NOT NULL", (sid,)).fetchone()
    assert row["prepared_done"] == 1, "kartu draft selesai masih dianggap aktif"

    hist = client.get("/api/chat/" + sid, headers=h).json()
    last = hist["messages"][-1]
    assert last["role"] == "assistant"
    assert "tersimpan" in last["text"], "pesan konfirmasi tidak tercatat di riwayat"
    assert last["suggestions"], "saran lanjutan tidak tersimpan"
