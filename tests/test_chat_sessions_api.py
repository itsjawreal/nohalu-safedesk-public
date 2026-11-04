"""Sprint 2.7 — kontrak API riwayat chat yang dipakai UI.

UI (chat.js) bergantung pada lima perilaku berikut:
1. `POST /api/chat` mengembalikan `session_id` yang bisa dikirim ulang sehingga
   pertanyaan berikutnya masuk ke percakapan yang sama (bukan sesi baru).
2. `GET /api/chat/sessions` hanya memuat sesi milik user, terbaru dulu, dengan
   judul + jumlah pesan; sesi kosong tidak ditampilkan.
3. `GET /api/chat/{id}` mengembalikan pesan berurutan (user, assistant) lengkap
   dengan sitasi untuk dirender ulang.
4. `PATCH` / `DELETE` hanya boleh oleh pemilik; admin boleh MEMBACA (audit) tapi
   tidak boleh menulis ke sesi orang lain.
5. Sesi yang dihapus tidak lagi muncul di daftar dan GET-nya 404.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "sessions_api.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402

client = TestClient(appmod.app)


def _h(uid):
    r = client.post("/api/auth/login", json={"user_id": uid})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


H_BUDI = _h("u_budi")
H_DEWI = _h("u_dewi")
H_ADMIN = _h("u_rina")


def _ask(headers, question, sid=None):
    body = {"question": question}
    if sid:
        body["session_id"] = sid
    r = client.post("/api/chat", headers=headers, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_session_id_roundtrip_keeps_one_conversation():
    first = _ask(H_BUDI, "Berapa cuti tahunan?")
    sid = first["session_id"]
    assert sid
    second = _ask(H_BUDI, "Bagaimana alur reimbursement?", sid)
    assert second["session_id"] == sid, "session_id harus dipakai ulang, bukan dibuat baru"

    hist = client.get("/api/chat/" + sid, headers=H_BUDI).json()
    roles = [m["role"] for m in hist["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"], roles
    assert hist["messages"][0]["text"] == "Berapa cuti tahunan?"
    assert hist["messages"][2]["text"] == "Bagaimana alur reimbursement?"
    print("  ok session_id_roundtrip_keeps_one_conversation")


def test_sessions_list_shape_and_ordering():
    sid = _ask(H_DEWI, "Kapan gaji dibayarkan?").get("session_id")
    data = client.get("/api/chat/sessions", headers=H_DEWI).json()
    assert isinstance(data.get("sessions"), list) and data["count"] == len(data["sessions"])
    mine = [s for s in data["sessions"] if s["id"] == sid]
    assert mine, data
    row = mine[0]
    for key in ("id", "title", "created_at", "updated_at", "status", "messages"):
        assert key in row, (key, row)
    assert row["messages"] >= 2
    # urut terbaru dulu
    stamps = [s["updated_at"] for s in data["sessions"]]
    assert stamps == sorted(stamps, reverse=True), stamps[:5]
    # user lain tidak melihat sesi ini
    other = client.get("/api/chat/sessions", headers=H_BUDI).json()
    assert sid not in {s["id"] for s in other["sessions"]}
    print("  ok sessions_list_shape_and_ordering")


def test_history_includes_citations_for_rerender():
    r = _ask(H_BUDI, "Apa itu Nusantara OS?")
    sid = r["session_id"]
    hist = client.get("/api/chat/" + sid, headers=H_BUDI).json()
    assistant = [m for m in hist["messages"] if m["role"] == "assistant"][-1]
    assert isinstance(assistant["citations"], list)
    assert assistant["citations"], "sitasi harus tersimpan agar riwayat bisa dirender"
    assert "interrupted" in assistant and assistant["interrupted"] is False
    assert assistant["created_at"]
    print("  ok history_includes_citations_for_rerender")


def test_admin_reads_but_cannot_write_other_session():
    sid = _ask(H_BUDI, "Apa kebijakan WFH?")["session_id"]
    r = client.get("/api/chat/" + sid, headers=H_ADMIN)
    assert r.status_code == 200, r.text  # admin boleh membaca untuk audit
    assert r.json()["messages"]
    w = client.post("/api/chat", headers=H_ADMIN, json={"question": "sisip", "session_id": sid})
    assert w.status_code == 403, w.text
    assert w.json()["detail"] == "session_forbidden"
    print("  ok admin_reads_but_cannot_write_other_session")


def test_deleted_session_leaves_list_and_history():
    sid = _ask(H_DEWI, "Bagaimana kebijakan parkir?")["session_id"]
    assert client.delete("/api/chat/sessions/" + sid, headers=H_DEWI).status_code == 200
    assert client.get("/api/chat/" + sid, headers=H_DEWI).status_code == 404
    ids = {s["id"] for s in client.get("/api/chat/sessions", headers=H_DEWI).json()["sessions"]}
    assert sid not in ids
    # idempotent: hapus dua kali tetap 404, bukan 500
    assert client.delete("/api/chat/sessions/" + sid, headers=H_DEWI).status_code == 404
    print("  ok deleted_session_leaves_list_and_history")


if __name__ == "__main__":
    test_session_id_roundtrip_keeps_one_conversation()
    test_sessions_list_shape_and_ordering()
    test_history_includes_citations_for_rerender()
    test_admin_reads_but_cannot_write_other_session()
    test_deleted_session_leaves_list_and_history()
    print("ALL CHAT SESSION API TESTS PASS")
