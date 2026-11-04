"""Sprint 3.6 — digest pertanyaan tanpa jawaban untuk perbaikan KB (HR/admin)."""
import datetime as dt
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "digest.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402

client = TestClient(appmod.app)


def _h(uid):
    r = client.post("/api/auth/login", json={"user_id": uid})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


def _log(question, reason, uid="u_budi"):
    ts = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    appmod.CONN.execute(
        """INSERT INTO audit_log (timestamp,user_id,user_role,question,
           permission_decision,allowed_sources,retrieved_sources,selected_sources,
           conflict_detected,answer,citations,refusal_reason,pii_masked,
           prompt_injection_detected,agent_version,event_type)
           VALUES (?,?,?,?,'allow','[]','[]','[]',0,NULL,'[]',?,0,0,'test','qa')""",
        (ts, uid, "employee", question, reason),
    )
    appmod.CONN.commit()


def test_digest_requires_hr_or_admin():
    assert client.get("/api/audit/refusals/digest", headers=_h("u_budi")).status_code == 403
    assert client.get("/api/audit/refusals/digest", headers=_h("u_andi")).status_code == 403
    assert client.get("/api/audit/refusals/digest", headers=_h("u_rina")).status_code == 200


def test_digest_groups_repeated_refusals():
    appmod.CONN.execute("DELETE FROM audit_log WHERE question LIKE 'DIGEST%'")
    appmod.CONN.commit()
    q = "DIGEST kapan kantor pindah?"
    for _ in range(3):
        _log(q, "no_data")
    _log("DIGEST siapa direktur utama?", "no_access")
    body = client.get("/api/audit/refusals/digest?days=30&top=10",
                      headers=_h("u_rina")).json()
    top = body["items"][0]
    assert top["question"] == q and top["count"] == 3 and top["reason"] == "no_data"
    assert body["by_reason"]["no_data"] >= 3
    assert body["total"] >= 4


def test_digest_ignores_answered_questions():
    appmod.CONN.execute("DELETE FROM audit_log WHERE question LIKE 'DIGEST-OK%'")
    appmod.CONN.commit()
    _log("DIGEST-OK berapa hari cuti?", None)
    body = client.get("/api/audit/refusals/digest", headers=_h("u_rina")).json()
    assert all("DIGEST-OK" not in i["question"] for i in body["items"])


def test_digest_aggregate_independent_of_top_n():
    """`by_reason`/`total` harus dari seluruh rentang, bukan hanya top-N."""
    appmod.CONN.execute("DELETE FROM audit_log WHERE question LIKE 'DIGEST-AGG%'")
    appmod.CONN.commit()
    for i in range(6):
        for _ in range(i + 1):
            _log(f"DIGEST-AGG pertanyaan {i}", "no_data")
    _log("DIGEST-AGG satu-satunya no_access", "no_access")
    body = client.get("/api/audit/refusals/digest?days=30&top=2",
                      headers=_h("u_rina")).json()
    assert body["returned"] == 2
    assert body["unique_questions"] >= 7
    # 1+2+3+4+5+6 = 21 no_data + 1 no_access
    assert body["by_reason"]["no_data"] >= 21
    assert body["by_reason"]["no_access"] >= 1
    assert body["total"] >= 22, body


def test_digest_offset_reaches_rows_past_top():
    """`top` saja membuat refusal ke-51 dst. tak pernah bisa dibaca.

    Halaman Digest HR dipakai untuk menyisir SEMUA pertanyaan gagal, jadi
    daftar yang berhenti di 50 baris pertama membuat dokumen yang hilang
    tidak pernah ketemu.
    """
    h = _h("u_rina")
    marker = "digest_page_probe"
    for i in range(6):
        _log(f"{marker} {i}", "no_data", uid="u_digest_page")
    try:
        full = client.get("/api/audit/refusals/digest?days=180&top=50",
                          headers=h).json()
        groups = full["groups"]
        page1 = client.get("/api/audit/refusals/digest?days=180&top=3&offset=0",
                           headers=h).json()
        page2 = client.get("/api/audit/refusals/digest?days=180&top=3&offset=3",
                           headers=h).json()
        assert page1["groups"] == groups
        assert (page1["offset"], page2["offset"]) == (0, 3)
        assert page1["limit"] == 3
        assert len(page1["items"]) == 3
        q1 = {i["question"] for i in page1["items"]}
        q2 = {i["question"] for i in page2["items"]}
        assert not (q1 & q2), "halaman tidak boleh tumpang tindih"
        assert page1["has_more"] is True
        # `total` tetap jumlah KEJADIAN (dipakai judul halaman); `groups`
        # adalah jumlah baris yang bisa dipaging.
        assert page1["total"] == page1["total_refusals"]
        assert page1["total"] >= page1["groups"]
    finally:
        appmod.CONN.execute("DELETE FROM audit_log WHERE user_id = 'u_digest_page'")
        appmod.CONN.commit()
