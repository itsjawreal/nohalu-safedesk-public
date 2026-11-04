"""P0 — pagination sungguhan.

Bug sebelumnya: `?limit=&offset=&status=` diterima di URL tetapi tidak pernah
dibaca endpoint. Akibatnya `/api/audit` selalu ~103 KB, `/api/documents`
selalu 70 baris, dan `?status=` di /api/requests tidak berpengaruh apa pun.
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "paging_api.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402
from authutil import login_headers  # noqa: E402

client = TestClient(appmod.app)


def _admin():
    return login_headers(client, "u_rina")


def test_documents_honours_limit_and_offset():
    h = _admin()
    full = client.get("/api/documents?limit=200", headers=h).json()
    assert full["total"] >= 60, full["total"]

    page1 = client.get("/api/documents?limit=10&offset=0", headers=h).json()
    page2 = client.get("/api/documents?limit=10&offset=10", headers=h).json()
    assert len(page1["documents"]) == 10
    assert page1["count"] == 10
    assert page1["total"] == full["total"]
    assert page1["has_more"] is True
    ids1 = [d["id"] for d in page1["documents"]]
    ids2 = [d["id"] for d in page2["documents"]]
    assert not set(ids1) & set(ids2), "halaman 1 dan 2 tidak boleh tumpang tindih"


def test_documents_last_page_has_no_more():
    h = _admin()
    total = client.get("/api/documents?limit=1", headers=h).json()["total"]
    last = client.get("/api/documents?limit=10&offset=%d" % (total - 3), headers=h).json()
    assert last["has_more"] is False
    assert len(last["documents"]) == 3


def test_documents_invalid_paging_falls_back_to_default():
    """Pagination tidak boleh membuat halaman gagal tampil."""
    h = _admin()
    r = client.get("/api/documents?limit=abc&offset=-5", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["limit"] > 0
    assert body["offset"] == 0


def test_documents_status_filter_applies():
    h = _admin()
    all_docs = client.get("/api/documents?limit=200", headers=h).json()["documents"]
    target = next(d["status"] for d in all_docs if d.get("status"))
    r = client.get("/api/documents?status=%s&limit=200" % target, headers=h).json()
    assert r["documents"], "filter status mengosongkan hasil"
    assert all(d["status"] == target for d in r["documents"])
    assert r["total"] < len(all_docs)


def test_documents_query_filter_applies():
    h = _admin()
    r = client.get("/api/documents?q=cuti&limit=200", headers=h).json()
    assert r["documents"], "pencarian 'cuti' tidak menemukan dokumen"
    for d in r["documents"]:
        hay = (d["title"] + d["source_type"] + d["id"]).lower()
        assert "cuti" in hay


def test_audit_is_paged_and_total_is_honest():
    h = _admin()
    # Pastikan ada isi: audit log bisa kosong pada DB tes yang baru dibuat.
    with appmod._DB_LOCK:
        n = appmod.CONN.execute("SELECT COUNT(*) c FROM audit_log").fetchone()["c"]
        for i in range(max(0, 12 - n)):
            appmod.CONN.execute(
                "INSERT INTO audit_log (timestamp,user_id,user_role,question,"
                "permission_decision,allowed_sources,retrieved_sources,"
                "selected_sources,answer,citations,agent_version,event_type)"
                " VALUES ('2026-01-01T00:00:00Z','u_budi','employee','probe',"
                "'allow','[]','[]','[]','jawaban','[]','1.0','qa')")
        appmod.CONN.commit()
    full = client.get("/api/audit?limit=200", headers=h).json()
    assert full["total"] >= 1
    small = client.get("/api/audit?limit=5", headers=h).json()
    assert len(small["audit"]) == 5
    assert small["count"] == 5
    assert small["total"] == full["total"]
    assert small["has_more"] is True
    assert small["limit"] == 5


def test_audit_status_filter_total_matches_returned_set():
    h = _admin()
    r = client.get("/api/audit?status=qa&limit=3", headers=h).json()
    assert all(a.get("event_type") == "qa" for a in r["audit"])
    # `total` harus jumlah yang COCOK, bukan jumlah yang dipindai.
    wider = client.get("/api/audit?status=qa&limit=200", headers=h).json()
    assert r["total"] == wider["total"]


def test_requests_status_filter_is_not_ignored():
    """Dulu `?status=` diterima lalu diabaikan: UI mengirim filter yang mati."""
    h = _admin()
    with appmod._DB_LOCK:
        if not appmod.CONN.execute("SELECT COUNT(*) c FROM requests").fetchone()["c"]:
            appmod.CONN.execute(
                "INSERT INTO requests (user_id,type,title,status,approver_roles,"
                "stages,stage_idx,history,created_at) VALUES "
                "('u_budi','cuti','probe','pending','[\"team_lead\"]',"
                "'[\"team_lead\"]',0,'[]','2026-01-01T00:00:00Z')")
            appmod.CONN.commit()
    allb = client.get("/api/requests?limit=200", headers=h).json()
    assert allb["all"], "butuh minimal satu pengajuan untuk tes"
    status = allb["all"][0]["status"]
    r = client.get("/api/requests?status=%s&limit=200" % status, headers=h).json()
    assert r["all"], "filter status mengosongkan hasil"
    assert all(x["status"] == status for x in r["all"])
    assert r["total"]["all"] <= allb["total"]["all"]


def test_requests_type_filter_applies():
    h = _admin()
    r = client.get("/api/requests?type=cuti&limit=200", headers=h).json()
    for key in ("mine", "approval", "all"):
        assert all(x["type"] == "cuti" for x in r[key])


def test_referensi_paged_with_facets():
    h = login_headers(client, "u_sari")
    full = client.get("/api/referensi?limit=200", headers=h).json()
    page = client.get("/api/referensi?limit=5&offset=0", headers=h).json()
    assert len(page["documents"]) == 5
    assert page["total"] == full["total"]
    assert page["has_more"] is True
    # Facet dihitung dari seluruh koleksi, bukan dari halaman yang dikirim.
    assert sum(page["facets"].values()) == page["total"]


def test_referensi_types_filter_uses_facets():
    h = login_headers(client, "u_sari")
    base = client.get("/api/referensi?limit=200", headers=h).json()
    stype = max(base["facets"], key=base["facets"].get)
    r = client.get("/api/referensi?types=%s&limit=200" % stype, headers=h).json()
    assert r["total"] == base["facets"][stype]
    assert all(d["source_type"] == stype for d in r["documents"])


def test_referensi_q_actually_filters_results():
    """`q` dulu hanya menyortir: hasil tetap SELURUH koleksi.

    Terukur di :8890 sebelum perbaikan: `?q=Makassar` -> `total: 60` (semua
    dokumen yang boleh dibaca), dokumen tak relevan hanya terdorong ke
    belakang. UI tampak benar karena `referensi.js` menyaring ulang satu
    halaman, sehingga bug tersembunyi sampai pengguna mencari istilah yang
    jarang (mis. "zzzznotexist" -> total 70).
    """
    # `doc_chat_makassar` hanya untuk team_lead/admin (seed.py:201), jadi
    # pencarian diuji dengan peran yang memang boleh membacanya.
    h = login_headers(client, "u_andi")
    hit = client.get("/api/referensi?q=Makassar&limit=200", headers=h).json()
    assert 1 <= hit["total"] < 10, hit["total"]
    assert hit["documents"], hit
    assert "makassar" in json.dumps(hit["documents"], ensure_ascii=False).lower()

    miss = client.get("/api/referensi?q=zzzznotexist&limit=200", headers=h).json()
    assert miss["total"] == 0
    assert miss["documents"] == []

    # Facet ikut mengecil: jumlahnya harus sama dengan total hasil pencarian,
    # bukan jumlah seluruh koleksi.
    assert sum(hit["facets"].values()) == hit["total"]

    # Query berisi spasi saja bukan pencarian: koleksi tidak boleh kosong.
    blank = client.get("/api/referensi?q=%20%20&limit=200", headers=h).json()
    assert blank["total"] > 10, blank["total"]


def test_referensi_facets_stable_across_pages():
    h = login_headers(client, "u_sari")
    a = client.get("/api/referensi?limit=5&offset=0", headers=h).json()
    b = client.get("/api/referensi?limit=5&offset=5", headers=h).json()
    assert a["facets"] == b["facets"], "facet berubah saat pindah halaman"
