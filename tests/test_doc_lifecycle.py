"""Sprint 4.3–4.5 — siklus hidup dokumen.

Yang dikunci di sini:
  * transisi status tidak bisa melompat sembarangan (draft → published boleh,
    draft → superseded tidak);
  * setiap perubahan isi menghasilkan baris versi, dan versi lama TIDAK
    hilang (bisa di-diff);
  * hapus = soft delete: dokumen hilang dari retrieval & perpustakaan, tetapi
    baris dan riwayat versinya tetap ada dan bisa dipulihkan;
  * export CSV mengikuti filter yang sama dengan tabel.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "lifecycle.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402
from authutil import login_headers  # noqa: E402

client = TestClient(appmod.app)
H_ADMIN = login_headers(client, "u_rina")
H_HR = login_headers(client, "u_sari")
H_EMP = login_headers(client, "u_budi")


def _new_doc(doc_id, **over):
    payload = {
        "id": doc_id, "title": "SOP Uji " + doc_id,
        "source_type": "sop", "effective_date": "2026-11-01",
        "workspace_ids": ["ws_onboarding"], "access_roles": ["employee", "hr", "admin"],
        "content": "Langkah satu.\nLangkah dua.\nLangkah tiga.",
        "confidentiality": "internal", "status": "draft",
    }
    payload.update(over)
    r = client.post("/api/documents", json=payload, headers=H_ADMIN)
    assert r.status_code == 200, r.text
    return r.json()


def test_create_records_first_version():
    _new_doc("doc_lc_a")
    r = client.get("/api/documents/doc_lc_a/versions", headers=H_ADMIN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] >= 1
    assert body["versions"][0]["version"] == "1"
    assert body["versions"][0]["current"] is True


def test_create_accepts_csv_string_and_returns_lists():
    """`workspace_ids` boleh dikirim sebagai string koma-separated.

    Regresi nyata: POST menerima string lalu menyimpannya apa adanya, sehingga
    `json.loads` mengembalikan STRING dan render tabel admin melempar
    "d.workspace_ids.join is not a function" — halaman berhenti di DOM lama
    (gejala: klik filter tidak mengubah apa pun). Kontrak respons harus selalu
    list, apa pun bentuk inputnya.
    """
    r = client.post("/api/documents", json={
        "id": "doc_lc_csv", "title": "SOP CSV", "source_type": "sop",
        "workspace_ids": "ws_onboarding, ws_it",
        "access_roles": "employee,hr",
        "content": "Isi dokumen uji.", "status": "draft",
    }, headers=H_ADMIN)
    assert r.status_code == 200, r.text

    row = next(d for d in client.get("/api/documents?q=doc_lc_csv", headers=H_ADMIN).json()["documents"]
               if d["id"] == "doc_lc_csv")
    assert row["workspace_ids"] == ["ws_onboarding", "ws_it"], row["workspace_ids"]
    assert row["access_roles"] == ["employee", "hr"], row["access_roles"]

    # Update juga harus menormalkan string, bukan menyimpan bentuk mentah.
    r = client.put("/api/documents/doc_lc_csv", json={"workspace_ids": "ws_it"},
                   headers=H_ADMIN)
    assert r.status_code == 200, r.text
    row = next(d for d in client.get("/api/documents?q=doc_lc_csv", headers=H_ADMIN).json()["documents"]
               if d["id"] == "doc_lc_csv")
    assert row["workspace_ids"] == ["ws_it"], row["workspace_ids"]


def test_legacy_string_row_does_not_break_clients():
    """Baris lama di DB yang tersimpan sebagai string tetap dibaca sebagai list.

    Memperbaiki penulisan tidak memperbaiki data yang sudah ada. Tanpa
    pembacaan defensif, satu baris warisan membuat seluruh tabel admin gagal
    dirender selamanya (tidak ada jalan bagi admin untuk memperbaikinya lewat UI).
    """
    _new_doc("doc_lc_legacy")
    appmod.CONN.execute(
        "UPDATE documents SET workspace_ids=?, access_roles=? WHERE id=?",
        ('"ws_onboarding"', 'employee', "doc_lc_legacy"))
    appmod.CONN.commit()
    row = next(d for d in client.get("/api/documents?q=doc_lc_legacy", headers=H_ADMIN).json()["documents"]
               if d["id"] == "doc_lc_legacy")
    assert row["workspace_ids"] == ["ws_onboarding"], row["workspace_ids"]
    assert row["access_roles"] == ["employee"], row["access_roles"]


def test_status_transition_is_validated():
    _new_doc("doc_lc_b", status="draft")
    # draft → superseded tidak sah (harus lewat published dulu)
    r = client.put("/api/documents/doc_lc_b", json={"status": "superseded"}, headers=H_ADMIN)
    assert r.status_code == 409, r.text
    assert r.json()["detail"]["code"] == "invalid_status_transition"

    # draft → review → published sah
    r = client.put("/api/documents/doc_lc_b", json={"status": "review"}, headers=H_ADMIN)
    assert r.status_code == 200, r.text
    r = client.put("/api/documents/doc_lc_b", json={"status": "published"}, headers=H_ADMIN)
    assert r.status_code == 200, r.text
    assert r.json()["doc_status"] == "published"

    # status tak dikenal tetap 422
    r = client.put("/api/documents/doc_lc_b", json={"status": "hilang"}, headers=H_ADMIN)
    assert r.status_code == 422


def test_edit_creates_new_version_and_keeps_old_content():
    _new_doc("doc_lc_c", content="isi asli")
    r = client.put("/api/documents/doc_lc_c", json={"content": "isi revisi"},
                   headers=H_ADMIN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version"] == "2"
    assert body["previous_version"] == "1"

    versions = client.get(
        "/api/documents/doc_lc_c/versions?include_content=true",
        headers=H_ADMIN).json()["versions"]
    by_ver = {v["version"]: v for v in versions}
    assert by_ver["1"]["content"] == "isi asli", "isi lama harus tersimpan"
    assert by_ver["2"]["content"] == "isi revisi"


def test_diff_between_versions():
    _new_doc("doc_lc_d", content="baris a\nbaris b")
    client.put("/api/documents/doc_lc_d", json={"content": "baris a\nbaris b diubah"},
               headers=H_ADMIN)
    r = client.get("/api/documents/doc_lc_d/diff?a=1&b=2", headers=H_ADMIN)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["same"] is False
    assert body["added_lines"] >= 1
    assert body["removed_lines"] >= 1
    assert "-baris b" in body["diff"]
    assert "+baris b diubah" in body["diff"]


def test_diff_missing_version_is_404():
    _new_doc("doc_lc_e")
    r = client.get("/api/documents/doc_lc_e/diff?a=1&b=99", headers=H_ADMIN)
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "version_not_found"


def test_versions_require_manageable_doc():
    r = client.get("/api/documents/doc_lc_e/versions", headers=H_EMP)
    assert r.status_code == 403


def test_soft_delete_hides_from_read_paths_but_keeps_versions():
    _new_doc("doc_lc_f", status="published", content="isi yang akan dihapus")
    # pastikan terlihat di perpustakaan sebelum dihapus
    before = client.get("/api/referensi?limit=200", headers=H_ADMIN).json()
    assert any(d["id"] == "doc_lc_f" for d in before["documents"])

    r = client.delete("/api/documents/doc_lc_f", headers=H_ADMIN)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "deleted"

    after = client.get("/api/referensi?limit=200", headers=H_ADMIN).json()
    assert not any(d["id"] == "doc_lc_f" for d in after["documents"]), \
        "dokumen terhapus tidak boleh muncul di perpustakaan"

    # hilang dari daftar default, tetapi muncul saat mode deleted diminta
    listed = client.get("/api/documents?limit=200", headers=H_ADMIN).json()
    assert not any(d["id"] == "doc_lc_f" for d in listed["documents"])
    trash = client.get("/api/documents?deleted=only&limit=200", headers=H_ADMIN).json()
    assert any(d["id"] == "doc_lc_f" for d in trash["documents"])

    # riwayat versi tetap ada
    v = client.get("/api/documents/doc_lc_f/versions", headers=H_ADMIN)
    assert v.status_code == 200
    assert v.json()["count"] >= 1


def test_restore_returns_document_to_library():
    _new_doc("doc_lc_g", status="published", content="isi yang dipulihkan")
    client.delete("/api/documents/doc_lc_g", headers=H_ADMIN)
    r = client.post("/api/documents/doc_lc_g/restore", headers=H_ADMIN)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "restored"
    again = client.get("/api/referensi?limit=200", headers=H_ADMIN).json()
    assert any(d["id"] == "doc_lc_g" for d in again["documents"])


def test_hr_can_restore_document_from_trash():
    """Regresi: restore lewat HTTP selalu 404 untuk HR.

    `_manageable_docs` lama memakai `filter_allowed_docs`, yang membuang baris
    `deleted_at`. Dokumen yang baru dihapus karena itu lenyap dari daftar yang
    boleh dikelola, sehingga `POST /restore` menolak dengan 404 — padahal
    justru dokumen itulah yang mau dipulihkan. Admin tidak terkena karena
    cabangnya membaca seluruh tabel.
    """
    _new_doc("doc_lc_hr_restore", status="published",
             access_roles=["employee", "hr", "admin"])
    assert client.delete("/api/documents/doc_lc_hr_restore",
                         headers=H_HR).status_code == 200
    r = client.post("/api/documents/doc_lc_hr_restore/restore", headers=H_HR)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "restored"


def test_deleted_document_cannot_be_edited():
    _new_doc("doc_lc_h")
    client.delete("/api/documents/doc_lc_h", headers=H_ADMIN)
    r = client.put("/api/documents/doc_lc_h", json={"title": "ganti"}, headers=H_ADMIN)
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "doc_deleted"


def test_delete_is_idempotent():
    _new_doc("doc_lc_i")
    first = client.delete("/api/documents/doc_lc_i", headers=H_ADMIN)
    second = client.delete("/api/documents/doc_lc_i", headers=H_ADMIN)
    assert first.status_code == 200 and second.status_code == 200
    assert second.json()["status"] == "already_deleted"


def test_employee_cannot_delete():
    _new_doc("doc_lc_j")
    r = client.delete("/api/documents/doc_lc_j", headers=H_EMP)
    assert r.status_code == 403


def test_audit_csv_export():
    r = client.get("/api/audit/export.csv?limit=5", headers=H_ADMIN)
    assert r.status_code == 200, r.text
    assert "text/csv" in r.headers["content-type"]
    assert "attachment" in r.headers["content-disposition"]
    text = r.content.decode("utf-8-sig")
    lines = text.splitlines()
    assert lines[0].startswith("id;timestamp;event_type")
    assert 1 <= len(lines) - 1 <= 5


def test_requests_csv_export_respects_scope():
    client.post("/api/requests", json={
        "type": "cuti", "title": "Cuti ekspor",
        "start_date": "2026-12-01", "end_date": "2026-12-02"},
        headers=H_EMP)
    r = client.get("/api/requests/export.csv?scope=mine", headers=H_EMP)
    assert r.status_code == 200, r.text
    text = r.content.decode("utf-8-sig")
    assert "Cuti ekspor" in text
    # HR tidak boleh melihat keranjang "mine" milik orang lain
    r2 = client.get("/api/requests/export.csv?scope=mine", headers=H_HR)
    assert r2.status_code == 200
    assert "Cuti ekspor" not in r2.content.decode("utf-8-sig")
