"""Regresi: POST /api/documents tidak boleh menerima dokumen kosong/invalid.

Bug produksi: POST hanya memvalidasi duplikat id. Payload tanpa title/source_type
(atau content kosong, confidentiality/status ngawur) tetap tersimpan sehingga
tabel admin menampilkan baris kosong dan retrieval bisa memuat dokumen tanpa isi.
PUT sudah memvalidasi title+source_type; sekarang keduanya konsisten.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

_TMP = tempfile.mkdtemp()
os.environ["NOHALU_DB_PATH"] = os.path.join(_TMP, "validation.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402

client = TestClient(appmod.app)


def _h(uid):
    r = client.post("/api/auth/login", json={"user_id": uid})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


H_HR = _h("u_sari")
H_EMP = _h("u_budi")

GOOD = {
    "title": "Dokumen Uji",
    "source_type": "reference",
    "content": "Isi dokumen uji yang cukup panjang.",
    "workspace_ids": ["ws_onboarding"],
    "access_roles": ["employee", "hr"],
    "confidentiality": "internal",
    "status": "published",
}


def test_post_rejects_missing_title():
    r = client.post("/api/documents", headers=H_HR, json={**GOOD, "title": ""})
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "title_required"
    print("  ok post_rejects_missing_title")


def test_post_rejects_missing_source_type():
    r = client.post("/api/documents", headers=H_HR, json={**GOOD, "source_type": "  "})
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "source_type_required"
    print("  ok post_rejects_missing_source_type")


def test_post_rejects_empty_content():
    r = client.post("/api/documents", headers=H_HR, json={**GOOD, "content": ""})
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "content_required"
    print("  ok post_rejects_empty_content")


def test_post_rejects_bad_enum():
    r = client.post("/api/documents", headers=H_HR, json={**GOOD, "confidentiality": "public"})
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "invalid_confidentiality"

    r = client.post("/api/documents", headers=H_HR, json={**GOOD, "status": "aktif"})
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "invalid_status"
    print("  ok post_rejects_bad_enum")


def test_post_accepts_valid_document_and_trims():
    r = client.post("/api/documents", headers=H_HR, json={**GOOD, "title": "  Dok Trim  "})
    assert r.status_code == 200, r.text
    doc_id = r.json()["id"]
    got = client.get(f"/api/documents/{doc_id}", headers=H_HR)
    assert got.status_code == 200, got.text
    assert got.json()["title"] == "Dok Trim", got.json()["title"]
    print("  ok post_accepts_valid_document_and_trims")


def test_put_rejects_empty_content():
    r = client.put("/api/documents/doc_glossary", headers=H_HR, json={"content": "   "})
    assert r.status_code == 422, r.text
    assert r.json()["detail"] == "content_required"
    print("  ok put_rejects_empty_content")


def test_employee_cannot_create():
    r = client.post("/api/documents", headers=H_EMP, json=GOOD)
    assert r.status_code == 403, r.text
    print("  ok employee_cannot_create")


if __name__ == "__main__":
    test_post_rejects_missing_title()
    test_post_rejects_missing_source_type()
    test_post_rejects_empty_content()
    test_post_rejects_bad_enum()
    test_post_accepts_valid_document_and_trims()
    test_put_rejects_empty_content()
    test_employee_cannot_create()
    print("DOC VALIDATION TESTS PASS")
