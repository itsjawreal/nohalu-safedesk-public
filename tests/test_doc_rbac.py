"""Regresi RBAC endpoint manajemen dokumen.

Bug produksi: `require_hr` hanya memeriksa role, bukan ACL dokumen. Akibatnya
HR (u_sari) bisa membaca isi dokumen `restricted` milik team_lead/admin lewat
GET /api/documents — termasuk "Proyek Kenari ... Anggaran awal Rp850.000.000"
— padahal GET /api/documents/{id} menolaknya dengan 404. HR juga bisa
MENGUBAH dokumen tersebut lewat PUT.

Semua tes offline (LLM dimatikan) dan memakai DB sementara.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

_TMP = tempfile.mkdtemp()
os.environ["NOHALU_DB_PATH"] = os.path.join(_TMP, "rbac.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402

client = TestClient(appmod.app)


def _h(uid):
    r = client.post("/api/auth/login", json={"user_id": uid})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


H_HR = _h("u_sari")
H_ADMIN = _h("u_rina")
H_EMP = _h("u_budi")
H_TL = _h("u_andi")


def _all_docs(headers):
    """Ambil SEMUA halaman /api/documents.

    Endpoint sekarang berpaginasi (default 50/halaman). Tes RBAC harus
    memeriksa keseluruhan himpunan, bukan halaman pertama.
    """
    ids, offset = [], 0
    while True:
        r = client.get("/api/documents?limit=200&offset=%d" % offset, headers=headers)
        assert r.status_code == 200, r.text
        body = r.json()
        ids.extend(d["id"] for d in body["documents"])
        if not body.get("has_more"):
            return ids
        offset += body["limit"]


def _ids(headers):
    return set(_all_docs(headers))


def test_employee_still_forbidden():
    """Karyawan biasa tetap 403 (bukan 200 kosong)."""
    assert client.get("/api/documents", headers=H_EMP).status_code == 403


def test_admin_sees_every_document():
    ids = _ids(H_ADMIN)
    assert "doc_project_kenari" in ids
    assert "doc_salary_structure" in ids
    # Sejak soft delete (Sprint 4.5), dokumen di tempat sampah sengaja tidak
    # muncul di daftar manajemen — admin melihat semua yang BELUM dihapus.
    live = appmod.CONN.execute(
        "SELECT id FROM documents WHERE deleted_at IS NULL").fetchall()
    assert len(ids) == len(live)


def test_hr_cannot_read_restricted_teamlead_doc():
    """Inti bug: dokumen Kenari (ACL team_lead/admin) tidak boleh muncul ke HR.

    Catatan: angka Rp850.000.000 juga muncul sah di doc_fin_annual_budget_2026
    (milik HR), jadi penanda kebocoran di sini adalah id + judul dokumennya.
    """
    r = client.get("/api/documents", headers=H_HR)
    assert r.status_code == 200
    ids = {d["id"] for d in r.json()["documents"]}
    assert "doc_project_kenari" not in ids, "HR tidak punya ws_project_kenari"
    assert "doc_chat_makassar" not in ids, "chat export TL/admin tidak untuk HR"
    titles = {d["title"] for d in r.json()["documents"]}
    assert "Proyek Kenari — Dokumen Internal" not in titles


def test_hr_keeps_own_restricted_docs():
    """Pembatasan tidak boleh mengurangi dokumen yang memang milik HR."""
    ids = _ids(H_HR)
    assert "doc_salary_structure" in ids      # access_roles hr+admin
    assert "doc_fin_tax_slip" in ids          # access_roles hr+admin
    assert "doc_fin_annual_budget_2026" in ids
    assert "doc_hr_contacts" in ids


def test_team_lead_cannot_use_management_endpoint():
    """Endpoint manajemen KB tetap hr/admin; TL memakai /api/referensi."""
    assert client.get("/api/documents", headers=H_TL).status_code == 403
    ref = client.get("/api/referensi", headers=H_TL)
    assert ref.status_code == 200
    ids = {d["id"] for d in ref.json()["documents"]}
    assert "doc_project_kenari" in ids        # TL punya ws_project_kenari
    assert "doc_salary_structure" not in ids  # tapi bukan HR


def test_hr_cannot_update_restricted_doc():
    """PUT juga harus dibatasi, bukan hanya GET (celah tulis)."""
    r = client.put(
        "/api/documents/doc_project_kenari",
        json={"title": "Proyek Kenari — Dokumen Internal",
              "source_type": "project_doc",
              "content": "Proyek Kenari dalam persiapan. Anggaran awal Rp850.000.000. Restricted.",
              "workspace_ids": ["ws_project_kenari"],
              "access_roles": ["team_lead", "admin"],
              "confidentiality": "restricted", "status": "indexed", "version": "1"},
        headers=H_HR,
    )
    assert r.status_code == 404, r.text
    # Bentuk detail tetap sama seperti sebelum aturan izin dipindah ke
    # `hr_actions`: kode + pesan, bukan string mentah.
    assert r.json()["detail"]["code"] == "doc_not_found"
    # isi asli tidak berubah
    row = appmod.CONN.execute(
        "SELECT content FROM documents WHERE id='doc_project_kenari'").fetchone()
    assert "850.000.000" in row["content"]


def test_hr_can_update_own_doc():
    r = client.put(
        "/api/documents/doc_salary_structure",
        json={"title": "Struktur Gaji & Grade Karyawan",
              "source_type": "salary_structure",
              "content": "uji",
              "workspace_ids": ["ws_finance", "ws_onboarding"],
              "access_roles": ["hr", "admin"],
              "confidentiality": "restricted", "status": "indexed", "version": "1"},
        headers=H_HR,
    )
    assert r.status_code == 200, r.text


def test_duplicate_create_is_conflict():
    payload = {"id": "doc_project_kenari", "title": "x", "source_type": "project_doc",
               "content": "x"}
    r = client.post("/api/documents", json=payload, headers=H_ADMIN)
    assert r.status_code == 409, r.text


def test_update_without_owner_keeps_metadata():
    """Edit parsial tidak boleh menghapus owner atau menurunkan confidentiality.

    Bug: form admin tidak mengirim owner/confidentiality, dan PUT lama menulis
    NULL + "internal" — dokumen restricted diam-diam jadi internal.
    """
    before = appmod.CONN.execute(
        "SELECT owner, confidentiality FROM documents WHERE id='doc_salary_structure'"
    ).fetchone()
    assert before["confidentiality"] == "restricted"

    r = client.put(
        "/api/documents/doc_salary_structure",
        json={"title": "Struktur Gaji & Grade Karyawan", "content": "isi baru"},
        headers=H_HR,
    )
    assert r.status_code == 200, r.text

    after = appmod.CONN.execute(
        "SELECT owner, confidentiality, content FROM documents WHERE id='doc_salary_structure'"
    ).fetchone()
    assert after["owner"] == before["owner"], "owner hilang karena edit parsial"
    assert after["confidentiality"] == "restricted", "confidentiality turun diam-diam"
    assert after["content"] == "isi baru"


def test_update_requires_title_and_source_type_when_creating_state():
    """title/source_type wajib ada di dokumen; payload parsial tetap aman."""
    r = client.put(
        "/api/documents/doc_hr_contacts",
        json={"title": ""},
        headers=H_HR,
    )
    assert r.status_code == 422, r.text
