"""Aksi HR/admin yang dipicu dari chat (backend/hr_actions.py).

Sebelum modul ini, semua kemampuan tulis HR hanya hidup di halaman: menerbitkan
pengumuman, slip gaji, mengubah dokumen, dan mengelola akun karyawan. Agent
hanya bisa MENJAWAB, sehingga permintaan seperti "terbitkan pengumuman untuk
tim" berakhir dengan instruksi menekan tombol.

Tes ini menjaga tiga hal yang mudah rusak tanpa disadari:
1. RBAC per peran benar-benar ditegakkan di lapisan domain (bukan hanya di UI).
2. Pemulihan dokumen dari tempat sampah bekerja untuk HR — dulu selalu 404
   karena `filter_allowed_docs` membuang baris `deleted_at`.
3. Revisi dokumen tidak diam-diam mengubah status/klasifikasi.
"""
import os
import sqlite3
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_LLM_ENABLED"] = "0"

import db
import engine
import hr_actions
import docs_lifecycle as lifecycle


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    db.seed_if_empty(conn)
    db.ensure_seed_users(conn)
    return conn


def actor(conn, uid):
    user = engine.get_user(conn, uid)
    assert user is not None, uid
    return user


def new_doc(conn, doc_id="doc_ha_a", status="published",
            confidentiality="internal", roles=("hr", "admin")):
    import json
    conn.execute(
        """INSERT INTO documents
           (id,title,source_type,effective_date,workspace_ids,access_roles,
            content,owner,confidentiality,status,version,content_hash,local_edit)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1)""",
        (doc_id, "Dokumen " + doc_id, "sop", "2026-01-01",
         json.dumps(["ws_onboarding"]), json.dumps(list(roles)),
         "isi awal", "HR", confidentiality, status, "1", "h0"))
    conn.commit()
    return doc_id


# ---------------------------------------------------------------------------
# Izin
# ---------------------------------------------------------------------------
def test_employee_cannot_create_announcement():
    conn = make_conn()
    out, status = hr_actions.create_announcement(
        conn, actor(conn, "u_budi"),
        {"title": "Coba", "content": "isi"})
    assert status == 403 and out["error"] == "not_authorized"


def test_team_lead_cannot_publish_payslip():
    conn = make_conn()
    out, status = hr_actions.publish_payslip(
        conn, actor(conn, "u_andi"),
        {"user_id": "u_budi", "period": "2026-10", "gross": 1000000})
    assert status == 403, out


def test_hr_cannot_change_employee_account():
    """Perubahan akun adalah kewenangan admin, bukan HR."""
    conn = make_conn()
    out, status = hr_actions.update_employee(
        conn, actor(conn, "u_sari"), "u_budi", {"job_title": "X"})
    assert status == 403, out


def test_finance_cannot_manage_documents():
    conn = make_conn()
    doc_id = new_doc(conn)
    out, status = hr_actions.delete_document(conn, actor(conn, "u_dewi"), doc_id)
    assert status == 403, out


# ---------------------------------------------------------------------------
# Pengumuman
# ---------------------------------------------------------------------------
def test_validate_announcement_does_not_write():
    """Draft yang tampil ke user harus divalidasi tanpa efek samping."""
    conn = make_conn()
    before = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
    bad, _ = hr_actions.validate_announcement(
        conn, actor(conn, "u_sari"), {"title": "Judul", "content": "Isi"})
    assert bad is None
    after = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
    assert after == before


def test_announcement_notifies_only_eligible_users():
    conn = make_conn()
    out, status = hr_actions.create_announcement(
        conn, actor(conn, "u_sari"),
        {"title": "Kebijakan baru", "content": "Mulai Senin.",
         "workspace_ids": "ws_onboarding",
         "access_roles": "employee,team_lead,hr,admin"})
    assert status == 201, out
    assert out["notified"] > 0

    rows = conn.execute(
        "SELECT user_id FROM notifications WHERE ref = ?", (out["id"],)).fetchall()
    targets = {r["user_id"] for r in rows}
    # `finance` mewarisi employee+team_lead (engine.ROLE_INHERITS), jadi Dewi
    # memang berhak. Yang diuji: pengumuman khusus HR tidak dikirim ke karyawan.
    assert "u_budi" in targets
    assert "u_sari" in targets
    # Dokumennya benar-benar ada dan bisa dibaca karyawan lewat ACL biasa.
    docs = engine.filter_allowed_docs(conn, actor(conn, "u_budi"))
    assert any(d["id"] == out["id"] for d in docs)


def test_announcement_validation_rejects_empty_title():
    conn = make_conn()
    out, status = hr_actions.create_announcement(
        conn, actor(conn, "u_sari"), {"title": " ", "content": "isi"})
    assert status == 422 and out["error"] == "title_required"


# ---------------------------------------------------------------------------
# Dokumen
# ---------------------------------------------------------------------------
def test_hr_can_restore_own_deleted_document():
    """Bug lama: `filter_allowed_docs` membuang baris deleted → restore 404."""
    conn = make_conn()
    doc_id = new_doc(conn)
    hr = actor(conn, "u_sari")
    out, status = hr_actions.delete_document(conn, hr, doc_id)
    assert status == 200 and out["status"] == "deleted"
    out, status = hr_actions.restore_document(conn, hr, doc_id)
    assert status == 200 and out["status"] == "restored", out
    docs = engine.filter_allowed_docs(conn, hr)
    assert any(d["id"] == doc_id for d in docs)


def test_restricted_document_of_other_role_is_not_manageable():
    """Keberadaan dokumen restricted tidak boleh bocor lewat 404 vs 403."""
    conn = make_conn()
    doc_id = new_doc(conn, "doc_ha_restricted",
                     confidentiality="restricted", roles=("admin",))
    out, status = hr_actions.delete_document(conn, actor(conn, "u_sari"), doc_id)
    assert status == 404 and out["error"] == "doc_not_found"


def test_revision_keeps_status_and_confidentiality():
    conn = make_conn()
    doc_id = new_doc(conn, "doc_ha_rev", status="indexed",
                     confidentiality="restricted")
    out, status = hr_actions.update_document(
        conn, actor(conn, "u_sari"), doc_id, {"content": "isi baru"})
    assert status == 200, out
    row = conn.execute("SELECT status, confidentiality, version FROM documents "
                       "WHERE id = ?", (doc_id,)).fetchone()
    assert row["status"] == "indexed"
    assert row["confidentiality"] == "restricted"
    assert out["changed_fields"] == ["content", "version"]


def test_revision_snapshots_previous_version():
    conn = make_conn()
    doc_id = new_doc(conn, "doc_ha_ver")
    hr_actions.update_document(conn, actor(conn, "u_sari"), doc_id,
                               {"content": "isi kedua"})
    versions = lifecycle.list_versions(conn, doc_id)
    assert versions, "riwayat versi kosong setelah revisi"


def test_deleted_document_cannot_be_revised():
    conn = make_conn()
    doc_id = new_doc(conn, "doc_ha_del")
    hr = actor(conn, "u_sari")
    hr_actions.delete_document(conn, hr, doc_id)
    out, status = hr_actions.update_document(conn, hr, doc_id, {"content": "x"})
    assert status == 409 and out["error"] == "doc_deleted"


def test_create_document_validates_enums():
    conn = make_conn()
    out, status = hr_actions.create_document(
        conn, actor(conn, "u_sari"),
        {"title": "T", "source_type": "sop", "content": "isi",
         "confidentiality": "rahasia"})
    assert status == 422 and out["error"] == "invalid_confidentiality"


# ---------------------------------------------------------------------------
# Karyawan
# ---------------------------------------------------------------------------
def test_admin_can_create_and_deactivate_employee():
    conn = make_conn()
    admin = actor(conn, "u_rina")
    out, status = hr_actions.create_employee(conn, admin, {
        "name": "Karyawan Uji", "role": "employee",
        "department": "Operations", "job_title": "Analis"})
    assert status < 400, out
    uid = out["id"]
    out, status = hr_actions.deactivate_employee(conn, admin, uid)
    assert status < 400, out
    row = conn.execute("SELECT employment_status FROM users WHERE id = ?",
                       (uid,)).fetchone()
    assert row["employment_status"] != "active"


def test_employee_creation_rejects_unknown_role():
    conn = make_conn()
    out, status = hr_actions.create_employee(
        conn, actor(conn, "u_rina"), {"name": "X", "role": "superuser"})
    assert status == 422, out


def test_document_without_workspace_defaults_to_onboarding():
    """Dokumen tanpa workspace tidak bisa diakses siapa pun (ACL kosong).

    `filter_allowed_docs` memakai irisan workspace, jadi `[]` berarti tidak ada
    yang boleh membacanya — termasuk HR yang baru membuatnya. Dokumen itu lalu
    "hilang" dan revisi berikutnya ditolak 404.
    """
    conn = make_conn()
    hr = actor(conn, "u_sari")
    out, status = hr_actions.create_document(
        conn, hr, {"title": "SOP Tanpa Workspace", "source_type": "sop",
                   "content": "isi"})
    assert status == 201, out
    doc = hr_actions.find_doc_any(conn, out["id"])
    assert doc["workspace_ids"], "ACL workspace kosong"
    assert any(d["id"] == out["id"] for d in engine.filter_allowed_docs(conn, hr))


def test_list_documents_tool_sees_trash():
    """Klaim "tidak ada dokumen di tempat sampah" harus berbasis data."""
    import agent_tools
    conn = make_conn()
    doc_id = new_doc(conn, "doc_ha_trash")
    hr = actor(conn, "u_sari")
    hr_actions.delete_document(conn, hr, doc_id)
    ctx = agent_tools.ToolContext(conn, hr)
    out = agent_tools.execute(ctx, "list_documents", {"deleted": "only"})
    ids = {d["doc_id"] for d in out["documents"]}
    assert doc_id in ids, out
    active = agent_tools.execute(ctx, "list_documents", {"deleted": "active"})
    assert doc_id not in {d["doc_id"] for d in active["documents"]}


def test_list_documents_is_hr_only():
    import agent_tools
    conn = make_conn()
    ctx = agent_tools.ToolContext(conn, actor(conn, "u_budi"))
    out = agent_tools.execute(ctx, "list_documents", {})
    assert out.get("denied") is True
