"""D5 — daftar dokumen tidak boleh membocorkan PII atau melewati audit.

Bug: `GET /api/documents` mengirim `content` MENTAH untuk setiap baris,
sementara `GET /api/documents/{id}` menyamarkannya lewat `mask_pii` dan menulis
`doc_view` ke audit. Membaca isi lewat daftar berarti melewati dua kontrol
sekaligus:

    LIST   -> "…telepon 0812-3456-7890 atau email sari@nusantarakreatif.id…"
    DETAIL -> "…telepon 0812****890 atau email s***@nusantarakreatif.id…"

Form edit tetap butuh isi asli. Karena itu ada jalur terpisah `?raw=1` yang
hanya untuk pengelola (hr/admin), hanya untuk dokumen yang boleh dikelola, dan
tercatat sebagai `doc_edit_open`.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "doc_list_privacy.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402
from authutil import login_headers  # noqa: E402

client = TestClient(appmod.app)
H_HR = login_headers(client, "u_sari")
H_ADMIN = login_headers(client, "u_rina")
H_EMP = login_headers(client, "u_budi")

DOC = "doc_hr_contacts"          # berisi telepon + email pribadi
PHONE = "0812-3456-7890"
EMAIL = "sari@nusantarakreatif.id"


def _row(doc_id, headers=H_HR):
    body = client.get("/api/documents?limit=200&q=" + doc_id, headers=headers).json()
    return next(d for d in body["documents"] if d["id"] == doc_id)


def test_list_never_sends_raw_content():
    row = _row(DOC)
    assert "content" not in row, "daftar masih mengirim isi mentah"
    assert PHONE not in str(row), "nomor telepon asli bocor lewat daftar"
    assert EMAIL not in str(row), "email asli bocor lewat daftar"


def test_list_excerpt_is_masked():
    row = _row(DOC)
    assert row["excerpt"], "excerpt hilang — tabel admin jadi kosong"
    assert PHONE not in row["excerpt"]
    assert "0812****890" in row["excerpt"], row["excerpt"]
    assert "s***@nusantarakreatif.id" in row["excerpt"], row["excerpt"]


def test_list_reports_content_length():
    """Tabel tetap bisa menampilkan ukuran isi tanpa menerima isinya."""
    row = _row(DOC)
    assert row["content_chars"] > 0
    assert row["content_chars"] == len(appmod.CONN.execute(
        "SELECT content FROM documents WHERE id=?", (DOC,)).fetchone()["content"])


def test_detail_still_masks_and_audits():
    r = client.get("/api/documents/" + DOC, headers=H_HR)
    assert r.status_code == 200
    assert PHONE not in r.json()["content"]
    events = [x.get("event_type")
              for x in client.get("/api/audit?limit=100", headers=H_ADMIN).json()["audit"]]
    assert "doc_view" in events


def test_raw_preview_returns_original_for_managers():
    r = client.get("/api/documents/" + DOC + "?raw=1", headers=H_HR)
    assert r.status_code == 200, r.text
    assert PHONE in r.json()["content"], "form edit butuh isi asli"
    assert EMAIL in r.json()["content"]


def test_raw_preview_is_audited_separately():
    before = [x.get("event_type")
              for x in client.get("/api/audit?limit=200", headers=H_ADMIN).json()["audit"]]
    client.get("/api/documents/" + DOC + "?raw=1", headers=H_HR)
    after = [x.get("event_type")
             for x in client.get("/api/audit?limit=200", headers=H_ADMIN).json()["audit"]]
    assert after.count("doc_edit_open") == before.count("doc_edit_open") + 1


def test_raw_preview_forbidden_for_employee():
    r = client.get("/api/documents/" + DOC + "?raw=1", headers=H_EMP)
    assert r.status_code == 403


def test_raw_preview_404_for_doc_outside_scope():
    """HR tidak boleh membuka isi asli dokumen restricted milik team_lead."""
    r = client.get("/api/documents/doc_project_kenari?raw=1", headers=H_HR)
    assert r.status_code == 404
    assert "850" not in r.text


def test_edit_roundtrip_preserves_pii():
    """Alur nyata form edit: raw → ubah judul → simpan. PII tidak boleh rusak."""
    raw = client.get("/api/documents/" + DOC + "?raw=1", headers=H_HR).json()
    payload = {"title": raw["title"], "content": raw["content"] + " Baris tambahan."}
    r = client.put("/api/documents/" + DOC, json=payload, headers=H_HR)
    assert r.status_code == 200, r.text
    stored = appmod.CONN.execute(
        "SELECT content FROM documents WHERE id=?", (DOC,)).fetchone()["content"]
    assert PHONE in stored, "PII tertimpa versi tersamar saat menyimpan"
    assert stored.endswith("Baris tambahan.")
