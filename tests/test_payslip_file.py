"""P1 — lampirkan & unduh berkas slip gaji.

Sebelum ini `payslips.file_ref` ada di DB tetapi tidak pernah bisa diisi dari
UI dan tidak ada endpoint unduh: HR menerbitkan slip berupa angka saja, dan
karyawan tidak punya dokumen resmi untuk diunduh.
"""
import io
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

_TMP = tempfile.mkdtemp()
os.environ["NOHALU_DB_PATH"] = os.path.join(_TMP, "slip.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402
from authutil import login_headers  # noqa: E402

client = TestClient(appmod.app)
H_HR = login_headers(client, "u_sari")
H_ADMIN = login_headers(client, "u_rina")
H_BUDI = login_headers(client, "u_budi")
H_DEWI = login_headers(client, "u_dewi")

PERIOD = "2026-10"


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_rate_buckets():
    """Kosongkan bucket rate-limit sebelum tiap tes di modul ini.

    Modul ini menerbitkan banyak slip dari akun HR yang sama dalam satu menit
    (kuota produksi 10/menit). Tanpa reset, tes ke-9 gagal 429 karena sisa
    kuota tes sebelumnya — bukan karena produk salah. Pengujian rate-limit
    sendiri ada di `test_request_api.py`/`test_login_ratelimit.py`.
    """
    import ratelimit
    ratelimit.reset()
    yield
    ratelimit.reset()


def _publish(uid="u_budi"):
    r = client.post("/api/payslips", json={
        "user_id": uid, "period": PERIOD, "gross": 9000000,
        "deductions": 500000, "net": 8500000}, headers=H_HR)
    assert r.status_code in (200, 201), r.text
    return r.json()


def _attach(headers=H_HR, uid="u_budi", data=b"%PDF-1.4 isi slip"):
    return client.post("/api/payslips/file",
                       files={"file": ("slip-oktober.pdf", data, "application/pdf")},
                       data={"user_id": uid, "period": PERIOD}, headers=headers)


def test_publish_then_attach_then_download():
    _publish()
    r = _attach()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "attached"
    assert body["bytes"] > 0
    assert "file_ref" not in body, "path penyimpanan tidak boleh bocor ke klien"

    mine = client.get("/api/me/payslips", headers=H_BUDI).json()
    slip = next(s for s in mine["payslips"] if s["period"] == PERIOD)
    assert slip["has_file"] is True
    assert "file_ref" not in slip, "path penyimpanan tidak boleh bocor ke klien"

    dl = client.get(f"/api/me/payslips/{slip['id']}/file", headers=H_BUDI)
    assert dl.status_code == 200, dl.text
    assert dl.content == b"%PDF-1.4 isi slip"
    assert "attachment" in dl.headers["content-disposition"]


def test_other_employee_cannot_download():
    _publish()
    _attach()
    slip = next(s for s in client.get("/api/me/payslips", headers=H_BUDI)
                .json()["payslips"] if s["period"] == PERIOD)
    r = client.get(f"/api/me/payslips/{slip['id']}/file", headers=H_DEWI)
    assert r.status_code == 404


def test_employee_cannot_attach():
    _publish()
    r = _attach(headers=H_BUDI)
    assert r.status_code == 403


def test_attach_requires_existing_payslip():
    r = client.post("/api/payslips/file",
                    files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")},
                    data={"user_id": "u_budi", "period": "1999-01"}, headers=H_HR)
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "payslip_not_found"


def test_unsupported_type_rejected_and_nothing_left_on_disk():
    _publish()
    import files as files_mod
    # Pakai path DB yang benar-benar dipegang aplikasi, bukan env: modul tes
    # lain (mis. test_me_api) menimpa NOHALU_DB_PATH saat diimpor, sehingga
    # `os.environ` di sini menunjuk DB milik tes lain.
    root = files_mod.store_root(appmod.DEFAULT_DB_PATH)
    before = sum(len(f) for _, _, f in os.walk(root))
    r = client.post("/api/payslips/file",
                    files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
                    data={"user_id": "u_budi", "period": PERIOD}, headers=H_HR)
    assert r.status_code == 415
    after = sum(len(f) for _, _, f in os.walk(root))
    assert after == before, "berkas yang ditolak tidak boleh tertinggal"


def test_replacing_file_deletes_the_old_one():
    _publish()
    _attach(data=b"%PDF-1.4 versi pertama")
    _attach(data=b"%PDF-1.4 versi kedua")
    import files as files_mod
    # Pakai path DB yang benar-benar dipegang aplikasi, bukan env: modul tes
    # lain (mis. test_me_api) menimpa NOHALU_DB_PATH saat diimpor, sehingga
    # `os.environ` di sini menunjuk DB milik tes lain.
    root = files_mod.store_root(appmod.DEFAULT_DB_PATH)
    pdfs = [f for _, _, fs in os.walk(root) for f in fs if f.endswith(".pdf")]
    assert len(pdfs) >= 1
    slip = next(s for s in client.get("/api/me/payslips", headers=H_BUDI)
                .json()["payslips"] if s["period"] == PERIOD)
    dl = client.get(f"/api/me/payslips/{slip['id']}/file", headers=H_BUDI)
    assert dl.content == b"%PDF-1.4 versi kedua"


def test_download_without_file_is_404_with_reason():
    """Baris slip lama (terbit sebelum PDF otomatis) tetap harus menjawab 404
    dengan alasan yang bisa ditindaklanjuti, bukan 500 atau berkas kosong.

    Sejak penerbitan membuat PDF sendiri, kondisi "tanpa berkas" hanya muncul
    pada data lama/seed. Dibuat langsung di DB supaya jalur ini tetap teruji.
    """
    client.post("/api/payslips", json={
        "user_id": "u_maya", "period": "2026-09", "gross": 7000000,
        "deductions": 0, "net": 7000000}, headers=H_HR)
    appmod.CONN.execute("UPDATE payslips SET file_ref = NULL WHERE user_id = ? AND period = ?",
                        ("u_maya", "2026-09"))
    appmod.CONN.commit()
    slip = next(s for s in client.get("/api/me/payslips", headers=login_headers(client, "u_maya"))
                .json()["payslips"] if s["period"] == "2026-09")
    r = client.get(f"/api/me/payslips/{slip['id']}/file",
                   headers=login_headers(client, "u_maya"))
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "payslip_no_file"


def test_publishing_creates_downloadable_pdf():
    """P1: menerbitkan slip harus langsung menghasilkan dokumen resmi.

    Sebelumnya slip hanya angka; tombol Unduh selalu 404 dan HR harus
    mengunggah PDF sendiri untuk setiap karyawan setiap bulan.
    """
    from pypdf import PdfReader
    client.post("/api/payslips", json={
        "user_id": "u_raka", "period": "2026-07", "gross": 8000000,
        "deductions": 900000, "net": 7100000}, headers=H_HR)
    h_raka = login_headers(client, "u_raka")
    slip = next(s for s in client.get("/api/me/payslips", headers=h_raka)
                .json()["payslips"] if s["period"] == "2026-07")
    assert slip["has_file"] is True
    r = client.get(f"/api/me/payslips/{slip['id']}/file", headers=h_raka)
    assert r.status_code == 200, r.text
    assert r.content.startswith(b"%PDF-")
    text = PdfReader(io.BytesIO(r.content)).pages[0].extract_text()
    assert "2026-07" in text
    assert "Rp7.100.000" in text
    # Karyawan lain tidak boleh mengunduh slip ini (404, bukan 403).
    assert client.get(f"/api/me/payslips/{slip['id']}/file",
                      headers=H_BUDI).status_code == 404


def test_list_payslips_for_hr_and_forbidden_for_employee():
    """HR butuh daftar slip untuk tahu periode mana yang belum terbit.

    Tanpa endpoint ini halaman Kelola hanya bisa menerbitkan slip baru dan
    tidak pernah menampilkan apa yang sudah ada.
    """
    _publish("u_budi")
    r = client.get("/api/payslips", headers=H_HR)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] >= 1
    row = next(x for x in body["payslips"] if x["user_id"] == "u_budi")
    assert row["user_name"] == "Budi Santoso"
    assert "file_ref" not in row, "path penyimpanan tidak boleh bocor ke klien"
    assert PERIOD in body["periods"]

    # Filter periode + pencarian nama.
    only = client.get(f"/api/payslips?period={PERIOD}&q=budi", headers=H_HR).json()
    assert only["count"] == 1
    none = client.get("/api/payslips?period=1999-01", headers=H_HR).json()
    assert none["count"] == 0

    # Karyawan biasa tidak boleh membaca daftar seluruh perusahaan.
    r = client.get("/api/payslips", headers=H_BUDI)
    assert r.status_code == 403
    assert r.json()["detail"]["code"] == "not_authorized"


def test_payslip_paging_contract_and_stable_periods():
    """`count` = baris di halaman ini, `total` = setelah filter.

    Dua bug nyata yang dikunci tes ini:
    1. `personal.list_payslips` mengembalikan `count` = total, lalu `spread`
       menimpanya sehingga `count` berarti total sementara `payslips` hanya
       satu halaman — angka "tanpa berkas" di UI ikut salah saat pindah halaman.
    2. Daftar periode diambil dari hasil filter, jadi memilih satu periode
       membuat dropdown hanya berisi periode itu dan pengguna tidak bisa
       pindah periode lagi.
    """
    _publish("u_budi")
    r = client.post("/api/payslips", json={
        "user_id": "u_maya", "period": "2026-09", "gross": 8000000,
        "deductions": 0, "net": 8000000}, headers=H_HR)
    assert r.status_code in (200, 201), r.text

    page = client.get("/api/payslips?limit=1&offset=0", headers=H_HR).json()
    assert page["count"] == 1, page["count"]
    assert page["total"] >= 2, page["total"]
    assert page["has_more"] is True
    assert len(page["payslips"]) == 1

    filtered = client.get(f"/api/payslips?period={PERIOD}&limit=50", headers=H_HR).json()
    assert all(p["period"] == PERIOD for p in filtered["payslips"])
    # Periode lain tetap terlihat walau filter aktif.
    assert "2026-09" in filtered["periods"], filtered["periods"]
    assert PERIOD in filtered["periods"]

    # `without_file` dihitung atas SELURUH hasil filter, bukan satu halaman.
    # Sejak penerbitan membuat PDF otomatis, kondisi ini hanya ada pada baris
    # lama — jadi dibuat langsung di DB, lalu diperiksa lintas halaman.
    for uid in ("u_fajar", "u_nadia"):
        client.post("/api/payslips", json={
            "user_id": uid, "period": "2026-06", "gross": 5000000,
            "deductions": 0, "net": 5000000}, headers=H_HR)
    appmod.CONN.execute(
        "UPDATE payslips SET file_ref = NULL WHERE period = '2026-06'")
    appmod.CONN.commit()
    page1 = client.get("/api/payslips?limit=1&offset=0", headers=H_HR).json()
    page2 = client.get("/api/payslips?limit=1&offset=1", headers=H_HR).json()
    assert page1["without_file"] >= 2, page1
    assert page1["without_file"] == page2["without_file"], (
        "angka tanpa-berkas tidak boleh berubah saat pindah halaman")


def test_export_payslips_csv():
    _publish("u_budi")
    r = client.get("/api/payslips/export.csv", headers=H_HR)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    text = r.content.decode("utf-8-sig")
    assert text.splitlines()[0].startswith("id;user_id;nama;periode")
    assert "u_budi" in text
    r = client.get("/api/payslips/export.csv", headers=H_BUDI)
    assert r.status_code == 403
