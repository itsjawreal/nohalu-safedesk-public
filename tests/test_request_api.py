"""Sprint 0 — kontrak HTTP endpoint pengajuan (status code + detail terstruktur).

Menguji lewat FastAPI TestClient supaya format error yang dilihat UI ikut
terverifikasi: `{"detail": {"code", "message", "fields"}}`.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

_TMP = tempfile.mkdtemp()
os.environ["NOHALU_DB_PATH"] = os.path.join(_TMP, "api.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app as appmod  # noqa: E402

client = TestClient(appmod.app)


def _h(uid):
    from authutil import login_headers
    return login_headers(client, uid)


def _post(path, uid, body=None):
    return client.post(path, json=body or {}, headers=_h(uid))


# ---------------------------------------------------------------------------
# Pratinjau jalur persetujuan (dipakai formulir sebelum submit)
# ---------------------------------------------------------------------------
def test_rules_preview_matches_what_submit_creates():
    """Pratinjau formulir harus sama persis dengan tahap yang benar-benar dibuat.

    Kalau keduanya berbeda, karyawan melihat "Team Lead lalu HR" di formulir
    dan menerima rantai lain setelah dikirim — persis kelas bug yang membuat
    approver menunggu di tahap yang tidak pernah ada.
    """
    r = client.get("/api/requests/rules?type=cuti", headers=_h("u_budi"))
    assert r.status_code == 200, r.text
    preview = r.json()
    assert preview["stages"] == ["team_lead", "hr"]
    assert preview["auto_approved"] is False

    created = _post("/api/requests", "u_budi",
                    {"type": "cuti", "start_date": "2027-03-01", "end_date": "2027-03-02"})
    assert created.status_code == 201, created.text
    assert created.json()["stages"] == preview["stages"]


def test_rules_preview_threshold_split():
    """Di bawah/tepat ambang satu tahap; di atas ambang dua tahap."""
    below = client.get("/api/requests/rules?type=reimbursement&amount=120000",
                       headers=_h("u_budi")).json()
    assert below["stages"] == ["team_lead"]
    assert below["finance_manager_required"] is False

    at = client.get("/api/requests/rules?type=reimbursement&amount=1000000",
                    headers=_h("u_budi")).json()
    assert at["stages"] == ["team_lead"], "tepat di ambang belum lewat Finance"

    above = client.get("/api/requests/rules?type=reimbursement&amount=1000001",
                       headers=_h("u_budi")).json()
    assert above["stages"] == ["team_lead", "finance"]
    assert above["finance_manager_required"] is True


def test_rules_preview_drops_self_approval_stage():
    """Team Lead yang mengajukan cuti tidak boleh melihat dirinya sebagai approver."""
    body = client.get("/api/requests/rules?type=cuti", headers=_h("u_andi")).json()
    assert body["stages"] == ["hr"]
    assert body["skipped"] == ["team_lead"]

    # Finance mengajukan klaim di atas ambang: tahap finance-nya dilewati.
    fin = client.get("/api/requests/rules?type=reimbursement&amount=2500000",
                     headers=_h("u_dewi")).json()
    assert fin["stages"] == ["team_lead"]
    assert fin["skipped"] == ["finance"]
    assert fin["finance_manager_required"] is True


def test_rules_preview_rejects_bad_input():
    bad_type = client.get("/api/requests/rules?type=liburan", headers=_h("u_budi"))
    assert bad_type.status_code == 400
    assert bad_type.json()["detail"]["code"] == "invalid_type"

    bad_amount = client.get("/api/requests/rules?type=reimbursement&amount=abc",
                            headers=_h("u_budi"))
    assert bad_amount.status_code == 422
    assert bad_amount.json()["detail"]["code"] == "amount_invalid"


# ---------------------------------------------------------------------------
# Self-approval lewat HTTP (probe A/B/C dari audit)
# ---------------------------------------------------------------------------
def test_http_hr_leave_self_stage_skipped():
    r = _post("/api/requests", "u_sari",
              {"type": "cuti", "start_date": "2027-01-05", "end_date": "2027-01-06"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["skipped_stages"] == ["hr"]
    rid = body["id"]
    r2 = _post(f"/api/requests/{rid}/decide", "u_andi",
               {"action": "approve", "note": "ok"})
    assert r2.status_code == 200, r2.text
    assert r2.json()["status"] == "approved"


def test_http_team_lead_leave_self_stage_skipped():
    r = _post("/api/requests", "u_andi",
              {"type": "cuti", "start_date": "2027-02-05", "end_date": "2027-02-06"})
    assert r.status_code == 201
    rid = r.json()["id"]
    r2 = _post(f"/api/requests/{rid}/decide", "u_sari",
               {"action": "approve", "note": "ok"})
    assert r2.status_code == 200 and r2.json()["status"] == "approved"


# ---------------------------------------------------------------------------
# Validasi: status code + detail terstruktur
# ---------------------------------------------------------------------------
def test_http_validation_error_shape():
    r = _post("/api/requests", "u_dimas",
              {"type": "cuti", "start_date": "2027-01-10", "end_date": "2027-01-05"})
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail["code"] == "end_before_start"
    assert detail["message"]
    assert "end_date" in detail["fields"]


def test_http_invalid_amount():
    for amount in ("-500000", "abc", "0"):
        r = _post("/api/requests", "u_dimas",
                  {"type": "reimbursement", "amount": amount})
        assert r.status_code == 422, (amount, r.text)
        assert r.json()["detail"]["code"] == "amount_invalid"


def test_http_cancel_flow():
    r = _post("/api/requests", "u_dimas",
              {"type": "reimbursement", "amount": "42000"})
    rid = r.json()["id"]
    r2 = client.post(f"/api/requests/{rid}/cancel", headers=_h("u_dimas"))
    assert r2.status_code == 200, r2.text
    assert r2.json()["status"] == "cancelled"
    r3 = _post(f"/api/requests/{rid}/decide", "u_andi", {"action": "approve"})
    assert r3.status_code == 400
    assert r3.json()["detail"]["code"] == "already_decided"


def test_http_override_permissions():
    r = _post("/api/requests", "u_dimas",
              {"type": "reimbursement", "amount": "43000"})
    rid = r.json()["id"]
    r2 = _post(f"/api/requests/{rid}/override", "u_andi",
               {"action": "approve", "note": "alasan yang cukup panjang"})
    assert r2.status_code == 403
    r3 = _post(f"/api/requests/{rid}/override", "u_rina",
               {"action": "approve", "note": "pendek"})
    assert r3.status_code == 422 and r3.json()["detail"]["code"] == "note_required"
    r4 = _post(f"/api/requests/{rid}/override", "u_rina",
               {"action": "approve", "note": "Approver tidak tersedia hari ini."})
    assert r4.status_code == 200 and r4.json()["status"] == "approved"


def test_http_idempotent_client_ref():
    payload = {"type": "reimbursement", "amount": "55000",
               "client_ref": "http-idem-1"}
    r1 = _post("/api/requests", "u_budi", payload)
    r2 = _post("/api/requests", "u_budi", payload)
    assert r1.status_code == 201 and r2.status_code == 200
    assert r1.json()["id"] == r2.json()["id"]


def test_http_rate_limit_requests():
    codes = [_post("/api/requests", "u_dewi",
                   {"type": "reimbursement", "amount": "1000"}).status_code
             for _ in range(appmod.REQUEST_RATE_LIMIT + 3)]
    assert 429 in codes, codes
    assert codes.count(429) >= 3


def test_audit_does_not_ship_full_allowed_sources():
    """Daftar izin dokumen tidak dikirim per baris.

    `allowed_sources` memuat 50+ id per baris dan halaman Audit tidak pernah
    menampilkannya — dulu 111 KB dari 236 KB respons. Yang dikirim hanya
    jumlahnya; daftar penuh tetap tersedia lewat `?include_sources=true`.
    """
    r = client.get("/api/audit", headers=_h("u_rina"))
    assert r.status_code == 200
    rows = r.json()["audit"]
    assert rows
    assert "allowed_sources" not in rows[0], "daftar penuh tidak boleh ikut"
    assert isinstance(rows[0].get("allowed_count"), int)
    # selected_sources tetap ada: itu yang dipakai halaman.
    assert "selected_sources" in rows[0]

    full = client.get("/api/audit?include_sources=true", headers=_h("u_rina"))
    assert full.status_code == 200
    assert isinstance(full.json()["audit"][0].get("allowed_sources"), list)


def test_audit_truncates_long_answers_by_default():
    """Jawaban penuh tidak dikirim per baris; potongan ditandai eksplisit.

    Tabel Audit hanya menampilkan 160 karakter pertama. Mengirim jawaban utuh
    membuat 36% respons tidak pernah terbaca. Yang penting: klien tidak boleh
    menyangka potongan itu jawaban utuh, jadi `answer_truncated` dan
    `answer_length` ikut dikirim.
    """
    appmod.CONN.execute(
        """INSERT INTO audit_log (timestamp,user_id,user_role,question,
           permission_decision,allowed_sources,retrieved_sources,selected_sources,
           conflict_detected,answer,citations,refusal_reason,pii_masked,
           prompt_injection_detected,agent_version,event_type)
           VALUES ('2026-01-01T00:00:00Z','u_budi','employee','PANJANG',
           'allow','[]','["doc_a","doc_b"]','[]',0,?,'[]',NULL,0,0,'test','qa')""",
        ("x" * 2000,),
    )
    appmod.CONN.commit()

    rows = client.get("/api/audit?user_id=u_budi", headers=_h("u_rina")).json()["audit"]
    row = next(r for r in rows if r["question"] == "PANJANG")
    assert row["answer_truncated"] is True
    assert row["answer_length"] == 2000
    # Potongan dipotong di batas kata dan ditutup elipsis, jadi panjangnya
    # sedikit di bawah batas — bukan tepat di tengah kata.
    assert len(row["answer"]) <= appmod.AUDIT_ANSWER_PREVIEW + 1
    assert row["answer"].endswith("…")
    assert "  " not in row["answer"]
    # retrieved_sources diganti jumlahnya: UI hanya memakai selected_sources.
    assert "retrieved_sources" not in row
    assert row["retrieved_count"] == 2

    full = client.get("/api/audit?user_id=u_budi&include_answers=true",
                      headers=_h("u_rina")).json()["audit"]
    full_row = next(r for r in full if r["question"] == "PANJANG")
    assert len(full_row["answer"]) == 2000
    assert full_row["answer_truncated"] is False


def test_http_audit_event_visible_to_hr():
    r = client.get("/api/audit?status=request_create", headers=_h("u_sari"))
    assert r.status_code == 200
    rows = r.json()["audit"]
    assert rows and rows[0]["event_type"] == "request_create"
    assert rows[0]["details"]["request_id"]


def test_audit_status_filter_accepts_event_type():
    """Filter ?status=request_create harus mengembalikan event tersebut."""
    r = client.get("/api/audit?status=request_create", headers=_h("u_rina"))
    assert r.status_code == 200
    rows = r.json()["audit"]
    assert rows, "harus ada event request_create"
    assert all(a["event_type"] == "request_create" for a in rows)
    # filter verdict lama tetap bekerja
    r2 = client.get("/api/audit?status=refused", headers=_h("u_rina"))
    assert r2.status_code == 200


def test_requests_paging_is_per_bucket_and_filter_is_scoped():
    """Satu offset bersama untuk tiga keranjang salah secara diam-diam.

    Memaging tabel "Semua Pengajuan" ikut menggeser antrean "Menunggu Saya",
    sehingga kartu approval yang belum diputuskan bisa hilang dari layar.
    Filter `status` juga dulu berlaku ke semua keranjang: memilih chip
    "approved" mengosongkan antrean yang justru masih pending.
    """
    for i in range(3):
        r = _post("/api/requests", "u_budi", {
            "type": "cuti", "title": "Cuti uji %d" % i,
            "start_date": "2027-06-%02d" % (1 + i * 2),
            "end_date": "2027-06-%02d" % (2 + i * 2),
        })
        assert r.status_code == 201, r.text

    lead = _h("u_andi")
    base = client.get("/api/requests", headers=lead).json()
    assert len(base["approval"]) >= 3, base["approval"]

    # Paging tabel "Semua Pengajuan" tidak boleh menggeser antrean approval.
    shifted = client.get("/api/requests?offset_all=2", headers=lead).json()
    assert [r["id"] for r in shifted["approval"]] == \
        [r["id"] for r in base["approval"]], "antrean approval ikut bergeser"
    assert shifted["pages"]["all"]["offset"] == 2
    assert shifted["pages"]["approval"]["offset"] == 0

    # Filter chip tabel "Semua" tidak boleh mengosongkan antrean pending.
    filtered = client.get("/api/requests?status_all=approved", headers=lead).json()
    assert len(filtered["approval"]) == len(base["approval"]), \
        "filter tabel ikut menyaring antrean approval"

    # Grafik sebaran memakai populasi penuh, bukan halaman/filter aktif.
    # `all` hanya berisi data untuk HR/admin (team lead tidak boleh melihat
    # pengajuan departemen lain), jadi populasi diuji dari akun HR.
    hr = _h("u_sari")
    hr_base = client.get("/api/requests", headers=hr).json()
    hr_filtered = client.get("/api/requests?status_all=approved", headers=hr).json()
    assert hr_filtered["summary"]["by_status"]["pending"] >= 3, hr_filtered["summary"]
    assert hr_filtered["summary"]["total"] == hr_base["summary"]["total"]
    assert hr_filtered["summary"]["total"] > len(hr_filtered["all"]), \
        "summary harus populasi penuh, bukan isi halaman"
