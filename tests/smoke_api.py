"""Smoke test for the FastAPI layer via TestClient (uses a temp DB)."""
import os
import sys
import io
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

# run against a throwaway DB
tmp = tempfile.mkdtemp()
os.environ["NOHALU_DB_PATH"] = os.path.join(tmp, "smoke.db")
# keep the smoke test offline/deterministic
os.environ["NOHALU_LLM_ENABLED"] = "0"

from fastapi.testclient import TestClient  # noqa: E402
import app  # noqa: E402

c = TestClient(app.app)


def j(resp):
    return resp.status_code, resp.json()


def main():
    assert j(c.get("/api/health"))[1]["status"] == "ok"
    # login Budi
    s_budi = j(c.post("/api/auth/login", json={"user_id": "u_budi"}))[1]
    h_budi = {"Authorization": f"Bearer {s_budi['token']}"}
    # chat
    r = j(c.post("/api/chat", json={"question": "Berapa tunjangan transport bulanan?"}, headers=h_budi))[1]
    assert "750.000" in r["answer"], r["answer"]
    assert len(r["citations"]) > 0
    assert r.get("audit_id") and r.get("limitations")
    # refusal (Kenari)
    r2 = j(c.post("/api/chat", json={"question": "Apa status Proyek Kenari?"}, headers=h_budi))[1]
    assert r2["refusal_reason"] == "no_access"
    # admin
    tok_admin = j(c.post("/api/auth/login", json={"user_id": "u_rina"}))[1]["token"]
    h_admin = {"Authorization": f"Bearer {tok_admin}"}
    assert j(c.get("/api/documents", headers=h_admin))[0] == 200
    assert j(c.get("/api/users", headers=h_admin))[0] == 200
    # permission grant then re-ask Kenari
    j(c.put("/api/users/u_budi/permissions",
            json={"allowed_workspaces": ["ws_onboarding", "ws_project_alpha", "ws_project_kenari"],
                  "role": "team_lead"}, headers=h_admin))
    r3 = j(c.post("/api/chat", json={"question": "Apa status Proyek Kenari?"}, headers=h_budi))[1]
    assert "850" in r3["answer"], r3["answer"]
    # restore Budi
    j(c.put("/api/users/u_budi/permissions",
            json={"allowed_workspaces": ["ws_onboarding", "ws_project_alpha"],
                  "role": "employee"}, headers=h_admin))
    # RBAC: Budi cannot list documents
    assert j(c.get("/api/documents", headers=h_budi))[0] == 403

    # ---- upload + parse + immediate retrieval ----
    # CSV
    csv_bytes = b"topik,nilai\nTunjangan makan siang,Rp200.000\n"
    up = c.post("/api/documents/upload",
                files={"file": ("makan.csv", csv_bytes, "text/csv")},
                data={"title": "Tunjangan Makan Siang", "source_type": "csv",
                      "effective_date": "2026-10-01", "workspace_ids": "ws_onboarding",
                      "access_roles": "employee,hr,team_lead,admin"}, headers=h_admin)
    assert up.status_code == 200, up.text
    assert up.json()["status"] == "indexed"
    rm = j(c.post("/api/chat", json={"question": "Berapa tunjangan makan siang?"}, headers=h_budi))[1]
    assert "200.000" in rm["answer"], rm["answer"]

    # DOCX
    import docx
    buf = io.BytesIO()
    d = docx.Document()
    d.add_paragraph("Kebijakan parkir: gratis untuk seluruh karyawan.")
    d.save(buf)
    up = c.post("/api/documents/upload",
                files={"file": ("parkir.docx", buf.getvalue(),
                                "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                data={"title": "Kebijakan Parkir", "source_type": "docx",
                      "effective_date": "2026-10-02", "workspace_ids": "ws_onboarding",
                      "access_roles": "employee,hr,team_lead,admin"}, headers=h_admin)
    assert up.status_code == 200, up.text
    rp = j(c.post("/api/chat", json={"question": "Bagaimana kebijakan parkir?"}, headers=h_budi))[1]
    assert "parkir" in rp["answer"].lower(), rp["answer"]

    # PDF
    try:
        from reportlab.pdfgen import canvas
        pdf_buf = io.BytesIO()
        cv = canvas.Canvas(pdf_buf)
        cv.drawString(72, 720, "Tunjangan internet Rp150.000 per bulan")
        cv.save()
        up = c.post("/api/documents/upload",
                    files={"file": ("internet.pdf", pdf_buf.getvalue(), "application/pdf")},
                    data={"title": "Tunjangan Internet", "source_type": "pdf",
                          "effective_date": "2026-10-03", "workspace_ids": "ws_onboarding",
                          "access_roles": "employee,hr,team_lead,admin"}, headers=h_admin)
        assert up.status_code == 200, up.text
        ri = j(c.post("/api/chat", json={"question": "Berapa tunjangan internet?"}, headers=h_budi))[1]
        assert "150.000" in ri["answer"], ri["answer"]
    except ImportError:
        print("  (pdf upload skipped: reportlab not installed)")

    # unsupported format -> 400
    bad = c.post("/api/documents/upload",
                 files={"file": ("x.xlsx", b"PK\x03\x04", "application/vnd.ms-excel")},
                 data={"title": "x"}, headers=h_admin)
    assert bad.status_code == 400, bad.text

    # non-admin upload -> 403
    denied = c.post("/api/documents/upload",
                    files={"file": ("x.txt", b"hello", "text/plain")},
                    data={"title": "x"}, headers=h_budi)
    assert denied.status_code == 403, denied.text

    # audit
    aud = j(c.get("/api/audit", headers=h_admin))[1]
    assert aud["audit"], "audit must have entries"
    assert any(a.get("engine_used") for a in aud["audit"])

    print("API SMOKE TEST PASS — endpoints, RBAC, grant, upload/parse, audit all OK")


def finance_flow_check():
    """Role finance: approver reimbursement tahap akhir (regresi)."""
    tok = j(c.post("/api/auth/login", json={"user_id": "u_dewi"}))[1]["token"]
    h = {"Authorization": f"Bearer {tok}"}
    # login + dashboard
    me = j(c.get("/api/me", headers=h))[1]
    assert me["role"] == "finance", me
    d = j(c.get("/api/dashboard", headers=h))[1]
    assert "approval_waiting" in d
    # payroll: jadwal terlihat, struktur gaji tidak
    p = j(c.get("/api/payroll", headers=h))[1]
    assert p["schedule"] and p["salary_structure"] is None
    # buat reimbursement -> team lead approve -> finance approve
    budi = j(c.post("/api/auth/login", json={"user_id": "u_budi"}))[1]["token"]
    hb = {"Authorization": f"Bearer {budi}"}
    req = j(c.post("/api/requests", json={
        "type": "reimbursement", "amount": "2500000", "note": "smoke"}, headers=hb))[1]
    assert req["stages"] == ["team_lead", "finance"], req
    rid = req["id"]
    andi = j(c.post("/api/auth/login", json={"user_id": "u_andi"}))[1]["token"]
    ha = {"Authorization": f"Bearer {andi}"}
    r1 = j(c.post(f"/api/requests/{rid}/decide", json={"action": "approve", "note": "tl"}, headers=ha))[1]
    assert r1["status"] == "pending" and r1["stage_idx"] == 1, r1
    r2 = j(c.post(f"/api/requests/{rid}/decide", json={"action": "approve", "note": "fin"}, headers=h))[1]
    assert r2["status"] == "approved", r2
    print("FINANCE APPROVAL FLOW PASS")


if __name__ == "__main__":
    main()
    finance_flow_check()
