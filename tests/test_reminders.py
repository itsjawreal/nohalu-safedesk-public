"""Sprint 3.5/3.6 — reminder job (idempotent) + digest refusal HR."""
import datetime as dt
import os
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "reminders.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"

import db  # noqa: E402
import portal  # noqa: E402
import reminders  # noqa: E402
from engine import get_user  # noqa: E402


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    db.seed_if_empty(conn)
    db.ensure_seed_users(conn)
    return conn


def _msgs(conn, uid="u_andi"):
    return [r["message"] for r in conn.execute(
        "SELECT message FROM notifications WHERE user_id=?", (uid,)).fetchall()]


def test_stale_approval_reminder_is_idempotent():
    conn = make_conn()
    conn.execute(
        """INSERT INTO requests (user_id,type,title,status,approver_roles,stages,
           stage_idx,history,created_at)
           VALUES ('u_budi','cuti','Cuti keluarga','pending','["team_lead"]',
                   '["team_lead"]',0,'[]',?)""",
        ((dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=48))
         .isoformat().replace("+00:00", "Z"),),
    )
    conn.commit()
    now = dt.datetime.now(dt.timezone.utc)
    first = reminders.job_stale_approvals(conn, now=now)
    second = reminders.job_stale_approvals(conn, now=now)
    assert first >= 1
    assert second == 0, "job harus idempotent dalam hari yang sama"
    assert any("Pengingat: pengajuan" in m for m in _msgs(conn))


def test_stale_approval_ignored_when_fresh():
    conn = make_conn()
    conn.execute(
        """INSERT INTO requests (user_id,type,title,status,approver_roles,stages,
           stage_idx,history,created_at)
           VALUES ('u_budi','cuti','Cuti baru','pending','["team_lead"]',
                   '["team_lead"]',0,'[]',?)""",
        (dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),),
    )
    conn.commit()
    assert reminders.job_stale_approvals(conn) == 0


def test_leave_starting_soon_notifies_requester_and_approver():
    conn = make_conn()
    start = portal._today() + dt.timedelta(days=1)
    conn.execute(
        """INSERT INTO requests (user_id,type,title,start_date,status,approver_roles,
           stages,stage_idx,history,created_at)
           VALUES ('u_budi','cuti','Cuti besok',?,'pending','["team_lead"]',
                   '["team_lead"]',0,'[]',?)""",
        (start.isoformat(), portal._now()),
    )
    conn.commit()
    assert reminders.job_leave_starting(conn) >= 2  # pengaju + team lead
    assert reminders.job_leave_starting(conn) == 0
    assert any("dijadwalkan mulai" in m for m in _msgs(conn, "u_budi"))


def test_new_payslip_reminder_once():
    conn = make_conn()
    conn.execute(
        """INSERT OR REPLACE INTO payslips (user_id,period,gross,deductions,net,
           published_at) VALUES ('u_budi','2026-10',1000,100,900,?)""",
        (dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),),
    )
    conn.commit()
    assert reminders.job_new_payslips(conn) >= 1
    assert reminders.job_new_payslips(conn) == 0
    assert any("Slip gaji periode 2026-10" in m for m in _msgs(conn, "u_budi"))


def test_run_once_isolates_job_failures(monkeypatch):
    conn = make_conn()

    def boom(*a, **k):
        raise RuntimeError("gagal")

    monkeypatch.setattr(reminders, "job_stale_approvals", boom)
    out = reminders.run_once(conn)
    assert out["job_stale_approvals"] == -1
    assert out["job_leave_starting"] >= 0


def test_publish_payslip_then_job_does_not_duplicate():
    """Penerbitan slip langsung + job reminder tidak boleh dobel."""
    import personal

    conn = make_conn()
    hr = get_user(conn, "u_sari")
    data, code = personal.publish_payslip(
        conn, hr, {"user_id": "u_budi", "period": "2026-11", "gross": 1000,
                   "deductions": 100})
    assert code == 201
    assert reminders.job_new_payslips(conn) == 0
    rows = conn.execute(
        "SELECT COUNT(*) AS n FROM notifications WHERE user_id='u_budi' "
        "AND category='payslip'").fetchone()
    assert rows["n"] == 1


def test_new_announcement_notifies_only_allowed_users():
    conn = make_conn()
    # Doc announcement khusus admin (workspace finance + role admin).
    conn.execute(
        """INSERT OR REPLACE INTO documents
           (id,title,source_type,effective_date,workspace_ids,access_roles,content,
            owner,confidentiality,status,version)
           VALUES ('doc_ann_test','Uji Pengumuman','hr_announcement',?,?,?,
                   'isi','HR','internal','indexed','1')""",
        (portal._today().isoformat(), '["ws_finance"]', '["admin"]'),
    )
    # Isolasi: hapus pengumuman seed lain agar hitungan hanya dari doc uji.
    conn.execute("DELETE FROM documents WHERE source_type=? AND id<>?",
                 (portal.ANNOUNCEMENT_SOURCE, "doc_ann_test"))
    conn.commit()
    assert reminders.job_new_announcements(conn) == 1, "hanya admin yang berhak"
    assert reminders.job_new_announcements(conn) == 0
    admin_msgs = conn.execute(
        "SELECT COUNT(*) AS n FROM notifications WHERE user_id='u_rina' "
        "AND category='announcement'").fetchone()["n"]
    emp_msgs = conn.execute(
        "SELECT COUNT(*) AS n FROM notifications WHERE user_id='u_budi' "
        "AND category='announcement'").fetchone()["n"]
    assert admin_msgs >= 1
    assert emp_msgs == 0, "user tanpa akses tidak boleh menerima pengumuman"
    # `ref` wajib terisi agar membaca satu pengumuman bisa membersihkan
    # notifikasinya secara presisi (bukan semua notifikasi pengumuman).
    refs = {r["ref"] for r in conn.execute(
        "SELECT ref FROM notifications WHERE user_id='u_rina' "
        "AND category='announcement'").fetchall()}
    assert refs == {"doc_ann_test"}, refs


def test_admin_endpoint_runs_jobs_and_rejects_non_admin():
    """Endpoint manual: admin-only, idempotent, memakai lock yang sama.

    Dijalankan di subprocess supaya tidak perlu me-reload modul global
    (reload `db`/`portal` di tengah sesi pytest mengacaukan DB test lain).
    """
    import subprocess
    import textwrap

    code = textwrap.dedent("""
        import os, sys, tempfile
        sys.path.insert(0, %r)
        os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "rem_api.db")
        os.environ["NOHALU_LLM_ENABLED"] = "0"
        os.environ["NOHALU_REMINDERS"] = "0"
        from fastapi.testclient import TestClient
        import app as appmod
        c = TestClient(appmod.app)
        def h(uid):
            r = c.post("/api/auth/login", json={"user_id": uid})
            assert r.status_code == 200, r.text
            return {"Authorization": "Bearer " + r.json()["token"]}
        emp = c.post("/api/admin/reminders/run", headers=h("u_budi"))
        adm = c.post("/api/admin/reminders/run", headers=h("u_rina"))
        assert emp.status_code == 403, emp.status_code
        assert adm.status_code == 200, adm.text
        jobs = adm.json()["jobs"]
        assert "job_stale_approvals" in jobs and "job_new_payslips" in jobs, jobs
        print("OK", sorted(jobs))
    """ % os.path.join(ROOT, "backend"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "OK" in out.stdout


def test_status_reports_last_run_and_result():
    conn = make_conn()
    before = reminders.status()
    assert before["running"] is False  # NOHALU_REMINDERS=0 di test
    reminders.run_once(conn)
    after = reminders.status()
    assert after["runs"] == before["runs"] + 1
    assert after["last_run"] and after["last_result"]
    assert set(after["last_result"]) == set(reminders.JOBS)


def test_leave_reminder_splits_request_and_approval_categories():
    """Approver memakai kategori `approval`, pengaju `request` (preferensi beda)."""
    conn = make_conn()
    start = portal._today() + dt.timedelta(days=2)
    conn.execute(
        """INSERT INTO requests (user_id,type,title,start_date,status,approver_roles,
           stages,stage_idx,history,created_at)
           VALUES ('u_budi','cuti','Cuti dua hari lagi',?,'pending','["team_lead"]',
                   '["team_lead"]',0,'[]',?)""",
        (start.isoformat(), portal._now()),
    )
    conn.commit()
    reminders.job_leave_starting(conn)
    rows = conn.execute(
        "SELECT user_id, category FROM notifications WHERE message LIKE 'Cuti #%'"
    ).fetchall()
    cats = {(r["user_id"], r["category"]) for r in rows}
    assert ("u_budi", "request") in cats
    assert ("u_andi", "approval") in cats


def test_purge_sessions_drops_only_dead_rows():
    """Tabel `sessions` append-only: baris mati harus dipangkas job.

    Terukur di produksi: 1858 baris dalam ~2 hari (866 kedaluwarsa, 891
    dicabut) dan tidak ada job pembersih sama sekali. `_session_live()`
    dipanggil setiap request ber-token, jadi tabel yang tumbuh selamanya
    memperlambat semua orang.
    """
    conn = make_conn()
    now = reminders._utcnow()
    iso = lambda d: d.isoformat()
    rows = [
        ("h_live", "u_budi", now + dt.timedelta(hours=5), None),
        ("h_expired", "u_budi", now - dt.timedelta(hours=1), None),
        ("h_revoked_old", "u_budi", now + dt.timedelta(hours=5),
         now - dt.timedelta(days=40)),
        ("h_revoked_new", "u_budi", now + dt.timedelta(hours=5),
         now - dt.timedelta(days=1)),
    ]
    for h, uid, exp, rev in rows:
        conn.execute(
            """INSERT INTO sessions (token_hash,user_id,issued_at,expires_at,revoked_at)
               VALUES (?,?,?,?,?)""",
            (h, uid, iso(now - dt.timedelta(days=1)), iso(exp),
             iso(rev) if rev else None))
    conn.commit()

    removed = reminders.job_purge_sessions(conn, now=now)
    left = {r["token_hash"] for r in conn.execute(
        "SELECT token_hash FROM sessions").fetchall()}
    assert removed == 2, removed
    assert left == {"h_live", "h_revoked_new"}, left
