"""Reminder proaktif in-process (Sprint 3.5).

Job berjalan di thread daemon dengan interval tetap. Aturan keras:
- tidak boleh menahan `_DB_LOCK` milik app (pakai koneksi thread-local sendiri),
- tidak boleh mematikan aplikasi (semua exception ditangkap per job),
- idempotent (dedupe_key di `notification_log`), jadi restart tidak menggandakan
  notifikasi yang sama.
"""
import datetime as _dt
import json
import os
import threading
import traceback

import portal
from db import connect_threadlocal

INTERVAL_SECONDS = int(os.environ.get("NOHALU_REMINDER_INTERVAL", "900"))  # 15 menit
STALE_APPROVAL_HOURS = int(os.environ.get("NOHALU_STALE_APPROVAL_HOURS", "24"))
LEAVE_LEAD_DAYS = int(os.environ.get("NOHALU_LEAVE_LEAD_DAYS", "3"))
# Pengumuman dengan effective_date lebih tua dari ini tidak lagi diberitakan.
ANNOUNCEMENT_WINDOW_DAYS = int(os.environ.get("NOHALU_ANNOUNCE_WINDOW_DAYS", "30"))

_started = False
_thread = None
_lock = threading.Lock()
# Status terakhir (dipakai /api/health & endpoint admin) — bukan sumber kebenaran,
# hanya observability supaya deploy bisa diverifikasi tanpa menebak.
_state = {"last_run": None, "last_result": None, "runs": 0}


def _parse(ts):
    if not ts:
        return None
    try:
        return _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def _utcnow():
    return _dt.datetime.now(_dt.timezone.utc)


def job_stale_approvals(conn, now=None) -> int:
    """Pengingat untuk pengajuan pending yang menunggu > 24 jam di tahap aktif."""
    now = now or _utcnow()
    rows = conn.execute(
        "SELECT id, type, title, created_at, stages, stage_idx FROM requests "
        "WHERE status = 'pending'"
    ).fetchall()
    sent = 0
    for r in rows:
        created = _parse(r["created_at"])
        if not created:
            continue
        age_h = (now - created).total_seconds() / 3600.0
        if age_h < STALE_APPROVAL_HOURS:
            continue
        try:
            stages = json.loads(r["stages"] or "[]")
        except Exception:
            stages = []
        idx = r["stage_idx"] or 0
        role = stages[idx] if idx < len(stages) else None
        if not role:
            continue
        holders = [u for u in portal._users_by_role(conn, role)]
        if not holders:
            continue
        days = int(age_h // 24)
        sent += portal.notify_once(
            conn, holders,
            f"Pengingat: pengajuan #{r['id']} ({r['title'] or r['type']}) "
            f"menunggu persetujuan Anda sejak {days} hari lalu.",
            link=portal.LINK_APPROVALS, category="approval",
            dedupe_key=f"stale-approval:{r['id']}:{role}:{now.date().isoformat()}",
        )
    conn.commit()
    return sent


def job_leave_starting(conn, now=None) -> int:
    """Pengingat cuti yang mulai <3 hari lagi tapi belum diputuskan."""
    now = now or _utcnow()
    today = portal._today()
    rows = conn.execute(
        "SELECT id, user_id, title, start_date, stages, stage_idx FROM requests "
        "WHERE status = 'pending' AND type = 'cuti' AND start_date IS NOT NULL"
    ).fetchall()
    sent = 0
    for r in rows:
        start = portal._parse_iso_date(r["start_date"])
        if not start:
            continue
        lead = (start - today).days
        if lead < 0 or lead > LEAVE_LEAD_DAYS:
            continue
        when = "hari ini" if lead == 0 else f"{lead} hari lagi"
        try:
            stages = json.loads(r["stages"] or "[]")
        except Exception:
            stages = []
        idx = r["stage_idx"] or 0
        role = stages[idx] if idx < len(stages) else None
        # Pengaju dapat kategori `request`, approver dapat `approval` — supaya
        # preferensi per kategori benar-benar berlaku (dulu keduanya `request`,
        # sehingga approver yang mematikan `request` kehilangan pengingat cuti).
        sent += portal.notify_once(
            conn, [r["user_id"]],
            f"Cuti #{r['id']} dijadwalkan mulai {when} ({r['start_date']}) "
            f"dan masih menunggu keputusan.",
            link=portal.LINK_REQUESTS, category="request",
            dedupe_key=f"leave-soon:{r['id']}:{today.isoformat()}",
        )
        if role:
            holders = [u for u in portal._users_by_role(conn, role)
                       if u != r["user_id"]]
            sent += portal.notify_once(
                conn, holders,
                f"Cuti #{r['id']} mulai {when} ({r['start_date']}) dan masih "
                f"menunggu keputusan Anda.",
                link=portal.LINK_APPROVALS, category="approval",
                dedupe_key=f"leave-soon-approver:{r['id']}:{today.isoformat()}",
            )
    conn.commit()
    return sent


def job_new_payslips(conn, now=None) -> int:
    """Slip gaji yang terbit dalam 24 jam terakhir dan belum diberitahukan."""
    now = now or _utcnow()
    rows = conn.execute(
        "SELECT user_id, period, published_at FROM payslips "
        "WHERE published_at IS NOT NULL"
    ).fetchall()
    sent = 0
    for r in rows:
        pub = _parse(r["published_at"])
        if not pub:
            continue
        if (now - pub).total_seconds() > 86400:
            continue
        sent += portal.notify_once(
            conn, [r["user_id"]],
            f"Slip gaji periode {r['period']} sudah tersedia.",
            link="/app/slip-gaji", category="payslip",
            dedupe_key=f"payslip:{r['user_id']}:{r['period']}",
        )
    conn.commit()
    return sent


# Nama (bukan objek fungsi) supaya monkeypatch pada modul tetap berlaku dan
# satu job gagal tidak menghentikan job lain.
def job_new_announcements(conn, now=None) -> int:
    """Beri tahu user saat ada pengumuman HR baru yang boleh mereka baca.

    RBAC dihormati: notifikasi hanya untuk user yang benar-benar boleh membuka
    dokumen (irisan workspace + role, semantik sama dengan `filter_allowed_docs`),
    bukan broadcast ke semua karyawan.

    Dokumen dipindai SEKALI lalu difilter per user. Versi lama memanggil
    `filter_allowed_docs` untuk setiap user, artinya seluruh tabel dokumen
    dibaca ulang N kali per putaran job.
    """
    from engine import ACTIVE_DOC_STATUS, effective_roles

    now = now or _utcnow()
    horizon = now - _dt.timedelta(days=ANNOUNCEMENT_WINDOW_DAYS)
    docs = []
    for row in conn.execute("SELECT * FROM documents").fetchall():
        d = {"id": row["id"], "title": row["title"],
             "source_type": row["source_type"],
             "effective_date": row["effective_date"],
             "status": row["status"] if "status" in row.keys() else "indexed",
             "workspace_ids": json.loads(row["workspace_ids"] or "[]"),
             "access_roles": json.loads(row["access_roles"] or "[]")}
        if d["source_type"] != portal.ANNOUNCEMENT_SOURCE:
            continue
        if d["status"] not in ACTIVE_DOC_STATUS:
            continue
        eff = portal._parse_iso_date(d["effective_date"])
        if eff and _dt.datetime.combine(eff, _dt.time.min,
                                        tzinfo=_dt.timezone.utc) < horizon:
            continue
        docs.append(d)
    if not docs:
        return 0

    users = conn.execute("SELECT * FROM users").fetchall()
    sent = 0
    for u in users:
        uws = set(json.loads(u["allowed_workspaces"] or "[]"))
        roles = effective_roles(u["role"])
        for d in docs:
            if not (uws & set(d["workspace_ids"]) and roles & set(d["access_roles"])):
                continue
            sent += portal.notify_once(
                conn, [u["id"]], f"Pengumuman HR baru: {d['title']}",
                link="/app/pengumuman", category="announcement",
                dedupe_key=f"ann:{d['id']}:{u['id']}", ref=d["id"],
            )
    conn.commit()
    return sent


CHAT_RETENTION_DAYS = int(os.environ.get("NOHALU_CHAT_RETENTION_DAYS", "180"))
EMPTY_SESSION_GRACE_HOURS = int(os.environ.get("NOHALU_EMPTY_SESSION_HOURS", "24"))
# Batas variabel per statement SQLite (999 di build lama) → 400 aman + cepat.
PURGE_CHUNK = 400


def job_purge_chat_sessions(conn, now=None) -> int:
    """Hapus percakapan yang sudah lewat masa retensi + sesi kosong.

    Retensi 180 hari adalah janji privasi: riwayat chat memuat pertanyaan
    pribadi karyawan (gaji, cuti, keluhan). Tanpa job ini tabel tumbuh tanpa
    batas dan data lama tersimpan selamanya.

    Sesi kosong (dibuat lalu ditinggalkan sebelum ada pesan) dibuang setelah
    `EMPTY_SESSION_GRACE_HOURS` — UI menyembunyikannya, jadi baris itu murni
    sampah yang memperlambat `GET /api/chat/sessions`.

    Batas waktu dihitung di SQL (`datetime()`) supaya string ISO-Z dan
    ISO+00:00 yang tercampur tetap terbaca, bukan dibandingkan sebagai teks.
    """
    now = now or _utcnow()
    cutoff = (now - _dt.timedelta(days=CHAT_RETENTION_DAYS)).isoformat()
    empty_cutoff = (now - _dt.timedelta(hours=EMPTY_SESSION_GRACE_HOURS)).isoformat()

    stale = conn.execute(
        """SELECT id FROM chat_sessions
           WHERE datetime(COALESCE(updated_at, created_at)) < datetime(?)""",
        (cutoff,),
    ).fetchall()
    empty = conn.execute(
        """SELECT s.id FROM chat_sessions s
           WHERE datetime(s.created_at) < datetime(?)
             AND NOT EXISTS (SELECT 1 FROM chat_messages m WHERE m.session_id = s.id)""",
        (empty_cutoff,),
    ).fetchall()
    ids = sorted({r["id"] for r in stale} | {r["id"] for r in empty})
    if not ids:
        return 0
    # Dihapus bertahap, bukan satu `IN (...)` raksasa: SQLite membatasi jumlah
    # variabel per statement (999 di build lama). Satu tahun pemakaian bisa
    # menghasilkan ribuan sesi kedaluwarsa sekaligus, dan job yang gagal di
    # tengah jalan akan mengulang kegagalan yang sama setiap 15 menit —
    # retensi berhenti tanpa ada yang menyadarinya.
    for i in range(0, len(ids), PURGE_CHUNK):
        chunk = ids[i:i + PURGE_CHUNK]
        marks = ",".join("?" * len(chunk))
        conn.execute("DELETE FROM chat_messages WHERE session_id IN (" + marks + ")",
                     chunk)
        conn.execute("DELETE FROM chat_sessions WHERE id IN (" + marks + ")", chunk)
    conn.commit()
    return len(ids)


SESSION_RETENTION_DAYS = int(os.environ.get("NOHALU_SESSION_RETENTION_DAYS", "30"))


def job_purge_sessions(conn, now=None) -> int:
    """Hapus baris `sessions` yang sudah mati (kedaluwarsa atau dicabut).

    Tabel ini append-only sejak login pertama: setiap login menambah satu baris,
    dan `logout` hanya menandai `revoked_at`. Di produksi terukur 1858 baris
    dalam ~2 hari (866 kedaluwarsa, 891 dicabut) tanpa satu pun job pembersih —
    artinya tabel tumbuh selamanya dan `_session_live()` (dipanggil setiap
    request ber-token) makin lambat. Baris mati tidak dibutuhkan lagi: token
    kedaluwarsa sudah ditolak oleh TTL, token dicabut sudah ditolak oleh
    ketiadaan baris.

    Masa simpan singkat (30 hari) tetap memberi jendela audit: insiden "token
    bocor" masih bisa ditelusuri dari `audit_log`, sedangkan tabel `sessions`
    sendiri bukan sumber audit.
    """
    now = now or _utcnow()
    cutoff = (now - _dt.timedelta(days=SESSION_RETENTION_DAYS)).isoformat()
    cur = conn.execute(
        """DELETE FROM sessions
           WHERE datetime(expires_at) < datetime(?)
              OR (revoked_at IS NOT NULL AND datetime(revoked_at) < datetime(?))""",
        (now.isoformat(), cutoff),
    )
    conn.commit()
    return cur.rowcount


JOBS = ("job_stale_approvals", "job_leave_starting", "job_new_payslips",
        "job_new_announcements", "job_purge_chat_sessions",
        "job_purge_sessions")


def run_once(conn, now=None) -> dict:
    """Jalankan semua job sekali. Return ringkasan jumlah notifikasi per job."""
    out = {}
    for name in JOBS:
        job = globals().get(name)
        if job is None:
            out[name] = -1
            continue
        try:
            out[name] = job(conn, now=now)
        except Exception:
            out[name] = -1
            traceback.print_exc()
    _state["last_run"] = _utcnow().isoformat().replace("+00:00", "Z")
    _state["last_result"] = dict(out)
    _state["runs"] = int(_state.get("runs") or 0) + 1
    return out


def _loop(db_path, interval):
    conn = connect_threadlocal(db_path)
    # Beri waktu startup selesai dulu (seed/repair) sebelum job pertama.
    threading.Event().wait(min(30, max(1, interval // 4)))
    while True:
        try:
            run_once(conn)
        except Exception:
            traceback.print_exc()
        threading.Event().wait(interval)


def start_scheduler(db_path, interval=None):
    """Idempotent: thread hanya dibuat sekali per proses."""
    global _started, _thread
    with _lock:
        if _started:
            return False
        if os.environ.get("NOHALU_REMINDERS", "1") == "0":
            return False
        _started = True
    t = threading.Thread(target=_loop, args=(db_path, interval or INTERVAL_SECONDS),
                         name="safedesk-reminders", daemon=True)
    t.start()
    _thread = t
    return True


def is_running() -> bool:
    """True bila thread scheduler masih hidup (dipakai /api/health)."""
    return bool(_thread and _thread.is_alive())


def status() -> dict:
    """Ringkasan scheduler untuk ops: hidup?, putaran terakhir, hasil per job."""
    return {
        "running": is_running(),
        "interval": INTERVAL_SECONDS,
        "runs": int(_state.get("runs") or 0),
        "last_run": _state.get("last_run"),
        "last_result": _state.get("last_result"),
    }
