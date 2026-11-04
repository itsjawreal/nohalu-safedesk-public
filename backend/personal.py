"""Data personal karyawan — cuti, absensi, slip gaji, org chart, tim.

Modul ini adalah satu-satunya sumber jawaban untuk pertanyaan "milik saya"
(engine chat) dan untuk endpoint `/api/me/*`. Semua fungsi menerima `user`
yang sudah terautentikasi dan **hanya** membaca baris milik user tersebut,
kecuali fungsi organisasi (org chart/team) yang memang publik internal tanpa
nominal gaji.

Zona waktu: `NOHALU_TZ_OFFSET_HOURS` (default +7 / WIB) — sama dengan portal.
"""
import datetime as _dt
import os
import re

BUSINESS_TZ_OFFSET = int(os.environ.get("NOHALU_TZ_OFFSET_HOURS", "7"))
DEFAULT_ENTITLEMENT = int(os.environ.get("NOHALU_LEAVE_ENTITLEMENT", "12"))
WORK_START = os.environ.get("NOHALU_WORK_START", "09:00")
WORK_END = os.environ.get("NOHALU_WORK_END", "18:00")

MONTHS_ID = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
             "Agustus", "September", "Oktober", "November", "Desember"]


# ---------------------------------------------------------------------------
# Waktu
# ---------------------------------------------------------------------------
def _tz():
    return _dt.timezone(_dt.timedelta(hours=BUSINESS_TZ_OFFSET))


def now():
    return _dt.datetime.now(_tz())


def today():
    return now().date()


def _iso(d):
    return d.isoformat() if d else None


def _parse_date(value):
    if not value:
        return None
    try:
        return _dt.date.fromisoformat(str(value).strip())
    except (ValueError, TypeError):
        return None


def is_workday(day: _dt.date | None = None) -> bool:
    """Senin–Jumat. Dipakai agar rekap & clock-in tahu bedanya hari kerja.

    Tanpa ini, pertanyaan "siapa yang belum absen hari ini?" pada Sabtu
    dijawab "11 karyawan belum clock-in" — terdengar seperti seluruh kantor
    membolos, padahal memang bukan hari kerja.
    """
    d = day or today()
    return d.isoweekday() <= 5


def _month_bounds(month: str | None):
    """Return (year, month, first_date, last_date) untuk 'YYYY-MM'."""
    if month:
        m = re.match(r"^(\d{4})-(\d{1,2})$", str(month).strip())
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            if 1 <= mo <= 12:
                first = _dt.date(y, mo, 1)
                last = _dt.date(y + (mo == 12), 1 if mo == 12 else mo + 1, 1) - _dt.timedelta(days=1)
                return y, mo, first, last
    t = today()
    first = t.replace(day=1)
    last = _dt.date(t.year + (t.month == 12), 1 if t.month == 12 else t.month + 1, 1) - _dt.timedelta(days=1)
    return t.year, t.month, first, last


# ---------------------------------------------------------------------------
# Profil & organisasi
# ---------------------------------------------------------------------------
_PROFILE_COLS = ("id", "name", "role", "department", "job_title", "manager_id",
                 "join_date", "employment_status", "location", "employee_no", "email")


def _row_val(row, key, default=None):
    try:
        return row[key]
    except (IndexError, KeyError):
        return default


def profile(conn, user) -> dict:
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
    if not row:
        return {}
    out = {k: _row_val(row, k) for k in _PROFILE_COLS}
    out["role"] = user.get("role") or out.get("role")
    mgr = None
    if out.get("manager_id"):
        mrow = conn.execute(
            "SELECT id,name,role,department,job_title FROM users WHERE id = ?",
            (out["manager_id"],)).fetchone()
        if mrow:
            mgr = {"id": mrow["id"], "name": mrow["name"], "role": mrow["role"],
                   "department": _row_val(mrow, "department"),
                   "job_title": _row_val(mrow, "job_title")}
    out["manager"] = mgr
    return out


def _public_person(row) -> dict:
    return {
        "id": row["id"], "name": row["name"], "role": row["role"],
        "department": _row_val(row, "department"),
        "job_title": _row_val(row, "job_title"),
        "location": _row_val(row, "location"),
        "employment_status": _row_val(row, "employment_status"),
    }


def org_chart(conn, user) -> dict:
    """Rantai atasan (ke atas) + bawahan langsung (ke bawah), tanpa gaji/PII."""
    me = profile(conn, user)
    chain, seen = [], {user["id"]}
    cur = me.get("manager_id")
    while cur and cur not in seen and len(chain) < 10:
        seen.add(cur)
        row = conn.execute("SELECT * FROM users WHERE id = ?", (cur,)).fetchone()
        if not row:
            break
        chain.append(_public_person(row))
        cur = _row_val(row, "manager_id")
    reports = []
    for row in conn.execute(
            "SELECT * FROM users WHERE manager_id = ? ORDER BY name", (user["id"],)).fetchall():
        reports.append(_public_person(row))
    return {"me": _public_person_self(me), "chain": chain, "reports": reports}


def _public_person_self(p: dict) -> dict:
    return {k: p.get(k) for k in ("id", "name", "role", "department", "job_title",
                                  "location", "employment_status")}


def team(conn, user) -> dict:
    """Atasan langsung, rekan setim (manager sama), dan bawahan langsung."""
    me = profile(conn, user)
    manager = me.get("manager")
    peers = []
    if me.get("manager_id"):
        for row in conn.execute(
                "SELECT * FROM users WHERE manager_id = ? AND id <> ? ORDER BY name",
                (me["manager_id"], user["id"])).fetchall():
            peers.append(_public_person(row))
    reports = [_public_person(r) for r in conn.execute(
        "SELECT * FROM users WHERE manager_id = ? ORDER BY name", (user["id"],)).fetchall()]
    return {"me": _public_person_self(me), "manager": manager, "peers": peers,
            "reports": reports}


# ---------------------------------------------------------------------------
# Cuti
# ---------------------------------------------------------------------------
def _leave_requests(conn, user_id, year):
    rows = conn.execute(
        """SELECT id,status,start_date,end_date,title,created_at FROM requests
           WHERE user_id = ? AND type = 'cuti'
             AND status IN ('pending','approved','rejected','cancelled')
           ORDER BY start_date""",
        (user_id,)).fetchall()
    out = []
    for r in rows:
        s, e = _parse_date(r["start_date"]), _parse_date(r["end_date"])
        if not s or not e:
            continue
        # hitung irisan dengan tahun yang diminta
        ys, ye = _dt.date(year, 1, 1), _dt.date(year, 12, 31)
        a, b = max(s, ys), min(e, ye)
        days = (b - a).days + 1 if b >= a else 0
        if days <= 0:
            continue
        out.append({"id": r["id"], "status": r["status"],
                    "start_date": r["start_date"], "end_date": r["end_date"],
                    "days": days, "title": r["title"]})
    return out


def leave_balance(conn, user, year: int | None = None) -> dict:
    """Saldo cuti = entitlement + carry_over - used(approved) - pending."""
    year = int(year or today().year)
    row = conn.execute(
        "SELECT entitlement, carry_over FROM leave_balances WHERE user_id = ? AND year = ?",
        (user["id"], year)).fetchone()
    entitlement = _row_val(row, "entitlement", DEFAULT_ENTITLEMENT) if row else DEFAULT_ENTITLEMENT
    carry = _row_val(row, "carry_over", 0) if row else 0
    entitlement = DEFAULT_ENTITLEMENT if entitlement is None else int(entitlement)
    carry = int(carry or 0)

    items = _leave_requests(conn, user["id"], year)
    used = sum(i["days"] for i in items if i["status"] == "approved")
    pending = sum(i["days"] for i in items if i["status"] == "pending")
    before_pending = entitlement + carry - used
    return {
        "year": year,
        "entitlement": entitlement,
        "carry_over": carry,
        "used": used,
        "pending": pending,
        # `remaining` SUDAH memotong cuti yang masih menunggu persetujuan.
        # `before_pending` adalah saldo bila semua pending ditolak. Dua angka ini
        # dulu hanya satu, dan model menyimpulkan sendiri bahwa pending "belum
        # terpotong" — jawaban HR jadi kontradiktif ("sisa 9 hari ... saldo
        # aktifnya masih 12"). Keduanya dikirim eksplisit agar tidak ditafsirkan.
        "remaining": before_pending - pending,
        "before_pending": before_pending,
        "requests": items,
    }


# ---------------------------------------------------------------------------
# Absensi
# ---------------------------------------------------------------------------
def _hhmm(value):
    """'09:10:00' -> '09:10'; None/'' -> None (bukan string kosong).

    Konsisten dengan API: field yang belum ada harus `null`, bukan `""`, agar
    klien tidak menampilkan nilai kosong sebagai jam yang valid.
    """
    if not value:
        return None
    return str(value)[:5]


def attendance_summary(conn, user, month: str | None = None) -> dict:
    y, m, first, last = _month_bounds(month)
    rows = conn.execute(
        """SELECT date, clock_in, clock_out, status, source FROM attendance
           WHERE user_id = ? AND date BETWEEN ? AND ? ORDER BY date""",
        (user["id"], first.isoformat(), last.isoformat())).fetchall()
    days = [{
        "date": r["date"], "clock_in": _hhmm(r["clock_in"]),
        "clock_out": _hhmm(r["clock_out"]), "status": r["status"],
        "source": _row_val(r, "source", "web"),
    } for r in rows]
    summary = {"present": 0, "late": 0, "leave": 0, "sick": 0, "wfh": 0}
    for d in days:
        summary[d["status"]] = summary.get(d["status"], 0) + 1
    return {
        "month": f"{y:04d}-{m:02d}",
        "label": f"{MONTHS_ID[m - 1]} {y}",
        "work_start": WORK_START, "work_end": WORK_END,
        "days": days, "summary": summary,
        "today": today().isoformat(),
        "today_is_workday": is_workday(),
        "today_record": next((d for d in days if d["date"] == today().isoformat()), None),
    }


def clock(conn, user, action: str | None = None) -> tuple[dict, int]:
    """Clock-in/out hari ini. Idempotent: klik ganda tidak membuat baris baru."""
    day = today().isoformat()
    row = conn.execute(
        "SELECT * FROM attendance WHERE user_id = ? AND date = ?",
        (user["id"], day)).fetchone()
    now_hm = now().strftime("%H:%M")

    if action not in (None, "", "in", "out"):
        return {"error": "invalid_action",
                "message": "Aksi harus 'in' atau 'out'."}, 400

    workday = is_workday()
    if row is None:
        if action == "out":
            return {"error": "not_clocked_in",
                    "message": "Anda belum clock-in hari ini."}, 409
        # Di luar hari kerja, clock-in tidak pernah dihitung terlambat: tidak
        # ada jam masuk yang berlaku. Tetap dicatat (karyawan boleh masuk
        # weekend), tetapi statusnya `present`, bukan `late`.
        late = workday and now_hm > WORK_START
        status = "late" if late else "present"
        conn.execute(
            """INSERT INTO attendance (user_id, date, clock_in, status, source)
               VALUES (?,?,?,?,?)""",
            (user["id"], day, now_hm, status, "web"))
        conn.commit()
        out = {"date": day, "clock_in": now_hm, "clock_out": None,
               "status": status, "idempotent": False, "workday": workday}
        if not workday:
            out["note"] = ("Hari ini bukan hari kerja (akhir pekan); catatan "
                           "kehadiran tidak dihitung terlambat.")
        return out, 201

    if action == "in":
        # sudah clock-in: kembalikan yang ada (bukan error) — mencegah duplikat
        return {"date": day, "clock_in": _hhmm(row["clock_in"]),
                "clock_out": _hhmm(row["clock_out"]), "status": row["status"],
                "idempotent": True}, 200

    if row["clock_out"]:
        return {"date": day, "clock_in": _hhmm(row["clock_in"]),
                "clock_out": _hhmm(row["clock_out"]), "status": row["status"],
                "idempotent": True}, 200

    conn.execute("UPDATE attendance SET clock_out = ? WHERE id = ?", (now_hm, row["id"]))
    conn.commit()
    return {"date": day, "clock_in": _hhmm(row["clock_in"]), "clock_out": now_hm,
            "status": row["status"], "idempotent": False}, 200


# ---------------------------------------------------------------------------
# Slip gaji
# ---------------------------------------------------------------------------
def payslips(conn, user) -> dict:
    """Slip milik user sendiri. HR/admin hanya melihat daftar miliknya juga.

    `has_file` dikirim terpisah dari `file_ref`: UI butuh tahu apakah tombol
    unduh bisa ditampilkan, tetapi path penyimpanan di server tidak perlu
    (dan tidak boleh) bocor ke klien.
    """
    rows = conn.execute(
        """SELECT id, period, gross, deductions, net, file_ref, published_at
           FROM payslips WHERE user_id = ? ORDER BY period DESC""",
        (user["id"],)).fetchall()
    items = [{
        "id": r["id"], "period": r["period"],
        "gross": r["gross"], "deductions": r["deductions"], "net": r["net"],
        "has_file": bool(_row_val(r, "file_ref")),
        "published_at": _row_val(r, "published_at"),
    } for r in rows]
    return {"payslips": items, "count": len(items)}


def list_payslips(conn, actor, period=None, q=None) -> dict:
    """Daftar slip untuk HR/admin (satu periode atau semua).

    Tanpa ini halaman Kelola hanya bisa MENERBITKAN slip: HR tidak punya cara
    melihat periode mana yang sudah terbit, siapa yang belum dapat slip, atau
    slip mana yang belum punya berkas. Endpoint `/api/me/payslips` sengaja
    tidak dipakai karena selalu memotong ke pemilik token.
    """
    if actor["role"] not in ("hr", "admin"):
        return {"error": "not_authorized",
                "message": "Hanya HR/admin yang dapat melihat daftar slip."}, 403
    # Daftar periode diambil dari SELURUH tabel, bukan dari hasil filter:
    # kalau tidak, memilih satu periode membuat dropdown hanya berisi periode
    # itu dan pengguna tidak bisa pindah periode lagi tanpa menghapus filter.
    # Query terpisah (bukan dari `rows`) supaya filter periode tetap dikerjakan
    # SQL, bukan dengan memindai semua slip di memori.
    periods = sorted(
        {r["period"] for r in conn.execute(
            "SELECT DISTINCT period FROM payslips WHERE period IS NOT NULL")},
        reverse=True)
    sql = """SELECT p.id, p.user_id, p.period, p.gross, p.deductions, p.net,
                    p.file_ref, p.published_at, u.name AS user_name
             FROM payslips p LEFT JOIN users u ON u.id = p.user_id"""
    args = []
    if period:
        sql += " WHERE p.period = ?"
        args.append(period)
    sql += " ORDER BY p.period DESC, u.name"
    rows = conn.execute(sql, args).fetchall()
    needle = (q or "").strip().lower()
    items = []
    for r in rows:
        name = _row_val(r, "user_name") or ""
        if needle and needle not in (name.lower() + " " + (r["user_id"] or "").lower()):
            continue
        items.append({
            "id": r["id"], "user_id": r["user_id"],
            "user_name": name or r["user_id"], "period": r["period"],
            "gross": r["gross"], "deductions": r["deductions"], "net": r["net"],
            "has_file": bool(_row_val(r, "file_ref")),
            "published_at": _row_val(r, "published_at"),
        })
    return {"payslips": items, "count": len(items), "periods": periods,
            "without_file": sum(1 for i in items if not i["has_file"])}, 200


def publish_payslip(conn, actor, payload: dict, db_path=None) -> tuple[dict, int]:
    """HR/admin menerbitkan slip untuk satu karyawan & periode.

    `db_path` (opsional) mengaktifkan pembuatan PDF slip otomatis. Tanpa ini
    slip hanya berupa angka di DB dan tombol "Unduh" selalu 404 — persis gap
    yang membuat 24 dari 24 slip seed tidak punya berkas. Pemanggil yang tidak
    ingin menulis ke disk (tes murni DB) cukup tidak mengirim `db_path`.
    """
    if actor["role"] not in ("hr", "admin"):
        return {"error": "not_authorized",
                "message": "Hanya HR/admin yang dapat menerbitkan slip."}, 403
    user_id = (payload.get("user_id") or "").strip()
    period = (payload.get("period") or "").strip()
    if not re.match(r"^\d{4}-(0[1-9]|1[0-2])$", period):
        return {"error": "invalid_period",
                "message": "Periode harus format YYYY-MM.", "fields": {"period": "YYYY-MM"}}, 422
    if not conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone():
        return {"error": "user_not_found", "message": "Karyawan tidak ditemukan."}, 404
    def _int(v, name):
        try:
            n = int(v)
        except (TypeError, ValueError):
            return None, {"error": "invalid_amount", "message": f"{name} harus angka.",
                          "fields": {name: "angka"}}
        if n < 0:
            return None, {"error": "invalid_amount", "message": f"{name} tidak boleh negatif."}
        return n, None
    gross, err = _int(payload.get("gross", 0), "gross")
    if err:
        return err, 422
    deductions, err = _int(payload.get("deductions", 0), "deductions")
    if err:
        return err, 422
    net = payload.get("net")
    if net is None:
        net = gross - deductions
    net, err = _int(net, "net")
    if err:
        return err, 422
    file_ref = payload.get("file_ref")
    if not file_ref and db_path:
        # PDF dibuat server, bukan diunggah HR: slip yang baru diterbitkan harus
        # langsung bisa diunduh karyawan tanpa langkah tambahan.
        import files as files_mod
        import payslip_pdf
        person = conn.execute("SELECT name FROM users WHERE id = ?",
                              (user_id,)).fetchone()
        pdf = payslip_pdf.build({
            "user_name": _row_val(person, "name") or user_id,
            "user_id": user_id, "period": period,
            "gross": gross, "deductions": deductions, "net": net,
        })
        meta = files_mod.save(db_path, f"slip-{period}.pdf", pdf, kind="payslips")
        file_ref = meta["ref"]
    conn.execute(
        """INSERT INTO payslips (user_id, period, gross, deductions, net, file_ref, published_at)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(user_id, period) DO UPDATE SET
             gross=excluded.gross, deductions=excluded.deductions, net=excluded.net,
             file_ref=COALESCE(excluded.file_ref, payslips.file_ref),
             published_at=excluded.published_at""",
        (user_id, period, gross, deductions, net, file_ref,
         now().isoformat(timespec="seconds")))
    # Kategori `payslip` supaya user bisa mengatur preferensi; import lokal
    # untuk menghindari siklus portal -> engine -> personal.
    # `notify_once` dengan kunci yang sama seperti job reminder: tanpa ini
    # penerbitan slip langsung + job 24 jam menghasilkan dua notifikasi identik.
    import portal
    portal.notify_once(conn, [user_id], f"Slip gaji periode {period} sudah tersedia.",
                       link="/app/slip-gaji", category="payslip",
                       dedupe_key=f"payslip:{user_id}:{period}")
    conn.commit()
    return {"user_id": user_id, "period": period, "net": net}, 201


def payslip_file(conn, db_path, user, slip_id):
    """Berkas slip milik user sendiri. 404 untuk slip orang lain (bukan 403).

    404 dipakai supaya keberadaan slip karyawan lain tidak bisa ditebak dari
    perbedaan status.
    """
    import files as files_mod
    row = conn.execute(
        "SELECT id, user_id, period, file_ref FROM payslips WHERE id = ?",
        (slip_id,)).fetchone()
    if not row or row["user_id"] != user["id"]:
        return None, 404, {"error": "payslip_not_found",
                           "message": "Slip tidak ditemukan."}
    ref = _row_val(row, "file_ref")
    if not ref:
        return None, 404, {"error": "payslip_no_file",
                           "message": "Slip ini belum punya berkas untuk diunduh."}
    path = files_mod.resolve(db_path, ref)
    if not path:
        return None, 404, {"error": "payslip_file_missing",
                           "message": "Berkas slip hilang dari penyimpanan."}
    return {"path": path, "period": row["period"],
            "filename": f"slip-gaji-{row['period']}{os.path.splitext(ref)[1]}",
            "content_type": files_mod.content_type(ref)}, 200, None


def attach_payslip_file(conn, db_path, actor, payload):
    """Terbitkan/perbarui slip dengan berkas lampiran (HR/admin).

    `file_ref` lama dihapus dari disk HANYA setelah baris DB berhasil ditulis,
    supaya kegagalan di tengah jalan tidak meninggalkan slip tanpa berkas.
    """
    import files as files_mod
    user_id = (payload.get("user_id") or "").strip()
    period = (payload.get("period") or "").strip()
    row = conn.execute(
        "SELECT id, file_ref FROM payslips WHERE user_id=? AND period=?",
        (user_id, period)).fetchone()
    if not row:
        return {"error": "payslip_not_found",
                "message": "Terbitkan slip dulu, baru lampirkan berkasnya."}, 404
    meta = payload.get("_file")
    old_ref = _row_val(row, "file_ref")
    conn.execute("UPDATE payslips SET file_ref=? WHERE id=?", (meta["ref"], row["id"]))
    conn.commit()
    if old_ref and old_ref != meta["ref"]:
        files_mod.delete(db_path, old_ref)
    # `file_ref` (path di server) sengaja TIDAK dikirim: klien hanya butuh
    # tahu bahwa berkas terpasang. Sama seperti `payslips()` yang memakai
    # `has_file`, bukan path penyimpanan.
    return {"id": row["id"], "user_id": user_id, "period": period,
            "filename": meta["name"], "bytes": meta["bytes"],
            "sha256": meta["sha256"], "has_file": True,
            "status": "attached"}, 200


# ---------------------------------------------------------------------------
# Intent personal untuk chat (ID + EN)
# ---------------------------------------------------------------------------
_LEAVE_WORDS = ("cuti", "leave", "libur", "off")
_LEAVE_BALANCE_WORDS = ("sisa", "saldo", "jatah", "tersisa", "remaining", "balance",
                        "left", "kuota", "quota", "how many")
_MANAGER_WORDS = ("atasan", "manajer", "manager", "boss", "supervisor", "report to",
                  "siapa atasan", "who is my")
_TEAM_WORDS = ("tim saya", "my team", "rekan tim", "anggota tim", "team members",
               "teman satu tim", "peers")
_PAYSLIP_WORDS = ("slip gaji", "payslip", "pay slip", "slip gajiku", "salary slip",
                  "take home pay", "gaji saya", "my salary", "my payslip")
_ATTENDANCE_WORDS = ("absensi", "presensi", "attendance", "clock in", "clock-in",
                     "clock out", "clock-out", "absen saya", "kehadiran saya",
                     "my attendance", "check in")
_PENDING_WORDS = ("menunggu", "pending", "waiting", "belum disetujui", "outstanding")
# Kata "gaji" pada pertanyaan yang BUKAN slip gaji milik sendiri (mis. "gaji
# anggota tim saya") tidak boleh dibajak menjadi intent `team`/`manager`; itu
# pertanyaan data sensitif yang harus ditangani jalur agent + RBAC.
_SALARY_WORDS = ("gaji", "salary", "upah", "payroll", "penghasilan", "take home pay",
                 "gajian")
_REQUEST_WORDS = ("pengajuan", "request", "approval", "persetujuan", "cuti")


def _norm(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").lower()).strip()


_PROCEDURAL_RE = re.compile(
    r"\b(bagaimana|gimana|caranya|cara|kebijakan|policy|prosedur|procedure|alur|"
    r"aturan|langkah|tahapan|syarat|ketentuan)\b|\bhow (do|to|can|should)\b")

# Penanda pertanyaan yang menanyakan NILAI/STATUS data milik sendiri, bukan
# prosedur. Dipakai bersama guard prosedural: "bagaimana absensi saya bulan ini"
# tetap personal, sedangkan "bagaimana kebijakan absensi" bukan.
_DATA_RE = re.compile(
    r"\b(berapa|sisa|saldo|jatah|tersisa|remaining|balance|kuota|quota|kapan|"
    r"rekap|status|bulan ini|minggu ini|tahun ini|hari ini|besok|kemarin|"
    r"this month|this year|today|tomorrow|yesterday|where|di mana|dimana|"
    r"how many|how much)\b")

_OWN_RE = re.compile(r"\b(saya|aku|ku|milik saya|punya saya|my|mine|me|i)\b")

_SCHEDULE_WORDS = ("jam berapa", "jam masuk", "jam kerja", "jam pulang", "jam mulai",
                   "jadwal kerja", "jadwal saya", "shift", "work hours", "work schedule",
                   "what time do i", "what time should i")

# "siapa nama saya", "nama saya siapa", "what is my name", "siapa saya".
# Sengaja TIDAK cocok dengan "siapa nama atasan saya" (itu intent `manager`)
# maupun "siapa nama karyawan X" (itu pertanyaan tentang orang lain).
_WHOAMI_RE = re.compile(
    r"\b(?:siapa|apa)\s+nama\s+(?:saya|aku)\b"
    r"|\bnama\s+(?:saya|aku)\s+(?:siapa|apa)\b"
    r"|\bsiapa\s+(?:saya|aku)\b"
    r"|\bwhat(?:'s|\s+is)\s+my\s+name\b"
    r"|\bwho\s+am\s+i\b")


def detect_personal_intent(question: str):
    """Return nama intent personal atau None. Deterministik & konservatif.

    Pertanyaan **prosedural/kebijakan** ("bagaimana cara...", "apa kebijakan...")
    TIDAK boleh dibajak menjadi jawaban data personal; pertanyaan seperti itu
    harus jatuh ke retrieval dokumen. Pengecualian: pertanyaan prosedural yang
    sekaligus menanyakan nilai/status milik sendiri ("bagaimana absensi saya
    bulan ini") tetap dijawab dari data.
    """
    q = _norm(question)
    if not q:
        return None
    procedural = bool(_PROCEDURAL_RE.search(q))
    has_data_marker = bool(_DATA_RE.search(q))
    has_mine = bool(_OWN_RE.search(q))
    if procedural and not (has_mine and has_data_marker):
        return None
    # Pertanyaan nominal gaji selain slip gaji sendiri = data sensitif; jangan
    # dijawab dari struktur organisasi. Serahkan ke jalur agent/RBAC.
    if any(w in q for w in _SALARY_WORDS) and not any(w in q for w in _PAYSLIP_WORDS):
        return None

    has_leave = any(w in q for w in _LEAVE_WORDS)

    # Saldo cuti: butuh kata saldo/sisa + cuti (atau "cuti saya berapa")
    if has_leave and any(w in q for w in _LEAVE_BALANCE_WORDS):
        return "leave_balance"
    if has_leave and has_mine and any(w in q for w in ("berapa", "how many", "how much")):
        return "leave_balance"

    # Atasan
    if any(w in q for w in _MANAGER_WORDS) and (has_mine or "atasan" in q or "manager" in q):
        if not any(w in q for w in ("kebijakan", "policy", "cuti", "darurat", "pengadaan")):
            return "manager"

    # Tim
    if any(w in q for w in _TEAM_WORDS):
        return "team"

    # Jadwal kerja milik sendiri ("jam berapa saya masuk besok?")
    if has_mine and any(w in q for w in _SCHEDULE_WORDS):
        return "work_schedule"

    # Slip gaji (harus eksplisit milik sendiri)
    if any(w in q for w in _PAYSLIP_WORDS):
        return "payslip"

    # Absensi
    if any(w in q for w in _ATTENDANCE_WORDS) and (has_mine or "absensi" in q or "attendance" in q):
        return "attendance"

    # Jumlah pengajuan menunggu
    if any(w in q for w in _PENDING_WORDS) and any(w in q for w in _REQUEST_WORDS):
        return "pending_requests"

    # Identitas penanya sendiri ("siapa nama saya?"). Diletakkan PALING AKHIR
    # supaya "siapa nama atasan saya" tetap dijawab intent `manager`, dan
    # pertanyaan tentang orang lain ("siapa nama karyawan Grade 3") tidak
    # dibajak menjadi profil penanya.
    if _WHOAMI_RE.search(q):
        return "whoami"
    return None


def other_person_target(conn, user, question):
    """Karyawan LAIN yang disebut namanya di pertanyaan, atau None.

    Dua bentuk pencocokan, dari yang paling ketat:
      1. nama lengkap — semua katanya muncul (sebagai kata utuh) di pertanyaan;
      2. nama depan — hanya bila unik di seluruh direktori.
    Bentuk (2) perlu karena user hampir selalu menulis nama depan saja ("sisa
    cuti Maya"). Keunikannya yang menjaga dari salah sasaran: bila dua karyawan
    bernama depan sama, tidak ada tebakan.
    Penanya sendiri selalu dikecualikan.
    """
    q = " ".join((question or "").lower().split())
    if not q:
        return None
    words = set(re.findall(r"[a-z]+", q))
    me = (user or {}).get("id")
    try:
        rows = conn.execute("SELECT id, name FROM users").fetchall()
    except Exception:  # noqa: BLE001 — deteksi bersifat best-effort
        return None
    people = [((r["name"] or "").strip(), r["id"]) for r in rows]
    first_counts = {}
    for full, _uid in people:
        first = (full.split() or [""])[0].lower()
        if len(first) > 2:
            first_counts[first] = first_counts.get(first, 0) + 1
    full_hit, first_hit = None, None
    for full, uid in people:
        if not full or uid == me:
            continue
        parts = [p for p in re.findall(r"[a-z]+", full.lower()) if len(p) > 2]
        if parts and all(p in words for p in parts):
            if full_hit is None or len(full) > len(full_hit[1]):
                full_hit = (uid, full)
            continue
        first = (full.split() or [""])[0].lower()
        if (len(first) > 2 and first_counts.get(first) == 1 and first in words
                and first_hit is None):
            first_hit = (uid, full)
    return full_hit or first_hit


_INTENT_LABEL = {
    "leave_balance": "cuti",
    "attendance": "absensi",
    "payslip": "slip gaji",
    "pending_requests": "pengajuan",
}

# Siapa yang MEMANG berwenang atas data itu. Slip gaji sengaja berbeda: bahkan
# atasan langsung tidak boleh melihatnya (lihat dokumen "Cara Melihat &
# Memahami Slip Gaji"), jadi menyebut "atasan langsung" di pesan penolakan
# slip gaji adalah informasi yang salah.
_INTENT_AUTHORITY = {
    "leave_balance": "atasan langsung, HR, atau admin perusahaan",
    "attendance": "atasan langsung, HR, atau admin perusahaan",
    "pending_requests": "atasan langsung, HR, atau admin perusahaan",
    "payslip": "pemilik akun yang bersangkutan, HR, atau admin perusahaan",
}


def _fmt_rp(n):
    try:
        return "Rp" + f"{int(n):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "Rp-"


def answer_personal(conn, user, question: str):
    """Jawab pertanyaan personal dari data. Return result-dict atau None.

    Result mengikuti bentuk `compose_answer` supaya bisa dipakai langsung oleh
    `answer_question` (answer, citations, selected_sources, engine_used).
    """
    # Pertanyaan tentang ORANG LAIN bukan data personal penanya. Tanpa gerbang
    # ini, "Berapa sisa cuti Maya?" dijawab dengan saldo cuti Budi (penanya):
    # user mendapat angka yang salah untuk subjek yang salah, tanpa penolakan
    # apa pun. Deteksi nama sudah ada di `other_person_target`, jadi di sini
    # cukup menyerahkan pertanyaannya ke jalur agent/RBAC.
    intent = detect_personal_intent(question)
    if not intent:
        return None
    # Pertanyaan DATA PERSONAL tentang orang lain bukan milik penanya. Menjawab
    # dengan data penanya sendiri adalah bug paling berbahaya di sini: user
    # mendapat angka yang salah untuk subjek yang salah, tanpa penolakan apa
    # pun ("Berapa sisa cuti Maya?" -> saldo cuti Budi). Struktur organisasi
    # ("siapa atasan Maya") tidak terpengaruh karena intent-nya bukan data
    # personal.
    target = other_person_target(conn, user, question)
    if target:
        return {
            "answer": ("Data " + _INTENT_LABEL.get(intent, "kepegawaian") +
                       " karyawan lain hanya bisa diakses oleh " +
                       _INTENT_AUTHORITY.get(intent,
                                             "HR atau admin perusahaan") +
                       ". Akun Anda belum mencakup data itu."),
            "citations": [], "conflict_detected": False, "conflict_note": "",
            "permission_decision": "deny", "refusal_reason": "no_access",
            "pii_masked": False, "prompt_injection_detected": False,
            "selected_sources": [], "retrieved_sources": [],
            "engine_used": "personal", "personal_intent": intent,
            "data_source": "personal",
        }

    def base(answer):
        return {"answer": answer, "citations": [], "conflict_detected": False,
                "conflict_note": "", "permission_decision": "allow",
                "refusal_reason": None, "pii_masked": False,
                "prompt_injection_detected": False, "selected_sources": [],
                "retrieved_sources": [], "engine_used": "personal",
                "personal_intent": intent, "data_source": "personal"}

    if intent == "leave_balance":
        lb = leave_balance(conn, user)
        lines = [
            f"Sisa cuti Anda tahun {lb['year']}: {lb['remaining']} hari "
            f"(hak {lb['entitlement']} hari"
            + (f" + carry-over {lb['carry_over']} hari" if lb["carry_over"] else "")
            + f", terpakai {lb['used']} hari, menunggu persetujuan {lb['pending']} hari)."
        ]
        if lb["requests"]:
            lines.append("Rincian pengajuan cuti:")
            for r in lb["requests"][:8]:
                lines.append(
                    f"- #{r['id']} {r['start_date']} s.d. {r['end_date']} "
                    f"({r['days']} hari) — status {r['status']}.")
        else:
            lines.append("Belum ada pengajuan cuti tercatat tahun ini.")
        lines.append("Cuti tahunan diajukan minimal 3 hari kerja sebelum tanggal mulai.")
        return base("\n".join(lines))

    if intent == "manager":
        p = profile(conn, user)
        mgr = p.get("manager")
        if not mgr:
            return base("Anda belum memiliki atasan langsung yang tercatat di sistem. "
                        "Hubungi HR untuk memperbarui data organisasi.")
        title = f" ({mgr['job_title']})" if mgr.get("job_title") else ""
        dept = f", departemen {mgr['department']}" if mgr.get("department") else ""
        return base(f"Atasan langsung Anda adalah {mgr['name']}{title}{dept}. "
                    f"Untuk hal administratif (cuti, reimbursement), atasan inilah "
                    f"yang menjadi tahap persetujuan pertama.")

    if intent == "team":
        t = team(conn, user)
        lines = []
        if t["manager"]:
            lines.append(f"Atasan: {t['manager']['name']} "
                         f"({t['manager'].get('job_title') or t['manager']['role']}).")
        if t["peers"]:
            lines.append("Rekan setim: " + ", ".join(
                f"{p['name']} ({p.get('job_title') or p['role']})" for p in t["peers"]) + ".")
        if t["reports"]:
            lines.append("Bawahan langsung: " + ", ".join(
                f"{p['name']} ({p.get('job_title') or p['role']})" for p in t["reports"]) + ".")
        if not lines:
            return base("Belum ada data tim yang tercatat untuk akun Anda.")
        return base("\n".join(lines))

    if intent == "work_schedule":
        q = _norm(question)
        target = today() + _dt.timedelta(days=1) if ("besok" in q or "tomorrow" in q) else today()
        hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"][target.isoweekday() - 1]
        label = f"{hari}, {target.day} {MONTHS_ID[target.month - 1]} {target.year}"
        lines = []
        if target.isoweekday() > 5:
            lines.append(f"{label} adalah akhir pekan — tidak ada jadwal kerja. "
                         f"Jam kerja normal Anda {WORK_START}–{WORK_END} WIB "
                         f"pada hari kerja (Senin–Jumat).")
        else:
            lines.append(f"Jadwal kerja Anda {label}: masuk {WORK_START} WIB, "
                         f"pulang {WORK_END} WIB.")
        lines.append(f"Clock-in setelah {WORK_START} dicatat sebagai terlambat; "
                     f"clock-in/clock-out dilakukan lewat menu Saya → Absensi.")
        a = attendance_summary(conn, user)
        if a["today_record"]:
            tr = a["today_record"]
            lines.append(f"Catatan hari ini: masuk {tr['clock_in'] or '-'} · "
                         f"keluar {tr['clock_out'] or '-'} · status {tr['status']}.")
        else:
            lines.append("Hari ini belum ada catatan clock-in.")
        p = profile(conn, user)
        if p.get("location"):
            lines.append(f"Lokasi kerja tercatat: {p['location']}.")
        return base("\n".join(lines))

    if intent == "payslip":
        data = payslips(conn, user)
        if not data["count"]:
            return base("Belum ada slip gaji yang diterbitkan untuk akun Anda. "
                        "Slip biasanya tersedia setelah payroll bulanan diproses; "
                        "hubungi Finance melalui #finance bila menurut Anda ini keliru.")
        lines = [f"Ada {data['count']} slip gaji untuk akun Anda:"]
        for s in data["payslips"][:12]:
            lines.append(f"- Periode {s['period']}: bersih {_fmt_rp(s['net'])} "
                         f"(bruto {_fmt_rp(s['gross'])}, potongan {_fmt_rp(s['deductions'])}).")
        lines.append("Buka menu Saya → Slip Gaji untuk melihat rinciannya.")
        return base("\n".join(lines))

    if intent == "attendance":
        a = attendance_summary(conn, user)
        lines = [f"Rekap absensi {a['label']}:"]
        if a["today_record"]:
            tr = a["today_record"]
            lines.append(f"Hari ini: masuk {tr['clock_in'] or '-'} · "
                         f"keluar {tr['clock_out'] or '-'} · status {tr['status']}.")
        else:
            lines.append(f"Hari ini ({a['today']}) belum ada catatan clock-in.")
        s = a["summary"]
        lines.append(f"Total bulan ini: hadir {s.get('present', 0)} hari, "
                     f"terlambat {s.get('late', 0)}, cuti {s.get('leave', 0)}, "
                     f"sakit {s.get('sick', 0)}, WFH {s.get('wfh', 0)}.")
        lines.append(f"Jam kerja normal {a['work_start']}–{a['work_end']} WIB.")
        return base("\n".join(lines))

    if intent == "pending_requests":
        rows = conn.execute(
            "SELECT id,type,title,status FROM requests WHERE status='pending' "
            "ORDER BY id DESC").fetchall()
        if user["role"] in ("hr", "admin", "team_lead", "finance"):
            waiting = []
            for r in rows:
                from portal import _request_row  # local import (hindari siklus)
                rr = _request_row(r)
                if rr["user_id"] != user["id"] and rr["current_role"] == user["role"]:
                    waiting.append(rr)
            mine = [r for r in rows if r["user_id"] == user["id"]]
            return base(
                f"Ada {len(waiting)} pengajuan menunggu keputusan Anda dan "
                f"{len(mine)} pengajuan Anda sendiri yang masih menunggu. "
                f"Buka menu Persetujuan untuk memprosesnya.")
        mine = conn.execute(
            "SELECT COUNT(*) AS n FROM requests WHERE user_id=? AND status='pending'",
            (user["id"],)).fetchone()["n"]
        return base(f"Anda memiliki {mine} pengajuan yang masih menunggu persetujuan. "
                    f"Status terbaru bisa dilihat di menu Pengajuan.")

    if intent == "whoami":
        # Pertanyaan identitas penanya: dijawab dari data akun, bukan dari
        # dokumen struktur organisasi (yang dulu dikutip dan justru menyebut
        # nama direktur, bukan nama penanya).
        p = profile(conn, user)
        parts = [f"Anda adalah {p.get('name') or user['name']}"]
        if p.get("job_title"):
            parts.append(f"({p['job_title']})")
        line = " ".join(parts)
        extra = []
        if p.get("department"):
            extra.append(f"departemen {p['department']}")
        if p.get("employee_no"):
            extra.append(f"NIK {p['employee_no']}")
        if p.get("join_date"):
            extra.append(f"bergabung {p['join_date']}")
        if extra:
            line += ", " + ", ".join(extra)
        line += "."
        return base(line)

    return None
