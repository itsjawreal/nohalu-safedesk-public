"""Nohalu SafeDesk — FastAPI application.

Mock auth (token = user_id). RBAC via role. All safety logic lives in engine.py.
Run: uvicorn app:app --port 8000  (from backend/)
"""
import hashlib
import os
import re
import json
import datetime
import uuid
import threading
import sys
import time

from fastapi import FastAPI, Header, HTTPException, Depends, Body, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from db import (connect_threadlocal, seed_if_empty, ensure_seed_docs,
                ensure_seed_users, backfill_chat_meta, row_to_doc, DEFAULT_DB_PATH,
                begin_request_scope, end_request_scope)
import engine
from engine import answer_question, get_user, filter_allowed_docs, mask_pii, AGENT_VERSION
import agent
import agent_tools
import auth
import llm
import ingest
import portal
import personal
import paging
import metrics
import rbac
import logging_setup
import backup
import docs_lifecycle as lifecycle
import hr_actions
import users_admin
import workspaces
import files
import csv

# "/docs" dipakai SPA (halaman dokumentasi produk); Swagger pindah ke /api/docs.
app = FastAPI(title="Nohalu SafeDesk", version=AGENT_VERSION,
              docs_url="/api/docs", redoc_url="/api/redoc",
              openapi_url="/api/openapi.json")

# Origin diambil dari environment; tidak ada domain yang di-hardcode. Isi
# `NOHALU_CORS_ORIGINS` dengan origin frontend Anda (dipisah koma), mis.
# "https://safedesk.example.com,http://localhost:3000". Bila kosong, browser
# lintas origin akan ditolak — server tetap melayani permintaan same-origin
# dan klien non-browser (curl, skrip).
_CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "NOHALU_CORS_ORIGINS", "",
).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS, allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- Log request + error terstruktur -------------------------------------
# Hanya jalur /api yang dicatat: aset statis (gambar, CSS, JS) akan membanjiri
# log tanpa memberi informasi apa pun. Endpoint health juga dilewati supaya
# probe uptime tidak dihitung sebagai lalu lintas pengguna.
_LOG_SKIP = ("/api/health",)

REQUEST_LOG = logging_setup.get_logger("request")


class RequestLogMiddleware:
    """Middleware ASGI murni untuk log request.

    Kenapa bukan `@app.middleware("http")`: BaseHTTPMiddleware membungkus
    `receive`, dan itu memutus `request.is_disconnected()` pada respons
    streaming (SSE). Akibatnya slot kuota SSE tidak dilepas saat klien menutup
    tab — regresi yang sudah pernah diperbaiki di endpoint. Middleware ini
    hanya mengintip `http.response.start` untuk status, tanpa menyentuh
    `receive`, jadi perilaku streaming tetap apa adanya.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        # Scope request dipasang untuk SEMUA request HTTP, bukan hanya /api:
        # transaksi yang ditinggalkan harus dibuang tanpa memandang jalur.
        # `id(scope)` cukup unik selama request ini hidup dan tidak butuh
        # import tambahan.
        token = begin_request_scope(str(id(scope)))
        try:
            return await self._handle(scope, receive, send, path, token)
        finally:
            # Jaring terakhir: bila handler menulis lalu melempar sebelum
            # `commit`, transaksi itu TIDAK boleh diwariskan ke request
            # berikutnya di thread yang sama (SQLite memakai satu transaksi
            # implisit per koneksi).
            #
            # Baris ini hanya menjangkau koneksi milik thread event loop
            # (endpoint async). Endpoint sinkron berjalan di threadpool, jadi
            # pembersihannya dilakukan oleh pemeriksaan scope di
            # `ThreadLocalConnection._conn()` — di sana `in_transaction`
            # diperiksa pada koneksi thread pekerja yang sebenarnya.
            CONN.discard_pending()
            end_request_scope(token)

    async def _handle(self, scope, receive, send, path, token):
        if not path.startswith("/api") or path in _LOG_SKIP:
            return await self.app(scope, receive, send)

        started = time.perf_counter()
        status = {"code": 0}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message.get("status", 0)
            await send(message)

        method = scope.get("method", "?")
        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:  # noqa: BLE001 — diteruskan ke handler FastAPI
            ms = int((time.perf_counter() - started) * 1000)
            logging_setup.event(
                REQUEST_LOG, "request_error", f"{method} {path} gagal",
                level=logging_setup.logging.ERROR,
                method=method, path=path, ms=ms, exc=type(exc).__name__)
            raise
        code = status["code"]
        ms = int((time.perf_counter() - started) * 1000)
        level = (logging_setup.logging.ERROR if code >= 500
                 else logging_setup.logging.WARNING if code >= 400
                 else logging_setup.logging.INFO)
        logging_setup.event(
            REQUEST_LOG, "request", f"{method} {path} -> {code}",
            level=level, method=method, path=path, status=code, ms=ms)
        # KPI §11: p95 & laju 5xx dihitung dari request nyata, bukan tebakan.
        # `record` hanya menyimpan angka (tanpa path ber-parameter mentah),
        # jadi tidak ada data pengguna yang masuk ke metrik.
        metrics.record(path, ms, code)


app.add_middleware(RequestLogMiddleware)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    """500 tanpa membocorkan detail internal ke klien.

    Klien menerima pesan umum + `request_id` yang bisa dicocokkan dengan log;
    traceback lengkap hanya ada di log server.
    """
    rid = uuid.uuid4().hex[:12]
    logging_setup.event(
        LOG, "unhandled_error", f"{type(exc).__name__}: {exc}",
        level=logging_setup.logging.ERROR, request_id=rid,
        method=request.method, path=request.url.path, exc=type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "internal_error",
                            "message": "Terjadi kesalahan pada server. "
                                       f"Kode laporan: {rid}",
                            "request_id": rid}})


# Koneksi per-thread: setiap request di threadpool punya cursor sendiri, jadi
# endpoint lain tidak perlu menunggu selama satu request menjalankan I/O LLM.
# Log terstruktur (JSON per baris). Dipasang sebelum seeding supaya kegagalan
# startup pun tercatat sebagai JSON, bukan traceback mentah.
LOG = logging_setup.get_logger("app")
logging_setup.event(LOG, "boot", "SafeDesk starting",
                    db=DEFAULT_DB_PATH, version=AGENT_VERSION)

CONN = connect_threadlocal(DEFAULT_DB_PATH)

# Integritas DB diperiksa sekali saat start: kerusakan halaman SQLite atau
# foreign key yang menggantung jauh lebih murah ditemukan di log startup
# daripada setelah pengguna melaporkan "data hilang".
_health = logging_setup.integrity_check(CONN)
logging_setup.event(
    LOG, "db_integrity",
    "database integrity ok" if _health["ok"] else "database integrity PROBLEM",
    level=logging_setup.logging.INFO if _health["ok"] else logging_setup.logging.ERROR,
    ok=_health["ok"], detail=str(_health.get("detail"))[:200],
    foreign_key_violations=_health.get("foreign_key_violations", 0))

seed_if_empty(CONN)
# Add seed docs that were introduced after the DB was first created (e.g.
# company profile, Kenari partner, salary structure, workflows).
ensure_seed_docs(CONN)
# Add seed users introduced later (e.g. role `finance` for approval > Rp1jt).
ensure_seed_users(CONN)
# Saldo cuti, absensi, dan slip contoh (idempotent) — data personal Sprint 1.
try:
    from seed import seed_personal
    seed_personal(CONN, db_path=DEFAULT_DB_PATH)
except Exception as _exc:  # seeding personal tidak boleh menggagalkan startup
    logging_setup.event(LOG, "seed_personal_skipped", str(_exc),
                        level=logging_setup.logging.WARNING)
# Lengkapi metadata pesan lama (status kartu draft + saran lanjutan) supaya
# riwayat yang sudah ada ikut benar tanpa menunggu sesi baru (idempotent).
backfill_chat_meta(CONN)
# Bereskan pengajuan pending yang macet karena tahap self-approval (idempotent).
portal.repair_self_stuck(CONN)
# Buang tahap Finance dari reimbursement lama yang nominalnya di bawah ambang
# (doc_approval_matrix). Idempotent; baris baru sudah benar sejak dibuat.
portal.repair_overrouted_finance(CONN)
# Cadangan harian (retensi 7 berkas). Thread daemon; kegagalan hanya dicatat.
try:
    if backup.start_daily(DEFAULT_DB_PATH):
        logging_setup.event(LOG, "backup_scheduled", "daily backup started",
                            dir=backup.backup_dir(DEFAULT_DB_PATH),
                            keep=backup.DEFAULT_KEEP)
except Exception as _exc:
    logging_setup.event(LOG, "backup_schedule_skipped", str(_exc),
                        level=logging_setup.logging.WARNING)

# Reminder proaktif (Sprint 3.5): thread daemon, idempotent, tidak menahan lock.
try:
    import reminders
    reminders.start_scheduler(DEFAULT_DB_PATH)
except Exception as _exc:  # scheduler tidak boleh menggagalkan startup
    logging_setup.event(LOG, "reminders_skipped", str(_exc),
                        level=logging_setup.logging.WARNING)

# FastAPI runs sync endpoints in a threadpool, but this app shares a single
# SQLite connection. Concurrent requests would interleave cursor usage on the
# same connection (sqlite3 allows only one active cursor writer at a time) and
# can raise 500s under rapid sends. A coarse module-level lock serializes DB
# access — simplest correct fix for an internal service.
_DB_LOCK = threading.RLock()


# Rate-limit in-memory + resolusi IP klien (lihat ratelimit.py).
from ratelimit import (  # noqa: E402
    CHAT_RATE_LIMIT, LOGIN_RATE_LIMIT,
    REQUEST_RATE_LIMIT, DECIDE_RATE_LIMIT, READ_RATE_LIMIT, PUBLIC_RATE_LIMIT,
    client_ip, rate_limit as _rate_limit,
)

def db_locked(func):
    """Decorator: run the route under the global DB lock.

    `finally` me-rollback transaksi yang masih terbuka bila handler melempar
    sebelum `commit`. SQLite memakai satu transaksi implisit per koneksi, dan
    koneksi dimiliki per-thread sementara thread FastAPI dipakai ulang:
    tanpa rollback di sini, tulisan parsial dari request yang gagal akan
    di-commit oleh request berikutnya di thread yang sama (terbukti di
    `test_db_concurrency.py`). Rollback di titik kegagalan lebih cepat dan
    lebih pasti daripada menunggu pemeriksaan scope di request berikutnya.
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        with _DB_LOCK:
            try:
                return func(*args, **kwargs)
            finally:
                CONN.discard_pending()

    return wrapper


def get_conn():
    return CONN


def _now_iso() -> str:
    """Waktu UTC berformat Z — sama dengan kolom waktu lain di DB."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _session_live(token_hash: str) -> bool:
    """Token sah secara kriptografis tapi bisa saja sudah dicabut (logout)."""
    row = CONN.execute(
        "SELECT revoked_at, expires_at FROM sessions WHERE token_hash = ?",
        (token_hash,)).fetchone()
    if row is None:
        # Token yang tidak pernah tercatat berarti dibuat sebelum tabel sesi
        # ada (token lama) — tetap ditolak: semua login baru mencatat sesi.
        return False
    return not row["revoked_at"]


def parse_token(authorization: str | None = Header(default=None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="missing_authorization")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="invalid_authorization")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=401, detail="invalid_token")
    try:
        claims = auth.parse_token(DEFAULT_DB_PATH, token)
    except auth.TokenError as e:
        raise HTTPException(status_code=401, detail=e.code)
    with _DB_LOCK:
        if not _session_live(claims["hash"]):
            raise HTTPException(status_code=401, detail="session_revoked")
        user = get_user(CONN, claims["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="invalid_token")
    # Akun nonaktif tidak boleh memakai token lama yang belum kedaluwarsa.
    # `deactivate()` sudah mencabut sesi, ini lapisan kedua untuk sesi yang
    # dibuat sebelum perbaikan ini (atau bila pencabutan gagal di tengah jalan).
    if not users_admin.is_active(user):
        raise HTTPException(status_code=401, detail="account_inactive")
    return user


def _role_guard(action: str, detail: str, name: str):
    """Dependency factory yang membaca izin dari `rbac.PERMISSIONS`.

    Satu sumber aturan: menambah/mengubah peran cukup di `rbac.py`, bukan
    mencari `role in (...)` yang tersebar di banyak endpoint. `name` dipakai
    apa adanya supaya `Depends(...)` di intropeksi (tes RBAC, log, OpenAPI)
    tetap terbaca sebagai `require_admin`/`require_hr`.
    """
    def guard(user: dict = Depends(parse_token)):
        if not rbac.allowed(action, user["role"]):
            raise HTTPException(status_code=403, detail=detail)
        return user
    guard.__name__ = name
    guard.rbac_action = action
    return guard


require_admin = _role_guard("users.manage", "admin_required", "require_admin")
"""Hanya admin: kelola karyawan, workspace, cadangan."""

require_hr = _role_guard("docs.manage", "hr_or_admin_required", "require_hr")
"""HR + admin: mengelola knowledge base dan membaca audit."""


# ---- Backup harian (admin) ----
@app.get("/api/admin/backups")
@db_locked
def list_backups(admin: dict = Depends(require_admin)):
    """Daftar cadangan + status retensi. Tidak membaca isi DB.

    Path absolut server sengaja TIDAK dikirim: klien hanya butuh nama berkas
    dan ukuran, sedangkan lokasi penyimpanan membocorkan struktur direktori
    (dan berubah-ubah antar lingkungan). Sama seperti `file_ref` slip gaji.
    """
    return {"keep": backup.DEFAULT_KEEP,
            "backups": backup.list_backups(DEFAULT_DB_PATH)}


@app.post("/api/admin/backups")
@db_locked
def run_backup(admin: dict = Depends(require_admin)):
    """Jalankan cadangan sekarang (dipakai cron dan tombol di UI)."""
    _rate_limit("backup:run:" + admin["id"], REQUEST_RATE_LIMIT)
    try:
        res = backup.run(DEFAULT_DB_PATH)
    except Exception as exc:  # disk penuh / izin -> jangan 500 mentah
        logging_setup.event(LOG, "backup_failed", str(exc),
                            level=logging_setup.logging.ERROR)
        raise HTTPException(status_code=503, detail={
            "code": "backup_failed",
            "message": "Cadangan gagal dibuat: " + str(exc)})
    logging_setup.event(LOG, "backup_created", res["name"],
                        bytes=res["bytes"], removed=len(res["removed"]))
    portal.audit_event(CONN, admin, "backup_create",
                       f"Cadangan DB dibuat: {res['name']}",
                       extra={"bytes": res["bytes"],
                              "removed": res["removed"]})
    CONN.commit()
    # `path` hanya untuk log/operator; jangan pernah dikirim ke klien.
    return {k: v for k, v in res.items() if k != "path"}


@app.get("/api/admin/kpi")
@db_locked
def admin_kpi(admin: dict = Depends(require_hr)):
    """KPI §11: latensi runtime + rasio grounded/refusal/approve dari DB.

    HR (bukan hanya admin) boleh melihat: angka ini tentang mutu jawaban dan
    kecepatan approval, bukan data pribadi siapa pun.

    `llm_usage` ditambahkan di sini (bukan di `metrics`) karena sumbernya
    penghitung in-memory milik modul `llm`: jumlah panggilan, token prompt/
    completion yang dilaporkan provider, ukuran payload terbesar, dan berapa
    kali jawaban terpotong. Tanpa ini, "payload membesar" hanya terasa sebagai
    keluhan konteks jebol dan tidak pernah bisa diukur.
    """
    _rate_limit("kpi:" + admin["id"], READ_RATE_LIMIT)
    out = metrics.report(CONN)
    out["llm_usage"] = llm.usage_stats()
    return out


@app.get("/api/health")
def health():
    """Health + status subsistem. `reminders` = thread scheduler hidup?"""
    sched = False
    detail = None
    try:
        import reminders
        sched = reminders.is_running()
        detail = reminders.status()
    except Exception:
        sched = False
    # `db` diperiksa ulang di sini (bukan hanya saat start): DB yang halaman
    # filenya rusak di tengah operasi harus terlihat dari probe uptime.
    db_state = "ok"
    try:
        row = CONN.execute("PRAGMA quick_check;").fetchone()
        if not row or row[0] != "ok":
            db_state = "degraded"
    except Exception:
        db_state = "unavailable"
    return {"status": "ok" if db_state == "ok" else "degraded",
            "agent": AGENT_VERSION,
            "llm": llm.status()["enabled"], "reminders": sched,
            "reminders_detail": detail, "db": db_state,
            "backup": {"running": backup.is_running(),
                       "keep": backup.DEFAULT_KEEP,
                       "count": len(backup.list_backups(DEFAULT_DB_PATH))}}


@app.get("/api/public/stats")
@db_locked
def public_stats(request: Request):
    """Agregat untuk halaman publik (landing). Tanpa autentikasi.

    Kenapa tanpa auth: dipanggil sebelum siapa pun masuk. Karena itu isinya
    dibatasi ke hitungan — tidak ada id, nama, email, judul, atau isi dokumen.
    Lihat `portal.public_stats()` untuk batas privasi lengkapnya.

    Rate limit per IP memakai bucket `PUBLIC_RATE_LIMIT`, lebih longgar dari
    READ karena satu kunjungan bisa memicu beberapa permintaan (prefetch,
    retry setelah navigasi). Cache 5 menit: angka ini bergerak secepat
    dokumen diunggah, bukan secepat halaman dimuat.
    """
    _rate_limit("public:stats:" + client_ip(request), PUBLIC_RATE_LIMIT)
    try:
        data = portal.public_stats(CONN)
    except Exception as exc:
        # Jangan pernah mengarang angka saat DB bermasalah: klien akan jatuh
        # ke fallback statis dan menandainya sebagai "angka contoh".
        logging_setup.event(LOG, "public_stats_failed", str(exc),
                            level=logging_setup.logging.ERROR)
        raise HTTPException(status_code=503, detail={
            "code": "stats_unavailable",
            "message": "Statistik sementara tidak tersedia."})
    data["generated_at"] = _now_iso()
    return JSONResponse(content=data, headers={
        "Cache-Control": "public, max-age=300"})


@app.get("/api/llm/status")
def llm_status():
    # never exposes the API key
    return llm.status(detail=False)


@app.post("/api/auth/login")
@db_locked
def login(payload: dict = Body(...), request: Request = None):
    host = client_ip(request)
    uid = payload.get("user_id") or payload.get("id")
    if not uid:
        # allow login by name
        name = payload.get("name")
        if name:
            row = CONN.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()
            uid = row["id"] if row else None
    # Bucket per (IP, user): satu orang yang klik berulang tidak mengunci
    # pengguna lain di balik proxy yang sama.
    _rate_limit("login:" + host + ":" + str(uid or "-"), LOGIN_RATE_LIMIT)
    user = get_user(CONN, uid) if uid else None
    if not user:
        raise HTTPException(status_code=401, detail="unknown_user")
    if not users_admin.is_active(user):
        # 401 (bukan 403): dari sisi klien ini sama dengan kredensial yang
        # tidak berlaku, dan UI sudah menangani 401 dengan kembali ke login.
        raise HTTPException(status_code=401, detail="account_inactive")
    # Token bertanda tangan (HMAC) + masa berlaku; yang disimpan hanya hash.
    issued = auth.issue_token(DEFAULT_DB_PATH, user["id"])
    ua = (request.headers.get("user-agent") if request else None) or None
    # Sudah di dalam @db_locked — jangan ambil lock lagi (dulu menyebabkan
    # deadlock karena `threading.Lock` tidak reentrant).
    CONN.execute(
        """INSERT INTO sessions
           (token_hash,user_id,issued_at,expires_at,revoked_at,user_agent,ip)
           VALUES (?,?,?,?,NULL,?,?)""",
        (auth.token_hash(issued["token"]), user["id"], _now_iso(),
         issued["expires_at"], ua, host))
    CONN.commit()
    return {"token": issued["token"], "user": user,
            "expires_at": issued["expires_at"],
            "ttl_seconds": issued["ttl_seconds"]}


@app.post("/api/auth/logout")
@db_locked
def logout(authorization: str | None = Header(default=None)):
    """Cabut token yang sedang dipakai. Selalu 200 walau token sudah mati."""
    token = (authorization or "")
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if token:
        CONN.execute(
            "UPDATE sessions SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
            (_now_iso(), auth.token_hash(token)))
        CONN.commit()
    return {"ok": True}


@app.post("/api/auth/logout-all")
@db_locked
def logout_all(user: dict = Depends(parse_token)):
    """Cabut SEMUA sesi user ini (dipakai bila token diduga bocor)."""
    cur = CONN.execute(
        "UPDATE sessions SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
        (_now_iso(), user["id"]))
    CONN.commit()
    portal.audit_event(CONN, user, "auth_logout_all",
                       f"Semua sesi dicabut untuk {user['id']}",
                       extra={"revoked": cur.rowcount})
    CONN.commit()
    return {"ok": True, "revoked": cur.rowcount}


@app.get("/api/me")
def me(user: dict = Depends(parse_token)):
    return user


# ---- Data personal (Sprint 1) ----
# Semua endpoint di bawah HANYA mengembalikan data milik user yang login.
@app.get("/api/me/profile")
@db_locked
def me_profile(user: dict = Depends(parse_token)):
    return personal.profile(CONN, user)


@app.get("/api/me/leave-balance")
@db_locked
def me_leave_balance(year: int | None = None, user: dict = Depends(parse_token)):
    _rate_limit("me:leave:" + user["id"], READ_RATE_LIMIT)
    return personal.leave_balance(CONN, user, year)


@app.get("/api/me/payslips")
@db_locked
def me_payslips(user: dict = Depends(parse_token)):
    _rate_limit("me:payslip:" + user["id"], READ_RATE_LIMIT)
    return personal.payslips(CONN, user)


@app.get("/api/me/attendance")
@db_locked
def me_attendance(month: str | None = None, user: dict = Depends(parse_token)):
    _rate_limit("me:attendance:" + user["id"], READ_RATE_LIMIT)
    return personal.attendance_summary(CONN, user, month)


@app.post("/api/me/attendance/clock")
@db_locked
def me_attendance_clock(payload: dict = Body(default={}),
                        user: dict = Depends(parse_token)):
    _rate_limit("me:clock:" + user["id"], DECIDE_RATE_LIMIT)
    data, code = personal.clock(CONN, user, (payload or {}).get("action"))
    return _raise_portal(data, code, ok=(200, 201))


@app.get("/api/org/chart")
@db_locked
def org_chart(user: dict = Depends(parse_token)):
    return personal.org_chart(CONN, user)


@app.get("/api/team")
@db_locked
def my_team(user: dict = Depends(parse_token)):
    return personal.team(CONN, user)


@app.post("/api/payslips")
@db_locked
def publish_payslip(payload: dict = Body(...), user: dict = Depends(parse_token)):
    """HR/admin menerbitkan slip gaji satu karyawan untuk satu periode.

    Penerbitan WAJIB tercatat di audit: ini tindakan yang menulis data
    finansial pribadi karyawan. Sebelumnya hanya unggah berkas (`payslip_file`)
    yang diaudit, sedangkan penerbitan nominalnya tidak meninggalkan jejak.
    """
    _rate_limit("payslip:publish:" + user["id"], REQUEST_RATE_LIMIT)
    data, code = personal.publish_payslip(CONN, user, payload,
                                          db_path=DEFAULT_DB_PATH)
    if code in (200, 201):
        portal.audit_event(
            CONN, user, "payslip_publish",
            f"Menerbitkan slip gaji {data.get('user_id')} periode {data.get('period')}",
            document_id=None,
            extra={"target_user": data.get("user_id"),
                   "period": data.get("period"),
                   "net": data.get("net")})
        CONN.commit()
    return _raise_portal(data, code, ok=(200, 201))


@app.get("/api/payslips")
@db_locked
def list_payslips(period: str | None = None, q: str | None = None,
                  limit: str | None = None, offset: str | None = None,
                  user: dict = Depends(parse_token)):
    """Daftar slip gaji untuk HR/admin (Kelola → Payroll).

    `limit`/`offset` dibaca sebagai `str` supaya nilai aneh jatuh ke default,
    bukan 422 — sama dengan endpoint daftar lain (lihat backend/paging.py).
    """
    _rate_limit("payslip:list:" + user["id"], READ_RATE_LIMIT)
    data, code = personal.list_payslips(CONN, user, period=period, q=q)
    if code != 200:
        return _raise_portal(data, code)
    page = paging.slice_page(data["payslips"], limit, offset, default=100)
    # `count` dari personal adalah jumlah SETELAH filter (= total), sedangkan
    # kontrak paging memakai `count` = jumlah baris di halaman ini. Kunci itu
    # dibuang di sini supaya semua endpoint daftar punya arti yang sama.
    extra = {k: v for k, v in data.items() if k not in ("payslips", "count")}
    return paging.spread(extra, "payslips", page)


@app.get("/api/payslips/export.csv")
@db_locked
def export_payslips_csv(period: str | None = None, q: str | None = None,
                        user: dict = Depends(parse_token)):
    """Ekspor slip (bukan berkas PDF-nya) sesuai filter yang sedang tampil."""
    _rate_limit("payslip:csv:" + user["id"], READ_RATE_LIMIT)
    data, code = personal.list_payslips(CONN, user, period=period, q=q)
    if code != 200:
        raise HTTPException(status_code=code, detail={
            "code": data.get("error", "error"),
            "message": data.get("message", "")})
    header = ["id", "user_id", "nama", "periode", "bruto", "potongan",
              "diterima", "punya_berkas", "diterbitkan"]
    rows = [[i["id"], i["user_id"], i["user_name"], i["period"], i["gross"],
             i["deductions"], i["net"], "YA" if i["has_file"] else "",
             i["published_at"] or ""]
            for i in data["payslips"]]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return _csv_response(f"slip-gaji-{stamp}.csv", header, rows)


@app.post("/api/payslips/file")
@db_locked
async def upload_payslip_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    period: str = Form(...),
    user: dict = Depends(parse_token),
):
    """Lampirkan berkas slip (PDF/gambar) ke slip yang sudah diterbitkan."""
    _rate_limit("payslip:file:" + user["id"], REQUEST_RATE_LIMIT)
    if user["role"] not in ("hr", "admin"):
        raise HTTPException(status_code=403, detail={
            "code": "not_authorized",
            "message": "Hanya HR/admin yang dapat melampirkan berkas slip."})
    declared = (file.headers.get("content-length") if file.headers else None)
    if declared and declared.isdigit() and int(declared) > files.MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="file_too_large")
    raw = await file.read(files.MAX_FILE_BYTES + 1)
    try:
        meta = files.save(DEFAULT_DB_PATH, file.filename or "slip.pdf", raw,
                          kind="payslips")
    except files.FileError as e:
        raise HTTPException(status_code=e.status, detail={
            "code": e.code, "message": e.message})
    data, code = personal.attach_payslip_file(
        CONN, DEFAULT_DB_PATH, user,
        {"user_id": user_id, "period": period, "_file": meta})
    if code != 200:
        # Berkas sudah terlanjur ditulis — jangan tinggalkan sampah di disk.
        files.delete(DEFAULT_DB_PATH, meta["ref"])
    else:
        portal.audit_event(CONN, user, "payslip_file",
                           f"Melampirkan berkas slip {period} untuk {user_id}",
                           extra={"user_id": user_id, "period": period,
                                  "bytes": meta["bytes"], "sha256": meta["sha256"]})
        CONN.commit()
    return _raise_portal(data, code, ok=(200, 201))


@app.get("/api/me/payslips/{slip_id}/file")
@db_locked
def download_payslip(slip_id: int, user: dict = Depends(parse_token)):
    """Unduh berkas slip milik sendiri."""
    _rate_limit("payslip:get:" + user["id"], READ_RATE_LIMIT)
    data, code, err = personal.payslip_file(CONN, DEFAULT_DB_PATH, user, slip_id)
    if code != 200:
        raise HTTPException(status_code=code, detail={
            "code": err["error"], "message": err["message"]})
    return FileResponse(data["path"], media_type=data["content_type"],
                        filename=data["filename"])


@app.get("/api/dashboard")
@db_locked
def api_dashboard(user: dict = Depends(parse_token)):
    return portal.dashboard(CONN, user)


@app.get("/api/schedule")
@db_locked
def api_schedule(user: dict = Depends(parse_token)):
    return portal.schedule(CONN, user)


@app.get("/api/payroll")
@db_locked
def api_payroll(user: dict = Depends(parse_token)):
    return portal.payroll(CONN, user)


@app.get("/api/directory")
@db_locked
def api_directory(user: dict = Depends(parse_token)):
    return portal.directory(CONN, user)


@app.get("/api/company")
@db_locked
def api_company(user: dict = Depends(parse_token)):
    return portal.company(CONN, user)


@app.get("/api/projects")
@db_locked
def api_projects(user: dict = Depends(parse_token)):
    return portal.projects(CONN, user)


def _raise_portal(data, code, ok=(200, 201)):
    """Ubah hasil portal.py menjadi respons HTTP yang benar.

    - error  -> HTTPException dengan detail terstruktur {code, message, fields}
    - sukses -> JSONResponse supaya status 201 tidak "downgrade" jadi 200
    """
    if code not in ok:
        detail = {"code": data.get("error", "error"),
                  "message": data.get("message") or data.get("error", "error")}
        if data.get("fields"):
            detail["fields"] = data["fields"]
        raise HTTPException(status_code=code, detail=detail)
    return JSONResponse(content=data, status_code=code)


@app.get("/api/requests/rules")
@db_locked
def api_request_rules(type: str = "cuti", amount: str | None = None,
                      user: dict = Depends(parse_token)):
    """Pratinjau rantai persetujuan untuk formulir pengajuan."""
    _rate_limit("req:rules:" + user["id"], READ_RATE_LIMIT)
    data, code = portal.request_rules(CONN, user, type, amount)
    return _raise_portal(data, code)


@app.post("/api/requests")
@db_locked
def api_create_request(payload: dict = Body(...), user: dict = Depends(parse_token)):
    _rate_limit("req:create:" + user["id"], REQUEST_RATE_LIMIT)
    data, code = portal.create_request(CONN, user, payload)
    return _raise_portal(data, code, ok=(200, 201))


@app.get("/api/requests")
@db_locked
def api_list_requests(user: dict = Depends(parse_token),
                      status: str | None = None, type: str | None = None,
                      limit: str | None = None, offset: str | None = None,
                      offset_mine: str | None = None,
                      offset_approval: str | None = None,
                      offset_all: str | None = None,
                      status_all: str | None = None,
                      type_all: str | None = None,
                      sort: str | None = None,
                      dir: str | None = None):
    """Pengajuan dalam tiga keranjang (mine/approval/all) + paging per keranjang.

    `?status=`/`?type=` dulu diterima tetapi tidak berpengaruh apa pun: UI
    mengirimnya, server mengabaikan, jadi filter tampak "tidak bekerja".

    Offset PER KERANJANG (`offset_mine`, `offset_approval`, `offset_all`) ada
    karena satu offset bersama salah secara diam-diam: memaging tabel "Semua
    Pengajuan" ikut menggeser antrean "Menunggu Saya", sehingga kartu approval
    yang belum diputuskan bisa hilang dari layar. `offset` tunggal tetap
    diterima sebagai fallback untuk klien lama.

    Filter khusus tabel "Semua Pengajuan" dipisah (`status_all`, `type_all`)
    karena `status`/`type` berlaku ke SEMUA keranjang: memilih chip "approved"
    dulu ikut mengosongkan antrean "Menunggu Saya", sehingga approver melihat
    layar kosong padahal ada pengajuan yang menunggu keputusannya.

    Urutan tabel `all` juga dikerjakan server (`sort`, `dir`). Sebelum ini UI
    mengurutkan di klien, yang hanya benar selama seluruh data ada di satu
    halaman: begitu server memotong, klik "Status ⇅" mengurutkan 100 baris
    pertama saja dan hasilnya terlihat seperti data yang salah.

    `summary` dihitung dari SELURUH keranjang `all` SEBELUM filter/paging:
    grafik sebaran harus menggambarkan populasi, bukan satu halaman.
    """
    _rate_limit("req:list:" + user["id"], READ_RATE_LIMIT)
    data = portal.list_requests(CONN, user)

    def _filter(rows, st=None, tp=None):
        if st:
            rows = [r for r in rows if (r.get("status") or "") == st]
        if tp:
            rows = [r for r in rows if (r.get("type") or "") == tp]
        return rows

    lim, off = paging.window(limit, offset, default=100)

    def _bucket_offset(raw):
        if raw is None:
            return off
        return paging.window(limit, raw, default=100)[1]

    offsets = {
        "mine": _bucket_offset(offset_mine),
        "approval": _bucket_offset(offset_approval),
        "all": _bucket_offset(offset_all),
    }

    sort_key = (sort or "id").strip().lower()
    sort_dir = -1 if (dir or "").strip().lower() in ("desc", "descending", "-1") else 1
    _SORT_FIELDS = {
        "id": lambda r: int(r.get("id") or 0),
        "type": lambda r: str(r.get("type") or ""),
        "user_id": lambda r: str(r.get("user_name") or r.get("user_id") or ""),
        "status": lambda r: str(r.get("status") or ""),
    }

    out, totals, pages = {}, {}, {}
    for key in ("mine", "approval", "all"):
        if key == "all":
            rows = _filter(data.get(key) or [], status_all, type_all)
            getter = _SORT_FIELDS.get(sort_key)
            if getter:
                rows = sorted(rows, key=getter, reverse=(sort_dir < 0))
        else:
            rows = _filter(data.get(key) or [], status, type)
        totals[key] = len(rows)
        start = offsets[key]
        out[key] = rows[start:start + lim]
        pages[key] = {"total": totals[key], "limit": lim, "offset": start,
                      "count": len(out[key]),
                      "has_more": start + len(out[key]) < totals[key]}

    raw_all = data.get("all") or []
    summary = {
        "by_status": {st: sum(1 for r in raw_all if (r.get("status") or "") == st)
                      for st in ("pending", "approved", "rejected", "cancelled")},
        "by_type": {tp: sum(1 for r in raw_all if (r.get("type") or "") == tp)
                    for tp in ("cuti", "reimbursement")},
        "total": len(raw_all),
    }

    return {"mine": out["mine"], "approval": out["approval"], "all": out["all"],
            "count": {k: len(v) for k, v in out.items()},
            "total": totals,
            "limit": lim, "offset": off,
            "offset_by_bucket": offsets,
            "pages": pages,
            "has_more": any(p["has_more"] for p in pages.values()),
            "summary": summary,
            "filters": {"status": status or "", "type": type or "",
                        "status_all": status_all or "", "type_all": type_all or "",
                        "sort": sort_key, "dir": "desc" if sort_dir < 0 else "asc"}}


@app.post("/api/requests/{req_id}/decide")
@db_locked
def api_decide_request(req_id: int, payload: dict = Body(...),
                       user: dict = Depends(parse_token)):
    _rate_limit("req:decide:" + user["id"], DECIDE_RATE_LIMIT)
    action = payload.get("action")
    note = payload.get("note", "")
    data, code = portal.decide_request(CONN, user, req_id, action, note)
    return _raise_portal(data, code)


@app.post("/api/requests/{req_id}/cancel")
@db_locked
def api_cancel_request(req_id: int, user: dict = Depends(parse_token)):
    _rate_limit("req:cancel:" + user["id"], DECIDE_RATE_LIMIT)
    data, code = portal.cancel_request(CONN, user, req_id)
    return _raise_portal(data, code)


@app.post("/api/requests/{req_id}/override")
@db_locked
def api_override_request(req_id: int, payload: dict = Body(...),
                         user: dict = Depends(parse_token)):
    _rate_limit("req:override:" + user["id"], DECIDE_RATE_LIMIT)
    data, code = portal.override_request(
        CONN, user, req_id, payload.get("action"), payload.get("note", ""))
    return _raise_portal(data, code)


@app.get("/api/notifications")
@db_locked
def api_notifications(category: str | None = None, limit: int = 50,
                      offset: int = 0, user: dict = Depends(parse_token)):
    _rate_limit("notif:" + user["id"], READ_RATE_LIMIT)
    if category and category != "all" and category not in portal.NOTIF_CATEGORIES:
        raise HTTPException(status_code=422, detail="invalid_category")
    return portal.notifications(CONN, user, category=category, limit=limit,
                                offset=offset)


# --- Preferensi (Sprint 3.2) ---
# Didaftarkan SEBELUM route ber-parameter `/api/notifications/{notif_id}` supaya
# path statis tidak tertangkap sebagai id (bug yang pernah terjadi di
# /api/chat/sessions).
@app.get("/api/notifications/prefs")
@db_locked
def api_notif_prefs(user: dict = Depends(parse_token)):
    return portal.notif_prefs(CONN, user)


@app.put("/api/notifications/prefs")
@db_locked
def api_notif_prefs_set(payload: dict = Body(...),
                        user: dict = Depends(parse_token)):
    _rate_limit("notif:prefs:" + user["id"], READ_RATE_LIMIT)
    data, code = portal.set_notif_prefs(CONN, user, payload)
    return _raise_portal(data, code, ok=(200,))


@app.post("/api/notifications/read-all")
@db_locked
def api_notif_read_all(payload: dict = Body(default={}),
                       user: dict = Depends(parse_token)):
    category = (payload or {}).get("category")
    if category and category != "all" and category not in portal.NOTIF_CATEGORIES:
        raise HTTPException(status_code=422, detail="invalid_category")
    return portal.mark_all_read(CONN, user, category=category)


# --- SSE (Sprint 3.4) ---
# Lock DB TIDAK boleh ditahan selama streaming: baca snapshot singkat, lepas,
# lalu yield. Koneksi ditutup ~55 detik supaya proxy/Vercel tidak memutus paksa
# tanpa event terakhir; EventSource klien otomatis reconnect.
# Batas koneksi SSE per user: tanpa ini satu tab yang macet (atau klien nakal)
# bisa membuka puluhan stream dan menghabiskan kuota koneksi proxy.
_SSE_ACTIVE = {}          # uid -> [slot_token, ...]
_SSE_LOCK = threading.Lock()
_SSE_SEQ = 0
SSE_MAX_PER_USER = int(os.environ.get("NOHALU_SSE_MAX_PER_USER", "4"))


def _sse_acquire(uid, ttl):
    """Ambil slot stream; kuota penuh → evict slot tertua (takeover).

    Kenapa takeover, bukan 429: proxy (Vercel) bisa menahan koneksi upstream
    beberapa detik setelah tab browser ditutup. Kalau kuota penuh dijawab 429,
    user sah yang cuma membuka/refresh beberapa tab bisa terkunci total sampai
    TTL habis — persis bug produksi "stream selalu 429".

    Takeover tetap membatasi sumber daya: jumlah stream hidup per user tidak
    pernah melebihi SSE_MAX_PER_USER, karena stream tertua dipaksa berhenti
    (generator-nya melihat tokennya sudah tidak ada di `_SSE_ACTIVE`).

    Return (token, evicted_count).
    """
    global _SSE_SEQ
    now_m = time.monotonic()
    with _SSE_LOCK:
        live = [t for t in _SSE_ACTIVE.get(uid, []) if now_m - t[1] < ttl]
        evicted = 0
        while len(live) >= max(1, SSE_MAX_PER_USER):
            live.pop(0)          # stream tertua: klien paling mungkin sudah hilang
            evicted += 1
        _SSE_SEQ += 1
        token = (_SSE_SEQ, now_m)   # unik walau dua stream mulai di detik sama
        live.append(token)
        _SSE_ACTIVE[uid] = live
        return token, evicted


def _sse_alive(uid, token):
    """True selama token masih memegang slot (False = sudah di-evict/release)."""
    with _SSE_LOCK:
        return token in (_SSE_ACTIVE.get(uid) or [])


def _sse_release(uid, token=None):
    with _SSE_LOCK:
        live = _SSE_ACTIVE.get(uid) or []
        if token is None:
            if live:
                live.pop()
        else:
            live = [t for t in live if t != token]
        if live:
            _SSE_ACTIVE[uid] = live
        else:
            _SSE_ACTIVE.pop(uid, None)


@app.get("/api/notifications/stream")
async def api_notif_stream(request: Request, user: dict = Depends(parse_token)):
    """Server-Sent Events: kirim `unread` setiap kali jumlahnya berubah.

    Async generator + `asyncio.sleep`: satu koneksi SSE tidak memakai thread
    dari threadpool (kalau sinkron, 40 klien bisa menghabiskan semua worker dan
    request lain ikut menggantung). Query DB dijalankan singkat di threadpool
    dengan `_DB_LOCK`, tidak pernah ditahan selama streaming.
    """
    import asyncio
    from fastapi.responses import StreamingResponse
    from starlette.concurrency import run_in_threadpool

    def _snapshot():
        with _DB_LOCK:
            return portal.unread_count(CONN, user["id"])

    # Bisa diperkecil lewat env agar test tidak menunggu 55 detik.
    max_s = float(os.environ.get("NOHALU_SSE_MAX_SECONDS", "30"))
    tick = float(os.environ.get("NOHALU_SSE_TICK", "2.5"))
    ping_every = float(os.environ.get("NOHALU_SSE_PING", "25"))

    slot, evicted = _sse_acquire(user["id"], max_s + 30)
    if evicted:
        # Stream lama dipaksa berhenti oleh stream baru (tab ditutup/refresh).
        logging_setup.event(LOG, "sse_evicted",
                            "slot SSE lama diambil alih",
                            count=evicted, user_id=user["id"])

    async def gen():
        last = None
        loop = asyncio.get_event_loop()
        started = loop.time()
        last_ping = started
        try:
            while loop.time() - started < max_s:
                # Slot sudah di-evict stream lain → berhenti, supaya jumlah
                # stream hidup tidak pernah melebihi kuota.
                #
                # Deteksi putus TIDAK memakai `request.is_disconnected()`:
                # hasilnya bergantung pada server/proxy dan bisa True prematur
                # (stream sah ikut mati). Yang dipakai adalah heartbeat di
                # bawah: `send` ke socket yang sudah ditutup langsung gagal →
                # CancelledError/OSError → `finally` melepas slot dalam ~1 tick.
                if not _sse_alive(user["id"], slot):
                    break
                n = await run_in_threadpool(_snapshot)
                if n != last:
                    last = n
                    yield "event: unread\ndata: " + json.dumps({"unread": n}) + "\n\n"
                now_m = loop.time()
                if now_m - last_ping >= ping_every:
                    last_ping = now_m
                    yield ": ping\n\n"
                # Heartbeat tiap tick: (a) membuat kegagalan `send` langsung
                # terlihat saat klien hilang → slot cepat dilepas, (b) menjaga
                # proxy tidak menutup koneksi karena idle.
                yield ": hb\n\n"
                await asyncio.sleep(tick)
            yield "event: reconnect\ndata: {}\n\n"
        except (asyncio.CancelledError, OSError):
            # CancelledError: server shutdown / batal; OSError: socket sudah
            # ditutup klien saat `send`. Dua-duanya berarti slot harus lepas.
            return
        finally:
            # Wajib: tanpa ini kuota per-user bocor setiap kali stream ditutup.
            _sse_release(user["id"], slot)

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform",
                 "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.post("/api/notifications/{notif_id}/read")
@db_locked
def api_notification_read(notif_id: int, user: dict = Depends(parse_token)):
    return portal.mark_notification_read(CONN, user, notif_id)


# --- Pengumuman HR (Sprint 3.3) ---
@app.get("/api/announcements")
@db_locked
def api_announcements(limit: int = 30, offset: int = 0,
                      user: dict = Depends(parse_token)):
    _rate_limit("announce:" + user["id"], READ_RATE_LIMIT)
    return portal.announcements(CONN, user, limit=limit, offset=offset)


@app.post("/api/announcements/read-all")
@db_locked
def api_announcements_read_all(user: dict = Depends(parse_token)):
    return portal.mark_all_announcements_read(CONN, user)


@app.post("/api/announcements/{doc_id}/read")
@db_locked
def api_announcement_read(doc_id: str, user: dict = Depends(parse_token)):
    data, code = portal.mark_announcement_read(CONN, user, doc_id)
    return _raise_portal(data, code, ok=(200,))


@app.post("/api/admin/reminders/run")
@db_locked
def api_run_reminders(admin: dict = Depends(require_admin)):
    """Jalankan job reminder sekarang (admin). Dipakai ops & verifikasi deploy.

    Job normal berjalan otomatis tiap 15 menit; endpoint ini hanya memicu
    putaran tambahan dan tetap idempotent (dedupe_key).
    """
    import reminders
    jobs = reminders.run_once(CONN)
    return {"jobs": jobs, "status": reminders.status()}


# --- Digest HR (Sprint 3.6): pertanyaan yang tidak terjawab ---
@app.get("/api/audit/refusals/digest")
@db_locked
def api_refusal_digest(days: int = 30, top: int = 10, offset: int = 0,
                       admin: dict = Depends(require_hr)):
    """Daftar pertanyaan yang tidak terjawab, untuk memperbaiki knowledge base.

    `offset` ditambahkan karena `top` saja berarti refusal ke-51 dst. tidak
    pernah bisa dibaca: halaman ini dipakai justru untuk menyisir SEMUA
    pertanyaan gagal, dan yang paling sering muncul belum tentu yang paling
    perlu diperbaiki (satu pertanyaan aneh yang muncul sekali bisa menandakan
    dokumen hilang). `total` di sini = jumlah baris grup (pertanyaan × alasan),
    bukan jumlah kejadian; jumlah kejadian ada di `total_refusals`.
    """
    days = max(1, min(int(days), 180))
    top = max(1, min(int(top), 50))
    offset = max(0, int(offset))
    since = (datetime.datetime.now(datetime.timezone.utc)
             - datetime.timedelta(days=days)).isoformat().replace("+00:00", "Z")
    rows = CONN.execute(
        """SELECT question, refusal_reason, COUNT(*) AS n
           FROM audit_log
           WHERE refusal_reason IS NOT NULL AND timestamp >= ?
           GROUP BY question, refusal_reason
           ORDER BY n DESC, question ASC LIMIT ? OFFSET ?""",
        (since, top, offset),
    ).fetchall()
    items = [{"question": r["question"], "reason": r["refusal_reason"],
              "count": r["n"]} for r in rows]
    # Agregat HARUS dari seluruh rentang, bukan dari top-N: dulu `by_reason`
    # dijumlahkan dari `items` saja sehingga total ikut mengecil saat `top`
    # kecil (dashboard melaporkan angka yang salah).
    agg = CONN.execute(
        """SELECT COALESCE(refusal_reason,'unknown') AS reason, COUNT(*) AS n
           FROM audit_log
           WHERE refusal_reason IS NOT NULL AND timestamp >= ?
           GROUP BY reason""",
        (since,),
    ).fetchall()
    by_reason = {r["reason"]: r["n"] for r in agg}
    unique_q = CONN.execute(
        """SELECT COUNT(*) AS n FROM (
             SELECT DISTINCT question FROM audit_log
             WHERE refusal_reason IS NOT NULL AND timestamp >= ?)""",
        (since,),
    ).fetchone()["n"]
    groups = CONN.execute(
        """SELECT COUNT(*) AS n FROM (
             SELECT question, refusal_reason FROM audit_log
             WHERE refusal_reason IS NOT NULL AND timestamp >= ?
             GROUP BY question, refusal_reason)""",
        (since,),
    ).fetchone()["n"]
    return {"days": days, "items": items, "by_reason": by_reason,
            # `total` dipertahankan = jumlah kejadian (dipakai judul halaman);
            # `groups` = jumlah baris yang bisa dipaging.
            "total": sum(by_reason.values()), "total_refusals": sum(by_reason.values()),
            "unique_questions": unique_q, "returned": len(items),
            "groups": groups, "limit": top, "offset": offset,
            "has_more": offset + len(items) < groups}


def _iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _clean_session_id(raw) -> str | None:
    """Validasi `session_id` kiriman klien.

    Nilai ini disimpan apa adanya sebagai PRIMARY KEY `chat_sessions.id` dan
    ikut dirangkai ke URL `/api/chat/{session_id}`. Tanpa validasi, klien bisa
    menulis id sembarang (path traversal, ribuan karakter, tanda kutip) yang
    tidak akan pernah cocok dengan id yang dibuat server, sehingga muncul sesi
    hantu di riwayat. Format yang diterima: 1-64 karakter alfanumerik/`_`/`-`,
    diawali alfanumerik — cukup untuk UUID v4 maupun id pendek di test.
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise HTTPException(status_code=400, detail={
            "code": "invalid_session_id",
            "message": "session_id harus berupa teks.",
            "fields": {"session_id": "teks"}})
    sid = raw.strip()
    if not sid:
        return None
    if not _SESSION_ID_RE.match(sid):
        raise HTTPException(status_code=400, detail={
            "code": "invalid_session_id",
            "message": "session_id tidak valid (maks 64 karakter: huruf, angka, _ atau -).",
            "fields": {"session_id": "format tidak valid"}})
    return sid


def _session_owned(user: dict, session_id: str):
    """Ambil sesi + pastikan milik user. Return (row|None, forbidden: bool).

    Admin boleh MEMBACA sesi orang lain lewat GET (kebutuhan audit), tetapi
    TIDAK boleh menulis ke sesi orang lain lewat POST /api/chat — kalau tidak,
    riwayat dan audit milik user lain bisa dicemari.
    """
    row = CONN.execute(
        "SELECT id, user_id FROM chat_sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row is None:
        return None, False
    if row["user_id"] != user["id"]:
        return row, True
    return row, False


def _touch_session(session_id: str, title_hint: str | None = None) -> None:
    """Perbarui updated_at; isi title dari pesan pertama bila masih kosong."""
    now = _iso_now()
    if title_hint:
        # Judul sesi dirender di daftar riwayat sidebar. Judul berisi newline
        # (atau kalimat injeksi) tidak boleh masuk apa adanya: satu baris baru
        # cukup untuk memecah tata letak item, dan teks instruksi tampil
        # sebagai judul percakapan resmi user.
        hint = engine.safe_label(title_hint, 120, "")
        if not hint:
            hint = "(percakapan)"
        CONN.execute(
            "UPDATE chat_sessions SET updated_at = ?, "
            "title = COALESCE(NULLIF(title, ''), ?) WHERE id = ?",
            (now, hint, session_id),
        )
    else:
        CONN.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                     (now, session_id))


@app.post("/api/chat")
def chat(payload: dict = Body(...), user: dict = Depends(parse_token)):
    """Tanya-jawab. Transaksi DB pendek; panggilan LLM di luar lock.

    Sebelumnya seluruh handler berjalan di bawah `_DB_LOCK` dan transaksi tulis
    dibiarkan terbuka selama panggilan LLM (bisa 180 s). Satu pertanyaan lambat
    membekukan semua endpoint dan abort klien meninggalkan `-journal` yang
    mengunci database. Sekarang: commit pesan user dulu, lepas lock, baru LLM.
    """
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="empty_question")
    if len(question) > 8000:
        raise HTTPException(status_code=422, detail={
            "code": "question_too_long",
            "message": "Pertanyaan maksimal 8000 karakter.",
            "fields": {"question": "maks 8000"}})
    _rate_limit("chat:" + user["id"], CHAT_RATE_LIMIT)

    requested = _clean_session_id(payload.get("session_id"))
    session_id = requested or str(uuid.uuid4())
    with _DB_LOCK:
        row, forbidden = _session_owned(user, session_id)
        if forbidden:
            # Jangan bocorkan isi sesi; cukup tolak.
            raise HTTPException(status_code=403, detail="session_forbidden")
        if row is None:
            CONN.execute(
                "INSERT INTO chat_sessions (id, user_id, created_at, updated_at, status)"
                " VALUES (?,?,?,?,?)",
                (session_id, user["id"], _iso_now(), _iso_now(), "active"),
            )
        CONN.execute(
            "INSERT INTO chat_messages (session_id, role, text, citations, created_at)"
            " VALUES (?,?,?,?,?)",
            (session_id, "user", question, "[]", _iso_now()),
        )
        _touch_session(session_id, question)
        CONN.commit()

    # ---- di luar lock: LLM boleh lambat tanpa memblokir endpoint lain ----
    # Agent loop: model memutuskan tool apa yang perlu dipanggil (profil, saldo
    # cuti, dokumen, draft pengajuan). Gerbang keamanan tetap di engine/portal.
    try:
        res = agent.run(CONN, user, question, session_id)
    except Exception as exc:  # jangan tinggalkan sesi tanpa penjelasan
        sys.stderr.write(f"[safedesk] chat failed: {exc}\n")
        with _DB_LOCK:
            CONN.execute(
                "INSERT INTO chat_messages (session_id, role, text, citations,"
                " created_at, interrupted) VALUES (?,?,?,?,?,1)",
                (session_id, "assistant",
                 "Tugas dihentikan karena gangguan pada server. Silakan coba lagi.",
                 "[]", _iso_now()),
            )
            _touch_session(session_id)
            CONN.commit()
        raise HTTPException(status_code=500, detail="chat_failed")

    with _DB_LOCK:
        CONN.execute(
            "INSERT INTO chat_messages (session_id, role, text, citations, created_at,"
            " trace, prepared, suggestions) VALUES (?,?,?,?,?,?,?,?)",
            (session_id, "assistant", res["answer"], json.dumps(res["citations"]),
             _iso_now(),
             json.dumps(res.get("agent_trace") or [], ensure_ascii=False),
             json.dumps(res.get("prepared"), ensure_ascii=False)
             if res.get("prepared") else None,
             # Saran ikut disimpan: dulu hanya hidup di respons HTTP, sehingga
             # membuka ulang sesi dari Riwayat kehilangan blok "Pertanyaan
             # lanjutan" pada jawaban TERAKHIR — padahal justru di situ user
             # paling butuh langkah berikutnya.
             json.dumps(res.get("suggestions") or [], ensure_ascii=False)),
        )
        _touch_session(session_id)
        CONN.commit()
    res["session_id"] = session_id
    return res


# ---- Konfirmasi draft agent (aksi tulis yang menunggu persetujuan) ----
# Aksi draft → tool penyiapnya. Dipakai setelah tombol "Setujui & kirim"
# dijalankan: pilihan saran disusun dari konteks aksi yang BARU selesai, bukan
# dari daftar dokumen acak.
_PREP_TOOL_FOR_ACTION = {
    # Keputusan approver: tanpa pemetaan ini, saran setelah tombol "Setujui &
    # kirim" jatuh ke daftar pengajuan milik SENDIRI — konteks yang salah untuk
    # user yang barusan memutuskan pengajuan orang lain.
    "decide_request": "prepare_decide_request",
    "decide_bulk": "prepare_decide_bulk",
    "override_request": "prepare_override",
    "create_announcement": "prepare_announcement",
    "publish_payslip": "prepare_payslip",
    "save_document": "prepare_document",
    "archive_document": "prepare_document_archive",
    "change_employee": "prepare_employee_change",
}


@app.post("/api/agent/confirm")
@db_locked
def agent_confirm(payload: dict = Body(...), user: dict = Depends(parse_token)):
    """Jalankan draft agent tanpa lewat LLM (tombol "Setujui & kirim").

    Token tetap diverifikasi di agent_tools (milik user, belum dipakai,
    belum kedaluwarsa, sesi sama). Endpoint ini hanya jalan pintas UI supaya
    user tidak perlu mengetik "ya" — jalur LLM tetap tersedia.
    """
    token = (payload.get("token") or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="token_required")
    _rate_limit("agent:confirm:" + user["id"], DECIDE_RATE_LIMIT)
    session_id = _clean_session_id(payload.get("session_id"))
    if not session_id:
        # Klien boleh tidak mengirim session_id (mis. state sesi belum sempat
        # dimuat saat kartu diklik). Token draft SUDAH menyimpan sesi asalnya,
        # dan tanpa nilai ini `_mark_prepared_done` tidak jalan: kartu yang
        # benar-benar sudah jadi pengajuan tetap tampil aktif, sementara pesan
        # "Pengajuan #N tersimpan" tidak pernah tercatat di riwayat.
        own = CONN.execute(
            "SELECT session_id FROM agent_prepared WHERE token = ? AND user_id = ?",
            (token, user["id"])).fetchone()
        if own and own["session_id"]:
            session_id = own["session_id"]
    if session_id:
        # Sesi harus milik user ini; bila belum ada, buat (sama seperti /api/chat).
        row, forbidden = _session_owned(user, session_id)
        if forbidden:
            raise HTTPException(status_code=403, detail="session_forbidden")
        if row is None:
            CONN.execute(
                "INSERT INTO chat_sessions (id, user_id, created_at, updated_at, status)"
                " VALUES (?,?,?,?,?)",
                (session_id, user["id"], _iso_now(), _iso_now(), "active"))
            CONN.commit()
    # Aksi draft dibaca SEBELUM dijalankan: setelah `submit_prepared`, barisnya
    # ditandai `used` dan pilihan saran kehilangan konteks (dulu selalu jatuh ke
    # saran dokumen acak).
    action = None
    try:
        row = CONN.execute(
            "SELECT action FROM agent_prepared WHERE token = ? AND user_id = ?",
            (token, user["id"])).fetchone()
        action = row["action"] if row else None
    except Exception:
        action = None
    ctx = agent_tools.ToolContext(CONN, user, session_id)
    out = agent_tools.execute(ctx, "submit_prepared", {"confirm_token": token})
    if out.get("error"):
        raise HTTPException(status_code=400, detail={
            "code": out.get("error"),
            "message": out.get("message") or "Draft tidak bisa dijalankan.",
        })
    # Catat hasilnya di percakapan supaya riwayat tidak "kehilangan" aksi yang
    # dijalankan lewat tombol (bukan lewat pesan user). Saran ikut disimpan
    # supaya blok "Pertanyaan lanjutan" tetap ada saat riwayat dibuka ulang.
    if session_id:
        trace_after = [{"tool": "submit_prepared"}]
        prep_tool = _PREP_TOOL_FOR_ACTION.get(action or "")
        if prep_tool:
            trace_after.insert(0, {"tool": prep_tool})
        saran = agent._suggestions(
            {"agent_trace": trace_after}, "", conn=CONN, user=user)
        CONN.execute(
            "INSERT INTO chat_messages (session_id, role, text, citations, created_at,"
            " suggestions) VALUES (?,?,?,?,?,?)",
            (session_id, "assistant",
             out.get("message") or "Draft dijalankan.", "[]", _iso_now(),
             json.dumps(saran, ensure_ascii=False)))
        _touch_session(session_id)
        CONN.commit()
        out["suggestions"] = saran
    return out


# ---- Riwayat chat ----
# PENTING: route statis `/api/chat/sessions` harus didaftarkan SEBELUM
# `/api/chat/{session_id}`, kalau tidak "sessions" dianggap id sesi (404).
@app.get("/api/chat/sessions")
@db_locked
def chat_sessions(limit: str | None = None, offset: str | None = None,
                  user: dict = Depends(parse_token)):
    """Daftar sesi milik user, terbaru dulu (judul, waktu, jumlah pesan).

    Sebelumnya `LIMIT 100` ditulis mati: user dengan >100 percakapan kehilangan
    sisanya tanpa cara mengambilnya, dan endpoint ini juga tidak punya
    rate-limit sama sekali (satu-satunya endpoint baca chat yang bolong).
    Paging sekarang seragam dengan daftar lain (`paging.slice_page`).

    Sesi kosong difilter di SQL, bukan setelah LIMIT — kalau tidak, satu
    halaman bisa berisi 10 sesi kosong dan tampil kosong semua.
    """
    _rate_limit("chat:sessions:" + user["id"], READ_RATE_LIMIT)
    lim, off = paging.window(limit, offset, default=50, cap=100)
    base = """FROM chat_sessions s
              WHERE s.user_id = ?
                AND EXISTS (SELECT 1 FROM chat_messages m WHERE m.session_id = s.id)"""
    total = CONN.execute("SELECT COUNT(*) AS n " + base, (user["id"],)).fetchone()["n"]
    rows = CONN.execute(
        """SELECT s.id, s.title, s.created_at, s.updated_at, s.status,
                  (SELECT COUNT(*) FROM chat_messages m WHERE m.session_id = s.id) AS n
           """ + base + """
           ORDER BY COALESCE(s.updated_at, s.created_at) DESC, s.id DESC
           LIMIT ? OFFSET ?""",
        (user["id"], lim, off),
    ).fetchall()
    items = [{
        "id": r["id"],
        "title": r["title"] or "Percakapan baru",
        "created_at": r["created_at"],
        "updated_at": r["updated_at"] or r["created_at"],
        "status": r["status"],
        "messages": r["n"],
    } for r in rows]
    return {"sessions": items, "count": len(items), "total": total,
            "limit": lim, "offset": off,
            "has_more": off + len(items) < total}


@app.patch("/api/chat/sessions/{session_id}")
@db_locked
def chat_session_rename(session_id: str, payload: dict = Body(...),
                        user: dict = Depends(parse_token)):
    title = (payload.get("title") or "").strip()
    if not title:
        raise HTTPException(status_code=422, detail={
            "code": "title_required", "message": "Judul tidak boleh kosong.",
            "fields": {"title": "wajib"}})
    if len(title) > 120:
        raise HTTPException(status_code=422, detail={
            "code": "title_too_long", "message": "Judul maksimal 120 karakter.",
            "fields": {"title": "maks 120"}})
    _, forbidden = _session_owned(user, session_id)
    row = CONN.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
    if not row or forbidden:
        raise HTTPException(status_code=404, detail="session_not_found")
    CONN.execute("UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                 (title, _iso_now(), session_id))
    CONN.commit()
    return {"id": session_id, "title": title}


@app.delete("/api/chat/sessions/{session_id}")
@db_locked
def chat_session_delete(session_id: str, user: dict = Depends(parse_token)):
    _, forbidden = _session_owned(user, session_id)
    row = CONN.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
    if not row or forbidden:
        raise HTTPException(status_code=404, detail="session_not_found")
    CONN.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
    CONN.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    CONN.commit()
    return {"id": session_id, "deleted": True}


@app.get("/api/chat/{session_id}")
@db_locked
def chat_history(session_id: str, user: dict = Depends(parse_token)):
    sess = CONN.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
    if not sess or (sess["user_id"] != user["id"] and user["role"] != "admin"):
        # admins may view any session
        raise HTTPException(status_code=404, detail="session_not_found")
    # Membaca percakapan orang lain adalah tindakan sensitif: catat agar ada
    # jejak siapa membuka sesi siapa (sebelumnya tidak tercatat sama sekali).
    if sess["user_id"] != user["id"]:
        portal.audit_event(CONN, user, "chat_session_read",
                           f"Admin membuka sesi chat {session_id} milik {sess['user_id']}",
                           extra={"session_id": session_id,
                                  "owner": sess["user_id"]})
        CONN.commit()
    rows = CONN.execute(
        "SELECT id, role, text, citations, created_at, interrupted, trace, prepared,"
        " prepared_done, suggestions"
        " FROM chat_messages WHERE session_id = ? ORDER BY id",
        (session_id,)).fetchall()
    def _trace(r):
        raw = r["trace"] if "trace" in r.keys() else None
        if not raw:
            return []
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return []

    def _prepared(r):
        raw = r["prepared"] if "prepared" in r.keys() else None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None
    def _json_list(r, key):
        raw = r[key] if key in r.keys() else None
        if not raw:
            return []
        try:
            val = json.loads(raw)
        except (ValueError, TypeError):
            return []
        return val if isinstance(val, list) else []

    messages = []
    for r in rows:
        messages.append({
            "role": r["role"],
            "text": r["text"],
            # `citations` boleh NULL untuk pesan lama / pesan yang belum
            # pernah punya sitasi (mis. pesan sistem). json.loads(None) dulu
            # melempar TypeError sehingga seluruh riwayat gagal dimuat.
            "citations": _json_list(r, "citations"),
            "interrupted": bool(r["interrupted"]) if "interrupted" in r.keys() else False,
            "trace": _trace(r),
            "prepared": _prepared(r),
            # Draft yang sudah dijalankan tetap dikirim (sebagai jejak), tetapi
            # UI tidak menampilkan tombolnya — lihat renderPrepared(prepared, done).
            "prepared_done": bool(r["prepared_done"]) if "prepared_done" in r.keys() else False,
            "suggestions": _json_list(r, "suggestions"),
            "created_at": r["created_at"],
        })

    # Jaring terakhir: jawaban terakhir tanpa saran (baris lama sebelum kolom
    # `suggestions` ada, atau sesi yang dibuat sebelum deploy) tetap diberi
    # saran. Tanpa ini, membuka sesi lama dari Riwayat berakhir tanpa satu pun
    # "Pertanyaan lanjutan" — persis keluhan yang dilaporkan.
    last_asst = None
    for m in reversed(messages):
        if m["role"] == "assistant":
            last_asst = m
            break
    if last_asst is not None and not last_asst["suggestions"]:
        try:
            last_asst["suggestions"] = agent._suggestions(
                {"agent_trace": last_asst["trace"] or [],
                 "prepared": last_asst["prepared"],
                 "prepared_done": last_asst["prepared_done"]},
                "", conn=CONN, user=user)
        except Exception:
            pass

    return {"session_id": session_id,
            "title": (sess["title"] if "title" in sess.keys() else None),
            "messages": messages}


# ---- Documents (admin) ----
# Aturan "dokumen mana yang boleh DIKELOLA" hidup di `hr_actions` supaya jalur
# HTTP dan jalur chat memakai satu implementasi. Dulu fungsi ini ada di sini,
# dan `filter_allowed_docs` membuang dokumen `deleted_at` — akibatnya
# POST /api/documents/{id}/restore selalu 404 untuk HR: dokumen yang baru
# dihapus langsung hilang dari daftar yang boleh dikelola.
def _manageable_docs(user: dict) -> list:
    """Dokumen yang boleh DIKELOLA user (termasuk yang ada di tempat sampah).

    `require_hr` hanya memeriksa role, bukan ACL dokumen. Tanpa pembatasan ini
    HR bisa membaca isi dokumen restricted milik team_lead/admin (mis. Proyek
    Kenari) lewat GET /api/documents — padahal GET /api/documents/{id}
    menolaknya. Admin tetap melihat semua.
    """
    return hr_actions.manageable_docs(CONN, user)


def _find_doc_any(doc_id: str):
    """Ambil dokumen apa pun statusnya (termasuk terhapus), untuk manajemen."""
    return hr_actions.find_doc_any(CONN, doc_id)


def _require_manageable(user: dict, doc_id: str) -> None:
    """404 (bukan 403) agar keberadaan dokumen restricted tidak bocor."""
    bad = hr_actions.require_manageable(CONN, user, doc_id)
    if bad:
        raise HTTPException(status_code=bad[1], detail={
            "code": bad[0].get("error"), "message": bad[0].get("message")})


@app.get("/api/documents")
@db_locked
def list_documents(user: dict = Depends(require_hr),
                   limit: str | None = None, offset: str | None = None,
                   q: str | None = None, status: str | None = None,
                   deleted: str | None = None):
    """Daftar dokumen yang boleh dikelola user, dengan paging + filter.

    `limit`/`offset` dulu diterima di URL tapi diabaikan (selalu 70 dokumen
    sekaligus). Sekarang dipotong di server; `total` menjelaskan ukuran penuh.

    `?deleted=only|all` menampilkan isi tempat sampah (soft delete, Sprint 4.5).
    Default `active` supaya dokumen terhapus tidak pernah muncul tanpa diminta.
    """
    _rate_limit("docs:list:" + user["id"], READ_RATE_LIMIT)
    docs = _manageable_docs(user)
    mode = (deleted or "active").strip().lower()
    if mode == "only":
        docs = [d for d in docs if d.get("deleted_at")]
    elif mode == "all":
        pass
    else:
        docs = [d for d in docs if not d.get("deleted_at")]
    if status:
        docs = [d for d in docs if (d.get("status") or "indexed") == status]
    if q:
        needle = q.strip().lower()
        if needle:
            docs = [d for d in docs
                    if needle in (d.get("title") or "").lower()
                    or needle in (d.get("source_type") or "").lower()
                    or needle in (d.get("id") or "").lower()]
    page = paging.slice_page(docs, limit, offset)
    # Isi dokumen TIDAK ikut di daftar. Sebelumnya `content` mentah dikirim
    # untuk setiap baris, sehingga dua kontrol terlewati sekaligus:
    #   * `mask_pii` — detail (`GET /api/documents/{id}`) menyamarkan telepon/
    #     email, daftar tidak: "0812-3456-7890" vs "0812****890";
    #   * audit `doc_view` — membaca isi lewat daftar tidak meninggalkan jejak.
    # Daftar hanya butuh metadata; isi penuh diambil lewat endpoint detail yang
    # sudah memasking + mencatat. `excerpt` di sini sudah tersamar.
    for d in page["items"]:
        body = d.pop("content", None) or ""
        excerpt, _ = mask_pii(body[:240])
        d["excerpt"] = excerpt
        d["content_chars"] = len(body)
    return paging.spread({"query": q or "", "status": status or "",
                          "deleted": mode}, "documents", page)


# Nilai yang diterima untuk kolom dokumen. Divalidasi di POST maupun PUT supaya
# data sampah tidak bisa masuk lewat jalur mana pun (dulu POST menerima string
# kosong sehingga dokumen tanpa judul/tipe/konten ikut tersimpan dan tampil di
# tabel admin sebagai baris kosong).
DOC_CONFIDENTIALITY = ("internal", "restricted")
# Sumber tunggal: docs_lifecycle (menambah `review` untuk alur draft → review →
# published). Sebelumnya daftar ini ditulis ulang di app.py dan UI, sehingga
# status baru bisa lolos validasi di satu tempat dan ditolak di tempat lain.
DOC_STATUSES = lifecycle.DOC_STATUSES


def _as_str_list(value, field: str) -> list:
    """Normalisasi `workspace_ids`/`access_roles` menjadi list[str].

    Kolom ini disimpan sebagai JSON array dan dibaca kembali lewat
    `json.loads` di `row_to_doc`. Dulu POST menerima string apa adanya
    ("ws_onboarding"), lalu menyimpannya sebagai `"\"ws_onboarding\""` —
    `json.loads` mengembalikan STRING, bukan list, sehingga render tabel admin
    (`d.workspace_ids.join(", ")`) melempar TypeError dan halaman berhenti di
    DOM lama. Menerima string koma-separated juga memudahkan klien curl.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    raise HTTPException(status_code=422, detail={
        "code": "invalid_field_type",
        "message": f"{field} harus berupa daftar atau string dipisah koma.",
        "fields": {field: "list|string"}})


def _validate_doc_payload(title, source_type, content, confidentiality, status):
    """Validasi field wajib + enum dokumen. Raise 422 bila tidak valid."""
    if not (title or "").strip():
        raise HTTPException(status_code=422, detail="title_required")
    # Judul dokumen dirender di daftar, panel sumber, dan notifikasi. Satu baris
    # baru cukup untuk memecah kartu DAN menambahkan baris instruksi di atas
    # isi dokumen pada konteks model. Tidak ada judul sah yang butuh newline.
    if any(ch in title for ch in ("\r", "\n", "\t")):
        raise HTTPException(status_code=422, detail={
            "code": "title_invalid_characters",
            "message": "Judul dokumen tidak boleh memuat baris baru atau tab.",
            "fields": {"title": "tanpa baris baru"}})
    if not (source_type or "").strip():
        raise HTTPException(status_code=422, detail="source_type_required")
    if not (content or "").strip():
        raise HTTPException(status_code=422, detail="content_required")
    if confidentiality not in DOC_CONFIDENTIALITY:
        raise HTTPException(status_code=422, detail="invalid_confidentiality")
    if status not in DOC_STATUSES:
        raise HTTPException(status_code=422, detail="invalid_status")


@app.post("/api/documents")
@db_locked
def create_document(payload: dict = Body(...), admin: dict = Depends(require_hr)):
    doc_id = payload.get("id") or ("doc_" + uuid.uuid4().hex[:8])
    if CONN.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone():
        raise HTTPException(status_code=409, detail="doc_id_exists")
    status = payload.get("status", "published")
    confidentiality = payload.get("confidentiality", "internal")
    _validate_doc_payload(payload.get("title"), payload.get("source_type"),
                          payload.get("content"), confidentiality, status)
    CONN.execute(
        """INSERT INTO documents
           (id,title,source_type,effective_date,workspace_ids,access_roles,content,owner,confidentiality,status,version,supersedes,content_hash,local_edit)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
        (doc_id, payload["title"].strip(), payload["source_type"].strip(), payload.get("effective_date"),
         json.dumps(_as_str_list(payload.get("workspace_ids"), "workspace_ids")
                    or ["ws_onboarding"]),
         json.dumps(_as_str_list(payload.get("access_roles",
                                             ["employee", "hr", "team_lead", "admin"]),
                                 "access_roles")),
         payload["content"], payload.get("owner"), confidentiality,
         status, payload.get("version", "1"), payload.get("supersedes"),
         hashlib.sha256((payload.get("content") or "").encode()).hexdigest()))
    lifecycle.snapshot(CONN, {
        "id": doc_id, "title": payload["title"].strip(),
        "content": payload["content"], "status": status,
        "version": str(payload.get("version", "1")),
        "content_hash": hashlib.sha256((payload.get("content") or "").encode()).hexdigest(),
    }, admin, note="versi awal")
    portal.audit_event(CONN, admin, "doc_create",
                       f"Membuat dokumen: {payload['title'].strip()}",
                       document_id=doc_id,
                       extra={"doc_id": doc_id, "status": status,
                              "confidentiality": confidentiality})
    CONN.commit()
    return {"id": doc_id, "status": "created", "doc_status": status}


@app.post("/api/documents/upload")
@db_locked
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    source_type: str | None = Form(None),
    effective_date: str | None = Form(None),
    workspace_ids: str | None = Form(None),
    access_roles: str | None = Form(None),
    owner: str | None = Form(None),
    confidentiality: str | None = Form("internal"),
    admin: dict = Depends(require_hr),
):
    """Upload PDF/DOCX/CSV/MD/TXT, extract text, store as an indexed document."""
    _rate_limit("doc:upload:" + admin["id"], REQUEST_RATE_LIMIT)
    # Batas ukuran diperiksa SEBELUM file dibaca penuh ke memori: tanpa ini,
    # satu file besar bisa menghabiskan RAM proses sebelum MAX_CHARS sempat
    # memotong teksnya (MAX_CHARS memotong hasil ekstraksi, bukan byte masuk).
    declared = (file.headers.get("content-length") if file.headers else None)
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file_too_large")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file_too_large")
    try:
        parsed = ingest.parse_file(file.filename or "upload", raw)
    except ingest.IngestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    doc_id = "doc_" + uuid.uuid4().hex[:8]
    name = os.path.splitext(os.path.basename(file.filename or "dokumen"))[0]
    doc_title = title or name
    if any(ch in doc_title for ch in ("\r", "\n", "\t")):
        raise HTTPException(status_code=422, detail={
            "code": "title_invalid_characters",
            "message": "Judul dokumen tidak boleh memuat baris baru atau tab.",
            "fields": {"title": "tanpa baris baru"}})
    ws = [x.strip() for x in (workspace_ids or "ws_onboarding").split(",") if x.strip()]
    roles = [x.strip() for x in (access_roles or "employee,hr,team_lead,admin").split(",") if x.strip()]
    conf = confidentiality or "internal"
    if conf not in DOC_CONFIDENTIALITY:
        raise HTTPException(status_code=422, detail="invalid_confidentiality")
    CONN.execute(
        """INSERT INTO documents
           (id,title,source_type,effective_date,workspace_ids,access_roles,content,owner,confidentiality,status,content_hash,local_edit)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,1)""",
        (doc_id, doc_title, source_type or parsed["source_type"], effective_date,
         json.dumps(ws), json.dumps(roles), parsed["text"], owner,
         conf, "indexed",
         hashlib.sha256(parsed["text"].encode()).hexdigest()))
    portal.audit_event(CONN, admin, "doc_upload",
                       f"Upload dokumen: {doc_title}",
                       document_id=doc_id,
                       extra={"doc_id": doc_id, "filename": file.filename,
                              "chars": len(parsed["text"]),
                              "truncated": parsed["truncated"]})
    CONN.commit()
    return {"id": doc_id, "title": doc_title,
            "source_type": source_type or parsed["source_type"],
            "chars": len(parsed["text"]), "truncated": parsed["truncated"],
            "status": "indexed"}


@app.put("/api/documents/{doc_id}")
@db_locked
def update_document(doc_id: str, payload: dict = Body(...), user: dict = Depends(require_hr)):
    """Update parsial (merge), bukan timpa penuh.

    Bug sebelumnya: payload tanpa `owner`/`confidentiality` menulis NULL dan
    "internal", sehingga mengedit dokumen restricted lewat UI diam-diam
    menurunkan klasifikasinya. Sekarang field yang tidak dikirim dipertahankan.
    """
    _require_manageable(user, doc_id)
    row = CONN.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    cur = row_to_doc(row)
    if cur.get("deleted_at"):
        raise HTTPException(status_code=409, detail={
            "code": "doc_deleted",
            "message": "Dokumen ini ada di tempat sampah. Pulihkan dulu sebelum diubah."})

    def pick(key, default=None):
        return payload[key] if key in payload else default

    title = pick("title", cur["title"])
    source_type = pick("source_type", cur["source_type"])
    effective_date = pick("effective_date", cur.get("effective_date"))
    workspace_ids = _as_str_list(pick("workspace_ids", cur.get("workspace_ids") or []),
                                  "workspace_ids")
    access_roles = _as_str_list(pick("access_roles", cur.get("access_roles") or []),
                                "access_roles")
    content = pick("content", cur.get("content") or "")
    owner = pick("owner", cur.get("owner"))
    confidentiality = pick("confidentiality", cur.get("confidentiality") or "internal")
    status = pick("status", cur.get("status") or "indexed")
    supersedes = pick("supersedes", cur.get("supersedes"))

    _validate_doc_payload(title, source_type, content, confidentiality, status)
    # Transisi status divalidasi di satu tempat (docs_lifecycle), bukan diterima
    # apa adanya seperti sebelumnya.
    try:
        lifecycle.check_transition(cur.get("status") or "indexed", status)
    except lifecycle.LifecycleError as e:
        raise HTTPException(status_code=e.status, detail={
            "code": e.code, "message": e.message, "fields": e.fields})

    # Versi: kalau isi berubah dan klien tidak menyebut versi baru, naikkan
    # otomatis. Tanpa ini dua revisi berbeda berbagi label "1" dan riwayat
    # versi jadi tidak bisa dipakai untuk diff.
    content_changed = (cur.get("content") or "") != content
    new_hash = hashlib.sha256((content or "").encode()).hexdigest()
    known = lifecycle.list_versions(CONN, doc_id)
    cur_label = str(cur.get("version") or "1")
    if "version" in payload and payload.get("version"):
        version = str(payload["version"])
    elif content_changed:
        # Label baru harus benar-benar belum terpakai, dan harus lebih tinggi
        # dari label dokumen maupun label riwayat mana pun. Kalau hanya
        # memakai `cur.version + 1`, dokumen yang pernah di-rollback bisa
        # menabrak baris versi lama.
        base = cur_label
        hist = lifecycle.latest_version(CONN, doc_id)
        if hist and hist != base:
            base = max([base, hist], key=lambda v: [
                int(x) if x.isdigit() else 0 for x in str(v).split(".")] + [0])
        version = lifecycle.next_version(base)
    else:
        version = cur_label

    if content_changed:
        if not known:
            # Dokumen lama (sebelum ada tabel versi): simpan isi yang sekarang
            # masih berlaku supaya revisi pertama tidak menghapus sejarahnya.
            lifecycle.snapshot(CONN, dict(cur, version=cur_label), user,
                               note="isi sebelum revisi")

    CONN.execute(
        """UPDATE documents SET title=?, source_type=?, effective_date=?, workspace_ids=?, access_roles=?,
           content=?, owner=?, confidentiality=?, status=?, version=?, supersedes=?, content_hash=?,
           local_edit=1 WHERE id=?""",
        (title.strip(), source_type.strip(), effective_date,
         json.dumps(workspace_ids), json.dumps(access_roles),
         content, owner, confidentiality, status, version, supersedes,
         new_hash, doc_id))
    changed = [k for k, before, after in (
        ("title", cur["title"], title.strip()),
        ("source_type", cur["source_type"], source_type.strip()),
        ("effective_date", cur.get("effective_date"), effective_date),
        ("workspace_ids", cur.get("workspace_ids") or [], workspace_ids),
        ("access_roles", cur.get("access_roles") or [], access_roles),
        ("content", cur.get("content") or "", content),
        ("owner", cur.get("owner"), owner),
        ("confidentiality", cur.get("confidentiality") or "internal", confidentiality),
        ("status", cur.get("status") or "indexed", status),
        ("version", cur.get("version") or "1", version),
        ("supersedes", cur.get("supersedes"), supersedes),
    ) if before != after]
    # Versi baru dicatat sebagai baris riwayat tersendiri supaya
    # GET /api/documents/{id}/versions langsung menampilkan isi terbaru juga.
    if content_changed:
        lifecycle.snapshot(CONN, {
            "id": doc_id, "title": title.strip(), "content": content,
            "status": status, "version": version, "content_hash": new_hash,
        }, user, note=payload.get("change_note") or "revisi")
    portal.audit_event(CONN, user, "doc_update",
                       f"Mengubah dokumen: {title.strip()}",
                       document_id=doc_id,
                       extra={"doc_id": doc_id, "changed_fields": changed,
                              "version": version,
                              "previous_version": cur.get("version") or "1"})
    CONN.commit()
    return {"id": doc_id, "status": "updated", "changed_fields": changed,
            "doc_status": status, "version": version,
            "previous_version": cur.get("version") or "1"}


@app.get("/api/documents/{doc_id}")
@db_locked
def preview_document(doc_id: str, raw: bool = False,
                     user: dict = Depends(parse_token)):
    """Pratinjau satu dokumen. `?raw=1` hanya untuk pengelola (form edit).

    Isi yang dikembalikan default SELALU tersamar (`mask_pii`) dan dicatat
    sebagai `doc_view`. Form edit butuh isi asli — kalau memakai versi tersamar,
    menyimpan akan MENIMPA nomor telepon/email asli dengan "0812****890".
    Karena itu jalur mentah dipisah: hanya hr/admin, hanya dokumen yang boleh
    dikelola, dan tercatat sebagai `doc_edit_open` (bukan `doc_view`).
    """
    if raw:
        if not rbac.allowed("docs.manage", user["role"]):
            raise HTTPException(status_code=403, detail="hr_or_admin_required")
        _require_manageable(user, doc_id)
        row = CONN.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="doc_not_found")
        d = row_to_doc(row)
        portal.audit_event(CONN, user, "doc_edit_open",
                           "Membuka isi asli untuk diedit: " + d["title"],
                           document_id=d["id"])
        CONN.commit()
        return {"id": d["id"], "title": d["title"],
                "effective_date": d.get("effective_date"),
                "source_type": d.get("source_type"),
                "version": d.get("version", "1"),
                "confidentiality": d.get("confidentiality", "internal"),
                "content_hash": d.get("content_hash"),
                "content": d.get("content") or ""}
    allowed = filter_allowed_docs(CONN, user, include_inactive=True)
    for d in allowed:
        if d["id"] == doc_id:
            content, masked = mask_pii(d["content"])
            # Audit per-view: siapa melihat dokumen apa, kapan (append-only).
            CONN.execute(
                """INSERT INTO audit_log
                   (timestamp,user_id,user_role,question,permission_decision,allowed_sources,
                    selected_sources,answer,pii_masked,agent_version,event_type,document_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
                    user["id"], user["role"],
                    "Melihat dokumen: " + d["title"], "view",
                    json.dumps([d["id"]]), json.dumps([d["id"]]),
                    None, int(masked), AGENT_VERSION, "doc_view", d["id"],
                ),
            )
            CONN.commit()
            return {"id": d["id"], "title": d["title"],
                    "effective_date": d.get("effective_date"),
                    "source_type": d.get("source_type"),
                    "version": d.get("version", "1"),
                    "confidentiality": d.get("confidentiality", "internal"),
                    "content_hash": d.get("content_hash"),
                    "content": content}
    raise HTTPException(status_code=404, detail="document_not_available")


# ---- Siklus hidup dokumen (Sprint 4.3–4.5) ----
@app.get("/api/documents/{doc_id}/versions")
@db_locked
def document_versions(doc_id: str, include_content: bool = False,
                      user: dict = Depends(require_hr)):
    """Riwayat versi dokumen, terbaru dulu.

    Versi terbaru yang berlaku selalu ikut ditampilkan walau baris riwayatnya
    belum ada (dokumen lama sebelum tabel versi dibuat), jadi daftar ini tidak
    pernah kosong untuk dokumen yang bisa dikelola.
    """
    _rate_limit("docs:versions:" + user["id"], READ_RATE_LIMIT)
    data, status = hr_actions.document_versions(CONN, user, doc_id)
    if status >= 400:
        raise HTTPException(status_code=status, detail={
            "code": data.get("error"), "message": data.get("message")})
    if include_content:
        # Isi tiap versi hanya diminta eksplisit oleh UI riwayat (berat);
        # jalur chat tidak pernah membutuhkannya.
        data["versions"] = lifecycle.list_versions(
            CONN, doc_id, include_content=True) or data["versions"]
    return data


@app.get("/api/documents/{doc_id}/diff")
@db_locked
def document_diff(doc_id: str, a: str, b: str, user: dict = Depends(require_hr)):
    """Diff teks antara dua versi (a = lama, b = baru)."""
    _rate_limit("docs:diff:" + user["id"], READ_RATE_LIMIT)
    _require_manageable(user, doc_id)
    try:
        data = lifecycle.diff_versions(CONN, doc_id, a, b)
    except lifecycle.LifecycleError as e:
        raise HTTPException(status_code=e.status, detail={
            "code": e.code, "message": e.message})
    portal.audit_event(CONN, user, "doc_diff",
                       f"Membandingkan versi {a} → {b}",
                       document_id=doc_id,
                       extra={"doc_id": doc_id, "from": a, "to": b,
                              "added": data["added_lines"],
                              "removed": data["removed_lines"]})
    CONN.commit()
    return data


@app.delete("/api/documents/{doc_id}")
@db_locked
def delete_document(doc_id: str, user: dict = Depends(require_hr)):
    """Soft delete: baris tetap ada, semua jalur baca menyaring `deleted_at`."""
    _rate_limit("docs:delete:" + user["id"], REQUEST_RATE_LIMIT)
    _require_manageable(user, doc_id)
    doc = _find_doc_any(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="doc_not_found")
    if doc.get("deleted_at"):
        return {"id": doc_id, "status": "already_deleted",
                "deleted_at": doc["deleted_at"], "deleted_by": doc.get("deleted_by")}
    data = lifecycle.soft_delete(CONN, doc_id, user)
    portal.audit_event(CONN, user, "doc_delete",
                       f"Menghapus dokumen: {doc['title']}",
                       document_id=doc_id,
                       extra={"doc_id": doc_id, "title": doc["title"],
                              "soft": True})
    CONN.commit()
    return data


@app.post("/api/documents/{doc_id}/restore")
@db_locked
def restore_document(doc_id: str, user: dict = Depends(require_hr)):
    """Pulihkan dokumen dari tempat sampah."""
    _rate_limit("docs:restore:" + user["id"], REQUEST_RATE_LIMIT)
    _require_manageable(user, doc_id)
    doc = _find_doc_any(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="doc_not_found")
    if not doc.get("deleted_at"):
        return {"id": doc_id, "status": "not_deleted"}
    data = lifecycle.restore(CONN, doc_id)
    portal.audit_event(CONN, user, "doc_restore",
                       f"Memulihkan dokumen: {doc['title']}",
                       document_id=doc_id, extra={"doc_id": doc_id})
    CONN.commit()
    return data


SYNONYMS = {
    "cuti": ["libur", "leave", "izin"],
    "reimbursement": ["reimburse", "klaim", "ganti rugi"],
    "gaji": ["payroll", "upah", "salary"],
    "lembur": ["overtime", "lemburan"],
    "wfh": ["remote", "kerja dari rumah"],
    "proyek": ["project", "proyek"],
}


def _doc_score(doc: dict, q: str) -> int:
    """Skor relevansi sederhana: judul > kategori/type > excerpt."""
    if not q:
        return 0
    q = q.lower()
    terms = {q}
    for key, syns in SYNONYMS.items():
        if key in q:
            terms.update(syns)
    score = 0
    title = (doc.get("title") or "").lower()
    stype = (doc.get("source_type") or "").lower()
    excerpt = (doc.get("excerpt") or "").lower()
    for t in terms:
        if t in title:
            score += 10
            if title.startswith(t):
                score += 5
        if t in stype:
            score += 4
        if t in excerpt:
            score += 1
    return score


@app.get("/api/referensi")
@db_locked
def api_referensi(user: dict = Depends(parse_token), q: str | None = None,
                  limit: str | None = None, offset: str | None = None,
                  category: str | None = None, types: str | None = None):
    """Perpustakaan dokumen untuk user aktif (hormati RBAC via filter_allowed_docs).
    Hanya metadata + ringkasan singkat (PII teredaksi); konten penuh lewat
    /api/documents/{id}. Query opsional `q` MENYARING dokumen yang tidak punya
    satu pun term cocok, lalu mengurutkan sisanya berdasarkan skor."""
    # Query berisi spasi saja tidak boleh dianggap pencarian: tanpa ini
    # `?q=%20` akan membuang SEMUA dokumen (skor 0 untuk semuanya).
    q = (q or "").strip() or None
    docs = filter_allowed_docs(CONN, user)
    # Peta dokumen yang telah digantikan dokumen lain (superseded_by).
    superseded_by = {}
    for row in CONN.execute("SELECT id, supersedes FROM documents WHERE supersedes IS NOT NULL").fetchall():
        superseded_by.setdefault(row["supersedes"], []).append(row["id"])

    out = []
    for d in docs:
        if d.get("status", "indexed") not in ("indexed", "published"):
            continue  # siklus hidup: draft/deactivated tidak ditampilkan
        excerpt, masked = mask_pii((d.get("content") or "")[:240])
        out.append({
            "id": d["id"],
            "title": d["title"],
            "source_type": d.get("source_type"),
            "effective_date": d.get("effective_date"),
            "confidentiality": d.get("confidentiality", "internal"),
            "status": d.get("status", "indexed"),
            "version": d.get("version", "1"),
            "supersedes": d.get("supersedes"),
            "superseded_by": superseded_by.get(d["id"], []),
            "workspace_ids": d.get("workspace_ids", []),
            "access_roles": d.get("access_roles", []),
            "excerpt": excerpt,
            "pii_masked": int(masked),
            "content_hash": d.get("content_hash"),
        })
    if q:
        for d in out:
            d["score"] = _doc_score(d, q)
        # `q` dulu hanya MENYORTIR: pencarian "Makassar" mengembalikan
        # `total: 60` (seluruh koleksi) dengan dokumen tak relevan di belakang,
        # sehingga UI tampak benar hanya karena klien menyaring ulang satu
        # halaman. Sekarang dokumen tanpa satu pun term yang cocok dibuang.
        out = [d for d in out if d.get("score", 0) > 0]
        out.sort(key=lambda d: d["effective_date"] or "", reverse=True)  # tie-break
        out.sort(key=lambda d: d.get("score", 0), reverse=True)          # stable
    else:
        out.sort(key=lambda d: d["effective_date"] or "", reverse=True)
    # Facet dihitung dari himpunan SETELAH pencarian teks tetapi SEBELUM filter
    # kategori — persis yang dibutuhkan UI untuk menampilkan "sebaran koleksi"
    # yang tidak ikut mengecil saat satu kategori dipilih.
    facets = {}
    for d in out:
        key = d.get("source_type") or "lain"
        facets[key] = facets.get(key, 0) + 1

    wanted = {t.strip().lower() for t in (types or "").split(",") if t.strip()}
    if category:
        wanted.add(category.strip().lower())
    if wanted:
        out = [d for d in out if (d.get("source_type") or "").lower() in wanted]
    page = paging.slice_page(out, limit, offset)
    return paging.spread({"query": q or "", "category": category or "",
                          "types": sorted(wanted), "facets": facets},
                         "documents", page)


@app.post("/api/chat/{audit_id}/feedback")
@db_locked
def chat_feedback(audit_id: int, payload: dict = Body(...),
                  user: dict = Depends(parse_token)):
    value = payload.get("value")
    if value not in (1, -1):
        raise HTTPException(status_code=400, detail="invalid_value")
    row = CONN.execute("SELECT id FROM audit_log WHERE id = ?", (audit_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="audit_not_found")
    CONN.execute(
        "INSERT INTO feedback (audit_id, user_id, value, created_at) VALUES (?,?,?,?)",
        (audit_id, user["id"], value,
         datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")),
    )
    CONN.commit()
    return {"ok": True}


# ---- Users & permissions (admin) ----
USER_COLS = ("id,name,role,allowed_workspaces,department,job_title,manager_id,"
             "join_date,employment_status,location,employee_no,email")


def _user_public(row) -> dict:
    keys = row.keys()
    out = {"id": row["id"], "name": row["name"], "role": row["role"],
           "allowed_workspaces": json.loads(row["allowed_workspaces"] or "[]")}
    for k in users_admin.PROFILE_FIELDS:
        if k in keys:
            out[k] = row[k]
    out["employment_status"] = out.get("employment_status") or "active"
    return out


def _users_filtered(q: str | None = None, role: str | None = None,
                    status: str | None = None) -> list:
    """Daftar karyawan setelah filter — satu sumber untuk tabel dan ekspor.

    Dulu filter ditulis langsung di handler daftar; ekspor CSV butuh aturan
    yang sama persis, dan dua salinan filter akan menyimpang cepat atau lambat.
    """
    rows = CONN.execute("SELECT " + USER_COLS + " FROM users ORDER BY name").fetchall()
    users = [_user_public(r) for r in rows]
    if q:
        needle = q.strip().lower()
        users = [u for u in users
                 if needle in (u["name"] or "").lower()
                 or needle in (u["id"] or "").lower()
                 or needle in (u.get("email") or "").lower()
                 or needle in (u.get("department") or "").lower()]
    if role:
        users = [u for u in users if u["role"] == role]
    if status:
        users = [u for u in users if (u.get("employment_status") or "active") == status]
    return users


@app.get("/api/users")
@db_locked
def list_users(admin: dict = Depends(require_admin),
               q: str | None = None, role: str | None = None,
               status: str | None = None,
               limit: str | None = None, offset: str | None = None):
    """Daftar karyawan + profil, dengan pencarian/paging.

    Profil (departemen, jabatan, atasan) ikut dikirim: tanpa itu tabel Kelola
    tidak bisa menampilkan struktur organisasi dan form edit tidak punya nilai
    awal. Endpoint ini khusus admin (bukan direktori publik).
    """
    _rate_limit("users:list:" + admin["id"], READ_RATE_LIMIT)
    users = _users_filtered(q, role, status)
    page = paging.slice_page(users, limit, offset, default=100)
    return paging.spread({"query": q or "", "role": role or "",
                          "status": status or "",
                          "roles": list(users_admin.ROLES)}, "users", page)


@app.get("/api/users/export.csv")
@db_locked
def export_users_csv(q: str | None = None, role: str | None = None,
                     status: str | None = None,
                     admin: dict = Depends(require_admin)):
    """Ekspor karyawan sesuai filter yang sedang tampil.

    Kolomnya sama dengan yang diterima `POST /api/users/import`, jadi berkas
    hasil ekspor bisa langsung diimpor ulang setelah diedit — itu yang membuat
    ekspor berguna untuk HR, bukan sekadar arsip.
    """
    _rate_limit("users:csv:" + admin["id"], READ_RATE_LIMIT)
    users = _users_filtered(q, role, status)
    header = ["id", "nama", "email", "role", "departemen", "jabatan",
              "atasan", "tanggal_masuk", "lokasi", "nomor_karyawan",
              "status", "workspace"]
    rows = [[u["id"], u["name"], u.get("email") or "", u["role"],
             u.get("department") or "", u.get("job_title") or "",
             u.get("manager_id") or "", u.get("join_date") or "",
             u.get("location") or "", u.get("employee_no") or "",
             u.get("employment_status") or "active",
             ",".join(u.get("allowed_workspaces") or [])]
            for u in users]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return _csv_response(f"karyawan-{stamp}.csv", header, rows)


@app.post("/api/users")
@db_locked
def create_user(payload: dict = Body(...), admin: dict = Depends(require_admin)):
    """Tambah karyawan baru (Sprint 4.1)."""
    _rate_limit("users:create:" + admin["id"], REQUEST_RATE_LIMIT)
    try:
        data = users_admin.create(CONN, admin, payload)
    except users_admin.UserError as e:
        raise HTTPException(status_code=e.status, detail={
            "code": e.code, "message": e.message, "fields": e.fields})
    portal.audit_event(CONN, admin, "user_create",
                       f"Menambah karyawan: {payload.get('name') or data['id']}",
                       extra={"user_id": data["id"], "role": data["role"]})
    CONN.commit()
    return data


@app.patch("/api/users/{user_id}")
@db_locked
def patch_user(user_id: str, payload: dict = Body(...),
               admin: dict = Depends(require_admin)):
    """Ubah sebagian profil karyawan (nama, peran, atasan, status, …)."""
    _rate_limit("users:update:" + admin["id"], REQUEST_RATE_LIMIT)
    try:
        data = users_admin.update(CONN, admin, user_id, payload)
    except users_admin.UserError as e:
        raise HTTPException(status_code=e.status, detail={
            "code": e.code, "message": e.message, "fields": e.fields})
    if data.get("changed_fields"):
        portal.audit_event(CONN, admin, "user_update",
                           f"Mengubah karyawan: {user_id}",
                           extra={"user_id": user_id,
                                  "changed_fields": data["changed_fields"]})
        CONN.commit()
    return data


@app.delete("/api/users/{user_id}")
@db_locked
def delete_user(user_id: str, admin: dict = Depends(require_admin)):
    """Nonaktifkan karyawan (soft). Baris tidak dihapus agar riwayat tetap utuh."""
    _rate_limit("users:delete:" + admin["id"], REQUEST_RATE_LIMIT)
    try:
        data = users_admin.deactivate(CONN, admin, user_id)
    except users_admin.UserError as e:
        raise HTTPException(status_code=e.status, detail={
            "code": e.code, "message": e.message, "fields": e.fields})
    portal.audit_event(CONN, admin, "user_deactivate",
                       f"Menonaktifkan karyawan: {user_id}",
                       extra={"user_id": user_id,
                              "sessions_revoked": data.get("sessions_revoked", 0)})
    CONN.commit()
    return data


@app.post("/api/users/import")
@db_locked
def import_users(payload: dict = Body(...), admin: dict = Depends(require_admin)):
    """Impor CSV karyawan. `dry_run` default True → laporan tanpa menulis.

    Menerima `{"csv": "..."}` (isi file) atau `{"rows": [{...}]}`. UI mengirim
    isi file apa adanya supaya backend yang memutuskan pemisah kolom dan
    normalisasi nama kolom — satu sumber aturan untuk dua jalur.
    """
    _rate_limit("users:import:" + admin["id"], REQUEST_RATE_LIMIT)
    dry = payload.get("dry_run", True)
    try:
        if "rows" in payload and isinstance(payload["rows"], list):
            # Jalur cepat untuk klien terstruktur: bungkus ulang jadi CSV.
            import io as _io
            fields = ["id", "name", "email", "role", "department", "job_title",
                      "manager_id", "join_date", "location", "employee_no",
                      "employment_status", "allowed_workspaces"]
            buf = _io.StringIO()
            w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in payload["rows"]:
                # Baris terstruktur boleh memakai nama kolom Indonesia
                # ("nama", "workspace"); petakan lewat alias yang sama dengan
                # CSV supaya dua jalur impor punya aturan identik.
                row = {}
                for k, v in (r or {}).items():
                    key = users_admin.CSV_ALIASES.get(str(k).strip().lower())
                    if key:
                        row[key] = v
                if isinstance(row.get("allowed_workspaces"), list):
                    row["allowed_workspaces"] = ",".join(row["allowed_workspaces"])
                w.writerow(row)
            text = buf.getvalue()
        else:
            text = payload.get("csv") or ""
        data = users_admin.import_csv(CONN, admin, text, dry_run=bool(dry))
    except users_admin.UserError as e:
        raise HTTPException(status_code=e.status, detail={
            "code": e.code, "message": e.message, "fields": e.fields})
    if not dry and (data["created"] or data["updated"]):
        portal.audit_event(CONN, admin, "user_import",
                           f"Impor karyawan: {data['created']} baru, "
                           f"{data['updated']} diperbarui, {data['failed']} gagal",
                           extra={"created": data["created"],
                                  "updated": data["updated"],
                                  "failed": data["failed"]})
        CONN.commit()
    return data


@app.put("/api/users/{user_id}/permissions")
@db_locked
def update_permissions(user_id: str, payload: dict = Body(...),
                       admin: dict = Depends(require_admin)):
    """Ubah peran/workspace lewat jalur yang SAMA dengan PATCH /api/users/{id}.

    Endpoint ini dulu menulis kolom langsung tanpa validasi, tanpa audit, dan
    tanpa perlindungan admin terakhir. Akibatnya satu request bisa:
      * menyimpan role yang tidak dikenal (`superadmin`), sehingga user itu
        kehilangan seluruh izin tanpa pesan error;
      * menurunkan admin terakhir dan mengunci konsol Kelola/Audit/Cadangan
        untuk semua orang secara permanen;
      * tidak meninggalkan jejak di audit sama sekali.
    Semua aturan sekarang hidup di `users_admin.update()` — satu sumber.
    """
    _rate_limit("users:perm:" + admin["id"], REQUEST_RATE_LIMIT)
    fields = {k: payload[k] for k in ("role", "allowed_workspaces") if k in payload}
    if not fields:
        raise HTTPException(status_code=422, detail={
            "code": "no_fields",
            "message": "Kirim minimal `role` atau `allowed_workspaces`.",
            "fields": {"role": "opsional", "allowed_workspaces": "opsional"}})
    try:
        data = users_admin.update(CONN, admin, user_id, fields)
    except users_admin.UserError as e:
        raise HTTPException(status_code=e.status, detail={
            "code": e.code, "message": e.message, "fields": e.fields})
    if data.get("changed_fields"):
        portal.audit_event(CONN, admin, "user_permissions",
                           f"Mengubah izin karyawan: {user_id}",
                           extra={"user_id": user_id,
                                  "changed_fields": data["changed_fields"]})
        CONN.commit()
    row = CONN.execute(
        "SELECT role, allowed_workspaces FROM users WHERE id = ?", (user_id,)).fetchone()
    # Bentuk respons lama (`role`/`allowed_workspaces`) dipertahankan supaya
    # klien yang sudah ada tidak perlu diubah; `changed_fields` ditambahkan.
    return {"id": user_id, "status": "permissions_updated",
            "role": row["role"] if row else None,
            "allowed_workspaces": json.loads(row["allowed_workspaces"] or "[]") if row else [],
            "changed_fields": data.get("changed_fields", [])}


# ---- Workspaces (admin) ----
@app.get("/api/workspaces")
@db_locked
def list_workspaces(admin: dict = Depends(require_admin)):
    """Daftar workspace + jumlah pemakaian (dokumen/karyawan).

    Angka pemakaian dipakai UI untuk mencegah admin menghapus workspace yang
    masih menopang akses dokumen — server tetap menolaknya (409), ini hanya
    supaya tombolnya bisa dinonaktifkan lebih awal.
    """
    _rate_limit("workspaces:" + admin["id"], READ_RATE_LIMIT)
    items = workspaces.list_all(CONN)
    for w in items:
        w.update(workspaces.usage(CONN, w["id"]))
    return {"workspaces": items, "count": len(items)}


@app.post("/api/workspaces")
@db_locked
def create_workspace(payload: dict = Body(...),
                     admin: dict = Depends(require_admin)):
    _rate_limit("workspace:create:" + admin["id"], REQUEST_RATE_LIMIT)
    try:
        data = workspaces.create(CONN, payload or {})
    except workspaces.WorkspaceError as e:
        raise HTTPException(status_code=e.status, detail={
            "code": e.code, "message": e.message, "fields": e.fields})
    portal.audit_event(CONN, admin, "workspace_create",
                       f"Membuat workspace {data['id']} ({data['name']})")
    CONN.commit()
    return data


@app.patch("/api/workspaces/{ws_id}")
@db_locked
def rename_workspace(ws_id: str, payload: dict = Body(...),
                     admin: dict = Depends(require_admin)):
    _rate_limit("workspace:rename:" + admin["id"], REQUEST_RATE_LIMIT)
    try:
        data = workspaces.rename(CONN, ws_id, payload or {})
    except workspaces.WorkspaceError as e:
        raise HTTPException(status_code=e.status, detail={
            "code": e.code, "message": e.message, "fields": e.fields})
    portal.audit_event(CONN, admin, "workspace_rename",
                       f"Mengganti nama workspace {ws_id} menjadi {data['name']}")
    CONN.commit()
    return data


@app.delete("/api/workspaces/{ws_id}")
@db_locked
def delete_workspace(ws_id: str, admin: dict = Depends(require_admin)):
    """Hapus workspace kosong. Yang masih dipakai ditolak 409.

    Sengaja tidak ada "hapus paksa": menghapus workspace yang menopang dokumen
    membuat dokumen itu tak terjangkau siapa pun — kehilangan akses diam-diam.
    """
    _rate_limit("workspace:delete:" + admin["id"], REQUEST_RATE_LIMIT)
    try:
        data = workspaces.delete(CONN, ws_id)
    except workspaces.WorkspaceError as e:
        raise HTTPException(status_code=e.status, detail={
            "code": e.code, "message": e.message, "fields": e.fields})
    portal.audit_event(CONN, admin, "workspace_delete",
                       f"Menghapus workspace {ws_id}")
    CONN.commit()
    return data


# ---- Audit (admin) ----
# Tabel Audit menampilkan jawaban sebagai potongan pendek — satu baris tabel
# tidak membaca 2 KB teks. Panjang ini adalah yang DITAMPILKAN (bukan lagi
# dipotong di klien), jadi tidak ada dua tempat yang menentukan batas berbeda.
# Jawaban penuh tetap tersedia lewat `?include_answers=true`.
# Batas ukuran unggahan dokumen (byte). 10 MB cukup untuk PDF/DOCX kebijakan
# perusahaan, tetapi mencegah satu request menghabiskan memori proses.
MAX_UPLOAD_BYTES = int(os.environ.get("NOHALU_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

AUDIT_ANSWER_PREVIEW = 160


def _clip_words(text, limit):
    """Potong di batas kata terakhir, lalu tambahkan elipsis.

    Memotong tepat di `limit` karakter menghasilkan baris seperti
    "…Approval Matrix, atu" — terbaca seperti data rusak, bukan ringkasan.
    """
    head = text[:limit]
    cut = head.rfind(" ")
    if cut > limit * 0.6:
        head = head[:cut]
    return head.rstrip(" ,.;:-") + "…"


@app.get("/api/audit")
@db_locked
def list_audit(admin: dict = Depends(require_hr),
               user_id: str | None = None, status: str | None = None,
               from_ts: str | None = None, to_ts: str | None = None,
               include_sources: bool = False, include_answers: bool = False,
               limit: str | None = None, offset: str | None = None):
    """Audit log dengan paging nyata.

    Dulu `LIMIT 500` ditulis mati di query: baris ke-501 dan seterusnya tidak
    pernah bisa dibaca (hilang tanpa jejak) dan `?limit=` diabaikan. Sekarang
    batas atas per halaman tetap ada, tetapi klien bisa menelusuri seluruh log.
    """
    where = " WHERE 1=1"
    params = []
    if user_id:
        where += " AND user_id = ?"; params.append(user_id)
    if from_ts:
        where += " AND timestamp >= ?"; params.append(from_ts)
    if to_ts:
        where += " AND timestamp <= ?"; params.append(to_ts)
    lim, off = paging.window(limit, offset)
    # `status` menyaring verdict/event_type SETELAH baris diambil (verdict
    # dihitung dari beberapa kolom), jadi filter itu tidak bisa dipindah ke SQL.
    # Tanpa status, paging dilakukan di SQL agar tidak menarik seluruh tabel.
    if status:
        q = "SELECT * FROM audit_log" + where + " ORDER BY id DESC LIMIT 5000"
        rows = CONN.execute(q, params).fetchall()
        total = len(rows)
    else:
        total = CONN.execute(
            "SELECT COUNT(*) c FROM audit_log" + where, params).fetchone()["c"]
        q = ("SELECT * FROM audit_log" + where
             + " ORDER BY id DESC LIMIT ? OFFSET ?")
        rows = CONN.execute(q, params + [lim, off]).fetchall()
    out = []
    for r in rows:
        rec = dict(r)
        for k in ("allowed_sources", "retrieved_sources", "selected_sources", "citations"):
            rec[k] = json.loads(rec[k]) if rec[k] else []
        # `allowed_sources` adalah daftar SELURUH dokumen yang boleh dibaca satu
        # peran (50+ id per baris). Halamannya tidak pernah menampilkannya, tapi
        # dulu ikut terkirim: 111 KB dari 236 KB respons untuk 104 baris, dan
        # membengkak linear terhadap jumlah dokumen. Yang dikirim adalah
        # jumlahnya; daftar penuhnya hanya bila diminta eksplisit.
        if not include_sources:
            rec["allowed_count"] = len(rec.pop("allowed_sources", []))
            # `excerpt` adalah potongan isi dokumen (±250 karakter) per sitasi.
            # Halaman Audit hanya memakai id/judul lewat `selected_sources`,
            # jadi potongan itu murni bobot: 11 KB dari 104 baris, dan naik
            # seiring panjang jawaban.
            rec["citations"] = [
                {k: c.get(k) for k in ("id", "title", "source_type", "effective_date")}
                for c in rec.get("citations") or []
            ]
            # `retrieved_sources` = dokumen yang diambil sebelum penyaringan.
            # UI hanya memakai `selected_sources` (yang benar-benar dikutip),
            # jadi daftar ini murni bobot: 4,3 KB dari 114 KB untuk 104 baris.
            rec["retrieved_count"] = len(rec.pop("retrieved_sources", []))
            # `answer` adalah jawaban penuh LLM (bisa 1-2 KB per baris). Tabel
            # Audit hanya menampilkan 160 karakter pertama. Mengirim sisanya
            # membuat 36% respons tidak pernah terbaca; versi penuh tetap ada
            # lewat `?include_answers=true` untuk investigasi.
            # Nama field tetap `answer` supaya klien lama tidak pecah; yang
            # berubah hanya panjangnya, dan pemotongan ditandai eksplisit.
            answer = rec.get("answer") or ""
            if not include_answers and len(answer) > AUDIT_ANSWER_PREVIEW:
                rec["answer"] = _clip_words(answer, AUDIT_ANSWER_PREVIEW)
                rec["answer_length"] = len(answer)
                rec["answer_truncated"] = True
            else:
                rec["answer_length"] = len(answer)
                rec["answer_truncated"] = False
        rec["details"] = json.loads(rec["details"]) if rec.get("details") else None
        event_type = rec.get("event_type") or "qa"
        verdict = "answered"
        if event_type != "qa":
            # Event non-chat (doc_view, request_*, approval_override, doc_update…)
            # diverifikasi lewat jenis eventnya, bukan refusal/konflik.
            verdict = event_type
        elif rec["refusal_reason"]:
            verdict = "refused"
        elif rec["conflict_detected"]:
            verdict = "conflict"
        elif rec["prompt_injection_detected"]:
            verdict = "injection_detected"
        # Filter `status` menerima dua bentuk: verdict lama (answered/refused/…)
        # dan nama event (doc_update/request_create/…). Dulu filter event
        # selalu kosong karena verdict non-chat dipaksa jadi "view".
        if status and verdict != status and event_type != status:
            continue
        rec["verdict"] = verdict
        out.append(rec)
    if status:
        # Filter status berjalan di memori; potong setelah penyaringan supaya
        # `total` mencerminkan jumlah yang cocok, bukan jumlah yang dipindai.
        total = len(out)
        out = out[off:off + lim]
    page = {"items": out, "count": len(out), "total": total,
            "limit": lim, "offset": off, "has_more": off + len(out) < total}
    return paging.spread({}, "audit", page)


# ---- Export CSV (Sprint 4.6) ----
def _csv_response(filename: str, header: list, rows: list):
    """CSV UTF-8 dengan BOM supaya Excel (Windows) tidak merusak karakter non-ASCII."""
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf, delimiter=";", quoting=csv.QUOTE_MINIMAL)
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    body = "\ufeff" + buf.getvalue()
    return Response(
        content=body.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/audit/export.csv")
@db_locked
def export_audit_csv(user_id: str | None = None, status: str | None = None,
                     from_ts: str | None = None, to_ts: str | None = None,
                     limit: str | None = None, offset: str | None = None,
                     admin: dict = Depends(require_hr)):
    """Ekspor audit sesuai filter yang sama dengan tabel (bukan seluruh tabel)."""
    _rate_limit("audit:csv:" + admin["id"], READ_RATE_LIMIT)
    data = list_audit(admin=admin, user_id=user_id, status=status,
                      from_ts=from_ts, to_ts=to_ts,
                      include_sources=False, include_answers=True,
                      limit=limit or "200", offset=offset)
    header = ["id", "timestamp", "event_type", "verdict", "user_id", "user_role",
              "question", "permission_decision", "conflict", "injection", "pii",
              "engine", "llm_model", "selected_sources", "answer"]
    rows = [[r.get("id"), r.get("timestamp"), r.get("event_type"), r.get("verdict"),
             r.get("user_id"), r.get("user_role"), (r.get("question") or "").replace("\n", " "),
             r.get("permission_decision"),
             "YA" if r.get("conflict_detected") else "",
             "YA" if r.get("prompt_injection_detected") else "",
             "YA" if r.get("pii_masked") else "",
             r.get("engine_used") or "", r.get("llm_model") or "",
             ", ".join(r.get("selected_sources") or []),
             (r.get("answer") or "").replace("\n", " ")]
            for r in data.get("audit") or []]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return _csv_response(f"audit-{stamp}.csv", header, rows)


@app.get("/api/requests/export.csv")
@db_locked
def export_requests_csv(status: str | None = None, type: str | None = None,
                        scope: str = "mine", limit: str | None = None,
                        offset: str | None = None,
                        user: dict = Depends(parse_token)):
    """Ekspor pengajuan. `scope` mengikuti keranjang UI: mine|approval|all."""
    _rate_limit("req:csv:" + user["id"], READ_RATE_LIMIT)
    data = api_list_requests(user=user, status=status, type=type,
                             limit=limit or "200", offset=offset)
    key = scope if scope in ("mine", "approval", "all") else "mine"
    header = ["id", "user_id", "user_name", "type", "title", "amount", "start_date",
              "end_date", "status", "current_role", "stage", "stages",
              "created_at", "decided_by", "decided_at", "decision_note", "note"]
    rows = [[r.get("id"), r.get("user_id"), r.get("user_name") or "",
             r.get("type"), r.get("title") or "", r.get("amount") or "",
             r.get("start_date") or "", r.get("end_date") or "",
             r.get("status"), r.get("current_role") or "",
             "%d/%d" % ((r.get("stage_idx") or 0) + 1, len(r.get("stages") or [])),
             " > ".join(r.get("stages") or []),
             r.get("created_at") or "", r.get("decided_by") or "",
             r.get("decided_at") or "",
             (r.get("decision_note") or "").replace("\n", " "),
             (r.get("note") or "").replace("\n", " ")]
            for r in data.get(key) or []]
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    return _csv_response(f"pengajuan-{key}-{stamp}.csv", header, rows)


# ---- Static frontend (single origin: UI + API) ----
# UI disajikan terpisah (repo frontend sendiri). Bila direktori frontend tidak
# ada di mesin ini, API tetap berjalan normal tanpa route statis.
# Bila direktori tidak ditemukan, API tetap berjalan normal (tanpa route statis).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FRONTENDS = [
    os.environ.get("NOHALU_FRONTEND_DIR", ""),
    os.path.join(_ROOT, "..", "app-nohalu-ui"),
]
FRONTEND_DIR = next((p for p in _FRONTENDS if p and os.path.isdir(p)), None)
if FRONTEND_DIR:
    _FRONTEND_ROOT = os.path.abspath(FRONTEND_DIR)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_or_static(full_path: str):
        """Serve file statis; fallback ke index.html untuk route SPA (/chat, /docs)."""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        target = os.path.abspath(os.path.join(_FRONTEND_ROOT, full_path))
        if not target.startswith(_FRONTEND_ROOT):
            raise HTTPException(status_code=404)
        if full_path and os.path.isfile(target):
            return FileResponse(target)
        return FileResponse(os.path.join(_FRONTEND_ROOT, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
