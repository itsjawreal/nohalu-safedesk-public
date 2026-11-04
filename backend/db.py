"""SQLite schema + connection helpers for Nohalu SafeDesk.

Pure stdlib sqlite3 — no external ORM dependency.
"""
import contextvars
import os
import hashlib
import sqlite3
import json
import threading

DEFAULT_DB_PATH = os.environ.get(
    "NOHALU_DB_PATH",
    os.path.join(os.path.dirname(__file__), "data", "safedesk.db"),
)


BUSY_TIMEOUT_MS = int(os.environ.get("NOHALU_DB_BUSY_TIMEOUT_MS", "5000"))
_INIT_DONE: set[str] = set()
_INIT_LOCK = threading.Lock()

# Penanda request aktif. Koneksi SQLite dimiliki per-thread, sementara thread
# FastAPI dipakai ulang antar request. Tanpa penanda ini, transaksi yang
# ditinggalkan request yang gagal (menulis lalu `raise` sebelum `commit`)
# akan ikut ter-commit oleh request BERIKUTNYA di thread yang sama — data
# parsial masuk DB tanpa pernah lolos validasi. Middleware request menetapkan
# nilai ini; `_conn()` membuang transaksi sisa saat nilainya berubah.
_REQUEST_SCOPE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "safedesk_request_scope", default=None)


def begin_request_scope(scope_id: str):
    """Tandai awal request; kembalikan token untuk `end_request_scope`."""
    return _REQUEST_SCOPE.set(scope_id)


def end_request_scope(token) -> None:
    try:
        _REQUEST_SCOPE.reset(token)
    except ValueError:
        # Token dari context lain (mis. thread yang di-reuse anyio) — abaikan;
        # scope berikutnya tetap menetapkan nilai baru.
        pass


def _log(event: str, msg: str, **fields) -> None:
    """Satu baris log terstruktur tanpa mengimpor logging_setup saat import.

    `print()` di sini dulu menulis baris non-JSON ke tengah stream log JSON,
    sehingga agregator yang mem-parse per baris gagal pada baris seed/backfill.
    Impor lokal menjaga db.py tetap bisa dipakai mandiri (tanpa konfigurasi log).
    """
    try:
        import logging_setup
        logging_setup.event(logging_setup.get_logger("db"), event, msg, **fields)
    except Exception:  # logging tidak boleh menggagalkan migrasi
        pass


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Buka koneksi SQLite dengan pengaturan tahan-konkurensi.

    - `journal_mode=WAL`: pembaca tidak diblokir penulis, dan transaksi yang
      terputus tidak meninggalkan `-journal` yang mengunci DB setelah restart.
    - `busy_timeout`: penulis kedua menunggu (bukan langsung
      `database is locked`) selama transaksi lain masih berjalan singkat.
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False,
                           timeout=BUSY_TIMEOUT_MS / 1000.0)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
    except sqlite3.OperationalError:
        # mis. DB readonly / :memory: -> tetap jalan dengan mode default
        pass
    key = os.path.abspath(db_path)
    with _INIT_LOCK:
        first = key not in _INIT_DONE
    if first:
        init_db(conn)
        with _INIT_LOCK:
            _INIT_DONE.add(key)
    return conn


class ThreadLocalConnection:
    """Proxy koneksi: satu koneksi SQLite per thread.

    FastAPI menjalankan endpoint sinkron di threadpool. Satu koneksi bersama
    membuat cursor saling menimpa dan memaksa lock global ditahan selama I/O
    jaringan (LLM), sehingga satu pertanyaan lambat membekukan seluruh aplikasi.
    Dengan koneksi per-thread, setiap request punya cursor sendiri dan transaksi
    bisa dibuat pendek.

    Objek ini meneruskan seluruh method sqlite3.Connection, jadi kode lama
    (`CONN.execute(...)`) tetap bekerja tanpa perubahan.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        scope = _REQUEST_SCOPE.get()
        if conn is None:
            conn = connect(self.db_path)
            self._local.conn = conn
            self._local.scope = scope
            return conn
        if getattr(self._local, "scope", None) != scope:
            # Thread ini dipakai ulang untuk request lain. Transaksi yang
            # masih terbuka milik request sebelumnya = tulisan parsial yang
            # tidak pernah di-commit; buang, jangan diwariskan.
            if conn.in_transaction:
                conn.rollback()
            self._local.scope = scope
        return conn

    def discard_pending(self) -> bool:
        """Rollback transaksi yang masih terbuka. True bila ada yang dibuang."""
        conn = getattr(self._local, "conn", None)
        if conn is not None and conn.in_transaction:
            conn.rollback()
            return True
        return False

    # -- method yang dipakai kode aplikasi -------------------------------
    def execute(self, *args, **kwargs):
        return self._conn().execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._conn().executemany(*args, **kwargs)

    def executescript(self, *args, **kwargs):
        return self._conn().executescript(*args, **kwargs)

    def commit(self):
        return self._conn().commit()

    def rollback(self):
        return self._conn().rollback()

    def close(self):
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def cursor(self):
        return self._conn().cursor()

    @property
    def row_factory(self):
        return self._conn().row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._conn().row_factory = value

    def __getattr__(self, name):
        return getattr(self._conn(), name)


def connect_threadlocal(db_path: str = DEFAULT_DB_PATH) -> ThreadLocalConnection:
    return ThreadLocalConnection(db_path)


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            allowed_workspaces TEXT NOT NULL   -- JSON array
        );
        CREATE TABLE IF NOT EXISTS workspaces (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source_type TEXT NOT NULL,
            effective_date TEXT,
            workspace_ids TEXT NOT NULL,       -- JSON array
            access_roles TEXT NOT NULL,        -- JSON array
            content TEXT NOT NULL,
            owner TEXT,
            confidentiality TEXT DEFAULT 'internal',
            status TEXT DEFAULT 'indexed'
        );
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT DEFAULT 'active'
        );
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            citations TEXT,                    -- JSON array
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id TEXT,
            user_role TEXT,
            question TEXT,
            permission_decision TEXT,
            allowed_sources TEXT,              -- JSON array of doc ids
            retrieved_sources TEXT,            -- JSON array of doc ids
            selected_sources TEXT,             -- JSON array of doc ids
            conflict_detected INTEGER DEFAULT 0,
            answer TEXT,
            citations TEXT,                    -- JSON array
            refusal_reason TEXT,
            pii_masked INTEGER DEFAULT 0,
            prompt_injection_detected INTEGER DEFAULT 0,
            agent_version TEXT
        );
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            type TEXT NOT NULL,                -- cuti | reimbursement
            title TEXT,
            amount TEXT,
            start_date TEXT,
            end_date TEXT,
            note TEXT,
            status TEXT DEFAULT 'pending',      -- pending | approved | rejected | cancelled
            approver_roles TEXT NOT NULL,       -- JSON array of roles allowed to decide
            stages TEXT,                        -- JSON array of roles (berjenjang)
            stage_idx INTEGER DEFAULT 0,
            history TEXT,                       -- JSON array of decisions
            created_at TEXT NOT NULL,
            decided_by TEXT,
            decided_at TEXT,
            decision_note TEXT
        );
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            message TEXT NOT NULL,
            link TEXT,
            is_read INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            category TEXT DEFAULT 'system'     -- approval|request|payslip|announcement|system
        );
        -- Preferensi per kategori (Sprint 3.2). Tidak ada baris = aktif.
        CREATE TABLE IF NOT EXISTS notification_prefs (
            user_id TEXT NOT NULL,
            category TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (user_id, category)
        );
        -- Idempotensi notifikasi proaktif: satu dedupe_key per user hanya sekali.
        CREATE TABLE IF NOT EXISTS notification_log (
            user_id TEXT NOT NULL,
            dedupe_key TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (user_id, dedupe_key)
        );
        -- Penanda baca pengumuman HR (Sprint 3.3).
        CREATE TABLE IF NOT EXISTS announcement_reads (
            user_id TEXT NOT NULL,
            doc_id TEXT NOT NULL,
            read_at TEXT NOT NULL,
            PRIMARY KEY (user_id, doc_id)
        );
        CREATE TABLE IF NOT EXISTS leave_balances (
            user_id TEXT NOT NULL,
            year INTEGER NOT NULL,
            entitlement INTEGER NOT NULL DEFAULT 12,
            carry_over INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (user_id, year)
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,               -- YYYY-MM-DD (waktu kantor)
            clock_in TEXT,                    -- HH:MM
            clock_out TEXT,
            status TEXT DEFAULT 'present',    -- present | late | leave | sick | wfh
            source TEXT DEFAULT 'web',
            UNIQUE (user_id, date)
        );
        CREATE TABLE IF NOT EXISTS payslips (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            period TEXT NOT NULL,             -- YYYY-MM
            gross INTEGER,
            deductions INTEGER,
            net INTEGER,
            file_ref TEXT,
            published_at TEXT,
            UNIQUE (user_id, period)
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            audit_id INTEGER NOT NULL,
            user_id TEXT NOT NULL,
            value INTEGER NOT NULL,             -- 1 = membantu, -1 = tidak
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS agent_prepared (
            token TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT,
            action TEXT NOT NULL,              -- create_request | cancel_request
            payload TEXT NOT NULL,             -- JSON, sudah divalidasi
            summary TEXT,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used INTEGER DEFAULT 0
        );
        -- Sesi login (token bertanda tangan). Yang disimpan hanya HASH token:
        -- bocornya tabel ini tidak memberi token yang bisa dipakai.
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            issued_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,          -- ISO UTC
            revoked_at TEXT,
            user_agent TEXT,
            ip TEXT
        );
        -- Riwayat versi dokumen (Sprint 4.4). Satu baris = satu isi dokumen pada
        -- satu versi. Versi terbaru SELALU juga ada di tabel `documents`; tabel
        -- ini menyimpan yang lama supaya diff dan rollback mungkin tanpa
        -- menggandakan tabel dokumen untuk query retrieval.
        CREATE TABLE IF NOT EXISTS document_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            version TEXT NOT NULL,
            title TEXT,
            content TEXT,
            content_hash TEXT,
            status TEXT,
            changed_by TEXT,
            changed_at TEXT NOT NULL,
            note TEXT,
            UNIQUE (doc_id, version)
        );
        """
    )
    _ensure_columns(conn, "audit_log", {
        "engine_used": "TEXT",
        "llm_model": "TEXT",
        "llm_ms": "INTEGER",
        "llm_error": "TEXT",
        "event_type": "TEXT DEFAULT 'qa'",
        "document_id": "TEXT",
        # Detail tambahan event non-chat (perubahan field, id pengajuan, dst).
        "details": "TEXT",
    })
    _ensure_columns(conn, "users", {
        # Sprint 1 — profil karyawan (dipakai org chart & intent personal)
        "department": "TEXT",
        "job_title": "TEXT",
        "manager_id": "TEXT",
        "join_date": "TEXT",
        "employment_status": "TEXT DEFAULT 'active'",
        "location": "TEXT",
        "employee_no": "TEXT",
        "email": "TEXT",
    })
    _ensure_columns(conn, "chat_sessions", {
        "title": "TEXT",
        "updated_at": "TEXT",
    })
    _ensure_columns(conn, "chat_messages", {
        # 1 = jawaban tidak selesai (server error/timeout) dan hanya penanda.
        "interrupted": "INTEGER DEFAULT 0",
        # Jejak tool yang dipanggil agent untuk pesan ini (JSON array),
        # dipakai UI untuk menampilkan langkah kerja dan untuk audit.
        "trace": "TEXT",
        # Draft aksi yang menunggu konfirmasi user (JSON: token/action/summary).
        # Disimpan supaya kartu konfirmasi tetap tampil saat riwayat dimuat ulang.
        "prepared": "TEXT",
        # 1 = draft pada pesan ini sudah DIJALANKAN lewat tombol konfirmasi.
        # Bedanya dengan `agent_prepared.used`: `used` juga 1 untuk draft yang
        # dibatalkan, sedangkan kartu yang dibatalkan memang harus hilang.
        # Tanpa kolom ini, membuka ulang riwayat memunculkan lagi tombol
        # "Setujui & kirim" untuk pengajuan yang SUDAH masuk.
        "prepared_done": "INTEGER DEFAULT 0",
        # Saran lanjutan yang dikirim bersama jawaban (JSON array). Dulu saran
        # hanya hidup di respons HTTP: begitu halaman dimuat ulang atau sesi
        # lama dibuka, blok "Pertanyaan lanjutan" hilang dari riwayat.
        "suggestions": "TEXT",
    })
    _ensure_columns(conn, "documents", {
        "status": "TEXT DEFAULT 'indexed'",
        # Soft delete (Sprint 4.5). Baris tetap ada supaya audit & versi tidak
        # kehilangan induknya; semua query retrieval menyaring `deleted_at IS NULL`.
        "deleted_at": "TEXT",
        "deleted_by": "TEXT",
        "version": "TEXT DEFAULT '1'",
        "supersedes": "TEXT",
        "content_hash": "TEXT",
        # Data terstruktur yang menyertai isi dokumen (mis. fase proyek, jam
        # kerja). Halaman non-chat menggambar grafik dari sini, bukan dari
        # mengurai prosa: angka di layar tidak boleh hasil tebakan regex.
        "meta": "TEXT",
        # 1 = dokumen sudah disunting admin/HR lewat API. Seed sync tidak boleh
        # menimpanya, kalau tidak setiap restart menghapus hasil edit.
        "local_edit": "INTEGER DEFAULT 0",
    })
    _ensure_columns(conn, "notifications", {
        "category": "TEXT DEFAULT 'system'",
        # `ref` = id objek sumber (mis. doc_id pengumuman). Tanpa ini, membaca
        # satu pengumuman tidak bisa membersihkan notifikasinya secara presisi.
        "ref": "TEXT",
    })
    _ensure_columns(conn, "requests", {
        "stages": "TEXT",
        "stage_idx": "INTEGER DEFAULT 0",
        "history": "TEXT",
        # Sprint 0 — integritas workflow
        "amount_value": "INTEGER",          # nominal rupiah asli (bukan TEXT mentah)
        "client_ref": "TEXT",               # idempotency key dari klien
        "cancelled_at": "TEXT",
        "self_approved": "INTEGER DEFAULT 0",
        "overridden_by": "TEXT",
    })
    # Idempotency: satu (user, client_ref) hanya boleh menghasilkan satu pengajuan.
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_requests_client_ref "
        "ON requests(user_id, client_ref) WHERE client_ref IS NOT NULL"
    )
    # Index untuk query yang benar-benar dipakai portal/notifikasi.
    conn.execute("CREATE INDEX IF NOT EXISTS ix_requests_status ON requests(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_requests_user ON requests(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_notifications_user "
                 "ON notifications(user_id, is_read)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_notifications_cat "
                 "ON notifications(user_id, category, is_read)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_chat_messages_session "
                 "ON chat_messages(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_audit_user ON audit_log(user_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_documents_status ON documents(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_documents_deleted ON documents(deleted_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_doc_versions_doc "
                 "ON document_versions(doc_id, version)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_sessions_user "
                 "ON sessions(user_id, revoked_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_users_manager ON users(manager_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_attendance_user_date "
                 "ON attendance(user_id, date)")
    conn.execute("CREATE INDEX IF NOT EXISTS ix_payslips_user ON payslips(user_id, period)")
    conn.commit()
    _backfill_amount_value(conn)


def _backfill_amount_value(conn: sqlite3.Connection) -> None:
    """Isi `amount_value` untuk baris lama yang masih menyimpan nominal sebagai TEXT.

    Idempotent: hanya menyentuh baris yang amount_value-nya masih NULL.
    """
    rows = conn.execute(
        "SELECT id, amount FROM requests WHERE amount_value IS NULL AND amount IS NOT NULL"
    ).fetchall()
    if not rows:
        return
    changed = 0
    for r in rows:
        digits = "".join(ch for ch in str(r["amount"]) if ch.isdigit())
        if not digits:
            continue
        conn.execute("UPDATE requests SET amount_value = ? WHERE id = ?",
                     (int(digits), r["id"]))
        changed += 1
    if changed:
        conn.commit()


def backfill_chat_meta(conn: sqlite3.Connection) -> dict:
    """Lengkapi metadata pesan lama setelah kolom baru ditambahkan.

    Dua hal yang tidak bisa diisi `ALTER TABLE`:

    1. `suggestions` — dulu saran hanya hidup di respons HTTP, jadi membuka
       ulang sesi kehilangan blok "Pertanyaan lanjutan" pada jawaban terakhir.
       Saran dihitung ulang dari `trace` + peran pemilik sesi (deterministik).
    2. `prepared_done` — status kartu draft lama. Baris yang dibuat sebelum
       kolom ini ada bernilai 0, sehingga kartu draft yang SUDAH dijalankan
       tampil lagi dengan tombol aktif. Ditandai dari `agent_prepared.used`:
       draft yang dijalankan ATAU dibatalkan sama-sama `used=1`, tetapi hanya
       yang dijalankan punya pengajuan yang benar-benar tersimpan. Pembeda
       yang dipakai: pesan hasil konfirmasi ("Pengajuan #N tersimpan") di
       sesi yang sama, tepat setelah kartu itu.

    3. `title` — sesi yang dibuat sebelum `_touch_session(title_hint)` ada
       (commit HOTFIX H) tersimpan dengan judul kosong, sehingga 86% riwayat
       prod tampil sebagai "Percakapan baru". Judul diisi dari pesan user
       PERTAMA di sesi itu (sumber yang sama dengan yang dipakai kode baru),
       dipotong 120 karakter seperti `_touch_session`.

    Idempotent: baris yang sudah terisi tidak disentuh.
    """
    out = {"suggestions": 0, "prepared_done": 0, "titles": 0}
    try:
        import agent as agent_mod  # local import: hindari siklus saat modul dimuat
        import engine as engine_mod
        rows = conn.execute(
            "SELECT m.id, m.text, m.trace, m.prepared_done, s.user_id"
            " FROM chat_messages m JOIN chat_sessions s ON s.id = m.session_id"
            " WHERE m.role = 'assistant' AND m.suggestions IS NULL"
        ).fetchall()
        users = {}
        for r in rows:
            uid = r["user_id"]
            if uid not in users:
                users[uid] = engine_mod.get_user(conn, uid)
            user = users[uid]
            if not user:
                continue
            try:
                trace = json.loads(r["trace"]) if r["trace"] else []
            except (ValueError, TypeError):
                trace = []
            try:
                items = agent_mod._suggestions(
                    {"agent_trace": trace, "prepared_done": bool(r["prepared_done"])},
                    r["text"] or "", conn=conn, user=user)
            except Exception:
                items = []
            conn.execute("UPDATE chat_messages SET suggestions=? WHERE id=?",
                         (json.dumps(items, ensure_ascii=False), r["id"]))
            out["suggestions"] += 1
        conn.commit()
    except Exception as exc:
        _log("backfill_skipped", "backfill saran dilewati", step="suggestions", error=str(exc))

    try:
        # Kartu draft lama: tandai selesai bila tokennya sudah dipakai DAN
        # ada pesan konfirmasi hasil di sesi yang sama.
        rows = conn.execute(
            "SELECT id, session_id, prepared FROM chat_messages"
            " WHERE prepared IS NOT NULL AND prepared_done = 0"
        ).fetchall()
        for r in rows:
            try:
                token = (json.loads(r["prepared"]) or {}).get("token") or ""
            except (ValueError, TypeError):
                token = ""
            if not token:
                continue
            p = conn.execute("SELECT used FROM agent_prepared WHERE token=?",
                             (token,)).fetchone()
            if not p or not p["used"]:
                continue
            done = conn.execute(
                "SELECT 1 FROM chat_messages WHERE session_id=? AND id>?"
                " AND role='assistant' AND text LIKE '%tersimpan%' LIMIT 1",
                (r["session_id"], r["id"])).fetchone()
            if done:
                conn.execute("UPDATE chat_messages SET prepared_done=1 WHERE id=?",
                             (r["id"],))
                out["prepared_done"] += 1
        conn.commit()
    except Exception as exc:
        _log("backfill_skipped", "backfill prepared_done dilewati", step="prepared_done", error=str(exc))

    try:
        # Saran yang dibuat SEBELUM `prepared_done` diketahui masih menawarkan
        # "Ya, kirim pengajuannya." pada draft yang sudah dijalankan. Dihitung
        # ulang supaya riwayat tidak menyuruh user mengirim dua kali.
        rows = conn.execute(
            "SELECT m.id, m.text, m.trace, s.user_id"
            " FROM chat_messages m JOIN chat_sessions s ON s.id = m.session_id"
            " WHERE m.role = 'assistant' AND m.prepared_done = 1"
            "   AND m.suggestions LIKE '%kirim pengajuannya%'"
        ).fetchall()
        users = {}
        for r in rows:
            uid = r["user_id"]
            if uid not in users:
                users[uid] = engine_mod.get_user(conn, uid)
            user = users[uid]
            if not user:
                continue
            try:
                trace = json.loads(r["trace"]) if r["trace"] else []
            except (ValueError, TypeError):
                trace = []
            items = agent_mod._suggestions(
                {"agent_trace": trace, "prepared_done": True},
                r["text"] or "", conn=conn, user=user)
            conn.execute("UPDATE chat_messages SET suggestions=? WHERE id=?",
                         (json.dumps(items, ensure_ascii=False), r["id"]))
            out["suggestions"] += 1
        conn.commit()
    except Exception as exc:
        _log("backfill_skipped", "backfill saran draft selesai dilewati", step="prepared_suggestions", error=str(exc))

    try:
        # Judul sesi lama: pakai pesan user PERTAMA (bukan yang terbaru) supaya
        # judul tetap menggambarkan topik awal percakapan, sama seperti yang
        # dihasilkan `_touch_session(title_hint)` untuk sesi baru.
        rows = conn.execute(
            "SELECT s.id, (SELECT m.text FROM chat_messages m"
            "               WHERE m.session_id = s.id AND m.role = 'user'"
            "               ORDER BY m.id LIMIT 1) AS first_user"
            " FROM chat_sessions s"
            " WHERE s.title IS NULL OR TRIM(s.title) = ''"
        ).fetchall()
        for r in rows:
            hint = (r["first_user"] or "").strip()
            if not hint:
                continue
            conn.execute("UPDATE chat_sessions SET title = ? WHERE id = ?",
                         (hint[:120], r["id"]))
            out["titles"] += 1
        conn.commit()
    except Exception as exc:
        _log("backfill_skipped", "backfill judul sesi dilewati", step="titles", error=str(exc))

    if out["suggestions"] or out["prepared_done"] or out["titles"]:
        _log("backfill_chat_meta", "backfill metadata chat selesai", **out)
    return out


def _ensure_columns(conn: sqlite3.Connection, table: str, cols: dict) -> None:
    existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, typ in cols.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {typ}")


def seed_if_empty(conn: sqlite3.Connection) -> bool:
    """Seed mock data only when tables are empty. Returns True if seeded."""
    cur = conn.execute("SELECT COUNT(*) AS n FROM documents")
    if cur.fetchone()["n"] > 0:
        return False
    from seed import SEED_USERS, SEED_WORKSPACES, SEED_DOCUMENTS  # local import

    conn.executemany(
        "INSERT INTO users (id, name, role, allowed_workspaces) VALUES (?,?,?,?)",
        [(u["id"], u["name"], u["role"], json.dumps(u["allowed_workspaces"])) for u in SEED_USERS],
    )
    conn.executemany(
        "INSERT INTO workspaces (id, name) VALUES (?,?)",
        [(w["id"], w["name"]) for w in SEED_WORKSPACES],
    )
    # Isi seluruh kolom yang dibandingkan `ensure_seed_docs`, supaya instalasi
    # baru tidak langsung "meng-update" 62 dokumen hanya karena content_hash NULL.
    conn.executemany(
        """INSERT INTO documents
           (id,title,source_type,effective_date,workspace_ids,access_roles,content,owner,
            confidentiality,version,supersedes,status,content_hash,meta)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (
                d["id"], d["title"], d["source_type"], d.get("effective_date"),
                json.dumps(d["workspace_ids"]), json.dumps(d["access_roles"]),
                d["content"], d.get("owner"), d.get("confidentiality", "internal"),
                d.get("version", "1"), d.get("supersedes"),
                d.get("status", "indexed"),
                hashlib.sha256((d["content"] or "").encode()).hexdigest(),
                _dump_meta(d.get("meta")),
            )
            for d in SEED_DOCUMENTS
        ],
    )
    conn.commit()
    return True


def ensure_seed_users(conn: sqlite3.Connection) -> list:
    """Idempotently sync seed users (termasuk profil karyawan) ke DB.

    `seed_if_empty` hanya berjalan saat tabel documents kosong, sehingga user
    baru (mis. role `finance` yang dibutuhkan approval > Rp1jt, atau rekan tim
    untuk org chart) tidak pernah masuk ke DB produksi yang sudah ter-seed.
    Fungsi ini menambal gap itu:
    - insert user yang belum ada (lengkap dengan profil)
    - update name/role/workspaces/profil bila berubah
    Mengembalikan daftar id yang baru ditambahkan.
    """
    from seed import SEED_USERS  # local import

    PROFILE = ("department", "job_title", "manager_id", "join_date",
               "employment_status", "location", "employee_no", "email")
    existing = {r["id"]: r for r in conn.execute("SELECT * FROM users").fetchall()}
    added, updated = [], []
    for u in SEED_USERS:
        ws = json.dumps(u["allowed_workspaces"], sort_keys=True)
        row = existing.get(u["id"])
        if row is None:
            cols = ["id", "name", "role", "allowed_workspaces"] + list(PROFILE)
            vals = [u["id"], u["name"], u["role"], ws] + [u.get(k) for k in PROFILE]
            conn.execute(
                "INSERT INTO users (%s) VALUES (%s)"
                % (",".join(cols), ",".join("?" * len(cols))), vals)
            added.append(u["id"])
            continue
        changed = (row["name"] != u["name"] or row["role"] != u["role"]
                   or json.dumps(json.loads(row["allowed_workspaces"] or "[]"),
                                 sort_keys=True) != ws)
        for k in PROFILE:
            keys = row.keys()
            if k in keys and (row[k] or None) != (u.get(k) or None):
                changed = True
                break
        if changed:
            conn.execute(
                "UPDATE users SET name=?, role=?, allowed_workspaces=?, "
                + ", ".join("%s=?" % k for k in PROFILE)
                + " WHERE id=?",
                [u["name"], u["role"], ws] + [u.get(k) for k in PROFILE] + [u["id"]])
            updated.append(u["id"])
    # Commit walau tidak ada perubahan: INSERT/UPDATE yang di-skip tetap bisa
    # meninggalkan transaksi tulis terbuka dan mengunci WAL untuk request lain.
    conn.commit()
    if added or updated:
        _log("seed_users", "seed pengguna selesai", added=added, updated=updated)
    return added


def ensure_seed_docs(conn: sqlite3.Connection) -> list:
    """Idempotently sync seed docs into the DB.

    - Inserts any seed doc that is not yet present.
    - Updates existing seed docs whose seed metadata changed (title/source_type/
      effective_date/workspace_ids/access_roles/content/owner/confidentiality),
      so perbaikan isi dokumen ikut ter-deploy tanpa menghapus data lain
      (chat history, audit, dokumen upload admin tetap utuh).
    Returns the list of doc ids that were newly inserted (kept for back-compat).
    """
    from seed import SEED_DOCUMENTS  # local import

    cols = ("id,title,source_type,effective_date,workspace_ids,access_roles,content,owner,"
            "confidentiality,version,supersedes,status,content_hash,local_edit,meta")
    existing = {r["id"]: r for r in conn.execute("SELECT " + cols + " FROM documents").fetchall()}
    added = []
    updated = []
    skipped_local = []
    for d in SEED_DOCUMENTS:
        row = existing.get(d["id"])
        if row is None:
            conn.execute(
                """INSERT INTO documents
                   (id,title,source_type,effective_date,workspace_ids,access_roles,content,owner,confidentiality,version,supersedes,status,content_hash,meta)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    d["id"], d["title"], d["source_type"], d.get("effective_date"),
                    json.dumps(d["workspace_ids"]), json.dumps(d["access_roles"]),
                    d["content"], d.get("owner"), d.get("confidentiality", "internal"),
                    d.get("version", "1"), d.get("supersedes"), d.get("status", "indexed"),
                    hashlib.sha256((d["content"] or "").encode()).hexdigest(),
                    _dump_meta(d.get("meta")),
                ),
            )
            added.append(d["id"])
            continue
        if row["local_edit"]:
            # Sudah disunting lewat admin/HR: jangan ditimpa seed.
            skipped_local.append(d["id"])
            continue
        new_fields = (
            d["title"], d["source_type"], d.get("effective_date"),
            json.dumps(d["workspace_ids"], sort_keys=True),
            json.dumps(d["access_roles"], sort_keys=True),
            d["content"], d.get("owner"), d.get("confidentiality", "internal"),
            d.get("version", "1"), d.get("supersedes"), d.get("status", "indexed"),
            hashlib.sha256((d["content"] or "").encode()).hexdigest(),
            _dump_meta(d.get("meta")),
        )
        old_fields = (
            row["title"], row["source_type"], row["effective_date"],
            row["workspace_ids"], row["access_roles"],
            row["content"], row["owner"], row["confidentiality"],
            row["version"], row["supersedes"], row["status"], row["content_hash"],
            row["meta"],
        )
        if new_fields != old_fields:
            conn.execute(
                """UPDATE documents SET title=?, source_type=?, effective_date=?,
                   workspace_ids=?, access_roles=?, content=?, owner=?, confidentiality=?,
                   version=?, supersedes=?, status=?, content_hash=?, meta=?
                   WHERE id=?""",
                (
                    d["title"], d["source_type"], d.get("effective_date"),
                    json.dumps(d["workspace_ids"], sort_keys=True),
                    json.dumps(d["access_roles"], sort_keys=True),
                    d["content"], d.get("owner"), d.get("confidentiality", "internal"),
                    d.get("version", "1"), d.get("supersedes"), d.get("status", "indexed"),
                    hashlib.sha256((d["content"] or "").encode()).hexdigest(),
                    _dump_meta(d.get("meta")),
                    d["id"],
                ),
            )
            updated.append(d["id"])
    # Lihat catatan di ensure_seed_users: commit tanpa syarat agar write-lock
    # tidak tertinggal saat semua dokumen sudah sinkron.
    conn.commit()
    if updated:
        _log("seed_docs_updated", "seed dokumen diperbarui", ids=updated)
    if skipped_local:
        # Dokumen yang disunting admin tidak ditimpa seed: catat supaya operator
        # tahu kenapa sebuah koreksi seed tidak muncul di DB lama.
        _log("seed_docs_skipped", "dokumen disunting admin, seed dilewati",
             ids=skipped_local)
    return added


def _dump_meta(meta):
    """Kebalikan `_load_meta`: dict → JSON teks, None bila kosong."""
    if not meta:
        return None
    return json.dumps(meta, ensure_ascii=False, sort_keys=True)


def _load_meta(raw):
    """Kolom `meta` menyimpan JSON; nilai rusak/kosong jadi None, bukan error.

    Dokumen tanpa meta tetap sah — halaman yang butuh grafik memeriksa None dan
    jatuh ke tampilan teks biasa.
    """
    if not raw:
        return None
    try:
        v = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return v if isinstance(v, dict) else None


def _load_str_list(raw) -> list:
    """Kolom list (`workspace_ids`, `access_roles`) selalu dibaca sebagai list.

    `json.loads('"ws_onboarding"')` mengembalikan STRING, bukan list. Baris
    lama yang tersimpan sebagai string membuat setiap klien yang memanggil
    `.join()`/`.map()` melempar TypeError dan halaman berhenti di DOM lama.
    Nilai tak terbaca diperlakukan sebagai daftar kosong: dokumen tetap tampil
    (judul/isi masih berguna), hanya izinnya yang harus diperbaiki admin.
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    try:
        val = json.loads(raw)
    except (TypeError, ValueError):
        val = raw
    if isinstance(val, str):
        return [x.strip() for x in val.split(",") if x.strip()]
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    return []


def row_to_doc(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "source_type": row["source_type"],
        "effective_date": row["effective_date"],
        "workspace_ids": _load_str_list(row["workspace_ids"]),
        "access_roles": _load_str_list(row["access_roles"]),
        "content": row["content"],
        "owner": row["owner"],
        "confidentiality": row["confidentiality"],
        "status": row["status"] if "status" in row.keys() else "indexed",
        "version": row["version"] if "version" in row.keys() else "1",
        "supersedes": row["supersedes"] if "supersedes" in row.keys() else None,
        "content_hash": row["content_hash"] if "content_hash" in row.keys() else None,
        "meta": _load_meta(row["meta"] if "meta" in row.keys() else None),
        # Soft delete: baris tetap ada untuk audit, tetapi semua jalur baca
        # (retrieval, perpustakaan, manajemen) menyaring nilai non-NULL.
        "deleted_at": row["deleted_at"] if "deleted_at" in row.keys() else None,
        "deleted_by": row["deleted_by"] if "deleted_by" in row.keys() else None,
    }
