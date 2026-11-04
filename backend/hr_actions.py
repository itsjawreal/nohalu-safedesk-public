"""hr_actions.py — aksi HR/admin yang bisa dipicu dari chat.

Sebelum modul ini, kemampuan "melakukan sesuatu" untuk HR hanya hidup di
halaman: menerbitkan slip gaji, mengumumkan pengumuman, mengubah data
karyawan, memindahkan status dokumen. Agent hanya bisa MENJAWAB. Akibatnya
pertanyaan "tolong terbitkan slip Budi untuk Oktober" dijawab dengan
instruksi menekan tombol di halaman — bukan dikerjakan.

Modul ini memuat **lapisan domain**, bukan tool: setiap fungsi menerima
`conn` + `actor` yang sudah terautentikasi, memeriksa izin sendiri, menulis
audit, lalu mengembalikan `(data, status)` seperti `portal.py`. Pemisahan ini
disengaja:

  * `agent_tools.py` tetap berisi kontrak model (skema + dua fase konfirmasi);
  * `app.py` bisa memakai fungsi yang sama untuk rute HTTP;
  * aturan izin tidak ditulis dua kali (dulu `_manageable_docs` hanya hidup
    di `app.py`, sehingga jalur chat harus menyalinnya).

Prinsip yang sama dengan `portal.py`: penolakan akses memakai 404 untuk
dokumen (keberadaan dokumen restricted tidak boleh bocor), dan setiap tulisan
meninggalkan jejak audit.
"""
import hashlib
import json
import os
import uuid

import docs_lifecycle as lifecycle
import engine
import personal
import portal
from docs_lifecycle import LifecycleError

# Sumber tunggal nilai yang sah — sama dengan yang dipakai app.py.
DOC_CONFIDENTIALITY = ("internal", "restricted")
DOC_STATUSES = lifecycle.DOC_STATUSES
ROLES = ("employee", "team_lead", "hr", "finance", "admin")
EMPLOYMENT_STATUSES = ("active", "inactive", "probation", "resigned")

MAX_DOC_CONTENT = int(os.environ.get("NOHALU_MAX_DOC_CHARS", "200000"))
MAX_ANNOUNCEMENT_TITLE = 160


def _err(code, message, status, fields=None):
    out = {"error": code, "message": message}
    if fields:
        out["fields"] = fields
    return out, status


def _now():
    return portal._now()


def _as_list(value, field="value"):
    """Normalisasi JSON-array/CSV menjadi list[str] (kontrak sama dengan app.py)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    raise ValueError(f"{field} harus berupa daftar atau string dipisah koma.")


def _content_hash(text):
    return hashlib.sha256((text or "").encode()).hexdigest()


# ---------------------------------------------------------------------------
# Izin
# ---------------------------------------------------------------------------
def _require_role(actor, roles, message):
    if (actor or {}).get("role") not in roles:
        return _err("not_authorized", message, 403)
    return None


def manageable_docs(conn, actor):
    """Dokumen yang boleh DIKELOLA actor (bukan sekadar dibaca).

    Aturan ini dulu hanya ada di `app.py::_manageable_docs`. HR bisa membaca
    dokumen yang lolos ACL-nya, tetapi TIDAK boleh mengelola dokumen restricted
    milik peran lain; admin melihat semuanya. Tanpa satu implementasi bersama,
    jalur chat akan menjadi pintu belakang yang melewati pembatasan itu.

    Dokumen TERHAPUS ikut disertakan. `filter_allowed_docs` membuangnya, dan
    itulah sebabnya pemulihan dari tempat sampah selalu 404 untuk HR: dokumen
    yang baru dihapus langsung hilang dari daftar yang boleh dikelola, padahal
    justru dokumen itu yang mau dipulihkan. Admin tidak terpengaruh karena
    cabangnya membaca seluruh tabel.
    """
    if actor["role"] == "admin":
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY effective_date DESC").fetchall()
        return [engine.row_to_doc(r) for r in rows]
    # ACL yang sama dengan `filter_allowed_docs`, tetapi TANPA menyaring
    # `deleted_at` dan TANPA menyaring status: manajemen butuh keduanya.
    uws = set(actor["allowed_workspaces"])
    roles = engine.effective_roles(actor["role"])
    out = []
    for row in conn.execute("SELECT * FROM documents").fetchall():
        d = engine.row_to_doc(row)
        if uws & set(d.get("workspace_ids") or []) and \
                roles & set(d.get("access_roles") or []):
            out.append(d)
    out.sort(key=lambda d: (d.get("effective_date") or ""), reverse=True)
    return out


def find_doc_any(conn, doc_id):
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return engine.row_to_doc(row) if row else None


def require_manageable(conn, actor, doc_id):
    """None bila boleh; dict error 404 bila tidak (keberadaan tidak dibocorkan)."""
    if doc_id not in {d["id"] for d in manageable_docs(conn, actor)}:
        return _err("doc_not_found",
                    "Dokumen tidak ditemukan atau bukan milik Anda kelola.", 404)
    return None


def known_workspaces(conn):
    try:
        return {r["id"] for r in conn.execute("SELECT id FROM workspaces")}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# 1. Pengumuman HR (dokumen `source_type='hr_announcement'`)
# ---------------------------------------------------------------------------
def validate_announcement(conn, actor, payload):
    """Validasi pengumuman TANPA menulis apa pun.

    Return `(err, status)`: `(None, 0)` bila sah. Dipakai tool draft supaya
    draft yang tampil ke user tidak mungkin ditolak saat tombol "Setujui"
    ditekan — validasi dan eksekusi memakai fungsi yang sama.
    """
    denied = _require_role(actor, ("hr", "admin"),
                           "Hanya HR/admin yang dapat menerbitkan pengumuman.")
    if denied:
        return denied
    title = (payload.get("title") or "").strip()
    content = (payload.get("content") or "").strip()
    if not title:
        return _err("title_required", "Judul pengumuman wajib diisi.", 422,
                    {"title": "wajib"})
    if len(title) > MAX_ANNOUNCEMENT_TITLE:
        return _err("title_too_long",
                    f"Judul maksimal {MAX_ANNOUNCEMENT_TITLE} karakter.", 422,
                    {"title": f"maks {MAX_ANNOUNCEMENT_TITLE}"})
    # Judul pengumuman dirender sebagai kartu DAN menjadi judul dokumen yang
    # masuk konteks model. Baris baru di sini cukup untuk menambah baris
    # instruksi di atas isi pada prompt.
    if any(ch in title for ch in ("\r", "\n", "\t")):
        return _err("title_invalid_characters",
                    "Judul tidak boleh memuat baris baru atau tab.", 422,
                    {"title": "tanpa baris baru"})
    if not content:
        return _err("content_required", "Isi pengumuman wajib diisi.", 422,
                    {"content": "wajib"})
    if len(content) > MAX_DOC_CONTENT:
        return _err("content_too_long",
                    f"Isi maksimal {MAX_DOC_CONTENT} karakter.", 422,
                    {"content": f"maks {MAX_DOC_CONTENT}"})
    try:
        workspace_ids = _as_list(payload.get("workspace_ids") or "ws_onboarding",
                                 "workspace_ids")
        access_roles = _as_list(payload.get("access_roles")
                                or "employee,team_lead,hr,admin", "access_roles")
    except ValueError as e:
        return _err("invalid_field_type", str(e), 422)
    known = known_workspaces(conn)
    unknown = [w for w in workspace_ids if known and w not in known]
    if unknown:
        return _err("unknown_workspace",
                    "Workspace tidak dikenal: " + ", ".join(unknown), 422,
                    {"workspace_ids": "|".join(sorted(known))})
    bad_roles = [r for r in access_roles if r not in ROLES]
    if bad_roles:
        return _err("invalid_role", "Peran tidak dikenal: " + ", ".join(bad_roles),
                    422, {"access_roles": "|".join(ROLES)})
    confidentiality = (payload.get("confidentiality") or "internal").strip()
    if confidentiality not in DOC_CONFIDENTIALITY:
        return _err("invalid_confidentiality",
                    "Klasifikasi harus internal atau restricted.", 422,
                    {"confidentiality": "internal|restricted"})
    effective_date = (payload.get("effective_date") or "").strip() or \
        personal.today().isoformat()
    if not portal._parse_iso_date(effective_date):
        return _err("invalid_effective_date",
                    "Tanggal efektif harus format YYYY-MM-DD.", 422,
                    {"effective_date": "YYYY-MM-DD"})
    return None, 0


def create_announcement(conn, actor, payload):
    """Terbitkan pengumuman HR dan beri tahu karyawan yang berhak.

    Pengumuman adalah dokumen biasa dengan `source_type='hr_announcement'`,
    sehingga RBAC, pencarian, dan halaman Pengumuman langsung memakainya.
    Notifikasi dikirim SETELAH dokumen tersimpan: kalau pengiriman gagal,
    pengumumannya tetap ada dan bisa diumumkan ulang, bukan sebaliknya.
    """
    bad, bad_status = validate_announcement(conn, actor, payload)
    if bad:
        return bad, bad_status
    title = (payload.get("title") or "").strip()
    content = (payload.get("content") or "").strip()
    workspace_ids = _as_list(payload.get("workspace_ids") or "ws_onboarding",
                             "workspace_ids")
    access_roles = _as_list(payload.get("access_roles")
                            or "employee,team_lead,hr,admin", "access_roles")
    confidentiality = (payload.get("confidentiality") or "internal").strip()
    effective_date = (payload.get("effective_date") or "").strip() or \
        personal.today().isoformat()

    doc_id = (payload.get("id") or "").strip() or ("doc_ann_" + uuid.uuid4().hex[:8])
    if conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone():
        return _err("doc_id_exists", f"ID dokumen sudah dipakai: {doc_id}", 409)

    h = _content_hash(content)
    conn.execute(
        """INSERT INTO documents
           (id,title,source_type,effective_date,workspace_ids,access_roles,content,
            owner,confidentiality,status,version,content_hash,local_edit)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,1)""",
        (doc_id, title, portal.ANNOUNCEMENT_SOURCE, effective_date,
         json.dumps(workspace_ids), json.dumps(access_roles), content,
         payload.get("owner") or "HR", confidentiality, "published", "1", h))
    lifecycle.snapshot(conn, {"id": doc_id, "title": title, "content": content,
                              "status": "published", "version": "1",
                              "content_hash": h}, actor, note="pengumuman baru")
    portal.audit_event(conn, actor, "announcement_create",
                       f"Menerbitkan pengumuman: {title}",
                       document_id=doc_id,
                       extra={"doc_id": doc_id, "workspaces": workspace_ids,
                              "roles": access_roles,
                              "confidentiality": confidentiality})
    conn.commit()

    # Notifikasi hanya untuk user yang benar-benar boleh membuka dokumen ini —
    # semantik sama dengan `engine.filter_allowed_docs` (irisan workspace+role).
    notified = _notify_announcement(conn, doc_id, title, workspace_ids, access_roles)
    return {"id": doc_id, "title": title, "status": "published",
            "effective_date": effective_date,
            "workspace_ids": workspace_ids, "access_roles": access_roles,
            "notified": notified}, 201


def _notify_announcement(conn, doc_id, title, workspace_ids, access_roles):
    """Kirim notifikasi ke karyawan aktif yang lolos ACL dokumen."""
    ws, roles = set(workspace_ids), set(access_roles)
    sent = 0
    for row in conn.execute(
            "SELECT id, role, allowed_workspaces FROM users "
            "WHERE COALESCE(employment_status,'active') = 'active'").fetchall():
        try:
            uws = set(json.loads(row["allowed_workspaces"] or "[]"))
        except (TypeError, ValueError):
            uws = set()
        if not (uws & ws):
            continue
        if not (engine.effective_roles(row["role"]) & roles):
            continue
        sent += portal.notify_once(
            conn, [row["id"]], f"Pengumuman HR baru: {title}",
            link="/app/pengumuman", category="announcement",
            dedupe_key=f"ann:{doc_id}:{row['id']}", ref=doc_id)
    conn.commit()
    return sent


# ---------------------------------------------------------------------------
# 2. Slip gaji
# ---------------------------------------------------------------------------
def publish_payslip(conn, actor, payload, db_path=None):
    """Terbitkan slip satu karyawan untuk satu periode.

    Mendelegasikan ke `personal.publish_payslip` supaya validasi nominal dan
    pembuatan PDF tetap satu sumber. Bedanya di sini: audit ditulis di sini
    karena jalur chat tidak lewat `app.py`.
    """
    denied = _require_role(actor, ("hr", "admin"),
                           "Hanya HR/admin yang dapat menerbitkan slip gaji.")
    if denied:
        return denied
    data, status = personal.publish_payslip(conn, actor, payload, db_path=db_path)
    if status >= 400:
        return data, status
    portal.audit_event(conn, actor, "payslip_publish",
                       f"Menerbitkan slip gaji {data.get('user_id')} "
                       f"periode {data.get('period')}",
                       extra={"target_user": data.get("user_id"),
                              "period": data.get("period"),
                              "net": data.get("net"), "via": "chat"})
    conn.commit()
    return data, status


def list_payslips(conn, actor, period=None, q=None, limit=20):
    """Daftar slip untuk HR/admin, dibatasi supaya konteks model tidak jebol."""
    data, status = personal.list_payslips(conn, actor, period=period, q=q)
    if status >= 400:
        return data, status
    try:
        limit = max(1, min(int(limit or 20), 50))
    except (TypeError, ValueError):
        limit = 20
    rows = data["payslips"]
    return {"count": len(rows), "periods": data.get("periods") or [],
            "without_file": data.get("without_file"),
            "payslips": rows[:limit],
            "truncated": len(rows) > limit}, 200


# ---------------------------------------------------------------------------
# 3. Data karyawan
# ---------------------------------------------------------------------------
def _users_admin():
    """Import lazy: `users_admin` mengimpor `rbac`, bukan `engine`."""
    import users_admin
    return users_admin


def create_employee(conn, actor, payload):
    denied = _require_role(actor, ("admin",),
                           "Hanya admin yang dapat menambah karyawan.")
    if denied:
        return denied
    ua = _users_admin()
    try:
        data = ua.create(conn, actor, payload)
    except ua.UserError as e:
        return _err(e.code, e.message, e.status, e.fields)
    portal.audit_event(conn, actor, "user_create",
                       f"Menambah karyawan: {payload.get('name') or data['id']}",
                       extra={"user_id": data["id"], "role": data["role"],
                              "via": "chat"})
    conn.commit()
    return data, 201


def update_employee(conn, actor, user_id, payload):
    denied = _require_role(actor, ("admin",),
                           "Hanya admin yang dapat mengubah data karyawan.")
    if denied:
        return denied
    ua = _users_admin()
    try:
        data = ua.update(conn, actor, user_id, payload)
    except ua.UserError as e:
        return _err(e.code, e.message, e.status, e.fields)
    if data.get("changed_fields"):
        portal.audit_event(conn, actor, "user_update",
                           f"Mengubah karyawan: {user_id}",
                           extra={"user_id": user_id,
                                  "changed_fields": data["changed_fields"],
                                  "via": "chat"})
        conn.commit()
    return data, 200


def deactivate_employee(conn, actor, user_id):
    denied = _require_role(actor, ("admin",),
                           "Hanya admin yang dapat menonaktifkan karyawan.")
    if denied:
        return denied
    ua = _users_admin()
    try:
        data = ua.deactivate(conn, actor, user_id)
    except ua.UserError as e:
        return _err(e.code, e.message, e.status, e.fields)
    portal.audit_event(conn, actor, "user_deactivate",
                       f"Menonaktifkan karyawan: {user_id}",
                       extra={"user_id": user_id,
                              "sessions_revoked": data.get("sessions_revoked", 0),
                              "via": "chat"})
    conn.commit()
    return data, 200


# ---------------------------------------------------------------------------
# 4. Siklus hidup dokumen
# ---------------------------------------------------------------------------
def _validate_doc(title, source_type, content, confidentiality, status):
    if not (title or "").strip():
        return _err("title_required", "Judul dokumen wajib diisi.", 422,
                    {"title": "wajib"})
    if any(ch in title for ch in ("\r", "\n", "\t")):
        return _err("title_invalid_characters",
                    "Judul tidak boleh memuat baris baru atau tab.", 422,
                    {"title": "tanpa baris baru"})
    if not (source_type or "").strip():
        return _err("source_type_required", "Tipe sumber wajib diisi.", 422,
                    {"source_type": "wajib"})
    if not (content or "").strip():
        return _err("content_required", "Isi dokumen wajib diisi.", 422,
                    {"content": "wajib"})
    if len(content) > MAX_DOC_CONTENT:
        return _err("content_too_long",
                    f"Isi maksimal {MAX_DOC_CONTENT} karakter.", 422,
                    {"content": f"maks {MAX_DOC_CONTENT}"})
    if confidentiality not in DOC_CONFIDENTIALITY:
        return _err("invalid_confidentiality",
                    "Klasifikasi harus internal atau restricted.", 422,
                    {"confidentiality": "internal|restricted"})
    if status not in DOC_STATUSES:
        return _err("invalid_status", "Status dokumen tidak dikenal.", 422,
                    {"status": "|".join(DOC_STATUSES)})
    return None


def create_document(conn, actor, payload):
    denied = _require_role(actor, ("hr", "admin"),
                           "Hanya HR/admin yang dapat membuat dokumen.")
    if denied:
        return denied
    status = (payload.get("status") or "published").strip()
    confidentiality = (payload.get("confidentiality") or "internal").strip()
    bad = _validate_doc(payload.get("title"), payload.get("source_type"),
                        payload.get("content"), confidentiality, status)
    if bad:
        return bad
    doc_id = (payload.get("id") or "").strip() or ("doc_" + uuid.uuid4().hex[:8])
    if conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone():
        return _err("doc_id_exists", f"ID dokumen sudah dipakai: {doc_id}", 409)
    # Dokumen TANPA workspace tidak bisa diakses siapa pun: `filter_allowed_docs`
    # memakai irisan workspace, sehingga daftar kosong berarti ACL kosong —
    # termasuk bagi HR yang baru membuatnya. Akibatnya dokumen itu langsung
    # "hilang" dan revisi berikutnya ditolak 404. Default-nya disamakan dengan
    # `POST /api/documents/upload` (ws_onboarding).
    workspace_ids = _as_list(payload.get("workspace_ids"), "workspace_ids") \
        or ["ws_onboarding"]
    known = known_workspaces(conn)
    unknown = [w for w in workspace_ids if known and w not in known]
    if unknown:
        return _err("unknown_workspace",
                    "Workspace tidak dikenal: " + ", ".join(unknown), 422,
                    {"workspace_ids": "|".join(sorted(known))})
    access_roles = _as_list(payload.get("access_roles")
                            or "employee,hr,team_lead,admin", "access_roles")
    bad_roles = [r for r in access_roles if r not in ROLES]
    if bad_roles:
        return _err("invalid_role", "Peran tidak dikenal: " + ", ".join(bad_roles),
                    422, {"access_roles": "|".join(ROLES)})
    title = payload["title"].strip()
    source_type = payload["source_type"].strip()
    content = payload["content"]
    h = _content_hash(content)
    version = str(payload.get("version") or "1")
    conn.execute(
        """INSERT INTO documents
           (id,title,source_type,effective_date,workspace_ids,access_roles,content,
            owner,confidentiality,status,version,supersedes,content_hash,local_edit)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
        (doc_id, title, source_type, payload.get("effective_date"),
         json.dumps(workspace_ids), json.dumps(access_roles), content,
         payload.get("owner"), confidentiality, status, version,
         payload.get("supersedes"), h))
    lifecycle.snapshot(conn, {"id": doc_id, "title": title, "content": content,
                              "status": status, "version": version,
                              "content_hash": h}, actor, note="versi awal")
    portal.audit_event(conn, actor, "doc_create",
                       f"Membuat dokumen: {title}", document_id=doc_id,
                       extra={"doc_id": doc_id, "status": status,
                              "confidentiality": confidentiality, "via": "chat"})
    conn.commit()
    return {"id": doc_id, "status": "created", "doc_status": status,
            "title": title, "version": version}, 201


def update_document(conn, actor, doc_id, payload):
    """Update parsial (merge). Field yang tidak dikirim dipertahankan.

    Logika ini dipindah apa adanya dari `app.py::update_document`: kalau
    payload tanpa `confidentiality` menulis "internal", mengedit dokumen
    restricted lewat chat akan menurunkan klasifikasinya tanpa peringatan.
    """
    denied = _require_role(actor, ("hr", "admin"),
                           "Hanya HR/admin yang dapat mengubah dokumen.")
    if denied:
        return denied
    bad = require_manageable(conn, actor, doc_id)
    if bad:
        return bad
    cur = find_doc_any(conn, doc_id)
    if cur is None:
        return _err("doc_not_found", "Dokumen tidak ditemukan.", 404)
    if cur.get("deleted_at"):
        return _err("doc_deleted",
                    "Dokumen ini ada di tempat sampah. Pulihkan dulu sebelum diubah.",
                    409)

    def pick(key, default=None):
        return payload[key] if key in payload else default

    title = pick("title", cur["title"])
    source_type = pick("source_type", cur["source_type"])
    effective_date = pick("effective_date", cur.get("effective_date"))
    try:
        workspace_ids = _as_list(pick("workspace_ids",
                                      cur.get("workspace_ids") or []),
                                 "workspace_ids")
        access_roles = _as_list(pick("access_roles",
                                     cur.get("access_roles") or []),
                                "access_roles")
    except ValueError as e:
        return _err("invalid_field_type", str(e), 422)
    content = pick("content", cur.get("content") or "")
    owner = pick("owner", cur.get("owner"))
    confidentiality = pick("confidentiality", cur.get("confidentiality") or "internal")
    status = pick("status", cur.get("status") or "indexed")
    supersedes = pick("supersedes", cur.get("supersedes"))

    bad = _validate_doc(title, source_type, content, confidentiality, status)
    if bad:
        return bad
    try:
        lifecycle.check_transition(cur.get("status") or "indexed", status)
    except LifecycleError as e:
        return _err(e.code, e.message, e.status, e.fields)

    content_changed = (cur.get("content") or "") != content
    new_hash = _content_hash(content)
    known = lifecycle.list_versions(conn, doc_id)
    cur_label = str(cur.get("version") or "1")
    if "version" in payload and payload.get("version"):
        version = str(payload["version"])
    elif content_changed:
        base = cur_label
        hist = lifecycle.latest_version(conn, doc_id)
        if hist and hist != base:
            base = max([base, hist], key=lambda v: [
                int(x) if x.isdigit() else 0 for x in str(v).split(".")] + [0])
        version = lifecycle.next_version(base)
    else:
        version = cur_label

    if content_changed and not known:
        lifecycle.snapshot(conn, dict(cur, version=cur_label), actor,
                           note="isi sebelum revisi")

    conn.execute(
        """UPDATE documents SET title=?, source_type=?, effective_date=?,
           workspace_ids=?, access_roles=?, content=?, owner=?, confidentiality=?,
           status=?, version=?, supersedes=?, content_hash=?, local_edit=1
           WHERE id=?""",
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
    if content_changed:
        lifecycle.snapshot(conn, {"id": doc_id, "title": title.strip(),
                                  "content": content, "status": status,
                                  "version": version, "content_hash": new_hash},
                           actor, note=payload.get("change_note") or "revisi")
    portal.audit_event(conn, actor, "doc_update",
                       f"Mengubah dokumen: {title.strip()}",
                       document_id=doc_id,
                       extra={"doc_id": doc_id, "changed_fields": changed,
                              "version": version,
                              "previous_version": cur.get("version") or "1",
                              "via": "chat"})
    conn.commit()
    return {"id": doc_id, "status": "updated", "changed_fields": changed,
            "doc_status": status, "version": version,
            "previous_version": cur.get("version") or "1"}, 200


def delete_document(conn, actor, doc_id):
    denied = _require_role(actor, ("hr", "admin"),
                           "Hanya HR/admin yang dapat menghapus dokumen.")
    if denied:
        return denied
    bad = require_manageable(conn, actor, doc_id)
    if bad:
        return bad
    doc = find_doc_any(conn, doc_id)
    if doc is None:
        return _err("doc_not_found", "Dokumen tidak ditemukan.", 404)
    if doc.get("deleted_at"):
        return {"id": doc_id, "status": "already_deleted",
                "deleted_at": doc["deleted_at"],
                "deleted_by": doc.get("deleted_by")}, 200
    data = lifecycle.soft_delete(conn, doc_id, actor)
    data["title"] = doc["title"]
    portal.audit_event(conn, actor, "doc_delete",
                       f"Menghapus dokumen: {doc['title']}",
                       document_id=doc_id,
                       extra={"doc_id": doc_id, "title": doc["title"],
                              "soft": True, "via": "chat"})
    conn.commit()
    return data, 200


def restore_document(conn, actor, doc_id):
    denied = _require_role(actor, ("hr", "admin"),
                           "Hanya HR/admin yang dapat memulihkan dokumen.")
    if denied:
        return denied
    bad = require_manageable(conn, actor, doc_id)
    if bad:
        return bad
    doc = find_doc_any(conn, doc_id)
    if doc is None:
        return _err("doc_not_found", "Dokumen tidak ditemukan.", 404)
    if not doc.get("deleted_at"):
        return {"id": doc_id, "status": "not_deleted"}, 200
    data = lifecycle.restore(conn, doc_id)
    data["title"] = doc["title"]
    portal.audit_event(conn, actor, "doc_restore",
                       f"Memulihkan dokumen: {doc['title']}",
                       document_id=doc_id,
                       extra={"doc_id": doc_id, "via": "chat"})
    conn.commit()
    return data, 200


# ---------------------------------------------------------------------------
# 5. Riwayat versi dokumen
# ---------------------------------------------------------------------------
def document_versions(conn, actor, doc_id):
    """Riwayat versi + versi yang berlaku (HR/admin, dokumen yang dikelola).

    Dipakai jalur chat dan rute HTTP supaya aturan "versi berlaku selalu ikut
    tampil walau baris riwayatnya belum ada" tidak ditulis dua kali.
    """
    denied = _require_role(actor, ("hr", "admin"),
                           "Hanya HR/admin yang dapat melihat riwayat versi dokumen.")
    if denied:
        return denied
    bad = require_manageable(conn, actor, doc_id)
    if bad:
        return bad
    items = lifecycle.list_versions(conn, doc_id, include_content=False)
    doc = find_doc_any(conn, doc_id) or {}
    current_version = str(doc.get("version") or "1")
    current = {
        "version": current_version,
        "title": doc.get("title"),
        "content_hash": doc.get("content_hash"),
        "status": doc.get("status"),
        "changed_at": None, "changed_by": None,
        "note": "versi berlaku", "current": True,
    }
    labels = {str(i["version"]) for i in items}
    if current_version not in labels:
        items = [current] + items
    else:
        for i in items:
            if str(i["version"]) == current_version:
                i["current"] = True
    return {"doc_id": doc_id, "title": doc.get("title"),
            "count": len(items), "current_version": current_version,
            "versions": items}, 200


# ---------------------------------------------------------------------------
# 6. Keputusan di luar rantai (override)
# ---------------------------------------------------------------------------
def override_request(conn, actor, request_id, action, note):
    """Delegate ke portal; override punya syarat catatan minimal 10 karakter."""
    try:
        rid = int(request_id)
    except (TypeError, ValueError):
        return _err("invalid_request_id", "`request_id` harus angka.", 422)
    return portal.override_request(conn, actor, rid, action, note)
