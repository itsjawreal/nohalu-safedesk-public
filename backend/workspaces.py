"""workspaces.py — CRUD workspace (Sprint 4, pelengkap user CRUD).

Workspace adalah unit izin: dokumen punya `workspace_ids`, karyawan punya
`allowed_workspaces`, dan sebuah dokumen hanya terbaca bila irisannya cocok
dengan role. Sebelum ini tabelnya hanya bisa dibaca (`GET /api/workspaces`),
jadi menambah unit organisasi baru berarti SQL langsung ke DB produksi —
padahal `users_admin` sudah menolak workspace yang tidak dikenal.

Aturan yang dijaga:

* `id` dibuat dari nama (slug) bila tidak dikirim, dan tidak bisa diubah
  setelah ada — mengganti id akan memutus semua referensi di dokumen/user.
* Nama wajib, unik (case-insensitive) supaya admin tidak membuat dua entri
  yang tampak sama di UI.
* Hapus ditolak (409) bila workspace masih dipakai dokumen atau karyawan.
  Menghapusnya akan membuat dokumen itu tak terjangkau siapa pun — kehilangan
  data diam-diam, bukan sekadar kerapian.
"""
import re

_SLUG_RE = re.compile(r"[^a-z0-9]+")


class WorkspaceError(Exception):
    def __init__(self, code, message, status=422, fields=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.fields = fields or {}


def _slug(name):
    base = _SLUG_RE.sub("_", (name or "").strip().lower()).strip("_")
    return "ws_" + (base or "baru")


def _clean_name(name):
    text = (name or "").strip()
    if not text:
        raise WorkspaceError("name_required", "Nama workspace wajib diisi.",
                             fields={"name": "wajib"})
    if len(text) > 80:
        raise WorkspaceError("name_too_long", "Nama maksimal 80 karakter.",
                             fields={"name": "maks 80"})
    return text


def _row(conn, ws_id):
    return conn.execute("SELECT id, name FROM workspaces WHERE id = ?",
                        (ws_id,)).fetchone()


def _unique_name(conn, name, exclude_id=None):
    rows = conn.execute("SELECT id, name FROM workspaces").fetchall()
    lowered = name.lower()
    for r in rows:
        if exclude_id and r["id"] == exclude_id:
            continue
        if (r["name"] or "").lower() == lowered:
            raise WorkspaceError("name_taken", "Nama workspace sudah dipakai.",
                                 status=409,
                                 fields={"name": r["id"]})


def list_all(conn):
    rows = conn.execute(
        "SELECT id, name FROM workspaces ORDER BY name COLLATE NOCASE").fetchall()
    return [{"id": r["id"], "name": r["name"]} for r in rows]


def create(conn, payload):
    name = _clean_name(payload.get("name"))
    _unique_name(conn, name)
    ws_id = (payload.get("id") or "").strip() or _slug(name)
    if not re.match(r"^[a-z0-9_]{3,40}$", ws_id):
        raise WorkspaceError(
            "invalid_id", "ID hanya huruf kecil, angka, dan garis bawah (3–40).",
            fields={"id": ws_id})
    if _row(conn, ws_id):
        raise WorkspaceError("id_taken", "ID workspace sudah dipakai.", status=409,
                             fields={"id": ws_id})
    conn.execute("INSERT INTO workspaces (id, name) VALUES (?,?)", (ws_id, name))
    conn.commit()
    return {"id": ws_id, "name": name, "status": "created"}


def rename(conn, ws_id, payload):
    row = _row(conn, ws_id)
    if not row:
        raise WorkspaceError("workspace_not_found", "Workspace tidak ditemukan.", 404)
    name = _clean_name(payload.get("name"))
    _unique_name(conn, name, exclude_id=ws_id)
    conn.execute("UPDATE workspaces SET name = ? WHERE id = ?", (name, ws_id))
    conn.commit()
    return {"id": ws_id, "name": name, "status": "renamed"}


def usage(conn, ws_id):
    """Berapa dokumen & karyawan yang memakai workspace ini."""
    docs = 0
    for r in conn.execute("SELECT workspace_ids FROM documents").fetchall():
        raw = r["workspace_ids"] or "[]"
        if '"' + ws_id + '"' in raw:
            docs += 1
    users = 0
    for r in conn.execute("SELECT allowed_workspaces FROM users").fetchall():
        raw = r["allowed_workspaces"] or "[]"
        if '"' + ws_id + '"' in raw:
            users += 1
    return {"documents": docs, "users": users}


def delete(conn, ws_id):
    if not _row(conn, ws_id):
        raise WorkspaceError("workspace_not_found", "Workspace tidak ditemukan.", 404)
    used = usage(conn, ws_id)
    if used["documents"] or used["users"]:
        raise WorkspaceError(
            "workspace_in_use",
            "Workspace masih dipakai; pindahkan dokumen/karyawan dulu.",
            status=409, fields={"documents": used["documents"],
                                "users": used["users"]})
    conn.execute("DELETE FROM workspaces WHERE id = ?", (ws_id,))
    conn.commit()
    return {"id": ws_id, "status": "deleted"}
