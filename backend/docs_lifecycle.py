"""docs_lifecycle.py — siklus hidup dokumen: transisi status, versi, diff,
soft delete.

Dipisah dari app.py supaya aturannya satu tempat. Sebelum ini `PUT
/api/documents/{id}` menerima status apa pun (draft → deactivated → published
dalam satu lompatan) dan tidak ada riwayat isi: begitu konten ditimpa, versi
lama hilang tanpa jejak padahal `version` dan `supersedes` sudah ada di skema.

Kontrak:
  * `check_transition(old, new)` — raise LifecycleError kalau lompatan tidak sah.
  * `snapshot(conn, doc, actor, note)` — simpan isi dokumen sebagai satu versi.
  * `list_versions(conn, doc_id)` — metadata versi, terbaru dulu.
  * `diff_versions(conn, doc_id, a, b)` — unified diff + statistik baris.
  * `soft_delete(conn, doc_id, actor)` / `restore(conn, doc_id, actor)`.
"""
import difflib
import hashlib
import json
import re

# `review` ditambahkan supaya alur draft → review → published bisa dijalankan
# tanpa memaksa dokumen "hidup" (published) lebih dulu.
DOC_STATUSES = ("draft", "review", "published", "indexed", "superseded", "deactivated")

# Status yang boleh dipakai retrieval chat (lihat engine.ACTIVE_DOC_STATUS).
ACTIVE_STATUSES = ("indexed", "published")

# Transisi yang diizinkan. Aturan dibuat longgar untuk arah "mundur" (kembali
# ke draft untuk diperbaiki) tetapi menolak lompatan yang menghilangkan review:
# draft tidak boleh langsung superseded, dan dokumen yang sudah deactivated
# tidak boleh melompat ke superseded.
ALLOWED_TRANSITIONS = {
    "draft": {"draft", "review", "published", "deactivated"},
    "review": {"review", "draft", "published", "deactivated"},
    "published": {"published", "draft", "superseded", "deactivated"},
    "indexed": {"indexed", "draft", "review", "published", "superseded", "deactivated"},
    "superseded": {"superseded", "published", "draft", "deactivated"},
    "deactivated": {"deactivated", "draft", "published"},
}


class LifecycleError(Exception):
    """Pelanggaran aturan siklus hidup (dipetakan ke HTTP 409/422 oleh app)."""

    def __init__(self, code: str, message: str, status: int = 409, fields=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.fields = fields or {}


def check_transition(old: str, new: str) -> None:
    """Validasi satu perubahan status. Status sama selalu sah (no-op)."""
    old = (old or "indexed").strip()
    new = (new or "indexed").strip()
    if new not in DOC_STATUSES:
        raise LifecycleError("invalid_status",
                             "Status dokumen tidak dikenal: " + new, 422,
                             {"status": "|".join(DOC_STATUSES)})
    if old == new:
        return
    allowed = ALLOWED_TRANSITIONS.get(old)
    if allowed is None:
        raise LifecycleError("invalid_status",
                             "Status lama tidak dikenal: " + old, 422)
    if new not in allowed:
        raise LifecycleError(
            "invalid_status_transition",
            f"Tidak bisa pindah dari {old} ke {new}. "
            f"Langkah yang sah: {', '.join(sorted(allowed))}.",
            409, {"status": "|".join(sorted(allowed))})


def next_version(current: str) -> str:
    """Naikkan nomor versi: 1 → 2, 1.4 → 1.5, rilis-a → rilis-a.1."""
    cur = (current or "1").strip()
    if re.fullmatch(r"\d+", cur):
        return str(int(cur) + 1)
    m = re.fullmatch(r"(.*)\.(\d+)", cur)
    if m:
        return f"{m.group(1)}.{int(m.group(2)) + 1}"
    return cur + ".1"


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def snapshot(conn, doc: dict, actor: dict | None, note: str | None = None) -> str:
    """Simpan isi `doc` sebagai satu baris versi. Idempotent per (doc, versi).

    Versi yang sudah ada TIDAK ditimpa: kalau label versinya sama tetapi isinya
    berbeda, baris lama dipertahankan dan versi baru memakai nomor berikutnya.
    Tanpa aturan ini, memperbaiki typo pada dokumen yang versinya masih "1"
    akan menghapus satu-satunya salinan isi lama.
    """
    doc_id = doc["id"]
    version = str(doc.get("version") or "1")
    h = doc.get("content_hash") or content_hash(doc.get("content") or "")
    # Cari label yang benar-benar kosong: `version` bisa sudah dipakai isi
    # lain (mis. seed memperbarui konten tanpa menaikkan label), dan INSERT
    # dengan label itu akan menabrak UNIQUE(doc_id, version).
    while True:
        existing = conn.execute(
            "SELECT content_hash FROM document_versions WHERE doc_id=? AND version=?",
            (doc_id, version)).fetchone()
        if existing is None:
            break
        if existing["content_hash"] == h:
            return version
        version = next_version(version)
    conn.execute(
        """INSERT INTO document_versions
           (doc_id, version, title, content, content_hash, status, changed_by, changed_at, note)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (doc_id, version, doc.get("title"), doc.get("content"), h,
         doc.get("status") or "indexed",
         (actor or {}).get("id"),
         _now(), note))
    return version


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def list_versions(conn, doc_id: str, include_content: bool = False) -> list:
    cols = ("id,version,title,content_hash,status,changed_by,changed_at,note"
            + (",content" if include_content else ""))
    rows = conn.execute(
        f"SELECT {cols} FROM document_versions WHERE doc_id=? ORDER BY id DESC",
        (doc_id,)).fetchall()
    out = []
    for r in rows:
        item = {"id": r["id"], "version": r["version"], "title": r["title"],
                "content_hash": r["content_hash"], "status": r["status"],
                "changed_by": r["changed_by"], "changed_at": r["changed_at"],
                "note": r["note"]}
        if include_content:
            item["content"] = r["content"]
        out.append(item)
    return out


def latest_version(conn, doc_id: str) -> str:
    """Label versi tertinggi yang tercatat untuk dokumen ini (default "1")."""
    rows = conn.execute(
        "SELECT version FROM document_versions WHERE doc_id=?", (doc_id,)).fetchall()
    best = "1"
    best_key = (0,)
    for r in rows:
        v = str(r["version"] or "1")
        m = re.fullmatch(r"(\d+)", v)
        key = (int(m.group(1)),) if m else (0,)
        if key > best_key or best_key == (0,):
            best, best_key = v, key
    return best


def _version_row(conn, doc_id: str, version: str):
    return conn.execute(
        "SELECT * FROM document_versions WHERE doc_id=? AND version=?",
        (doc_id, version)).fetchone()


def diff_versions(conn, doc_id: str, a: str, b: str) -> dict:
    """Unified diff antara dua versi (a = lama, b = baru)."""
    ra = _version_row(conn, doc_id, str(a))
    rb = _version_row(conn, doc_id, str(b))
    if ra is None or rb is None:
        raise LifecycleError("version_not_found",
                             f"Versi {a if ra is None else b} tidak ada untuk dokumen ini.",
                             404)
    old = (ra["content"] or "").splitlines()
    new = (rb["content"] or "").splitlines()
    lines = list(difflib.unified_diff(
        old, new, fromfile=f"v{a}", tofile=f"v{b}", lineterm="", n=3))
    added = sum(1 for x in lines if x.startswith("+") and not x.startswith("+++"))
    removed = sum(1 for x in lines if x.startswith("-") and not x.startswith("---"))
    return {
        "doc_id": doc_id,
        "from": {"version": ra["version"], "title": ra["title"],
                 "status": ra["status"], "changed_at": ra["changed_at"],
                 "changed_by": ra["changed_by"], "chars": len(ra["content"] or "")},
        "to": {"version": rb["version"], "title": rb["title"],
               "status": rb["status"], "changed_at": rb["changed_at"],
               "changed_by": rb["changed_by"], "chars": len(rb["content"] or "")},
        "same": (ra["content_hash"] == rb["content_hash"]),
        "added_lines": added,
        "removed_lines": removed,
        "diff": "\n".join(lines),
        "truncated": len(lines) > 4000,
    }


def soft_delete(conn, doc_id: str, actor: dict) -> dict:
    conn.execute(
        "UPDATE documents SET deleted_at=?, deleted_by=?, local_edit=1 WHERE id=?",
        (_now(), actor["id"], doc_id))
    return {"id": doc_id, "status": "deleted", "deleted_at": _now(),
            "deleted_by": actor["id"]}


def restore(conn, doc_id: str) -> dict:
    conn.execute(
        "UPDATE documents SET deleted_at=NULL, deleted_by=NULL, local_edit=1 WHERE id=?",
        (doc_id,))
    return {"id": doc_id, "status": "restored"}


def diff_payload(diff: dict) -> str:
    """Ringkas diff untuk kolom `details` audit (tanpa seluruh isi dokumen)."""
    return json.dumps({
        "doc_id": diff["doc_id"], "from": diff["from"]["version"],
        "to": diff["to"]["version"], "added": diff["added_lines"],
        "removed": diff["removed_lines"], "same": diff["same"],
    }, ensure_ascii=False)
