"""files.py — penyimpanan berkas lampiran (slip gaji) di disk.

Kenapa bukan BLOB di SQLite: file PDF slip bisa ratusan KB per karyawan per
bulan, dan menyimpannya di baris yang sama membuat setiap query `payslips`
ikut membaca seluruh isi berkas. Yang disimpan di DB hanya *referensi* relatif
(`file_ref`), isinya di `<dir data>/files/`.

Keamanan:
  * nama berkas dibuat server (bukan dari nama unggahan) → tidak ada path
    traversal dari `../` atau nama dengan garis miring;
  * ekstensi dibatasi daftar putih;
  * ukuran dibatasi `MAX_FILE_BYTES` sebelum ditulis.
"""
import hashlib
import os
import uuid

ALLOWED_EXT = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".txt": "text/plain",
    ".csv": "text/csv",
}

MAX_FILE_BYTES = int(os.environ.get("NOHALU_MAX_FILE_BYTES",
                                    str(10 * 1024 * 1024)))


class FileError(Exception):
    def __init__(self, code, message, status=400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def store_root(db_path: str) -> str:
    """Direktori berkas, bersebelahan dengan DB (ikut pindah saat DB dipindah)."""
    root = os.path.join(os.path.dirname(os.path.abspath(db_path)) or ".", "files")
    os.makedirs(root, exist_ok=True)
    return root


def save(db_path: str, filename: str, data: bytes, kind: str = "doc") -> dict:
    """Tulis berkas dengan nama buatan server. Return metadata untuk DB."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXT:
        raise FileError("unsupported_file_type",
                        "Jenis berkas tidak didukung. Gunakan PDF, PNG, JPG, TXT, atau CSV.",
                        415)
    if not data:
        raise FileError("empty_file", "Berkas kosong.")
    if len(data) > MAX_FILE_BYTES:
        raise FileError("file_too_large",
                        f"Berkas melebihi batas {MAX_FILE_BYTES // (1024 * 1024)} MB.",
                        413)
    digest = hashlib.sha256(data).hexdigest()
    name = f"{kind}-{uuid.uuid4().hex[:12]}{ext}"
    folder = os.path.join(store_root(db_path), kind)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    # `xb` = gagal bila nama sudah ada; UUID membuat tabrakan praktis nol, dan
    # mode ini mencegah dua request menimpa berkas satu sama lain.
    with open(path, "xb") as fh:
        fh.write(data)
    return {
        "ref": os.path.join(kind, name),      # relatif terhadap store_root
        "name": os.path.basename(filename or name),
        "bytes": len(data),
        "sha256": digest,
        "content_type": ALLOWED_EXT[ext],
        "ext": ext,
    }


def resolve(db_path: str, ref: str) -> str | None:
    """Path absolut untuk `ref` relatif, atau None bila keluar dari store_root."""
    if not ref:
        return None
    root = os.path.abspath(store_root(db_path))
    path = os.path.abspath(os.path.join(root, ref))
    if not path.startswith(root + os.sep):
        return None
    return path if os.path.isfile(path) else None


def content_type(ref: str) -> str:
    return ALLOWED_EXT.get(os.path.splitext(ref or "")[1].lower(),
                           "application/octet-stream")


def delete(db_path: str, ref: str) -> bool:
    path = resolve(db_path, ref)
    if not path:
        return False
    try:
        os.remove(path)
        return True
    except OSError:
        return False
