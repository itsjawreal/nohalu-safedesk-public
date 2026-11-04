"""backup.py — cadangan SQLite harian + retensi.

Kenapa `sqlite3.Connection.backup()` dan bukan `cp`: menyalin berkas DB yang
sedang ditulis (mode WAL) bisa menghasilkan salinan yang tidak konsisten atau
kehilangan isi WAL yang belum di-checkpoint. API backup SQLite membuat snapshot
transaksional yang aman walau ada penulis aktif.

Nama berkas memuat tanggal (`safedesk-YYYYMMDD-HHMM.db`) sehingga:
  * backup yang sama tidak menimpa dirinya sendiri dalam satu hari;
  * retensi cukup mengurutkan nama, tanpa membaca isi berkas.

Retensi menyimpan `KEEP` berkas terbaru. Nilainya dari env supaya operator bisa
memperpanjang tanpa mengubah kode.
"""
import os
import re
import sqlite3
import time

DEFAULT_KEEP = int(os.environ.get("NOHALU_BACKUP_KEEP", "7"))
NAME_RE = re.compile(r"^safedesk-\d{8}-\d{4}(-\d+)?\.db$")


def backup_dir(db_path: str) -> str:
    """Folder cadangan di samping DB (`<dir DB>/backups`)."""
    base = os.path.dirname(os.path.abspath(db_path)) or "."
    return os.environ.get("NOHALU_BACKUP_DIR") or os.path.join(base, "backups")


def list_backups(db_path: str) -> list:
    """Daftar berkas cadangan, terbaru dulu (urut nama = urut waktu)."""
    folder = backup_dir(db_path)
    try:
        names = [n for n in os.listdir(folder) if NAME_RE.match(n)]
    except FileNotFoundError:
        return []
    names.sort(reverse=True)
    out = []
    for n in names:
        full = os.path.join(folder, n)
        try:
            st = os.stat(full)
        except OSError:
            continue
        out.append({"name": n, "bytes": st.st_size, "mtime": int(st.st_mtime)})
    return out


def prune(db_path: str, keep: int = None) -> list:
    """Hapus cadangan lama, sisakan `keep` terbaru. Return nama yang dihapus."""
    keep = DEFAULT_KEEP if keep is None else max(0, int(keep))
    removed = []
    for item in list_backups(db_path)[keep:]:
        try:
            os.remove(os.path.join(backup_dir(db_path), item["name"]))
            removed.append(item["name"])
        except OSError:
            pass
    return removed


def run(db_path: str, keep: int = None, stamp: str = None) -> dict:
    """Buat satu cadangan + terapkan retensi.

    `stamp` (opsional, "YYYYMMDD-HHMM") dipakai tes agar nama deterministik.
    """
    folder = backup_dir(db_path)
    os.makedirs(folder, exist_ok=True)
    stamp = stamp or time.strftime("%Y%m%d-%H%M")
    name = f"safedesk-{stamp}.db"
    target = os.path.join(folder, name)
    if os.path.exists(target):
        # Dua pemanggilan dalam menit yang sama: tetap buat berkas baru dengan
        # akhiran agar cadangan sebelumnya tidak tertimpa.
        n = 2
        while os.path.exists(target):
            target = os.path.join(folder, f"safedesk-{stamp}-{n}.db")
            name = f"safedesk-{stamp}-{n}.db"
            n += 1
    src = sqlite3.connect(db_path, timeout=10)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)  # snapshot transaksional
    finally:
        dst.close()
        src.close()
    size = os.path.getsize(target)
    removed = prune(db_path, keep)
    return {"name": name, "bytes": size, "path": target,
            "removed": removed, "kept": len(list_backups(db_path))}


# ---------------------------------------------------------------------------
# Penjadwal harian
# ---------------------------------------------------------------------------
# Kenapa thread sendiri, bukan cron: di mesin ini tidak ada crontab sama sekali
# (`crontab -l` → no crontab), dan menambah dependensi sistem membuat
# deployment bergantung pada hal di luar repo. Satu thread daemon dengan
# interval 24 jam cukup untuk satu instance.
_STARTED = False
_LOCK = None


def start_daily(db_path: str, interval_seconds: int = None, first_delay: int = 300) -> bool:
    """Jalankan cadangan tiap `interval_seconds` (default 24 jam). Idempotent."""
    global _STARTED, _LOCK
    import threading

    if os.environ.get("NOHALU_BACKUP_ENABLED", "1") == "0":
        return False
    if _LOCK is None:
        _LOCK = threading.Lock()
    with _LOCK:
        if _STARTED:
            return False
        _STARTED = True
    interval = int(interval_seconds or os.environ.get("NOHALU_BACKUP_INTERVAL", 86400))

    def _loop():
        threading.Event().wait(max(1, first_delay))
        while True:
            try:
                run(db_path)
            except Exception as exc:  # disk penuh, izin, dst.
                try:
                    import logging_setup
                    logging_setup.event(
                        logging_setup.get_logger("backup"), "backup_failed",
                        str(exc), level=logging_setup.logging.ERROR)
                except Exception:
                    pass
            threading.Event().wait(max(60, interval))

    threading.Thread(target=_loop, name="safedesk-backup", daemon=True).start()
    return True


def is_running() -> bool:
    import threading
    return any(t.name == "safedesk-backup" and t.is_alive()
               for t in threading.enumerate())
