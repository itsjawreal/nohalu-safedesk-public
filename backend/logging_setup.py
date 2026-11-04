"""logging_setup.py — log terstruktur (JSON per baris) + pemeriksaan integritas DB.

Kenapa JSON: di produksi log dibaca mesin (grep `"level":"error"`, filter
`"event":"request"`, hitung `"status":500`). Format bebas seperti
`print(f"... {exc}")` tidak bisa diagregasi, dan satu traceback multi-baris
memecah satu peristiwa menjadi banyak baris.

Kontrak tiap baris:
    {"ts": "...Z", "level": "info", "logger": "safedesk.api",
     "event": "request", "msg": "...", ...field tambahan}

`event` selalu ada supaya filter tidak bergantung pada teks pesan.
"""
import json
import logging
import os
import sys
import time

_CONFIGURED = False


class _DynamicStream(logging.StreamHandler):
    """StreamHandler yang selalu menulis ke `sys.stdout` TERKINI.

    `StreamHandler` biasa menyimpan objek stream saat dibuat. Di bawah pytest
    (capsys) atau saat proses mengganti `sys.stdout` (reload, worker), objek
    itu ditutup sementara logger masih memegangnya — hasilnya
    "Logging error ... I/O operation on closed file" di setiap baris, dan
    pesan asli hilang dari output. Resolusi per-emit membuat log tidak pernah
    menulis ke stream mati.
    """

    def __init__(self, stream=None):
        super().__init__(stream or sys.stdout)
        self._fixed = stream  # None = ikuti sys.stdout

    @property
    def stream(self):
        return self._fixed if self._fixed is not None else sys.stdout

    @stream.setter
    def stream(self, value):
        self._fixed = value

    def emit(self, record):
        # Saat interpreter shutdown, `sys.stdout` bisa sudah ditutup/None.
        # Tanpa penjagaan ini, setiap baris log terakhir memunculkan
        # "--- Logging error ---" ke stderr dan menutupi pesan aslinya.
        stream = self.stream
        if stream is None or getattr(stream, "closed", False):
            return
        try:
            super().emit(record)
        except (ValueError, OSError):
            return


class JsonFormatter(logging.Formatter):
    """Formatter satu baris JSON. Exception tetap ikut sebagai field `exc`."""

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            # `event` diisi lewat `extra={"event": ...}`; default = nama level
            # supaya baris tanpa event tetap punya kunci yang sama.
            "event": getattr(record, "event", "log"),
            "msg": record.getMessage(),
        }
        for key, value in getattr(record, "fields", {}).items():
            if key not in out:
                out[key] = value
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        try:
            return json.dumps(out, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            out["msg"] = str(out.get("msg"))
            return json.dumps(out, ensure_ascii=False)


def setup(level: str | None = None, stream=None) -> logging.Logger:
    """Pasang handler JSON sekali saja (idempotent, aman dipanggil ulang)."""
    global _CONFIGURED
    root = logging.getLogger("safedesk")
    if _CONFIGURED:
        return root
    lvl = (level or os.environ.get("NOHALU_LOG_LEVEL", "info")).upper()
    handler = _DynamicStream(stream)
    handler.setFormatter(JsonFormatter())
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, lvl, logging.INFO))
    root.propagate = False
    _CONFIGURED = True
    return root


def get_logger(name: str = "api") -> logging.Logger:
    setup()
    return logging.getLogger("safedesk." + name)


def event(logger: logging.Logger, name: str, msg: str, level: int = logging.INFO, **fields):
    """Satu peristiwa terstruktur. Field tambahan masuk ke baris JSON."""
    logger.log(level, msg, extra={"event": name, "fields": fields})


def integrity_check(conn) -> dict:
    """`PRAGMA integrity_check` + `foreign_key_check` saat start.

    Hasilnya dilaporkan sebagai log, bukan exception: DB yang rusak sebagian
    masih bisa melayani baca, dan mematikan proses hanya membuat pengguna
    kehilangan akses tanpa memberi waktu menyelamatkan data.
    """
    out = {"ok": True, "detail": "ok", "foreign_key_violations": 0}
    try:
        row = conn.execute("PRAGMA integrity_check;").fetchone()
        detail = (row[0] if row else "no result")
        out["detail"] = detail
        out["ok"] = (detail == "ok")
    except Exception as exc:  # DB tidak bisa dibaca sama sekali
        out.update(ok=False, detail=f"integrity_check failed: {exc}")
        return out
    try:
        rows = conn.execute("PRAGMA foreign_key_check;").fetchall()
        out["foreign_key_violations"] = len(rows)
        if rows:
            out["ok"] = False
            out["fk_sample"] = [tuple(r) for r in rows[:5]]
    except Exception as exc:
        out["fk_error"] = str(exc)
    return out
