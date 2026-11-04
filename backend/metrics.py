"""metrics.py — KPI runtime yang dulu "tidak terukur" (plan §11).

Plan §11 mendaftar lima metrik dengan kolom "Sekarang" berisi *tidak terukur*:
p95 latensi chat, error 5xx/hari, waktu approve rata-rata, rasio jawaban
grounded, dan refusal untuk data yang ada. Tanpa angka, tidak ada cara
membuktikan sebuah perubahan memperbaiki atau memperburuk.

Dua sumber, sengaja dipisah:

* **runtime** (in-memory) — latensi & status per request, diisi middleware.
  Hilang saat restart; itu wajar untuk observabilitas jendela pendek dan
  menghindari menulis satu baris DB untuk setiap request.
* **db** (turunan) — rasio grounded/refusal dari `audit_log`, waktu approve
  dari `requests`. Ini tahan restart karena datanya memang sudah tersimpan.

Jendela bergulir
----------------
Sampel runtime dibatasi dua lapis:

* **waktu** (`NOHALU_METRIC_WINDOW`, default 3600 s) — p95 selalu berarti
  "satu jam terakhir". Tanpa ini, satu lonjakan deploy pagi tetap menaikkan
  p95 sampai proses di-restart, dan angka itu tidak lagi menggambarkan
  kondisi sekarang.
* **jumlah** (`NOHALU_METRIC_SAMPLES`, default 5000) — pagar memori untuk
  trafik tinggi. Kalau jumlah yang memotong lebih dulu, `covered_seconds`
  di laporan memberi tahu bahwa jendela efektif lebih pendek dari yang
  diminta; angka tidak pernah diam-diam berubah arti.

Semua sampel disimpan di satu deque, bukan per-path: total dan rincian
per-path jadi selalu berjumlah sama. Tidak ada sampling — persentil dihitung
dari seluruh sampel yang masih di jendela.
"""
import datetime as _dt
import json
import os
import re
import threading
import time
from collections import deque

WINDOW_SECONDS = int(os.environ.get("NOHALU_METRIC_WINDOW", "3600"))
MAX_SAMPLES = int(os.environ.get("NOHALU_METRIC_SAMPLES", "5000"))

# Endpoint yang sengaja menahan koneksi terbuka (SSE) atau memindahkan berkas
# besar. Durasi request-nya mengukur lama koneksi hidup, bukan lama kerja
# server — memasukkannya ke p95 membuat satu tab browser terbuka tampak
# seperti "endpoint 5 detik" dan menenggelamkan regresi yang nyata.
_LONG_LIVED = (
    "/api/notifications/stream",
    "/api/chat/export",
)

_LOCK = threading.Lock()
# (ts_epoch, path, ms, status)
_SAMPLES = deque(maxlen=MAX_SAMPLES)
_SINCE = time.time()


def reset() -> None:
    """Kosongkan semua sampel (dipakai tes)."""
    global _SINCE
    with _LOCK:
        _SAMPLES.clear()
        _SINCE = time.time()


def record(path, ms, status) -> None:
    """Catat satu request. Aman dipanggil dari thread mana pun."""
    try:
        ms = int(ms)
        status = int(status)
    except (TypeError, ValueError):
        return
    p = str(path or "")
    if p.startswith(_LONG_LIVED):
        return
    with _LOCK:
        _SAMPLES.append((time.time(), p, ms, status))


def _prune(now) -> None:
    """Buang sampel di luar jendela waktu. Dipanggil dengan `_LOCK` dipegang."""
    cutoff = now - WINDOW_SECONDS
    while _SAMPLES and _SAMPLES[0][0] < cutoff:
        _SAMPLES.popleft()


def _p95(values):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


# Segmen yang jelas merupakan parameter, bukan nama endpoint. Tanpa ini
# `/api/documents/doc_17` dan `/api/documents/doc_42` menjadi dua baris
# terpisah, dan laporan p95 berubah jadi daftar id — bukan daftar endpoint.
_ID_SEGMENT = re.compile(
    r"^(?:[0-9]+"                    # 42
    r"|[0-9a-f]{8,}"                 # uuid tanpa tanda hubung / sha pendek
    r"|[A-Za-z]+_[0-9]+"             # doc_17, ws_12, u_9
    r"|[A-Za-z]+_[0-9a-f]{8,})$")    # u_1a2b3c4d5e



def _group(path):
    """Satukan path ber-parameter supaya `/api/chat/abc` dan `/api/chat/xyz`
    tidak menjadi dua baris terpisah di laporan."""
    parts = path.split("/")
    if len(parts) >= 4 and parts[1] == "api" and parts[2] == "chat" and len(parts) == 4:
        return "/api/chat/{session_id}"
    if len(parts) >= 4 and parts[1] == "api":
        grouped = [parts[0], parts[1]]
        for i, seg in enumerate(parts[2:], start=2):
            grouped.append("{id}" if _ID_SEGMENT.match(seg) else seg)
        return "/".join(grouped)
    return path


def snapshot() -> dict:
    """Ringkasan runtime: p95 & jumlah error per path dalam jendela."""
    now = time.time()
    with _LOCK:
        _prune(now)
        rows = list(_SAMPLES)
        started = _SINCE

    grouped = {}
    for _ts, path, ms, status in rows:
        key = _group(path)
        acc = grouped.setdefault(key, {"latencies": [], "errors": 0, "count": 0})
        acc["latencies"].append(ms)
        acc["count"] += 1
        if status >= 500:
            acc["errors"] += 1

    total = len(rows)
    errors = sum(1 for _t, _p, _m, s in rows if s >= 500)
    paths = {}
    for key, acc in grouped.items():
        paths[key] = {
            "count": acc["count"],
            "p95_ms": _p95(acc["latencies"]),
            "errors_5xx": acc["errors"],
        }

    # `covered_seconds` = rentang yang BENAR-BENAR terwakili sampel, bukan
    # uptime proses. Proses yang baru hidup 20 detik punya jendela efektif 20
    # detik — itu fakta, bukan anomali, dan tidak boleh dilaporkan sebagai
    # "batas sampel tercapai".
    covered = max(0, int(now - started))
    if rows:
        covered = min(covered, max(0, int(now - rows[0][0])))
    covered = min(covered, WINDOW_SECONDS)

    # Hanya true kalau deque penuh memotong data sebelum batas waktu habis:
    # artinya jendela efektif lebih pendek dari yang diminta KARENA VOLUME,
    # dan p95 tidak lagi mencakup satu jam penuh.
    sample_capped = len(rows) >= MAX_SAMPLES and WINDOW_SECONDS > covered

    chat = paths.get("/api/chat/{session_id}") or paths.get("/api/chat") or {}
    since = _dt.datetime.fromtimestamp(max(started, now - WINDOW_SECONDS),
                                      _dt.timezone.utc)
    return {
        "since": since.isoformat().replace("+00:00", "Z"),
        "window_seconds": WINDOW_SECONDS,
        "covered_seconds": covered,
        "sample_capped": sample_capped,
        "uptime_seconds": max(0, int(now - started)),
        "samples": total,
        "requests": total,
        "errors_5xx": errors,
        "error_rate": round(errors / total, 6) if total else 0.0,
        "chat_p95_ms": chat.get("p95_ms"),
        "paths": dict(sorted(paths.items(), key=lambda kv: -kv[1]["count"])[:20]),
    }


def _parse(ts):
    if not ts:
        return None
    try:
        return _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def db_snapshot(conn) -> dict:
    """KPI yang bisa diturunkan dari data tersimpan (tahan restart)."""
    rows = conn.execute(
        "SELECT answer, citations, refusal_reason FROM audit_log").fetchall()
    total = len(rows)
    grounded = 0
    refused = 0
    for r in rows:
        try:
            cites = json.loads(r["citations"] or "[]")
        except (TypeError, ValueError):
            cites = []
        if cites:
            grounded += 1
        if r["refusal_reason"]:
            refused += 1

    decided = conn.execute(
        "SELECT created_at, decided_at FROM requests "
        "WHERE decided_at IS NOT NULL AND created_at IS NOT NULL").fetchall()
    hours = []
    for r in decided:
        a, b = _parse(r["created_at"]), _parse(r["decided_at"])
        if a and b and b >= a:
            hours.append((b - a).total_seconds() / 3600.0)

    multi = conn.execute(
        """SELECT COUNT(*) AS n FROM (
             SELECT session_id FROM chat_messages WHERE role = 'user'
             GROUP BY session_id HAVING COUNT(*) > 1)""").fetchone()["n"]
    sessions = conn.execute(
        "SELECT COUNT(DISTINCT session_id) AS n FROM chat_messages").fetchone()["n"]

    return {
        "audit_rows": total,
        # Hitungan absolut disertakan di samping rasio: rasio bergerak ke arah
        # mana pun tergantung isi DB, sedangkan hitungan selalu bisa dipakai
        # untuk memverifikasi bahwa satu baris baru benar-benar terhitung.
        "grounded_rows": grounded,
        "refused_rows": refused,
        "grounded_rate": round(grounded / total, 4) if total else None,
        "refusal_rate": round(refused / total, 4) if total else None,
        "approve_avg_hours": round(sum(hours) / len(hours), 2) if hours else None,
        "approve_samples": len(hours),
        "multi_turn_rate": round(multi / sessions, 4) if sessions else None,
    }


def report(conn) -> dict:
    """Gabungan runtime + DB, dipakai `GET /api/admin/kpi`."""
    return {"runtime": snapshot(), "data": db_snapshot(conn)}
