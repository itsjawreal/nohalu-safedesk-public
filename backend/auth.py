"""auth.py — sesi bertanda tangan (HMAC) + revokasi.

Kenapa bukan `Bearer <user_id>` lagi: token lama adalah **id user plaintext**,
jadi siapa pun yang menebak/melihat id (`u_rina` = admin) bisa masuk tanpa
login. Modul ini menggantinya dengan token yang:

  v1.<user_id>.<exp_unix>.<jti>.<sig>

`sig` = HMAC-SHA256(secret, "v1.<user_id>.<exp>.<jti>") dalam base64url tanpa
padding. `jti` (nonce acak) membuat dua login pada detik yang sama tetap
menghasilkan token berbeda — tanpa itu, hash-nya bentrok di tabel `sessions`.
Token hanya bisa dibuat server (butuh secret), punya masa berlaku, dan bisa
dicabut lebih awal lewat tabel `sessions`.

Secret diambil dari `NOHALU_SESSION_SECRET`. Bila tidak diset, secret dibuat
sekali lalu disimpan di samping DB (`<db>.session-secret`, mode 0600) supaya
sesi tidak mati setiap restart pada instalasi demo. Untuk produksi nyata,
set env-nya agar secret tidak ikut tersalin bersama file DB.
"""
import base64
import datetime as _dt
import hashlib
import hmac
import os
import secrets
import threading

TOKEN_PREFIX = "v1"
DEFAULT_TTL_HOURS = 12

_SECRET_LOCK = threading.Lock()
_SECRET_CACHE = {}


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def _secret_path(db_path: str) -> str:
    return (db_path or "safedesk.db") + ".session-secret"


def _load_or_create_secret(db_path: str) -> bytes:
    """Ambil secret dari env, atau file di samping DB (dibuat sekali, 0600)."""
    env = os.environ.get("NOHALU_SESSION_SECRET")
    if env:
        return env.encode("utf-8")

    key = os.path.abspath(db_path or "safedesk.db")
    with _SECRET_LOCK:
        cached = _SECRET_CACHE.get(key)
        if cached:
            return cached
        path = _secret_path(key)
        try:
            with open(path, "rb") as fh:
                data = fh.read().strip()
            if data:
                _SECRET_CACHE[key] = data
                return data
        except FileNotFoundError:
            pass
        data = secrets.token_bytes(32)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
        except OSError as exc:  # read-only FS: tetap jalan, sesi per proses
            # Modul ini sengaja tanpa dependensi: impor lokal supaya
            # `auth.py` tetap bisa dipakai mandiri (mis. skrip CLI).
            try:
                import logging_setup
                logging_setup.event(
                    logging_setup.get_logger("auth"), "secret_not_persisted",
                    "session secret tidak bisa ditulis; memakai secret memori",
                    level=logging_setup.logging.WARNING, error=str(exc))
            except Exception:
                pass
        _SECRET_CACHE[key] = data
        return data


def reset_secret_cache() -> None:
    """Dipakai tes yang mengganti DB/secret di tengah proses."""
    with _SECRET_LOCK:
        _SECRET_CACHE.clear()


def ttl_hours() -> float:
    try:
        val = float(os.environ.get("NOHALU_SESSION_TTL_HOURS", DEFAULT_TTL_HOURS))
    except ValueError:
        return DEFAULT_TTL_HOURS
    return val if val > 0 else DEFAULT_TTL_HOURS


def _sign(secret: bytes, payload: str) -> str:
    return _b64(hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).digest())


def issue_token(db_path: str, user_id: str, ttl_hours_: float | None = None) -> dict:
    """Buat token + metadata. Return {token, expires_at, ttl_seconds}."""
    ttl = ttl_hours_ if ttl_hours_ is not None else ttl_hours()
    exp = int((_dt.datetime.now(_dt.timezone.utc)
               + _dt.timedelta(hours=ttl)).timestamp())
    jti = secrets.token_hex(8)
    payload = f"{TOKEN_PREFIX}.{user_id}.{exp}.{jti}"
    secret = _load_or_create_secret(db_path)
    return {
        "token": f"{payload}.{_sign(secret, payload)}",
        "expires_at": _dt.datetime.fromtimestamp(
            exp, _dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "ttl_seconds": int(ttl * 3600),
    }


def token_hash(token: str) -> str:
    """Sidik jari token untuk tabel `sessions` (token asli tidak disimpan)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class TokenError(Exception):
    """Token tidak sah / kedaluwarsa. `code` dipakai sebagai detail API."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def parse_token(db_path: str, token: str) -> dict:
    """Verifikasi tanda tangan + masa berlaku. Return {user_id, exp, hash}.

    TIDAK menyentuh DB: pemanggil yang memeriksa revokasi lewat `sessions`.
    """
    parts = (token or "").split(".")
    if len(parts) != 5 or parts[0] != TOKEN_PREFIX:
        raise TokenError("invalid_token")
    _, user_id, exp_raw, jti, sig = parts
    if not user_id or not exp_raw or not jti:
        raise TokenError("invalid_token")
    try:
        exp = int(exp_raw)
    except ValueError:
        raise TokenError("invalid_token")
    payload = f"{TOKEN_PREFIX}.{user_id}.{exp_raw}.{jti}"
    expected = _sign(_load_or_create_secret(db_path), payload)
    if not hmac.compare_digest(expected, sig):
        raise TokenError("invalid_token")
    now = int(_dt.datetime.now(_dt.timezone.utc).timestamp())
    if exp <= now:
        raise TokenError("token_expired")
    return {"user_id": user_id, "exp": exp, "hash": token_hash(token)}
