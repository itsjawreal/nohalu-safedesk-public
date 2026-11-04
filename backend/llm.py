"""Optional generative LLM layer — Inception Labs mercury-3 (reasoning high).

The deterministic engine remains the source of truth for permissions, retrieval,
conflict, refusal, PII and injection. This module only *phrases* an answer from an
already-verified draft + allowed context. It is never called for refusals.

Config (env or backend/.env):
  NOHALU_LLM_API_KEY   required to enable
  NOHALU_LLM_BASE_URL  default https://api.inceptionlabs.ai/v1/chat/completions
  NOHALU_LLM_MODEL     default mercury-3
  NOHALU_LLM_REASONING default high
  NOHALU_LLM_ENABLED   "0" to force off, "1" to force on (default: on if key set)
  NOHALU_LLM_TIMEOUT   seconds, default 90
"""
import os
import json
import re
import time
import threading
import urllib.request
import urllib.error


def _log(event, msg, **fields):
    """Log terstruktur tanpa pernah menulis API key.

    Modul ini dulu hanya menulis lewat `sys.stderr` di pemanggil, sehingga
    kegagalan provider tidak bisa diagregasi (`grep '"event":"llm_error"'`)
    dan tidak memuat durasi. Import dilakukan di dalam fungsi supaya modul
    tetap bisa dipakai tanpa `logging_setup` (mis. skrip uji cepat).
    """
    try:
        import logging_setup
        import logging
        logging_setup.event(logging_setup.get_logger("llm"), event, msg,
                            level=logging.WARNING, **fields)
    except Exception:  # pragma: no cover — logging tidak boleh mematikan jawaban
        pass


def _load_dotenv():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()

BASE_URL = os.environ.get("NOHALU_LLM_BASE_URL", "https://api.inceptionlabs.ai/v1/chat/completions")
MODEL = os.environ.get("NOHALU_LLM_MODEL", "mercury-3")
REASONING = os.environ.get("NOHALU_LLM_REASONING", "high")
TIMEOUT = int(os.environ.get("NOHALU_LLM_TIMEOUT", "90"))

# ---- Circuit breaker ----
# Provider yang sedang down tidak boleh menahan SETIAP request selama seluruh
# anggaran retry. Tanpa pemutus, 5 pengguna yang bertanya bersamaan saat
# provider mati menumpuk 5 x 120 detik, dan tiap pengguna baru mengulanginya.
# Setelah `THRESHOLD` kegagalan berurutan, panggilan berikutnya langsung gagal
# (fallback deterministik) sampai `COOLDOWN` detik berlalu, lalu dicoba lagi.
_CB_LOCK = threading.Lock()
_CB_STATE = {"failures": 0, "open_until": 0.0}

# ---- Akuntansi pemakaian ----
# Sebelum ini tidak ada satu angka pun tentang berapa token yang benar-benar
# dipakai: `llm_ms` hanya mengukur waktu, sehingga lonjakan biaya/konteks tidak
# terlihat sampai tagihan datang. Penghitung ini in-memory (hilang saat restart,
# sama seperti `metrics`), cukup untuk menjawab "payload kita membesar atau
# tidak" dari `/api/llm/status`.
_USAGE_LOCK = threading.Lock()
_USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
          "max_prompt_tokens": 0, "max_prompt_chars": 0, "truncated": 0}


def _record_usage(body: dict, prompt_chars: int) -> None:
    usage = (body or {}).get("usage") or {}
    try:
        pt = int(usage.get("prompt_tokens") or 0)
    except (TypeError, ValueError):
        pt = 0
    try:
        ct = int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        ct = 0
    truncated = ((body or {}).get("choices") or [{}])[0].get("finish_reason") == "length"
    with _USAGE_LOCK:
        _USAGE["calls"] += 1
        _USAGE["prompt_tokens"] += pt
        _USAGE["completion_tokens"] += ct
        _USAGE["max_prompt_tokens"] = max(_USAGE["max_prompt_tokens"], pt)
        _USAGE["max_prompt_chars"] = max(_USAGE["max_prompt_chars"], int(prompt_chars or 0))
        if truncated:
            _USAGE["truncated"] += 1


def usage_stats() -> dict:
    """Penghitung pemakaian sejak proses hidup (dipakai `/api/llm/status`)."""
    with _USAGE_LOCK:
        return dict(_USAGE)


def reset_usage() -> None:
    with _USAGE_LOCK:
        for k in _USAGE:
            _USAGE[k] = 0


def _cb_threshold() -> int:
    # Dibaca per-panggilan (bukan konstanta import) supaya bisa diubah saat
    # runtime dan diuji tanpa memuat ulang modul.
    return int(os.environ.get("NOHALU_LLM_CB_THRESHOLD", "4"))


def _cb_cooldown() -> float:
    return float(os.environ.get("NOHALU_LLM_CB_COOLDOWN", "45"))


def _cb_is_open() -> bool:
    with _CB_LOCK:
        return _CB_STATE["open_until"] > time.time()


def _cb_record(ok: bool) -> None:
    with _CB_LOCK:
        if ok:
            _CB_STATE["failures"] = 0
            _CB_STATE["open_until"] = 0.0
            return
        _CB_STATE["failures"] += 1
        if _CB_STATE["failures"] >= _cb_threshold():
            _CB_STATE["open_until"] = time.time() + _cb_cooldown()


def reset_circuit() -> None:
    """Buka kembali pemutus (dipakai tes dan setelah konfigurasi diperbaiki)."""
    with _CB_LOCK:
        _CB_STATE["failures"] = 0
        _CB_STATE["open_until"] = 0.0


def api_key() -> str:
    return os.environ.get("NOHALU_LLM_API_KEY", "").strip()


def enabled() -> bool:
    flag = os.environ.get("NOHALU_LLM_ENABLED")
    if flag is not None:
        return flag.strip().lower() not in ("0", "false", "no", "")
    return bool(api_key())


def status(detail: bool = True) -> dict:
    """Ringkasan konfigurasi. `detail=False` untuk respons HTTP publik.

    `/api/llm/status` sengaja tanpa auth (UI memakainya untuk badge
    "mode ekstraktif"), tetapi URL provider internal bukan informasi yang perlu
    dibagikan ke siapa pun yang membuka endpoint itu.
    """
    out = {
        "enabled": enabled(),
        "model": MODEL,
        "reasoning": REASONING,
        "has_key": bool(api_key()),
        # Pemutus sirkuit terbuka = jawaban sedang memakai jalur deterministik
        # walau `enabled` masih true. Tanpa field ini, badge UI menampilkan
        # "LLM aktif" saat provider justru sedang diparkir, dan operator
        # menyimpulkan "LLM tidak dipakai" padahal baru saja gagal.
        "circuit_open": _cb_is_open(),
    }
    if detail:
        out["base_url"] = BASE_URL
        # Pemakaian nyata (token dari provider + ukuran payload kita). Tanpa
        # angka ini, "payload membesar" hanya bisa dirasakan sebagai keluhan
        # konteks jebol, bukan diukur.
        out["usage"] = usage_stats()
    return out


# Token kontrol / penanda protokol internal model yang TIDAK boleh sampai ke
# user. Terukur di :8890: satu jawaban diawali
# "<|inc_tool_call_ml|_c_calls> <|inc_tool_call_ml|invoke name=\"...\">" —
# model gagal memformat tool-call dan membocorkan penanda mentahnya. Teks
# seperti ini bukan jawaban; membiarkannya membuat aplikasi terasa rusak.
_CONTROL_TOKEN_RE = re.compile(
    # Token kontrol bergaya "<|nama|>" maupun "<|nama|_varian>".
    r"<\|[^<>]{0,120}>"
    # Fragmen markup tool-call yang kehilangan pembungkusnya.
    r"|</?(?:tool_call|function_call|invoke|parameter)[^<>]{0,80}>")


def _clean_text(text: str) -> str:
    """Buang token kontrol/penanda protokol yang bocor ke isi jawaban."""
    if not text:
        return text
    cleaned = _CONTROL_TOKEN_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _post(payload: dict, timeout: int | None = None) -> tuple[dict, int]:
    """POST ke provider. Return (body, elapsed_ms). Raise RuntimeError bila gagal.

    Retry HANYA untuk kegagalan yang memang layak diulang: 429 (rate limit),
    5xx (gangguan provider), dan timeout/putus koneksi. 4xx lain tidak diulang:
    400 karena prompt ditolak content_filter atau 401 karena key salah tidak
    akan berubah hasilnya, dan mengulangnya hanya menambah latensi user.

    Sebelum ini tidak ada retry sama sekali: satu 429 sesaat langsung menjatuhkan
    seluruh jawaban ke fallback deterministik, padahal percobaan kedua biasanya
    berhasil. Jumlah percobaan dibatasi (default 3) supaya provider yang mati
    total tidak menahan satu pertanyaan selama menit-menitan.
    """
    key = api_key()
    if not key:
        raise RuntimeError("NOHALU_LLM_API_KEY not set")
    if _cb_is_open():
        # Kegagalan cepat, bukan menunggu timeout lagi. Kode error sengaja
        # dibedakan supaya dashboard bisa membedakan "provider sedang dimatikan
        # sementara oleh pemutus" dari "satu request kebetulan gagal".
        _log("llm_circuit_open", "LLM circuit breaker open")
        raise RuntimeError("LLM circuit open: provider recently failing")
    data = json.dumps(payload).encode("utf-8")
    prompt_chars = len(data)
    attempts = max(1, int(os.environ.get("NOHALU_LLM_ATTEMPTS", "3")))
    backoff = float(os.environ.get("NOHALU_LLM_BACKOFF", "0.6"))
    # Anggaran total satu panggilan (termasuk semua percobaan). Tanpa ini,
    # 3 percobaan x timeout 90 dtk = 270 dtk per step agent, dan satu
    # pertanyaan dengan MAX_STEPS=6 bisa menahan lebih dari 25 menit.
    budget = float(os.environ.get("NOHALU_LLM_BUDGET", "120"))
    t0 = time.time()
    last_error = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(
            BASE_URL, data=data, method="POST",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
        )
        try:
            remaining = budget - (time.time() - t0)
            if remaining <= 1:
                raise RuntimeError(last_error or "LLM time budget exhausted")
            with urllib.request.urlopen(
                    req, timeout=min(timeout or TIMEOUT, remaining)) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            _cb_record(True)
            _record_usage(body, prompt_chars)
            return body, int((time.time() - t0) * 1000)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:300]
            retryable = e.code == 429 or 500 <= e.code <= 599
            # Provider yang mengirim `Retry-After` sudah memberi tahu kapan
            # boleh mencoba lagi. Mengabaikannya dan menunggu backoff sendiri
            # berarti menabrak rate limit persis saat masih ditutup — pola yang
            # membuat 429 berubah menjadi blokir lebih panjang.
            retry_after = None
            try:
                raw_after = e.headers.get("Retry-After") if e.headers else None
                if raw_after is not None:
                    retry_after = max(0.0, float(raw_after))
            except (TypeError, ValueError, AttributeError):
                retry_after = None
            _log("llm_http_error", f"LLM HTTP {e.code}",
                 status=e.code, attempt=attempt, retryable=retryable,
                 retry_after=retry_after,
                 elapsed_ms=int((time.time() - t0) * 1000))
            if not retryable or attempt >= attempts:
                # 4xx NON-retryable BUKAN bukti provider sedang down: 400
                # content_filter / 401 key salah / 413 payload terlalu besar
                # berasal dari satu request tertentu, bukan dari kesehatan
                # provider. Dulu semuanya menambah hitungan breaker, sehingga
                # empat prompt user yang kebetulan ditolak content_filter
                # memarkir LLM untuk SEMUA user selama cooldown. Terbukti:
                # 4x HTTP 400 berurutan -> `_cb_is_open()` True.
                if retryable:
                    _cb_record(False)
                raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e
            last_error = f"LLM HTTP {e.code}: {detail}"
            if retry_after is not None:
                # Tetap dibatasi anggaran total: menunggu 60 dtk saat sisa
                # anggaran 10 dtk hanya menunda kegagalan yang sama.
                remaining = budget - (time.time() - t0)
                time.sleep(min(retry_after, max(0.0, remaining - 1)))
                continue
        except urllib.error.URLError as e:
            _log("llm_network_error", f"LLM network error: {e.reason}",
                 attempt=attempt, elapsed_ms=int((time.time() - t0) * 1000))
            if attempt >= attempts:
                _cb_record(False)
                raise RuntimeError(f"LLM network error: {e.reason}") from e
            last_error = f"LLM network error: {e.reason}"
        except (TimeoutError, OSError) as e:
            _log("llm_timeout", f"LLM network error: {e}",
                 attempt=attempt, elapsed_ms=int((time.time() - t0) * 1000))
            if attempt >= attempts:
                _cb_record(False)
                raise RuntimeError(f"LLM network error: {e}") from e
            last_error = f"LLM network error: {e}"
        # Jeda linear + sedikit acak agar banyak request tidak menghantam
        # provider serentak pada detik yang sama.
        time.sleep(min(backoff * attempt, 5.0) + (attempt - 1) * 0.15)
    raise RuntimeError(last_error or "LLM request failed")


def generate(system: str, user: str, max_tokens: int | None = None,
             reasoning: str | None = None) -> dict:
    """Call mercury-3 tanpa tool. Returns {text, ms, usage, model}. Raises on failure.

    NOTE: with reasoning_effort=high the model may spend >1000 tokens on hidden
    reasoning. Sending a low max_tokens can therefore yield an empty completion.
    By default we omit max_tokens and let the API decide.
    """
    payload = {
        "model": MODEL,
        "reasoning_effort": reasoning or REASONING,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    body, ms = _post(payload)
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM empty choices: {str(body)[:200]}")
    text = (choices[0].get("message") or {}).get("content") or ""
    return {
        "text": _clean_text(text),
        # `length` = jawaban berhenti di tengah. Pemanggil (rephrase/open) harus
        # bisa menolak mengirim kalimat yang tidak selesai sebagai fakta.
        "finish_reason": choices[0].get("finish_reason"),
        "truncated": choices[0].get("finish_reason") == "length",
        "ms": ms,
        "usage": body.get("usage", {}),
        "model": body.get("model", MODEL),
    }


def chat(messages: list, tools: list | None = None, max_tokens: int | None = None,
         reasoning: str | None = None, tool_choice: str = "auto") -> dict:
    """Satu langkah percakapan dengan dukungan native tool-calling.

    Berbeda dari `generate()`, fungsi ini mengembalikan pesan asisten UTUH
    (termasuk `tool_calls`) supaya pemanggil bisa menjalankan loop agent.
    `reasoning_details` yang dikembalikan provider tidak perlu di-echo balik —
    sudah diuji: percakapan tool-call tetap konsisten tanpanya.

    Return:
      {text, tool_calls:[{id,name,arguments,raw_arguments,arguments_error}],
       finish_reason, truncated, ms, usage, model}
    """
    payload = {
        "model": MODEL,
        "reasoning_effort": reasoning or REASONING,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    body, ms = _post(payload)
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM empty choices: {str(body)[:200]}")
    choice = choices[0]
    msg = choice.get("message") or {}
    calls = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments")
        # Argumen yang tidak bisa di-parse TIDAK boleh menjadi `{}`.
        #
        # Bug terukur: provider kadang mengirim JSON terpotong
        # ('{"category": "promo') — lalu `args` menjadi objek kosong dan tool
        # tetap dijalankan. Untuk tool mutasi yang semua parameternya punya
        # default (clock_attendance, mark_notifications_read, dst.) hasilnya
        # adalah AKSI NYATA dengan parameter yang tidak pernah diminta model.
        # `arguments_error` membuat pemanggil bisa menolak panggilan itu, bukan
        # menebak isinya.
        arguments_error = None
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) and raw_args.strip() else (raw_args or {})
        except json.JSONDecodeError:
            args = {}
            arguments_error = "invalid_json"
        if not isinstance(args, dict):
            args = {}
            arguments_error = arguments_error or "not_an_object"
        calls.append({
            "id": tc.get("id") or f"call_{len(calls)}",
            "name": fn.get("name") or "",
            "arguments": args,
            "raw_arguments": raw_args if isinstance(raw_args, str) else json.dumps(args),
            "arguments_error": arguments_error,
        })
    return {
        "text": _clean_text(msg.get("content") or ""),
        "tool_calls": calls,
        "finish_reason": choice.get("finish_reason"),
        # `length` = model kehabisan token. Teksnya berhenti di tengah kalimat
        # ("Cuti tahunan Anda adalah 12 hari dan dapat"). Dulu teks itu dikirim
        # apa adanya sebagai jawaban final. Penanda ini membuat pemanggil bisa
        # meminta lanjutan / menandai jawaban sebagai tidak lengkap.
        "truncated": choice.get("finish_reason") == "length",
        "ms": ms,
        "usage": body.get("usage", {}),
        "model": body.get("model", MODEL),
    }
