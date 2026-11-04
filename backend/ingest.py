"""Document ingestion: extract text from PDF, DOCX, CSV, Markdown, TXT.

No network. Returns plain text + a source_type label. Raises IngestError with a
user-safe message on unsupported/corrupt files.
"""
import io
import os
import csv

MAX_BYTES = 10 * 1024 * 1024          # 10 MB
MAX_CHARS = 200_000                   # keep documents bounded


class IngestError(Exception):
    pass


def _source_type(ext: str) -> str:
    return {
        ".pdf": "pdf", ".docx": "docx", ".csv": "csv",
        ".md": "markdown", ".markdown": "markdown", ".txt": "text",
    }.get(ext, "text")


def _pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise IngestError("Parser PDF tidak tersedia di server (pypdf belum terpasang).")
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as e:
        raise IngestError(f"Gagal membaca PDF: {str(e)[:120]}")


def _docx(data: bytes) -> str:
    try:
        import docx
    except ImportError:
        raise IngestError("Parser DOCX tidak tersedia di server (python-docx belum terpasang).")
    try:
        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs if p.text and p.text.strip()]
        for table in d.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        return "\n".join(parts)
    except Exception as e:
        raise IngestError(f"Gagal membaca DOCX: {str(e)[:120]}")


def _csv(data: bytes) -> str:
    text = data.decode("utf-8-sig", errors="replace")
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except Exception as e:
        raise IngestError(f"Gagal membaca CSV: {str(e)[:120]}")
    if not rows:
        return ""
    header = rows[0]
    out = []
    for r in rows[1:]:
        pairs = []
        for i, val in enumerate(r):
            key = header[i] if i < len(header) and header[i].strip() else f"kolom{i+1}"
            if val.strip():
                pairs.append(f"{key}: {val.strip()}")
        if pairs:
            out.append("; ".join(pairs))
    return "\n".join(out) if out else text


def _text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    # collapse 3+ blank lines
    out = []
    blank = 0
    for ln in lines:
        if ln.strip():
            blank = 0
            out.append(ln)
        else:
            blank += 1
            if blank <= 1:
                out.append("")
    return "\n".join(out).strip()


def parse_file(filename: str, data: bytes) -> dict:
    if not data:
        raise IngestError("File kosong.")
    if len(data) > MAX_BYTES:
        raise IngestError("File terlalu besar (maks 10 MB).")
    ext = os.path.splitext((filename or "").lower())[1]
    if ext == ".pdf":
        text = _pdf(data)
    elif ext == ".docx":
        text = _docx(data)
    elif ext == ".csv":
        text = _csv(data)
    elif ext in (".md", ".markdown", ".txt"):
        text = _text(data)
    else:
        raise IngestError(f"Format tidak didukung: {ext or 'tanpa ekstensi'}. "
                          "Gunakan PDF, DOCX, CSV, MD, atau TXT.")
    text = _clean(text)
    if not text.strip():
        raise IngestError("Tidak ada teks yang bisa diekstrak dari file.")
    truncated = False
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n\n[... konten dipotong ...]"
        truncated = True
    return {"text": text, "source_type": _source_type(ext), "ext": ext, "truncated": truncated}
