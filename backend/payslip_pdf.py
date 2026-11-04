"""payslip_pdf.py — PDF slip gaji sederhana tanpa dependensi luar.

Kenapa ditulis manual: satu-satunya pustaka PDF di venv (`pypdf`) bisa membaca
dan menggabung, tetapi tidak bisa menaruh teks baru di halaman. Menambah
`reportlab` (~10 MB) hanya untuk satu halaman slip tidak sepadan, dan slip
demo harus benar-benar bisa diunduh — bukan tombol yang selalu 404.

Yang dihasilkan: satu halaman A4, teks Helvetica, tanpa gambar. Font standar
PDF (Helvetica) tidak perlu ditanam, jadi berkasnya hanya beberapa KB.
"""
from datetime import datetime

PAGE_W, PAGE_H = 595, 842  # A4 dalam poin


def _esc(text):
    """Escape karakter khusus sintaks string PDF.

    Tanpa ini, nama seperti `O'Brien` atau tanda kurung di judul membuat
    struktur PDF rusak dan pembaca PDF menolak berkasnya.
    """
    out = str(text if text is not None else "")
    for bad in ("\\", "(", ")"):
        out = out.replace(bad, "\\" + bad)
    # Font standar memakai WinAnsi; karakter di luar ASCII aman diganti.
    return out.encode("latin-1", "replace").decode("latin-1")


def _rupiah(value):
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        n = 0
    return "Rp" + format(n, ",").replace(",", ".")


def build(slip: dict, company: str = "PT Nusantara Kreatif") -> bytes:
    """Return isi PDF untuk satu slip.

    `slip` butuh: user_name, user_id, period, gross, deductions, net.
    Field yang tidak ada ditampilkan sebagai "-" alih-alih membuat PDF gagal.
    """
    name = slip.get("user_name") or slip.get("user_id") or "-"
    period = slip.get("period") or "-"
    gross = slip.get("gross")
    deductions = slip.get("deductions")
    net = slip.get("net")
    issued = datetime.now().strftime("%d-%m-%Y")

    lines = [
        (24, 760, "F2", 18, company),
        (24, 736, "F1", 11, "Slip Gaji Karyawan"),
        (24, 706, "F2", 11, "Periode"),
        (140, 706, "F1", 11, period),
        (24, 688, "F2", 11, "Nama"),
        (140, 688, "F1", 11, name),
        (24, 670, "F2", 11, "ID Karyawan"),
        (140, 670, "F1", 11, slip.get("user_id") or "-"),
        (24, 630, "F2", 11, "Rincian"),
        (24, 608, "F1", 11, "Gaji bruto"),
        (400, 608, "F1", 11, _rupiah(gross)),
        (24, 590, "F1", 11, "Potongan"),
        (400, 590, "F1", 11, "-" + _rupiah(deductions)),
        (24, 560, "F2", 12, "Diterima bersih"),
        (400, 560, "F2", 12, _rupiah(net)),
        (24, 500, "F1", 9,
         "Dokumen ini dibuat otomatis oleh SafeDesk dan sah tanpa tanda tangan basah."),
        (24, 486, "F1", 9, "Pertanyaan: hubungi HR (hr@nusantarakreatif.co.id)."),
        (24, 460, "F1", 9, "Diterbitkan: " + issued),
    ]

    stream = ["BT"]
    for x, y, font, size, text in lines:
        stream.append(f"/{font} {size} Tf 1 0 0 1 {x} {y} Tm ({_esc(text)}) Tj")
    stream.append("ET")
    # Garis pemisah di atas blok rincian.
    stream.append("0.6 w 24 620 m 571 620 l S")
    stream.append("0.6 w 24 578 m 571 578 l S")
    content = "\n".join(stream).encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
         "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> "
         "/Contents 6 0 R >>").encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
        b"/Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
        + content + b"\nendstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(i).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += ("%010d 00000 n \n" % off).encode()
    out += (b"trailer\n<< /Size " + str(len(objects) + 1).encode()
            + b" /Root 1 0 R >>\nstartxref\n"
            + str(xref_at).encode() + b"\n%%EOF\n")
    return bytes(out)
