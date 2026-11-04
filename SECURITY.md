# Security Notes

Dokumen ini menjelaskan apa yang **sudah** diamankan, apa yang **sengaja dibuat
longgar untuk demo**, dan apa yang harus diganti sebelum dipakai di lingkungan
nyata. Tujuannya agar pembaca tidak salah menilai: kode ini adalah backend demo
yang jujur, bukan produk yang siap produksi.

## 1. Autentikasi: demo, bukan produksi

`POST /api/auth/login` menerima `user_id` tanpa password. Siapa pun yang tahu ID
karyawan bisa masuk sebagai orang itu.

Yang **sudah** benar di sekitarnya:

- Token **ditandatangani server** dengan HMAC-SHA256 (`v1.<user>.<exp>.<jti>.<sig>`).
  Klien tidak bisa membuat token sendiri.
- Token punya masa berlaku (`NOHALU_SESSION_TTL_HOURS`).
- Token bisa **dicabut** per-jti: `/api/auth/logout` dan `/api/auth/logout-all`
  membuat token mati seketika, bukan sekadar dihapus dari browser.
- `NOHALU_SESSION_SECRET` wajib diisi di produksi. Bila kosong, server membuat
  secret acak sekali dan menyimpannya di samping file DB (mode `0600`).

**Ganti dengan SSO/OIDC/SCIM** sebelum dipakai sungguhan. Semua endpoint
memanggil satu fungsi identitas; titik gantinya tunggal.

## 2. Otorisasi: diperiksa sebelum retrieval

Ini bagian yang paling penting dan sudah benar sejak awal.

- Dokumen difilter `workspace ACL ∩ role` **sebelum** masuk kandidat retrieval.
- Refusal untuk "tidak ada" dan "tidak boleh" memakai kalimat identik, termasuk
  daftar pertanyaan lanjutan — sehingga penolakan tidak bisa dipakai sebagai
  oracle untuk menebak data mana yang eksis.
- RBAC diuji matriks penuh (`tests/test_rbac_matrix.py`).

## 3. Prompt injection

Isi dokumen diperlakukan sebagai **data**, bukan instruksi. Pertahanan berlapis
ada di `engine.py`, bukan di prompt LLM:

- 17 bahasa, encoding berlapis (base64 ganda, hex, ROT13, reverse)
- Homoglyph Cyrillic/Yunani, pemisah per huruf, ASCII smuggling
- Dievaluasi **per klausa** — satu kalimat sah tidak menjadi perisai untuk
  perintah kedua

Karena gate ini deterministik, mematikan LLM tidak melemahkan pertahanan.

## 4. Data pribadi

PII (NIK, telepon, email, nomor rekening) dimasking kecuali user punya izin
eksplisit. Slip gaji hanya bisa dibaca pemiliknya — **atasan pun tidak**, dan ini
diuji.

Dataset mock (`backend/seed.py`) memakai identitas fiktif. Nomor telepon memakai
pola contoh `0812-3456-7890`, email memakai domain `nusantarakreatif.id` yang bukan
milik siapa pun, dan NIK tidak diisi. Jangan menyalin dataset ini ke sistem nyata
tanpa mengganti seluruh identitas.

## 5. Batas laju dan kuota

Aktif secara default: login (20/menit), chat (60/menit), baca (240/menit),
pengajuan (30/menit), endpoint publik (120/menit), serta kuota SSE per user.
Semua dapat diubah lewat environment.

## 6. Upload berkas

- Format diizinkan: PDF, DOCX, CSV, MD, TXT. Maksimum 10 MB (`NOHALU_MAX_UPLOAD_BYTES`).
- Tipe diperiksa dari isi, bukan hanya ekstensi.
- Nama berkas tidak pernah dipakai sebagai path; ID dokumen selalu di-generate.
- Parsing berjalan di proses yang sama — untuk beban produksi, jalankan parser di
  worker terpisah dengan batas memori.

## 7. Audit

`audit_log` bersifat append-only: tidak ada endpoint update/delete. Setiap
keputusan (allow/deny, injeksi terdeteksi, konflik, akses dokumen) tercatat
dengan timestamp dan user.

## 8. Yang belum ada (jujur)

- Belum ada 2FA, sesi perangkat, atau rotasi kunci otomatis.
- Belum ada enkripsi at-rest untuk SQLite. Gunakan volume terenkripsi.
- Belum ada WAF. Letakkan di belakang reverse proxy dengan rate limit tambahan.
- Backup harian ada (`backup.py`), tetapi belum ada uji restore otomatis.

## Melaporkan masalah

Buka issue di repo ini. Untuk temuan sensitif, jangan sertakan eksploit di issue
publik — hubungi pemelihara lebih dulu.

## Cakupan

Repo ini adalah backend. Frontend berada di repo terpisah dan tidak termasuk di
sini. Tidak ada kredensial, kunci API, atau konfigurasi infrastruktur di dalam
repo ini; semua rahasia dibaca dari environment.
