# Nohalu SafeDesk

**Chat kerja aman dengan sumber, izin, dan audit.**

Backend untuk asisten chat internal perusahaan. Karyawan bertanya soal kebijakan,
SOP, status proyek, dan operasional kerja; agent hanya menjawab dari knowledge base
yang **diizinkan untuk user tersebut**, selalu menyertakan **sumber**, mendeteksi
**konflik** antar-dokumen, menolak secara **netral** bila data tidak ada atau tidak
boleh diakses, **memasking PII**, menolak **prompt injection**, dan mencatat
**audit log** append-only.

Dibuat untuk **Tencent Cloud Hackathon Indonesia 2026 — Challenge A: The Workplace Brain**.

```bash
git clone <repo-ini> && cd nohalu-safedesk-public
python -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt
cp backend/.env.example backend/.env
.venv/bin/python -m pytest -q          # 1276 passed, 1 skipped
.venv/bin/python run.py                # http://localhost:8000
```

Tanpa API key apa pun, tanpa layanan eksternal. SQLite dibuat otomatis dan diisi
dataset mock 70 dokumen / 12 karyawan saat pertama kali dijalankan.

---

## Masalah yang diselesaikan

Informasi perusahaan tersebar di grup chat, Drive, dan task board. Karyawan
menanyakan hal yang sama berulang kali, sementara staf senior menjadi bottleneck.
Solusi "tempel chatbot ke semua dokumen" menciptakan masalah lebih besar: model
tanpa kontrol akses akan membocorkan gaji, dokumen restricted, dan data pribadi.

SafeDesk membalik urutannya. **Izin diperiksa sebelum pencarian**, bukan disaring
setelah jawaban jadi. Dokumen di luar akses tidak pernah masuk kandidat retrieval,
sehingga penolakan tidak bisa membocorkan keberadaan data.

## Tujuh prinsip yang diuji, bukan diklaim

| # | Prinsip | Bukti |
|---|---|---|
| 1 | Setiap jawaban faktual punya citation (judul, tanggal efektif, kutipan) | `test_citation_hygiene.py` |
| 2 | Permission-aware: filter workspace ACL ∩ role **sebelum** retrieval | `test_doc_rbac.py`, `test_rbac_matrix.py` |
| 3 | Refusal netral — "tidak ada" dan "tidak boleh" memakai kalimat identik | `test_engine.py` |
| 4 | Conflict detection: pengumuman resmi terbaru menang, yang lama disebut | `test_engine.py` |
| 5 | PII masking (NIK, telepon, email, rekening) kecuali izin eksplisit | `test_personal_data.py` |
| 6 | Prompt injection: isi dokumen = data, bukan instruksi | `test_injection_hardening.py` (543 tes) |
| 7 | Audit trail append-only | `test_admin_guards.py` |

### Pertahanan prompt injection

Ini bagian yang paling banyak diuji (**543 kasus**). Yang ditutup:

- 22 bahasa (ID, EN, ES, PT, FR, DE, IT, NL, SV, PL, RU, UK, RO, CS, AR, TR,
  VI, TH, ZH, JA, KO, HI), termasuk instruksi campur bahasa dalam satu kalimat
- Bentuk berimbuhan: artikel Arab `ال`, sufiks imperatif `-وا`/`-ي`, perfektif
  Rusia `про-`, dan harakat/tatweel yang dulu meloloskan seluruh pola Arab
- Encoding berlapis: base64 (termasuk berlapis ganda), hex, ROT13, teks terbalik
- Penyamaran karakter: homoglyph Cyrillic + Yunani (termasuk varian ambigu ν/η/ρ)
- Pemisah per huruf (`i.g.n.o.r.e`), ASCII smuggling, URL/HTML di dalam dokumen
- **Per klausa, bukan per pesan** — satu kalimat sah tidak boleh menjadi perisai
  untuk perintah kedua di kalimat berikutnya

Deteksi berjalan di jalur engine, jadi tetap aktif walaupun LLM dimatikan.
Biaya terukur pada 1.392 pemindaian campuran (potongan dokumen asli + muatan
serangan, termasuk teks Arab/CJK): **median 1,1 ms, p95 2,7 ms, p99 3,5 ms**.
ROT13 dan reverse dibatasi ≤400 char, hex ≤2000 char supaya masukan besar tidak
membuat pemindaian meledak.

Pemindaian dokumen di-cache per isi (digest, batas total 8 MB, bisa diatur lewat
`NOHALU_STRIP_CACHE_CHARS`). Satu kueri memanggil pembersihan 3-4 kali pada
dokumen yang sama; tanpa cache, `tool_search_policies` pada korpus 19 KB butuh
2,3-2,8 detik CPU. Dengan cache: 8-12 ms setelah pemanasan. Dokumen 200 ribu
karakter (batas `MAX_CHARS`) turun dari ~7 detik ke ~0,2 detik.

Cakupan bahasa kini 22. Yang belum punya pola verba+objek sendiri: bahasa
Nordik lain di luar Swedia (Norwegia/Denmark/Finlandia), Yunani, Ibrani,
Bengali, dan Melayu non-Indonesia. Muatan dalam bahasa itu masih tertangkap
bila memakai frasa Inggris atau lewat jalur encoding.

## Arsitektur

```text
        ┌──────────────┐
        │  Frontend    │  (repo terpisah; API bisa dipakai tanpa UI)
        └──────┬───────┘
               │ Bearer <token>
        ┌──────▼────────────────────────────────────────┐
        │  FastAPI (backend/app.py)                     │
        │  rate limit · CORS · request log · SSE        │
        └──────┬────────────────────────────────────────┘
               │
        ┌──────▼─────────────┐
        │  agent.py          │  loop tool-calling, batas langkah & waktu
        └──────┬─────────────┘
               │
   ┌───────────▼────────────┐        ┌──────────────────┐
   │  engine.py             │───────▶│  llm.py          │ (opsional)
   │  permission router     │        │  mercury-3       │
   │  retrieval             │◀───────│  grounded rewrite│
   │  conflict · refusal    │        └──────────────────┘
   │  PII · injection       │
   │  citation · audit      │
   └───────────┬────────────┘
               │
        ┌──────▼─────────┐
        │  SQLite (WAL)  │  dokumen · user · audit · sesi chat
        └────────────────┘
```

**Engine deterministik adalah sumber kebenaran.** LLM hanya menulis ulang jawaban
yang sudah lolos gate keamanan; ia tidak pernah melihat dokumen di luar izin, dan
tidak pernah menerima permintaan yang seharusnya ditolak. Bila LLM gagal, mati, atau
tidak dikonfigurasi, sistem jatuh ke jawaban ekstraktif — bukan ke jawaban kosong.

## LLM opsional

Tanpa konfigurasi LLM, semua fitur tetap jalan (retrieval, sitasi, konflik, refusal,
PII, audit) dengan jawaban ekstraktif dari dokumen. Untuk mengaktifkan penulisan
ulang generatif:

```env
NOHALU_LLM_ENABLED=1
NOHALU_LLM_API_KEY=<kunci-anda>
NOHALU_LLM_MODEL=mercury-3
NOHALU_LLM_REASONING=high
```

Kunci dibaca **hanya dari environment/server** — tidak pernah dari klien. Endpoint
`/api/chat` mengabaikan parameter `use_llm` dari request.

## Menjalankan tes

```bash
.venv/bin/python -m pytest -q                    # suite penuh
.venv/bin/python tests/run_testcases.py          # 25 skenario (butuh LLM aktif)
.venv/bin/python tests/smoke_api.py              # smoke API + RBAC + upload + audit
```

Di mesin tanpa repo frontend, 2 tes UI di-skip otomatis (1274 passed, 3 skipped). Sisanya berjalan offline:
tidak ada panggilan jaringan, tidak ada biaya, tidak butuh kunci.

CI menjalankan suite yang sama di Python 3.11 dan 3.12 dengan `NOHALU_LLM_ENABLED=0`
— lihat `.github/workflows/backend.yml`.

## Akun demo

Login mock: `POST /api/auth/login` dengan `{"user_id": "u_budi"}`. Tidak ada password
karena ini dataset demo. **Lihat [SECURITY.md](SECURITY.md) sebelum memakai di
lingkungan nyata.**

| ID | Nama | Role | Catatan |
|---|---|---|---|
| `u_budi` | Budi Santoso | employee | akses terbatas — baseline demo |
| `u_andi` | Andi Pratama | team_lead | punya akses proyek restricted |
| `u_sari` | Sari Dewi | hr | data gaji & personal |
| `u_dewi` | Dewi Lestari | finance | approval reimbursement |
| `u_rina` | Rina Kusuma | admin | akses penuh |

## API

Auth: `Authorization: Bearer <token>`.

| Method | Path | Keterangan |
|---|---|---|
| POST | `/api/auth/login` | `{user_id}` → `{token, user}` |
| POST | `/api/auth/logout` | cabut token (jti) |
| GET | `/api/me` | profil dari token |
| POST | `/api/chat` | `{question, session_id?}` → jawaban + citations + flags |
| POST | `/api/agent/confirm` | setujui draft aksi yang disiapkan agent |
| GET | `/api/chat/{session_id}` | riwayat sesi |
| GET | `/api/documents` | daftar dokumen (terfilter izin) |
| POST | `/api/documents/upload` | admin: PDF/DOCX/CSV/MD/TXT → ekstrak + index |
| GET | `/api/requests` | pengajuan cuti/reimbursement |
| POST | `/api/requests/{id}/decide` | approval berjenjang |
| GET | `/api/audit` | admin: audit log (filter user/status/tanggal) |
| GET | `/api/notifications/stream` | SSE notifikasi real-time |

Dokumentasi interaktif: `/api/docs`.

Field penting pada respons `/api/chat`: `answer`, `citations`, `conflict_detected`,
`conflict_note`, `prompt_injection_detected`, `permission_decision`, `engine_used`,
`llm_error`, `suggestions`, `prepared`, `audit_id`, `session_id`.

## Struktur

```text
backend/
  app.py            FastAPI routes, auth, rate limit, SSE
  engine.py         permission router, retrieval, conflict, refusal, PII, injection, audit
  agent.py          loop tool-calling dengan batas langkah/waktu
  agent_tools.py    tool yang boleh dipanggil agent
  portal.py         dashboard, jadwal, payroll, direktori, proyek, approval, notifikasi
  personal.py       cuti, absensi, slip gaji
  hr_actions.py     aksi HR (approval, override)
  rbac.py           matriks izin role
  workspaces.py     ACL workspace
  docs_lifecycle.py versi & diff dokumen
  ingest.py         ekstraksi PDF/DOCX/CSV/MD/TXT
  llm.py            lapisan mercury-3 (grounded, opsional)
  auth.py           sesi HMAC + revokasi jti
  ratelimit.py      kuota per endpoint
  backup.py         backup harian + retensi
  reminders.py      job pengingat
  metrics.py        KPI & latensi
  db.py             skema SQLite + seed sync
  seed.py           dataset mock (70 dokumen, 12 karyawan)
  paging.py         pagination
  files.py          validasi upload
  payslip_pdf.py    generator slip gaji
  logging_setup.py  structured logging JSON
  users_admin.py    CRUD user
tests/              1276 tes
docs/               accountability sheet
```

## Keputusan desain

- **Permission sebelum retrieval.** Bukan filter pasca-jawaban. Ini yang membuat
  refusal tidak bisa dipakai sebagai oracle.
- **Refusal tunggal.** Akses ditolak dan data tidak ada memakai kalimat yang sama
  persis, termasuk daftar pertanyaan lanjutan yang sama. Kalau berbeda, selisihnya
  sendiri sudah membocorkan bahwa data itu ada.
- **Engine deterministik sebagai basis.** 25 skenario bisa dijalankan offline dan
  hasilnya reproducible; LLM tidak bisa mengubah keputusan izin.
- **Audit append-only.** Tidak ada endpoint update/delete untuk `audit_log`.

## Lisensi

MIT — lihat [LICENSE](LICENSE).
