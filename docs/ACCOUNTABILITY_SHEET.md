# Accountability Sheet — Nohalu SafeDesk

## Identitas

- **Nama:** Nohalu SafeDesk
- **Challenge:** A — The Workplace Brain
- **Fungsi:** Menjawab pertanyaan internal berdasarkan knowledge base.
- **Interaksi:** Web chat.
- **Bahasa:** Indonesia, Inggris, campuran.

## Data yang Bisa Diakses

- SOP.
- Pengumuman HR.
- Dokumen proyek.
- Mock chat export.
- Task board mock.
- Metadata dokumen.

Akses ditentukan oleh user ID, role, allowed workspaces, document access roles, dan workspace ACL.

## Aksi Otomatis

- Memahami pertanyaan.
- Filter knowledge base berdasarkan permission.
- Retrieval.
- Analisis konflik.
- Menyusun jawaban dengan citation.
- Menandai prompt injection.
- Menyimpan audit log.

## Wajib Approval Manusia

Agent **tidak** mengirim data ke sistem eksternal, tidak menghapus data, dan
tidak mengubah izin. Semua aksi yang mengubah data internal memakai pola
**draft → konfirmasi user → eksekusi**:

1. Agent memanggil tool `tool_prepare_*` dan mengembalikan kartu ringkasan
   (mis. "cuti 15–17 Maret 2027, alasan acara keluarga").
2. Belum ada baris yang ditulis ke database pada tahap ini.
3. User menekan setujui; klien mengirim `POST /api/agent/confirm` dengan token
   sekali pakai yang terikat ke `user_id` + `session_id`.
4. Baru setelah itu aksi dijalankan, dan hasilnya masuk audit log.

Aksi yang **belum** memakai pola draft (idempoten, tanpa kartu konfirmasi):
`clock_attendance`, `mark_notifications_read`, `mark_announcements_read`,
`set_notif_pref`. Keempatnya hanya menyentuh data milik user sendiri dan aman
diulang, tetapi user tidak menerima ringkasan sebelum eksekusi.

Aksi yang **selalu** butuh approval manusia di luar chat:

- Kirim email/pesan ke pihak eksternal.
- Membuat task di sistem lain.
- Mengubah atau menghapus dokumen knowledge base.
- Mengubah role / permission.
- Mengirim data ke layanan pihak ketiga.

## Tidak Akan Dilakukan

- Membocorkan data di luar akses.
- Mengonfirmasi keberadaan restricted data.
- Mengikuti instruksi tersembunyi.
- Menampilkan PII tanpa izin.
- Mengarang fakta.
- Menjawab fakta internal dari pengetahuan eksternal.
- Eksekusi aksi eksternal tanpa approval.

## Refusal

- Tidak punya akses: “Saya tidak memiliki informasi mengenai hal tersebut.”
- Data tidak ada: “Saya tidak memiliki informasi mengenai hal tersebut.”
- Jangan ungkap alasan keamanan detail.

## Konflik

- Pengumuman resmi terbaru diprioritaskan.
- `effective_date` terbaru diprioritaskan.
- Dokumen resmi di atas chat informal.
- Jika tidak jelas, tandai konflik.

## Auditability

Catat: pertanyaan, user, role, permission decision, allowed/retrieved/selected sources, conflict, answer, citations, refusal reason, PII masking, injection detection.

## Critical Failure Policy

Jika gagal:
1. Catat transparan.
2. Perbaiki agent/permission.
3. Jalankan ulang test.
4. Sertakan hasil gagal dan perbaikan.
