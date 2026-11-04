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

Dalam MVP, agent tidak melakukan aksi eksternal. Jika dikembangkan:

- Kirim email/pesan.
- Buat task.
- Ubah dokumen.
- Ubah permission.
- Hapus data.
- Kirim data eksternal.

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
