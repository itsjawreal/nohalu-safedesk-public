"""Tool chat yang memperluas bantuan agent di luar tanya-jawab.

Sebelum ini banyak kemampuan backend hanya bisa dipicu dari halaman:
ringkasan kerja, jadwal, tanggal gajian, proyek, profil perusahaan, pratinjau
rantai approval, struktur organisasi, dan preferensi notifikasi. Akibatnya
model menjawab "tidak tersedia" untuk pertanyaan yang datanya ada, atau
menghitung sendiri tanggal gajian.

Tes ini menjaga tiga hal:
1. Tool baru mengembalikan data nyata (bukan stub kosong).
2. Batas peran tetap berlaku (karyawan biasa tidak melihat data approver).
3. Aksi ringan benar-benar mengubah state dan idempoten.
"""
import json
import os
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_LLM_ENABLED"] = "0"

import agent
import agent_tools
import db
import engine
import personal
import portal


def make_conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_db(conn)
    db.seed_if_empty(conn)
    db.ensure_seed_users(conn)
    return conn


def ctx_for(conn, uid="u_budi"):
    # `engine.get_user` yang mem-parse `allowed_workspaces` dari JSON. Membaca
    # baris mentah membuat ACL selalu kosong dan tes menguji organisasi yang
    # tidak pernah terjadi di produksi.
    user = engine.get_user(conn, uid)
    assert user is not None, uid
    return agent_tools.ToolContext(conn, user)


# ---------------------------------------------------------------------------
# Skema & registri
# ---------------------------------------------------------------------------
def test_every_schema_has_implementation():
    names = {s["function"]["name"] for s in agent_tools.TOOL_SCHEMAS}
    assert names == set(agent_tools.TOOL_FUNCS), (
        sorted(names - set(agent_tools.TOOL_FUNCS)),
        sorted(set(agent_tools.TOOL_FUNCS) - names))


def test_new_tools_are_offered_to_the_model():
    names = {s["function"]["name"] for s in agent_tools.TOOL_SCHEMAS}
    for n in ("get_dashboard", "get_schedule", "get_payroll", "get_projects",
              "get_company", "get_request_rules", "get_notif_prefs",
              "get_org_chart", "mark_notifications_read",
              "mark_announcements_read", "set_notif_pref"):
        assert n in names, n


def test_new_tools_counted_as_personal_data():
    """`limitations` harus bilang 'data kepegawaian', bukan 'dokumen'."""
    for n in ("get_dashboard", "get_schedule", "get_payroll", "get_projects",
              "get_company", "get_request_rules", "get_notif_prefs",
              "get_org_chart", "mark_notifications_read",
              "mark_announcements_read", "set_notif_pref"):
        assert n in agent._PERSONAL_TOOLS, n


# ---------------------------------------------------------------------------
# Ringkasan & navigasi
# ---------------------------------------------------------------------------
def test_dashboard_returns_own_request_stats():
    conn = make_conn()
    out = agent_tools.execute(ctx_for(conn), "get_dashboard", {})
    assert out["me"]["id"] == "u_budi"
    assert out["requests"]["total"] >= 0
    assert isinstance(out["available_documents"], int)
    assert out["available_documents"] > 0
    # Karyawan biasa tidak boleh menerima antrean approval.
    assert "approval_waiting" not in out
    assert "org" not in out


def test_dashboard_gives_approver_their_queue():
    conn = make_conn()
    out = agent_tools.execute(ctx_for(conn, "u_andi"), "get_dashboard", {})
    assert out["me"]["id"] == "u_andi"
    assert "approval_waiting_count" in out
    assert isinstance(out["approval_waiting"], list)


def test_dashboard_hr_sees_org_aggregate():
    conn = make_conn()
    budi = engine.get_user(conn, "u_budi")
    # Seed tidak membuat pengajuan; buat satu supaya agregat organisasi
    # benar-benar berisi dan tes tidak lulus hanya karena angka nol.
    out, status = portal.create_request(conn, budi, {
        "type": "cuti", "title": "Cuti tahunan",
        "start_date": "2026-12-01", "end_date": "2026-12-02",
    })
    assert status < 400, out

    data = agent_tools.execute(ctx_for(conn, "u_sari"), "get_dashboard", {})
    assert data["org"]["total"] >= 1
    assert set(data["org"]["by_type"]) == {"cuti", "reimbursement"}
    assert data["org"]["by_type"]["cuti"] >= 1


def test_schedule_returns_working_hours_doc():
    conn = make_conn()
    out = agent_tools.execute(ctx_for(conn), "get_schedule", {})
    ids = {i["id"] for i in out["items"]}
    assert "doc_hr_working_hours" in ids
    assert any(i.get("meta") for i in out["items"]), out["items"]


def test_payroll_next_date_is_computed_by_server():
    conn = make_conn()
    out = agent_tools.execute(ctx_for(conn), "get_payroll", {})
    assert out["next_pay_date"], out
    assert isinstance(out["days_until"], int)
    assert out["days_until"] >= 0
    # Karyawan biasa tidak menerima struktur gaji.
    assert out["salary_structure"] is None


def test_payroll_sends_ready_made_day_label():
    """Nama hari tidak boleh dihitung model.

    Terukur: model menulis "Rabu, 27 November 2026" untuk tanggal yang
    sebenarnya Jumat. Label dari server menghilangkan kelas kesalahan ini.
    """
    conn = make_conn()
    out = agent_tools.execute(ctx_for(conn), "get_payroll", {})
    iso = out["next_pay_date"]
    label = out["next_pay_date_label"]

    import datetime as dt
    expected_day = ("Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu",
                    "Minggu")[dt.date.fromisoformat(iso).weekday()]
    assert label.startswith(expected_day), (iso, label)
    assert str(dt.date.fromisoformat(iso).year) in label


def test_payroll_hr_gets_salary_structure():
    conn = make_conn()
    out = agent_tools.execute(ctx_for(conn, "u_sari"), "get_payroll", {})
    assert out["salary_structure"], "HR harus menerima struktur gaji"


def test_projects_respect_rbac():
    conn = make_conn()
    budi = agent_tools.execute(ctx_for(conn), "get_projects", {})
    budi_ids = {p["id"] for p in budi["projects"]}
    assert "doc_project_alpha" in budi_ids
    assert "doc_kenari_partner" not in budi_ids


def test_company_returns_grouped_sections():
    conn = make_conn()
    out = agent_tools.execute(ctx_for(conn), "get_company", {})
    assert out["count"] >= 1
    assert out["groups"], "profil perusahaan harus punya pengelompokan"
    ids = {s["id"] for s in out["sections"]}
    assert "doc_company_mission_values" in ids


def test_request_rules_preview_matches_form_contract():
    conn = make_conn()
    out = agent_tools.execute(ctx_for(conn), "get_request_rules", {"type": "cuti"})
    assert out["type"] == "cuti"
    assert out["stages"] == ["team_lead", "hr"]

    big = agent_tools.execute(
        ctx_for(conn), "get_request_rules",
        {"type": "reimbursement", "amount": "2000000"})
    assert big["finance_manager_required"] is True
    assert "finance" in big["stages"]

    small = agent_tools.execute(
        ctx_for(conn), "get_request_rules",
        {"type": "reimbursement", "amount": "100000"})
    assert small["finance_manager_required"] is False
    assert "finance" not in small["stages"]


def test_request_rules_without_amount_returns_both_branches():
    """Tanpa nominal, tool dulu error → model MENGARANG nominal 2 juta.

    Terukur di :8890: "Siapa saja approver untuk pengajuan saya?" memanggil
    `get_request_rules(type=reimbursement)` tanpa nominal, mendapat
    `tool_error`, lalu memanggil lagi dengan `amount="2000000"` — angka yang
    tidak pernah disebut user — hanya supaya tool mau menjawab.
    """
    conn = make_conn()
    out = agent_tools.execute(
        ctx_for(conn), "get_request_rules", {"type": "reimbursement"})
    assert out.get("error") is None, out
    assert out["amount_given"] is False
    assert out["approval_threshold"] > 0
    low = out["variants"]["at_or_below_threshold"]
    high = out["variants"]["above_threshold"]
    assert low["finance_manager_required"] is False
    assert high["finance_manager_required"] is True
    assert "finance" in high["stages"]
    assert "jangan mengarang" in out["note"].lower(), out["note"]
    # Note diturunkan dari data nyata per varian, bukan kalimat hardcoded.
    assert low["stages"] or low["skipped"], low


def test_request_rules_note_reflects_skipped_self_approval_stage():
    """Approver yang juga pemegang peran tidak boleh diberi janji "Team Lead"."""
    conn = make_conn()
    out = agent_tools.execute(
        ctx_for(conn, "u_andi"), "get_request_rules", {"type": "reimbursement"})
    assert "team_lead dilewati" in out["note"], out["note"]


def test_request_rules_registers_policy_document_as_evidence():
    """Rantai approval berasal dari SOP, jadi harus bersitasi dokumen.

    Terukur di produksi sebelum perbaikan:
    "Siapa yang menyetujui reimbursement Rp800.000?" -> `get_request_rules`
    dipanggil, jawabannya benar, tetapi cites=0 dan `data_source=personal`
    sehingga `limitations` berbunyi "data kepegawaian Anda" padahal isinya
    persis `doc_approval_matrix`.
    """
    conn = make_conn()
    ctx = ctx_for(conn)
    agent_tools.execute(ctx, "get_request_rules",
                        {"type": "reimbursement", "amount": "800000"})
    assert "doc_approval_matrix" in ctx.evidence, ctx.evidence.keys()

    # Tanpa nominal pun kedua cabang tetap berasal dari dokumen yang sama.
    ctx2 = ctx_for(conn)
    agent_tools.execute(ctx2, "get_request_rules", {"type": "reimbursement"})
    assert "doc_approval_matrix" in ctx2.evidence, ctx2.evidence.keys()

    # Cuti memakai kebijakan cuti, bukan approval matrix (dokumen pendamping
    # yang tidak relevan akan menjadi sitasi palsu saat judul tidak disebut).
    ctx3 = ctx_for(conn)
    agent_tools.execute(ctx3, "get_request_rules", {"type": "cuti"})
    assert "doc_leave_policy" in ctx3.evidence, ctx3.evidence.keys()
    assert "doc_approval_matrix" not in ctx3.evidence


def test_request_rules_never_cites_document_the_user_cannot_read():
    """ACL tetap dipegang `ctx.allowed`; tool tidak boleh membocorkan dokumen.

    `ctx.allowed` dibatasi di sini untuk meniru peran yang tidak punya akses ke
    approval matrix. Tool harus tetap menjawab (data approval dihitung dari
    tabel), tetapi tanpa mendaftarkan dokumen terlarang sebagai bukti.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    restricted = [d for d in engine.filter_allowed_docs(conn, user)
                  if d["id"] != "doc_approval_matrix"]
    ctx = agent_tools.ToolContext(conn, user, allowed=restricted)
    out = agent_tools.execute(ctx, "get_request_rules",
                             {"type": "reimbursement", "amount": "800000"})
    assert out.get("error") is None, out
    assert "doc_approval_matrix" not in ctx.evidence


def test_request_rules_rejects_unknown_type():
    conn = make_conn()
    out = agent_tools.execute(ctx_for(conn), "get_request_rules", {"type": "lembur"})
    assert out.get("error") == "tool_error"
    assert "cuti" in (out.get("message") or "").lower()


def test_org_chart_has_chain_and_reports():
    conn = make_conn()
    out = agent_tools.execute(ctx_for(conn), "get_org_chart", {})
    assert out["me"]["id"] == "u_budi"
    assert [p["name"] for p in out["chain"]], out["chain"]
    # Tidak ada data gaji/kontak pribadi di bagan organisasi.
    blob = repr(out).lower()
    for bad in ("salary", "gaji", "phone", "telepon", "email"):
        assert bad not in blob, bad


# ---------------------------------------------------------------------------
# Aksi ringan
# ---------------------------------------------------------------------------
def test_mark_notifications_read_updates_state():
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    ctx = agent_tools.ToolContext(conn, user)
    portal._notify(conn, [user["id"]], "Uji notifikasi", category="system")
    conn.commit()
    before = portal.unread_count(conn, user["id"])
    assert before >= 1

    out = agent_tools.execute(ctx, "mark_notifications_read", {"category": "all"})
    assert out["ok"] is True
    assert out["updated"] >= 1
    assert out["unread"] == 0


def test_mark_notifications_read_rejects_unknown_category():
    conn = make_conn()
    out = agent_tools.execute(
        ctx_for(conn), "mark_notifications_read", {"category": "gaji"})
    assert out.get("error") == "tool_error"


def test_mark_announcements_read_marks_only_allowed_docs():
    conn = make_conn()
    ctx = ctx_for(conn)
    out = agent_tools.execute(ctx, "mark_announcements_read", {})
    assert out["ok"] is True
    allowed = {d["id"] for d in agent_tools.engine.filter_allowed_docs(conn, ctx.user)}
    rows = conn.execute(
        "SELECT doc_id FROM announcement_reads WHERE user_id=?",
        (ctx.user["id"],)).fetchall()
    assert rows, "harus ada pengumuman yang ditandai"
    assert {r["doc_id"] for r in rows} <= allowed


def test_mark_announcements_read_separates_announcement_and_notification_unread():
    """`unread` global pernah dilaporkan model sebagai sisa PENGUMUMAN.

    Terukur di :8890: "Tandai semua pengumuman sudah saya baca" dijawab
    "masih ada 6 pengumuman yang belum terbaca", padahal 6 itu notifikasi
    kategori `request`/`payslip`. Tool harus memberi dua angka terpisah.
    """
    conn = make_conn()
    ctx = ctx_for(conn)
    portal._notify(conn, [ctx.user["id"]], "Pengajuan menunggu",
                   category="request")
    conn.commit()
    out = agent_tools.execute(ctx, "mark_announcements_read", {})
    assert out.get("error") is None, out
    assert out["unread_announcements"] == 0, out
    assert out["unread_notifications"] >= 1, out
    assert "unread_announcements" in out["note"], out


def test_set_notif_pref_toggles_and_system_is_locked():
    conn = make_conn()
    ctx = ctx_for(conn)
    out = agent_tools.execute(
        ctx, "set_notif_pref", {"category": "approval", "enabled": False})
    cats = {c["category"]: c for c in out["categories"]}
    assert cats["approval"]["enabled"] is False
    assert cats["system"]["locked"] is True

    locked = agent_tools.execute(
        ctx, "set_notif_pref", {"category": "system", "enabled": False})
    assert locked.get("error") == "tool_error"
    assert "tidak dapat dimatikan" in (locked.get("message") or "").lower()


def test_get_notif_prefs_lists_all_categories():
    conn = make_conn()
    out = agent_tools.execute(ctx_for(conn), "get_notif_prefs", {})
    cats = [c["category"] for c in out["categories"]]
    assert cats == list(portal.NOTIF_CATEGORIES)


# ---------------------------------------------------------------------------
# Kebocoran
# ---------------------------------------------------------------------------
def test_no_new_tool_returns_salary_or_other_people_contact():
    conn = make_conn()
    ctx = ctx_for(conn)
    for name, args in [
        ("get_dashboard", {}), ("get_schedule", {}), ("get_payroll", {}),
        ("get_projects", {}), ("get_company", {}), ("get_org_chart", {}),
        ("get_notif_prefs", {}),
    ]:
        out = agent_tools.execute(ctx, name, args)
        assert "error" not in out, (name, out)
        blob = repr(out).lower()
        assert "0812-3456-7890" not in blob, name


# ---------------------------------------------------------------------------
# UI: setiap tool wajib punya label manusiawi
# ---------------------------------------------------------------------------
def test_every_tool_has_a_human_label_in_the_ui():
    """Tool tanpa label tampil sebagai `snake_case` mentah di jejak agent.

    Saat 11 tool baru ditambahkan, UI belum mengenal satupun nama itu sehingga
    user melihat "get_request_rules" di balik jawaban. Tes ini mengikat kedua
    repo: menambah tool di backend tanpa label akan gagal di sini, bukan
    ketahuan dari screenshot.
    """
    import re
    ui = os.path.join(ROOT, "..", "app-nohalu-ui", "js", "chat", "agent.js")
    if not os.path.exists(ui):
        pytest.skip("repo UI tidak ada di mesin ini")
    src = open(ui, encoding="utf-8").read()
    block = src.split("const TOOL_LABEL = {", 1)[1].split("};", 1)[0]
    labels = set(re.findall(r"^\s{2}([a-z_]+):", block, re.M))
    missing = sorted(set(agent_tools.TOOL_FUNCS) - labels)
    assert not missing, f"tool tanpa label UI: {missing}"


def test_no_duplicate_tool_names_in_ui_labels():
    """Dua kunci sama akan membuat label terakhir diam-diam menimpa."""
    import re
    ui = os.path.join(ROOT, "..", "app-nohalu-ui", "js", "chat", "agent.js")
    if not os.path.exists(ui):
        pytest.skip("repo UI tidak ada di mesin ini")
    src = open(ui, encoding="utf-8").read()
    block = src.split("const TOOL_LABEL = {", 1)[1].split("};", 1)[0]
    keys = re.findall(r"^\s{2}([a-z_]+):", block, re.M)
    assert len(keys) == len(set(keys)), "label ganda di TOOL_LABEL"


# ---------------------------------------------------------------------------
# Riwayat keputusan: "siapa yang menyetujui" harus dari data, bukan karangan
# ---------------------------------------------------------------------------
def test_my_requests_exposes_who_decided():
    """Baris lama menyimpan `decided_by` tanpa `history`.

    Model pernah menjawab "catatan audit tidak tersedia" untuk klaim yang
    kolom `decided_by`-nya jelas berisi `u_andi`. Nama penyetuju harus ikut
    terkirim supaya jawaban bersumber data, bukan rekonstruksi rantai peran.
    """
    conn = make_conn()
    budi = engine.get_user(conn, "u_budi")
    out, status = portal.create_request(conn, budi, {
        "type": "reimbursement", "title": "Klaim tiket",
        "amount": "800000", "note": "tiket",
    })
    assert status < 400, out
    req_id = out["id"]
    andi = engine.get_user(conn, "u_andi")
    decided, code = portal.decide_request(conn, andi, req_id, "approve", "ok")
    assert code < 400, decided

    data = agent_tools.execute(ctx_for(conn), "get_my_requests", {})
    row = next(r for r in data["requests"] if r["id"] == req_id)
    assert row["status"] == "approved"
    assert row.get("decided_by_name") == "Andi Pratama", row
    assert row.get("decided_at"), row


def test_my_requests_history_carries_approver_name():
    """`history` harus ikut: tanpa itu model mengarang rantai peran umum."""
    conn = make_conn()
    budi = engine.get_user(conn, "u_budi")
    out, status = portal.create_request(conn, budi, {
        "type": "cuti", "title": "Cuti keluarga",
        "start_date": "2026-12-10", "end_date": "2026-12-12",
    })
    assert status < 400, out
    req_id = out["id"]
    andi = engine.get_user(conn, "u_andi")
    portal.decide_request(conn, andi, req_id, "approve", "silahkan")

    data = agent_tools.execute(ctx_for(conn), "get_my_requests", {})
    row = next(r for r in data["requests"] if r["id"] == req_id)
    assert row.get("history"), row
    names = {h.get("by_name") for h in row["history"]}
    assert "Andi Pratama" in names, row["history"]


def test_pending_request_reports_current_holder():
    """"Sekarang di meja siapa" harus menyebut nama, bukan peran mentah."""
    conn = make_conn()
    budi = engine.get_user(conn, "u_budi")
    out, status = portal.create_request(conn, budi, {
        "type": "cuti", "title": "Cuti akhir tahun",
        "start_date": "2026-12-20", "end_date": "2026-12-22",
    })
    assert status < 400, out
    req_id = out["id"]

    data = agent_tools.execute(ctx_for(conn), "get_my_requests", {})
    row = next(r for r in data["requests"] if r["id"] == req_id)
    assert row["status"] == "pending"
    assert row.get("waiting_on_role") == "team_lead"
    assert row.get("waiting_on_name") == "Andi Pratama", row


def test_history_is_capped_and_has_no_internal_ids_only():
    """Riwayat panjang tidak boleh membanjiri konteks model."""
    conn = make_conn()
    budi = engine.get_user(conn, "u_budi")
    out, status = portal.create_request(conn, budi, {
        "type": "reimbursement", "title": "Klaim besar",
        "amount": "2000000", "note": "perjalanan",
    })
    assert status < 400, out
    data = agent_tools.execute(ctx_for(conn), "get_my_requests", {})
    row = next(r for r in data["requests"] if r["id"] == out["id"])
    assert len(row.get("history") or []) <= 12


# ---------------------------------------------------------------------------
# Jadwal cuti tim: satu kueri rentang, bukan 7 panggilan kehadiran
# ---------------------------------------------------------------------------
def _approved_leave(conn, uid, start, end, title="Cuti"):
    """Buat cuti lalu setujui TL + HR supaya statusnya `approved`."""
    user = engine.get_user(conn, uid)
    out, code = portal.create_request(conn, user, {
        "type": "cuti", "title": title,
        "start_date": start, "end_date": end,
    })
    assert code < 400, out
    req_id = out["id"]
    andi = engine.get_user(conn, "u_andi")
    step, code = portal.decide_request(conn, andi, req_id, "approve", "ok")
    assert code < 400, step
    sari = engine.get_user(conn, "u_sari")
    step, code = portal.decide_request(conn, sari, req_id, "approve", "ok")
    assert code < 400, step
    return req_id


def test_team_leave_returns_approved_leave_in_range():
    """Team lead cukup satu panggilan untuk "siapa cuti minggu depan"."""
    conn = make_conn()
    req_id = _approved_leave(conn, "u_budi", "2027-03-08", "2027-03-10")
    data = agent_tools.execute(ctx_for(conn, "u_andi"), "get_team_leave", {
        "start_date": "2027-03-01", "end_date": "2027-03-14"})
    assert "error" not in data, data
    row = next(r for r in data["on_leave"] if r["request_id"] == req_id)
    assert row["name"] == "Budi Santoso", row
    assert row["status"] == "approved", row
    assert data["summary"]["approved"] == 1, data["summary"]
    assert data["scope"] == "bawahan langsung Anda", data


def test_team_leave_denied_for_employee():
    """Karyawan biasa tidak boleh membaca jadwal cuti rekan kerja."""
    conn = make_conn()
    data = agent_tools.execute(ctx_for(conn, "u_budi"), "get_team_leave", {
        "start_date": "2027-03-01", "end_date": "2027-03-14"})
    assert data.get("denied") is True, data
    assert data.get("error") == "tool_error", data


def test_team_leave_days_in_range_counts_overlap_only():
    """Cuti 1–10 yang ditanya 5–6 hanya 2 hari, bukan 10."""
    conn = make_conn()
    req_id = _approved_leave(conn, "u_budi", "2027-04-01", "2027-04-10")
    data = agent_tools.execute(ctx_for(conn, "u_andi"), "get_team_leave", {
        "start_date": "2027-04-05", "end_date": "2027-04-06"})
    row = next(r for r in data["on_leave"] if r["request_id"] == req_id)
    assert row["days_in_range"] == 2, row
    assert row["start_date"] == "2027-04-01", row
    assert row["end_date"] == "2027-04-10", row


def test_team_leave_rejects_range_over_92_days():
    conn = make_conn()
    data = agent_tools.execute(ctx_for(conn, "u_andi"), "get_team_leave", {
        "start_date": "2027-01-01", "end_date": "2027-06-30"})
    assert data.get("error") == "tool_error", data
    assert "92" in data.get("message", ""), data


def test_team_leave_can_exclude_pending():
    """`include_pending=False` hanya cuti yang sudah pasti."""
    conn = make_conn()
    budi = engine.get_user(conn, "u_budi")
    out, code = portal.create_request(conn, budi, {
        "type": "cuti", "title": "Cuti menunggu HR",
        "start_date": "2027-05-04", "end_date": "2027-05-05",
    })
    assert code < 400, out
    req_id = out["id"]
    andi = engine.get_user(conn, "u_andi")
    portal.decide_request(conn, andi, req_id, "approve", "ok")

    args = {"start_date": "2027-05-01", "end_date": "2027-05-07"}
    with_pending = agent_tools.execute(ctx_for(conn, "u_andi"),
                                       "get_team_leave", dict(args))
    ids = {r["request_id"] for r in with_pending["on_leave"]}
    assert req_id in ids, with_pending

    only_approved = agent_tools.execute(
        ctx_for(conn, "u_andi"), "get_team_leave",
        {**args, "include_pending": False})
    ids = {r["request_id"] for r in only_approved["on_leave"]}
    assert req_id not in ids, only_approved


# ---------------------------------------------------------------------------
# System prompt tidak boleh menjanjikan tool yang menolak peran itu
# ---------------------------------------------------------------------------
def test_capabilities_do_not_promise_team_leave_to_employee():
    """Dulu `_capabilities` menjanjikan "jadwal cuti tim" ke SEMUA peran.

    Model lalu menawarkannya ke karyawan biasa, dan `get_team_leave` menolak.
    Janji yang tidak bisa ditepati adalah bug, bukan sekadar gaya bahasa.
    """
    employee = agent._capabilities("employee")
    assert "jadwal cuti" not in employee, employee
    for role in ("team_lead", "finance", "hr", "admin"):
        assert "jadwal cuti" in agent._capabilities(role), role


def test_capability_claims_match_tool_permissions():
    """Setiap peran yang ditawari jadwal cuti benar-benar boleh memanggilnya."""
    conn = make_conn()
    for uid in ("u_budi", "u_andi", "u_sari", "u_dewi", "u_rina"):
        ctx = ctx_for(conn, uid)
        role = ctx.user["role"]
        offered = "jadwal cuti" in agent._capabilities(role)
        out = agent_tools.execute(ctx, "get_team_leave", {
            "start_date": "2027-07-01", "end_date": "2027-07-03"})
        denied = out.get("denied") is True
        assert offered != denied, (uid, role, offered, out)


def test_team_leave_treats_string_false_as_false():
    """Model bisa mengirim `"false"` (string) — itu truthy di Python."""
    conn = make_conn()
    budi = engine.get_user(conn, "u_budi")
    out, code = portal.create_request(conn, budi, {
        "type": "cuti", "title": "Cuti menunggu HR",
        "start_date": "2027-08-10", "end_date": "2027-08-11",
    })
    assert code < 400, out
    req_id = out["id"]
    portal.decide_request(conn, engine.get_user(conn, "u_andi"),
                          req_id, "approve", "ok")

    data = agent_tools.execute(ctx_for(conn, "u_andi"), "get_team_leave", {
        "start_date": "2027-08-01", "end_date": "2027-08-31",
        "include_pending": "false"})
    ids = {r["request_id"] for r in data["on_leave"]}
    assert req_id not in ids, data


# ---------------------------------------------------------------------------
# Dokumen dirujuk lewat JUDUL, bukan doc_id
# ---------------------------------------------------------------------------
def test_archive_document_resolves_title():
    """User menyebut judul; model hampir tidak pernah tahu `doc_id`."""
    import json as _json
    conn = make_conn()
    conn.execute(
        """INSERT INTO documents
           (id,title,source_type,effective_date,workspace_ids,access_roles,
            content,owner,confidentiality,status,version,content_hash,local_edit)
           VALUES ('doc_title_a','SOP Perjalanan Dinas','sop','2026-01-01',
                   ?,?, 'isi','HR','internal','published','1','h',1)""",
        (_json.dumps(["ws_onboarding"]), _json.dumps(["employee", "hr", "admin"])))
    conn.commit()
    ctx = ctx_for(conn, "u_sari")
    out = agent_tools.execute(ctx, "prepare_document_archive",
                              {"document": "SOP Perjalanan Dinas", "action": "delete"})
    assert out.get("confirm_token"), out
    assert "SOP Perjalanan Dinas" in out["summary"]


def test_archive_document_rejects_unknown_title():
    conn = make_conn()
    ctx = ctx_for(conn, "u_sari")
    out = agent_tools.execute(ctx, "prepare_document_archive",
                              {"document": "Dokumen Yang Tidak Ada"})
    assert out.get("error"), out


def test_draft_suggestion_matches_action_kind():
    """Saran "Ya, kirim pengajuannya" salah untuk draft pengumuman/slip."""
    cases = {
        "create_announcement": "pengumuman",
        "publish_payslip": "slip",
        "save_document": "dokumen",
        "change_employee": "perubahan",
        "create_request": "pengajuan",
    }
    for action, word in cases.items():
        out = agent._suggestions(
            {"prepared": {"token": "cfm_x", "action": action}}, "q")
        assert out and word in out[0].lower(), (action, out)


# ---------------------------------------------------------------------------
# Keputusan massal (`prepare_decide_bulk`)
# ---------------------------------------------------------------------------
def _pending_request(conn, uid, **over):
    user = engine.get_user(conn, uid)
    payload = {"type": "cuti", "title": "Cuti uji",
               "start_date": "2027-09-06", "end_date": "2027-09-07"}
    payload.update(over)
    out, code = portal.create_request(conn, user, payload)
    assert code < 400, out
    return out["id"]


def test_decide_bulk_approves_all_eligible_in_one_draft():
    """TL memutuskan tiga pengajuan sekaligus lewat SATU draft.

    Sebelum tool ini ada, model memanggil `prepare_decide_request` berulang
    kali dan aturan satu-draft-per-giliran menggugurkan semuanya kecuali yang
    terakhir — user melihat "3 disetujui" padahal hanya satu yang berubah.
    """
    conn = make_conn()
    ids = [
        _pending_request(conn, "u_maya"),
        _pending_request(conn, "u_fajar"),
        _pending_request(conn, "u_budi", type="reimbursement", title="Klaim taksi",
                         amount="250000"),
    ]
    ctx = ctx_for(conn, "u_andi")
    prep = agent_tools.execute(ctx, "prepare_decide_bulk",
                               {"request_ids": ", ".join(map(str, ids)),
                                "action": "approve", "note": "lengkap"})
    assert prep.get("confirm_token"), prep
    assert "3 pengajuan" in prep["summary"], prep
    # belum ada efek sebelum submit
    assert all(conn.execute("SELECT status FROM requests WHERE id=?", (i,))
               .fetchone()["status"] == "pending" for i in ids)

    res = agent_tools.execute(ctx, "submit_prepared",
                              {"confirm_token": prep["confirm_token"]})
    assert res.get("ok") is True, res
    assert len(res["decided"]) == 3, res
    assert res["failed"] == [], res
    # reimbursement 250rb hanya lewat team_lead -> approved penuh
    assert conn.execute("SELECT status FROM requests WHERE id=?", (ids[2],)) \
        .fetchone()["status"] == "approved"
    # cuti lanjut ke tahap HR, bukan selesai
    assert conn.execute("SELECT stage_idx FROM requests WHERE id=?", (ids[0],)) \
        .fetchone()["stage_idx"] == 1

    # token sekali pakai
    again = agent_tools.execute(ctx, "submit_prepared",
                                {"confirm_token": prep["confirm_token"]})
    assert again["error"] == "tool_error", again


def test_decide_bulk_skips_and_reports_reasons():
    """Nomor yang tidak bisa diputuskan dilewati, bukan menggagalkan semuanya."""
    conn = make_conn()
    done_id = _pending_request(conn, "u_maya")
    portal.decide_request(conn, engine.get_user(conn, "u_andi"), done_id,
                          "approve", "ok")
    live_id = _pending_request(conn, "u_fajar")

    ctx = ctx_for(conn, "u_andi")
    prep = agent_tools.execute(ctx, "prepare_decide_bulk", {
        "request_ids": f"{live_id},{done_id},999999", "action": "approve"})
    assert prep.get("confirm_token"), prep
    assert "dilewati" in prep["summary"], prep

    res = agent_tools.execute(ctx, "submit_prepared",
                              {"confirm_token": prep["confirm_token"]})
    assert res["ok"] is True, res
    assert [d["id"] for d in res["decided"]] == [live_id], res
    # yang sudah diputuskan tidak berubah statusnya karena ikut di-bulk
    assert conn.execute("SELECT stage_idx FROM requests WHERE id=?", (done_id,)) \
        .fetchone()["stage_idx"] == 1


def test_decide_bulk_rejects_own_request():
    """Pemisahan tugas: approver tidak boleh menyetujui pengajuannya sendiri."""
    conn = make_conn()
    rid = _pending_request(conn, "u_andi")
    conn.execute("UPDATE requests SET stages=?, stage_idx=0, status='pending' "
                 "WHERE id=?", ('["team_lead"]', rid))
    conn.commit()

    ctx = ctx_for(conn, "u_andi")
    out = agent_tools.execute(ctx, "prepare_decide_bulk",
                              {"request_ids": str(rid), "action": "approve"})
    assert out["error"] == "tool_error", out
    assert "sendiri" in out["message"], out
    assert conn.execute("SELECT status FROM requests WHERE id=?", (rid,)) \
        .fetchone()["status"] == "pending"


def test_decide_bulk_without_eligible_keeps_token_unused():
    """Bila semua ditolak, tidak ada draft yang boleh tersisa bisa dieksekusi."""
    conn = make_conn()
    rid = _pending_request(conn, "u_maya")   # menunggu team_lead
    ctx = ctx_for(conn, "u_budi")            # employee biasa
    out = agent_tools.execute(ctx, "prepare_decide_bulk",
                              {"request_ids": str(rid), "action": "approve"})
    assert out["error"] == "tool_error", out
    assert "team_lead" in out["message"], out
    assert conn.execute("SELECT COUNT(*) FROM agent_prepared").fetchone()[0] == 0


def test_decide_bulk_rejects_invalid_id_text():
    conn = make_conn()
    ctx = ctx_for(conn, "u_andi")
    out = agent_tools.execute(ctx, "prepare_decide_bulk",
                              {"request_ids": "12,abc", "action": "approve"})
    assert out["error"] == "tool_error", out
    assert "abc" in out["message"], out
    # daftar kosong juga ditolak, bukan diam-diam jadi draft kosong
    empty = agent_tools.execute(ctx, "prepare_decide_bulk",
                                {"request_ids": "  ", "action": "approve"})
    assert empty["error"] == "tool_error", empty


def test_decide_bulk_rejects_invalid_action():
    conn = make_conn()
    rid = _pending_request(conn, "u_maya")
    out = agent_tools.execute(ctx_for(conn, "u_andi"), "prepare_decide_bulk",
                              {"request_ids": str(rid), "action": "hapus"})
    assert out["error"] == "tool_error", out
    assert "approve" in out["message"], out


# ---------------------------------------------------------------------------
# Override di luar rantai approval (`prepare_override`)
# ---------------------------------------------------------------------------
def test_override_requires_hr_or_admin():
    conn = make_conn()
    rid = _pending_request(conn, "u_maya")
    for uid in ("u_budi", "u_andi", "u_dewi"):
        out = agent_tools.execute(ctx_for(conn, uid), "prepare_override", {
            "request_id": rid, "action": "approve",
            "note": "approver tidak tersedia"})
        assert out["error"] == "tool_error", (uid, out)
        assert out.get("denied") is True, (uid, out)
    assert conn.execute("SELECT status FROM requests WHERE id=?", (rid,)) \
        .fetchone()["status"] == "pending"


def test_override_requires_meaningful_reason():
    conn = make_conn()
    rid = _pending_request(conn, "u_maya")
    ctx = ctx_for(conn, "u_sari")
    out = agent_tools.execute(ctx, "prepare_override", {
        "request_id": rid, "action": "approve", "note": "urgent"})
    assert out["error"] == "tool_error", out
    assert "10" in out["message"], out
    assert conn.execute("SELECT COUNT(*) FROM agent_prepared").fetchone()[0] == 0


def test_override_approves_pending_request_and_records_actor():
    conn = make_conn()
    rid = _pending_request(conn, "u_maya")
    ctx = ctx_for(conn, "u_sari")
    note = "Approver sedang cuti, keputusan dibutuhkan hari ini"
    prep = agent_tools.execute(ctx, "prepare_override", {
        "request_id": rid, "action": "approve", "note": note})
    assert prep.get("confirm_token"), prep
    assert "OVERRIDE" in prep["summary"], prep

    res = agent_tools.execute(ctx, "submit_prepared",
                              {"confirm_token": prep["confirm_token"]})
    assert res["ok"] is True, res
    row = conn.execute("SELECT * FROM requests WHERE id=?", (rid,)).fetchone()
    assert row["status"] == "approved"
    assert row["overridden_by"] == "u_sari"
    assert row["decision_note"] == note
    # jejak audit tidak boleh hilang
    audit = conn.execute("SELECT * FROM audit_log WHERE event_type=?",
                         ("approval_override",)).fetchone()
    assert audit is not None and audit["user_id"] == "u_sari"


def test_override_rejects_decided_or_missing_request():
    conn = make_conn()
    rid = _pending_request(conn, "u_maya")
    portal.decide_request(conn, engine.get_user(conn, "u_andi"), rid,
                          "approve", "ok")
    portal.decide_request(conn, engine.get_user(conn, "u_sari"), rid,
                          "approve", "ok")
    ctx = ctx_for(conn, "u_rina")
    note = "Approver tidak tersedia hari ini"
    out = agent_tools.execute(ctx, "prepare_override", {
        "request_id": rid, "action": "approve", "note": note})
    assert out["error"] == "tool_error", out
    assert "approved" in out["message"], out

    missing = agent_tools.execute(ctx, "prepare_override", {
        "request_id": 987654, "action": "approve", "note": note})
    assert missing["error"] == "tool_error", missing
    assert "tidak ditemukan" in missing["message"], missing


def test_override_rejects_bad_request_id():
    conn = make_conn()
    out = agent_tools.execute(ctx_for(conn, "u_rina"), "prepare_override",
                              {"request_id": "dua", "action": "approve",
                               "note": "approver tidak tersedia"})
    assert out["error"] == "tool_error", out
    assert "angka" in out["message"], out


# ---------------------------------------------------------------------------
# Riwayat versi dokumen dari chat
# ---------------------------------------------------------------------------
def test_document_versions_resolves_title_and_marks_current():
    conn = make_conn()
    ctx = ctx_for(conn, "u_sari")
    created = agent_tools.execute(ctx, "prepare_document", {
        "title": "SOP Versi Chat", "source_type": "sop",
        "content": "Versi satu isi dokumen."})
    assert created.get("confirm_token"), created
    done = agent_tools.execute(ctx, "submit_prepared",
                               {"confirm_token": created["confirm_token"]})
    assert done.get("ok"), done
    doc_id = done.get("id") or done.get("doc_id")
    assert doc_id, done

    # Giliran baru: aturan satu-draft-per-giliran berlaku seperti request HTTP
    # berikutnya, jadi konteks dibuat ulang.
    ctx2 = ctx_for(conn, "u_sari")
    revised = agent_tools.execute(ctx2, "prepare_document", {
        "document": "SOP Versi Chat", "content": "Versi dua isi dokumen."})
    assert revised.get("confirm_token"), revised
    agent_tools.execute(ctx2, "submit_prepared",
                        {"confirm_token": revised["confirm_token"]})

    out = agent_tools.execute(ctx_for(conn, "u_sari"), "get_document_versions",
                              {"document": "SOP Versi Chat"})
    assert out.get("error") is None, out
    assert out["count"] >= 2, out
    assert any(v.get("current") for v in out["versions"]), out
    assert out["current_version"] == "2", out


def test_document_versions_requires_hr_and_known_title():
    conn = make_conn()
    denied = agent_tools.execute(ctx_for(conn, "u_budi"),
                                 "get_document_versions",
                                 {"document": "SOP Versi Chat"})
    assert denied["error"] == "tool_error" and denied.get("denied"), denied
    unknown = agent_tools.execute(ctx_for(conn, "u_sari"),
                                  "get_document_versions",
                                  {"document": "Dokumen Tidak Ada"})
    assert unknown["error"] == "tool_error", unknown


def _revise_doc(conn, uid, title, content):
    """Buat/revisi dokumen lewat jalur tool (satu draft per giliran)."""
    ctx = ctx_for(conn, uid)
    args = {"content": content, "source_type": "sop"}
    if title:
        args["title"] = title
    else:
        args["document"] = "SOP Diff Chat"
    out = agent_tools.execute(ctx, "prepare_document", args)
    assert out.get("confirm_token"), out
    done = agent_tools.execute(ctx, "submit_prepared",
                               {"confirm_token": out["confirm_token"]})
    assert done.get("ok"), done
    return done.get("id") or done.get("doc_id")


def test_document_diff_defaults_to_two_latest_versions():
    conn = make_conn()
    _revise_doc(conn, "u_sari", "SOP Diff Chat", "Baris satu.\nBaris dua.")
    _revise_doc(conn, "u_sari", "", "Baris satu.\nBaris dua diubah.\nBaris tiga.")

    out = agent_tools.execute(ctx_for(conn, "u_sari"), "get_document_diff",
                              {"document": "SOP Diff Chat"})
    assert out.get("error") is None, out
    assert out["from"]["version"] == "1" and out["to"]["version"] == "2", out
    assert out["added_lines"] >= 1 and out["removed_lines"] >= 1, out
    assert "Baris dua diubah." in out["diff"], out["diff"]
    assert out["title"] == "SOP Diff Chat", out


def test_document_diff_requires_hr_and_known_title():
    conn = make_conn()
    denied = agent_tools.execute(ctx_for(conn, "u_budi"), "get_document_diff",
                                 {"document": "SOP Diff Chat"})
    assert denied["error"] == "tool_error" and denied.get("denied"), denied
    unknown = agent_tools.execute(ctx_for(conn, "u_sari"), "get_document_diff",
                                  {"document": "Dokumen Tidak Ada"})
    assert unknown["error"] == "tool_error", unknown


def test_document_diff_without_two_versions_is_explained():
    """Dokumen baru punya satu versi: jawab jelas, jangan lempar traceback."""
    conn = make_conn()
    _revise_doc(conn, "u_sari", "SOP Satu Versi", "Hanya satu versi.")
    out = agent_tools.execute(ctx_for(conn, "u_sari"), "get_document_diff",
                              {"document": "SOP Satu Versi"})
    assert out["error"] == "tool_error", out
    assert "satu versi" in out["message"], out


# ---------------------------------------------------------------------------
# Urutan daftar dokumen (dulu: 24 panggilan sampai step_limit)
# ---------------------------------------------------------------------------
def test_list_documents_sort_oldest_answers_staleness_question():
    conn = make_conn()
    ctx = ctx_for(conn, "u_sari")
    out = agent_tools.execute(ctx, "list_documents",
                              {"sort": "oldest", "limit": 5})
    assert out.get("error") is None, out
    assert out["sort"] == "oldest", out
    dates = [d["effective_date"] for d in out["documents"]]
    assert dates == sorted(dates), dates
    assert "effective_date" in out["sort_note"], out


def test_list_documents_sort_rejects_unknown_and_defaults_newest():
    conn = make_conn()
    ctx = ctx_for(conn, "u_sari")
    out = agent_tools.execute(ctx, "list_documents", {"sort": "acak"})
    assert out.get("error") is None, out
    assert out["sort"] == "newest", out
    dates = [d["effective_date"] for d in out["documents"]]
    assert dates == sorted(dates, reverse=True), dates


# ---------------------------------------------------------------------------
# Daftar karyawan menurut status (dulu: jawaban "tidak ada daftarnya")
# ---------------------------------------------------------------------------
def test_list_employees_filters_by_status_and_role():
    conn = make_conn()
    ctx = ctx_for(conn, "u_rina")
    agent_tools.execute(ctx, "prepare_employee_change",
                        {"action": "update", "employee": "Budi",
                         "employment_status": "probation"})
    token = ctx.prepared["token"]
    assert agent_tools.execute(ctx, "submit_prepared",
                               {"confirm_token": token}).get("ok")

    out = agent_tools.execute(ctx_for(conn, "u_rina"), "list_employees",
                              {"status": "probation"})
    assert out.get("error") is None, out
    names = [p["name"] for p in out["employees"]]
    assert names == ["Budi Santoso"], out
    assert out["by_status"]["probation"] == 1, out

    leads = agent_tools.execute(ctx_for(conn, "u_rina"), "list_employees",
                                {"role": "team_lead"})
    assert [p["name"] for p in leads["employees"]] == ["Andi Pratama"], leads


def test_list_employees_is_hr_only_and_validates_filters():
    conn = make_conn()
    denied = agent_tools.execute(ctx_for(conn, "u_budi"), "list_employees", {})
    assert denied["error"] == "tool_error" and denied.get("denied"), denied
    bad = agent_tools.execute(ctx_for(conn, "u_sari"), "list_employees",
                              {"status": "kontrak"})
    assert bad["error"] == "tool_error" and "Status tidak dikenal" in bad["message"], bad


def test_employee_status_enum_matches_validator():
    """Enum skema harus sama dengan yang diterima validator.

    Bug nyata: skema menawarkan `probation`, tetapi `users_admin.STATUSES`
    tidak memuatnya sehingga draft selalu gagal saat disubmit.
    """
    import users_admin
    import hr_actions
    assert users_admin.STATUSES == hr_actions.EMPLOYMENT_STATUSES


# ---------------------------------------------------------------------------
# Sisa cuti bawahan langsung
# ---------------------------------------------------------------------------
def test_team_leave_balance_for_direct_report_only():
    conn = make_conn()
    andi = ctx_for(conn, "u_andi")
    out = agent_tools.execute(andi, "get_team_leave_balance", {"employee": "Maya"})
    assert out.get("error") is None, out
    assert out["employee"]["name"] == "Maya Anggraini", out
    assert "remaining" in out and "before_pending" in out, out

    # Rina bukan bawahan Andi -> ditolak, bukan "tidak ditemukan"
    other = agent_tools.execute(andi, "get_team_leave_balance", {"employee": "Rina"})
    assert other["error"] == "tool_error", other
    assert other.get("denied") is True, other


def test_team_leave_balance_role_boundaries():
    conn = make_conn()
    emp = agent_tools.execute(ctx_for(conn, "u_budi"),
                              "get_team_leave_balance", {"employee": "Maya"})
    assert emp["error"] == "tool_error" and emp.get("denied"), emp

    # HR punya cakupan lebih luas lewat tool lain; tool ini mengarahkannya.
    hr = agent_tools.execute(ctx_for(conn, "u_sari"),
                             "get_team_leave_balance", {"employee": "Maya"})
    assert hr["error"] == "tool_error", hr
    assert "get_employee_leave" in hr["message"], hr


# ---------------------------------------------------------------------------
# Bacaan operasional admin: penolakan harus sesuai peran nyata, bukan tebakan
# ---------------------------------------------------------------------------
def test_admin_can_read_kpi_and_backups_from_chat():
    """Dulu model menjawab "tidak memiliki informasi" / "peran tidak mencakup"
    untuk KPI dan daftar backup, padahal endpoint API-nya terbuka untuk peran
    itu. Agent jadi tampak tidak tahu kemampuannya sendiri."""
    conn = make_conn()
    ctx = agent_tools.ToolContext(conn, engine.get_user(conn, "u_rina"))
    kpi = agent_tools.execute(ctx, "get_kpi", {})
    assert "error" not in kpi, kpi
    assert "data" in kpi and "runtime" in kpi, kpi

    b = agent_tools.execute(ctx, "list_backups", {})
    assert "error" not in b, b
    assert isinstance(b.get("backups"), list), b
    # Path absolut tidak boleh bocor lewat tool.
    assert "/" not in json.dumps(b.get("backups") or []), b


def test_hr_can_read_kpi_but_not_backup_list():
    conn = make_conn()
    ctx = agent_tools.ToolContext(conn, engine.get_user(conn, "u_sari"))
    assert "error" not in agent_tools.execute(ctx, "get_kpi", {})
    denied = agent_tools.execute(ctx, "list_backups", {})
    assert denied.get("error") == "tool_error", denied


def test_employee_cannot_read_kpi_or_backups():
    conn = make_conn()
    ctx = agent_tools.ToolContext(conn, engine.get_user(conn, "u_budi"))
    assert agent_tools.execute(ctx, "get_kpi", {}).get("error") == "tool_error"
    assert agent_tools.execute(ctx, "list_backups", {}).get("error") == "tool_error"


def test_company_requests_returns_server_computed_total():
    """Total nominal harus dihitung server dari SELURUH hasil filter.

    Dulu `get_company_requests` hanya mengirim 25 baris tanpa agregat, sehingga
    "berapa total reimbursement disetujui bulan ini" dijawab "daftarnya
    terpotong, saya belum bisa memberi total" — padahal datanya lengkap.
    `amount` tersimpan sebagai TEXT, jadi penjumlahan naif menghasilkan None.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_budi")
    for amount in ("150000", "250000"):
        out, status = portal.create_request(conn, user, {
            "type": "reimbursement", "title": "Taksi", "amount": amount,
            "note": "transport"})
        assert status < 400, out
        rid = out["id"]
        conn.execute("UPDATE requests SET status='approved' WHERE id=?", (rid,))
    conn.commit()
    ctx = agent_tools.ToolContext(conn, engine.get_user(conn, "u_sari"))
    r = agent_tools.execute(ctx, "get_company_requests",
                            {"status": "approved", "type": "reimbursement"})
    assert "error" not in r, r
    assert r["count"] == 2, r
    assert r["amount_total"] == 400000, r
    assert isinstance(r["amount_total"], int) and r["amount_total"] > 0, r
    assert r["amount_count"] == r["count"], r
    assert r["note"], "model butuh penjelasan arti amount_total"


def test_company_requests_total_none_for_leave_requests():
    """Cuti tidak punya nominal: jangan laporkan total 0 yang menyesatkan."""
    conn = make_conn()
    ctx = agent_tools.ToolContext(conn, engine.get_user(conn, "u_sari"))
    r = agent_tools.execute(ctx, "get_company_requests",
                            {"status": "approved", "type": "cuti"})
    assert r["amount_total"] is None, r
    assert r["amount_count"] == 0, r


def test_attendance_ships_ready_made_late_text():
    """Keterlambatan dihitung server, bukan model.

    Terukur: model menulis "terlambat 5 menit" dari 09:05 vs 09:00, lalu
    gerbang `ungrounded_quantity` menolak seluruh jawaban karena angka itu
    tidak ada di payload. Sekarang payload membawa `late_minutes_text`.
    """
    conn = make_conn()
    out = agent_tools.execute(ctx_for(conn), "get_attendance", {})
    rec = out["today_record"]
    if not rec or not rec.get("clock_in"):
        pytest.skip("seed tidak punya catatan absensi hari ini")
    late = out["late_minutes_text"]
    if late:
        assert late.endswith("menit")
        assert late.split()[0].isdigit()
        assert late in rec["late_note"]
    assert "late_minutes_text" in rec
    assert "Jangan menghitung" in out["note"]


def test_late_minutes_text_is_zero_safe():
    assert agent_tools._late_minutes_text("08:45", "09:00") is None
    assert agent_tools._late_minutes_text(None, "09:00") is None
    assert agent_tools._late_minutes_text("09:05", "09:00") == "5 menit"


def test_admin_payroll_answer_survives_salary_gate():
    """Struktur gaji dari TOOL sah untuk HR/admin.

    Bug nyata: `invented_salary` hanya memeriksa dokumen, sehingga jawaban
    admin yang menyalin `salary_structure` dari `get_payroll` dibuang dan
    jatuh ke jalur deterministik ("hubungi Finance") padahal datanya sudah
    diambil dan perannya berhak.
    """
    conn = make_conn()
    ctx = ctx_for(conn, "u_rina")
    out = agent_tools.execute(ctx, "get_payroll", {})
    assert out["salary_structure"], "admin harus menerima struktur gaji"

    text = "Struktur gaji 2026: Grade 2 (staff) Rp8.000.000–Rp12.000.000."
    fixed, err = agent._validate_answer(text, ctx, "Tampilkan payroll bulan ini")
    assert err is None, err
    assert fixed

    # Tanpa tool dan tanpa dokumen, klaim grade tetap ditolak.
    ctx2 = ctx_for(conn, "u_budi")
    _, err2 = agent._validate_answer(text, ctx2, "Tampilkan payroll bulan ini")
    # Urutan gerbang bisa mendahului (klaim tanpa bukti) — yang penting
    # jawaban itu TIDAK lolos untuk karyawan biasa.
    assert err2 in ("invented_salary", "ungrounded_claim"), err2


def _make_requests(conn, uid="u_budi", n=3, type_="reimbursement"):
    """Buat pengajuan nyata; seed sengaja tidak menyertakan data pengajuan."""
    user = engine.get_user(conn, uid)
    made = []
    for i in range(n):
        payload = {"type": type_, "title": f"Klaim uji {i}"}
        if type_ == "reimbursement":
            payload["amount"] = str(100000 * (i + 1))
        else:
            payload["start_date"] = "2026-12-01"
            payload["end_date"] = "2026-12-02"
        out, code = portal.create_request(conn, user, payload)
        assert code < 400, out
        made.append(out)
    return made


def test_company_requests_filter_by_month():
    """Filter bulan harus satu panggilan, bukan diulang per status.

    Terukur sebelum fix: model memanggil `get_company_requests` 17 kali dengan
    kombinasi status/jenis, lalu menjawab "belum punya parameter filter bulan".
    Datanya ada; parameternya yang tidak.
    """
    conn = make_conn()
    _make_requests(conn, n=2)
    ctx = ctx_for(conn, "u_sari")
    this_month = agent_tools.execute(ctx, "get_company_requests", {})["requests"][0]
    month = str(this_month["created_at"])[:7]

    out = agent_tools.execute(ctx_for(conn, "u_sari"), "get_company_requests",
                              {"month": month})
    assert out["month"] == month
    assert out["count"] >= 1
    for r in out["requests"]:
        stamp = str(r.get("created_at") or r.get("start_date") or "")
        assert stamp[:7] == month, r

    # Bulan tanpa data mengembalikan nol, bukan seluruh isi tabel.
    empty = agent_tools.execute(ctx_for(conn, "u_sari"), "get_company_requests",
                                {"month": "1999-01"})
    assert empty["count"] == 0
    assert empty["amount_total"] is None


def test_my_requests_returns_server_computed_total():
    """Total milik user dihitung dari SELURUH baris, bukan yang tampil.

    Terukur sebelum fix: jawaban berbunyi "daftarnya terpotong, angka total
    belum final" padahal server memegang semua baris.
    """
    conn = make_conn()
    _make_requests(conn, n=3)
    out = agent_tools.execute(ctx_for(conn), "get_my_requests",
                              {"type": "reimbursement"})
    assert out["count"] == 3
    assert isinstance(out["amount_total"], int)
    # 100k + 200k + 300k
    assert out["amount_total"] == 600000
    assert out["amount_count"] == 3
    assert out["shown"] <= 15
    assert "amount_total" in out["note"]


def test_my_requests_month_scopes_the_total():
    conn = make_conn()
    _make_requests(conn, n=2)
    all_out = agent_tools.execute(ctx_for(conn), "get_my_requests",
                                  {"type": "reimbursement"})
    assert all_out["count"] == 2
    month = str(all_out["requests"][0]["created_at"])[:7]

    scoped = agent_tools.execute(ctx_for(conn), "get_my_requests",
                                 {"type": "reimbursement", "month": month})
    assert scoped["count"] == 2
    assert scoped["amount_total"] == all_out["amount_total"]

    # Bulan mustahil: nol dan tanpa total, bukan seluruh data.
    none_out = agent_tools.execute(ctx_for(conn), "get_my_requests",
                                   {"month": "1999-01"})
    assert none_out["count"] == 0
    assert none_out["amount_total"] is None


def test_export_is_never_claimed_as_a_sent_file():
    """Chat tidak bisa mengirim berkas; prompt harus melarang klaim palsu.

    Terukur: model menulis "Berikut ekspor CSV" lalu menempel tabel dan
    menyebutnya file CSV. Itu menyesatkan user.
    """
    import personal
    user = engine.get_user(make_conn(), "u_budi")
    prompt = agent._system_prompt(user, {"name": "Budi Santoso"}, "2026-11-02")
    assert "Ekspor CSV/PDF TIDAK tersedia dari chat" in prompt
    assert 'jangan menulis "berikut ekspor csv"' in prompt.lower()


def test_clock_needs_a_command_not_an_intent():
    """Niat pulang bukan perintah mencatat absensi.

    Terukur: "Saya mau pulang sekarang, sudah tidak enak badan" memanggil
    `clock_attendance` dan mencatat clock-out. Itu mengubah data kehadiran
    atas dasar keluhan, bukan permintaan.
    """
    conn = make_conn()
    ctx = agent_tools.ToolContext(conn, engine.get_user(conn, "u_maya"),
                                  question="Saya mau pulang sekarang, sudah tidak enak badan")
    out = agent_tools.execute(ctx, "clock_attendance", {"action": "out"})
    assert out.get("error") == "tool_error"
    assert "niat" in (out.get("message") or "").lower()

    # Kalimat yang sama, tetapi dengan perintah eksplisit, tetap lolos gerbang
    # niat (pesan error yang tersisa hanya soal state absensi, bukan gerbang).
    ctx2 = agent_tools.ToolContext(conn, engine.get_user(conn, "u_maya"),
                                   question="Tolong clock-out sekarang")
    out2 = agent_tools.execute(ctx2, "clock_attendance", {"action": "out"})
    assert "niat" not in (out2.get("message") or "").lower(), out2


def test_clock_without_question_is_still_allowed():
    """Jalur tombol UI tidak punya teks pertanyaan; jangan ikut terblokir."""
    conn = make_conn()
    ctx = agent_tools.ToolContext(conn, engine.get_user(conn, "u_maya"))
    out = agent_tools.execute(ctx, "clock_attendance", {"action": "in"})
    assert "error" not in out, out


# ---------------------------------------------------------------------------
# Gerbang "niat vs perintah" untuk aksi tulis yang langsung berefek
# ---------------------------------------------------------------------------
def test_notification_mark_read_needs_a_command():
    """Kalimat musyawarah tidak boleh menandai notifikasi terbaca.

    Terukur di :8890: "Mungkin notifikasinya perlu ditandai terbaca" menandai
    11 notifikasi menjadi terbaca tanpa user pernah memerintahkan.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_maya")
    portal._notify(conn, [user["id"]], "Pesan uji", category="system")
    conn.commit()
    before = portal.unread_count(conn, user["id"])
    assert before >= 1

    for q in ("Mungkin notifikasinya perlu ditandai terbaca",
              "Kayaknya notifikasi saya sudah kebanyakan",
              "Kalau bisa notifikasinya jangan menumpuk"):
        ctx = agent_tools.ToolContext(conn, user, question=q)
        out = agent_tools.execute(ctx, "mark_notifications_read", {"category": "all"})
        assert out.get("error") == "tool_error", (q, out)
        assert "belum memerintahkan" in (out.get("message") or "")
        assert portal.unread_count(conn, user["id"]) == before, q

    # Perintah eksplisit tetap jalan.
    ctx = agent_tools.ToolContext(conn, user, question="Tandai semua notifikasi sudah dibaca")
    out = agent_tools.execute(ctx, "mark_notifications_read", {"category": "all"})
    assert out.get("ok") is True, out
    assert portal.unread_count(conn, user["id"]) == 0


def test_notification_mark_read_without_question_still_runs():
    """Jalur tombol UI tidak punya teks user; jangan ikut terblokir."""
    conn = make_conn()
    user = engine.get_user(conn, "u_maya")
    portal._notify(conn, [user["id"]], "Pesan uji", category="system")
    conn.commit()
    ctx = agent_tools.ToolContext(conn, user)
    out = agent_tools.execute(ctx, "mark_notifications_read", {"category": "all"})
    assert out.get("ok") is True, out
    assert portal.unread_count(conn, user["id"]) == 0


def test_announcements_mark_read_needs_a_command():
    conn = make_conn()
    user = engine.get_user(conn, "u_maya")
    ctx = agent_tools.ToolContext(conn, user, question="Sepertinya pengumumannya perlu dibaca semua")
    out = agent_tools.execute(ctx, "mark_announcements_read", {})
    assert out.get("error") == "tool_error", out

    ctx2 = agent_tools.ToolContext(conn, user, question="Tandai semua pengumuman sudah dibaca")
    out2 = agent_tools.execute(ctx2, "mark_announcements_read", {})
    assert "error" not in out2, out2


def test_notif_pref_needs_a_command():
    """Niat mengubah preferensi bukan perintah mengubahnya."""
    conn = make_conn()
    user = engine.get_user(conn, "u_maya")

    def enabled(cat):
        rows = portal.notif_prefs(conn, user)["categories"]
        return next(r["enabled"] for r in rows if r["category"] == cat)

    was = enabled("approval")
    ctx = agent_tools.ToolContext(
        conn, user, question="Kayaknya notifikasi approval ganggu, mungkin dimatikan")
    out = agent_tools.execute(ctx, "set_notif_pref",
                              {"category": "approval", "enabled": False})
    assert out.get("error") == "tool_error", out
    assert enabled("approval") == was, "preferensi berubah tanpa perintah"

    ctx2 = agent_tools.ToolContext(conn, user, question="Tolong matikan notifikasi approval")
    out2 = agent_tools.execute(ctx2, "set_notif_pref",
                               {"category": "approval", "enabled": False})
    assert "error" not in out2, out2
    assert enabled("approval") is False


def test_notifications_payload_exposes_duplicate_count():
    """Model boleh meringkas "muncul N kali" tanpa ditolak gerbang kuantitas.

    Terukur di :8890: pengumuman yang sama masuk 3 baris, model menulis
    "muncul 3x", dan angka 3 tidak ada di payload — jawaban yang benar dibuang
    `ungrounded_quantity` lalu user menerima dokumen reimbursement.
    """
    conn = make_conn()
    user = engine.get_user(conn, "u_maya")
    for _ in range(3):
        portal._notify(conn, [user["id"]], "Maintenance Sistem", category="system")
    portal._notify(conn, [user["id"]], "Slip gaji terbit", category="payslip")
    conn.commit()

    ctx = agent_tools.ToolContext(conn, user)
    out = agent_tools.execute(ctx, "get_notifications", {"limit": 10})
    assert out["shown"] == len(out["items"])
    by_msg = {}
    for it in out["items"]:
        by_msg.setdefault(it["message"], set()).add(it["same_message_count"])
    assert by_msg["Maintenance Sistem"] == {3}, by_msg
    assert by_msg["Slip gaji terbit"] == {1}, by_msg

    # Angka 3 kini ada di korpus tool, jadi ringkasan model tidak lagi "karangan".
    corpus = "\n".join(ctx.tool_text)
    assert agent._ungrounded_quantities("Pengumuman itu muncul 3 kali.", corpus) == set()


if __name__ == "__main__":
    test_every_schema_has_implementation()
    test_dashboard_returns_own_request_stats()
    print("OK")
