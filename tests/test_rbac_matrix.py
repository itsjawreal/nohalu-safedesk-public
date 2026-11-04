"""Sprint 5.2 — matriks izin sebagai satu sumber + penjagaan menyeluruh.

Dua hal yang dikunci:

1. **Tidak ada rute API yang lupa dijaga.** Rute publik harus tepat daftar
   yang memang publik (health, status LLM, login, logout). Rute lain wajib
   memakai dependency auth. Ini menangkap rute baru yang lupa diberi guard —
   kelas bug yang tidak akan terlihat sampai ada yang memindainya.

2. **Guard peran membaca `rbac.PERMISSIONS`.** Kalau `require_hr` dan matriks
   berbeda pendapat, tes gagal di sini, bukan di produksi.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

os.environ["NOHALU_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "rbac.db")
os.environ["NOHALU_LLM_ENABLED"] = "0"
os.environ["NOHALU_REMINDERS"] = "0"
os.environ["NOHALU_BACKUP_ENABLED"] = "0"

from fastapi.routing import APIRoute  # noqa: E402
import app as appmod  # noqa: E402
import rbac  # noqa: E402

# `/api/public/stats` sengaja publik: landing memanggilnya sebelum login.
# Isinya hanya agregat (tanpa id/nama/judul/isi dokumen) dan dibatasi
# rate-limit per IP — lihat tests/test_public_stats.py.
PUBLIC = {"/api/health", "/api/llm/status", "/api/auth/login",
          "/api/auth/logout", "/api/public/stats"}
AUTH_DEPS = {"parse_token", "require_admin", "require_hr"}


def _api_routes():
    return [r for r in appmod.app.routes
            if isinstance(r, APIRoute) and r.path.startswith("/api")]


def _dep_names(route):
    return {d.call.__name__ for d in route.dependant.dependencies}


def test_every_api_route_is_guarded():
    unguarded = []
    for r in _api_routes():
        if r.path in PUBLIC:
            continue
        if not (_dep_names(r) & AUTH_DEPS):
            unguarded.append((r.path, sorted(r.methods)))
    assert not unguarded, f"rute tanpa auth: {unguarded}"


def test_public_routes_are_exactly_the_expected_ones():
    public = {(r.path, m) for r in _api_routes() if not (_dep_names(r) & AUTH_DEPS)
              for m in r.methods}
    assert {p for p, _ in public} == PUBLIC, public


def test_matrix_is_complete_and_consistent():
    for action in rbac.PERMISSIONS:
        assert rbac.PERMISSIONS[action], action
        for role in rbac.PERMISSIONS[action]:
            assert role in rbac.ROLES, (action, role)
    # Aksi yang tidak dikenal harus ditolak, bukan default-allow.
    assert rbac.allowed("tidak.ada", "admin") is False


def test_admin_only_and_hr_guards_match_matrix():
    admin_roles = rbac.roles_for("users.manage")
    assert admin_roles == ("admin",), admin_roles
    hr_roles = rbac.roles_for("docs.manage")
    assert set(hr_roles) == {"hr", "admin"}, hr_roles


def test_employee_cannot_manage_docs_or_users():
    assert not rbac.allowed("docs.manage", "employee")
    assert not rbac.allowed("users.manage", "employee")
    assert not rbac.allowed("audit.read", "employee")
    # Tapi boleh membaca dokumen & memakai chat.
    assert rbac.allowed("docs.read", "employee")
    assert rbac.allowed("chat.use", "employee")


def test_team_lead_can_approve_but_not_manage_users():
    assert rbac.allowed("requests.approve", "team_lead")
    assert not rbac.allowed("users.manage", "team_lead")
    # Override lintas tahap tetap HR/admin.
    assert not rbac.allowed("requests.override", "team_lead")


def test_finance_can_approve_and_read_payroll_but_not_manage():
    assert rbac.allowed("requests.approve", "finance")
    assert rbac.allowed("payroll.read", "finance")
    assert not rbac.allowed("payroll.manage", "finance")
