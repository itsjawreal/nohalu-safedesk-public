"""rbac.py — satu sumber kebenaran untuk siapa boleh melakukan apa.

Sebelum ini aturan peran tersebar: `require_admin`/`require_hr` di app.py,
`actor["role"] in (...)` di personal.py, `portal.py`, `users_admin.py`, dan
`engine.py`. Akibatnya menambah peran baru berarti mencari semua tempat itu,
dan satu tempat yang terlewat menjadi celah izin.

Modul ini hanya memetakan **aksi → peran yang diizinkan**. Penegakan tetap di
depannya masing-masing (FastAPI dependency, fungsi domain), supaya import tidak
melingkar dan modul domain tetap bisa diuji tanpa HTTP.

Dipakai oleh:
  * `app.py` — dependency `require(...)` untuk rute baru;
  * `tests/test_rbac_matrix.py` — memastikan setiap rute ber-guard peran
    benar-benar memakai guard yang dideklarasikan di sini.
"""

ROLES = ("employee", "team_lead", "hr", "finance", "admin")

# aksi → peran yang boleh
PERMISSIONS = {
    # ---- dokumen & pengetahuan ----
    "docs.read": ("employee", "team_lead", "hr", "finance", "admin"),
    "docs.manage": ("hr", "admin"),
    "docs.delete": ("hr", "admin"),
    # ---- personal (selalu milik sendiri) ----
    "me.read": ("employee", "team_lead", "hr", "finance", "admin"),
    "me.clock": ("employee", "team_lead", "hr", "finance", "admin"),
    # ---- pengajuan ----
    "requests.create": ("employee", "team_lead", "hr", "finance", "admin"),
    "requests.read": ("employee", "team_lead", "hr", "finance", "admin"),
    "requests.approve": ("team_lead", "hr", "finance", "admin"),
    "requests.override": ("hr", "admin"),
    # ---- payroll ----
    "payroll.read": ("employee", "team_lead", "hr", "finance", "admin"),
    "payroll.manage": ("hr", "admin"),
    # ---- administrasi ----
    "users.manage": ("admin",),
    "audit.read": ("hr", "admin"),
    "backup.manage": ("admin",),
    "workspaces.manage": ("admin",),
    # ---- komunikasi ----
    "announcements.manage": ("hr", "admin"),
    "notifications.read": ("employee", "team_lead", "hr", "finance", "admin"),
    "chat.use": ("employee", "team_lead", "hr", "finance", "admin"),
}


def allowed(action: str, role: str) -> bool:
    """True bila `role` boleh melakukan `action`. Aksi tak dikenal = tolak."""
    return role in PERMISSIONS.get(action, ())


def roles_for(action: str) -> tuple:
    return PERMISSIONS.get(action, ())
