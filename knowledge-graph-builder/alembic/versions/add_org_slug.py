"""add slug and logo_url to organizations

Revision ID: add_org_slug
Revises: org_inv_subgraph_grants
Create Date: 2026-05-18 00:00:00.000000

Adds the organization subdomain ``slug`` (unique, indexed, NOT NULL) and an
optional ``logo_url``. ``slug`` is added nullable, backfilled from each org's
name, then flipped to NOT NULL — a NOT NULL column is never added directly to
a populated table.

The slugify logic below is an intentional frozen copy of ``app/utils/slug.py``
— migrations must stay reproducible even if that utility changes later.

Note: the revision id is kept short on purpose — alembic_version.version_num
is VARCHAR(32).
"""

import re
import unicodedata

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "add_org_slug"
down_revision = "org_inv_subgraph_grants"
branch_labels = None
depends_on = None

# Frozen copy of app/utils/slug.py — see module docstring.
_RESERVED = {
    "www",
    "api",
    "app",
    "admin",
    "mail",
    "smtp",
    "ftp",
    "auth",
    "oauth",
    "login",
    "logout",
    "register",
    "dashboard",
    "static",
    "assets",
    "cdn",
    "public",
    "docs",
    "doc",
    "status",
    "billing",
    "support",
    "help",
    "blog",
    "dev",
    "staging",
    "test",
    "internal",
    "system",
    "oraclous",
    "ns1",
    "ns2",
}
_MAX = 63


def _slugify(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name or "")
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    hyphenated = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower())
    return hyphenated.strip("-")[:_MAX].strip("-") or "org"


def upgrade():
    # Phase 1 — add both columns nullable (IF NOT EXISTS keeps it idempotent).
    op.execute("ALTER TABLE organizations ADD COLUMN IF NOT EXISTS slug VARCHAR(63)")
    op.execute(
        "ALTER TABLE organizations ADD COLUMN IF NOT EXISTS logo_url VARCHAR(512)"
    )

    # Phase 2 — backfill slug, oldest org first so the oldest keeps the bare
    # slug on a collision.
    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, name FROM organizations ORDER BY created_at")
    ).fetchall()
    used: set[str] = set()
    for row in rows:
        base = _slugify(row.name)
        candidate = base
        if candidate in _RESERVED or candidate in used:
            stem = base[: _MAX - 5].rstrip("-") or "org"
            n = 2
            while candidate in _RESERVED or candidate in used:
                candidate = f"{stem}-{n}"
                n += 1
        used.add(candidate)
        conn.execute(
            sa.text("UPDATE organizations SET slug = :s WHERE id = :id"),
            {"s": candidate, "id": row.id},
        )

    # Phase 3 — enforce NOT NULL + unique index. The unique index creation
    # fails loudly if the backfill somehow produced a duplicate.
    op.execute("ALTER TABLE organizations ALTER COLUMN slug SET NOT NULL")
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ix_organizations_slug "
        "ON organizations (slug)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_organizations_slug")
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS slug")
    op.execute("ALTER TABLE organizations DROP COLUMN IF EXISTS logo_url")
