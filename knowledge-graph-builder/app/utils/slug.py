"""Slug generation for organization subdomains.

A slug is the URL-safe label used as an organization's subdomain
(``<slug>.oraclous.com``). Slugs are lowercase, DNS-label-safe, unique, and
generated from the organization name at creation time.

NOTE: the Alembic migration that backfills slugs intentionally inlines a
frozen copy of this logic — it must NOT import this module, so that the
migration stays reproducible even if these rules change later.
"""

import re
import unicodedata
from collections.abc import Callable

# Subdomain labels that must never be assigned to an organization — they
# collide with platform hosts/paths or are otherwise reserved.
RESERVED_SLUGS: frozenset[str] = frozenset(
    {
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
)

# DNS labels are capped at 63 characters.
MAX_SLUG_LENGTH = 63


def slugify(name: str) -> str:
    """Convert an arbitrary name into a DNS-label-safe slug.

    Lowercases, strips accents, replaces every run of non ``[a-z0-9]``
    characters with a single hyphen, trims leading/trailing hyphens, and caps
    the length at 63. Returns ``"org"`` when the input reduces to nothing.
    """
    normalized = unicodedata.normalize("NFKD", name)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    hyphenated = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower())
    trimmed = hyphenated.strip("-")[:MAX_SLUG_LENGTH].strip("-")
    return trimmed or "org"


def is_reserved(slug: str) -> bool:
    """Return True if the slug is on the reserved list (case-insensitive)."""
    return slug.lower() in RESERVED_SLUGS


def generate_unique_slug(name: str, exists: Callable[[str], bool]) -> str:
    """Generate a unique, non-reserved slug from ``name``.

    ``exists`` is a predicate returning True when a candidate slug is already
    taken. A reserved or taken candidate gets a numeric suffix (``-2``, ``-3``,
    …) until a free one is found.
    """
    base = slugify(name)
    if not is_reserved(base) and not exists(base):
        return base
    # Reserve room for the "-N" suffix within the 63-char DNS limit.
    stem = base[: MAX_SLUG_LENGTH - 5].rstrip("-") or "org"
    suffix = 2
    while True:
        candidate = f"{stem}-{suffix}"
        if not is_reserved(candidate) and not exists(candidate):
            return candidate
        suffix += 1
