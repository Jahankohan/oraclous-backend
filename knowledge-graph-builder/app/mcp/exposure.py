"""The exposure allowlist guard — ADR-023 D6, enforced (TASK-232).

ADR-023 D6 makes exposure over MCP allowlist-governed: a capability is exposed
only when explicitly marked exposable, and administratively dangerous
operations — credential / key rotation, graph or data deletion, permission and
grant management, service-account management — are *excluded by default*.

The registry (`registry.py`) **is** the allowlist: the projection is
registry-driven, so adding a REST endpoint never auto-exposes it — a capability
reaches MCP only when someone deliberately adds a `CapabilitySpec`. That makes
the allowlist a convention.

This module makes D6 *enforced*, not merely conventional. It defines the set of
dangerous-operation patterns and `assert_safe_registry()` — a build-time
validator `build_mcp_server()` calls before projecting anything. If any
registered `CapabilitySpec` matches a dangerous pattern the build **fails
loudly** with `DangerousCapabilityError`, so a dangerous capability can never
be silently exposed even if a future edit adds one to the registry.

The rule, stated precisely:

    A capability is exposed over MCP iff it is explicitly in the registry
    (the allowlist) AND it does not match a dangerous pattern (the guard).

The two halves are complementary: the registry is the *allow* side (nothing is
exposed unless added), the guard is the *deny* side (a dangerous spec is
rejected even if added). Curation alone could let a reviewer slip; the guard
makes the dangerous classes structurally unreachable.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.mcp.registry import CapabilitySpec

logger = get_logger(__name__)


class DangerousCapabilityError(RuntimeError):
    """Raised at build time when the registry contains a dangerous capability.

    A hard failure by design: a dangerous operation must never be silently
    projected to an MCP tool. The error names the offending spec and the rule
    it broke so the fix is obvious — remove the spec, or, if the operation is
    genuinely safe, narrow the dangerous pattern with a reviewed change here.
    """


# --- the dangerous-operation patterns (ADR-023 D6) --------------------------
# Each entry is one administratively dangerous class. A `CapabilitySpec` is
# dangerous if it matches ANY pattern. The patterns are deliberately
# conservative — path *substrings* and HTTP methods — so a new endpoint in a
# dangerous family is caught without needing this list updated per endpoint.

# HTTP methods that destroy data. A DELETE-method capability is never exposed.
_DANGEROUS_METHODS: frozenset[str] = frozenset({"DELETE"})

# REST path substrings that mark a dangerous capability family. Matched
# case-insensitively against the spec's `path` (and its async-job status path).
#   * `/permissions`      — permission / grant management (ADR-023 D6).
#   * `/service-accounts` — service-account management (ADR-023 D6).
#   * `rotate` / `/keys`  — credential / key rotation (ADR-023 D6).
_DANGEROUS_PATH_SUBSTRINGS: tuple[str, ...] = (
    "/permissions",
    "/service-accounts",
    "/rotate",
    "/keys",
)

# MCP tool-name family prefixes that are dangerous regardless of path shape.
# The registry namespaces tools by family (ADR-024 D7-R); a tool named in one
# of these families is a grant- / account- / key-management primitive.
_DANGEROUS_NAME_PREFIXES: tuple[str, ...] = (
    "permission.",
    "permissions.",
    "service-account.",
    "service_account.",
    "key.",
    "keys.",
)


def _dangerous_reason(spec: CapabilitySpec) -> str | None:
    """Why `spec` is dangerous, or None if it is safe to expose.

    Checks, in order: a data-destroying HTTP method; a dangerous path
    substring (on the spec's path and, for async-job specs, its status path);
    a dangerous tool-name family prefix.
    """
    if spec.method.upper() in _DANGEROUS_METHODS:
        return f"method {spec.method.upper()} destroys data (DELETE is excluded by D6)"

    if spec.status_method.upper() in _DANGEROUS_METHODS:
        return (
            f"status_method {spec.status_method.upper()} destroys data "
            "(DELETE is excluded by D6)"
        )

    paths = [spec.path]
    if spec.status_path:
        paths.append(spec.status_path)
    for path in paths:
        lowered = path.lower()
        for substring in _DANGEROUS_PATH_SUBSTRINGS:
            if substring in lowered:
                return (
                    f"path {path!r} matches dangerous pattern {substring!r} "
                    "(credential/grant/account management is excluded by D6)"
                )

    name_lower = spec.name.lower()
    for prefix in _DANGEROUS_NAME_PREFIXES:
        if name_lower.startswith(prefix):
            return (
                f"tool name {spec.name!r} is in the dangerous family "
                f"{prefix!r} (excluded by D6)"
            )

    return None


def is_dangerous(spec: CapabilitySpec) -> bool:
    """True if `spec` matches an ADR-023 D6 dangerous-operation pattern."""
    return _dangerous_reason(spec) is not None


def assert_safe_registry(registry: tuple[CapabilitySpec, ...]) -> None:
    """Fail loudly if `registry` exposes any dangerous capability (ADR-023 D6).

    Called by `build_mcp_server()` *before* projection. The registry is the
    allowlist; this guard enforces D6 — a capability may be exposed only if it
    is explicitly in the registry AND not dangerous. A dangerous spec raises
    `DangerousCapabilityError`, aborting the build, so a dangerous operation
    can never be silently projected to an MCP tool.
    """
    offenders: list[str] = []
    for spec in registry:
        reason = _dangerous_reason(spec)
        if reason is not None:
            offenders.append(f"  - {spec.name} ({spec.method} {spec.path}): {reason}")

    if offenders:
        raise DangerousCapabilityError(
            "MCP exposure allowlist (ADR-023 D6) violated — the registry "
            "contains administratively dangerous capabilities that must not be "
            "exposed over MCP:\n"
            + "\n".join(offenders)
            + "\n\nThe registry is the allowlist; this guard enforces D6. "
            "Remove the offending CapabilitySpec(s) — credential/key rotation, "
            "data deletion, permission/grant management and service-account "
            "management belong in a consumer, never on the generic surface."
        )

    logger.info(
        "MCP exposure guard: %d capabilities checked against the ADR-023 D6 "
        "dangerous-operation patterns — all safe to expose.",
        len(registry),
    )
