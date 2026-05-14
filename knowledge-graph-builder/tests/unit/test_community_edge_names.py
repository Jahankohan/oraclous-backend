"""Pin community edge names so writer/reader can't drift again.

The bug this guards against: the sync ``_run_leiden_community_detection``
in ``analytics_service.py`` and the Celery detector in
``community_tasks.py`` used to write *different* relationship types
(``MEMBER_OF`` / ``PARENT_OF`` vs ``IN_COMMUNITY`` / ``PARENT_COMMUNITY``),
silently breaking every reader path on whichever subset of graphs the
sync detector wrote.

These tests grep the actual Cypher in those two writers and the
registry, asserting they all agree.
"""

import pytest


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


@pytest.mark.unit
def test_celery_detector_writes_in_community_and_parent_community():
    src = _read("app/tasks/community_tasks.py")
    # Member edge
    assert "[r:IN_COMMUNITY {graph_id: $graph_id, level: $level}]" in src, (
        "Celery detector must write IN_COMMUNITY edges with graph_id+level"
    )
    # Parent edge with direction (child)-[:PARENT_COMMUNITY]->(parent)
    assert "(child)-[:PARENT_COMMUNITY {graph_id: $graph_id}]->(parent)" in src, (
        "Celery detector must write PARENT_COMMUNITY from child to parent"
    )


@pytest.mark.unit
def test_sync_detector_writes_in_community_and_parent_community():
    src = _read("app/services/analytics_service.py")
    # The sync ``_run_leiden_community_detection`` path
    assert "[:IN_COMMUNITY {graph_id: $graph_id, level: $level}]" in src
    assert "(child)-[:PARENT_COMMUNITY {graph_id: $graph_id}]->(parent)" in src


@pytest.mark.unit
def test_no_legacy_member_of_or_parent_of_in_community_writers():
    """Reject any reintroduction of MEMBER_OF/PARENT_OF to __Community__.

    ``MEMBER_OF`` and ``PARENT_OF`` are used elsewhere — entity dedup in
    ``entity_resolver.py`` and user-team ReBAC in ``rebac_service.py``.
    Those are different features; they must never re-enter the community
    detection paths.
    """
    analytics = _read("app/services/analytics_service.py")
    tasks = _read("app/tasks/community_tasks.py")
    for src, label in ((analytics, "analytics_service"), (tasks, "community_tasks")):
        assert ":MEMBER_OF" not in src.replace("[:MEMBER_OF]", ""), (
            f"{label} contains a community-context :MEMBER_OF edge — "
            "use IN_COMMUNITY instead"
        )
        assert ":PARENT_OF" not in src, (
            f"{label} contains a :PARENT_OF edge — use PARENT_COMMUNITY instead"
        )


@pytest.mark.unit
def test_registry_member_rel_matches_writers():
    """The kind registry must agree with the writer/reader convention."""
    from app.schemas.community_kinds import get_kind

    assert get_kind("entity").member_rel == "IN_COMMUNITY"
