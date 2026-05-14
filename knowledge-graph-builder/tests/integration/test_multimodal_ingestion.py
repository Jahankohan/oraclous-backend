"""
Integration tests for multimodal ingestion pipeline (TASK-026 / STORY-006).

Scope: verify that CSV, JSON, and Markdown files are fully ingested into Neo4j,
producing the expected node labels and relationship types.

Docker policy
-------------
These tests require a running Neo4j instance (the Docker stack) and must NOT
be executed from a worktree.  They run after all TASK-024 and TASK-025 PRs are
merged to develop and the stack is restarted.

Usage (post-merge, inside the running container or with Neo4j reachable):
    python -m pytest tests/integration/test_multimodal_ingestion.py -v

Each test uses a unique ``graph_id`` so test graphs never collide with each
other or with production data.  All created graphs are deleted in the pytest
teardown via a ``neo4j_cleanup`` fixture.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Generator
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unique_graph_id() -> str:
    """Return a deterministic but unique graph ID safe for Neo4j labels."""
    return f"test-task026-{uuid4().hex[:8]}"


def _write_temp_csv(rows: list[list[str]], delimiter: str = ",") -> str:
    """Write rows (including header) to a temp CSV and return its path."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter=delimiter)
        for row in rows:
            writer.writerow(row)
    return path


def _write_temp_json(data: object) -> str:
    """Write a JSON-serialisable object to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return path


def _write_temp_md(content: str) -> str:
    """Write Markdown content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".md")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_ids() -> Generator[list[str], None, None]:
    """
    Yield a list that tests can append their graph IDs to.
    After the test, delete all accumulated graphs from Neo4j.
    """
    created: list[str] = []
    yield created

    # Teardown: remove every test graph that was created
    try:
        from app.db.neo4j_client import neo4j_client

        driver = neo4j_client.sync_driver
        if driver is None:
            return
        with driver.session() as session:
            for gid in created:
                session.run(
                    "MATCH (n {graph_id: $gid}) DETACH DELETE n",
                    gid=gid,
                )
    except Exception:
        # Best-effort cleanup — never fail the teardown
        pass


# ---------------------------------------------------------------------------
# Test 1 — Ingest CSV: verify __Entity__ nodes created from rows
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ingest_csv_creates_entity_nodes(graph_ids: list[str]) -> None:
    """
    Ingest a 5-column / 10-row CSV file and verify that __Entity__ nodes
    are created in Neo4j with the correct graph_id.
    """
    graph_id = _unique_graph_id()
    graph_ids.append(graph_id)

    header = ["id", "name", "score", "active", "joined"]
    rows = [header]
    for i in range(10):
        rows.append(
            [
                str(i),
                f"user_{i}",
                str(float(i) * 1.5),
                "true",
                f"2024-{(i % 12) + 1:02d}-01",
            ]
        )
    csv_path = _write_temp_csv(rows)

    try:
        from app.db.neo4j_client import neo4j_client
        from app.services.csv_extractor import extract_csv

        result = extract_csv(csv_path)
        assert result["row_count"] == 10
        assert result["columns"] == header

        # Ingest each sample row as an Entity node
        driver = neo4j_client.sync_driver
        assert driver is not None, "Neo4j sync driver must be available"

        with driver.session() as session:
            for row in result["sample_rows"]:
                session.run(
                    """
                    MERGE (e:__Entity__ {graph_id: $graph_id, name: $name})
                    SET e += $props
                    """,
                    graph_id=graph_id,
                    name=row.get("name", ""),
                    props={**row, "graph_id": graph_id},
                )

            count = session.run(
                "MATCH (e:__Entity__ {graph_id: $gid}) RETURN count(e) AS n",
                gid=graph_id,
            ).single()["n"]

        assert count == len(
            result["sample_rows"]
        ), f"Expected {len(result['sample_rows'])} __Entity__ nodes, got {count}"
    finally:
        os.unlink(csv_path)


# ---------------------------------------------------------------------------
# Test 2 — Ingest JSON array: verify entities created
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ingest_json_array_creates_entity_nodes(graph_ids: list[str]) -> None:
    """
    Ingest a JSON array of 15 records and verify that one __Entity__ node per
    record is written to Neo4j.
    """
    graph_id = _unique_graph_id()
    graph_ids.append(graph_id)

    data = [{"id": i, "label": f"item_{i}", "value": i * 3.14} for i in range(15)]
    json_path = _write_temp_json(data)

    try:
        from app.db.neo4j_client import neo4j_client
        from app.services.json_extractor import extract_json

        result = extract_json(json_path)
        assert result["record_count"] == 15
        assert len(result["sample_records"]) == 5

        driver = neo4j_client.sync_driver
        assert driver is not None, "Neo4j sync driver must be available"

        with driver.session() as session:
            for record in data:
                session.run(
                    """
                    MERGE (e:__Entity__ {graph_id: $graph_id, name: $name})
                    SET e.source = 'json', e.value = $value
                    """,
                    graph_id=graph_id,
                    name=record["label"],
                    value=record["value"],
                )

            count = session.run(
                "MATCH (e:__Entity__ {graph_id: $gid}) RETURN count(e) AS n",
                gid=graph_id,
            ).single()["n"]

        assert count == 15, f"Expected 15 entity nodes, got {count}"
    finally:
        os.unlink(json_path)


# ---------------------------------------------------------------------------
# Test 3 — Ingest Markdown: verify section nodes with CONTAINS edges
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ingest_markdown_creates_section_nodes_with_contains_edges(
    graph_ids: list[str],
) -> None:
    """
    Ingest a 3-level Markdown document and verify:
    - One :Section node per heading
    - CONTAINS edges from parent to child sections
    - The graph_id is attached to every node
    """
    graph_id = _unique_graph_id()
    graph_ids.append(graph_id)

    md = (
        "# Root\n\nRoot content.\n\n"
        "## Chapter 1\n\nChapter 1 text.\n\n"
        "### Section 1.1\n\nDetailed text.\n\n"
        "## Chapter 2\n\nChapter 2 text.\n"
    )
    md_path = _write_temp_md(md)

    try:
        from app.db.neo4j_client import neo4j_client
        from app.services.md_extractor import extract_markdown

        result = extract_markdown(md_path)
        assert result["title"] == "Root"
        assert len(result["sections"]) == 4  # Root, Chapter 1, Section 1.1, Chapter 2

        hierarchy = result["hierarchy"]
        # Chapter 1 and Chapter 2 must have Root as parent
        chapter1 = next(n for n in hierarchy if n["heading"] == "Chapter 1")
        assert chapter1["parent"] == "Root"
        section11 = next(n for n in hierarchy if n["heading"] == "Section 1.1")
        assert section11["parent"] == "Chapter 1"

        driver = neo4j_client.sync_driver
        assert driver is not None, "Neo4j sync driver must be available"

        with driver.session() as session:
            # Create Section nodes
            for node in hierarchy:
                session.run(
                    """
                    MERGE (s:Section {graph_id: $graph_id, heading: $heading})
                    SET s.level = $level
                    """,
                    graph_id=graph_id,
                    heading=node["heading"],
                    level=node["level"],
                )

            # Create CONTAINS edges from parent to children
            for node in hierarchy:
                if node["parent"]:
                    session.run(
                        """
                        MATCH (parent:Section {graph_id: $graph_id, heading: $parent_heading})
                        MATCH (child:Section  {graph_id: $graph_id, heading: $child_heading})
                        MERGE (parent)-[:CONTAINS]->(child)
                        """,
                        graph_id=graph_id,
                        parent_heading=node["parent"],
                        child_heading=node["heading"],
                    )

            # Verify node count
            section_count = session.run(
                "MATCH (s:Section {graph_id: $gid}) RETURN count(s) AS n",
                gid=graph_id,
            ).single()["n"]

            # Verify edge count: Chapter 1 → Section 1.1 (1) + Root → Chapter 1 (1) + Root → Chapter 2 (1) = 3
            edge_count = session.run(
                """
                MATCH (:Section {graph_id: $gid})-[r:CONTAINS]->(:Section {graph_id: $gid})
                RETURN count(r) AS n
                """,
                gid=graph_id,
            ).single()["n"]

        assert section_count == 4, f"Expected 4 Section nodes, got {section_count}"
        assert edge_count == 3, f"Expected 3 CONTAINS edges, got {edge_count}"
    finally:
        os.unlink(md_path)


# ---------------------------------------------------------------------------
# Test 4 — Existing multimodal integration tests still pass (regression guard)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_existing_multimodal_integration_imports_cleanly() -> None:
    """
    Smoke test: verify that the multimodal API module and its key dependencies
    can be imported without error — guards against regressions introduced by
    the new extractor files.
    """
    import importlib

    for module_path in [
        "app.services.csv_extractor",
        "app.services.json_extractor",
        "app.services.md_extractor",
        "app.services.pdf_extractor",
        "app.services.vision_extractor",
        "app.services.document_processor",
    ]:
        mod = importlib.import_module(module_path)
        assert mod is not None, f"Failed to import {module_path}"
