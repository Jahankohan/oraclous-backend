"""
Unit tests for the ChatHistoryEntry schema (TASK-050).

These verify the typed shape returned by GET /api/v1/graphs/{graph_id}/chat/history
matches what the frontend's `ChatHistoryEntry` interface expects.
"""

from datetime import UTC, datetime

import pytest

from app.schemas.chat_schemas import ChatHistoryEntry, SourceInfo


class TestChatHistoryEntry:
    @pytest.mark.unit
    def test_user_turn_minimum_fields(self):
        """A user prompt only requires role + content."""
        entry = ChatHistoryEntry(role="user", content="Who is the CEO?")
        assert entry.role == "user"
        assert entry.content == "Who is the CEO?"
        assert entry.sources is None
        assert entry.created_at is None

    @pytest.mark.unit
    def test_assistant_turn_with_sources(self):
        """An assistant reply may carry a list of grounding sources."""
        sources = [
            SourceInfo(node_id="n-1", relevance_score=0.92, content="…"),
            SourceInfo(node_id="n-2", relevance_score=0.81, content="…"),
        ]
        entry = ChatHistoryEntry(
            role="assistant",
            content="The CEO is John Doe.",
            sources=sources,
            created_at=datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC),
        )
        assert entry.role == "assistant"
        assert entry.sources is not None
        assert len(entry.sources) == 2
        assert entry.sources[0].node_id == "n-1"
        assert entry.created_at is not None

    @pytest.mark.unit
    def test_serialises_to_frontend_shape(self):
        """JSON shape must match the frontend `ChatHistoryEntry` TypeScript interface:
        { role, content, sources?, created_at? }.
        """
        entry = ChatHistoryEntry(
            role="user",
            content="hi",
            sources=None,
            created_at=datetime(2026, 4, 28, 12, 0, 0, tzinfo=UTC),
        )
        data = entry.model_dump(mode="json")
        assert set(data.keys()) == {"role", "content", "sources", "created_at"}
        assert data["role"] == "user"
        assert data["content"] == "hi"
        assert data["sources"] is None
        # ISO 8601 string
        assert data["created_at"].startswith("2026-04-28T12:00:00")

    @pytest.mark.unit
    def test_role_required(self):
        """Role is a required field."""
        with pytest.raises(Exception):
            ChatHistoryEntry(content="missing role")  # type: ignore[call-arg]

    @pytest.mark.unit
    def test_content_required(self):
        """Content is a required field."""
        with pytest.raises(Exception):
            ChatHistoryEntry(role="user")  # type: ignore[call-arg]
