"""Smoke tests for the chat-history endpoints module (STORY-031 / TASK-104).

These cover the pure helpers + schema serialization. The endpoints
themselves are exercised against a real DB by the integration test in
``tests/integration/test_chat_rls_integration.py`` (which gains
endpoint coverage in a follow-up commit if needed).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints.chat_history import (
    _check_feedback_rating,
    _conv_to_summary,
    _msg_to_response,
    _parse_uuid_or_404,
)
from app.models.chat import (
    ChatConversation,
    ChatMessage,
    ChatMessageToolCall,
)
from app.schemas.chat_schemas import ConversationsListResponse


def _make_conv(**overrides) -> ChatConversation:
    base = dict(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        graph_id=uuid.uuid4(),
        agent_id=None,
        title="A title",
        created_at=datetime(2026, 5, 15, tzinfo=UTC),
        last_message_at=datetime(2026, 5, 15, tzinfo=UTC),
        deleted_at=None,
    )
    base.update(overrides)
    c = ChatConversation()
    for k, v in base.items():
        setattr(c, k, v)
    return c


def _make_msg(**overrides) -> ChatMessage:
    base = dict(
        id=uuid.uuid4(),
        conversation_id=uuid.uuid4(),
        session_id=None,
        role="assistant",
        content="Hi there.",
        created_at=datetime(2026, 5, 15, tzinfo=UTC),
        model="claude-opus-4-7",
        provider="anthropic",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=120,
        cost_usd=None,
        reasoning_mode="direct",
        retriever_used="hybrid",
        error=None,
        cancelled=False,
        sources=None,
        feedback_rating=None,
        feedback_comment=None,
        feedback_at=None,
    )
    base.update(overrides)
    m = ChatMessage()
    for k, v in base.items():
        setattr(m, k, v)
    return m


def _make_tool_call(**overrides) -> ChatMessageToolCall:
    base = dict(
        id=uuid.uuid4(),
        message_id=uuid.uuid4(),
        sequence_index=0,
        tool_name="graph_search",
        args_json={"q": "test"},
        result_summary="3 nodes",
        result_compressed=None,
        result_compression=None,
        result_uncompressed_size_bytes=None,
        result_truncated=False,
        latency_ms=42,
        error=None,
        started_at=datetime(2026, 5, 15, tzinfo=UTC),
    )
    base.update(overrides)
    t = ChatMessageToolCall()
    for k, v in base.items():
        setattr(t, k, v)
    return t


@pytest.mark.unit
class TestUuidParser:
    def test_valid_uuid_returns_uuid(self):
        u = uuid.uuid4()
        assert _parse_uuid_or_404(str(u), "thing") == u

    def test_malformed_raises_404(self):
        with pytest.raises(HTTPException) as exc:
            _parse_uuid_or_404("not-a-uuid", "thing")
        assert exc.value.status_code == 404
        assert "Unknown thing" in exc.value.detail

    def test_empty_raises_404(self):
        with pytest.raises(HTTPException) as exc:
            _parse_uuid_or_404("", "thing")
        assert exc.value.status_code == 404


@pytest.mark.unit
class TestFeedbackRatingCheck:
    @pytest.mark.parametrize("rating", [1, -1])
    def test_valid_values_pass(self, rating):
        assert _check_feedback_rating(rating) == rating

    @pytest.mark.parametrize("rating", [0, 2, -2, 100])
    def test_invalid_values_raise_400(self, rating):
        with pytest.raises(HTTPException) as exc:
            _check_feedback_rating(rating)
        assert exc.value.status_code == 400


@pytest.mark.unit
class TestConvToSummary:
    def test_includes_all_fields(self):
        conv = _make_conv(agent_id=uuid.uuid4())
        summary = _conv_to_summary(conv)
        assert summary.id == str(conv.id)
        assert summary.agent_id == str(conv.agent_id)
        assert summary.title == "A title"
        assert summary.created_at == conv.created_at
        assert summary.last_message_at == conv.last_message_at

    def test_null_agent_id_stays_null(self):
        conv = _make_conv(agent_id=None)
        summary = _conv_to_summary(conv)
        assert summary.agent_id is None


@pytest.mark.unit
class TestMsgToResponse:
    def test_assistant_with_no_tool_calls(self):
        msg = _make_msg()
        resp = _msg_to_response(msg, [])
        assert resp.role == "assistant"
        assert resp.model == "claude-opus-4-7"
        assert resp.tool_calls == []
        assert resp.cost_usd is None  # column is None on the model

    def test_assistant_with_tool_calls(self):
        msg = _make_msg()
        tc = _make_tool_call(message_id=msg.id)
        resp = _msg_to_response(msg, [tc])
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].tool_name == "graph_search"
        assert resp.tool_calls[0].args_json == {"q": "test"}

    def test_feedback_round_trips(self):
        feedback_time = datetime(2026, 5, 16, tzinfo=UTC)
        msg = _make_msg(
            feedback_rating=1,
            feedback_comment="great",
            feedback_at=feedback_time,
        )
        resp = _msg_to_response(msg, [])
        assert resp.feedback_rating == 1
        assert resp.feedback_comment == "great"
        assert resp.feedback_at == feedback_time


@pytest.mark.unit
class TestConversationsListResponseShape:
    def test_empty_list_serialises(self):
        page = ConversationsListResponse(items=[], next_cursor=None)
        data = page.model_dump(mode="json")
        assert data == {"items": [], "next_cursor": None}

    def test_cursor_set_when_more_pages(self):
        conv = _make_conv()
        page = ConversationsListResponse(
            items=[_conv_to_summary(conv)],
            next_cursor="2026-05-15T00:00:00",
        )
        data = page.model_dump(mode="json")
        assert data["next_cursor"] == "2026-05-15T00:00:00"
        assert len(data["items"]) == 1
