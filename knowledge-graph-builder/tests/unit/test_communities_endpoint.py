"""
Unit tests for the communities endpoint helpers and schema (TASK-050).
"""

import pytest

from app.api.v1.endpoints.communities import _derive_label
from app.schemas.community_schemas import Community


class TestDeriveLabel:
    @pytest.mark.unit
    def test_uses_first_sentence_of_summary(self):
        label = _derive_label(
            "c-1", "Tech companies in the Bay Area. Includes startups and giants."
        )
        assert label == "Tech companies in the Bay Area"

    @pytest.mark.unit
    def test_truncates_long_summary_without_period(self):
        long = "a" * 150
        label = _derive_label("c-1", long)
        assert len(label) == 80

    @pytest.mark.unit
    def test_falls_back_to_synthetic_label_when_summary_missing(self):
        label = _derive_label("c-abc-123", None)
        assert label == "Community c"  # first segment before "-"

    @pytest.mark.unit
    def test_falls_back_when_summary_is_empty_string(self):
        label = _derive_label("c-abc-123", "")
        assert label.startswith("Community ")

    @pytest.mark.unit
    def test_uses_id_prefix_when_no_dashes(self):
        label = _derive_label("abcd1234efgh", None)
        assert label == "Community abcd1234"

    @pytest.mark.unit
    def test_handles_newline_terminator_in_summary(self):
        label = _derive_label(
            "c-1", "Title\nLong description that should not be in the label"
        )
        assert label == "Title"


class TestCommunitySchema:
    @pytest.mark.unit
    def test_schema_round_trip(self):
        c = Community(
            community_id="c-1",
            level=0,
            label="Tech",
            size=10,
            summary="Tech companies",
        )
        data = c.model_dump(mode="json")
        assert data == {
            "community_id": "c-1",
            "level": 0,
            "label": "Tech",
            "size": 10,
            "summary": "Tech companies",
        }

    @pytest.mark.unit
    def test_summary_optional(self):
        c = Community(community_id="c-1", level=0, label="Anon", size=0)
        assert c.summary is None

    @pytest.mark.unit
    def test_size_must_be_non_negative(self):
        with pytest.raises(Exception):
            Community(community_id="c-1", level=0, label="x", size=-1)
