"""Hermetic unit tests for the markdown primitive (TASK-222 / STORY-034).

Pure translation tests — a small in-memory markdown string is decomposed and
the resulting `StructuralRepresentation` is asserted. No database, no I/O.
"""

from app.recipes.primitives import (
    ExtractionMode,
    MarkdownPrimitive,
    Primitive,
    StructuralRepresentation,
    UnitKind,
)

_MD = """\
# Handbook

Intro paragraph.

## Onboarding

Onboarding body text that is deliberately long enough to exercise the \
SAMPLE-mode preview truncation path so the bounded example value is genuinely \
shorter than the full section content emitted in FULL mode for this same \
chunk of markdown. We pad it with extra sentences so the full body comfortably \
exceeds the bounded preview length the primitive applies in SAMPLE mode, which \
keeps the truncation assertion meaningful rather than accidentally trivial.

### Day One

First-day checklist.

## Benefits

Benefits body.
"""


class TestMarkdownPrimitive:
    def test_conforms_to_primitive_protocol(self):
        assert isinstance(MarkdownPrimitive(), Primitive)
        assert MarkdownPrimitive().source_type == "text"

    def test_document_and_chunks(self):
        rep = MarkdownPrimitive().decompose(_MD, ExtractionMode.FULL)
        assert isinstance(rep, StructuralRepresentation)
        assert rep.source_type == "text"

        docs = [u for u in rep.units if u.kind == UnitKind.DOCUMENT]
        chunks = [u for u in rep.units if u.kind == UnitKind.CHUNK]
        assert len(docs) == 1
        assert docs[0].name == "Handbook"
        # Four headings: Handbook, Onboarding, Day One, Benefits.
        assert len(chunks) == 4
        assert [c.name for c in chunks] == [
            "Handbook",
            "Onboarding",
            "Day One",
            "Benefits",
        ]

    def test_heading_hierarchy_containment(self):
        rep = MarkdownPrimitive().decompose(_MD, ExtractionMode.FULL)
        doc = next(u for u in rep.units if u.kind == UnitKind.DOCUMENT)
        chunks = {u.name: u for u in rep.units if u.kind == UnitKind.CHUNK}

        # H1 is parented to the document.
        assert chunks["Handbook"].parent_id == doc.unit_id
        # H2 "Onboarding" nests under H1 "Handbook".
        assert chunks["Onboarding"].parent_id == chunks["Handbook"].unit_id
        # H3 "Day One" nests under H2 "Onboarding".
        assert chunks["Day One"].parent_id == chunks["Onboarding"].unit_id
        # H2 "Benefits" pops back up to H1 "Handbook".
        assert chunks["Benefits"].parent_id == chunks["Handbook"].unit_id

    def test_chunk_role_and_heading_level(self):
        rep = MarkdownPrimitive().decompose(_MD, ExtractionMode.FULL)
        chunks = {u.name: u for u in rep.units if u.kind == UnitKind.CHUNK}
        assert chunks["Handbook"].role == "free_text"
        assert chunks["Handbook"].metadata["heading_level"] == 1
        assert chunks["Day One"].metadata["heading_level"] == 3

    def test_sample_mode_truncates_body(self):
        sample = MarkdownPrimitive().decompose(_MD, ExtractionMode.SAMPLE)
        full = MarkdownPrimitive().decompose(_MD, ExtractionMode.FULL)
        sample_chunks = {u.name: u for u in sample.units if u.kind == UnitKind.CHUNK}
        full_chunks = {u.name: u for u in full.units if u.kind == UnitKind.CHUNK}

        sample_body = sample_chunks["Onboarding"].sample_values[0]
        full_body = full_chunks["Onboarding"].sample_values[0]
        # SAMPLE bounds the body; FULL emits the complete section content.
        assert len(sample_body) <= 280
        assert len(full_body) > len(sample_body)
        assert full_body.startswith(sample_body)

    def test_shape_signature_depends_on_outline(self):
        rep_a = MarkdownPrimitive().decompose(_MD, ExtractionMode.SAMPLE)
        rep_b = MarkdownPrimitive().decompose(_MD, ExtractionMode.FULL)
        # Signature depends on the heading outline, not the mode.
        assert rep_a.shape_signature == rep_b.shape_signature
        assert rep_a.shape_signature.startswith("text(md:4sections")
