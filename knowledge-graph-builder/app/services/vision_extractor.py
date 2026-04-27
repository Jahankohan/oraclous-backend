"""
Vision-based entity and relationship extraction from images.

Primary model : Claude 3.5 Sonnet (via Anthropic SDK)
Fallback model: GPT-4o (via OpenAI SDK)

The extractor returns a structured dict with "entities" and "relationships"
and also provides a helper to serialise that output as human-readable text
so it can be fed directly into the existing text-based ingestion pipeline.

Diagram mode (added by TASK-025):
    When the image is identified as a technical diagram — either through the
    ``likely_diagram`` flag in metadata (set by pdf_extractor / TASK-024) or
    through filename heuristics — the extractor switches to a structured prompt
    that returns nodes + edges instead of entities + relationships.
"""

import base64
import json
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_EXTRACTION_PROMPT = """\
Analyze this image and extract all entities and relationships visible in it.

Return ONLY valid JSON matching this exact schema — no prose, no markdown fences:
{{
  "entities": [
    {{"name": "string", "type": "string", "description": "string"}}
  ],
  "relationships": [
    {{"source": "string", "target": "string", "type": "string", "description": "string"}}
  ]
}}

Rules:
- Entity types are singular nouns (Person, Organization, System, Service, Concept, etc.)
- Relationship types are UPPER_SNAKE_CASE verbs (WORKS_FOR, DEPENDS_ON, CALLS, CONTAINS, etc.)
- Extract ALL components and connections visible — be exhaustive for diagrams
- Do not invent information not visible in the image
- If nothing can be extracted return {{"entities": [], "relationships": []}}

Context: {context}
"""

_DIAGRAM_PROMPT = """\
This image appears to be a technical diagram (architecture, UML, flowchart, or similar).
Extract the components and their relationships in this exact JSON structure:
{
  "nodes": [{"id": "...", "label": "...", "type": "component|service|database|process|actor|other"}],
  "edges": [{"from": "...", "to": "...", "label": "...", "type": "connection|data_flow|dependency|inheritance|other"}],
  "diagram_type": "architecture|uml_class|uml_sequence|flowchart|er|other",
  "description": "one-sentence summary"
}
Only return valid JSON. If you cannot identify structured components, return {"nodes": [], "edges": [], "diagram_type": "other", "description": "..."}.
"""

# Keywords in filenames that suggest a technical diagram
_DIAGRAM_FILENAME_KEYWORDS = {"arch", "diagram", "flow", "uml", "schema", "model"}

# File extensions that may carry diagram images (SVG is treated as image/png for API)
_DIAGRAM_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg"}

_SUPPORTED_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _is_diagram_mode(
    image_path: str, metadata: dict[str, Any] | None
) -> bool:
    """
    Return True when diagram-structured extraction should be used.

    Triggers:
    1. ``metadata["likely_diagram"]`` is True (set by pdf_extractor / TASK-024).
    2. File extension is in _DIAGRAM_IMAGE_EXTENSIONS AND the filename (stem)
       contains one of the _DIAGRAM_FILENAME_KEYWORDS.
    """
    if metadata and metadata.get("likely_diagram"):
        return True

    path = Path(image_path)
    if path.suffix.lower() in _DIAGRAM_IMAGE_EXTENSIONS:
        stem_lower = path.stem.lower()
        if any(kw in stem_lower for kw in _DIAGRAM_FILENAME_KEYWORDS):
            return True

    return False


class VisionExtractor:
    """
    Extract entities and relationships from images.

    Usage::

        result = vision_extractor.extract_from_image(
            "/tmp/diagram.png",
            context="AWS architecture diagram",
        )
        text = vision_extractor.to_text(result)
        # Feed `text` into the standard ingestion pipeline

    Diagram mode::

        result = vision_extractor.extract(
            "/tmp/arch_overview.png",
            metadata={"likely_diagram": True},
        )
        # Returns {"nodes": [...], "edges": [...], "diagram_type": "...", "description": "..."}
    """

    def extract(
        self,
        image_path: str,
        context: str = "",
        model: str = "claude",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Extract structured information from an image.

        When diagram mode is triggered (via ``metadata["likely_diagram"]`` or
        filename heuristics), returns a diagram-structured dict with ``nodes``
        and ``edges``.  Otherwise delegates to ``extract_from_image`` and
        returns the standard ``{"entities": [...], "relationships": [...]}``.

        Args:
            image_path: Absolute path to the image.
            context: Optional domain hint.
            model: "claude" (default) or "gpt4o".
            metadata: Optional dict; ``likely_diagram`` key triggers diagram mode.

        Returns:
            Diagram dict OR standard entities/relationships dict.
        """
        if _is_diagram_mode(image_path, metadata):
            return self._extract_diagram(image_path, model)
        return self.extract_from_image(image_path, context=context, model=model)

    def extract_from_image(
        self,
        image_path: str,
        context: str = "",
        model: str = "claude",
    ) -> dict[str, Any]:
        """
        Extract entities and relationships from an image file.

        Args:
            image_path: Absolute path to the image (PNG, JPG, WEBP).
            context: Optional domain hint, e.g. "AWS architecture diagram".
            model: "claude" (default) or "gpt4o".

        Returns:
            {"entities": [...], "relationships": [...]}
        """
        image_b64 = self._to_base64(image_path)
        media_type = self._media_type(image_path)
        prompt = _EXTRACTION_PROMPT.format(
            context=context or "No additional context provided."
        )

        if model == "gpt4o":
            return self._extract_gpt4o(image_b64, media_type, prompt)
        return self._extract_claude(image_b64, media_type, prompt)

    # ── Serialisation helper ──────────────────────────────────────────────────

    @staticmethod
    def to_text(result: dict[str, Any], context: str = "") -> str:
        """
        Convert vision extraction output to human-readable text suitable for
        the existing text ingestion pipeline.

        Example output::

            AWS Lambda is a Service. Serverless compute service.
            API Gateway is a Service. HTTP routing service.
            AWS Lambda DEPENDS ON API Gateway.
        """
        lines: list[str] = []
        if context:
            lines.append(f"Source context: {context}\n")

        for entity in result.get("entities", []):
            name = entity.get("name", "")
            etype = entity.get("type", "Entity")
            desc = entity.get("description", "")
            line = f"{name} is a {etype}."
            if desc:
                line += f" {desc}"
            lines.append(line)

        for rel in result.get("relationships", []):
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            rtype = rel.get("type", "RELATED_TO").replace("_", " ").lower()
            desc = rel.get("description", "")
            line = f"{src} {rtype} {tgt}."
            if desc:
                line += f" Context: {desc}"
            lines.append(line)

        return "\n".join(lines)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _call_claude(self, image_b64: str, media_type: str, prompt: str) -> str:
        """Call Claude vision API and return raw text response."""
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "anthropic package required for Claude vision. "
                "Install it with: pip install anthropic"
            )

        if not settings.ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not configured. "
                "Set it in .env to enable Claude vision extraction."
            )

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.content[0].text

    def _call_gpt4o(self, image_b64: str, media_type: str, prompt: str) -> str:
        """Call GPT-4o vision API and return raw text response."""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai package required for GPT-4o vision. "
                "Install it with: pip install openai"
            )

        if not settings.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. "
                "Set it in .env to enable GPT-4o vision extraction."
            )

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_b64}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.choices[0].message.content

    def _extract_claude(
        self, image_b64: str, media_type: str, prompt: str
    ) -> dict[str, Any]:
        return self._parse(self._call_claude(image_b64, media_type, prompt))

    def _extract_gpt4o(
        self, image_b64: str, media_type: str, prompt: str
    ) -> dict[str, Any]:
        return self._parse(self._call_gpt4o(image_b64, media_type, prompt))

    @staticmethod
    def _to_base64(image_path: str) -> str:
        with open(image_path, "rb") as fh:
            return base64.standard_b64encode(fh.read()).decode("utf-8")

    @staticmethod
    def _media_type(image_path: str) -> str:
        suffix = Path(image_path).suffix.lower()
        return _SUPPORTED_EXTENSIONS.get(suffix, "image/png")

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Strip markdown code fences from an LLM response."""
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            inner = lines[1:-1] if stripped.endswith("```") else lines[1:]
            stripped = "\n".join(inner)
        return stripped

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        """Parse JSON from LLM response, stripping markdown fences if present."""
        stripped = VisionExtractor._strip_fences(text)
        try:
            data = json.loads(stripped)
            return {
                "entities": data.get("entities", []),
                "relationships": data.get("relationships", []),
            }
        except json.JSONDecodeError as exc:
            logger.error(
                f"Failed to parse vision response as JSON: {exc}\n"
                f"Response (first 500 chars): {stripped[:500]}"
            )
            return {"entities": [], "relationships": []}

    def _extract_diagram(
        self,
        image_path: str,
        model: str = "claude",
    ) -> dict[str, Any]:
        """
        Run diagram-specific structured extraction.

        Returns a dict with ``nodes``, ``edges``, ``diagram_type``,
        and ``description``.
        """
        image_b64 = self._to_base64(image_path)
        media_type = self._media_type(image_path)
        prompt = _DIAGRAM_PROMPT

        if model == "gpt4o":
            raw_text = self._call_gpt4o(image_b64, media_type, prompt)
        else:
            raw_text = self._call_claude(image_b64, media_type, prompt)

        stripped = self._strip_fences(raw_text)
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            logger.error(
                f"Failed to parse diagram response as JSON: {exc}\n"
                f"Response (first 500 chars): {stripped[:500]}"
            )
            data = {}

        return {
            "nodes": data.get("nodes", []),
            "edges": data.get("edges", []),
            "diagram_type": data.get("diagram_type", "other"),
            "description": data.get("description", ""),
        }


vision_extractor = VisionExtractor()
