"""
Vision-based entity and relationship extraction from images.

Primary model : Claude 3.5 Sonnet (via Anthropic SDK)
Fallback model: GPT-4o (via OpenAI SDK)

The extractor returns a structured dict with "entities" and "relationships"
and also provides a helper to serialise that output as human-readable text
so it can be fed directly into the existing text-based ingestion pipeline.
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

_SUPPORTED_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


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
    """

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

    def _extract_claude(
        self, image_b64: str, media_type: str, prompt: str
    ) -> dict[str, Any]:
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
        return self._parse(response.content[0].text)

    def _extract_gpt4o(
        self, image_b64: str, media_type: str, prompt: str
    ) -> dict[str, Any]:
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
        return self._parse(response.choices[0].message.content)

    @staticmethod
    def _to_base64(image_path: str) -> str:
        with open(image_path, "rb") as fh:
            return base64.standard_b64encode(fh.read()).decode("utf-8")

    @staticmethod
    def _media_type(image_path: str) -> str:
        suffix = Path(image_path).suffix.lower()
        return _SUPPORTED_EXTENSIONS.get(suffix, "image/png")

    @staticmethod
    def _parse(text: str) -> dict[str, Any]:
        """Parse JSON from LLM response, stripping markdown fences if present."""
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            inner = lines[1:-1] if stripped.endswith("```") else lines[1:]
            stripped = "\n".join(inner)
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


vision_extractor = VisionExtractor()
