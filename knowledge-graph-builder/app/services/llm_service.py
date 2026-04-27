from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.services.credential_service import credential_service

logger = get_logger(__name__)

# ── SAME_AS disambiguation ────────────────────────────────────────────────────

_DISAMBIGUATE_MODEL = "gpt-4o-mini"
_DISAMBIGUATE_SYSTEM = (
    "You are a precise entity disambiguation assistant. "
    "Answer only in the structured format requested."
)


async def disambiguate_entities(
    name_a: str,
    type_a: str,
    context_a: list[str],
    name_b: str,
    type_b: str,
    context_b: list[str],
) -> dict:
    """Ask the LLM whether two entities refer to the same real-world entity.

    Returns a dict with keys:
        decision:   "YES" | "NO"
        confidence: "HIGH" | "MEDIUM" | "LOW"
        reason:     str (one sentence)

    On any LLM failure or parse error, returns the fail-safe default:
        {"decision": "NO", "confidence": "LOW", "reason": "parse_error"}

    Security: name_a, name_b, context_a, context_b are only embedded in the
    LLM prompt string — they are never interpolated into Cypher queries.
    """
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        logger.warning("disambiguate_entities: OPENAI_API_KEY not set — returning NO/LOW")
        return {"decision": "NO", "confidence": "LOW", "reason": "no_api_key"}

    context_a_str = ", ".join(context_a[:3]) if context_a else "none"
    context_b_str = ", ".join(context_b[:3]) if context_b else "none"

    prompt = (
        f'Are these two entities the same real-world entity? Answer YES or NO, '
        f'and provide a brief reason.\n\n'
        f'Entity A: "{name_a}" (type: {type_a})\n'
        f'Context A: {context_a_str}\n\n'
        f'Entity B: "{name_b}" (type: {type_b})\n'
        f'Context B: {context_b_str}\n\n'
        f'Answer format:\n'
        f'DECISION: YES | NO\n'
        f'CONFIDENCE: HIGH | MEDIUM | LOW\n'
        f'REASON: <one sentence>'
    )

    try:
        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=_DISAMBIGUATE_MODEL,
            messages=[
                {"role": "system", "content": _DISAMBIGUATE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("disambiguate_entities: LLM call failed: %s", exc)
        return {"decision": "NO", "confidence": "LOW", "reason": "parse_error"}

    # Defensive line-by-line parse
    decision = "NO"
    confidence = "LOW"
    reason = "parse_error"
    try:
        for line in raw.splitlines():
            line = line.strip()
            upper = line.upper()
            if upper.startswith("DECISION:"):
                val = line[len("DECISION:"):].strip().upper()
                if val in {"YES", "NO"}:
                    decision = val
            elif upper.startswith("CONFIDENCE:"):
                val = line[len("CONFIDENCE:"):].strip().upper()
                if val in {"HIGH", "MEDIUM", "LOW"}:
                    confidence = val
            elif upper.startswith("REASON:"):
                reason = line[len("REASON:"):].strip()
    except Exception as exc:
        logger.warning("disambiguate_entities: parse failed on %r: %s", raw, exc)
        return {"decision": "NO", "confidence": "LOW", "reason": "parse_error"}

    logger.debug(
        "disambiguate_entities: %r vs %r → decision=%s confidence=%s",
        name_a, name_b, decision, confidence,
    )
    return {"decision": decision, "confidence": confidence, "reason": reason}


class LLMService:
    """Service for managing LLM providers and graph transformation"""

    def __init__(self):
        self.llm = None
        self.graph_transformer = None
        self.current_provider = None
        self.current_model = None

    async def initialize_llm(
        self,
        user_id: str,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> bool:
        """Initialize LLM based on provider and user credentials"""

        try:
            # Get user credentials for the provider
            if provider == "openai":
                api_key = await credential_service.get_openai_token(user_id)
                if not api_key:
                    # Fallback to service-level key
                    api_key = settings.OPENAI_API_KEY

                if not api_key:
                    logger.error("No OpenAI API key available")
                    return False

                self.llm = ChatOpenAI(
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    max_retries=3,
                    request_timeout=60,
                )

            elif provider == "anthropic":
                api_key = await credential_service.get_anthropic_token(user_id)
                if not api_key:
                    api_key = settings.ANTHROPIC_API_KEY

                if not api_key:
                    logger.error("No Anthropic API key available")
                    return False

                self.llm = ChatAnthropic(
                    api_key=api_key,
                    model=(
                        model
                        if model.startswith("claude")
                        else "claude-3-haiku-20240307"
                    ),
                    temperature=temperature,
                    max_retries=3,
                    timeout=60,
                )

            elif provider == "google":
                api_key = await credential_service.get_user_credentials(
                    user_id, "google"
                )
                api_key = api_key.get("access_token") if api_key else None

                if not api_key:
                    logger.error("No Google API key available")
                    return False

                self.llm = ChatGoogleGenerativeAI(
                    google_api_key=api_key,
                    model=model if model.startswith("gemini") else "gemini-1.5-flash",
                    temperature=temperature,
                )
            else:
                logger.error(f"Unsupported LLM provider: {provider}")
                return False

            # Initialize graph transformer with correct import
            self.graph_transformer = LLMGraphTransformer(
                llm=self.llm,
                allowed_nodes=None,  # Will be set based on schema
                allowed_relationships=None,  # Will be set based on schema
                strict_mode=False,
            )

            self.current_provider = provider
            self.current_model = model

            logger.info(f"LLM initialized: {provider} - {model}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LLM {provider}: {e}")
            return False

    def set_schema(self, schema_config: dict[str, Any]):
        """Configure allowed entities and relationships"""
        if self.graph_transformer and schema_config:
            entities = schema_config.get("entities", [])
            relationships = schema_config.get("relationships", [])

            if entities:
                self.graph_transformer.allowed_nodes = entities
            if relationships:
                self.graph_transformer.allowed_relationships = relationships

    def set_dynamic_schema(self, schema: dict[str, Any]):
        """Configure LLM transformer with dynamic schema"""

        if not self.llm:
            logger.warning("LLM not initialized, cannot set dynamic schema")
            return

        self.graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=(
                schema.get("entities", []) if schema.get("entities") else None
            ),
            allowed_relationships=(
                schema.get("relationships", []) if schema.get("relationships") else None
            ),
            strict_mode=False,  # Allow creative discovery within boundaries
            node_properties=["name", "description", "type", "confidence"],
            relationship_properties=["confidence", "source"],
        )

        logger.info(
            f"Updated LLM schema with {len(schema.get('entities', []))} entity types and {len(schema.get('relationships', []))} relationship types"
        )

    def is_initialized(self) -> bool:
        """Check if LLM is properly initialized"""
        return self.llm is not None and self.graph_transformer is not None


# Global LLM service instance
llm_service = LLMService()
