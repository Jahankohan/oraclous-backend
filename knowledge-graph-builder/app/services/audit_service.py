"""Audit event logging for public agent calls (STORY-022) and SA security events (ORA-316)."""

import hashlib
import time
import uuid

from app.core.logging import get_logger

logger = get_logger(__name__)


async def log_public_call(
    driver,
    graph_id: str,
    agent_id: str,
    integration_key_last4: str,
    input_text: str,
    response_text: str,
) -> None:
    """Write one :AuditEvent node for a public agent call.

    Uses SHA-256 hashes of input and response — plaintext never stored.
    """
    event_id = str(uuid.uuid4())
    now = int(time.time())
    input_hash = hashlib.sha256(input_text.encode()).hexdigest()
    response_hash = hashlib.sha256(response_text.encode()).hexdigest()

    try:
        await driver.execute_query(
            """
            CREATE (:AuditEvent {
                event_id:                $event_id,
                event_type:              "agent_public_call",
                graph_id:                $graph_id,
                agent_id:                $agent_id,
                integration_key_last4:   $key_last4,
                input_hash:              $input_hash,
                response_hash:           $response_hash,
                timestamp:               $now
            })
            """,
            {
                "event_id": event_id,
                "graph_id": graph_id,
                "agent_id": agent_id,
                "key_last4": integration_key_last4,
                "input_hash": input_hash,
                "response_hash": response_hash,
                "now": now,
            },
        )
    except Exception as exc:
        # Audit failure must not propagate to the caller
        logger.error("Failed to write AuditEvent: %s", exc)


async def log_sa_security_event(
    session,
    event_type: str,
    sa_id: str,
    actor_user_id: str,
    home_graph_id: str,
    tenant_id: str,
    key_prefix: str | None = None,
) -> None:
    """Write one :SecurityAuditLog node + [:FOR_SA]->(:AgentServiceAccount) edge.

    Caller provides the session — no try/except here so exceptions propagate and
    the caller's transaction rolls back atomically.  Raw API key and bcrypt hash
    MUST NOT be passed as key_prefix.
    """
    audit_log_id = str(uuid.uuid4())
    await session.run(
        """
        CREATE (a:SecurityAuditLog {
            audit_log_id:   $audit_log_id,
            event_type:     $event_type,
            sa_id:          $sa_id,
            actor_user_id:  $actor_user_id,
            home_graph_id:  $home_graph_id,
            tenant_id:      $tenant_id,
            key_prefix:     $key_prefix,
            timestamp:      datetime()
        })
        WITH a
        MATCH (sa:AgentServiceAccount {service_account_id: $sa_id, tenant_id: $tenant_id})
        CREATE (a)-[:FOR_SA]->(sa)
        """,
        {
            "audit_log_id": audit_log_id,
            "event_type": event_type,
            "sa_id": sa_id,
            "actor_user_id": actor_user_id,
            "home_graph_id": home_graph_id,
            "tenant_id": tenant_id,
            "key_prefix": key_prefix,
        },
    )
