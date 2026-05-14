"""Celery task for async webhook agent execution with optional egress (STORY-022)."""

import asyncio
import socket
import time
from urllib.parse import urlparse

from app.core.logging import get_logger

logger = get_logger(__name__)


def _validate_egress_url(url: str) -> None:
    """Raise ValueError if egress_url points to a private/internal host (SSRF guard)."""
    import ipaddress

    parsed = urlparse(url)
    host = parsed.hostname or ""
    _BLOCKED = {"localhost", "metadata.google.internal", "169.254.169.254"}
    _PRIVATE = [
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
        ipaddress.ip_network("127.0.0.0/8"),
        ipaddress.ip_network("169.254.0.0/16"),
        ipaddress.ip_network("0.0.0.0/8"),
        ipaddress.ip_network("::1/128"),
        ipaddress.ip_network("fc00::/7"),
    ]
    if host.lower() in _BLOCKED:
        raise ValueError(f"egress_url host '{host}' is not allowed (SSRF protection)")
    try:
        addr = ipaddress.ip_address(host)
        for net in _PRIVATE:
            if addr in net:
                raise ValueError(
                    f"egress_url host '{host}' is in a private range (SSRF protection)"
                )
    except ValueError as exc:
        if "SSRF" in str(exc):
            raise
        # hostname — resolve
        try:
            resolved = socket.gethostbyname(host)
            resolved_addr = ipaddress.ip_address(resolved)
            for net in _PRIVATE:
                if resolved_addr in net:
                    raise ValueError(
                        "egress_url resolves to private IP (SSRF protection)"
                    )
        except OSError:
            pass  # DNS failure is acceptable; caller's problem


def _run_webhook_async(
    graph_id: str,
    agent_id: str,
    slug: str,
    message: str,
    session_id: str | None,
    context: dict,
    egress_url: str | None,
    integration_key_last4: str,
) -> None:
    """Synchronous wrapper that runs the async agent execution in a new event loop."""
    asyncio.run(
        _async_webhook_core(
            graph_id=graph_id,
            agent_id=agent_id,
            slug=slug,
            message=message,
            session_id=session_id,
            context=context,
            egress_url=egress_url,
            integration_key_last4=integration_key_last4,
        )
    )


async def _async_webhook_core(
    graph_id: str,
    agent_id: str,
    slug: str,
    message: str,
    session_id: str | None,
    context: dict,
    egress_url: str | None,
    integration_key_last4: str,
) -> None:
    from neo4j import GraphDatabase

    from app.core.config import settings
    from app.services.agent_executor import AgentExecutor
    from app.services.audit_service import log_public_call

    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )
    try:
        async with driver.session(database=settings.NEO4J_DATABASE) as _:
            pass
    except Exception:
        pass

    # Use the async driver for agent execution
    from neo4j import AsyncGraphDatabase

    async_driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )
    try:
        executor = await AgentExecutor.from_neo4j(async_driver, graph_id, agent_id)
        result = await executor.run(message, session_id)

        await log_public_call(
            driver=async_driver,
            graph_id=graph_id,
            agent_id=agent_id,
            integration_key_last4=integration_key_last4,
            input_text=message,
            response_text=result.response,
        )

        if egress_url:
            try:
                _validate_egress_url(egress_url)
                await _fire_egress(
                    egress_url,
                    {
                        "agent_id": agent_id,
                        "session_id": result.session_id,
                        "response": result.response,
                        "provenance": (
                            result.provenance.model_dump()
                            if result.provenance
                            else None
                        ),
                        "context_echo": context,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
            except Exception as exc:
                logger.error("Egress POST to %s failed: %s", egress_url, exc)
    except Exception as exc:
        logger.error("Webhook agent execution failed for agent %s: %s", agent_id, exc)
    finally:
        await async_driver.close()


async def _fire_egress(url: str, payload: dict) -> None:
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=payload)
        logger.info("Egress POST to %s returned %s", url, resp.status_code)


try:
    from app.services.background_jobs import celery_app

    @celery_app.task(name="webhook_tasks.run_webhook_agent_task")
    def run_webhook_agent_task(
        graph_id: str,
        agent_id: str,
        slug: str,
        message: str,
        session_id: str | None,
        context: dict,
        egress_url: str | None,
        integration_key_last4: str,
    ) -> None:
        _run_webhook_async(
            graph_id=graph_id,
            agent_id=agent_id,
            slug=slug,
            message=message,
            session_id=session_id,
            context=context,
            egress_url=egress_url,
            integration_key_last4=integration_key_last4,
        )

except Exception:
    # Celery not configured in test environments — define a no-op stub
    class _StubTask:
        def delay(self, **kwargs):
            logger.warning("Celery not available — webhook task not queued")

    run_webhook_agent_task = _StubTask()
