"""All Cypher templates touching ``:__Chat__`` nodes (STORY-031 / TASK-106).

ADR-020 reserves the ``:__Chat__`` namespace label for chat artifacts
projected from Postgres. This module is the **single source of truth**
for every Cypher statement that reads or writes a chat-namespaced node.

Rules enforced by ``tests/unit/test_chat_namespace_isolation.py``:

1. Any Cypher pattern matching ``:Conversation`` or ``:ChatTurn``
   anywhere in ``app/`` outside this module FAILS the CI test.
2. Every chat-projected node MUST carry the ``:__Chat__`` namespace
   label. Forgetting it would let analytics queries (community
   detection, retrieval, degree centrality, etc.) accidentally pick
   up chat nodes — the requirement Reza flagged.

The companion Celery task ``app.tasks.chat_projection`` is the only
caller of the WRITE templates. READ templates are reserved for future
memory / retrieval features that explicitly opt in to chat data.

Schema mirror:

* Postgres ``chat_conversations`` row → ``(:__Chat__:Conversation)``
* Postgres ``chat_messages`` row     → ``(:__Chat__:ChatTurn)``

Source of truth remains Postgres. The Neo4j projection is a derived
shadow; failures here never block the user-facing chat write path.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────── #
# WRITE templates — idempotent (MERGE-based) so retries are safe.
# ──────────────────────────────────────────────────────────────────────────── #

UPSERT_CONVERSATION = """
MERGE (c:__Chat__:Conversation { conversation_id: $conversation_id })
SET c.user_id = $user_id,
    c.graph_id = $graph_id,
    c.agent_id = $agent_id,
    c.title = $title,
    c.created_at = $created_at,
    c.last_message_at = $last_message_at
RETURN c.conversation_id AS conversation_id
""".strip()


UPSERT_CHAT_TURN = """
MATCH (c:__Chat__:Conversation { conversation_id: $conversation_id })
MERGE (t:__Chat__:ChatTurn { message_id: $message_id })
SET t.role = $role,
    t.snippet = $snippet,
    t.created_at = $created_at,
    t.conversation_id = $conversation_id
MERGE (t)-[:IN_CONVERSATION]->(c)
RETURN t.message_id AS message_id
""".strip()


# Outbound link to a knowledge-graph node. The target keeps its own
# labels (no :__Chat__ contamination) and the edge gets a :__Chat__
# property-free type — relationship types don't carry the namespace
# label, but the source node does, so cleanup queries can still find
# everything via the source.
LINK_CONVERSATION_TO_GRAPH = """
MATCH (c:__Chat__:Conversation { conversation_id: $conversation_id })
MATCH (g:Graph:__Platform__ { graph_id: $graph_id })
MERGE (c)-[:IN_GRAPH]->(g)
""".strip()


LINK_CONVERSATION_TO_AGENT = """
MATCH (c:__Chat__:Conversation { conversation_id: $conversation_id })
MATCH (a:Agent:__Platform__ { agent_id: $agent_id })
MERGE (c)-[:HANDLED_BY]->(a)
""".strip()


# Soft-delete a conversation's projection. Called from the
# conversation-delete sweeper task. Hard-deletes the namespaced nodes;
# linked KG nodes (Documents, Entities, Agents) are untouched because
# DETACH DELETE only removes the matched nodes and their relationships.
DELETE_CONVERSATION_PROJECTION = """
MATCH (c:__Chat__:Conversation { conversation_id: $conversation_id })
OPTIONAL MATCH (t:__Chat__:ChatTurn)-[:IN_CONVERSATION]->(c)
DETACH DELETE c, t
""".strip()


# ──────────────────────────────────────────────────────────────────────────── #
# READ templates — for future memory / analytics features that opt in.
# Not exercised by v1 chat handlers (Postgres is the source of truth for
# the read endpoints in TASK-104). Kept here so all chat Cypher lives
# in one place.
# ──────────────────────────────────────────────────────────────────────────── #

COUNT_CHAT_NODES = """
MATCH (n:__Chat__) RETURN count(n) AS count
""".strip()


COUNT_CHAT_NODES_FOR_GRAPH = """
MATCH (n:__Chat__:Conversation { graph_id: $graph_id })
OPTIONAL MATCH (t:__Chat__:ChatTurn)-[:IN_CONVERSATION]->(n)
RETURN count(DISTINCT n) AS conversations, count(DISTINCT t) AS turns
""".strip()


# Lookup conversation projection by id. Used by future memory features
# that traverse from :__Chat__:Conversation outward; never called by
# the user-facing chat read endpoints.
GET_CONVERSATION = """
MATCH (c:__Chat__:Conversation { conversation_id: $conversation_id })
RETURN c
""".strip()


# Lookup chat turns for a conversation. Same caveat as above.
LIST_CHAT_TURNS = """
MATCH (t:__Chat__:ChatTurn)-[:IN_CONVERSATION]->(c:__Chat__:Conversation { conversation_id: $conversation_id })
RETURN t.message_id AS message_id,
       t.role        AS role,
       t.snippet     AS snippet,
       t.created_at  AS created_at
ORDER BY t.created_at ASC
""".strip()
