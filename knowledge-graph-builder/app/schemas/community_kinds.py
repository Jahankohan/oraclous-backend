"""Community-kind registry (STORY-4a).

A single source of truth for "what kinds of communities can this graph
hold." Endpoints, the analytics service, the agent toolkit, and any
future tasks all read from the same registry instead of hardcoding
Neo4j labels and relationship types.

Adding a new community kind = add one ``CommunityKindSpec`` entry to the
``COMMUNITY_KINDS`` dict below. No endpoint, service, or schema code
change required. The discovery endpoint ``GET /communities/kinds``
exposes the registry to API consumers so the frontend can also stay
metadata-driven.

Design notes
------------
- ``kind`` is the stable wire identifier. Never rename without a
  deprecation cycle — it appears in API query params, JSON payloads, and
  Celery task names.
- ``detector_task_name`` may be ``None`` when a kind is read-only — e.g.,
  the chunk-community Louvain script that produced :ChunkCommunity nodes
  during STORY-024 demo prep has not yet been promoted to a Celery task,
  but the nodes still need to be queryable. ``POST .../detect`` returns
  HTTP 405 with a clear message when ``detector_task_name is None``.
- ``hierarchical`` distinguishes Leiden (entity, multi-level) from Louvain
  (chunk, flat). Callers can branch on this to decide whether the
  ``level`` field on a Community is meaningful or always ``None``.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommunityKindSpec:
    """Describes one kind of community Neo4j can hold for a graph.

    Property-name fields (``id_property``, ``size_property``) exist because
    the two kinds we have today were materialized by different scripts that
    chose different conventions. Rather than retroactively renaming
    properties on live data, the registry tells callers what to query.
    """

    # Stable wire identifier. Used in API query params, JSON payloads,
    # Celery task names. Lowercase, snake_case, no spaces.
    kind: str

    # Human-readable name for UI / error messages.
    display_name: str

    # Neo4j label of the community node itself.
    community_label: str

    # Neo4j label of the node type members of this community are.
    member_label: str

    # Neo4j relationship that connects (member)-[REL]->(community).
    # Note the direction: members point AT their community.
    member_rel: str

    # Property on the community node that holds its stable id. Different
    # community kinds historically chose different names (``id`` for the
    # entity-Leiden output, ``community_id`` for the chunk-Louvain output).
    id_property: str

    # Property on the community node that holds its member count.
    # Entity-Leiden writes ``entity_count``; chunk-Louvain writes ``size``.
    size_property: str

    # True for hierarchical clusterings (Leiden) where ``level`` matters,
    # False for flat clusterings (Louvain) where ``level`` is always None.
    # Hierarchical kinds also have a ``PARENT_COMMUNITY`` edge between
    # adjacent levels and a ``parent_id`` property on each community node.
    hierarchical: bool

    # True when the member relationship has a ``level`` property the
    # detail-fetch query must match on (entity-Leiden does this; chunk
    # Louvain does not).
    member_rel_has_level: bool

    # Celery task name that detects this kind, or None when detection
    # has not been wired yet (read-only kind). Endpoints return HTTP 405
    # on POST .../detect when this is None.
    detector_task_name: str | None

    # Property on the community node that holds the summary embedding
    # (STORY-4c). Entity-Leiden uses ``embedding`` (the Celery detector's
    # convention, predates STORY-4b); chunk uses ``summary_embedding``
    # (under the ``summary_*`` namespace alongside ``summary_keywords``).
    # The vector index over this property is named ``{index_name}``.
    embedding_property: str

    # Name of the Neo4j vector index over the community node, scoped to
    # the kind. ``ensure_community_vector_indexes`` creates this index on
    # startup if it doesn't exist. ``find_communities`` agent tool
    # (STORY-4d) queries it via ``db.index.vector.queryNodes``.
    index_name: str


# Adding a new community kind: append one entry below. No other change
# required — endpoint code, analytics queries, agent tools, and the
# discovery endpoint all read from this dict.
COMMUNITY_KINDS: dict[str, CommunityKindSpec] = {
    "entity": CommunityKindSpec(
        kind="entity",
        display_name="Entity community (Leiden, hierarchical)",
        community_label="__Community__",
        member_label="__Entity__",
        member_rel="IN_COMMUNITY",
        id_property="id",
        size_property="entity_count",
        hierarchical=True,
        member_rel_has_level=True,
        detector_task_name="community_tasks.detect_communities_task",
        embedding_property="embedding",
        index_name="community_embeddings_entity",
    ),
    "chunk": CommunityKindSpec(
        kind="chunk",
        display_name="Chunk community (Louvain, flat)",
        community_label="ChunkCommunity",
        member_label="Chunk",
        member_rel="IN_CHUNK_COMMUNITY",
        id_property="community_id",
        size_property="size",
        hierarchical=False,
        member_rel_has_level=False,
        # Detection not yet wired. The original Louvain run was a one-shot
        # script during STORY-024 demo prep. Promoting it to a Celery task
        # is tracked as a follow-up story; until then, only the 59 chunk
        # communities already in the Eurail graph are visible.
        detector_task_name=None,
        embedding_property="summary_embedding",
        index_name="community_embeddings_chunk",
    ),
    # Future kinds slot in here without touching endpoint or service code:
    #
    # "sub_entity": CommunityKindSpec(
    #     kind="sub_entity",
    #     display_name="Within-entity property clustering",
    #     community_label="SubEntityCluster",
    #     member_label="EntityProperty",
    #     member_rel="IN_SUB_ENTITY_COMMUNITY",
    #     hierarchical=False,
    #     detector_task_name="community_tasks.detect_sub_entity_clusters_task",
    # ),
}


class UnknownCommunityKindError(ValueError):
    """Raised when a caller references a community kind that is not in
    ``COMMUNITY_KINDS``. Endpoints translate this to HTTP 400 with the
    valid kind list in the message body."""

    def __init__(self, kind: str) -> None:
        valid = ", ".join(sorted(COMMUNITY_KINDS.keys()))
        super().__init__(f"Unknown community kind: {kind!r}. Valid kinds: {valid}")
        self.kind = kind
        self.valid_kinds = sorted(COMMUNITY_KINDS.keys())


def get_kind(kind: str) -> CommunityKindSpec:
    """Resolve a kind id. Raises ``UnknownCommunityKindError`` if unknown."""
    if kind not in COMMUNITY_KINDS:
        raise UnknownCommunityKindError(kind)
    return COMMUNITY_KINDS[kind]


def all_kinds() -> list[CommunityKindSpec]:
    """Return every registered kind, in insertion order."""
    return list(COMMUNITY_KINDS.values())


def kind_for_community_label(label: str) -> CommunityKindSpec | None:
    """Reverse-lookup: given a Neo4j label, return the matching kind spec.

    Used by endpoints that receive a ``community_id`` without a ``?kind=``
    parameter (the ID is globally unique, so we can sniff the kind by
    looking at the node's labels in Neo4j).
    Returns None if no kind matches the label.
    """
    for spec in COMMUNITY_KINDS.values():
        if spec.community_label == label:
            return spec
    return None
