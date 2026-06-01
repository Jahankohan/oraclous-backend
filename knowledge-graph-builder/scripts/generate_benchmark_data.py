"""
Generate the ORA-413 benchmark dataset for the memory benchmark suite.

Produces deterministic (seeded-RNG) JSON files in benchmarks/data/ and
benchmarks/queries/ covering:
  - stable_facts.json        400 facts  (sessions 1-4)
  - volatile_facts.json      300 facts  (sessions 5-7, 2-4 temporal versions each)
  - contradictions.json      200 facts  (sessions 8-9, original + one contradicting update)
  - cross_agent.json         100 facts  (session 10, agent_a writes, agent_b reads)
  - evaluation_queries.json  100 queries across 5 temporal-distance buckets

Usage:
    python scripts/generate_benchmark_data.py

Output path is relative to the knowledge-graph-builder/ directory, so run from there.
"""

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

SEED = 42
RNG = random.Random(SEED)

BENCHMARKS_DIR = Path(__file__).parent.parent / "benchmarks"
DATA_DIR = BENCHMARKS_DIR / "data"
QUERIES_DIR = BENCHMARKS_DIR / "queries"

AGENT_A = "benchmark_agent_a"
AGENT_B = "benchmark_agent_b"

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.replace(tzinfo=UTC).isoformat()


def _date(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Domain templates: diverse real-world-flavored fact seeds
# ---------------------------------------------------------------------------

PEOPLE = [
    "Alice Chen",
    "Bob Nakamura",
    "Carol Osei",
    "David Petrov",
    "Eva Martinez",
    "Frank Kowalski",
    "Grace Liu",
    "Henry Abara",
    "Isabel Torres",
    "James Park",
    "Karen Schmidt",
    "Leo Nguyen",
    "Maya Patel",
    "Noah Williams",
    "Olivia Brown",
    "Patrick Dubois",
    "Quinn Adeyemi",
    "Rosa Fernandez",
    "Sam Tanaka",
    "Tina Müller",
    "Umar Hassan",
    "Vera Ivanova",
    "Walter Ferreira",
    "Xia Zhao",
    "Yuki Sato",
    "Zara Ahmed",
    "Aaron Cole",
    "Bella Stone",
    "Carlos Rivera",
    "Diana Popescu",
    "Ethan Fox",
    "Fatima Alawi",
    "George King",
    "Helena Björk",
    "Ivan Sokolov",
    "Julia Weiss",
    "Kevin Okafor",
    "Lena Fischer",
    "Marco Esposito",
    "Nina Johansson",
]

COMPANIES = [
    "Nexon Technologies",
    "VertexAI Corp",
    "CrystalData Inc",
    "PeakSoft Ltd",
    "OrbitSystems",
    "BlueStar Analytics",
    "TerraCloud",
    "NovaMind",
    "SparkLabs",
    "ZenithWorks",
    "Apex Dynamics",
    "Lighthouse AI",
    "Fusion Networks",
    "CloudForge",
    "DataBridge",
    "ClearPath Inc",
    "ProtonWave",
    "NanoSoft",
    "AlphaStream",
    "QuantumLeap Systems",
]

ROLES = [
    "Software Engineer",
    "Senior Engineer",
    "Staff Engineer",
    "Principal Engineer",
    "Engineering Manager",
    "Director of Engineering",
    "VP of Engineering",
    "CTO",
    "Product Manager",
    "Senior PM",
    "Data Scientist",
    "ML Engineer",
    "DevOps Engineer",
    "Site Reliability Engineer",
    "QA Lead",
]

SKILLS = [
    "Python",
    "Rust",
    "Go",
    "TypeScript",
    "Neo4j",
    "PostgreSQL",
    "Redis",
    "Kubernetes",
    "Docker",
    "FastAPI",
    "React",
    "PyTorch",
    "TensorFlow",
    "Kafka",
    "Spark",
    "Airflow",
    "Terraform",
    "GraphQL",
    "gRPC",
]

LOCATIONS = [
    "Berlin",
    "Tokyo",
    "London",
    "San Francisco",
    "Singapore",
    "Amsterdam",
    "Paris",
    "Toronto",
    "Sydney",
    "New York",
    "Zurich",
    "Seoul",
]

PATENT_PREDICATES = [
    "holds_patent",
    "co-invented",
    "filed_patent_for",
]

FACT_STABLE_TEMPLATES: list[dict] = []

# Generate stable templates at module level so they're deterministic
_rng_init = random.Random(SEED)


def _build_stable_templates() -> list[dict]:
    templates = []

    # Person → role at company (130 facts — allow repeats; real-world data does too)
    for _i in range(130):
        person = _rng_init.choice(PEOPLE)
        company = _rng_init.choice(COMPANIES)
        templates.append(
            {
                "subject": person,
                "predicate": "works_at",
                "object": company,
                "content": f"{person} works at {company}.",
            }
        )

    # Person → known_for skill
    for _ in range(100):
        person = _rng_init.choice(PEOPLE)
        skill = _rng_init.choice(SKILLS)
        templates.append(
            {
                "subject": person,
                "predicate": "known_for",
                "object": skill,
                "content": f"{person} is known for expertise in {skill}.",
            }
        )

    # Person → based_in location
    for _ in range(80):
        person = _rng_init.choice(PEOPLE)
        city = _rng_init.choice(LOCATIONS)
        templates.append(
            {
                "subject": person,
                "predicate": "based_in",
                "object": city,
                "content": f"{person} is based in {city}.",
            }
        )

    # Company → headquarters
    for _ in range(50):
        company = _rng_init.choice(COMPANIES)
        city = _rng_init.choice(LOCATIONS)
        templates.append(
            {
                "subject": company,
                "predicate": "headquartered_in",
                "object": city,
                "content": f"{company} is headquartered in {city}.",
            }
        )

    # Person → patent
    for _ in range(40):
        person = _rng_init.choice(PEOPLE)
        pred = _rng_init.choice(PATENT_PREDICATES)
        patent_id = (
            f"PCT/IB{_rng_init.randint(2020, 2025)}/{_rng_init.randint(100000, 999999)}"
        )
        templates.append(
            {
                "subject": person,
                "predicate": pred,
                "object": patent_id,
                "content": f"{person} {pred.replace('_', ' ')} {patent_id}.",
            }
        )

    return templates[:400]


STABLE_TEMPLATES = _build_stable_templates()

# ---------------------------------------------------------------------------
# Stable facts (400, sessions 1-4)
# ---------------------------------------------------------------------------


def _generate_stable_facts() -> list[dict]:
    facts = []
    ingestion_base = _date(2024, 1, 15)

    for i, tmpl in enumerate(STABLE_TEMPLATES):
        session_num = (i // 100) + 1  # sessions 1-4
        facts.append(
            {
                "memory_id_hint": f"stable_{i:04d}",
                "session_id": f"benchmark_session_{session_num:02d}",
                "agent_id": AGENT_A,
                "type": "semantic",
                "content": tmpl["content"],
                "subject": tmpl["subject"],
                "predicate": tmpl["predicate"],
                "object": tmpl["object"],
                "confidence": 1.0,
                "scope": "agent",
                "source": "agent",
                "valid_from": _iso(ingestion_base + timedelta(days=i % 10)),
                "valid_to": None,
            }
        )

    return facts


# ---------------------------------------------------------------------------
# Volatile facts (300, sessions 5-7, 2-4 versions per fact)
# ---------------------------------------------------------------------------

VOLATILE_ROLE_TEMPLATES = [
    {
        "subject": person,
        "predicate": "holds_position",
        "versions": [
            ROLES[i % len(ROLES)],
            ROLES[(i + 2) % len(ROLES)],
            ROLES[(i + 4) % len(ROLES)],
        ],
    }
    for i, person in enumerate(PEOPLE * 8)
]


def _generate_volatile_facts() -> list[dict]:
    facts = []
    base_date = _date(2022, 1, 1)

    for i in range(300):
        person = PEOPLE[i % len(PEOPLE)]
        session_num = 5 + (i // 100)  # sessions 5-7
        num_versions = RNG.randint(2, 4)
        roles = RNG.sample(ROLES, num_versions)

        versions = []
        current = base_date + timedelta(days=i * 3)
        for j, role in enumerate(roles):
            step = timedelta(days=RNG.randint(30, 180))
            v_from = current
            v_to = (current + step) if j < num_versions - 1 else None
            versions.append(
                {
                    "object": role,
                    "valid_from": _iso(v_from),
                    "valid_to": _iso(current + step) if v_to else None,
                }
            )
            current = current + step

        facts.append(
            {
                "memory_id_hint": f"volatile_{i:04d}",
                "session_id": f"benchmark_session_{session_num:02d}",
                "agent_id": AGENT_A,
                "type": "semantic",
                "subject": person,
                "predicate": "holds_position",
                "versions": versions,
                "scope": "agent",
                "source": "agent",
            }
        )

    return facts


# ---------------------------------------------------------------------------
# Contradictory facts (200, sessions 8-9)
# ---------------------------------------------------------------------------


def _generate_contradictions() -> list[dict]:
    facts = []
    base_date = _date(2023, 6, 1)

    for i in range(200):
        person = PEOPLE[i % len(PEOPLE)]
        company_a = COMPANIES[i % len(COMPANIES)]
        company_b = COMPANIES[(i + 7) % len(COMPANIES)]
        session_num = 8 + (i // 100)  # sessions 8-9

        original_date = base_date + timedelta(days=i * 2)
        contradiction_date = original_date + timedelta(days=RNG.randint(14, 90))

        facts.append(
            {
                "memory_id_hint": f"contradiction_{i:04d}",
                "session_id": f"benchmark_session_{session_num:02d}",
                "agent_id": AGENT_A,
                "type": "semantic",
                "subject": person,
                "predicate": "employed_by",
                "original": {
                    "object": company_a,
                    "content": f"{person} is employed by {company_a}.",
                    "valid_from": _iso(original_date),
                    "valid_to": _iso(contradiction_date),
                    "confidence": 0.9,
                },
                "contradiction": {
                    "object": company_b,
                    "content": f"{person} is employed by {company_b}.",
                    "valid_from": _iso(contradiction_date),
                    "valid_to": None,
                    "confidence": 0.95,
                },
                "scope": "agent",
                "source": "agent",
            }
        )

    return facts


# ---------------------------------------------------------------------------
# Cross-agent facts (100, session 10)
# ---------------------------------------------------------------------------


def _generate_cross_agent_facts() -> list[dict]:
    facts = []
    base_date = _date(2024, 3, 1)

    for i in range(100):
        person = PEOPLE[i % len(PEOPLE)]
        company = COMPANIES[(i + 3) % len(COMPANIES)]

        facts.append(
            {
                "memory_id_hint": f"cross_agent_{i:04d}",
                "session_id": "benchmark_session_10",
                "agent_id": AGENT_A,
                "type": "semantic",
                "subject": person,
                "predicate": "leads_team_at",
                "object": company,
                "content": f"{person} leads a team at {company}.",
                "confidence": 0.95,
                "scope": "organization",
                "source": "agent",
                "valid_from": _iso(base_date + timedelta(days=i)),
                "valid_to": None,
                "query_agent_id": AGENT_B,
            }
        )

    return facts


# ---------------------------------------------------------------------------
# Evaluation queries (100 across 5 buckets)
# ---------------------------------------------------------------------------


def _generate_queries(
    stable: list[dict],
    volatile: list[dict],
    contradictions: list[dict],
    cross_agent: list[dict],
) -> list[dict]:
    queries = []

    # Bucket 1: Current (30 queries) — from stable facts
    for i in range(30):
        fact = stable[i * 13 % len(stable)]
        queries.append(
            {
                "id": f"q_current_{i:02d}",
                "bucket": "current",
                "text": f"What is the relationship between {fact['subject']} and {fact['object']}?",
                "expected_answer": fact["content"],
                "expected_memory_hint": fact["memory_id_hint"],
                "temporal_filter": "current",
                "graph_segment": "stable",
            }
        )

    # Bucket 2: Point-in-time (recent, 20 queries) — volatile facts at last-version date
    for i in range(20):
        fact = volatile[i * 14 % len(volatile)]
        last_version = fact["versions"][-1]
        # Query at a date within 15 days after the last version's valid_from
        q_date = datetime.fromisoformat(
            last_version["valid_from"].replace("Z", "+00:00")
        )
        q_date = q_date + timedelta(days=15)
        queries.append(
            {
                "id": f"q_pit_recent_{i:02d}",
                "bucket": "point_in_time_recent",
                "text": f"What position does {fact['subject']} hold?",
                "expected_answer": f"{fact['subject']} holds_position {last_version['object']}",
                "expected_object": last_version["object"],
                "expected_memory_hint": fact["memory_id_hint"],
                "temporal_filter": "current",
                "at_date": _iso(q_date),
                "graph_segment": "volatile",
            }
        )

    # Bucket 3: Point-in-time (distant, 20 queries) — volatile facts at first-version date
    for i in range(20):
        fact = volatile[(i * 17 + 5) % len(volatile)]
        first_version = fact["versions"][0]
        # Query at a date 10 days after the first version's valid_from
        q_date = datetime.fromisoformat(
            first_version["valid_from"].replace("Z", "+00:00")
        )
        q_date = q_date + timedelta(days=10)
        queries.append(
            {
                "id": f"q_pit_distant_{i:02d}",
                "bucket": "point_in_time_distant",
                "text": f"What was the role of {fact['subject']} at that time?",
                "expected_answer": f"{fact['subject']} holds_position {first_version['object']}",
                "expected_object": first_version["object"],
                "expected_memory_hint": fact["memory_id_hint"],
                "temporal_filter": "all",
                "at_date": _iso(q_date),
                "graph_segment": "volatile",
            }
        )

    # Bucket 4: Range (15 queries) — volatile facts across a date range
    for i in range(15):
        fact = volatile[(i * 19 + 3) % len(volatile)]
        v0 = fact["versions"][0]
        v_last = fact["versions"][-1]
        from_dt = datetime.fromisoformat(v0["valid_from"].replace("Z", "+00:00"))
        to_dt_str = v_last["valid_to"] or v_last["valid_from"]
        to_dt = datetime.fromisoformat(to_dt_str.replace("Z", "+00:00")) + timedelta(
            days=30
        )
        queries.append(
            {
                "id": f"q_range_{i:02d}",
                "bucket": "range",
                "text": f"List all positions held by {fact['subject']} between {from_dt.date()} and {to_dt.date()}.",
                "expected_objects": [v["object"] for v in fact["versions"]],
                "expected_memory_hint": fact["memory_id_hint"],
                "temporal_filter": "all",
                "from_date": _iso(from_dt),
                "to_date": _iso(to_dt),
                "graph_segment": "volatile",
            }
        )

    # Bucket 5: Post-contradiction (15 queries) — must return resolved (new) version
    for i in range(15):
        fact = contradictions[i * 13 % len(contradictions)]
        queries.append(
            {
                "id": f"q_post_contradiction_{i:02d}",
                "bucket": "post_contradiction",
                "text": f"Who employs {fact['subject']} currently?",
                "expected_answer": fact["contradiction"]["content"],
                "expected_object": fact["contradiction"]["object"],
                "expected_memory_hint": fact["memory_id_hint"],
                "temporal_filter": "current",
                "graph_segment": "contradictory",
            }
        )

    return queries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    QUERIES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating stable_facts.json (400 facts)...")
    stable = _generate_stable_facts()
    (DATA_DIR / "stable_facts.json").write_text(json.dumps(stable, indent=2))
    print(f"  ✓ {len(stable)} stable facts written")

    print("Generating volatile_facts.json (300 facts)...")
    volatile = _generate_volatile_facts()
    (DATA_DIR / "volatile_facts.json").write_text(json.dumps(volatile, indent=2))
    total_versions = sum(len(f["versions"]) for f in volatile)
    print(
        f"  ✓ {len(volatile)} volatile facts ({total_versions} total versions) written"
    )

    print("Generating contradictions.json (200 facts)...")
    contradictions = _generate_contradictions()
    (DATA_DIR / "contradictions.json").write_text(json.dumps(contradictions, indent=2))
    print(f"  ✓ {len(contradictions)} contradiction pairs written")

    print("Generating cross_agent.json (100 facts)...")
    cross_agent = _generate_cross_agent_facts()
    (DATA_DIR / "cross_agent.json").write_text(json.dumps(cross_agent, indent=2))
    print(f"  ✓ {len(cross_agent)} cross-agent facts written")

    print("Generating evaluation_queries.json (100 queries)...")
    queries = _generate_queries(stable, volatile, contradictions, cross_agent)
    (QUERIES_DIR / "evaluation_queries.json").write_text(json.dumps(queries, indent=2))
    bucket_counts = {}
    for q in queries:
        bucket_counts[q["bucket"]] = bucket_counts.get(q["bucket"], 0) + 1
    print(f"  ✓ {len(queries)} queries written: {bucket_counts}")

    print("\nAll benchmark data generated successfully.")
    print(f"  Output: {BENCHMARKS_DIR.resolve()}")


if __name__ == "__main__":
    main()
