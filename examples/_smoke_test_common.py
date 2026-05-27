"""
Shared smoke-test infrastructure for the per-provider scripts in
this directory (smoke_test_ollama.py, smoke_test_openrouter.py,
smoke_test_azure.py).

Provides the Workload dataclass, the canonical WORKLOADS list, and
the run_smoke_test() runner. Each per-provider script handles its
own env-var defaults and argparse, then calls run_smoke_test().
"""
import logging

from dataclasses import dataclass
from pathlib import Path

from llm_router_ledger import (
    send_message,
    UsageTracker,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Workload:
    """
    One smoke-test workload: a realistic prompt shape with a ledger
    purpose label.
    """
    use_case: str
    system: str
    user: str
    purpose: str


# ---- Natural-language question to SQL ----
# Exercises a small PostgreSQL schema plus a JOIN + GROUP BY + COUNT
# query.

NLQ_TO_SQL_SYSTEM = (
    "You are a SQL assistant. Given a PostgreSQL database schema, "
    "generate a SQL query that answers the user's question.\n\n"
    "Rules:\n"
    "- Return ONLY the SQL query, no explanations, no markdown\n"
    "- Use PostgreSQL syntax"
)

NLQ_TO_SQL_SCHEMA = """
CREATE TABLE suppliers (
    supplier_id VARCHAR(20) PRIMARY KEY,
    supplier_name VARCHAR(100) NOT NULL,
    country VARCHAR(50) NOT NULL
);

CREATE TABLE raw_materials (
    material_id VARCHAR(20) PRIMARY KEY,
    material_name VARCHAR(100) NOT NULL,
    supplier_id VARCHAR(20) NOT NULL REFERENCES suppliers(supplier_id),
    unit_cost DECIMAL(10,4)
);
""".strip()

NLQ_TO_SQL_QUESTION = (
    "How many different raw materials does each supplier provide? "
    "Show supplier name and count."
)


WORKLOADS: list[Workload] = [
    Workload(
        use_case="nlq-to-sql",
        system=NLQ_TO_SQL_SYSTEM,
        user=(
            f"Schema:\n{NLQ_TO_SQL_SCHEMA}\n\n"
            f"Question: {NLQ_TO_SQL_QUESTION}"
        ),
        purpose="nlq-to-sql",
    ),
]


def run_smoke_test(
    *,
    endpoint_name: str,
    log_path: Path,
    project_id: str,
) -> int:
    """
    Run every Workload in WORKLOADS against endpoint_name, writing
    paired llm_request and llm_response events to log_path. Returns
    0 on success.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = UsageTracker(
        log_path=log_path,
        project_id=project_id,
    )
    try:
        for i, workload in enumerate(WORKLOADS, start=1):
            text, usage, gen_id = send_message(
                endpoint_name=endpoint_name,
                system=workload.system,
                user=workload.user,
                tracker=tracker,
                purpose=workload.purpose,
            )
            logger.info(
                "test=%d use_case=%s response=%r usage=%s"
                " generation_id=%s",
                i,
                workload.use_case,
                text,
                usage,
                gen_id,
            )
        logger.info("ledger: %s", log_path.resolve())
        return 0
    finally:
        tracker.close()
