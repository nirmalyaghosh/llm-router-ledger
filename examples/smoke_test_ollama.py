"""
End-to-end smoke test against a local Ollama server.

Runs a set of representative LLM workloads through send_message()
and writes paired llm_request / llm_response events to the JSONL
ledger for each call. Each Workload entry exercises one realistic
use case (system prompt, user content, ledger purpose).

To add coverage for another use case, define its constants below
and append a Workload entry to WORKLOADS. The companion script
examples/smoke_test_openrouter.py runs the same WORKLOADS against
OpenRouter instead of Ollama.

Prerequisites:
- Ollama running on http://localhost:11434
- A chat model pulled locally that matches the model field of
  your endpoint in llm_endpoints.yaml (the default endpoint
  local-llama in examples/llm_endpoints.example.yaml uses
  qwen2.5vl:3b; substitute whatever you have pulled).
- llm_endpoints.yaml in the working directory.

Run from the project root:
    python examples/smoke_test_ollama.py
    python examples/smoke_test_ollama.py --endpoint local-llama
"""
import argparse
import datetime
import logging
import os
import sys

from dataclasses import dataclass
from importlib.metadata import (
    PackageNotFoundError,
    version,
)
from pathlib import Path

from llm_router_ledger import (
    send_message,
    UsageTracker,
)


logger = logging.getLogger(__name__)

try:
    _LIBRARY_VERSION = version("llm-router-ledger")
except PackageNotFoundError:
    _LIBRARY_VERSION = "0.0.0+local"

os.environ.setdefault("OLLAMA_API_KEY", "ollama")
os.environ.setdefault("LRL_RUN_TAG", "smoke")
os.environ.setdefault(
    "LRL_RUN_LABEL",
    f"{_LIBRARY_VERSION}-ollama-verify-{datetime.date.today().isoformat()}",
)


@dataclass(frozen=True)
class Workload:
    """
    One smoke-test workload: a realistic prompt shape
    with a ledger purpose label.
    """
    use_case: str
    system: str
    user: str
    purpose: str


# ---- Natural-language question to SQL ----
# Exercises a small PostgreSQL schema plus a JOIN + GROUP BY + COUNT query.

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test an Ollama endpoint via llm-router-ledger "
            "using a set of representative LLM workloads."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default="local-llama",
        help="Endpoint name from llm_endpoints.yaml (default: local-llama).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    log_path = Path("logs/ollama_smoke_test_token_usage.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = UsageTracker(
        log_path=log_path,
        project_id="ollama-smoke-test",
    )
    try:
        for i, workload in enumerate(WORKLOADS, start=1):
            text, usage, gen_id = send_message(
                endpoint_name=args.endpoint,
                system=workload.system,
                user=workload.user,
                tracker=tracker,
                purpose=workload.purpose,
            )
            logger.info(
                "test=%d use_case=%s response=%r usage=%s generation_id=%s",
                i, workload.use_case, text, usage, gen_id,
            )
        logger.info("ledger: %s", log_path.resolve())
        return 0
    finally:
        tracker.close()


if __name__ == "__main__":
    sys.exit(main())
