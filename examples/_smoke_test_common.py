"""
Shared smoke-test infrastructure for the per-provider scripts in
this directory (smoke_test_ollama.py, smoke_test_openrouter.py,
smoke_test_azure.py).

Provides the Workload dataclass, the canonical WORKLOADS list, and
the run_smoke_test() runner. Each per-provider script handles its
own env-var defaults and argparse, then calls run_smoke_test().

run_embedding_smoke_test() is the create_embeddings() counterpart,
used by smoke_test_openrouter_embeddings.py.
"""
import logging

from dataclasses import dataclass
from pathlib import Path

from llm_router_ledger import (
    create_embeddings,
    load_config,
    send_message,
    UsageTracker,
)


logger = logging.getLogger(__name__)


# ---- Embedding workload ----
# The fallback corpus, used when no --input-file is given. Five short
# passages on model architecture: three on Mixture of Experts, then one
# on speculative decoding and one on LSTMs. The last two are adjacent
# enough to be plausible neighbours but distinct enough that they
# should not cluster with the MoE passages, which makes the set usable
# for a rough similarity check as well as a plumbing test. One passage
# is Chinese, so the multilingual models get exercised on something
# other than English tokenisation.
#
# Deliberately small: the point of a default run is to prove the path
# end to end and record what it cost, not to benchmark throughput.
# Point --input-file at a real corpus to measure a realistic bill.

EMBEDDING_TEXTS: list[str] = [
    "Mixture of Experts (MoE) is a machine learning design that "
    "splits a large neural network into smaller sub-networks called "
    "experts, managed by a router (or gating network) that activates "
    "only the most relevant experts for each piece of input data.",
    "混合专家模型是一种通过路由机制为每个输入只激活少量专家子网络的架构，"
    "从而以低计算成本实现巨大的模型容量。",
    "Sparse activation means only the top-k experts run for any given "
    "token, so parameter count can grow without a matching rise in "
    "inference cost.",
    "Speculative decoding pairs a small draft model with a larger "
    "verifier, accepting the drafted tokens only where the verifier "
    "agrees, which cuts latency without changing the output "
    "distribution.",
    "Long short-term memory networks read a sequence one step at a "
    "time through a recurrent cell with input, forget, and output "
    "gates, which is what limits how far their training can be "
    "parallelised.",
]


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


def _format_usd(value: float | None) -> str:
    """
    Helper function used to render a USD amount for the log line.
    Embedding calls run into the 1e-8 range, so the usual 2 decimals
    would print every one of them as 0.00. Returns "unreported" for
    None, which is what a provider that sends no cost field gets.
    """
    if value is None:
        return "unreported"
    return f"{value:.8f}"


def load_embedding_texts(path: Path) -> list[str]:
    """
    Read one text per line from path, dropping blank lines and
    surrounding whitespace. Raises ValueError if nothing is left, since
    embedding an empty batch would bill nothing and prove nothing.
    """
    texts = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8",
        ).splitlines()
        if line.strip()
    ]
    if not texts:
        raise ValueError(
            f"No non-blank lines in {path}"
        )
    return texts


def run_embedding_smoke_test(
    *,
    endpoint_name: str,
    log_path: Path,
    project_id: str,
    texts: list[str] | None = None,
) -> int:
    """
    Embed texts (default EMBEDDING_TEXTS) against endpoint_name,
    writing paired llm_request and llm_response events to log_path.
    Returns 0 on success.

    Reports two costs per call. reported is the provider's own charge
    for the call, taken from the response and written to the ledger's
    usage_details; it is authoritative. estimated is what the
    endpoint's configured input_per_1m rate implies for the tokens
    billed. A gap between them means the config's pricing is stale or
    the request was served by an upstream on a different rate, which is
    exactly what the ledger exists to surface.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if texts is None:
        texts = EMBEDDING_TEXTS
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Loaded once here rather than left to create_embeddings so the
    # endpoint's own cost config is available for the estimate below.
    config = load_config()
    endpoint = config.endpoints[endpoint_name]

    tracker = UsageTracker(
        log_path=log_path,
        project_id=project_id,
    )
    try:
        vectors, usage, gen_id = create_embeddings(
            endpoint_name=endpoint_name,
            texts=texts,
            config=config,
            tracker=tracker,
            purpose="document-embedding",
        )
        estimated = (
            endpoint.cost.estimate_cost(
                input_tokens=usage["prompt_tokens"],
                output_tokens=0,
            )
            if endpoint.cost
            else None
        )
        reported = usage.get("cost")
        logger.info(
            "vectors=%d dimensions=%d tokens=%d"
            " reported_cost_usd=%s estimated_cost_usd=%s"
            " upstream=%s generation_id=%s",
            len(vectors),
            usage["dimensions"],
            usage["prompt_tokens"],
            _format_usd(reported),
            _format_usd(estimated),
            usage.get("upstream_provider", "unreported"),
            gen_id,
        )
        logger.info("ledger: %s", log_path.resolve())
        return 0
    finally:
        tracker.close()


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
