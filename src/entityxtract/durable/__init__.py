"""entityxtract.durable — Temporal-backed durable agentic extraction.

Quick start::

    from entityxtract.durable import run_extraction_job, DocumentRef, ExtractionJobRequest, EntitySpec

    request = ExtractionJobRequest(
        doc_ref=DocumentRef(uri="file:///path/to/doc.pdf", doc_type="pdf"),
        entities=[EntitySpec(name="Authors", kind="table", example_schema="[...]", instructions="...")],
    )
    results = await run_extraction_job(request)

See ``entityxtract.durable.worker`` for running the Temporal worker.
"""

from .types import (
    ApprovalDecision,
    Deps,
    DocumentRef,
    EntitySpec,
    EntityProgress,
    EntityStatus,
    ExtractionJobRequest,
    JobProgress,
    SuggestedEntity,
    ValidationReport,
)

# Lazy imports — these pull in temporalio and pydantic_ai which require
# API keys at model-init time.  Importing the module is fine; constructing
# the TemporalAgent is deferred to first use.


def __getattr__(name: str):
    """Lazy import for heavy objects that need runtime API keys."""
    if name == "EntityExtractionWorkflow":
        from .workflow import EntityExtractionWorkflow
        return EntityExtractionWorkflow

    _client_names = {
        "approve_entity",
        "get_client",
        "get_job_progress",
        "reject_entity",
        "run_extraction_job",
        "start_extraction_job",
    }
    if name in _client_names:
        from . import client as _client
        return getattr(_client, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Types
    "ApprovalDecision",
    "Deps",
    "DocumentRef",
    "EntitySpec",
    "EntityProgress",
    "EntityStatus",
    "ExtractionJobRequest",
    "JobProgress",
    "SuggestedEntity",
    "ValidationReport",
    # Workflow
    "EntityExtractionWorkflow",
    # Client helpers
    "approve_entity",
    "get_client",
    "get_job_progress",
    "reject_entity",
    "run_extraction_job",
    "start_extraction_job",
]
