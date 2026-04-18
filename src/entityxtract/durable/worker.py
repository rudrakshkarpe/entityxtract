"""Temporal worker entry point for entityxtract.

Usage:
    # Start the Temporal dev server first:
    temporal server start-dev

    # Then run the worker:
    python -m entityxtract.durable.worker

Environment
-----------
Reads from ``entityxtract/.env`` (via :mod:`entityxtract.config`):

* ``OPENAI_API_KEY``      — shared between the extractor (LangChain) and
  the Pydantic AI coordinator (OpenAI SDK).
* ``OPENAI_API_BASE``     — LangChain-style env var.  Mirrored into
  ``OPENAI_BASE_URL`` if the latter is unset so the OpenAI SDK (used by
  the Pydantic AI coordinator) points at the same endpoint.
* ``OPENAI_DEFAULT_MODEL``— Raw model name used by LangChain in the
  extractor activity (e.g. ``google/gemini-2.5-flash``).  Also used to
  build the coordinator's model identifier as
  ``openai:<OPENAI_DEFAULT_MODEL>``.
* ``ENTITYXTRACT_COORDINATOR_MODEL`` — optional override for the
  coordinator's model identifier (must be a Pydantic AI model string
  like ``openai:gpt-4o-mini`` or ``anthropic:claude-3-5-haiku-latest``).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin

from entityxtract import config as _config  # noqa: F401  (triggers .env load)

from .activities import (
    extract_entity_activity,
    load_document_activity,
    render_pdf_pages_activity,
    validate_extraction_activity,
)
from .agent import get_temporal_coordinator
from .workflow import EntityExtractionWorkflow


def _prepare_openai_env() -> str:
    """Mirror OPENAI_API_BASE → OPENAI_BASE_URL and return coordinator model id."""
    api_base = os.environ.get("OPENAI_API_BASE")
    if api_base and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = api_base

    override = os.environ.get("ENTITYXTRACT_COORDINATOR_MODEL")
    if override:
        return override

    raw_model = os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
    return f"openai:{raw_model}"


async def run_worker(
    target_host: str = "localhost:7233",
    task_queue: str = "entityxtract",
    namespace: str = "default",
) -> None:
    """Connect to Temporal and run the entityxtract worker."""
    coordinator_model = _prepare_openai_env()

    coordinator = get_temporal_coordinator(coordinator_model)
    EntityExtractionWorkflow.__pydantic_ai_agents__ = [coordinator]

    print(f"Coordinator model: {coordinator_model}")
    print(f"Coordinator activities: {[a.__name__ for a in coordinator.temporal_activities]}")

    client = await Client.connect(
        target_host,
        namespace=namespace,
        plugins=[PydanticAIPlugin()],
    )

    restrictions = SandboxRestrictions.default.with_passthrough_modules(
        "entityxtract",
        "langchain_openai",
        "langchain_core",
        "langchain",
        "urllib3",
        "requests",
        "httpcore",
        "httpx",
        "openai",
        "polars",
        "pdfminer",
        "pypdfium2",
    )
    runner = SandboxedWorkflowRunner(restrictions=restrictions)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        worker = Worker(
            client,
            task_queue=task_queue,
            workflows=[EntityExtractionWorkflow],
            activities=[
                load_document_activity,
                render_pdf_pages_activity,
                extract_entity_activity,
                validate_extraction_activity,
            ],
            activity_executor=executor,
            workflow_runner=runner,
        )
        print(f"Worker listening on task queue {task_queue!r}...")
        await worker.run()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
