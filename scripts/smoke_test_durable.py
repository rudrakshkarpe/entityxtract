"""End-to-end smoke test for the durable + agentic extraction pipeline.

Runs the full workflow against a real Temporal dev server using the
``attention-is-all-you-need.pdf`` fixture in ``tests/data/`` and prints
the extracted Authors table.

Prerequisites
-------------
1. Install the Temporal CLI once:
       brew install temporal

2. Populate ``entityxtract/.env`` with an OpenAI-compatible endpoint:

       OPENAI_API_KEY=sk-or-v1-...
       OPENAI_API_BASE=https://openrouter.ai/api/v1
       OPENAI_DEFAULT_MODEL=google/gemini-2.5-flash

3. In terminal A start the dev server:
       temporal server start-dev

4. In terminal B (same virtualenv) start the worker:
       python -m entityxtract.durable.worker

5. In terminal C run this smoke test:
       python scripts/smoke_test_durable.py

You should see the workflow run to completion and print an extracted
``Authors`` table.  Visit http://localhost:8233 for the Temporal UI and
inspect the workflow's history, activities, and event timeline.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

# Importing entityxtract triggers .env loading via entityxtract.config.
from entityxtract import config as _config  # noqa: F401
from entityxtract.durable import (
    DocumentRef,
    EntitySpec,
    ExtractionJobRequest,
    run_extraction_job,
)
from entityxtract.extractor_types import ExtractionConfig, FileInputMode


# The PDF lives at <repo>/entityxtract/tests/data/attention-is-all-you-need.pdf.
# Resolve it relative to this script so the smoke test is CWD-independent.
SAMPLE_PDF = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "data"
    / "attention-is-all-you-need.pdf"
)


def _resolve_model() -> str:
    """Return the raw model name used by the extractor (LangChain)."""
    model = os.environ.get("ENTITYXTRACT_SMOKE_MODEL") or os.environ.get(
        "OPENAI_DEFAULT_MODEL"
    )
    if not model:
        print(
            "ERROR: set OPENAI_DEFAULT_MODEL in .env (e.g. "
            "'google/gemini-2.5-flash') or export ENTITYXTRACT_SMOKE_MODEL.",
            file=sys.stderr,
        )
        sys.exit(1)

    # If the caller accidentally supplied a pydantic-ai prefix like
    # "openai:foo", strip it — the extractor activity uses LangChain
    # which wants the raw model name.
    if ":" in model and not model.startswith("http"):
        model = model.split(":", 1)[1]

    return model


def _check_env() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "ERROR: OPENAI_API_KEY is missing.  Set it in entityxtract/.env "
            "or export it in your shell.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not SAMPLE_PDF.exists():
        print(
            f"ERROR: sample PDF not found at {SAMPLE_PDF}.  Place the "
            "'attention-is-all-you-need.pdf' fixture under tests/data/.",
            file=sys.stderr,
        )
        sys.exit(1)


async def main() -> None:
    _check_env()
    model = _resolve_model()

    authors_spec = EntitySpec(
        name="Authors",
        kind="table",
        example_schema=json.dumps([
            {
                "Name": "Foo Bar",
                "Organization": "University of Nowhere",
                "Email": "foo@bar.com",
            },
            {
                "Name": "Jane Doe",
                "Organization": "Institute of Something",
                "Email": "",
            },
        ]),
        instructions=(
            "Extract all authors from the document with their Name, "
            "Organization, and Email.  Typically found on the first page."
        ),
        required=True,
    )

    request = ExtractionJobRequest(
        doc_ref=DocumentRef(
            uri=SAMPLE_PDF.resolve().as_uri(),
            doc_type="pdf",
            page_range=(0, 2),  # first two pages are enough for Authors
        ),
        entities=[authors_spec],
        config=ExtractionConfig(
            model_name=model,
            temperature=0.0,
            file_input_modes=[FileInputMode.FILE],
            calculate_costs=False,
        ),
        auto_detect=False,
        require_human_approval=False,
    )

    # Fresh workflow_id every run so stale executions from earlier
    # attempts don't get replayed against a missing file.
    workflow_id = f"smoke-test-durable-{uuid.uuid4().hex[:8]}"

    print(f"Model:       {model}")
    print(f"Endpoint:    {os.environ.get('OPENAI_API_BASE', '<default openai>')}")
    print(f"Document:    {SAMPLE_PDF}")
    print(f"Entities:    {[e.name for e in request.entities]}")
    print(f"Workflow id: {workflow_id}")
    print("Submitting workflow to Temporal...\n")

    results = await run_extraction_job(
        request,
        workflow_id=workflow_id,
    )

    print("=== Results ===")
    print(json.dumps(results, indent=2)[:4000])

    authors = results.get("Authors")
    if not authors:
        print("\nFAIL: no Authors entity in result.", file=sys.stderr)
        sys.exit(2)
    print("\nOK: smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
