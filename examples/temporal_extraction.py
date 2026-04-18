"""Example: Durable entity extraction with Temporal.

Prerequisites:
    1. Install Temporal CLI:  brew install temporal
    2. Start dev server:      temporal server start-dev
    3. Start the worker:      python -m entityxtract.durable.worker
    4. Run this script:       python examples/temporal_extraction.py

The worker must be running in a separate terminal for the workflow to
execute.  If the worker crashes and is restarted, the extraction
resumes from where it left off — no duplicate LLM calls.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from entityxtract.durable import (
    DocumentRef,
    EntitySpec,
    ExtractionJobRequest,
    run_extraction_job,
)
from entityxtract.extractor_types import ExtractionConfig, FileInputMode

SAMPLE_PDF = Path(__file__).parent.parent / "tests" / "data" / "attention-is-all-you-need.pdf"


def build_request() -> ExtractionJobRequest:
    """Build an extraction request for the sample PDF."""
    authors_spec = EntitySpec(
        name="Authors",
        kind="table",
        example_schema=json.dumps([
            {"Name": "Foo Bar", "Organization": "University of Nowhere", "Email": "foo@bar.com"},
            {"Name": "Jane Doe", "Organization": "Institute of Something", "Email": ""},
        ]),
        instructions=(
            "Extract all authors from the document with their Name, Organization, and Email. "
            "Typically found on the first page."
        ),
        required=True,
    )

    benchmarks_spec = EntitySpec(
        name="Benchmarks",
        kind="table",
        example_schema=json.dumps([
            {"Model": "ModelA", "BLEU_EN_DE": 31.2, "BLEU_EN_FR": None, "Training_Cost": "5.4e18"},
            {"Model": "ModelB", "BLEU_EN_DE": 28.9, "BLEU_EN_FR": 42.5, "Training_Cost": "1.7e19"},
        ]),
        instructions=(
            "Extract benchmark results comparing different neural machine translation models. "
            "Include BLEU scores for EN-DE and EN-FR, plus training costs."
        ),
        required=True,
    )

    return ExtractionJobRequest(
        doc_ref=DocumentRef(
            uri=SAMPLE_PDF.resolve().as_uri(),
            doc_type="pdf",
            page_range=(0, 8),
        ),
        entities=[authors_spec, benchmarks_spec],
        config=ExtractionConfig(
            model_name="google/gemini-2.5-flash",
            temperature=0.0,
            file_input_modes=[FileInputMode.FILE],
            calculate_costs=True,
        ),
        auto_detect=False,
        require_human_approval=False,
    )


async def main() -> None:
    request = build_request()

    print(f"Starting durable extraction for: {request.doc_ref.uri}")
    print(f"Entities to extract: {[e.name for e in request.entities]}")
    print()

    results = await run_extraction_job(
        request,
        workflow_id="example-attention-paper",
    )

    print("=== Extraction Results ===")
    for name, data in results.items():
        print(f"\n--- {name} ---")
        print(json.dumps(data, indent=2)[:1000])

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
