"""Tool functions used by the entityxtract coordinator agent.

Each function here is registered on the Pydantic AI ``Agent`` via
``@coordinator_agent.tool``.  When wrapped in a ``TemporalAgent``,
they automatically run as Temporal activities (so I/O is allowed).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic_ai import RunContext

from entityxtract.extractor_types import ExtractionResult

from .types import (
    ApprovalDecision,
    Deps,
    DocumentRef,
    EntitySpec,
    SuggestedEntity,
    ValidationReport,
)


async def extract_entity(
    ctx: RunContext[Deps],
    entity_spec_json: str,
    doc_ref_json: str,
) -> str:
    """Extract a single entity from the document using LLM.

    Args:
        entity_spec_json: JSON-serialized EntitySpec.
        doc_ref_json: JSON-serialized DocumentRef.

    Returns:
        JSON string of ExtractionResult.
    """
    from .activities import extract_entity_activity

    deps = ctx.deps
    payload = {
        "doc_ref": json.loads(doc_ref_json),
        "entity_spec": json.loads(entity_spec_json),
        "config": {
            "model_name": deps.model_name,
            "temperature": deps.temperature,
            "file_input_modes": deps.file_input_modes,
            "calculate_costs": deps.calculate_costs,
        },
    }
    return await extract_entity_activity(json.dumps(payload))


async def validate_extraction(
    ctx: RunContext[Deps],
    entity_name: str,
    extracted_data_json: str,
    entity_spec_json: str,
) -> str:
    """Validate the extracted data against the entity schema.

    Args:
        entity_name: Name of the entity.
        extracted_data_json: JSON string of the extracted data.
        entity_spec_json: JSON-serialized EntitySpec.

    Returns:
        JSON string of ValidationReport.
    """
    from .activities import validate_extraction_activity

    payload = {
        "entity_name": entity_name,
        "extracted_data": json.loads(extracted_data_json),
        "entity_spec": json.loads(entity_spec_json),
    }
    return await validate_extraction_activity(json.dumps(payload))


async def detect_entities(
    ctx: RunContext[Deps],
    doc_ref_json: str,
) -> str:
    """Auto-detect extractable entities in a document.

    Analyses the document text and suggests entities that can be extracted.

    Args:
        doc_ref_json: JSON-serialized DocumentRef.

    Returns:
        JSON list of SuggestedEntity objects.
    """
    from .activities import load_document_activity

    text = await load_document_activity(doc_ref_json)

    preview = text[:3000] if len(text) > 3000 else text
    suggestions = [
        SuggestedEntity(
            name="Auto-detected content",
            kind="table",
            description=f"Document contains {len(text)} characters of extractable content",
            confidence=0.5,
        )
    ]
    return json.dumps([s.model_dump() for s in suggestions])


async def request_human_approval(
    ctx: RunContext[Deps],
    entity_name: str,
    extracted_data_json: str,
) -> str:
    """Request human approval for an extraction result.

    In the Temporal workflow, this is bridged to a signal/wait pattern.
    The tool itself returns a pending decision — the workflow handles
    the actual wait.

    Args:
        entity_name: Name of the entity awaiting approval.
        extracted_data_json: JSON string of extracted data to review.

    Returns:
        JSON string of ApprovalDecision (initially pending).
    """
    decision = ApprovalDecision(
        entity_name=entity_name,
        approved=False,
        reason="Awaiting human review",
    )
    return decision.model_dump_json()
