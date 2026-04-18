"""Temporal workflow for durable entity extraction.

The workflow itself is deterministic — all I/O happens in activities
(either the hand-written ones in ``activities.py`` or the auto-generated
ones created by ``TemporalAgent``).
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import Any, Optional

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from entityxtract.durable.activities import (
        load_document_activity,
        render_pdf_pages_activity,
    )
    from entityxtract.durable.types import (
        ApprovalDecision,
        Deps,
        EntityProgress,
        EntityStatus,
        ExtractionJobRequest,
        JobProgress,
    )


@workflow.defn
class EntityExtractionWorkflow:
    """Durable, agentic entity extraction orchestrated by Temporal.

    The TemporalAgent and its auto-generated activities are registered
    on the Worker via the ``PydanticAIPlugin`` / ``__pydantic_ai_agents__``.
    Inside the workflow we call the agent by importing the cached
    singleton at run-time (inside the sandbox pass-through block above).

    Signals:
        approve_entity  — approve (optionally edit) an extracted entity
        reject_entity   — reject an entity and provide a reason

    Queries:
        get_progress        — returns ``JobProgress`` snapshot
        get_partial_results — returns completed extraction results so far
    """

    def __init__(self) -> None:
        self._progress = JobProgress()
        self._results: dict[str, Any] = {}
        self._approval_decisions: dict[str, ApprovalDecision] = {}

    # ----- Signals -----

    @workflow.signal
    async def approve_entity(self, entity_name: str, edited_data_json: Optional[str] = None) -> None:
        edited = json.loads(edited_data_json) if edited_data_json else None
        self._approval_decisions[entity_name] = ApprovalDecision(
            entity_name=entity_name,
            approved=True,
            edited_data=edited,
        )

    @workflow.signal
    async def reject_entity(self, entity_name: str, reason: str = "") -> None:
        self._approval_decisions[entity_name] = ApprovalDecision(
            entity_name=entity_name,
            approved=False,
            reason=reason,
        )

    # ----- Queries -----

    @workflow.query
    def get_progress(self) -> dict:
        return self._progress.model_dump()

    @workflow.query
    def get_partial_results(self) -> dict[str, Any]:
        return dict(self._results)

    # ----- Main entry point -----

    @workflow.run
    async def run(self, request_json: str) -> str:
        """Execute the extraction job.

        Accepts and returns JSON strings for maximum Temporal
        serialization compatibility.
        """
        request = ExtractionJobRequest.model_validate_json(request_json)
        self._progress.started_at = JobProgress.now_iso()

        for entity in request.entities:
            self._progress.entity_progress[entity.name] = EntityProgress(
                name=entity.name,
            )

        deps = Deps(
            model_name=request.config.model_name,
            temperature=request.config.temperature,
            file_input_modes=[m.value for m in request.config.file_input_modes],
            calculate_costs=request.config.calculate_costs,
        )

        doc_text = await workflow.execute_activity(
            load_document_activity,
            request.doc_ref.model_dump_json(),
            start_to_close_timeout=timedelta(seconds=60),
        )

        prompt = self._build_agent_prompt(request, doc_text)

        from entityxtract.durable.agent import get_temporal_coordinator

        coordinator = get_temporal_coordinator(request.config.model_name)
        result = await coordinator.run(prompt, deps=deps)

        try:
            self._results = json.loads(result.output) if isinstance(result.output, str) else {}
        except (json.JSONDecodeError, TypeError):
            self._results = {"raw_output": str(result.output)}

        for entity in request.entities:
            ep = self._progress.entity_progress.get(entity.name)
            if ep:
                ep.status = EntityStatus.COMPLETED if entity.name in self._results else EntityStatus.FAILED

        if request.require_human_approval:
            await self._wait_for_approvals(request)

        self._progress.completed_at = JobProgress.now_iso()
        return json.dumps(self._results)

    # ----- Helpers (pure, deterministic) -----

    def _build_agent_prompt(self, request: ExtractionJobRequest, doc_text: str) -> str:
        entities_desc = []
        for entity in request.entities:
            entities_desc.append(
                f"- {entity.name} (kind={entity.kind}, required={entity.required}): "
                f"{entity.instructions}"
            )

        doc_ref_json = request.doc_ref.model_dump_json()
        entity_specs_json = json.dumps([e.model_dump() for e in request.entities])

        parts = [
            "Extract the following entities from the provided document.",
            "",
            "Entities:",
            *entities_desc,
            "",
            f"Document reference (pass to tools as-is): {doc_ref_json}",
            f"Entity specifications (pass each to tools): {entity_specs_json}",
            "",
            f"Document text preview (first 2000 chars):\n{doc_text[:2000]}",
        ]

        if request.auto_detect:
            parts.append(
                "\nAlso run detect_entities to discover additional extractable entities."
            )

        parts.append(
            "\nReturn a JSON object mapping entity names to their extraction results."
        )
        return "\n".join(parts)

    async def _wait_for_approvals(self, request: ExtractionJobRequest) -> None:
        """Block until every extracted entity has been approved or rejected."""
        for entity in request.entities:
            if entity.name not in self._results:
                continue

            ep = self._progress.entity_progress.get(entity.name)
            if ep:
                ep.status = EntityStatus.AWAITING_APPROVAL

            try:
                await workflow.wait_condition(
                    lambda name=entity.name: name in self._approval_decisions,
                    timeout=timedelta(hours=24),
                )
            except asyncio.TimeoutError:
                workflow.logger.warning(
                    f"Approval timeout for {entity.name}, auto-approving"
                )
                self._approval_decisions[entity.name] = ApprovalDecision(
                    entity_name=entity.name,
                    approved=True,
                    reason="Auto-approved after 24h timeout",
                )

            decision = self._approval_decisions[entity.name]
            if decision.approved and decision.edited_data is not None:
                self._results[entity.name] = decision.edited_data
            elif not decision.approved:
                del self._results[entity.name]
                if ep:
                    ep.status = EntityStatus.FAILED
                    ep.last_error = f"Rejected: {decision.reason}"
