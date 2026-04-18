"""Client helper to start and monitor extraction workflows.

Usage:
    from entityxtract.durable.client import run_extraction_job
    results = await run_extraction_job(request, workflow_id="my-job-1")
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from temporalio.client import Client
from temporalio.common import WorkflowIDConflictPolicy, WorkflowIDReusePolicy

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin

from .types import ExtractionJobRequest, JobProgress
from .workflow import EntityExtractionWorkflow


async def get_client(
    target_host: str = "localhost:7233",
    namespace: str = "default",
) -> Client:
    """Create a Temporal client with the Pydantic AI plugin."""
    return await Client.connect(
        target_host,
        namespace=namespace,
        plugins=[PydanticAIPlugin()],
    )


async def run_extraction_job(
    request: ExtractionJobRequest,
    *,
    workflow_id: Optional[str] = None,
    target_host: str = "localhost:7233",
    task_queue: str = "entityxtract",
    namespace: str = "default",
    client: Optional[Client] = None,
) -> dict[str, Any]:
    """Submit an extraction job and wait for the result.

    This is the primary convenience function for running durable
    extractions from application code.

    Args:
        request: The extraction job specification.
        workflow_id: Optional stable ID. Defaults to a UUID.
        target_host: Temporal server address.
        task_queue: Task queue the worker listens on.
        namespace: Temporal namespace.
        client: Optional pre-configured Client instance.

    Returns:
        Dict mapping entity names to their extraction results.
    """
    if client is None:
        client = await get_client(target_host, namespace=namespace)

    wf_id = workflow_id or f"entityxtract-{uuid.uuid4()}"

    result_json = await client.execute_workflow(
        EntityExtractionWorkflow.run,
        request.model_dump_json(),
        id=wf_id,
        task_queue=task_queue,
        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
        id_conflict_policy=WorkflowIDConflictPolicy.TERMINATE_EXISTING,
    )

    return json.loads(result_json) if isinstance(result_json, str) else result_json


async def start_extraction_job(
    request: ExtractionJobRequest,
    *,
    workflow_id: Optional[str] = None,
    target_host: str = "localhost:7233",
    task_queue: str = "entityxtract",
    namespace: str = "default",
    client: Optional[Client] = None,
):
    """Start an extraction job without waiting for the result.

    Returns a workflow handle that can be used to query progress,
    send signals, or await the final result later.
    """
    if client is None:
        client = await get_client(target_host, namespace=namespace)

    wf_id = workflow_id or f"entityxtract-{uuid.uuid4()}"

    handle = await client.start_workflow(
        EntityExtractionWorkflow.run,
        request.model_dump_json(),
        id=wf_id,
        task_queue=task_queue,
        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE,
        id_conflict_policy=WorkflowIDConflictPolicy.TERMINATE_EXISTING,
    )
    return handle


async def get_job_progress(
    workflow_id: str,
    *,
    target_host: str = "localhost:7233",
    namespace: str = "default",
    client: Optional[Client] = None,
) -> JobProgress:
    """Query a running extraction workflow for its current progress."""
    if client is None:
        client = await get_client(target_host, namespace=namespace)

    handle = client.get_workflow_handle(workflow_id)
    progress_dict = await handle.query(EntityExtractionWorkflow.get_progress)
    return JobProgress.model_validate(progress_dict)


async def approve_entity(
    workflow_id: str,
    entity_name: str,
    edited_data: Any = None,
    *,
    target_host: str = "localhost:7233",
    namespace: str = "default",
    client: Optional[Client] = None,
) -> None:
    """Send an approval signal to a running extraction workflow."""
    if client is None:
        client = await get_client(target_host, namespace=namespace)

    handle = client.get_workflow_handle(workflow_id)
    edited_json = json.dumps(edited_data) if edited_data is not None else None
    await handle.signal(EntityExtractionWorkflow.approve_entity, entity_name, edited_json)


async def reject_entity(
    workflow_id: str,
    entity_name: str,
    reason: str = "",
    *,
    target_host: str = "localhost:7233",
    namespace: str = "default",
    client: Optional[Client] = None,
) -> None:
    """Send a rejection signal to a running extraction workflow."""
    if client is None:
        client = await get_client(target_host, namespace=namespace)

    handle = client.get_workflow_handle(workflow_id)
    await handle.signal(EntityExtractionWorkflow.reject_entity, entity_name, reason)
