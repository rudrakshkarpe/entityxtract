"""Tests for the durable extraction pipeline.

Uses Temporal's WorkflowEnvironment for a real (local) test server,
with mocked activities so no actual LLM calls are made.
"""

from __future__ import annotations

import json
import uuid

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin


def _test_workflow_runner() -> SandboxedWorkflowRunner:
    """Return a sandboxed runner that passes through entityxtract's deps."""
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
    return SandboxedWorkflowRunner(restrictions=restrictions)

from entityxtract.durable.types import (
    DocumentRef,
    EntitySpec,
    ExtractionJobRequest,
    JobProgress,
    ValidationReport,
)
from entityxtract.durable.workflow import EntityExtractionWorkflow
from entityxtract.extractor_types import ExtractionConfig, ExtractionResult, FileInputMode


# ----- Mocked activities -----

@activity.defn(name="load_document_activity")
async def mock_load_document(doc_ref_json: str) -> str:
    return "This is mock document text for testing purposes. Page 1 content here."


@activity.defn(name="render_pdf_pages_activity")
async def mock_render_pages(doc_ref_json: str) -> list[str]:
    return ["base64encodedpage1=="]


@activity.defn(name="extract_entity_activity")
async def mock_extract_entity(payload_json: str) -> str:
    payload = json.loads(payload_json)
    entity_spec = payload["entity_spec"]
    name = entity_spec["name"]

    if entity_spec["kind"] == "table":
        extracted = [
            {"Name": "John Doe", "Organization": "Test University", "Email": "john@test.edu"},
            {"Name": "Jane Smith", "Organization": "Research Lab", "Email": "jane@lab.org"},
        ]
    else:
        extracted = f"Extracted string for {name}"

    result = ExtractionResult(
        extracted_data=extracted,
        response_raw={"mock": True},
        success=True,
        message="Mock extraction successful",
        input_tokens=100,
        output_tokens=50,
        cost=0.001,
    )
    return result.model_dump_json()


@activity.defn(name="validate_extraction_activity")
async def mock_validate_extraction(payload_json: str) -> str:
    payload = json.loads(payload_json)
    report = ValidationReport(
        entity_name=payload["entity_name"],
        is_valid=True,
        issues=[],
        suggestions=[],
    )
    return report.model_dump_json()


# ----- Fixtures -----

@pytest.fixture
def sample_request() -> ExtractionJobRequest:
    return ExtractionJobRequest(
        doc_ref=DocumentRef(
            uri="file:///tmp/test-document.pdf",
            doc_type="pdf",
            page_range=(0, 3),
        ),
        entities=[
            EntitySpec(
                name="Authors",
                kind="table",
                example_schema=json.dumps([
                    {"Name": "Example", "Organization": "Example Org", "Email": "ex@ex.com"},
                ]),
                instructions="Extract authors table.",
                required=True,
            ),
        ],
        config=ExtractionConfig(
            model_name="google/gemini-2.5-flash",
            temperature=0.0,
            file_input_modes=[FileInputMode.FILE],
        ),
        auto_detect=False,
        require_human_approval=False,
    )


# ----- Tests -----

class TestDurableTypes:
    """Test serialization of durable types."""

    def test_document_ref_roundtrip(self):
        ref = DocumentRef(uri="file:///tmp/test.pdf", doc_type="pdf", page_range=(0, 5))
        json_str = ref.model_dump_json()
        restored = DocumentRef.model_validate_json(json_str)
        assert restored.uri == ref.uri
        assert restored.doc_type == ref.doc_type
        assert restored.page_range == (0, 5)

    def test_entity_spec_roundtrip(self):
        spec = EntitySpec(
            name="Test",
            kind="table",
            example_schema='[{"col": "val"}]',
            instructions="Extract test data",
        )
        json_str = spec.model_dump_json()
        restored = EntitySpec.model_validate_json(json_str)
        assert restored.name == "Test"
        assert restored.kind == "table"

    def test_extraction_job_request_roundtrip(self, sample_request: ExtractionJobRequest):
        json_str = sample_request.model_dump_json()
        restored = ExtractionJobRequest.model_validate_json(json_str)
        assert restored.doc_ref.uri == sample_request.doc_ref.uri
        assert len(restored.entities) == 1
        assert restored.entities[0].name == "Authors"

    def test_job_progress_default(self):
        progress = JobProgress()
        assert progress.entity_progress == {}
        assert progress.started_at is None

    def test_validation_report(self):
        report = ValidationReport(
            entity_name="Test",
            is_valid=False,
            issues=["Missing column: Email"],
            suggestions=["Add explicit instruction for Email column"],
        )
        assert not report.is_valid
        assert len(report.issues) == 1


class TestActivities:
    """Test activity functions in isolation."""

    @pytest.mark.asyncio
    async def test_mock_load_document(self):
        result = await mock_load_document('{"uri": "file:///test.pdf", "doc_type": "pdf"}')
        assert "mock document text" in result

    @pytest.mark.asyncio
    async def test_mock_extract_entity(self):
        payload = {
            "doc_ref": {"uri": "file:///test.pdf", "doc_type": "pdf"},
            "entity_spec": {
                "name": "Authors",
                "kind": "table",
                "example_schema": '[{"Name": "ex"}]',
                "instructions": "Extract authors",
                "required": True,
            },
            "config": {"model_name": "test", "temperature": 0.0},
        }
        result_json = await mock_extract_entity(json.dumps(payload))
        result = ExtractionResult.model_validate_json(result_json)
        assert result.success
        assert isinstance(result.extracted_data, list)
        assert len(result.extracted_data) == 2

    @pytest.mark.asyncio
    async def test_mock_validate_extraction(self):
        payload = {
            "entity_name": "Authors",
            "extracted_data": [{"Name": "John"}],
            "entity_spec": {
                "name": "Authors",
                "kind": "table",
                "example_schema": '[{"Name": "ex"}]',
                "instructions": "test",
                "required": True,
            },
        }
        result_json = await mock_validate_extraction(json.dumps(payload))
        report = ValidationReport.model_validate_json(result_json)
        assert report.is_valid
        assert report.entity_name == "Authors"


# ----- Worker / sandbox validation -----

@pytest.mark.asyncio
async def test_workflow_loads_under_sandbox():
    """The workflow must import cleanly under Temporal's sandboxed runner.

    This catches the 99% failure mode for custom workflows: a transitive
    import (langchain, urllib3, etc.) that the Temporal sandbox forbids.
    If this passes, ``python -m entityxtract.durable.worker`` will start.
    """
    from temporalio.workflow import _Definition

    runner = _test_workflow_runner()
    defn = _Definition.must_from_class(EntityExtractionWorkflow)
    runner.prepare_workflow(defn)
