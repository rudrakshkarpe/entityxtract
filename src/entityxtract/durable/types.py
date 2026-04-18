"""Serializable types for the Temporal durable extraction pipeline.

These types travel through Temporal workflow history, so they must be
Pydantic-serializable and stay within the 2 MB payload limit.
Binary document data is deliberately excluded — use ``DocumentRef``
and materialise bytes inside activities via ``load_document_activity``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from entityxtract.extractor_types import (
    ExtractionConfig,
    ExtractionResult,
    ExtractableObjectTypes,
    FileInputMode,
)


class DocumentRef(BaseModel):
    """Lightweight reference to a document that fits in workflow history."""

    uri: str = Field(
        ...,
        description="file:// path, s3:// URI, or https:// URL to the source document",
    )
    doc_type: str = Field(
        ...,
        description="pdf, txt, png, etc.",
    )
    page_range: Optional[tuple[int, int]] = Field(
        default=None,
        description="Optional (start, end) 0-indexed page slice for PDFs",
    )


class EntitySpec(BaseModel):
    """Serializable description of one entity to extract.

    Mirrors TableToExtract / StringToExtract but as a plain JSON-safe dict
    so it can travel through Temporal payloads.
    """

    name: str
    kind: str = Field(description="'table' or 'string'")
    example_schema: str = Field(
        description="JSON representation of example_table or example_string",
    )
    instructions: str
    required: bool = True


class ExtractionJobRequest(BaseModel):
    """Top-level input to EntityExtractionWorkflow."""

    doc_ref: DocumentRef
    entities: list[EntitySpec]
    config: ExtractionConfig = Field(default_factory=ExtractionConfig)
    auto_detect: bool = False
    require_human_approval: bool = False


class EntityStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"


class EntityProgress(BaseModel):
    name: str
    status: EntityStatus = EntityStatus.PENDING
    attempts: int = 0
    last_error: Optional[str] = None


class JobProgress(BaseModel):
    """Queryable progress snapshot returned by ``get_progress``."""

    entity_progress: dict[str, EntityProgress] = Field(default_factory=dict)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


class ValidationReport(BaseModel):
    """Result of a structural/semantic validation pass."""

    entity_name: str
    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class SuggestedEntity(BaseModel):
    """An entity auto-detected in the document."""

    name: str
    kind: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)


class ApprovalDecision(BaseModel):
    entity_name: str
    approved: bool
    edited_data: Optional[Any] = None
    reason: Optional[str] = None


@dataclass
class Deps:
    """Serializable dependencies passed to TemporalAgent.run().

    API keys flow through here instead of via env-var reads inside the
    workflow (which are forbidden).
    """

    openai_api_key: str = ""
    openai_api_base: str = "https://openrouter.ai/api/v1"
    model_name: str = "google/gemini-2.5-flash"
    temperature: float = 0.0
    file_input_modes: list[str] = field(default_factory=lambda: ["file"])
    calculate_costs: bool = False
