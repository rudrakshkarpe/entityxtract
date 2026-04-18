"""Non-agent Temporal activities for document I/O.

These are registered directly on the Worker (not via TemporalAgent)
because they are infrastructure operations, not agent tools.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from temporalio import activity
from temporalio.exceptions import ApplicationError

from .types import DocumentRef, EntitySpec


def _resolve_uri(uri: str) -> Path:
    """Turn a file:// URI into a local Path, or raise for unsupported schemes."""
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        return Path(parsed.path)
    raise ApplicationError(
        f"Unsupported URI scheme: {parsed.scheme!r}. Only file:// is currently supported.",
        type="UnsupportedURIScheme",
        non_retryable=True,
    )


@activity.defn
async def load_document_activity(doc_ref_json: str) -> str:
    """Materialise a DocumentRef into extracted text content.

    Returns the document text as a JSON string so it fits in history
    without exceeding the 2 MB payload limit for most documents.
    """
    from entityxtract.extractor_types import Document

    doc_ref = DocumentRef.model_validate_json(doc_ref_json)
    path = _resolve_uri(doc_ref.uri)

    if not path.exists():
        raise ApplicationError(
            f"Document not found: {path}",
            type="DocumentNotFound",
            non_retryable=True,
        )

    page_range = tuple(doc_ref.page_range) if doc_ref.page_range else None
    doc = Document(file_path=path, page_range=page_range)
    return doc.text


@activity.defn
async def render_pdf_pages_activity(doc_ref_json: str) -> list[str]:
    """Render PDF pages as base64-encoded JPEG strings.

    Returns a list of base64 strings, one per page.  For very large
    documents, consider splitting into smaller page ranges.
    """
    from entityxtract.pdf.extractor import pdf_to_image

    doc_ref = DocumentRef.model_validate_json(doc_ref_json)
    path = _resolve_uri(doc_ref.uri)

    if doc_ref.doc_type.lower() != "pdf":
        raise ApplicationError(
            f"render_pdf_pages only supports PDFs, got {doc_ref.doc_type!r}",
            type="InvalidDocType",
            non_retryable=True,
        )

    with open(path, "rb") as f:
        pdf_bytes = f.read()

    if doc_ref.page_range:
        from entityxtract.pdf.extractor import trim_pdf_pages
        pdf_bytes = trim_pdf_pages(pdf_bytes, doc_ref.page_range[0], doc_ref.page_range[1])

    images = pdf_to_image(pdf_bytes, scale=2, combine_pages=False)
    if not isinstance(images, list):
        images = [images]

    from io import BytesIO

    result: list[str] = []
    for img in images:
        if getattr(img, "mode", None) != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        result.append(base64.b64encode(buf.getvalue()).decode())

    activity.heartbeat(f"Rendered {len(result)} pages")
    return result


@activity.defn
async def extract_entity_activity(payload_json: str) -> str:
    """Run the core LLM extraction for a single entity.

    Accepts and returns JSON strings to keep payloads serializable.
    This wraps the existing ``extract_object`` logic but is called
    from the coordinator agent as a tool.
    """
    from entityxtract.extractor import extract_object
    from entityxtract.extractor_types import (
        Document,
        ExtractionConfig,
        FileInputMode,
        StringToExtract,
        TableToExtract,
    )
    import polars as pl

    payload = json.loads(payload_json)
    doc_ref = DocumentRef.model_validate(payload["doc_ref"])
    entity_spec = EntitySpec.model_validate(payload["entity_spec"])
    config_dict = payload.get("config", {})

    path = _resolve_uri(doc_ref.uri)
    page_range = tuple(doc_ref.page_range) if doc_ref.page_range else None
    doc = Document(file_path=path, page_range=page_range)

    if entity_spec.kind == "table":
        schema_data = json.loads(entity_spec.example_schema)
        example_df = pl.DataFrame(schema_data)
        obj = TableToExtract(
            name=entity_spec.name,
            example_table=example_df,
            instructions=entity_spec.instructions,
            required=entity_spec.required,
        )
    else:
        obj = StringToExtract(
            name=entity_spec.name,
            example_string=entity_spec.example_schema,
            instructions=entity_spec.instructions,
            required=entity_spec.required,
        )

    config = ExtractionConfig(
        model_name=config_dict.get("model_name", "google/gemini-2.5-flash"),
        temperature=config_dict.get("temperature", 0.0),
        max_retries=1,  # Temporal handles retries
        file_input_modes=[FileInputMode(m) for m in config_dict.get("file_input_modes", ["file"])],
        calculate_costs=config_dict.get("calculate_costs", False),
    )

    activity.heartbeat(f"Extracting {entity_spec.name}")
    result = extract_object(doc, obj, config)
    return result.model_dump_json()


@activity.defn
async def validate_extraction_activity(payload_json: str) -> str:
    """Structurally validate an extraction result.

    Checks that required fields are present, data types look reasonable,
    and the result isn't empty.  Returns a ValidationReport as JSON.
    """
    from .types import ValidationReport

    payload = json.loads(payload_json)
    entity_name = payload["entity_name"]
    extracted_json = payload.get("extracted_data")
    entity_spec = EntitySpec.model_validate(payload["entity_spec"])

    issues: list[str] = []
    suggestions: list[str] = []

    if extracted_json is None:
        issues.append("Extraction returned no data")
    elif entity_spec.kind == "table":
        if isinstance(extracted_json, list):
            if len(extracted_json) == 0:
                issues.append("Extracted table is empty (zero rows)")
            else:
                schema = json.loads(entity_spec.example_schema)
                if isinstance(schema, list) and schema:
                    expected_cols = set(schema[0].keys())
                    actual_cols = set(extracted_json[0].keys())
                    missing = expected_cols - actual_cols
                    if missing:
                        issues.append(f"Missing columns: {missing}")
                        suggestions.append(
                            f"Retry extraction with explicit instruction to include columns: {missing}"
                        )
        else:
            issues.append(f"Expected a list of dicts for table, got {type(extracted_json).__name__}")
    elif entity_spec.kind == "string":
        if not isinstance(extracted_json, (str, dict)):
            issues.append(f"Expected string or dict, got {type(extracted_json).__name__}")
        if isinstance(extracted_json, str) and len(extracted_json.strip()) == 0:
            issues.append("Extracted string is empty")

    report = ValidationReport(
        entity_name=entity_name,
        is_valid=len(issues) == 0,
        issues=issues,
        suggestions=suggestions,
    )
    return report.model_dump_json()
