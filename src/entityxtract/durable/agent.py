"""Pydantic AI coordinator agent wrapped as a TemporalAgent.

The coordinator agent orchestrates entity extraction: it calls the
``extract_entity`` tool for each entity, validates results, and can
retry with refined prompts.  When wrapped in ``TemporalAgent``, every
model call and tool invocation becomes a Temporal activity automatically.

Because ``TemporalAgent`` requires at least one pre-registered Model
instance, we provide a factory function ``build_temporal_coordinator``
that callers invoke with their desired model(s).  A convenience
``get_temporal_coordinator`` returns a cached singleton using the
default ``openai:gpt-4o`` model.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from temporalio import workflow

from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import TemporalAgent
from pydantic_ai.models import infer_model, Model

from .types import Deps

COORDINATOR_INSTRUCTIONS = """\
You are the entityxtract coordinator agent.  Your job is to reliably
extract structured data from documents.

You will be given:
- A list of entity specifications (name, kind, schema, instructions)
- A reference to the source document

For EACH entity, follow these steps:
1. Call ``extract_entity`` with the entity spec and document reference.
2. Call ``validate_extraction`` on the result.
3. If validation fails and you have attempts remaining, analyse the
   issues/suggestions and call ``extract_entity`` again with a refined
   prompt (add the validation feedback to the instructions).
4. After a maximum of 3 extraction attempts per entity, move on.

If ``auto_detect`` is enabled, first call ``detect_entities`` to
discover additional entities in the document.

If ``require_human_approval`` is enabled, call
``request_human_approval`` for each successful extraction before
marking it complete.

Always return a JSON object mapping entity names to their final
extraction results (the raw JSON returned by ``extract_entity``).
"""


def _make_agent() -> Agent[Deps, str]:
    """Create a fresh coordinator Agent with tools registered."""
    agent = Agent(
        model=None,
        deps_type=Deps,
        name="entityxtract_coordinator",
        instructions=COORDINATOR_INSTRUCTIONS,
    )

    from . import tools as _tools

    agent.tool(_tools.extract_entity)
    agent.tool(_tools.validate_extraction)
    agent.tool(_tools.detect_entities)
    agent.tool(_tools.request_human_approval)
    return agent


def build_temporal_coordinator(
    default_model: str | Model = "openai:gpt-4o",
    extra_models: Optional[dict[str, str | Model]] = None,
) -> TemporalAgent[Deps, str]:
    """Build a TemporalAgent wrapping the coordinator.

    Args:
        default_model: Model string or instance (requires API key env var).
        extra_models: Optional dict of additional named models.
    """
    agent = _make_agent()

    default = infer_model(default_model) if isinstance(default_model, str) else default_model
    agent._model = default  # type: ignore[assignment]

    models: dict[str, Model] = {}
    if extra_models:
        for name, m in extra_models.items():
            models[name] = infer_model(m) if isinstance(m, str) else m

    return TemporalAgent(
        agent,
        models=models if models else None,
        activity_config=workflow.ActivityConfig(
            start_to_close_timeout=timedelta(seconds=120),
        ),
        model_activity_config=workflow.ActivityConfig(
            start_to_close_timeout=timedelta(seconds=120),
        ),
    )


_cached_coordinator: Optional[TemporalAgent[Deps, str]] = None


def get_temporal_coordinator(
    default_model: str | Model = "openai:gpt-4o",
) -> TemporalAgent[Deps, str]:
    """Return a cached TemporalAgent singleton (created on first call).

    This avoids importing openai (and needing an API key) at module load time.
    """
    global _cached_coordinator
    if _cached_coordinator is None:
        _cached_coordinator = build_temporal_coordinator(default_model)
    return _cached_coordinator
