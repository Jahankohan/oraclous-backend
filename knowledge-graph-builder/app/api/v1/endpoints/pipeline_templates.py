"""Pipeline template catalogue endpoints (ORA-352).

Routes:
  GET /pipeline-templates              — list all built-in template summaries
  GET /pipeline-templates/{id}         — fetch a single full template (nodes + edges + params)

No authentication is required: templates are static, read-only, and not tenant-specific.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.schemas.pipeline_template_schemas import (
    PIPELINE_TEMPLATES,
    PipelineTemplate,
    PipelineTemplateCatalogueResponse,
    PipelineTemplateSummary,
)

router = APIRouter()


@router.get(
    "/pipeline-templates",
    response_model=PipelineTemplateCatalogueResponse,
    summary="List built-in pipeline template summaries",
)
async def list_pipeline_templates() -> PipelineTemplateCatalogueResponse:
    """Return lightweight metadata for all built-in pipeline templates.

    The response omits nodes and edges to keep payload small; fetch a single
    template by id to get the full skeleton for canvas loading.
    """
    summaries = [
        PipelineTemplateSummary(
            id=t.id,
            name=t.name,
            description=t.description,
            domain=t.domain,
            tags=t.tags,
            node_count=len(t.nodes),
            param_count=len(t.params),
        )
        for t in PIPELINE_TEMPLATES.values()
    ]
    return PipelineTemplateCatalogueResponse(templates=summaries, total=len(summaries))


@router.get(
    "/pipeline-templates/{template_id}",
    response_model=PipelineTemplate,
    summary="Get a single pipeline template with full nodes, edges, and params",
)
async def get_pipeline_template(template_id: str) -> PipelineTemplate:
    """Return the full template skeleton including nodes, edges, and parameter descriptors.

    The frontend uses this to:
    1. Display a parameter substitution dialog (one field per ``params`` entry).
    2. Instantiate nodes and edges on the React Flow canvas after substitution.
    """
    template = PIPELINE_TEMPLATES.get(template_id)
    if template is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline template '{template_id}' not found.",
        )
    return template
