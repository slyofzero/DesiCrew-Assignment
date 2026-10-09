"""Subgraphs package exporting all 12 document-specific extraction nodes and stubs."""

from agentic_idp_pipeline.subgraphs.aadhar import (
    extract_aadhaar_node,
    extract_aadhaar_stub,
)
from agentic_idp_pipeline.subgraphs.pan import (
    extract_pan_node,
    extract_pan_stub,
)
from agentic_idp_pipeline.subgraphs.passport import (
    extract_passport_node,
    extract_passport_stub,
)
from agentic_idp_pipeline.subgraphs.driving_licence import (
    extract_dl_node,
    extract_dl_stub,
)
from agentic_idp_pipeline.subgraphs.nach import (
    extract_nach_node,
    extract_nach_stub,
)
from agentic_idp_pipeline.subgraphs.fatca import (
    extract_fatca_node,
    extract_fatca_stub,
)
from agentic_idp_pipeline.subgraphs.benefit_illustration import (
    extract_benefit_illustration_node,
    extract_benefit_illustration_stub,
)
from agentic_idp_pipeline.subgraphs.moral_hazard import (
    extract_moral_hazard_node,
    extract_moral_hazard_stub,
)
from agentic_idp_pipeline.subgraphs.multiple_policies import (
    extract_multiple_policies_node,
    extract_multiple_policies_stub,
)
from agentic_idp_pipeline.subgraphs.suitability_profiler import (
    extract_suitability_profiler_node,
    extract_suitability_profiler_stub,
)
from agentic_idp_pipeline.subgraphs.assignment_form import (
    extract_assignment_form_node,
    extract_assignment_form_stub,
)
from agentic_idp_pipeline.subgraphs.proposal_form import (
    extract_proposal_form_node,
    extract_proposal_form_stub,
)

__all__ = [
    "extract_aadhaar_node",
    "extract_aadhaar_stub",
    "extract_pan_node",
    "extract_pan_stub",
    "extract_passport_node",
    "extract_passport_stub",
    "extract_dl_node",
    "extract_dl_stub",
    "extract_nach_node",
    "extract_nach_stub",
    "extract_fatca_node",
    "extract_fatca_stub",
    "extract_benefit_illustration_node",
    "extract_benefit_illustration_stub",
    "extract_moral_hazard_node",
    "extract_moral_hazard_stub",
    "extract_multiple_policies_node",
    "extract_multiple_policies_stub",
    "extract_suitability_profiler_node",
    "extract_suitability_profiler_stub",
    "extract_assignment_form_node",
    "extract_assignment_form_stub",
    "extract_proposal_form_node",
    "extract_proposal_form_stub",
]
