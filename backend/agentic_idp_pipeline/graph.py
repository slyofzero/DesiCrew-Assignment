import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from typing_extensions import TypedDict

from langgraph.constants import END, START
from langgraph.graph.state import StateGraph

from agentic_idp_pipeline.classifier import classify_document_with_llm
from agentic_idp_pipeline.schemas import (
    DOCUMENT_TARGET_FIELDS,
    DocumentType,
    FieldResult,
)
from agentic_idp_pipeline.subgraphs import (
    extract_aadhaar_node,
    extract_aadhaar_stub,
    extract_pan_node,
    extract_pan_stub,
    extract_passport_node,
    extract_passport_stub,
    extract_dl_node,
    extract_dl_stub,
    extract_nach_node,
    extract_nach_stub,
    extract_fatca_node,
    extract_fatca_stub,
    extract_benefit_illustration_node,
    extract_benefit_illustration_stub,
    extract_moral_hazard_node,
    extract_moral_hazard_stub,
    extract_multiple_policies_node,
    extract_multiple_policies_stub,
    extract_suitability_profiler_node,
    extract_suitability_profiler_stub,
    extract_assignment_form_node,
    extract_assignment_form_stub,
    extract_proposal_form_node,
    extract_proposal_form_stub,
)


class AgenticIDPState(TypedDict, total=False):
    """
    Unified state for the Agentic IDP LangGraph workflow.
    `ocr_items` stores a singular array of 2D tuples: [(<line>, <confidence>), ...].
    """
    file_path: str
    filename: str
    preview_img: str
    override_type: Optional[DocumentType]

    # Singular array of 2D tuples: (<line>, <confidence>)
    ocr_items: List[Tuple[str, float]]
    ocr_boxes: List[List[List[float]]]

    # Classification Reasoner Node outputs
    document_type: DocumentType
    classification_confidence: float
    classification_reasoning: str

    # Field Extraction & Triage outputs
    fields: List[FieldResult]
    overall_confidence: float
    needs_review: bool
    flagged_count: int

    # Agentic Thought Trajectory
    thought_trajectory: List[str]
    processing_time_sec: float


def reasoning_classification_node(state: AgenticIDPState) -> Dict[str, Any]:
    """
    First Node: LangGraph Classification Reasoner.
    Analyzes the 2D tuple array [(line, confidence)] to reason over
    headers, layout anchors, and statutory body markers.
    """
    override_type = state.get("override_type")
    if override_type is not None:
        doc_type = override_type
        conf = 1.0 if override_type != DocumentType.UNKNOWN else 0.30
        reasoning = f"Manually designated document type as '{override_type.value}' by human reviewer."
        thought = (
            f"Step 1 [Agentic Reasoner - Document Classification]: "
            f"Manual override applied -> '{doc_type.value}' (100% confidence). {reasoning}"
        )
        return {
            "document_type": doc_type,
            "classification_confidence": conf,
            "classification_reasoning": reasoning,
            "thought_trajectory": [thought],
        }

    ocr_items = state.get("ocr_items", [])
    if not ocr_items:
        return {
            "document_type": DocumentType.UNKNOWN,
            "classification_confidence": 0.0,
            "classification_reasoning": "No OCR text extracted from file.",
            "thought_trajectory": [
                "Step 1 [Agentic Reasoner - Document Classification]: Failed - No OCR tokens extracted."
            ],
        }

    # Perform LLM Semantic Classification with reasoning on 2D tuples
    doc_type, conf, reasoning = classify_document_with_llm(ocr_items)

    thought = (
        f"Step 1 [Agentic Reasoner - Document Classification]: "
        f"Identified as '{doc_type.value}' ({int(conf * 100)}% confidence). "
        f"Reasoning: {reasoning}"
    )

    return {
        "document_type": doc_type,
        "classification_confidence": conf,
        "classification_reasoning": reasoning,
        "thought_trajectory": [thought],
    }


def route_document_type(state: AgenticIDPState) -> str:
    """
    Conditional edge router:
    Examines the classified document_type produced by the first reasoning node
    and routes the state to the corresponding specialized document extraction subgraph stub.
    """
    doc_type = state.get("document_type", DocumentType.UNKNOWN)

    route_map = {
        DocumentType.AADHAAR: "extract_aadhaar_stub",
        DocumentType.PAN: "extract_pan_stub",
        DocumentType.DRIVING_LICENCE: "extract_dl_stub",
        DocumentType.PASSPORT: "extract_passport_stub",
        DocumentType.NACH_MANDATE: "extract_nach_stub",
        DocumentType.FATCA: "extract_fatca_stub",
        DocumentType.BENEFIT_ILLUSTRATION: "extract_benefit_illustration_stub",
        DocumentType.MORAL_HAZARD: "extract_moral_hazard_stub",
        DocumentType.MULTIPLE_POLICIES: "extract_multiple_policies_stub",
        DocumentType.SUITABILITY_PROFILER: "extract_suitability_profiler_stub",
        DocumentType.ASSIGNMENT_FORM: "extract_assignment_form_stub",
        DocumentType.PROPOSAL_FORM: "extract_proposal_form_stub",
        DocumentType.UNKNOWN: "extract_unknown_stub",
    }
    return route_map.get(doc_type, "extract_unknown_stub")


def _run_subgraph_stub(
    doc_type: DocumentType, subgraph_name: str, state: AgenticIDPState
) -> Dict[str, Any]:
    """
    Lean stub for document-specific extraction subgraphs.
    Directly receives ocr_items as the singular array of 2D tuples [(<line>, <confidence>)].
    """
    trajectory = list(state.get("thought_trajectory", []))
    trajectory.append(
        f"Step 2 [Conditional Router -> Subgraph]: Routed to '{subgraph_name}' based on classified type '{doc_type.value}'."
    )

    if doc_type == DocumentType.UNKNOWN:
        trajectory.append("Step 3 [Stub Execution]: Skipped for unclassified document.")
        return {
            "fields": [],
            "overall_confidence": state.get("classification_confidence", 0.0),
            "flagged_count": 0,
            "needs_review": True,
            "thought_trajectory": trajectory,
        }

    ocr_lines: List[Tuple[str, float]] = state.get("ocr_items", [])
    target_field_names = DOCUMENT_TARGET_FIELDS.get(doc_type, [])

    # Stub extraction: looks for direct label matches in 2D tuples or generates placeholder
    fields: List[FieldResult] = []
    for field_name in target_field_names:
        cand_val = None
        cand_conf = 0.85

        for line, conf in ocr_lines:
            lower = line.lower()
            # Match field label in OCR line
            if any(term in lower for term in field_name.lower().split() if len(term) > 3):
                cand_val = line.strip()
                cand_conf = conf
                break

        fields.append(
            FieldResult(
                field_name=field_name,
                value=cand_val,
                confidence=round(cand_conf, 2),
                is_flagged=(cand_conf < 0.85 or not cand_val),
                flag_reason=None if cand_val else f"'{field_name}' stub pending specialized subgraph implementation.",
            )
        )

    flagged_count = sum(1 for f in fields if f.is_flagged)
    overall_conf = (
        round(sum(f.confidence for f in fields) / len(fields), 2)
        if fields
        else state.get("classification_confidence", 0.0)
    )

    trajectory.append(
        f"Step 3 [Stub Execution ({subgraph_name})]: Processed {len(ocr_lines)} 2D OCR tuples [(line, conf)]. "
        f"Generated {len(fields)} target field stubs for upcoming subgraph implementation."
    )

    return {
        "fields": fields,
        "overall_confidence": overall_conf,
        "flagged_count": flagged_count,
        "needs_review": flagged_count > 0,
        "thought_trajectory": trajectory,
    }


# Note: All 12 specialized document subgraphs are imported directly from subgraphs/


def extract_unknown_stub(state: AgenticIDPState) -> Dict[str, Any]:
    return _run_subgraph_stub(DocumentType.UNKNOWN, "Unknown Document Subgraph Stub", state)


# ---------------------------------------------------------------------------
# LangGraph Assembly with Conditional Routing
# ---------------------------------------------------------------------------
def create_agentic_idp_graph():
    """
    Constructs and compiles the StateGraph for the Agentic IDP Pipeline.
    Conditionally routes from reasoning_classifier to specialized document subgraphs.
    """
    workflow = StateGraph(AgenticIDPState)

    # 1. Reasoning Node (Classifier)
    workflow.add_node("reasoning_classifier", reasoning_classification_node)

    # 2. Document-Specific Subgraphs
    subgraph_stubs = {
        "extract_aadhaar_stub": extract_aadhaar_node,
        "extract_pan_stub": extract_pan_node,
        "extract_dl_stub": extract_dl_node,
        "extract_passport_stub": extract_passport_node,
        "extract_nach_stub": extract_nach_node,
        "extract_fatca_stub": extract_fatca_node,
        "extract_benefit_illustration_stub": extract_benefit_illustration_node,
        "extract_moral_hazard_stub": extract_moral_hazard_node,
        "extract_multiple_policies_stub": extract_multiple_policies_node,
        "extract_suitability_profiler_stub": extract_suitability_profiler_node,
        "extract_assignment_form_stub": extract_assignment_form_node,
        "extract_proposal_form_stub": extract_proposal_form_node,
        "extract_unknown_stub": extract_unknown_stub,
    }


    for node_name, stub_func in subgraph_stubs.items():
        workflow.add_node(node_name, stub_func)
        workflow.add_edge(node_name, END)

    # Start -> Classifier Reasoning Node
    workflow.add_edge(START, "reasoning_classifier")

    # Conditional Edge: Router from reasoning_classifier to each specialized subgraph stub
    workflow.add_conditional_edges(
        "reasoning_classifier",
        route_document_type,
        {name: name for name in subgraph_stubs.keys()},
    )

    return workflow.compile()
