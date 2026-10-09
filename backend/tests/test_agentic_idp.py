import pytest

from agentic_idp_pipeline.graph import create_agentic_idp_graph, route_document_type
from agentic_idp_pipeline.proximity import build_proximity_map
from agentic_idp_pipeline.schemas import DocumentType
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
from agentic_idp_pipeline.validators import (
    validate_aadhaar,
    validate_amount,
    validate_date,
    validate_dl_number,
    validate_ifsc,
    validate_pan,
    validate_passport_number,
)


def test_subgraphs_exports():
    """Verify all 12 document extraction nodes and stubs are cleanly exported."""
    nodes = [
        (extract_aadhaar_node, extract_aadhaar_stub),
        (extract_pan_node, extract_pan_stub),
        (extract_passport_node, extract_passport_stub),
        (extract_dl_node, extract_dl_stub),
        (extract_nach_node, extract_nach_stub),
        (extract_fatca_node, extract_fatca_stub),
        (extract_benefit_illustration_node, extract_benefit_illustration_stub),
        (extract_moral_hazard_node, extract_moral_hazard_stub),
        (extract_multiple_policies_node, extract_multiple_policies_stub),
        (extract_suitability_profiler_node, extract_suitability_profiler_stub),
        (extract_assignment_form_node, extract_assignment_form_stub),
        (extract_proposal_form_node, extract_proposal_form_stub),
    ]
    for node, stub in nodes:
        assert callable(node)
        assert callable(stub)


def test_agentic_graph_compilation():
    """Verify LangGraph workflow compiles with conditional edges and all 13 branch nodes."""
    graph = create_agentic_idp_graph()
    assert graph is not None
    expected_nodes = {
        "__start__",
        "reasoning_classifier",
        "extract_aadhaar_stub",
        "extract_pan_stub",
        "extract_dl_stub",
        "extract_passport_stub",
        "extract_nach_stub",
        "extract_fatca_stub",
        "extract_benefit_illustration_stub",
        "extract_moral_hazard_stub",
        "extract_multiple_policies_stub",
        "extract_suitability_profiler_stub",
        "extract_assignment_form_stub",
        "extract_proposal_form_stub",
        "extract_unknown_stub",
    }
    for expected in expected_nodes:
        assert expected in graph.nodes


def test_router_conditional_edges():
    """Verify router maps each document type to its corresponding specialized node."""
    mapping = {
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
    for doc_type, expected_node in mapping.items():
        assert route_document_type({"document_type": doc_type}) == expected_node


def test_spatial_proximity_map_construction():
    """Verify 2D proximity map correctly calculates relative directions and distances."""
    ocr_items = [
        ("Passport No.", 0.95),
        ("X1234567", 0.99),
    ]
    # Token 1 is at y=100, Token 2 is at y=140 (directly below)
    ocr_boxes = [
        [[100.0, 90.0], [200.0, 90.0], [200.0, 110.0], [100.0, 110.0]],
        [[100.0, 130.0], [200.0, 130.0], [200.0, 150.0], [100.0, 150.0]],
    ]

    proximity_map = build_proximity_map(ocr_boxes, ocr_items)
    assert len(proximity_map) == 2

    # Check relationships for Token 0 (Passport No.)
    t0_relations = proximity_map[0]["spatial_neighbors"]
    assert len(t0_relations) == 1
    rel = t0_relations[0]
    assert rel["target_text"] == "X1234567"
    assert rel["rel_direction_y"] == "BELOW"
    assert rel["distance"] == 40.0


def test_validators_suite():
    """Verify all statutory validators in agentic IDP pipeline."""
    # Passport
    valid_p, val_p, _ = validate_passport_number("X1234567")
    assert valid_p and val_p == "X1234567"

    # Aadhaar
    valid_a, val_a, _ = validate_aadhaar("123456789012")
    assert valid_a and val_a == "1234 5678 9012"

    # PAN
    valid_pan, val_pan, _ = validate_pan("ABCDE1234F")
    assert valid_pan and val_pan == "ABCDE1234F"

    # DL
    valid_dl, _, _ = validate_dl_number("MH12 2021 0001234")
    assert valid_dl

    # IFSC
    valid_ifsc, val_ifsc, _ = validate_ifsc("SBIN0002712")
    assert valid_ifsc and val_ifsc == "SBIN0002712"

    # Date
    valid_d, val_d, _ = validate_date("01/01/2030")
    assert valid_d and val_d == "01/01/2030"

    # Amount
    valid_amt, val_amt, _ = validate_amount("1,00,000.00")
    assert valid_amt and val_amt == "100000.00"

    # Policy Number
    from agentic_idp_pipeline.validators import validate_policy_number
    valid_pol, pol_val, _ = validate_policy_number("1500131601025")
    assert valid_pol is True
    assert pol_val == "1500131601025"

    # Form version barcode rejection
    valid_bad, _, pen = validate_policy_number("P029111911095V020")
    assert valid_bad is False

