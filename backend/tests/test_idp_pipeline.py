import pytest
from fastapi.testclient import TestClient

from idp_pipeline.classifier import classify_document
from idp_pipeline.extractor import extract_fields_for_type
from idp_pipeline.schemas import DocumentType
from idp_pipeline.validators import (
    validate_aadhaar,
    validate_amount,
    validate_date,
    validate_dl_number,
    validate_ifsc,
    validate_pan,
    validate_passport_number,
)
from main import app


client = TestClient(app)


def test_format_validators():
    # IFSC
    valid, val, pen = validate_ifsc("SBIN0002712")
    assert valid is True
    assert val == "SBIN0002712"
    assert pen == 0.0

    # Auto-fix character 'O' instead of '0' in IFSC
    valid_fix, fixed_val, pen_fix = validate_ifsc("SBINO002712")
    assert valid_fix is True
    assert fixed_val == "SBIN0002712"
    assert pen_fix > 0.0

    # PAN
    valid_pan, pan_val, _ = validate_pan("ABCDE1234F")
    assert valid_pan is True
    assert pan_val == "ABCDE1234F"

    # Aadhaar
    valid_aadh, aadh_val, _ = validate_aadhaar("123456789012")
    assert valid_aadh is True
    assert aadh_val == "1234 5678 9012"

    # Date
    valid_d, date_val, _ = validate_date("18/12/1979")
    assert valid_d is True
    assert date_val == "18/12/1979"

    # Amount
    valid_amt, amt_val, _ = validate_amount("50,000.00")
    assert valid_amt is True
    assert amt_val == "50000.00"

    # Passport
    valid_pass, pass_val, _ = validate_passport_number("X1234567")
    assert valid_pass is True
    assert pass_val == "X1234567"

    # DL
    valid_dl, _, _ = validate_dl_number("MH12 2021 0001234")
    assert valid_dl is True


def test_document_classification():
    # Aadhaar anchors
    lines_aadhaar = ["Government of India", "Mera Aadhaar", "UIDAI", "1234 5678 9012"]
    doc_type, conf = classify_document(lines_aadhaar)
    assert doc_type == DocumentType.AADHAAR
    assert conf >= 0.85

    # NACH Mandate anchors
    lines_nach = ["NACH Mandate Instruction", "HDFC Life", "Bank A/c Number", "UMRN"]
    doc_type, conf = classify_document(lines_nach)
    assert doc_type == DocumentType.NACH_MANDATE
    assert conf >= 0.85

    # FATCA anchors
    lines_fatca = ["Annexure Form", "Section 285BA", "Tax Residency", "TIN / PAN"]
    doc_type, conf = classify_document(lines_fatca)
    assert doc_type == DocumentType.FATCA
    assert conf >= 0.85


def test_field_extraction_and_flagging():
    # Test NACH mandate with ambiguous IFSC
    lines_nach = ["31004258912", "JFscSB1N0221", "State Bank of India", "50,000", "Monthly"]
    confs_nach = [0.95, 0.65, 0.95, 0.90, 0.90]
    fields = extract_fields_for_type(DocumentType.NACH_MANDATE, lines_nach, confs_nach)
    
    assert len(fields) == 5
    # The IFSC should be flagged for review due to confidence < 0.85
    ifsc_field = next(f for f in fields if f.field_name == "IFSC Code")
    assert ifsc_field.is_handwritten is True
    assert ifsc_field.is_flagged is True
    assert ifsc_field.confidence < 0.85


def test_api_endpoints():
    # Samples list
    res = client.get("/api/idp/samples")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) >= 10

    # Technical report
    res_rep = client.get("/api/idp/report")
    assert res_rep.status_code == 200
    rep_data = res_rep.json()
    assert rep_data["confidence_threshold"] == 0.85
    assert "threshold_rationale" in rep_data
