from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    AADHAAR = "Aadhaar Card"
    PAN = "PAN Card"
    DRIVING_LICENCE = "Driving Licence"
    PASSPORT = "Passport"
    NACH_MANDATE = "NACH / ECS Mandate"
    FATCA = "FATCA Annexure Form"
    BENEFIT_ILLUSTRATION = "Benefit Illustration Declaration"
    MORAL_HAZARD = "Moral Hazard Questionnaire"
    MULTIPLE_POLICIES = "Multiple Policies Consent Form"
    SUITABILITY_PROFILER = "Suitability Profiler Declaration"
    UNKNOWN = "Unknown Document"


# Exact required fields mapping as per Question 3 specifications
DOCUMENT_TARGET_FIELDS: Dict[DocumentType, List[str]] = {
    DocumentType.AADHAAR: [
        "Aadhaar Number",
        "Full Name",
        "Date of Birth",
        "Address",
    ],
    DocumentType.PAN: [
        "PAN Number",
        "Full Name",
        "Father's Name",
        "Date of Birth",
    ],
    DocumentType.DRIVING_LICENCE: [
        "DL Number",
        "Name",
        "Date of Issue",
        "Valid Till date",
    ],
    DocumentType.PASSPORT: [
        "Passport Number",
        "Date of Birth",
        "Date of Expiry",
        "MRZ Line 2",
    ],
    DocumentType.NACH_MANDATE: [
        "Bank Account Number",
        "IFSC Code",
        "Bank Name",
        "Amount (figures)",
        "Frequency",
    ],
    DocumentType.FATCA: [
        "Policy Number",
        "TIN / PAN",
        "Father's Name",
        "Place of Birth",
        "Nationality",
    ],
    DocumentType.BENEFIT_ILLUSTRATION: [
        "Application Number",
        "Policyholder Name",
        "Date",
        "Place",
    ],
    DocumentType.MORAL_HAZARD: [
        "Application Number",
        "Name of Life Assured",
        "Nominee Relationship",
        "Date",
        "Place",
    ],
    DocumentType.MULTIPLE_POLICIES: [
        "Proposer Name",
        "Reason for Multiple Policies (selected checkbox)",
        "Date",
        "Place",
    ],
    DocumentType.SUITABILITY_PROFILER: [
        "Application Number",
        "Name of Life Assured",
        "Name of Agent/SP",
        "Date",
        "Place",
    ],
}


class FieldResult(BaseModel):
    field_name: str
    value: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_handwritten: bool = False
    is_flagged: bool = False
    flag_reason: Optional[str] = None
    human_verified: bool = False


class DocumentProcessResponse(BaseModel):
    document_id: str
    filename: str
    preview_url: Optional[str] = None
    document_type: DocumentType
    classification_confidence: float
    fields: List[FieldResult]
    overall_confidence: float
    needs_review: bool
    flagged_count: int
    processing_time_sec: float
    raw_ocr_lines: Optional[List[str]] = None


class TriageUpdateRequest(BaseModel):
    document_id: str
    field_name: str
    updated_value: str
    confirm_as_verified: bool = True


class ReclassifyRequest(BaseModel):
    document_id: str
    chosen_type: DocumentType
