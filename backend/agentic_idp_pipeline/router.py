import json
import os
import shutil
import tempfile
from typing import Dict, List, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from agentic_idp_pipeline.extractor import extract_text_from_file, process_document
from agentic_idp_pipeline.schemas import (
    DOCUMENT_TARGET_FIELDS,
    DocumentProcessResponse,
    DocumentType,
    FieldResult,
    ReclassifyRequest,
    TriageUpdateRequest,
)

router = APIRouter(prefix="/idp", tags=["IDP Pipeline"])

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.dirname(BASE_DIR)
Q3_DIR = os.path.join(WORKSPACE_DIR, "Question 3")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "idp_uploads")
AUDIT_FILE = os.path.join(BASE_DIR, "data", "idp_triage_audit.json")
TEMP_PREVIEW_DIR = os.path.join(tempfile.gettempdir(), "idp_previews")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_PREVIEW_DIR, exist_ok=True)
os.makedirs(os.path.dirname(AUDIT_FILE), exist_ok=True)


# In-memory cache for processed documents to allow quick reclassification and previews
_processed_docs_cache: Dict[str, DocumentProcessResponse] = {}
_doc_file_paths: Dict[str, str] = {}


@router.get("/samples")
def list_sample_documents():
    """Lists available pre-loaded sample documents from the Question 3 dataset."""
    if not os.path.exists(Q3_DIR):
        return []

    samples = []
    sample_files = sorted(os.listdir(Q3_DIR))
    for fname in sample_files:
        p = os.path.join(Q3_DIR, fname)
        if os.path.isfile(p):
            ext = fname.rsplit(".", 1)[-1].lower()
            if ext in ["png", "jpg", "jpeg", "pdf"]:
                doc_id = fname.rsplit(".", 1)[0]
                _doc_file_paths[doc_id] = p
                samples.append({
                    "doc_id": doc_id,
                    "filename": fname,
                    "size_bytes": os.path.getsize(p),
                    "file_type": ext,
                })
    return samples


@router.get("/preview/{doc_id}")
def get_document_preview(doc_id: str):
    """Serves the preview image for a document."""
    # 1. Check in temporary preview directory (rendered from PDFs)
    if os.path.exists(TEMP_PREVIEW_DIR):
        for fname in os.listdir(TEMP_PREVIEW_DIR):
            if fname.startswith(doc_id) and fname.endswith(".png"):
                return FileResponse(os.path.join(TEMP_PREVIEW_DIR, fname), media_type="image/png")

    # 2. Check in uploaded files
    for fname in os.listdir(UPLOAD_DIR):
        if fname.startswith(doc_id):
            file_path = os.path.join(UPLOAD_DIR, fname)
            if file_path.lower().endswith(".pdf"):
                prev = os.path.join(TEMP_PREVIEW_DIR, f"{fname.rsplit('.', 1)[0]}_preview.png")
                if os.path.exists(prev):
                    return FileResponse(prev, media_type="image/png")
            return FileResponse(file_path)

    # 3. Check in sample files
    if os.path.exists(Q3_DIR):
        for fname in os.listdir(Q3_DIR):
            if fname.startswith(doc_id):
                file_path = os.path.join(Q3_DIR, fname)
                if file_path.lower().endswith(".pdf"):
                    prev = os.path.join(TEMP_PREVIEW_DIR, f"{fname.rsplit('.', 1)[0]}_preview.png")
                    if os.path.exists(prev):
                        return FileResponse(prev, media_type="image/png")
                return FileResponse(file_path)

    raise HTTPException(status_code=404, detail="Document image preview not found")



@router.post("/process-sample/{filename}", response_model=DocumentProcessResponse)
def process_sample_document(filename: str):
    """Runs extraction on a selected sample document from the Question 3 repository."""
    file_path = os.path.join(Q3_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Sample file {filename} not found")

    result = process_document(file_path, original_filename=filename)
    _processed_docs_cache[result.document_id] = result
    _doc_file_paths[result.document_id] = file_path
    return result


@router.post("/upload", response_model=DocumentProcessResponse)
async def upload_and_process_document(file: UploadFile = File(...)):
    """Uploads a document file and immediately runs classification and extraction."""
    safe_name = os.path.basename(file.filename or "uploaded_doc.png")
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = process_document(file_path, original_filename=safe_name)
    _processed_docs_cache[result.document_id] = result
    _doc_file_paths[result.document_id] = file_path
    return result


@router.post("/reclassify", response_model=DocumentProcessResponse)
def reclassify_document(payload: ReclassifyRequest):
    """
    Allows user to switch the document type dropdown.
    Re-evaluates fields for the chosen type instantly.
    """
    cached = _processed_docs_cache.get(payload.document_id)
    file_path = _doc_file_paths.get(payload.document_id)

    if not cached or not file_path:
        raise HTTPException(status_code=404, detail="Document not found in current session cache")

    updated = process_document(file_path, cached.filename, override_type=payload.chosen_type)
    _processed_docs_cache[payload.document_id] = updated
    return updated


@router.post("/triage/update", response_model=DocumentProcessResponse)
def update_triage_field(payload: TriageUpdateRequest):
    """Applies a human reviewer's correction or verification on a field."""
    cached = _processed_docs_cache.get(payload.document_id)
    if not cached:
        raise HTTPException(status_code=404, detail="Document not found")

    updated_fields: List[FieldResult] = []
    for f in cached.fields:
        if f.field_name == payload.field_name:
            updated_fields.append(
                FieldResult(
                    field_name=f.field_name,
                    value=payload.updated_value,
                    confidence=1.0,
                    is_handwritten=f.is_handwritten,
                    is_flagged=False,
                    flag_reason=None,
                    human_verified=True,
                )
            )
        else:
            updated_fields.append(f)

    flagged_count = sum(1 for f in updated_fields if f.is_flagged)
    overall_conf = round(sum(f.confidence for f in updated_fields) / len(updated_fields), 2)

    cached.fields = updated_fields
    cached.flagged_count = flagged_count
    cached.overall_confidence = overall_conf
    cached.needs_review = flagged_count > 0
    _processed_docs_cache[payload.document_id] = cached

    # Append to local audit log
    audit_entry = {
        "document_id": payload.document_id,
        "field_name": payload.field_name,
        "new_value": payload.updated_value,
        "verified": payload.confirm_as_verified,
    }
    try:
        entries = []
        if os.path.exists(AUDIT_FILE):
            with open(AUDIT_FILE, "r") as f:
                entries = json.load(f)
        entries.append(audit_entry)
        with open(AUDIT_FILE, "w") as f:
            json.dump(entries, f, indent=2)
    except Exception:
        pass

    return cached


@router.get("/report")
def get_technical_report():
    """Returns the technical deliverable report required for Question 3."""
    return {
        "confidence_threshold": 0.85,
        "threshold_rationale": (
            "In insurance onboarding and financial underwriting (KYC, ECS mandates, proposal verification), "
            "faulty extraction of critical identifiers (IFSC, PAN, Bank A/C) causes monetary failure. "
            "A confidence threshold of 0.85 provides optimal Straight-Through Processing (STP) for clean "
            "printed cards (Aadhaar, PAN, Passport, DL score >= 0.90) while reliably capturing ambiguous "
            "cursive strokes and character substitutions in handwritten fields (which score 0.60 - 0.84) "
            "into the human review queue."
        ),
        "handwritten_vs_printed_handling": (
            "Printed documents follow rigid fonts with high contrast, enabling exact regex validation and "
            "high confidence. Handwritten documents (NACH mandates, FATCA, Moral Hazard) exhibit variable stroke "
            "thickness, cursive ligatures, and character ambiguities (e.g., '1' vs 'I', '0' vs 'O', 'SBJNO' vs 'SBIN0'). "
            "Our pipeline detects handwritten sections, executes regex-guided OCR normalization, and applies a strict "
            "handwriting uncertainty penalty to ensure potential ambiguities are flagged for human triage."
        ),
        "observed_failure_cases": [
            "Handwritten IFSC on ECS form: 'SBIN0002712' frequently read as 'SBJNO27F12' or 'JFscSB1N0221'.",
            "Handwritten Place name in Benefit Illustration: 'West Bihar' occasionally read as 'Wes 4 Bihar'.",
            "Handwritten Date strokes: Confusion between single-digit months (03 vs 04) and century shortcuts (20 vs 26).",
            "Nominee Relationship in Moral Hazard: Cursive 'Nephew' vs 'Father' stroke variance.",
        ],
    }
