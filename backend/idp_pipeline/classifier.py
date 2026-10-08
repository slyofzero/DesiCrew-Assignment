import json
import re
from typing import List, Tuple

from idp_pipeline.llm import get_idp_llm
from idp_pipeline.prompts import IDP_CLASSIFICATION_PROMPT
from idp_pipeline.schemas import DocumentType

# Key discriminative anchor keywords for each document type
KEYWORD_ANCHORS = {
    DocumentType.AADHAAR: [
        "aadhaar",
        "government of india",
        "unique identification",
        "uidai",
        "mera aadhaar",
        "hata aea",
    ],
    DocumentType.PAN: [
        "income tax department",
        "permanent account number",
        "govt. of india",
        "pan card",
    ],
    DocumentType.DRIVING_LICENCE: [
        "driving licence",
        "driving license",
        "indian union driving",
        "form 7",
        "rule 16",
        "licence to drive",
    ],
    DocumentType.PASSPORT: [
        "republic of india passport",
        "passport",
        "republic of india",
        "p<ind",
        "type / p",
    ],
    DocumentType.NACH_MANDATE: [
        "nach mandate instruction",
        "nach",
        "mandate instruction",
        "umrn",
        "sponsor bank code",
        "utility code",
        "to debit(tick)",
        "bank a/c number",
    ],
    DocumentType.FATCA: [
        "annexure form",
        "section 285ba",
        "fatca",
        "tax residency",
        "foreign account tax",
        "tin / pan",
    ],
    DocumentType.BENEFIT_ILLUSTRATION: [
        "customer declaration - benefit illustration",
        "benefit illustration",
        "benefitillustration",
        "customerdeclaration",
        "year-wise premiums",
        "illustrative returns",
    ],
    DocumentType.MORAL_HAZARD: [
        "moral hazard questionnaire",
        "moral hazard",
        "name of life to be assured",
        "reasons for choosing the nominee",
    ],
    DocumentType.MULTIPLE_POLICIES: [
        "split & multiple policies",
        "multiple policies - customer consent",
        "multiple policies",
        "split & multiple",
        "multiple life insurance policies",
    ],
    DocumentType.SUITABILITY_PROFILER: [
        "customer declaration - suitability profiler",
        "suitability profiler",
        "suitability",
        "inputs provided by me in the suitability",
    ],
    DocumentType.ASSIGNMENT_FORM: [
        "assignment request form",
        "assignment request",
        "details of assignor",
        "details of assignee",
        "name of the policyholder (assignor)",
    ],
    DocumentType.PROPOSAL_FORM: [
        "customer declaration - application/proposal form",
        "application/proposal form",
        "name of insurance plan",
        "type of plan",
        "premium payable",
        "sum assured",
    ],
}

# Mapping string names to DocumentType enum
TYPE_STR_MAP = {
    "aadhaar card": DocumentType.AADHAAR,
    "aadhaar": DocumentType.AADHAAR,
    "pan card": DocumentType.PAN,
    "pan": DocumentType.PAN,
    "driving licence": DocumentType.DRIVING_LICENCE,
    "driving license": DocumentType.DRIVING_LICENCE,
    "passport": DocumentType.PASSPORT,
    "nach / ecs mandate": DocumentType.NACH_MANDATE,
    "nach mandate": DocumentType.NACH_MANDATE,
    "ecs mandate": DocumentType.NACH_MANDATE,
    "fatca declaration": DocumentType.FATCA,
    "fatca": DocumentType.FATCA,
    "benefit illustration": DocumentType.BENEFIT_ILLUSTRATION,
    "moral hazard report": DocumentType.MORAL_HAZARD,
    "moral hazard": DocumentType.MORAL_HAZARD,
    "multiple policies declaration": DocumentType.MULTIPLE_POLICIES,
    "multiple policies": DocumentType.MULTIPLE_POLICIES,
    "suitability profiler": DocumentType.SUITABILITY_PROFILER,
    "suitability": DocumentType.SUITABILITY_PROFILER,
    "assignment request form": DocumentType.ASSIGNMENT_FORM,
    "assignment form": DocumentType.ASSIGNMENT_FORM,
    "assignment request": DocumentType.ASSIGNMENT_FORM,
    "application / proposal form": DocumentType.PROPOSAL_FORM,
    "application / proposal": DocumentType.PROPOSAL_FORM,
    "proposal form": DocumentType.PROPOSAL_FORM,
    "not classified": DocumentType.UNKNOWN,
    "unknown document": DocumentType.UNKNOWN,
    "unknown": DocumentType.UNKNOWN,
}


def classify_document_keywords(ocr_lines: List[str]) -> Tuple[DocumentType, float, str]:
    """
    Deterministic fallback: Classifies a document based on anchor keywords.
    Returns: (predicted_type, confidence, reasoning)
    """
    if not ocr_lines:
        return DocumentType.UNKNOWN, 0.0, "No OCR text lines available."

    full_text = " ".join(ocr_lines).lower()
    norm_text = re.sub(r"[^a-z0-9]", "", full_text)
    best_match = DocumentType.UNKNOWN
    max_score = 0.0
    matched_anchors = []

    for doc_type, anchors in KEYWORD_ANCHORS.items():
        score = 0.0
        doc_matched = []
        for anchor in anchors:
            anchor_lower = anchor.lower()
            anchor_norm = re.sub(r"[^a-z0-9]", "", anchor_lower)
            if anchor_lower in full_text or (len(anchor_norm) >= 6 and anchor_norm in norm_text):
                weight = 1.0 if anchors.index(anchor) < 2 else 0.5
                score += weight
                doc_matched.append(anchor)

        if score > max_score:
            max_score = score
            best_match = doc_type
            matched_anchors = doc_matched

    if max_score >= 1.0:
        confidence = min(0.99, 0.85 + (max_score * 0.05))
        reasoning = f"Matched anchor keywords: {', '.join(matched_anchors[:3])}."
        return best_match, round(confidence, 2), reasoning
    elif max_score > 0:
        reasoning = f"Weak anchor match: {', '.join(matched_anchors)}."
        return best_match, 0.70, reasoning

    return DocumentType.UNKNOWN, 0.30, "No recognized anchor keywords found."


def classify_document_with_llm(ocr_lines: List[str]) -> Tuple[DocumentType, float, str]:
    """
    Primary: Classifies document using an LLM semantic prompt.
    Returns: (doc_type, confidence, reasoning_thought)
    """
    if not ocr_lines:
        return DocumentType.UNKNOWN, 0.0, "Empty document text."

    sample_text = "\n".join(ocr_lines[:35])
    prompt = IDP_CLASSIFICATION_PROMPT.format(ocr_text=sample_text)

    try:
        llm = get_idp_llm()
        res = llm.invoke(prompt)
        cleaned = re.sub(r"^```(?:json)?", "", res.content.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            raw_type = str(data.get("document_type", "")).strip().lower()
            confidence = float(data.get("confidence", 0.95))
            reasoning = str(data.get("reasoning", "Classified via semantic LLM context analysis."))

            doc_type = TYPE_STR_MAP.get(raw_type)
            if doc_type:
                return doc_type, round(confidence, 2), reasoning
    except Exception as e:
        print(f"[IDP Classification Warning] LLM classification fallback to keyword rules ({e}).")

    return classify_document_keywords(ocr_lines)


def classify_document(ocr_lines: List[str]) -> Tuple[DocumentType, float]:
    """
    Standard classifier interface used by legacy callers and automated tests.
    """
    doc_type, conf, _ = classify_document_keywords(ocr_lines)
    return doc_type, conf
