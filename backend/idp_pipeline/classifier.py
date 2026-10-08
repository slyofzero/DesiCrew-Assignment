from typing import List, Tuple
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
}


def classify_document(ocr_lines: List[str]) -> Tuple[DocumentType, float]:
    """
    Classifies a document based on extracted OCR text lines using anchored frequency & weighted matching.
    Returns: (predicted_type, confidence)
    """
    if not ocr_lines:
        return DocumentType.UNKNOWN, 0.0

    full_text = " ".join(ocr_lines).lower()

    best_match = DocumentType.UNKNOWN
    max_score = 0.0

    for doc_type, anchors in KEYWORD_ANCHORS.items():
        score = 0.0
        for anchor in anchors:
            if anchor in full_text:
                # Stronger weights for primary titles (first 2 anchors)
                weight = 1.0 if anchors.index(anchor) < 2 else 0.5
                score += weight

        if score > max_score:
            max_score = score
            best_match = doc_type

    if max_score >= 1.0:
        # Confidence calculation scaled by match strength
        confidence = min(0.99, 0.85 + (max_score * 0.05))
        return best_match, round(confidence, 2)
    elif max_score > 0:
        return best_match, 0.70

    return DocumentType.UNKNOWN, 0.30
