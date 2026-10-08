"""Central prompt templates for the Intelligent Document Processing (IDP) Pipeline."""

IDP_CLASSIFICATION_PROMPT = """You are an expert Indian KYC and insurance document classifier.
Given the raw OCR lines extracted from a document, classify it into EXACTLY ONE of the following 12 document types:
- Aadhaar Card
- PAN Card
- Driving Licence
- Passport
- NACH / ECS Mandate
- FATCA Declaration
- Benefit Illustration
- Moral Hazard Report
- Multiple Policies Declaration
- Suitability Profiler
- Assignment Request Form
- Application / Proposal Form
- Unknown

Document Classification Guidelines:
- "Aadhaar Card": Contains UIDAI, Government of India, Aadhaar number (12 digits), Mera Aadhaar Meri Pehchan.
- "PAN Card": Contains Income Tax Department, Permanent Account Number, Father's Name, PAN format.
- "Driving Licence": Contains Union of India, Driving Licence, Form 7, Rule 16, Transport Department.
- "Passport": Contains Republic of India, Passport, Passport No, Nationality, Place of Birth.
- "NACH / ECS Mandate": Contains NACH Mandate Instruction, ECS, UMRN, Sponsor Bank, IFSC, Bank a/c number.
- "FATCA Declaration": Contains Annexure Form, Section 285BA, FATCA, Tax Residency, TIN.
- "Benefit Illustration": Heading is specifically 'Customer Declaration - Benefit Illustration' or contains 'Benefit Illustration'.
- "Moral Hazard Report": Heading is specifically 'Moral Hazard Questionnaire', reasons for choosing nominee.
- "Multiple Policies Declaration": Heading is specifically 'Split & Multiple Policies - Customer Consent form'.
- "Suitability Profiler": Heading is specifically 'Customer Declaration - Suitability Profiler'.
- "Assignment Request Form": Heading is specifically 'Assignment Request Form', Details of Assignor, Details of Assignee.
- "Application / Proposal Form": Heading is specifically 'Customer Declaration - Application/Proposal Form', and has 'Type of Plan', 'Name of Insurance Plan', 'Sum Assured'.

DISAMBIGUATION RULE: Many different insurance forms include a field label called "Application/Proposal Form Number". Do NOT classify as "Application / Proposal Form" just because that field label appears! Always classify by the PRIMARY HEADER TITLE of the document (e.g. Benefit Illustration, Suitability Profiler, FATCA Annexure Form, Moral Hazard).

Respond ONLY with valid JSON wrapped in ```json ... ```:
{{
  "document_type": "<Exact name from above list>",
  "confidence": <float between 0.50 and 1.00>,
  "reasoning": "<Concise 1-2 sentence explanation of why this document type was chosen based on anchors found>"
}}

--- RAW OCR LINES ---
{ocr_text}
--- END RAW OCR LINES ---
"""

IDP_FIELD_VERIFICATION_PROMPT = """You are an expert Document Information Verification and Typo Repair Engine.
A deterministic heuristic rule engine has scanned the document and produced initial candidate values for target fields.
Your job is to cross-verify these candidates against the raw OCR lines, repair obvious optical typos, and provide an audit thought.

Document Type: {doc_type}

--- HEURISTIC EXTRACTED CANDIDATES ---
{candidates_json}
--- END CANDIDATES ---

--- RAW OCR TEXT LINES ---
{ocr_text}
--- END RAW OCR TEXT LINES ---

Instructions:
1. Cross-check each extracted value against the raw OCR lines.
2. If a value has an obvious optical typo or spacing error (e.g., 'S/OKumar' -> 'S/O Kumar', 'StodoBankofIndia' -> 'State Bank of India', or letterhead merged into an address), fix it.
3. If an extracted value is already correct, keep it as is.
4. If a field was marked as not detected but is clearly present in the OCR text, provide the correct value.
5. Provide a step-by-step 'thought' explaining the audit findings and any optical typo corrections made.

Respond ONLY with valid JSON wrapped in ```json ... ```:
{{
  "thought": "<Step-by-step audit reflection explaining the verification and any typo fixes>",
  "verified_fields": {{
    "<FieldName>": {{
      "value": "<string or null>",
      "confidence": <float between 0.0 and 1.0>,
      "correction_applied": <true or false>,
      "reasoning": "<brief explanation of value validity or fix>"
    }}
  }}
}}
"""

TECHNICAL_REPORT_DATA = {
    "confidence_threshold": 0.85,
    "max_cot_trajectory": 5,
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
        "Our hybrid architecture combines LLM semantic document classification, deterministic spatial-regex "
        "extraction, and LLM verification to resolve character ambiguities and assign justified confidence scores."
    ),
    "observed_failure_cases": [
        "Handwritten IFSC on ECS form: 'SBIN0002712' frequently read as 'SBJNO27F12' or 'JFscSB1N0221'.",
        "Handwritten Place name in Benefit Illustration: 'West Bihar' occasionally read as 'Wes 4 Bihar'.",
        "Handwritten Date strokes: Confusion between single-digit months (03 vs 04) and century shortcuts (20 vs 26).",
        "Nominee Relationship in Moral Hazard: Cursive 'Nephew' vs 'Father' stroke variance.",
    ],
}
