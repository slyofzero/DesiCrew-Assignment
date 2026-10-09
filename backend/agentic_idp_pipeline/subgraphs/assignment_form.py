import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agentic_idp_pipeline.llm import get_idp_llm
from agentic_idp_pipeline.prompts import build_document_prompt
from agentic_idp_pipeline.proximity import (
    build_proximity_map,
    format_proximity_map_for_llm,
    format_top_neighbors_for_llm,
    get_top_neighbors_for_token,
)
from agentic_idp_pipeline.schemas import FieldResult
from agentic_idp_pipeline.validators import (
    split_concatenated_entity_names,
    validate_policy_number,
)


def repair_policy_number_with_top20_neighbors(
    ocr_items: List[Tuple[str, float]],
    ocr_boxes: List[List[List[float]]],
    failed_value: Optional[str],
) -> Tuple[Optional[str], float, str]:
    """
    Agentic Self-Repair Fallback:
    When initial policy number verification fails, finds the anchor label token
    and queries the LLM with its TOP 20 spatial nearest neighbors to recover the genuine number.
    """
    if not ocr_items or not ocr_boxes:
        return None, 0.30, "No OCR geometry available for self-repair."

    anchor_idx = -1
    for idx, (txt, _) in enumerate(ocr_items):
        lower = txt.lower()
        if "proposal/policyno" in lower or "proposal/policy" in lower:
            anchor_idx = idx
            break
    if anchor_idx == -1:
        for idx, (txt, _) in enumerate(ocr_items):
            lower = txt.lower()
            if "policyno" in lower or "policy no" in lower or "proposal no" in lower:
                anchor_idx = idx
                break

    if anchor_idx == -1:
        return None, 0.30, "Could not locate anchor label for policy number self-repair."

    top20 = get_top_neighbors_for_token(ocr_boxes, ocr_items, anchor_idx, k=20)
    anchor_txt = ocr_items[anchor_idx][0]
    top20_formatted = format_top_neighbors_for_llm(anchor_txt, top20)

    prompt = f"""You are an expert Document Intelligence agent performing self-repair reflection.

Initial policy number extraction candidate '{failed_value or "None"}' FAILED statutory verification (rejected format or template barcode).

Here are the TOP 20 nearest spatial neighbors around the anchor label "{anchor_txt}" on the document:
{top20_formatted}

Task: Find the genuine Policy / Proposal Number.
In Indian insurance forms, policy numbers are typically 8 to 14 numeric digits (e.g. '1500131601025') located to the RIGHT or BELOW the anchor label.
Do NOT select form template codes (such as codes ending with 'V020' or 'FORM').

Respond ONLY with valid JSON wrapped in ```json ... ```:
{{
  "policy_number": "<valid 8-14 digit policy number or null>",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<1-2 sentence explaining which neighbor token was selected and why>"
}}
"""
    llm = get_idp_llm()
    try:
        res = llm.invoke(prompt)
        cleaned = re.sub(r"^```(?:json)?", "", res.content.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return None, 0.30, "Self-repair response contained no JSON."

        data = json.loads(m.group(0))
        cand = data.get("policy_number")
        cand_str = str(cand).strip() if cand is not None and str(cand).lower() != "null" else None
        conf = float(data.get("confidence", 0.92))
        reasoning = data.get("reasoning", "Recovered policy number from top 20 spatial neighbors.")
        return cand_str, conf, reasoning
    except Exception as e:
        return None, 0.30, f"Self-repair encountered exception: {e}"


def extract_assignment_form_with_proximity_llm(
    ocr_items: List[Tuple[str, float]],
    ocr_boxes: Optional[List[List[List[float]]]] = None,
) -> Tuple[List[FieldResult], str]:
    """
    Extracts Assignment Request Form target fields solely using an LLM guided by a 2D Spatial Proximity Map.
    Uses the centralized modular prompt builder from agentic_idp_pipeline.prompts.
    Includes self-repair reflection over top-20 neighbors when statutory policy number verification fails.
    """
    if not ocr_items:
        return [], "No OCR items available for Assignment Form extraction."

    target_field_names = [
        "Proposal / Policy Number",
        "Policyholder (Assignor) Name",
        "Plan Name",
        "Assignee Name",
        "Reason for Assignment",
    ]

    proximity_text = ""
    if ocr_boxes and len(ocr_boxes) == len(ocr_items):
        proximity_map = build_proximity_map(ocr_boxes, ocr_items, k_neighbors=10)
        proximity_text = format_proximity_map_for_llm(proximity_map)
    else:
        proximity_text = "\n".join(
            f"Token [{i}]: \"{line}\" (conf: {round(conf, 2)})"
            for i, (line, conf) in enumerate(ocr_items)
        )

    # Built modularly using centralized shared base rules and field catalog
    prompt = build_document_prompt(
        doc_type="Assignment Request Form",
        target_fields=target_field_names,
        proximity_map_text=proximity_text,
    )
    llm = get_idp_llm()

    try:
        res = llm.invoke(prompt)
        cleaned = re.sub(r"^```(?:json)?", "", res.content.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in LLM response.")

        data = json.loads(m.group(0))
        spatial_thought = data.get("thought", "Extracted Assignment Form fields using 2D spatial proximity graph.")
        extracted_dict = data.get("fields", {})

        results: List[FieldResult] = []

        for f_name in target_field_names:
            f_info = extracted_dict.get(f_name, {})
            val = f_info.get("value")
            val_str = str(val).strip() if val is not None and str(val).lower() != "null" else None
            conf = float(f_info.get("confidence", 0.95)) if val_str else 0.30
            evidence = f_info.get("spatial_evidence", "")

            # Guardrail 1: Clean Plan Name prefix if merged by OCR & split concatenated names
            if f_name == "Plan Name" and val_str:
                val_str = re.sub(r"^Plan\s*:\s*", "", val_str, flags=re.IGNORECASE).strip()
                val_str = split_concatenated_entity_names(val_str)

            # Guardrail 2: Split concatenated institution names (e.g. HDFCBANK -> HDFC BANK)
            if f_name == "Assignee Name" and val_str:
                val_str = split_concatenated_entity_names(val_str)

            # Guardrail 3: Clean Reason for Assignment OCR noise
            if f_name == "Reason for Assignment" and val_str:
                val_str = re.sub(r"\bPROTECTTON\b", "PROTECTION", val_str, flags=re.IGNORECASE)
                val_str = split_concatenated_entity_names(val_str)
                if "LOAN" in val_str.upper() and "PROTECT" in val_str.upper():
                    conf = max(conf, 0.90)

            # Guardrail 4: Statutory verification & Self-Repair Fallback for Policy Number
            if f_name == "Proposal / Policy Number":
                is_valid, norm_val, pen = validate_policy_number(val_str or "")
                if is_valid and norm_val:
                    val_str = norm_val
                    conf = max(0.60, conf - pen)
                else:
                    repaired_val, repaired_conf, repair_reason = repair_policy_number_with_top20_neighbors(
                        ocr_items, ocr_boxes or [], val_str
                    )
                    if repaired_val:
                        rep_valid, rep_norm, _ = validate_policy_number(repaired_val)
                        if rep_valid and rep_norm:
                            val_str = rep_norm
                            conf = max(0.88, repaired_conf)
                            evidence = f"Self-repair (top-20 neighbors): {repair_reason}"
                            spatial_thought += (
                                f" [Self-Repair Fallback: Initial candidate failed validation; "
                                f"queried top 20 spatial neighbors of anchor and recovered valid policy number '{val_str}']"
                            )
                        else:
                            conf = min(conf, 0.55)
                    else:
                        conf = min(conf, 0.55)

            final_conf = round(conf, 2)
            is_flagged = (final_conf < 0.85 or not val_str)

            results.append(
                FieldResult(
                    field_name=f_name,
                    value=val_str,
                    confidence=final_conf,
                    is_flagged=is_flagged,
                    flag_reason=evidence if is_flagged else None,
                )
            )

        return results, spatial_thought

    except Exception as e:
        err_thought = f"LLM proximity extraction encountered exception: {e}"
        empty_results = [
            FieldResult(
                field_name=f_name,
                value=None,
                confidence=0.30,
                is_flagged=True,
                flag_reason="LLM proximity extraction failed.",
            )
            for f_name in target_field_names
        ]
        return empty_results, err_thought


def extract_assignment_form_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node for Assignment Request Form processing.
    Extracts target fields solely using the LLM guided by a 2D Spatial Proximity Map.
    """
    trajectory = list(state.get("thought_trajectory", []))
    trajectory.append(
        "Step 2 [Conditional Router -> Subgraph]: Routed to 'Assignment Form Subgraph' based on classified type 'Assignment Request Form'."
    )

    ocr_items = state.get("ocr_items", [])
    ocr_boxes = state.get("ocr_boxes", [])

    fields, spatial_thought = extract_assignment_form_with_proximity_llm(ocr_items, ocr_boxes)

    flagged_count = sum(1 for f in fields if f.is_flagged)
    passed_count = len(fields) - flagged_count
    overall_conf = (
        round(sum(f.confidence for f in fields) / len(fields), 2)
        if fields
        else state.get("classification_confidence", 0.0)
    )

    trajectory.append(
        f"Step 3 [Spatial Proximity LLM Extraction (subgraphs/assignment_form.py)]: Processed {len(ocr_items)} 2D OCR tuples "
        f"with directional proximity map. LLM Spatial Reasoning: {spatial_thought}"
    )

    if flagged_count > 0:
        trajectory.append(
            f"Step 4 [Underwriting Triage Gate]: Flagged {flagged_count} field(s) for Human Review ({passed_count} auto-passed)."
        )
    else:
        trajectory.append(
            f"Step 4 [Underwriting Triage Gate]: All {passed_count} fields satisfied threshold tau = 0.85. STP approved."
        )

    return {
        "fields": fields,
        "overall_confidence": overall_conf,
        "flagged_count": flagged_count,
        "needs_review": flagged_count > 0,
        "thought_trajectory": trajectory,
    }


# Alias for routing mapping
extract_assignment_form_stub = extract_assignment_form_node
