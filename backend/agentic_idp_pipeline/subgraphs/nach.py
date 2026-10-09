import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agentic_idp_pipeline.llm import get_idp_llm
from agentic_idp_pipeline.prompts import build_document_prompt
from agentic_idp_pipeline.proximity import build_proximity_map, format_proximity_map_for_llm
from agentic_idp_pipeline.schemas import FieldResult
from agentic_idp_pipeline.validators import (
    split_concatenated_entity_names,
    validate_amount,
    validate_ifsc,
)


def extract_nach_with_proximity_llm(
    ocr_items: List[Tuple[str, float]],
    ocr_boxes: Optional[List[List[List[float]]]] = None,
) -> Tuple[List[FieldResult], str]:
    """
    Extracts NACH Mandate target fields solely using an LLM guided by a 2D Spatial Proximity Map.
    Uses the centralized modular prompt builder.
    """
    if not ocr_items:
        return [], "No OCR items available for NACH Mandate extraction."

    target_field_names = [
        "Bank Account Number",
        "IFSC Code",
        "Bank Name",
        "Amount (figures)",
        "Frequency",
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

    prompt = build_document_prompt("NACH / ECS Mandate", target_field_names, proximity_text)
    llm = get_idp_llm()

    try:
        res = llm.invoke(prompt)
        cleaned = re.sub(r"^```(?:json)?", "", res.content.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            raise ValueError("No JSON object found in LLM response.")

        data = json.loads(m.group(0))
        spatial_thought = data.get("thought", "Extracted NACH Mandate fields using 2D spatial proximity graph.")
        extracted_dict = data.get("fields", {})

        results: List[FieldResult] = []

        for f_name in target_field_names:
            f_info = extracted_dict.get(f_name, {})
            val = f_info.get("value")
            val_str = str(val).strip() if val is not None and str(val).lower() != "null" else None
            conf = float(f_info.get("confidence", 0.95)) if val_str else 0.30

            # Statutory validation guardrails
            if f_name == "IFSC Code" and val_str:
                is_valid, formatted, pen = validate_ifsc(val_str)
                if is_valid and formatted:
                    val_str = formatted
                    conf = max(0.60, conf - pen)
                else:
                    conf = min(conf, 0.65)
            elif f_name == "Amount (figures)" and val_str:
                is_valid, norm_amt, pen = validate_amount(val_str)
                if is_valid and norm_amt:
                    val_str = norm_amt
                    conf = max(0.60, conf - pen)

            if val_str:
                val_str = split_concatenated_entity_names(val_str)

            final_conf = round(conf, 2)
            is_flagged = (final_conf < 0.85 or not val_str)
            evidence = f_info.get("spatial_evidence", "")

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


def extract_nach_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node for NACH / ECS Mandate processing.
    Extracts target fields solely using the LLM guided by a 2D Spatial Proximity Map.
    """
    trajectory = list(state.get("thought_trajectory", []))
    trajectory.append(
        "Step 2 [Conditional Router -> Subgraph]: Routed to 'NACH / ECS Mandate Subgraph' based on classified type 'NACH / ECS Mandate'."
    )

    ocr_items = state.get("ocr_items", [])
    ocr_boxes = state.get("ocr_boxes", [])

    fields, spatial_thought = extract_nach_with_proximity_llm(ocr_items, ocr_boxes)

    flagged_count = sum(1 for f in fields if f.is_flagged)
    passed_count = len(fields) - flagged_count
    overall_conf = (
        round(sum(f.confidence for f in fields) / len(fields), 2)
        if fields
        else state.get("classification_confidence", 0.0)
    )

    trajectory.append(
        f"Step 3 [Spatial Proximity LLM Extraction (subgraphs/nach.py)]: Processed {len(ocr_items)} 2D OCR tuples "
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
extract_nach_stub = extract_nach_node
