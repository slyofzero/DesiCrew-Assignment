import os
import time
from typing import List, Optional, Tuple

import tempfile
import fitz  # pymupdf
from rapidocr_onnxruntime import RapidOCR

from agentic_idp_pipeline.graph import create_agentic_idp_graph
from agentic_idp_pipeline.schemas import (
    DocumentProcessResponse,
    DocumentType,
)

TEMP_PREVIEW_DIR = os.path.join(tempfile.gettempdir(), "idp_previews")
os.makedirs(TEMP_PREVIEW_DIR, exist_ok=True)

# Global singleton OCR engine to avoid reloading ONNX models on each request
_ocr_engine = None


def get_ocr_engine() -> RapidOCR:
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = RapidOCR()
    return _ocr_engine


def extract_text_with_boxes(
    file_path: str,
) -> Tuple[List[Tuple[str, float]], str, List[List[List[float]]]]:
    """
    Runs RapidOCR on an image or PDF file.
    Returns: (ocr_items, preview_path, bounding_boxes)
    """
    engine = get_ocr_engine()

    # If PDF, render first page to PNG in a temp directory (never polluting Question 3 or source dirs)
    if file_path.lower().endswith(".pdf"):
        doc = fitz.open(file_path)
        page = doc[0]
        pix = page.get_pixmap(dpi=150)
        base_name = os.path.basename(file_path).rsplit(".", 1)[0]
        img_preview_path = os.path.join(TEMP_PREVIEW_DIR, f"{base_name}_preview.png")
        pix.save(img_preview_path)
        target_path = img_preview_path
    else:
        target_path = file_path


    result, _ = engine(target_path)
    if not result:
        return [], target_path, []

    ocr_items: List[Tuple[str, float]] = [
        (str(item[1]), float(item[2])) for item in result
    ]
    boxes: List[List[List[float]]] = [item[0] for item in result]

    return ocr_items, target_path, boxes


def extract_text_from_file(file_path: str) -> Tuple[List[Tuple[str, float]], str]:
    """
    Backwards-compatible wrapper returning (ocr_items, preview_path).
    """
    ocr_items, preview_img, _ = extract_text_with_boxes(file_path)
    return ocr_items, preview_img


def process_document(
    file_path: str,
    original_filename: str,
    override_type: Optional[DocumentType] = None,
) -> DocumentProcessResponse:
    """
    Main Agentic IDP pipeline entry point executed upon any UI or API invocation.
    Invokes the LangGraph StateGraph workflow with a Reasoning Classification Node.
    """
    start_time = time.time()
    ocr_items, preview_img, ocr_boxes = extract_text_with_boxes(file_path)

    initial_state = {
        "file_path": file_path,
        "filename": original_filename,
        "preview_img": preview_img,
        "override_type": override_type,
        "ocr_items": ocr_items,
        "ocr_boxes": ocr_boxes,
        "thought_trajectory": [],
    }

    # Execute the LangGraph StateGraph
    agentic_idp_graph = create_agentic_idp_graph()
    final_state = agentic_idp_graph.invoke(initial_state)

    elapsed = round(time.time() - start_time, 2)
    doc_id = os.path.basename(file_path).rsplit(".", 1)[0]
    raw_lines = [item[0] for item in ocr_items[:15]]

    return DocumentProcessResponse(
        document_id=doc_id,
        filename=original_filename,
        preview_url=f"/api/idp/preview/{doc_id}",
        document_type=final_state.get("document_type", DocumentType.UNKNOWN),
        classification_confidence=final_state.get("classification_confidence", 0.0),
        fields=final_state.get("fields", []),
        overall_confidence=final_state.get("overall_confidence", 0.0),
        needs_review=final_state.get("needs_review", False),
        flagged_count=final_state.get("flagged_count", 0),
        processing_time_sec=elapsed,
        raw_ocr_lines=raw_lines,
        thought_trajectory=final_state.get("thought_trajectory", []),
    )
