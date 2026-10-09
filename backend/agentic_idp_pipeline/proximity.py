import math
from typing import Any, Dict, List, Optional, Tuple


def compute_centroid(box: List[List[float]]) -> Tuple[float, float]:
    """Computes the (cx, cy) centroid of a 4-point bounding box polygon."""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def get_relative_directions(
    cx_src: float, cy_src: float, cx_tgt: float, cy_tgt: float, tolerance: float = 15.0
) -> Tuple[str, str, float]:
    """
    Computes:
    - Relative direction on the X-axis: 'RIGHT', 'LEFT', or 'ALIGNED_X'
    - Relative direction on the Y-axis: 'BELOW', 'ABOVE', or 'ALIGNED_Y'
    - Absolute Euclidean distance between centroids
    """
    dx = cx_tgt - cx_src
    dy = cy_tgt - cy_src
    dist = math.sqrt(dx * dx + dy * dy)

    if dx > tolerance:
        dir_x = "RIGHT"
    elif dx < -tolerance:
        dir_x = "LEFT"
    else:
        dir_x = "ALIGNED_X"

    if dy > tolerance:
        dir_y = "BELOW"
    elif dy < -tolerance:
        dir_y = "ABOVE"
    else:
        dir_y = "ALIGNED_Y"

    return dir_x, dir_y, round(dist, 1)


def build_proximity_map(
    boxes: List[List[List[float]]],
    ocr_items: List[Tuple[str, float]],
    k_neighbors: int = 10,
) -> List[Dict[str, Any]]:
    """
    Constructs a spatial proximity map between each extracted text line.
    For each text item, computes its k nearest neighbors (default 10) with:
    - Relative direction on X-axis (LEFT / RIGHT / ALIGNED_X)
    - Relative direction on Y-axis (ABOVE / BELOW / ALIGNED_Y)
    - Absolute Euclidean distance
    """
    if not boxes or not ocr_items or len(boxes) != len(ocr_items):
        return []

    n = len(boxes)
    centroids = [compute_centroid(b) for b in boxes]
    proximity_map: List[Dict[str, Any]] = []

    for i in range(n):
        src_text, src_conf = ocr_items[i]
        src_cx, src_cy = centroids[i]

        neighbors = []
        for j in range(n):
            if i == j:
                continue
            tgt_text, tgt_conf = ocr_items[j]
            tgt_cx, tgt_cy = centroids[j]
            dir_x, dir_y, dist = get_relative_directions(src_cx, src_cy, tgt_cx, tgt_cy)

            neighbors.append({
                "target_index": j,
                "target_text": tgt_text,
                "target_confidence": round(tgt_conf, 2),
                "rel_direction_x": dir_x,
                "rel_direction_y": dir_y,
                "distance": dist,
            })

        # Sort neighbors by absolute distance ascending and keep top k
        neighbors.sort(key=lambda item: item["distance"])
        top_k = neighbors[:k_neighbors]

        proximity_map.append({
            "index": i,
            "text": src_text,
            "confidence": round(src_conf, 2),
            "centroid": (round(src_cx, 1), round(src_cy, 1)),
            "spatial_neighbors": top_k,
        })

    return proximity_map


def get_top_neighbors_for_token(
    boxes: List[List[List[float]]],
    ocr_items: List[Tuple[str, float]],
    anchor_index: int,
    k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Computes top k (e.g. 20) spatial neighbors for a specific anchor token index.
    Used for focused self-repair reflection when a field fails statutory validation.
    """
    if not boxes or not ocr_items or anchor_index < 0 or anchor_index >= len(boxes):
        return []

    centroids = [compute_centroid(b) for b in boxes]
    src_cx, src_cy = centroids[anchor_index]
    neighbors = []

    for j in range(len(boxes)):
        if j == anchor_index:
            continue
        tgt_text, tgt_conf = ocr_items[j]
        tgt_cx, tgt_cy = centroids[j]
        dir_x, dir_y, dist = get_relative_directions(src_cx, src_cy, tgt_cx, tgt_cy)

        neighbors.append({
            "target_index": j,
            "target_text": tgt_text,
            "target_confidence": round(tgt_conf, 2),
            "rel_direction_x": dir_x,
            "rel_direction_y": dir_y,
            "distance": dist,
        })

    neighbors.sort(key=lambda item: item["distance"])
    return neighbors[:k]


def format_top_neighbors_for_llm(anchor_text: str, neighbors: List[Dict[str, Any]]) -> str:
    """
    Formats the top-K neighbors surrounding a specific anchor label for self-reflection.
    """
    lines = [f"Anchor Token: \"{anchor_text}\""]
    for i, nb in enumerate(neighbors, 1):
        idx = nb["target_index"]
        txt = nb["target_text"]
        conf = nb["target_confidence"]
        dx = nb["rel_direction_x"]
        dy = nb["rel_direction_y"]
        d = nb["distance"]
        lines.append(
            f"  Neighbor #{i} [Token {idx}]: \"{txt}\" (conf: {conf}) -> Rel pos: [X: {dx}, Y: {dy}, distance: {d}px]"
        )
    return "\n".join(lines)


def format_proximity_map_for_llm(proximity_map: List[Dict[str, Any]]) -> str:
    """
    Formats the proximity map into a clean, human-readable layout graph for LLM prompts.
    """
    lines = []
    for item in proximity_map:
        text = item["text"]
        conf = item["confidence"]
        cx, cy = item["centroid"]
        neighbors_str = []
        for nb in item["spatial_neighbors"]:
            tgt = nb["target_text"]
            dx = nb["rel_direction_x"]
            dy = nb["rel_direction_y"]
            d = nb["distance"]
            neighbors_str.append(f"'{tgt}' [X: {dx}, Y: {dy}, dist: {d}px]")

        lines.append(
            f"Token [{item['index']}]: \"{text}\" (conf: {conf}, pos: ({cx},{cy}))\n"
            f"  -> Nearest Neighbors: " + "; ".join(neighbors_str)
        )

    return "\n\n".join(lines)

