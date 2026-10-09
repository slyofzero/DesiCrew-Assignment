"""Composable prompt builder constructing document-specific prompts from shared rules and field catalogs."""

from typing import List

from agentic_idp_pipeline.prompts.base import (
    BASE_EXTRACTION_PREAMBLE,
    BASE_RESPONSE_FORMAT,
)
from agentic_idp_pipeline.prompts.fields import FIELD_EXTRACTION_INSTRUCTIONS


def build_document_prompt(
    doc_type: str,
    target_fields: List[str],
    proximity_map_text: str = "",
) -> str:
    """
    Constructs a complete document extraction prompt by composing:
    1. Universal base preamble & extraction rules (word splitting, prefix stripping, spatial rules).
    2. Document-specific target field definitions pulled from the modular fields catalog.
    3. The 2D spatial proximity layout graph.
    4. Strongly-typed JSON output schema matching target_fields.
    """
    preamble = BASE_EXTRACTION_PREAMBLE.format(doc_type=doc_type)

    field_lines = []
    for i, field_name in enumerate(target_fields, 1):
        instruction = FIELD_EXTRACTION_INSTRUCTIONS.get(
            field_name,
            f"Extract the exact value for '{field_name}' positioned adjacent to its label.",
        )
        field_lines.append(f"{i}. \"{field_name}\": {instruction}")

    fields_text = "\n".join(field_lines)

    # Dynamic JSON schema template for fields
    schema_entries = []
    for field_name in target_fields:
        schema_entries.append(
            f'    "{field_name}": {{\n'
            f'      "value": "<Extracted string value or null if not found>",\n'
            f'      "confidence": <float between 0.0 and 1.0>,\n'
            f'      "spatial_evidence": "<Spatial neighbor used e.g. \'Directly RIGHT of {field_name} label\'>"\n'
            f'    }}'
        )
    fields_schema_text = ",\n".join(schema_entries)

    response_format = BASE_RESPONSE_FORMAT.format(fields_json_schema=fields_schema_text)

    prompt = (
        f"{preamble}\n\n"
        f"Use 2D SPATIAL REASONING to extract the {len(target_fields)} required target fields:\n"
        f"{fields_text}\n\n"
        f"--- 2D SPATIAL PROXIMITY MAP ---\n"
        f"{proximity_map_text}\n"
        f"--- END PROXIMITY MAP ---\n\n"
        f"{response_format}"
    )

    return prompt
