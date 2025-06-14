from typing import Dict, List, Any
from fuzzywuzzy import fuzz
from fuzzywuzzy import process



def handle_validation_errors(validation_result: dict, metadata_fields: dict) -> dict:
    """Handle validation errors with detailed feedback."""
    error_details = {
        "invalid_fields": validation_result["invalid_fields"],
        "invalid_values": validation_result["invalid_values"],
        "suggestions": generate_field_suggestions(validation_result, metadata_fields),
        "closest_matches": find_closest_valid_values(validation_result, metadata_fields)
    }
    return error_details

def generate_field_suggestions(validation_result: dict, metadata_fields: dict) -> List[str]:
    """Generate suggestions for invalid fields."""
    suggestions = []
    for field in validation_result["invalid_fields"]:
        # Find similar field names
        similar_fields = find_similar_fields(field, metadata_fields)
        if similar_fields:
            suggestions.append(f"Did you mean: {', '.join(similar_fields)}")
    return suggestions

def find_similar_fields(field: str, metadata_fields: dict) -> List[str]:
    """Find similar field names using fuzzy matching."""
    field_names = [f["fieldName"] for f in metadata_fields.get("singleValueFields", [])]
    field_names.extend([f["fieldName"] for f in metadata_fields.get("multiValueFields", [])])
    
    # Get top 3 matches with score > 70
    matches = process.extract(field, field_names, scorer=fuzz.ratio, limit=3)
    return [match[0] for match in matches if match[1] > 70]

def find_closest_valid_values(validation_result: dict, metadata_fields: dict) -> Dict[str, List[str]]:
    """Find closest valid values for invalid values."""
    closest_matches = {}
    for field, value in validation_result["invalid_values"].items():
        # Find the field in metadata
        field_metadata = find_field_metadata(field, metadata_fields)
        if field_metadata and "selectValues" in field_metadata:
            valid_values = [v["value"] for v in field_metadata["selectValues"]]
            # Get top 3 matches with score > 70
            matches = process.extract(value, valid_values, scorer=fuzz.ratio, limit=3)
            closest_matches[field] = [match[0] for match in matches if match[1] > 70]
    return closest_matches

def find_field_metadata(field: str, metadata_fields: dict) -> dict:
    """Find metadata for a specific field."""
    for f in metadata_fields.get("singleValueFields", []):
        if f["fieldName"] == field or f["identifier"] == field:
            return f
    for f in metadata_fields.get("multiValueFields", []):
        if f["fieldName"] == field or f["identifier"] == field:
            return f
    return None

def get_cached_metadata(headers: dict) -> dict:
    """Get metadata with caching."""
    from tools import get_cached_metadata as tools_get_cached_metadata
    return tools_get_cached_metadata(headers) 