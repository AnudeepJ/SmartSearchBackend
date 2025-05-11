import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import json
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def preprocess_query(query: str) -> str:
    """Preprocess the query to handle common patterns and edge cases."""
    # Remove extra whitespace
    query = " ".join(query.split())
    
    # Handle common date patterns
    query = normalize_date_references(query)
    
    # Handle common field name variations
    query = normalize_field_references(query)
    
    return query

def normalize_date_references(query: str) -> str:
    """Normalize date references in the query."""
    # Handle "last week", "last month", etc.
    date_patterns = {
        r'last week': '7 days ago',
        r'last month': '30 days ago',
        r'last year': '365 days ago',
        r'today': '0 days ago',
        r'yesterday': '1 day ago'
    }
    
    for pattern, replacement in date_patterns.items():
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    
    return query

def normalize_field_references(query: str) -> str:
    """Normalize field name references in the query."""
    # Common field name variations
    field_variations = {
        'doc type': 'Type',
        'doc status': 'Status',
        'document type': 'Type',
        'document status': 'Status',
        'created date': 'Date Created',
        'modified date': 'Date Modified',
        'review date': 'Date Reviewed'
    }
    
    for variation, standard in field_variations.items():
        query = re.sub(variation, standard, query, flags=re.IGNORECASE)
    
    return query

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

def enhance_search_results(results: dict) -> dict:
    """Enhance search results with additional context and metadata."""
    return {
        "results": results,
        "metadata": {
            "total_count": len(results.get("documents", [])),
            "search_fields_used": extract_used_fields(results),
            "confidence_scores": calculate_confidence_scores(results)
        }
    }

def extract_used_fields(results: dict) -> List[str]:
    """Extract fields used in the search."""
    used_fields = set()
    for doc in results.get("documents", []):
        for field in doc.keys():
            used_fields.add(field)
    return list(used_fields)

def calculate_confidence_scores(results: dict) -> Dict[str, float]:
    """Calculate confidence scores for search results."""
    confidence_scores = {}
    for doc in results.get("documents", []):
        # Calculate confidence based on field match quality
        score = 0.0
        total_fields = len(doc)
        if total_fields > 0:
            # Simple scoring based on field presence
            score = sum(1 for v in doc.values() if v is not None) / total_fields
        confidence_scores[doc.get("id", "unknown")] = score
    return confidence_scores

def get_cached_metadata(headers: dict) -> dict:
    """Get metadata with caching."""
    # TODO: Implement caching logic
    from tools import call_metadata
    return call_metadata(headers)

def call_search_api_with_retry(search_params: dict, headers: dict, max_retries: int = 3) -> dict:
    """Call search API with retry logic."""
    from tools import call_search_api
    import time
    
    for attempt in range(max_retries):
        try:
            return call_search_api(search_params, headers)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1 * (attempt + 1))  # Exponential backoff 