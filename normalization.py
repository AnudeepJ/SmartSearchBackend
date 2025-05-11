from typing import Dict, List, Any, Union
from datetime import datetime, timedelta
import re
from fuzzywuzzy import process

def normalize_fields(params: Dict[str, Any], metadata_fields: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Normalize field names and values in the parameters.
    
    Args:
        params: The parameters to normalize
        metadata_fields: The metadata fields from the API
        verbose: Whether to print detailed normalization information
        
    Returns:
        Dict containing normalized parameters
    """
    normalized_params = {}
    
    # Get searchable fields from metadata
    searchable_fields = {
        field["identifier"]: field 
        for field in metadata_fields.get("searchSchema", {}).get("fields", [])
        if field.get("searchable", False)
    }
    
    # Normalize each parameter
    for field_name, value in params.items():
        # Normalize field name
        normalized_field = normalize_field_name(field_name, searchable_fields)
        if not normalized_field:
            if verbose:
                print(f"⚠️ Could not normalize field name: {field_name}")
            continue
            
        # Normalize value based on field type
        field_metadata = searchable_fields[normalized_field]
        normalized_value = normalize_value(value, field_metadata)
        
        if normalized_value is not None:
            normalized_params[normalized_field] = normalized_value
            
    if verbose:
        print_normalization_details(params, normalized_params)
        
    return normalized_params

def normalize_field_name(field_name: str, searchable_fields: Dict[str, Any]) -> str:
    """Normalize a field name to its standard identifier."""
    # Direct match
    if field_name in searchable_fields:
        return field_name
        
    # Try to find similar field
    matches = process.extract(field_name, searchable_fields.keys(), limit=1)
    if matches and matches[0][1] >= 90:  # High similarity threshold
        return matches[0][0]
        
    return ""

def normalize_value(value: Any, field_metadata: Dict[str, Any]) -> Any:
    """Normalize a value based on its field type."""
    if value is None:
        return None
        
    # Handle lists
    if isinstance(value, list):
        return [normalize_value(v, field_metadata) for v in value]
        
    # Handle date fields
    if field_metadata.get("dataType") == "date":
        return normalize_date(value)
        
    # Handle select fields
    if field_metadata.get("selectValues"):
        return normalize_select_value(value, field_metadata["selectValues"])
        
    # Handle boolean fields
    if field_metadata.get("dataType") == "boolean":
        return normalize_boolean(value)
        
    # Handle string fields
    if field_metadata.get("dataType") == "string":
        return str(value).strip()
        
    return value

def normalize_date(value: Any) -> str:
    """Normalize a date value to YYYY-MM-DD format."""
    if isinstance(value, str):
        # Try to parse common date formats
        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"]:
            try:
                return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
                
        # Try to parse relative dates
        if re.match(r"last\s+(\d+)\s+(day|week|month|year)s?", value.lower()):
            match = re.match(r"last\s+(\d+)\s+(day|week|month|year)s?", value.lower())
            if match:
                number = int(match.group(1))
                unit = match.group(2)
                delta = {
                    "day": timedelta(days=number),
                    "week": timedelta(weeks=number),
                    "month": timedelta(days=30 * number),
                    "year": timedelta(days=365 * number)
                }[unit]
                return (datetime.now() - delta).strftime("%Y-%m-%d")
                
    return None

def normalize_select_value(value: str, allowed_values: List[str]) -> str:
    """Normalize a select value to match allowed values."""
    # Direct match
    if value in allowed_values:
        return value
        
    # Try to find similar value
    matches = process.extract(value, allowed_values, limit=1)
    if matches and matches[0][1] >= 90:  # High similarity threshold
        return matches[0][0]
        
    return None

def normalize_boolean(value: Any) -> bool:
    """Normalize a value to boolean."""
    if isinstance(value, bool):
        return value
        
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ["true", "yes", "1", "y"]:
            return True
        if value in ["false", "no", "0", "n"]:
            return False
            
    return None

def print_normalization_details(original: Dict[str, Any], normalized: Dict[str, Any]) -> None:
    """Print detailed normalization information."""
    print("\nNormalization Results:")
    print("\nOriginal Parameters:")
    for field, value in original.items():
        print(f"- {field}: {value}")
        
    print("\nNormalized Parameters:")
    for field, value in normalized.items():
        print(f"- {field}: {value}")
        
    print("\nChanges:")
    for field in set(original.keys()) | set(normalized.keys()):
        if field in original and field in normalized:
            if original[field] != normalized[field]:
                print(f"- {field}: {original[field]} → {normalized[field]}")
        elif field in original:
            print(f"- Removed: {field}")
        else:
            print(f"- Added: {field}")


