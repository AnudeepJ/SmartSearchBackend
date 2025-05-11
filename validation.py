from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def validate_llm_response(params: Dict[str, Any], metadata_fields: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
    """
    Validate the LLM response against metadata fields.
    
    Args:
        params: The parameters from LLM response
        metadata_fields: The metadata fields from the API
        verbose: Whether to print detailed validation information
        
    Returns:
        Dict containing validation results and cleaned output
    """
    invalid_fields = []
    invalid_values = {}
    cleaned_output = {}
    
    # Get searchable fields from metadata
    searchable_fields = {}
    print("\nValidating Fields:")
    
    # Process single value fields
    for field in metadata_fields.get("searchSchema", {}).get("singleValueFields", []):
        if field.get("searchable", False):
            searchable_fields[field["identifier"]] = field
            searchable_fields[field["fieldName"]] = field
            print(f"Added single value field - Identifier: {field['identifier']}, FieldName: {field['fieldName']}")
    
    # Process multi value fields
    for field in metadata_fields.get("searchSchema", {}).get("multiValueFields", []):
        if field.get("searchable", False):
            searchable_fields[field["identifier"]] = field
            searchable_fields[field["fieldName"]] = field
            print(f"Added multi value field - Identifier: {field['identifier']}, FieldName: {field['fieldName']}")
    
    print(f"\nTotal searchable fields: {len(searchable_fields)}")
    print(f"Available field keys: {list(searchable_fields.keys())}")
    
    # Validate each parameter
    for field_name, value in params.items():
        print(f"\nValidating field: {field_name}")
        # Check if field exists and is searchable
        if field_name not in searchable_fields:
            print(f"Field {field_name} not found in searchable fields")
            invalid_fields.append(field_name)
            continue
            
        field_metadata = searchable_fields[field_name]
        print(f"Found field metadata: {field_metadata}")
        
        # Handle date fields
        if field_metadata.get("dataType") == "date":
            if not is_valid_date(value):
                invalid_values[field_name] = value
                continue
                
        # Handle multi-value fields
        elif field_metadata.get("selectValues"):
            select_values = [v["value"] for v in field_metadata["selectValues"]]
            print(f"Checking value '{value}' against select values: {select_values}")
            # Case-insensitive match for select values
            if not is_valid_select_value(value, select_values):
                invalid_values[field_name] = value
                continue
                
        # Handle boolean fields
        elif field_metadata.get("dataType") == "boolean":
            if not isinstance(value, bool):
                invalid_values[field_name] = value
                continue
                
        # Handle string fields
        elif field_metadata.get("dataType") == "string":
            if not isinstance(value, str):
                invalid_values[field_name] = value
                continue
                
        # If validation passes, add to cleaned output
        cleaned_output[field_name] = value
        print(f"Field {field_name} validated successfully")
        
    # Prepare validation result
    validation_result = {
        "valid": len(invalid_fields) == 0 and len(invalid_values) == 0,
        "invalid_fields": invalid_fields,
        "invalid_values": invalid_values,
        "cleaned_output": cleaned_output
    }
    
    if verbose:
        print_validation_details(validation_result)
        
    return validation_result

def is_valid_date(value: Any) -> bool:
    """Check if a value is a valid date string."""
    if not isinstance(value, str):
        return False
        
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def is_valid_select_value(value: Any, select_values: List[str]) -> bool:
    """Check if a value is valid for a select field."""
    if not select_values:
        return True
        
    # Handle both single string values and lists of values
    if isinstance(value, list):
        # For lists, check if all values are valid
        return all(
            any(v.lower() == allowed_value.lower() for allowed_value in select_values)
            for v in value
        )
    else:
        # For single values, check if it matches any allowed value
        return any(value.lower() == allowed_value.lower() for allowed_value in select_values)

def print_validation_details(validation_result: Dict[str, Any]) -> None:
    """Print detailed validation information."""
    print("\nValidation Results:")
    print(f"Valid: {validation_result['valid']}")
    
    if validation_result["invalid_fields"]:
        print("\nInvalid Fields:")
        for field in validation_result["invalid_fields"]:
            print(f"- {field}")
            
    if validation_result["invalid_values"]:
        print("\nInvalid Values:")
        for field, value in validation_result["invalid_values"].items():
            print(f"- {field}: {value}")
            
    print("\nCleaned Output:")
    print(json.dumps(validation_result["cleaned_output"], indent=2))

def get_field_metadata(field_name: str, metadata_fields: Dict[str, Any]) -> Dict[str, Any]:
    """Get metadata for a specific field."""
    for field in metadata_fields.get("searchSchema", {}).get("fields", []):
        if field["identifier"] == field_name:
            return field
    return {}

def find_similar_fields(field_name: str, metadata_fields: Dict[str, Any], threshold: int = 80) -> List[Tuple[str, int]]:
    """Find fields similar to the given field name."""
    searchable_fields = [
        field["identifier"] 
        for field in metadata_fields.get("searchSchema", {}).get("fields", [])
        if field.get("searchable", False)
    ]
    
    return process.extract(field_name, searchable_fields, scorer=fuzz.ratio, limit=3)

def find_closest_valid_values(value: str, allowed_values: List[str], threshold: int = 80) -> List[Tuple[str, int]]:
    """Find the closest valid values for a given input."""
    return process.extract(value, allowed_values, scorer=fuzz.ratio, limit=3)


