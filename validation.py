from typing import Dict, List, Tuple, Any
import json
from datetime import datetime
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re
import logging

logger = logging.getLogger(__name__)

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
        field_found = field_name in searchable_fields
        
        # Special handling for date range fields (field1, field2, fieldQualifier)
        if not field_found:
            # Check if this is a date range component
            if field_name.endswith("1") or field_name.endswith("2"):
                base_field = field_name[:-1]  # Remove the "1" or "2"
                if base_field in searchable_fields:
                    base_field_metadata = searchable_fields[base_field]
                    if base_field_metadata.get("dataType") == "DATE":
                        field_found = True
                        print(f"Recognized {field_name} as date range component of {base_field}")
            elif field_name.endswith("Qualifier"):
                base_field = field_name.replace("Qualifier", "")
                if base_field in searchable_fields:
                    base_field_metadata = searchable_fields[base_field]
                    if base_field_metadata.get("dataType") == "DATE":
                        field_found = True
                        print(f"Recognized {field_name} as date qualifier for {base_field}")
        
        if not field_found:
            print(f"Field {field_name} not found in searchable fields")
            invalid_fields.append(field_name)
            continue
            
        # Get field metadata (handle both direct fields and date range components)
        field_metadata = None
        if field_name in searchable_fields:
            field_metadata = searchable_fields[field_name]
        elif field_name.endswith("1") or field_name.endswith("2"):
            # Date range component - get metadata from base field
            base_field = field_name[:-1]
            if base_field in searchable_fields:
                field_metadata = searchable_fields[base_field]
        elif field_name.endswith("Qualifier"):
            # Date qualifier - get metadata from base field
            base_field = field_name.replace("Qualifier", "")
            if base_field in searchable_fields:
                field_metadata = searchable_fields[base_field]
        
        if not field_metadata:
            print(f"Could not find metadata for field: {field_name}")
            invalid_fields.append(field_name)
            continue
            
        print(f"Found field metadata: {field_metadata}")
        
        # Handle date fields with comprehensive validation
        data_type = field_metadata.get("dataType", "").lower()
        if data_type == "date":
            # Handle date qualifiers separately
            if field_name.endswith("Qualifier"):
                valid_qualifiers = ["BETWEEN", "ON", "BEFORE", "AFTER", "GTE", "LTE"]
                if value not in valid_qualifiers:
                    invalid_values[field_name] = f"Invalid qualifier '{value}'. Must be one of: {', '.join(valid_qualifiers)}"
                    continue
            else:
                # Date value validation
                if not is_valid_date(value):
                    invalid_values[field_name] = f"Invalid date format for {field_name}: {value}"
                    continue
                
        # Handle multi-value fields
        elif field_metadata.get("selectValues"):
            select_values = [v["value"] for v in field_metadata["selectValues"]]
            print(f"Checking value '{value}' against select values: {select_values}")
            # Case-insensitive match for select values
            if not is_valid_select_value(value, select_values):
                invalid_values[field_name] = value
                continue
                
        # Handle boolean fields (case-insensitive)
        elif data_type == "boolean":
            if not isinstance(value, bool) and not normalize_boolean_for_validation(value):
                invalid_values[field_name] = value
                continue
                
        # Handle string fields (case-insensitive)
        elif data_type == "string":
            # Accept any value that can be converted to string
            pass
                
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
    """
    Check if a value is a valid date string in API format or can be normalized.
    Supports API format, common formats, and natural language expressions.
    """
    if not isinstance(value, str):
        return False
        
    # API format: YYYY-MM-DDTHH:mm:ss.sssZ
    api_date_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$'
    
    if re.match(api_date_pattern, value):
        try:
            # Validate the actual datetime
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
            return True
        except ValueError:
            return False
    
    # Common date formats
    common_formats = [
        "%Y-%m-%d",           # 2025-06-14
        "%d/%m/%Y",           # 14/06/2025
        "%m/%d/%Y",           # 06/14/2025
        "%Y/%m/%d",           # 2025/06/14
        "%d-%m-%Y",           # 14-06-2025
        "%Y-%m-%d %H:%M:%S",  # 2025-06-14 10:30:00
        "%Y-%m-%dT%H:%M:%S",  # 2025-06-14T10:30:00
    ]
    
    for fmt in common_formats:
        try:
            datetime.strptime(value, fmt)
            return True
        except ValueError:
            continue
    
    # Natural language expressions that can be normalized
    natural_language_patterns = [
        r'^today$',
        r'^yesterday$',
        r'^tomorrow$',
        r'^last\s+(week|month|year)$',
        r'^this\s+(week|month|year)$',
        r'^last\s+\d+\s+(days?|weeks?|months?|years?)$',
        r'^\d+\s+(days?|weeks?|months?|years?)\s+ago$',
    ]
    
    value_lower = value.lower().strip()
    for pattern in natural_language_patterns:
        if re.match(pattern, value_lower):
            logger.debug(f"Recognized natural language date: '{value}'")
            return True
    
    logger.debug(f"Could not validate date: '{value}'")
    return False

def validate_date_field_structure(field_name: str, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate date field structure for range queries.
    
    Args:
        field_name: The base field name (e.g., "registered")
        params: All parameters to check for range structure
        
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Check for range structure: field1, field2, fieldQualifier
    field1_key = f"{field_name}1"
    field2_key = f"{field_name}2"
    qualifier_key = f"{field_name}Qualifier"
    
    has_field1 = field1_key in params
    has_field2 = field2_key in params
    has_qualifier = qualifier_key in params
    has_simple = field_name in params
    
    # Validate range structure consistency
    if has_field1 or has_field2 or has_qualifier:
        # Range query detected
        if not (has_field1 and has_field2 and has_qualifier):
            missing = []
            if not has_field1:
                missing.append(field1_key)
            if not has_field2:
                missing.append(field2_key)
            if not has_qualifier:
                missing.append(qualifier_key)
            
            errors.append(f"Incomplete date range for {field_name}. Missing: {', '.join(missing)}")
            return False, errors
        
        # Validate individual date values
        if not is_valid_date(params[field1_key]):
            errors.append(f"Invalid date format for {field1_key}: {params[field1_key]}")
        
        if not is_valid_date(params[field2_key]):
            errors.append(f"Invalid date format for {field2_key}: {params[field2_key]}")
        
        # Validate qualifier
        valid_qualifiers = ["BETWEEN", "ON", "BEFORE", "AFTER", "GTE", "LTE"]
        if params[qualifier_key] not in valid_qualifiers:
            errors.append(f"Invalid qualifier for {qualifier_key}: {params[qualifier_key]}. Must be one of: {', '.join(valid_qualifiers)}")
        
        # Cannot have both simple and range format
        if has_simple:
            errors.append(f"Cannot use both simple date ({field_name}) and range format ({field1_key}, {field2_key}) for the same field")
    
    elif has_simple:
        # Simple date query
        if not is_valid_date(params[field_name]):
            errors.append(f"Invalid date format for {field_name}: {params[field_name]}")
    
    return len(errors) == 0, errors

def get_date_field_names(metadata_fields: Dict[str, Any]) -> List[str]:
    """Extract all date field identifiers from metadata."""
    date_fields = []
    
    # Process single value fields
    for field in metadata_fields.get("searchSchema", {}).get("singleValueFields", []):
        if field.get("searchable", False) and field.get("dataType") == "DATE":
            date_fields.append(field["identifier"])
    
    # Process multi value fields  
    for field in metadata_fields.get("searchSchema", {}).get("multiValueFields", []):
        if field.get("searchable", False) and field.get("dataType") == "DATE":
            date_fields.append(field["identifier"])
    
    return date_fields

def normalize_boolean_for_validation(value: Any) -> bool:
    """Check if a value can be normalized to boolean for validation purposes."""
    if isinstance(value, bool):
        return True
        
    if isinstance(value, str):
        value = value.lower().strip()
        if value in ["true", "yes", "1", "y", "false", "no", "0", "n"]:
            return True
            
    return False

def is_valid_select_value(value: Any, select_values: List[str]) -> bool:
    """Check if a value is valid for a select field."""
    if not select_values:
        return True
        
    # Handle comma-separated strings from LLM
    if isinstance(value, str) and "," in value:
        values_to_check = [v.strip() for v in value.split(",")]
    elif isinstance(value, list):
        values_to_check = value
    else:
        values_to_check = [str(value)]
    
    # Check if all values are valid (case-insensitive)
    return all(
        any(v.lower() == allowed_value.lower() for allowed_value in select_values)
        for v in values_to_check
    )

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




