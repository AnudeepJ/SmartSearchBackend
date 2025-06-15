from typing import Dict, List, Any, Union, Optional
from datetime import datetime, timedelta
import re
from fuzzywuzzy import process
import logging
import semantic_mapping
from SearchPromtUtils import SearchPromptUtils

logger = logging.getLogger(__name__)

def normalize_fields(params: Dict[str, Any], metadata_fields: Dict[str, Any], query: str = "", verbose: bool = False) -> Dict[str, Any]:
    """
    Normalize field names and values in the parameters with enhanced date handling.
    
    Args:
        params: The parameters to normalize
        metadata_fields: The metadata fields from the API
        query: Original user query (for date range context)
        verbose: Whether to print detailed normalization information
        
    Returns:
        Dict containing normalized parameters
    """
    try:
        # First pass: Detect and convert date ranges based on query context
        if query:
            params = detect_and_convert_date_ranges(query, params, metadata_fields)
        
        # Second pass: Normalize date range structures
        params = normalize_date_range_structure(params, metadata_fields)
        
        # Third pass: Standard field and value normalization
        normalized_params = {}
        
        # Get searchable fields from metadata
        searchable_fields = {}
        
        # Process single value fields
        for field in metadata_fields.get("searchSchema", {}).get("singleValueFields", []):
            if field.get("searchable", False):
                searchable_fields[field["identifier"]] = field
        
        # Process multi value fields
        for field in metadata_fields.get("searchSchema", {}).get("multiValueFields", []):
            if field.get("searchable", False):
                searchable_fields[field["identifier"]] = field
        
        # Normalize each parameter
        for field_name, value in params.items():
            # Skip date range qualifier fields during normalization (they're handled separately)
            if field_name.endswith("Qualifier") and field_name.replace("Qualifier", "") in searchable_fields:
                normalized_params[field_name] = value  # Keep qualifiers as-is
                continue
                
            # Normalize field name
            normalized_field = normalize_field_name(field_name, searchable_fields)
            
            if not normalized_field:
                # Check if this is a date range field (field1, field2)
                if field_name.endswith("1") or field_name.endswith("2"):
                    base_field = field_name[:-1]  # Remove the "1" or "2"
                    if base_field in searchable_fields:
                        normalized_field = field_name  # Keep the range field as-is
                    else:
                        if verbose:
                            print(f"⚠️ Could not normalize field name: {field_name}")
                        continue
                else:
                    if verbose:
                        print(f"⚠️ Could not normalize field name: {field_name}")
                    continue
                
            # Normalize value based on field type
            if normalized_field in searchable_fields:
                field_metadata = searchable_fields[normalized_field]
            elif normalized_field.endswith("1") or normalized_field.endswith("2"):
                # For range fields, get metadata from base field
                base_field = normalized_field[:-1]
                field_metadata = searchable_fields.get(base_field)
            else:
                field_metadata = None
                
            if field_metadata:
                # Skip date normalization for range fields that were already processed
                is_date_range_field = (normalized_field.endswith("1") or normalized_field.endswith("2")) and \
                                     field_metadata.get("dataType") == "DATE"
                
                if is_date_range_field:
                    # Date range fields were already processed in normalize_date_range_structure
                    # Keep the value as-is to preserve format (simple for BEFORE/AFTER, ISO for BETWEEN)
                    normalized_params[normalized_field] = value
                elif field_metadata.get("dataType") == "DATE":
                    # Simple date fields - convert to simple YYYY-MM-DD format for ON qualifier
                    if 'T' in value and value.endswith('Z'):
                        # Convert ISO format to simple YYYY-MM-DD for ON qualifier
                        simple_date = value.split('T')[0]
                        normalized_params[normalized_field] = simple_date
                    else:
                        # Try to normalize to simple format
                        try:
                            from datetime import datetime
                            # Try common date formats and convert to simple YYYY-MM-DD
                            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"]:
                                try:
                                    parsed_date = datetime.strptime(value, fmt)
                                    simple_date = parsed_date.strftime("%Y-%m-%d")
                                    normalized_params[normalized_field] = simple_date
                                    break
                                except ValueError:
                                    continue
                            else:
                                # If can't parse, keep original value
                                normalized_params[normalized_field] = value
                        except Exception:
                            # Fallback: keep original value
                            normalized_params[normalized_field] = value
                else:
                    # Normal field normalization
                    normalized_value = normalize_value(value, field_metadata)
                    if normalized_value is not None:
                        normalized_params[normalized_field] = normalized_value
            else:
                # Keep unknown fields as-is (might be qualifiers or other special fields)
                normalized_params[normalized_field] = value
                
        if verbose:
            print_normalization_details(params, normalized_params)
            
        return normalized_params
        
    except Exception as e:
        logger.error(f"Error normalizing fields: {str(e)}")
        return params

def normalize_field_name(field_name: str, searchable_fields: Dict[str, Any]) -> str:
    """Normalize a field name to its standard identifier."""
    # Direct match
    if field_name in searchable_fields:
        return field_name
    
    # IMPORTANT: Don't do fuzzy matching for date range components
    # These should be preserved as-is (registered1, registered2, registeredQualifier)
    if (field_name.endswith("1") or field_name.endswith("2") or field_name.endswith("Qualifier")):
        return ""
        
    # Try to find similar field for non-date-range fields
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
        
    # Handle date fields (case-insensitive)
    data_type = field_metadata.get("dataType", "").lower()
    if data_type == "date":
        return normalize_date(value)
        
    # Handle select fields
    if field_metadata.get("selectValues"):
        # Extract the actual values from the selectValues structure
        allowed_values = [v["value"] if isinstance(v, dict) else v for v in field_metadata["selectValues"]]
        return normalize_select_value(value, allowed_values)
        
    # Handle boolean fields (case-insensitive)
    if data_type == "boolean":
        return normalize_boolean(value)
        
    # Handle string fields (case-insensitive)
    if data_type == "string":
        return str(value).strip()
        
    return value

def normalize_date(value: Any) -> str:
    """
    Normalize a date value to API-required ISO format.
    Handles natural language, common formats, and API format requirements.
    """
    if not isinstance(value, str):
        return None
        
    # If already in API format, validate and return
    api_date_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$'
    if re.match(api_date_pattern, value):
        try:
            datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
            return value  # Already in correct format
        except ValueError:
            pass
    
    # Try natural language parsing first
    try:
        parse_result = SearchPromptUtils.parse_natural_date(value)
        if parse_result['success'] and parse_result['start_date']:
            return SearchPromptUtils.convert_to_iso_format(parse_result['start_date'])
    except Exception as e:
        logger.debug(f"Natural language date parsing failed for '{value}': {str(e)}")
    
    # Try to parse common date formats
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
            parsed_date = datetime.strptime(value, fmt)
            return SearchPromptUtils.convert_to_iso_format(parsed_date)
        except ValueError:
            continue
            
    # Try to parse relative dates with regex
    relative_patterns = [
        (r"(\d+)\s+(day|days)\s+ago", lambda m: timedelta(days=-int(m.group(1)))),
        (r"(\d+)\s+(week|weeks)\s+ago", lambda m: timedelta(weeks=-int(m.group(1)))),
        (r"(\d+)\s+(month|months)\s+ago", lambda m: timedelta(days=-30 * int(m.group(1)))),
        (r"(\d+)\s+(year|years)\s+ago", lambda m: timedelta(days=-365 * int(m.group(1)))),
        (r"last\s+(\d+)\s+(day|days)", lambda m: timedelta(days=-int(m.group(1)))),
        (r"last\s+(\d+)\s+(week|weeks)", lambda m: timedelta(weeks=-int(m.group(1)))),
        (r"last\s+(\d+)\s+(month|months)", lambda m: timedelta(days=-30 * int(m.group(1)))),
        (r"last\s+(\d+)\s+(year|years)", lambda m: timedelta(days=-365 * int(m.group(1)))),
    ]
    
    value_lower = value.lower().strip()
    for pattern, delta_func in relative_patterns:
        match = re.match(pattern, value_lower)
        if match:
            try:
                delta = delta_func(match)
                result_date = datetime.now() + delta
                return SearchPromptUtils.convert_to_iso_format(result_date)
            except Exception as e:
                logger.debug(f"Error parsing relative date '{value}': {str(e)}")
                continue
    
    logger.warning(f"Could not normalize date value: '{value}'")
    return None

def normalize_date_range_structure(params: Dict[str, Any], metadata_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize date range structures in parameters.
    Converts natural language date ranges to proper API format.
    
    Args:
        params: Parameters that may contain date ranges
        metadata_fields: Metadata to identify date fields
        
    Returns:
        Normalized parameters with proper date range structures
    """
    try:
        # Build semantic mappings to identify date fields
        semantic_data = semantic_mapping.build_semantic_mappings(metadata_fields)
        date_fields = semantic_data.get('date_fields', [])
        
        normalized_params = params.copy()
        processed_fields = set()
        
        for field_name, value in params.items():
            # Skip if already processed as part of a range
            if field_name in processed_fields:
                continue
                
            # Determine the base field name for metadata lookup
            base_field_name = field_name
            if field_name.endswith("1") or field_name.endswith("2"):
                base_field_name = field_name[:-1]  # Remove "1" or "2"
            elif field_name.endswith("Qualifier"):
                base_field_name = field_name.replace("Qualifier", "")
                
            # Check if this is a date field (using base field name)
            field_metadata = None
            for field in metadata_fields.get("searchSchema", {}).get("singleValueFields", []) + \
                         metadata_fields.get("searchSchema", {}).get("multiValueFields", []):
                if field.get("identifier") == base_field_name and field.get("dataType") == "DATE":
                    field_metadata = field
                    break
                    
            if not field_metadata:
                continue
                
            # Check for existing range structure (using base field name)
            field1_key = f"{base_field_name}1"
            field2_key = f"{base_field_name}2"
            qualifier_key = f"{base_field_name}Qualifier"
            
            if field1_key in params or field2_key in params or qualifier_key in params:
                # Range structure exists, normalize based on qualifier type
                qualifier = params.get(qualifier_key, "BETWEEN").upper()
                
                # Process field1 with qualifier-specific date formatting
                if field1_key in params:
                    field1_value = params[field1_key]
                    
                    if qualifier in ["BEFORE", "AFTER"]:
                        # For BEFORE/AFTER: Convert to simple YYYY-MM-DD format
                        simple_date_pattern = r'^\d{4}-\d{2}-\d{2}$'
                        if re.match(simple_date_pattern, field1_value):
                            # Already in simple format, keep as-is
                            normalized_params[field1_key] = field1_value
                        else:
                            # Convert to simple format
                            # Check if it's already an ISO format with T
                            if 'T' in field1_value:
                                # Extract YYYY-MM-DD part from ISO format
                                simple_date = field1_value.split('T')[0]
                                normalized_params[field1_key] = simple_date
                            else:
                                # Try to parse and convert to simple format
                                try:
                                    from datetime import datetime
                                    # Try common date formats
                                    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"]:
                                        try:
                                            parsed_date = datetime.strptime(field1_value, fmt)
                                            simple_date = parsed_date.strftime("%Y-%m-%d")
                                            normalized_params[field1_key] = simple_date
                                            break
                                        except ValueError:
                                            continue
                                    else:
                                        # Fallback: keep original value
                                        normalized_params[field1_key] = field1_value
                                except Exception:
                                    # Fallback: keep original value
                                    normalized_params[field1_key] = field1_value
                    else:
                        # For BETWEEN/ON: Use full ISO format
                        normalized_date = normalize_date(field1_value)
                        if normalized_date:
                            normalized_params[field1_key] = normalized_date
                    
                    processed_fields.add(field1_key)
                    
                # Only process field2 for BETWEEN qualifier
                if qualifier == "BETWEEN" and field2_key in params:
                    normalized_date = normalize_date(params[field2_key])
                    if normalized_date:
                        normalized_params[field2_key] = normalized_date
                    processed_fields.add(field2_key)
                elif qualifier in ["BEFORE", "AFTER"] and field2_key in params:
                    # Remove field2 for BEFORE/AFTER qualifiers as it shouldn't exist
                    logger.info(f"Removing unnecessary {field2_key} for {qualifier} qualifier")
                    processed_fields.add(field2_key)  # Mark as processed to ignore it
                    
                # Process qualifier
                if qualifier_key in params:
                    if qualifier in ["BETWEEN", "ON", "BEFORE", "AFTER", "GTE", "LTE"]:
                        normalized_params[qualifier_key] = qualifier
                    processed_fields.add(qualifier_key)
                    
            else:
                # Simple date field - use simple YYYY-MM-DD format for ON qualifier
                if 'T' in value and value.endswith('Z'):
                    # Convert ISO format to simple YYYY-MM-DD for ON qualifier
                    simple_date = value.split('T')[0]
                    normalized_params[field_name] = simple_date
                else:
                    # Try to normalize to simple format
                    try:
                        from datetime import datetime
                        # Try common date formats and convert to simple YYYY-MM-DD
                        for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"]:
                            try:
                                parsed_date = datetime.strptime(value, fmt)
                                simple_date = parsed_date.strftime("%Y-%m-%d")
                                normalized_params[field_name] = simple_date
                                break
                            except ValueError:
                                continue
                        else:
                            # If can't parse, keep original value
                            normalized_params[field_name] = value
                    except Exception:
                        # Fallback: keep original value
                        normalized_params[field_name] = value
                    
            # Mark this base field as processed (including all its range components)
            processed_fields.add(base_field_name)
            processed_fields.add(f"{base_field_name}1")
            processed_fields.add(f"{base_field_name}2")
            processed_fields.add(f"{base_field_name}Qualifier")
        
        return normalized_params
        
    except Exception as e:
        logger.error(f"Error normalizing date range structure: {str(e)}")
        return params

def detect_and_convert_date_ranges(query: str, params: Dict[str, Any], metadata_fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect date range intentions in query and convert simple dates to range format if needed.
    
    Args:
        query: Original user query
        params: Current parameters
        metadata_fields: Metadata fields
        
    Returns:
        Parameters with date ranges converted as appropriate
    """
    try:
        # Build semantic mappings
        semantic_data = semantic_mapping.build_semantic_mappings(metadata_fields)
        
        # Extract date concepts and detect range intentions
        date_concepts = SearchPromptUtils.extract_date_concepts_from_query(query)
        query_lower = query.lower()
        
        # Detect range keywords
        range_keywords = ['between', 'from', 'to', 'range', 'last week', 'this month', 'last month']
        has_range_intent = any(keyword in query_lower for keyword in range_keywords)
        
        if not has_range_intent:
            return params
            
        updated_params = params.copy()
        
        # Look for date fields that could be converted to ranges
        for field_name, value in params.items():
            # Check if this is a date field
            field_metadata = None
            for field in metadata_fields.get("searchSchema", {}).get("singleValueFields", []) + \
                         metadata_fields.get("searchSchema", {}).get("multiValueFields", []):
                if field.get("identifier") == field_name and field.get("dataType") == "DATE":
                    field_metadata = field
                    break
                    
            if not field_metadata:
                continue
                
            # Try to parse the value as a natural language date expression
            try:
                parse_result = SearchPromptUtils.parse_natural_date(value)
                if parse_result['success']:
                    qualifier = parse_result['qualifier']
                    field1_key = f"{field_name}1"
                    field2_key = f"{field_name}2"
                    qualifier_key = f"{field_name}Qualifier"
                    
                    if qualifier == 'BETWEEN' and parse_result['end_date']:
                        # BETWEEN: Use full ISO format for both dates
                        updated_params[field1_key] = SearchPromptUtils.convert_to_iso_format(parse_result['start_date'])
                        updated_params[field2_key] = SearchPromptUtils.convert_to_iso_format(parse_result['end_date'])
                        updated_params[qualifier_key] = "BETWEEN"
                        
                        # Remove the original simple date field
                        if field_name in updated_params:
                            del updated_params[field_name]
                            
                        logger.info(f"Converted date field '{field_name}' to BETWEEN range format")
                        
                    elif qualifier in ['BEFORE', 'AFTER']:
                        # BEFORE/AFTER: Use simple YYYY-MM-DD format, only field1
                        simple_date = parse_result['start_date'].strftime("%Y-%m-%d")
                        updated_params[field1_key] = simple_date
                        updated_params[qualifier_key] = qualifier
                        
                        # Remove the original simple date field
                        if field_name in updated_params:
                            del updated_params[field_name]
                            
                        logger.info(f"Converted date field '{field_name}' to {qualifier} format with simple date: {simple_date}")
                    
            except Exception as e:
                logger.debug(f"Could not convert date field '{field_name}' to range: {str(e)}")
                continue
        
        return updated_params
        
    except Exception as e:
        logger.error(f"Error detecting and converting date ranges: {str(e)}")
        return params

def normalize_select_value(value: Any, allowed_values: List[str]) -> str:
    """Normalize a select value to match allowed values."""
    # Handle comma-separated values from LLM
    if isinstance(value, str) and "," in value:
        values_to_normalize = [v.strip() for v in value.split(",")]
        normalized_values = []
        
        for val in values_to_normalize:
            # Direct match
            if val in allowed_values:
                normalized_values.append(val)
            else:
                # Try to find similar value
                matches = process.extract(val, allowed_values, limit=1)
                if matches and matches[0][1] >= 90:  # High similarity threshold
                    normalized_values.append(matches[0][0])
        
        return ",".join(normalized_values) if normalized_values else None
    
    # Handle single values
    value_str = str(value)
    
    # Direct match
    if value_str in allowed_values:
        return value_str
        
    # Try to find similar value
    matches = process.extract(value_str, allowed_values, limit=1)
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


