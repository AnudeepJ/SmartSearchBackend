from typing import Dict, Any, List
import json
import requests
import time
from datetime import datetime
from mapping import get_modified_identifier
import semantic_mapping
import re
from config import config

# Constants
with open("return_fields.json") as f:
    RETURN_FIELDS = json.load(f)
    USER_ID = "1879053256"
    ORG_ID = "1"

# API Endpoints - Now dynamically loaded from environment config
METADATA_API_URL = config.get_metadata_url()
SEARCH_API_URL = config.get_search_url()

# Cache configuration
CACHE_TTL = 900  # 5 minutes in seconds
metadata_cache = {
    "data": None,
    "timestamp": None
}

def get_cached_metadata(headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Get metadata with caching.
    
    Args:
        headers: The request headers
        
    Returns:
        Dict containing metadata fields
    """
    current_time = time.time()
    
    # Check if cache is valid
    if (metadata_cache["data"] is not None and 
        metadata_cache["timestamp"] is not None and 
        current_time - metadata_cache["timestamp"] < CACHE_TTL):
        return metadata_cache["data"]
        
    # Fetch fresh metadata
    try:
        metadata = call_metadata(headers)
        metadata_cache["data"] = metadata
        metadata_cache["timestamp"] = current_time
        return metadata
    except Exception as e:
        # If cache exists but is expired, return it as fallback
        if metadata_cache["data"] is not None:
            return metadata_cache["data"]
        raise e

def call_metadata(headers: Dict[str, str]) -> Dict[str, Any]:
    """
    Call the metadata API.
    
    Args:
        headers: The request headers
        
    Returns:
        Dict containing metadata fields
    """
    try:
        response = requests.get(
            METADATA_API_URL,
            headers=headers,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch metadata: {str(e)}")

def transform_search_response(search_results: Dict[str, Any], search_params: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform search results to support UI filter management.
    For each filter field:
    - Shows currently applied values from search parameters
    - Provides complete metadata including all possible select values
    - Enables UI to show current selections and allow modifications
    
    Args:
        search_results: The raw search results from the API
        search_params: The search parameters used (current filter selections)
        metadata: The metadata containing field definitions and possible values
        
    Returns:
        Dict containing:
        - results: Search results
        - metadata: Search metadata
        - filters: List of filter objects with:
          - applied_filters: Current selections
          - filter: Complete field metadata with all possible values
    """
    print("\n=== Debug: Transforming Search Response ===")
    
    # Get the filters that were prepared before the API call
    filters = search_results.get("_prepared_filters", [])
    print(f"Found {len(filters)} prepared filters")
    
    return {
        "results": {
            "documents": search_results.get("documents", []),
            "totalResults": str(search_results.get("totalCount", 0))
        },
        "metadata": {
            "total_count": search_results.get("totalCount", 0),
            "search_fields_used": [k for k in search_params.keys() if k not in ["returnFields", "userId", "orgId"]],
            "confidence_scores": {
                "unknown": 1.0
            }
        },
        "filters": filters  # List of filters with current selections and all possible values
    }

def convert_ids_to_display_values(ids: List[str], field_metadata: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Convert ID values back to display values with both ID and text.
    
    Args:
        ids: List of ID values to convert
        field_metadata: Field metadata containing selectValues
        
    Returns:
        List of objects with 'id' and 'display' keys
    """
    result = []
    for id_value in ids:
        # Try to find the display value for this ID
        display_value = None
        for select_value in field_metadata.get("selectValues", []):
            if str(select_value.get("id", "")) == str(id_value):
                display_value = select_value.get("value", id_value)
                break
        
        # If no display value found, use the ID as display
        if display_value is None:
            display_value = id_value
            
        result.append({
            "id": id_value,
            "display": display_value
        })
    
    return result

def format_date_for_ui(date_value: str) -> str:
    """
    Format API date value for UI display.
    
    Args:
        date_value: Date in API format (YYYY-MM-DDTHH:mm:ss.sssZ)
        
    Returns:
        Human-readable date string
    """
    try:
        # Parse API format
        if re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$', date_value):
            parsed_date = datetime.strptime(date_value, "%Y-%m-%dT%H:%M:%S.%fZ")
            # Format for UI display
            return parsed_date.strftime("%B %d, %Y")  # e.g., "June 14, 2025"
        
        # If already in simple format
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', date_value):
            parsed_date = datetime.strptime(date_value, "%Y-%m-%d")
            return parsed_date.strftime("%B %d, %Y")
        
        # Return as-is if can't parse
        return date_value
        
    except Exception:
        return date_value

def prepare_date_filter_for_ui(field_name: str, params: Dict[str, Any], field_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare date field filter for UI display.
    Handles both simple dates and date ranges.
    
    Args:
        field_name: Base field name (e.g., "registered")
        params: All search parameters
        field_metadata: Field metadata for the date field
        
    Returns:
        Date filter object with proper UI formatting
    """
    # Check for range structure
    field1_key = f"{field_name}1"
    field2_key = f"{field_name}2"
    qualifier_key = f"{field_name}Qualifier"
    
    has_range = field1_key in params or field2_key in params or qualifier_key in params
    has_simple = field_name in params
    
    if has_range:
        # Date range structure
        start_date = params.get(field1_key, "")
        end_date = params.get(field2_key, "")
        qualifier = params.get(qualifier_key, "BETWEEN")
        
        # Format dates for display
        display_start = format_date_for_ui(start_date) if start_date else ""
        display_end = format_date_for_ui(end_date) if end_date else ""
        
        # Create range description and compatibility values based on qualifier
        if qualifier == "BETWEEN" and display_start and display_end:
            range_display = f"{display_start} to {display_end}"
            compatibility_value = f"{start_date},{end_date}"
            full_id = f"{start_date},{end_date},{qualifier}"
        elif qualifier == "BEFORE" and display_start:
            range_display = f"Before {display_start}"
            compatibility_value = start_date  # Only start date for BEFORE
            full_id = f"{start_date},{qualifier}"
        elif qualifier == "AFTER" and display_start:
            range_display = f"After {display_start}"
            compatibility_value = start_date  # Only start date for AFTER  
            full_id = f"{start_date},{qualifier}"
        else:
            range_display = f"{display_start} ({qualifier.lower()})"
            compatibility_value = start_date
            full_id = f"{start_date},{qualifier}"
        
        applied_filters = {
            field_name: [compatibility_value],          # Optimized for compatibility
            f"{field_name}_display": [range_display],   # Human-readable range
            f"{field_name}_full": [{
                "id": full_id,
                "display": range_display,
                "type": "date_range",
                "start_date": start_date,
                "end_date": end_date if qualifier == "BETWEEN" else None,
                "qualifier": qualifier,
                "start_display": display_start,
                "end_display": display_end if qualifier == "BETWEEN" else None
            }]
        }
        
    elif has_simple:
        # Simple date structure
        date_value = params[field_name]
        display_value = format_date_for_ui(date_value)
        
        applied_filters = {
            field_name: [date_value],                   # Original value
            f"{field_name}_display": [display_value],   # Human-readable
            f"{field_name}_full": [{
                "id": date_value,
                "display": display_value,
                "type": "date_single"
            }]
        }
    else:
        applied_filters = {
            field_name: [],
            f"{field_name}_display": [],
            f"{field_name}_full": []
        }
    
    return {
        "applied_filters": applied_filters,
        "filter": {
            "fieldName": field_metadata.get("fieldName"),
            "identifier": field_name,
            "dataType": "DATE",
            "sortable": field_metadata.get("sortable", False),
            "searchable": field_metadata.get("searchable", True),
            "modifiedFieldName": field_metadata.get("fieldName"),
            "typeIdentifier": field_metadata.get("typeIdentifier", 0),
            "selectValues": [],  # Date fields don't have selectValues
            "dateField": True,   # Flag for UI to handle differently
            "supportedQualifiers": ["BETWEEN", "ON", "BEFORE", "AFTER", "GTE", "LTE"]
        }
    }

def get_date_field_names(metadata: Dict[str, Any]) -> List[str]:
    """Get list of date field identifiers from metadata."""
    date_fields = []
    
    # Process single value fields
    for field in metadata.get("searchSchema", {}).get("singleValueFields", []):
        if field.get("searchable", False) and field.get("dataType") == "DATE":
            date_fields.append(field["identifier"])
    
    # Process multi value fields  
    for field in metadata.get("searchSchema", {}).get("multiValueFields", []):
        if field.get("searchable", False) and field.get("dataType") == "DATE":
            date_fields.append(field["identifier"])
    
    return date_fields

def prepare_filters_for_ui(search_params: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Prepare filter objects for UI before API call with enhanced date handling.
    This ensures we have the correct field names and metadata before any transformations.
    Includes both ID values, human-readable display values, and special date field handling.
    """
    print("\n=== Debug: Preparing Filters for UI ===")
    print("Search params:", {k: v for k, v in search_params.items() if k not in ["returnFields", "userId", "orgId"]})
    
    filters = []
    date_fields = get_date_field_names(metadata)
    processed_fields = set()  # Track processed fields to avoid duplicates
    
    print(f"Date fields identified: {date_fields}")
    
    for field_name, values in search_params.items():
        # Skip non-search fields
        if field_name in ["returnFields", "userId", "orgId"]:
            continue
            
        # Skip if already processed (for date range components)
        if field_name in processed_fields:
            continue
            
        print(f"\nProcessing field: {field_name}")
        
        # Check if this is a date range component (field1, field2, fieldQualifier)
        is_date_component = False
        base_field = None
        
        if field_name.endswith("1") or field_name.endswith("2"):
            potential_base = field_name[:-1]
            if potential_base in date_fields:
                is_date_component = True
                base_field = potential_base
        elif field_name.endswith("Qualifier"):
            potential_base = field_name.replace("Qualifier", "")
            if potential_base in date_fields:
                is_date_component = True
                base_field = potential_base
        
        # If this is a date component, process the entire date structure
        if is_date_component:
            if base_field not in processed_fields:
                print(f"Processing date field structure for: {base_field}")
                
                # Find metadata for the base date field
                base_field_metadata = None
                for field in metadata.get("searchSchema", {}).get("singleValueFields", []):
                    if field.get("identifier") == base_field:
                        base_field_metadata = field
                        break
                if not base_field_metadata:
                    for field in metadata.get("searchSchema", {}).get("multiValueFields", []):
                        if field.get("identifier") == base_field:
                            base_field_metadata = field
                            break
                
                if base_field_metadata:
                    date_filter = prepare_date_filter_for_ui(base_field, search_params, base_field_metadata)
                    filters.append(date_filter)
                    
                    # Mark all related fields as processed
                    processed_fields.add(base_field)
                    processed_fields.add(f"{base_field}1")
                    processed_fields.add(f"{base_field}2") 
                    processed_fields.add(f"{base_field}Qualifier")
                    
                    print(f"Added date filter for {base_field}")
                    print(f"  - Display: {date_filter['applied_filters'].get(f'{base_field}_display', [])}")
            continue
        
        # Check if this is a simple date field
        if field_name in date_fields:
            print(f"Processing simple date field: {field_name}")
            
            # Find metadata for the date field
            field_metadata = None
            for field in metadata.get("searchSchema", {}).get("singleValueFields", []):
                if field.get("identifier") == field_name:
                    field_metadata = field
                    break
            if not field_metadata:
                for field in metadata.get("searchSchema", {}).get("multiValueFields", []):
                    if field.get("identifier") == field_name:
                        field_metadata = field
                        break
            
            if field_metadata:
                date_filter = prepare_date_filter_for_ui(field_name, search_params, field_metadata)
                filters.append(date_filter)
                processed_fields.add(field_name)
                
                print(f"Added simple date filter for {field_name}")
                print(f"  - Display: {date_filter['applied_filters'].get(f'{field_name}_display', [])}")
            continue
        
        # Handle non-date fields (existing logic)
        print(f"Processing non-date field: {field_name}")
        
        # Find the field metadata
        field_metadata = None
        for field in metadata.get("searchSchema", {}).get("singleValueFields", []):
            if field.get("identifier") == field_name:
                field_metadata = field
                break
        if not field_metadata:
            for field in metadata.get("searchSchema", {}).get("multiValueFields", []):
                if field.get("identifier") == field_name:
                    field_metadata = field
                    break
        
        if field_metadata:
            # Get current selections and ensure they're in a list
            if isinstance(values, str):
                current_values = [v.strip() for v in values.split(",")]
            elif isinstance(values, list):
                current_values = values
            else:
                current_values = [str(values)]
            
            # Convert IDs to display values with both ID and text
            applied_values = convert_ids_to_display_values(current_values, field_metadata)
            
            # Create filter object with:
            # 1. Current selections with both IDs and display values (applied_filters)
            # 2. Complete field metadata including all possible values (filter)
            filter_obj = {
                "applied_filters": {
                    field_name: [val["id"] for val in applied_values],  # Original ID array for compatibility
                    f"{field_name}_display": [val["display"] for val in applied_values],  # Human-readable values
                    f"{field_name}_full": applied_values  # Full objects with both id and display
                },
                "filter": {
                    "fieldName": field_metadata.get("fieldName"),
                    "identifier": field_name,  # Original identifier for UI
                    "dataType": field_metadata.get("dataType"),
                    "sortable": field_metadata.get("sortable", False),
                    "searchable": field_metadata.get("searchable", False),
                    "modifiedFieldName": field_metadata.get("fieldName"),
                    "typeIdentifier": field_metadata.get("typeIdentifier", 0),
                    "selectValues": field_metadata.get("selectValues", [])  # All possible values for UI
                }
            }
            filters.append(filter_obj)
            processed_fields.add(field_name)
            
            print(f"Added filter for {field_name}")
            print(f"  - IDs: {[val['id'] for val in applied_values]}")
            print(f"  - Display: {[val['display'] for val in applied_values]}")
        else:
            print(f"No metadata found for {field_name}")
    
    print(f"\nTotal filters prepared: {len(filters)}")
    return filters



def get_original_identifier(field_name: str) -> str:
    """
    Map API field names back to original metadata field identifiers.
    Returns the original field name if no mapping exists.
    """
    reverse_mapping = {
        "vendorDocumentNumber": "vendordocumentnumber",
        "vendorRev": "vendorrev",
        "contractorDocumentNumber": "contractordocumentnumber",
        "contractorRev": "contractorrev",
        "packageNumberValues": "packagenumber",
        "contractNumberValues": "contractnumber",
        "docstatus": "statusid",  # Map docstatus back to statusid
        "vdrCodeValues": "vdrcode",
        "categoryValues": "category",
        "attribute1Values": "attribute1",
        "attribute2Values": "attribute2",
        "attribute3Values": "attribute3",
        "attribute4Values": "attribute4",
        "selectList1Values": "selectList1",
        "selectList2Values": "selectList2",
        "selectList3Values": "selectList3",
        "selectList4Values": "selectList4",
        "selectList5Values": "selectList5",
        "selectList6Values": "selectList6",
        "selectList7Values": "selectList7",
        "selectList8Values": "selectList8",
        "selectList9Values": "selectList9",
        "selectList10Values": "selectList10",
        "disciplineValues": "discipline"  # Map disciplineValues back to discipline
    }
    return reverse_mapping.get(field_name, field_name)

def call_search_api(params: Dict[str, Any], headers: Dict[str, str], metadata: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Call the search API with the given parameters.
    
    Args:
        params: The search parameters
        headers: The request headers
        metadata: Optional metadata for field mapping and ID conversion
    """
    try:
        print("\n=== Debug: Starting Search API Call ===")
        print("Input params:", {k: v for k, v in params.items() if k not in ["returnFields", "userId", "orgId"]})
        
        # First prepare filters for UI using original field names
        filters = prepare_filters_for_ui(params, metadata)
        print(f"\nPrepared {len(filters)} filters")
        
        # Transform parameters using field mapping and metadata
        processed_params = {}
        for field_name, value in params.items():
            # Skip system fields
            if field_name in ["returnFields", "userId", "orgId"]:
                continue
                
            # Get the correct field name for API
            api_field_name = get_modified_identifier(field_name)
            print(f"Processing field for API: {field_name} -> {api_field_name}")
        processed_params = map_search_params(params, metadata)
        print(f"Processed params: {processed_params}")
        # Create request body with all required fields
        request_body = {
            **params,  # Include processed search parameters
            "returnFields": RETURN_FIELDS,
            "userId": USER_ID,
            "orgId": ORG_ID
        }
        print(f"Request url: {SEARCH_API_URL}")
        response = requests.post(
            SEARCH_API_URL,
            headers=headers,
            json=request_body,
            timeout=30
        )
        
        if response.status_code != 200:
            error_msg = f"Search API error: {response.text}"
            print(f"\nError calling search API: {error_msg}")
            raise Exception(error_msg)
        
        # Get search results and add the prepared filters
        search_results = response.json()
        search_results["_prepared_filters"] = filters
        
        # Transform the response to include filters
        return transform_search_response(search_results, params, metadata)
        
    except Exception as e:
        print(f"\nError in call_search_api: {str(e)}")
        raise



def map_search_params(raw_params: dict, metadata: dict) -> dict:
    """
    1. Rename each raw key via get_modified_identifier.
    2. Split comma-separated values.
    3. Lookup each token in metadata.multiValueFields by ORIGINAL identifier,
       replacing with its 'id' (if any) or leaving it alone.
    4. Re-join tokens into a CSV string.
    """
    # Build a reverse map from API key back to metadata identifier
    # e.g. "docstatus" → "statusid"
    field_mapping = {
        "vendordocumentnumber": "vendorDocumentNumber",
        "vendorrev": "vendorRev",
        "contractordocumentnumber": "contractorDocumentNumber",
        "contractorrev": "contractorRev",
        "packagenumber": "packageNumberValues",
        "contractnumber": "contractNumberValues",
        "statusid": "docstatus",
        "vdrcode": "vdrCodeValues",
        "category": "categoryValues",
        "attribute1": "attribute1Values",
        "attribute2": "attribute2Values",
        "attribute3": "attribute3Values",
        "attribute4": "attribute4Values",
        "selectList1": "selectList1Values",
        "selectList2": "selectList2Values",
        "selectList3": "selectList3Values",
        "selectList4": "selectList4Values",
        "selectList5": "selectList5Values",
        "selectList6": "selectList6Values",
        "selectList7": "selectList7Values",
        "selectList8": "selectList8Values",
        "selectList9": "selectList9Values",
        "selectList10": "selectList10Values"
    }
    reverse_map = {api: orig for orig, api in field_mapping.items()}

    mv_fields = metadata["searchSchema"]["multiValueFields"]
    result = {}

    for raw_key, raw_val in raw_params.items():
        # 1) rename the key
        api_key = get_modified_identifier(raw_key)

        # 2) find the original metadata identifier
        orig_key = reverse_map.get(api_key, raw_key)

        # 3) split into tokens
        tokens = [t.strip() for t in str(raw_val).split(",") if t.strip()]

        # 4) lookup the metadata field
        meta_field = next((f for f in mv_fields if f["identifier"] == orig_key), None)

        mapped = []
        for tok in tokens:
            mapped_tok = tok
            if meta_field:
                for sv in meta_field.get("selectValues", []):
                    if sv.get("value") == tok:
                        # replace with id if present
                        mapped_tok = str(sv["id"]) if "id" in sv else tok
                        break
            mapped.append(mapped_tok)

        # 5) re-join
        result[api_key] = ",".join(mapped)

    return result







# if __name__ == "__main__":
    # call_metadata(headers)
    # jsonParams = {
    #     "userId": "1477096489",
    #     "orgId": "9146",
    #     "returnFields": RETURN_FIELDS}
    # call_search_api(jsonParams)
