from typing import Dict, Any, List
import json
import requests
import time
from datetime import datetime
from mapping import get_modified_identifier

# Constants
with open("return_fields.json") as f:
    RETURN_FIELDS = json.load(f)
    USER_ID = "1477096489"
    ORG_ID = "9146"

# API Endpoints
METADATA_API_URL = "https://au1.aconex.com/mobile/rest/projects/26851/metadata/documents"
SEARCH_API_URL = "https://au1.aconex.com/mobile/rest/projects/26851/documents/search/filter?content_search=false&sort_direction=DESC&page_number=1&sort=registered"

# Cache configuration
CACHE_TTL = 300  # 5 minutes in seconds
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
            timeout=10
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

def prepare_filters_for_ui(search_params: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Prepare filter objects for UI before API call.
    This ensures we have the correct field names and metadata before any transformations.
    Includes both ID values and human-readable display values.
    """
    print("\n=== Debug: Preparing Filters for UI ===")
    print("Search params:", {k: v for k, v in search_params.items() if k not in ["returnFields", "userId", "orgId"]})
    
    filters = []
    for field_name, values in search_params.items():
        # Skip non-search fields
        if field_name in ["returnFields", "userId", "orgId"]:
            continue
            
        print(f"\nProcessing field: {field_name}")
            
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
            **processed_params,  # Include processed search parameters
            "returnFields": RETURN_FIELDS,
            "userId": USER_ID,
            "orgId": ORG_ID
        }
        
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
