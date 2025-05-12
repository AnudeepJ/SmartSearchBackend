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

def prepare_filters_for_ui(search_params: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Prepare filter objects for UI before API call.
    This ensures we have the correct field names and metadata before any transformations.
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
            
            # Create filter object with:
            # 1. Current selections (applied_filters)
            # 2. Complete field metadata including all possible values (filter)
            filter_obj = {
                "applied_filters": {
                    field_name: current_values  # Use original field name for UI
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
        else:
            print(f"No metadata found for {field_name}")
    
    print(f"\nTotal filters prepared: {len(filters)}")
    return filters

def get_modified_identifier(field_name: str) -> str:
    """
    Map metadata field identifiers to search API field names.
    Returns the original field name if no mapping exists.
    """
    field_mapping = {
        "vendordocumentnumber": "vendorDocumentNumber",
        "vendorrev": "vendorRev",
        "contractordocumentnumber": "contractorDocumentNumber",
        "contractorrev": "contractorRev",
        "packagenumber": "packageNumberValues",
        "contractnumber": "contractNumberValues",
        "statusid": "docstatus",  # Map statusid to docstatus
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
    return field_mapping.get(field_name, field_name)

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

def enhance_search_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance search results with additional metadata.
    
    Args:
        results: The raw search results
        
    Returns:
        Dict containing enhanced search results
    """
    # Get documents and totalResults from results
    documents = results.get("documents", [])
    total_results = results.get("totalResults", "0")
    
    # Create enhanced response
    enhanced = {
        "documents": documents,
        "totalResults": total_results,
        "metadata": {
            "total_count": len(documents),
            "search_fields_used": extract_used_fields(results),
            "confidence_scores": calculate_confidence_scores(results)
        }
    }
    
    return enhanced

def extract_used_fields(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract fields used in the search results."""
    used_fields = {}
    
    for doc in results.get("documents", []):
        for field, value in doc.items():
            if field not in used_fields:
                used_fields[field] = set()
            if isinstance(value, list):
                used_fields[field].update(value)
            else:
                used_fields[field].add(value)
                
    return {k: list(v) for k, v in used_fields.items()}

def calculate_confidence_scores(results: Dict[str, Any]) -> Dict[str, float]:
    """Calculate confidence scores for search results."""
    scores = {}
    
    for doc in results.get("documents", []):
        doc_id = doc.get("id")
        if doc_id:
            # Calculate score based on field match quality
            score = 0.0
            total_fields = 0
            
            for field, value in doc.items():
                if field not in ["id", "timestamp"]:
                    total_fields += 1
                    # Simple scoring based on field presence
                    score += 1.0
                    
            if total_fields > 0:
                scores[doc_id] = score / total_fields
                
    return scores

def format_error_response(error: Exception) -> Dict[str, Any]:
    """Format error response for API."""
    return {
        "error": {
            "message": str(error),
            "type": error.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        }
    }

def prepare_filters_for_api(filters: dict) -> dict:
    """
    Normalize filter keys for API:
    - Lowercase first character of each key.
    - If a value is a list, join all elements into a comma-separated string.
    - If a value is not a list, use it directly.
    """
    api_filters = {}
    for field, value in filters.items():
        # Lowercase only the first character of field name
        api_key = field[0].lower() + field[1:]

        # Determine API value
        if isinstance(value, list):
            api_value = ",".join(map(str, value))
        else:
            api_value = value

        api_filters[api_key] = api_value
    return api_filters

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
