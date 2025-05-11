from typing import Dict, Any
import json
import requests
import time
from datetime import datetime

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

def call_search_api(params: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
    """Call the search API with the given parameters."""
    try:
        # Convert list values to comma-separated strings
        processed_params = {}
        for key, value in params.items():
            if isinstance(value, list):
                processed_params[key] = ",".join(value)
            else:
                processed_params[key] = value

        # Create request body with all required fields
        request_body = {
            **processed_params,  # Include processed search parameters
            "returnFields": RETURN_FIELDS,
            "userId": USER_ID,
            "orgId": ORG_ID
        }
        
        print("\nSearch API Request Body:")
        print(json.dumps(request_body, indent=2))
        
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
            
        return response.json()
        
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








# if __name__ == "__main__":
    # call_metadata(headers)
    # jsonParams = {
    #     "userId": "1477096489",
    #     "orgId": "9146",
    #     "returnFields": RETURN_FIELDS}
    # call_search_api(jsonParams)
