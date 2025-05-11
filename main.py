from typing import Annotated, Dict, List, Any
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import CohereChatRequest, ChatDetails, OnDemandServingMode
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os
import json
from datetime import datetime
import time

import normalization
import validation
from tools import call_metadata, call_search_api, format_error_response
from helpers import (
    preprocess_query,
    handle_validation_errors,
    enhance_search_results,
    get_cached_metadata,
    call_search_api_with_retry
)

# OCI Configuration
CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/workshop', CONFIG_PROFILE)

# Service endpoint and model details
SERVICE_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
LLM_MODEL = "cohere.command-r-plus-08-2024"
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaaboqoghc43bbqvei5y4pmnzia36wuh7fpcxn6fia7pfyelyg4rj7a"

# Initialize Generative AI Client
llm_client = GenerativeAiInferenceClient(
    config=config,
    service_endpoint=SERVICE_ENDPOINT,
    retry_strategy=oci.retry.NoneRetryStrategy(),
    timeout=(10, 240)
)

# Fastapi APP
app = FastAPI(title="Aconex Search Assistant API")

# Request models
class SearchRequest(BaseModel):
    query: str

# Response Models
class ClarificationResponse(BaseModel):
    classification: str
    suggestions: list = []
    closest_matches: dict = {}

class SearchResponse(BaseModel):
    """Response model for search results."""
    documents: List[Dict[str, Any]]
    total_count: int
    applied_filters: Dict[str, Any]  # New field to store applied filters
    message: str = "Search completed successfully"

def build_preamble(metadata_fields: dict, query: str) -> str:
    """Build the enhanced preamble for the LLM."""
    # Extract only essential field information for searchable fields
    essential_fields = []
    
    # Process single value fields
    for field in metadata_fields.get("searchSchema", {}).get("singleValueFields", []):
        if field.get("searchable", False):  # Only include searchable fields
            field_info = {
                "fieldName": field.get("fieldName"),
                "identifier": field.get("identifier"),
                "dataType": field.get("dataType"),
                "selectValues": [v["value"] for v in field.get("selectValues", [])] if field.get("selectValues") else None
            }
            essential_fields.append(field_info)
    
    # Process multi value fields
    for field in metadata_fields.get("searchSchema", {}).get("multiValueFields", []):
        if field.get("searchable", False):  # Only include searchable fields
            field_info = {
                "fieldName": field.get("fieldName"),
                "identifier": field.get("identifier"),
                "dataType": field.get("dataType"),
                "selectValues": [v["value"] for v in field.get("selectValues", [])] if field.get("selectValues") else None
            }
            essential_fields.append(field_info)

    preamble = f"""
You are an AI assistant specialized in document search using structured metadata filters. Your task is to map user queries to the correct metadata fields and values.

Your task is to:
1. Understand the user's query.
2. Identify relevant metadata fields based on the query.
3. Match user-provided values to the **closest valid options** from the metadata above.
4. Return a structured **JSON object** with only valid field names and values.

Important Instructions:
- Only use metadata fields and values exactly as listed above.
- If a field is mentioned but the value is vague or partial (e.g., "administration"), choose the **best matching** allowed value.
- If multiple fields are relevant (e.g., discipline, status, date), include them all.
- If the user mentions a **person**, map it to appropriate fields such as `"Created By"`, `"Modified By"`, or `"Reviewed By"` — use whichever exists in the metadata.
- If a **year or date** is mentioned, return it using the appropriate date field (e.g., `"Date Created"`, `"Date Reviewed"`).
- For date ranges, use:
  - `"fieldname_gte"` for start date
  - `"fieldname_lte"` for end date
- If domain-specific keywords appear (e.g., "safety", "signage"), try to match them to the **closest discipline** or category field.
- If the user provides a general or partial term (e.g., "administration"), and multiple matching values exist (e.g., "AD - Administration", "Administration - HR"), return **all valid matching values** from the metadata
- Do not guess field names or values — use only what is in the metadata list.
- Return only one best-matching value per field, unless the query clearly asks for multiple.
- The response must be a clean, valid JSON object:
  - No comments
  - No markdown
  - No extra text — **just the JSON**

Example Response Format:
{{"discipline": ["EL - Electrical", "Electrical", "electrical"]}}

Available Fields (all fields are searchable):
{json.dumps(essential_fields, indent=2)}

User Query: "{query}"
"""
    print("\nBuilt Preamble:")
    print(preamble)
    return preamble

@app.post("/search", response_model=dict, responses={422: {"model": ClarificationResponse}})
def search_documents(
    req: SearchRequest,
    access_token: Annotated[str, Header(..., max_length=8024)],
    x_api_key: Annotated[str, Header(...)],
    accept_language: Annotated[str, Header()] = "en_US",
    application_type: Annotated[str, Header()] = "ANDROID PHONE",
    app_high_compliance: Annotated[str, Header(alias="app-high-compliance")] = "false",
):
    """Handle document search requests."""
    print("\n=== Starting Search Request ===")
    print(f"Query: {req.query}")
    
    # Prepare headers
    headers = {
        "authorization": access_token,
        "x-api-key": x_api_key,
        "accept-language": accept_language,
        "application-type": application_type,
        "app-high-compliance": app_high_compliance,
        "client-version": "25.1.0",
        "accept-encoding": "gzip"
    }
    
    try:
        # 1. Get metadata
        print("\n1. Fetching Metadata...")
        metadata = get_cached_metadata(headers)
        
        print("\nMetadata Structure:")
        print(f"Keys in metadata: {list(metadata.keys())}")
        print(f"Keys in searchSchema: {list(metadata.get('searchSchema', {}).keys())}")
        
        # 2. Preprocess query
        print("\n2. Preprocessing Query...")
        processed_query = req.query.strip()
        print(f"Processed Query: {processed_query}")
        
        # 3. Build preamble
        print("\n3. Building Preamble...")
        preamble = build_preamble(metadata, processed_query)
        print("Preamble built successfully")
        
        # 4. Call LLM with retries
        print("\n4. Calling LLM...")
        max_retries = 1
        retry_count = 0
        llm_response = {}
        
        while retry_count < max_retries:
            try:
                llm_response = call_llm(processed_query, preamble)
                if llm_response:  # If we got a non-empty response
                    break
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Empty LLM response, retrying... (Attempt {retry_count + 1}/{max_retries})")
            except Exception as e:
                print(f"LLM call failed: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying... (Attempt {retry_count + 1}/{max_retries})")
                else:
                    raise
        
        # 5. Parse LLM response
        print("\n5. Parsing LLM Response...")
        try:
            search_params = json.loads(llm_response)
            print(f"Parsed Search Parameters: {json.dumps(search_params, indent=2)}")
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response: {str(e)}")
            print(f"Raw LLM response: {llm_response}")
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Could not parse LLM response",
                    "raw_response": llm_response
                }
            )
        
        # 6. Call search API
        print("\n6. Calling Search API...")
        search_results = call_search_api(search_params, headers)
        print(f"Search Results: {json.dumps(search_results, indent=2)}")
        
        # 7. Prepare response with filters
        print("\n7. Preparing Response...")
        
        # Get the filter metadata for each applied filter
        filters = []
        for field_name, values in search_params.items():
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
                filters.append({
                    "applied_filters": {
                        field_name: values if isinstance(values, list) else [values]
                    },
                    "filter": field_metadata
                })
        
        response = {
            "results": {
                "documents": search_results.get("documents", []),
                "totalResults": str(search_results.get("totalCount", 0))
            },
            "metadata": {
                "total_count": search_results.get("totalCount", 0),
                "search_fields_used": list(search_params.keys()),
                "confidence_scores": {
                    "unknown": 1.0
                }
            },
            "filters": filters
        }
        
        print("\n=== Search Request Completed ===")
        return response
        
    except Exception as e:
        print(f"\nError in search_documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

def call_llm(prompt: str, preamble: str) -> str:
    """Call the LLM with the given prompt and preamble."""
    try:
        print("\nLLM Request Details:")
        print(f"Preamble: {preamble}")
        print(f"Prompt: {prompt}")
        
        llm_chat_request = CohereChatRequest()
        llm_chat_request.preamble_override = preamble
        llm_chat_request.message = prompt
        llm_chat_request.is_stream = False
        llm_chat_request.max_tokens = 500
        llm_chat_request.is_force_single_step = True

        chat_detail = ChatDetails()
        chat_detail.serving_mode = OnDemandServingMode(model_id=LLM_MODEL)
        chat_detail.compartment_id = COMPARTMENT_ID
        chat_detail.chat_request = llm_chat_request

        print("\nSending request to LLM...")
        chat_response = llm_client.chat(chat_detail)
        response_text = chat_response.data.chat_response.text
        
        print(f"\nRaw LLM Response: {response_text}")
        
        if not response_text:
            print("Warning: LLM returned empty response")
            return "{}"
            
        return response_text
        
    except Exception as e:
        print(f"Error in call_llm: {str(e)}")
        print(f"Error type: {type(e)}")
        raise

def parse_llm_response(llm_response: str) -> dict:
    """Parse the LLM response into a dictionary of search parameters."""
    try:
        if not llm_response:
            return {}
            
        # Try to parse as JSON
        return json.loads(llm_response)
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response: {str(e)}")
        return {}

def get_llm_response(preamble: str) -> str:
    """Get response from LLM with retries."""
    max_retries = 1
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            llm_response = call_llm("", preamble)  # Empty prompt since preamble contains the query
            if llm_response:  # If we got a non-empty response
                return llm_response
            retry_count += 1
            if retry_count < max_retries:
                print(f"Empty LLM response, retrying... (Attempt {retry_count + 1}/{max_retries})")
        except Exception as e:
            print(f"LLM call failed: {str(e)}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying... (Attempt {retry_count + 1}/{max_retries})")
            else:
                raise
    
    return "{}"  # Return empty JSON if all retries fail

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
