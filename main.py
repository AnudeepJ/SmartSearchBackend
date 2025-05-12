from typing import Annotated, Dict, List, Any
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import CohereChatRequest, ChatDetails, OnDemandServingMode
import os
import json
from datetime import datetime
import time
import logging
from logging.handlers import RotatingFileHandler
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
from mapping import get_modified_identifier

# Configure logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a logger
logger = logging.getLogger("search_api")
logger.setLevel(logging.DEBUG)

# Create handlers
log_file = os.path.join(log_dir, "search_api.log")
file_handler = RotatingFileHandler(
    log_file, 
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
console_handler = logging.StreamHandler()

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Initialize FastAPI app first
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

# Constants for field identifiers
VENDOR_DOCUMENT_NUMBER = "vendordocumentnumber"
VENDOR_DOCUMENT_NUMBER_CAMEL_CASE = "vendorDocumentNumber"
VENDOR_REV = "vendorrev"
VENDOR_REV_CAMEL_CASE = "vendorRev"
CONTRACTOR_DOCUMENT_NUMBER = "contractordocumentnumber"
CONTRACTOR_DOCUMENT_NUMBER_CAMEL_CASE = "contractorDocumentNumber"
CONTRACTOR_REV = "contractorrev"
CONTRACT_REV_CAMEL_CASE = "contractorRev"
PACKAGE_NUMBER = "packagenumber"
PACKAGE_NUMBER_VALUES = "packageNumberValues"
CONTRACT_NUMBER = "contractnumber"
CONTRACT_NUMBER_VALUES = "contractNumberValues"
DOCUMENT_STATUS = "statusid"
DOC_STATUS_ID_SMALL_CASE = "docstatus"
VDR_CODE = "vdrcode"
VDR_CODE_VALUES = "vdrCodeValues"
CATEGORY = "category"
CATEGORY_VALUES = "categoryValues"
ATTRIBUTE_1 = "attribute1"
ATTRIBUTE_2 = "attribute2"
ATTRIBUTE_3 = "attribute3"
ATTRIBUTE_4 = "attribute4"
SELECT_LIST_1 = "selectList1"
SELECT_LIST_2 = "selectList2"
SELECT_LIST_3 = "selectList3"
SELECT_LIST_4 = "selectList4"
SELECT_LIST_5 = "selectList5"
SELECT_LIST_6 = "selectList6"
SELECT_LIST_7 = "selectList7"
SELECT_LIST_8 = "selectList8"
SELECT_LIST_9 = "selectList9"
SELECT_LIST_10 = "selectList10"
ATTRIBUTE_1_VALUES = "attribute1Values"
ATTRIBUTE_2_VALUES = "attribute2Values"
ATTRIBUTE_3_VALUES = "attribute3Values"
ATTRIBUTE_4_VALUES = "attribute4Values"
SELECT_LIST_1_VALUES = "selectList1Values"
SELECT_LIST_2_VALUES = "selectList2Values"
SELECT_LIST_3_VALUES = "selectList3Values"
SELECT_LIST_4_VALUES = "selectList4Values"
SELECT_LIST_5_VALUES = "selectList5Values"
SELECT_LIST_6_VALUES = "selectList6Values"
SELECT_LIST_7_VALUES = "selectList7Values"
SELECT_LIST_8_VALUES = "selectList8Values"
SELECT_LIST_9_VALUES = "selectList9Values"
SELECT_LIST_10_VALUES = "selectList10Values"

def get_modified_identifier(identifier: str) -> str:
    """Map metadata field identifiers to search API field names."""
    mapping = {
        VENDOR_DOCUMENT_NUMBER: VENDOR_DOCUMENT_NUMBER_CAMEL_CASE,
        VENDOR_REV: VENDOR_REV_CAMEL_CASE,
        CONTRACTOR_DOCUMENT_NUMBER: CONTRACTOR_DOCUMENT_NUMBER_CAMEL_CASE,
        CONTRACTOR_REV: CONTRACT_REV_CAMEL_CASE,
        PACKAGE_NUMBER: PACKAGE_NUMBER_VALUES,
        CONTRACT_NUMBER: CONTRACT_NUMBER_VALUES,
        DOCUMENT_STATUS: DOC_STATUS_ID_SMALL_CASE,
        VDR_CODE: VDR_CODE_VALUES,
        CATEGORY: CATEGORY_VALUES,
        ATTRIBUTE_1: ATTRIBUTE_1_VALUES,
        ATTRIBUTE_2: ATTRIBUTE_2_VALUES,
        ATTRIBUTE_3: ATTRIBUTE_3_VALUES,
        ATTRIBUTE_4: ATTRIBUTE_4_VALUES,
        SELECT_LIST_1: SELECT_LIST_1_VALUES,
        SELECT_LIST_2: SELECT_LIST_2_VALUES,
        SELECT_LIST_3: SELECT_LIST_3_VALUES,
        SELECT_LIST_4: SELECT_LIST_4_VALUES,
        SELECT_LIST_5: SELECT_LIST_5_VALUES,
        SELECT_LIST_6: SELECT_LIST_6_VALUES,
        SELECT_LIST_7: SELECT_LIST_7_VALUES,
        SELECT_LIST_8: SELECT_LIST_8_VALUES,
        SELECT_LIST_9: SELECT_LIST_9_VALUES,
        SELECT_LIST_10: SELECT_LIST_10_VALUES
    }
    return mapping.get(identifier, identifier)

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
- For multiple values, use comma-separated strings instead of arrays. For example:
  - Correct: {{"discipline": "EL - Electrical,Electrical,electrical"}}
  - Incorrect: {{"discipline": ["EL - Electrical", "Electrical", "electrical"]}}
- The response must be a clean, valid JSON object:
  - No comments
  - No markdown
  - No extra text — **just the JSON**

Example Response Format:
{{"discipline": "EL - Electrical,Electrical,electrical"}}

Available Fields (all fields are searchable):
{json.dumps(essential_fields, indent=2)}

User Query: "{query}"
"""
    logger.debug("\nBuilt Preamble:")
    logger.debug(preamble)
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
    logger.info("\n=== Starting Search Request ===")
    logger.info(f"Query: {req.query}")
    
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
        logger.info("\n1. Fetching Metadata...")
        metadata = get_cached_metadata(headers)
        
        logger.debug("\nMetadata Summary:")
        logger.debug(f"Total fields: {len(metadata.get('searchSchema', {}).get('singleValueFields', [])) + len(metadata.get('searchSchema', {}).get('multiValueFields', []))}")
        
        # 2. Preprocess query
        logger.info("\n2. Preprocessing Query...")
        processed_query = req.query.strip()
        logger.info(f"Processed Query: {processed_query}")
        
        # 3. Build preamble
        logger.info("\n3. Building Preamble...")
        preamble = build_preamble(metadata, processed_query)
        logger.info("Preamble built successfully")
        
        # 4. Call LLM with retries
        logger.info("\n4. Calling LLM...")
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
                    logger.warning(f"Empty LLM response, retrying... (Attempt {retry_count + 1}/{max_retries})")
            except Exception as e:
                logger.error(f"LLM call failed: {str(e)}")
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Retrying... (Attempt {retry_count + 1}/{max_retries})")
                else:
                    raise
        
        # 5. Parse LLM response
        logger.info("\n5. Parsing LLM Response...")
        try:
            search_params = json.loads(llm_response)
            logger.debug(f"Raw Search Parameters: {json.dumps(search_params, indent=2)}")
            #
            # # Transform the parameters to use correct field names and values
            # transformed_params = {
            #     "returnFields": [],
            #     "userId": "1477096489",
            #     "orgId": "9146"
            # }
            #
            # for field_name, value in search_params.items():
            #     # Get the correct field name for API using the mapping
            #     api_field_name = get_modified_identifier(field_name)
            #     print(f"Processing field: {field_name} -> {api_field_name}")
            #
            #     # Find the field in metadata to get correct IDs
            #     field_metadata = None
            #     print(f"Looking for field {field_name} in metadata...")
            #     print("Available single value fields:", [f.get("identifier") for f in metadata.get("searchSchema", {}).get("singleValueFields", [])])
            #     print("Available multi value fields:", [f.get("identifier") for f in metadata.get("searchSchema", {}).get("multiValueFields", [])])
            #
            #     for field in metadata.get("searchSchema", {}).get("singleValueFields", []):
            #         if field.get("identifier") == field_name:
            #             field_metadata = field
            #             print(f"Found metadata in singleValueFields: {field}")
            #             break
            #     if not field_metadata:
            #         for field in metadata.get("searchSchema", {}).get("multiValueFields", []):
            #             if field.get("identifier") == field_name:
            #                 field_metadata = field
            #                 print(f"Found metadata in multiValueFields: {field}")
            #                 break
            #
            #     # Always add the field to transformed_params, even if metadata is not found
            #     if isinstance(value, list):
            #         transformed_params[api_field_name] = ",".join(value)
            #     else:
            #         transformed_params[api_field_name] = value
            #
            #     print(f"Current transformed_params: {transformed_params}")
            #
            # search_params = transformed_params
            logger.debug(f"Transformed Search Parameters: {json.dumps(search_params, indent=2)}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            logger.error(f"Raw LLM response: {llm_response}")
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Could not parse LLM response",
                    "raw_response": llm_response
                }
            )
        
        # 6. Call search API
        logger.info("\n6. Calling Search API...")
        response = call_search_api(search_params, headers, metadata)
        logger.debug(f"Search Results Summary: {len(response.get('results', {}).get('documents', []))} documents found")
        
        logger.info("\n=== Search Request Completed ===")
        return response
        
    except Exception as e:
        logger.error(f"\nError in search_documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

def call_llm(prompt: str, preamble: str) -> str:
    """Call the LLM with the given prompt and preamble."""
    try:
        logger.debug("\nLLM Request Summary:")
        logger.debug(f"Preamble length: {len(preamble)} chars")
        logger.debug(f"Prompt length: {len(prompt)} chars")
        
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

        logger.info("\nSending request to LLM...")
        chat_response = llm_client.chat(chat_detail)
        response_text = chat_response.data.chat_response.text
        
        logger.debug(f"\nLLM Response Summary: {len(response_text)} chars")
        
        if not response_text:
            logger.warning("Warning: LLM returned empty response")
            return "{}"
            
        return response_text
        
    except Exception as e:
        logger.error(f"Error in call_llm: {str(e)}")
        logger.error(f"Error type: {type(e)}")
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
