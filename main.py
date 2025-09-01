from typing import Annotated, Dict, List, Any
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import CohereChatRequest, ChatDetails, OnDemandServingMode
import os
import json
from datetime import datetime, timedelta
import ast
import time
import logging
from logging.handlers import RotatingFileHandler
import normalization
import validation
from SearchPromtUtils import SearchPromptUtils
from tools import call_search_api
from helpers import (
    handle_validation_errors,
    get_cached_metadata
)
from mapping import get_modified_identifier, get_field_id
import semantic_mapping
import re

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

# Environment Configuration
from config import config as env_config

# OCI Configuration
CONFIG_PROFILE = "DEFAULT"
oci_config = oci.config.from_file('~/.oci/workshop', CONFIG_PROFILE)

# Service endpoint and model details
SERVICE_ENDPOINT = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
LLM_MODEL = "cohere.command-r-plus-08-2024"
COMPARTMENT_ID = "ocid1.compartment.oc1..aaaaaaaaboqoghc43bbqvei5y4pmnzia36wuh7fpcxn6fia7pfyelyg4rj7a"

# Initialize Generative AI Client with intelligent retry strategy
llm_client = GenerativeAiInferenceClient(
    config=oci_config,
    service_endpoint=SERVICE_ENDPOINT,
    retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,  # Enable automatic retries
    timeout=(10, 60)  # Reduced to 60 seconds
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



RELATIVE_DATE_QUALIFIERS = {
    "TODAY",
    "TOMORROW",
    "YESTERDAY",
    "LAST7DAYS",
    "LAST30DAYS",
    "LAST90DAYS",
    "NEXT7DAYS",
    "NEXT30DAYS",
    "NEXT90DAYS",
}

def _to_date_only_str(value: Any) -> str:
    try:
        s = str(value).strip()
        if "T" in s:
            return s.split("T", 1)[0]
        return s
    except Exception:
        return str(value)

def _to_between_iso_str(value: Any) -> str:
    try:
        s = str(value).strip()
        if "T" in s:
            return s
        # Append standard time per API expectation
        return f"{s}T05:30:00.000Z"
    except Exception:
        return str(value)

def normalize_date_field_components(
    params: Dict[str, Any],
    date_identifiers: set
) -> Dict[str, Any]:
    """Normalize date field inputs to consistent API components.

    Rules applied:
    - If a base date field (e.g., "registered", "revisionDate") is present with a qualifier:
      - For ON: keep base field as is (e.g., registered = YYYY-MM-DD)
      - For BEFORE/AFTER: move the date to base+"1" and set qualifier
      - For BETWEEN: ensure base+"1" and base+"2" exist; leave qualifier as BETWEEN
    - If base+"1"/base+"2"/base+"Qualifier" are already present, leave as-is.
    - Applies only to identifiers found in date_identifiers.
    """
    try:
        updated = dict(params)
        # Find base date fields present in params
        candidate_bases = set()
        for key in list(updated.keys()):
            if key in date_identifiers:
                candidate_bases.add(key)
            elif (key.endswith("1") or key.endswith("2")) and key[:-1] in date_identifiers:
                candidate_bases.add(key[:-1])
            elif key.endswith("Qualifier") and key[:-9] in date_identifiers:
                candidate_bases.add(key[:-9])

        for base in candidate_bases:
            qualifier_key = f"{base}Qualifier"
            range_start_key = f"{base}1"
            range_end_key = f"{base}2"

            qualifier = updated.get(qualifier_key)
            base_value_present = base in updated
            has_start = range_start_key in updated
            has_end = range_end_key in updated

            if qualifier:
                q_upper = str(qualifier).upper()
                if q_upper in RELATIVE_DATE_QUALIFIERS:
                    # Relative windows: send only the qualifier, no dates
                    if base_value_present:
                        del updated[base]
                    if has_start:
                        del updated[range_start_key]
                    if has_end:
                        del updated[range_end_key]
                elif q_upper == "ON":
                    # Use base1 with YYYY-MM-DD; remove base and base2
                    if base_value_present and not has_start:
                        updated[range_start_key] = _to_date_only_str(updated[base])
                        del updated[base]
                    if has_start:
                        updated[range_start_key] = _to_date_only_str(updated[range_start_key])
                    if has_end:
                        del updated[range_end_key]
                elif q_upper in ("BEFORE", "AFTER"):
                    # Use base1 with YYYY-MM-DD; remove base and base2
                    if base_value_present and not has_start:
                        updated[range_start_key] = _to_date_only_str(updated[base])
                        del updated[base]
                    if has_start:
                        updated[range_start_key] = _to_date_only_str(updated[range_start_key])
                    if has_end:
                        del updated[range_end_key]
                elif q_upper == "BETWEEN":
                    # Use base1/base2 in ISO with T05:30:00.000Z; remove base
                    if base_value_present:
                        if not has_start:
                            updated[range_start_key] = updated[base]
                        if not has_end:
                            updated[range_end_key] = updated[base]
                        del updated[base]
                    updated[range_start_key] = _to_between_iso_str(updated.get(range_start_key))
                    updated[range_end_key] = _to_between_iso_str(updated.get(range_end_key))
                else:
                    # Unknown qualifier: leave structure untouched
                    pass
            else:
                # No qualifier; if we have base+1/base+2 only, leave; else leave base as-is
                pass

        return updated
    except Exception as e:
        logger.warning(f"normalize_date_field_components failed: {str(e)}")
        return params

def convert_values_to_ids(value: Any, field_metadata: Dict[str, Any], field_name: str) -> str:
    """Convert field values to their corresponding IDs."""
    if isinstance(value, str) and "," in value:
        # Handle comma-separated values
        values = [v.strip() for v in value.split(",")]
        ids = []
        for val in values:
            field_id = get_field_id(field_metadata, val)
            if field_id:
                ids.append(field_id)
            else:
                logger.warning(f"No ID found for value '{val}' in field '{field_name}', keeping original value")
                ids.append(val)
        return ",".join(ids)
    else:
        # Single value
        single_value = str(value)
        field_id = get_field_id(field_metadata, single_value)
        if field_id:
            logger.debug(f"Converted '{single_value}' to ID '{field_id}' for field '{field_name}'")
            return field_id
        else:
            logger.warning(f"No ID found for value '{single_value}' in field '{field_name}', keeping original value")
            return single_value

def format_text_values(value: Any) -> str:
    """Format values as text (for fields that don't need ID conversion)."""
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    elif isinstance(value, str) and "," in value:
        return value  # Already comma-separated
    else:
        return str(value)

def transform_parameters_for_api(params: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform validated and normalized parameters for the Aconex Search API with date handling.
    
    Args:
        params: The validated and normalized search parameters
        metadata: The metadata containing field definitions
        
    Returns:
        Dict containing parameters ready for the Aconex Search API
    """
    try:
        # Import the return fields and constants
        with open("return_fields.json") as f:
            return_fields = json.load(f)
        
        transformed_params = {
            "returnFields": return_fields,
            "userId": "1879053256",
            "orgId": "1"
        }
        
        # Get date field identifiers for special handling
        date_fields = []
        for field in metadata.get("searchSchema", {}).get("singleValueFields", []) + \
                   metadata.get("searchSchema", {}).get("multiValueFields", []):
            if field.get("searchable", False) and field.get("dataType") == "DATE":
                date_fields.append(field["identifier"])
        
        logger.debug(f"Date fields identified: {date_fields}")
        
        # Normalize date field components (e.g., revisionDate + Qualifier → revisionDate1 + Qualifier)
        params = normalize_date_field_components(params, set(date_fields))
        
        for field_name, value in params.items():
            logger.debug(f"Transforming field: {field_name} = {value}")
            
            # Handle date field qualifiers and range components directly
            if field_name.endswith("Qualifier"):
                base_field = field_name.replace("Qualifier", "")
                if base_field in date_fields:
                    # Date qualifiers go directly to API without mapping
                    transformed_params[field_name] = value
                    logger.debug(f"Added date qualifier directly: {field_name} = {value}")
                    continue
            
            # Handle date range components (field1, field2)
            if (field_name.endswith("1") or field_name.endswith("2")) and field_name[:-1] in date_fields:
                # Date range components go directly to API without mapping
                transformed_params[field_name] = value
                logger.debug(f"Added date range component directly: {field_name} = {value}")
                continue
            
            # Check if this is a simple date field
            if field_name in date_fields:
                # Simple date fields go directly to API without mapping
                transformed_params[field_name] = value
                logger.debug(f"Added date field directly: {field_name} = {value}")
                continue
            
            # Get the correct field name for API using the mapping (for non-date fields)
            api_field_name = get_modified_identifier(field_name)
            logger.debug(f"Mapped to API field: {field_name} -> {api_field_name}")
            
            # Find the field in metadata to get correct IDs if needed
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
            
            # Special handling: certain API fields must be arrays (e.g., vdrCodeValues)
            array_expected_fields = {
                VDR_CODE_VALUES,
                CATEGORY_VALUES,
                ATTRIBUTE_1_VALUES,
                ATTRIBUTE_2_VALUES,
                ATTRIBUTE_3_VALUES,
                ATTRIBUTE_4_VALUES,
                SELECT_LIST_1_VALUES,
                SELECT_LIST_2_VALUES,
                SELECT_LIST_3_VALUES,
                SELECT_LIST_4_VALUES,
                SELECT_LIST_5_VALUES,
                SELECT_LIST_6_VALUES,
                SELECT_LIST_7_VALUES,
                SELECT_LIST_8_VALUES,
                SELECT_LIST_9_VALUES,
                SELECT_LIST_10_VALUES,
            }

            if api_field_name in array_expected_fields:
                # Coerce to array robustly (handle list, comma-strings, and bracketed list strings)
                transformed_value = []
                if isinstance(value, list):
                    transformed_value = [str(v).strip() for v in value if v is not None]
                elif isinstance(value, str):
                    s = value.strip()
                    if s.startswith("[") and s.endswith("]"):
                        parsed = None
                        try:
                            # Try JSON first
                            parsed = json.loads(s)
                        except Exception:
                            try:
                                # Fallback to Python literal (e.g., "['x','y']")
                                parsed = ast.literal_eval(s)
                            except Exception:
                                parsed = None
                        if isinstance(parsed, list):
                            transformed_value = [str(v).strip() for v in parsed if v is not None]
                        else:
                            transformed_value = [s]
                    else:
                        parts = [p.strip() for p in s.split(",")] if "," in s else [s]
                        transformed_value = [p for p in parts if p]
                else:
                    transformed_value = [str(value)]
            else:
                # Handle value transformation - some fields need IDs, others use text values
                if field_metadata and field_metadata.get("selectValues"):
                    # Fields that typically need ID conversion
                    id_required_fields = {"doctype", "statusid", "status", "documenttype"}
                    
                    if field_name.lower() in id_required_fields:
                        # Ensure list inputs are converted to comma-separated before ID mapping
                        if isinstance(value, list):
                            value = ",".join(str(v) for v in value)
                        # Convert values to IDs for these fields
                        transformed_value = convert_values_to_ids(value, field_metadata, field_name)
                    else:
                        # Keep text values for other fields (like discipline, category, etc.)
                        transformed_value = format_text_values(value)
                else:
                    # No selectValues, use the value as-is (includes date fields handled above)
                    transformed_value = format_text_values(value)
            
            if transformed_value:  # Only add if we have a valid value
                transformed_params[api_field_name] = transformed_value
                logger.debug(f"Added to transformed params: {api_field_name} = {transformed_value}")
        
        # Debug: Show final transformed parameters
        logger.info("Final transformed parameters for API:")
        for key, value in transformed_params.items():
            if key not in ["returnFields", "userId", "orgId"]:  # Skip standard fields
                logger.info(f"  {key}: {value}")
        
        return transformed_params
        
    except Exception as e:
        logger.error(f"Error transforming parameters: {str(e)}")
        raise Exception(f"Failed to transform parameters for API: {str(e)}")

def preprocess_query_with_dates(query: str) -> str:
    """
    Preprocess query to convert relative dates (today, yesterday, tomorrow) to actual dates.
    This ensures the LLM receives concrete dates instead of relative terms.
    
    Args:
        query: Original user query
        
    Returns:
        Query with relative dates converted to actual dates
    """
    try:
        # Get current date
        current_date = datetime.now()
        current_date_str = current_date.strftime("%Y-%m-%d")
        yesterday_str = (current_date - timedelta(days=1)).strftime("%Y-%m-%d")
        tomorrow_str = (current_date + timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Create processed query
        processed_query = query.lower()
        
        # Convert relative dates to actual dates (comprehensive list)
        replacements = [
            ('today', current_date_str),
            ('tdy', current_date_str),
            ('now', current_date_str),
            ('this day', current_date_str),
            ('yesterday', yesterday_str),
            ('yest', yesterday_str),
            ('tomorrow', tomorrow_str),
            ('tmr', tomorrow_str),
            ('tom', tomorrow_str)
        ]
        
        for relative_term, actual_date in replacements:
            if relative_term in processed_query:
                # Replace the term but preserve the case context of the original query
                original_index = query.lower().find(relative_term)
                if original_index != -1:
                    # Preserve original capitalization pattern
                    query = query[:original_index] + actual_date + query[original_index + len(relative_term):]
                    logger.info(f"Converted '{relative_term}' to '{actual_date}' in query")
        
        return query
        
    except Exception as e:
        logger.warning(f"Error preprocessing dates in query: {str(e)}")
        return query  # Return original query if preprocessing fails

def build_preamble(metadata_fields: dict, query: str) -> str:
    """Build the enhanced preamble for the LLM with semantic mapping and date handling."""
    try:
        # Build semantic mappings from metadata
        semantic_data = semantic_mapping.build_semantic_mappings(metadata_fields)
        
        # Preprocess query and detect intent (use the processed query, not original)
        cleaned = SearchPromptUtils.preprocess_query(query)
        has_date_intent = SearchPromptUtils.has_date_intent(cleaned)
        date_concepts = SearchPromptUtils.extract_date_concepts_from_query(cleaned)
        
        # Re-check date intent after preprocessing if not detected initially
        if not has_date_intent and ('modified' in query.lower() or 'updated' in query.lower() or 'created' in query.lower()):
            # Check if query now contains actual dates after preprocessing
            if re.search(r'\d{4}-\d{2}-\d{2}', query):
                has_date_intent = True
                logger.debug("Date intent detected after preprocessing due to actual date in query")
        
        # Extract essential non-date fields
        essential_fields = []
        for field in metadata_fields.get("searchSchema", {}).get("singleValueFields", []):
            if field.get("searchable", False) and field.get("dataType") != "DATE":
                field_info = {
                    "fieldName": field.get("fieldName"),
                    "identifier": field.get("identifier"),
                    "dataType": field.get("dataType"),
                    "selectValues": [v["value"] for v in field.get("selectValues", [])] if field.get("selectValues") else None
                }
                essential_fields.append(field_info)

        for field in metadata_fields.get("searchSchema", {}).get("multiValueFields", []):
            if field.get("searchable", False) and field.get("dataType") != "DATE":
                field_info = {
                    "fieldName": field.get("fieldName"),
                    "identifier": field.get("identifier"),
                    "dataType": field.get("dataType"),
                    "selectValues": [v["value"] for v in field.get("selectValues", [])] if field.get("selectValues") else None
                }
                essential_fields.append(field_info)

        # Get date fields if needed
        date_fields = []
        if has_date_intent:
            date_fields = SearchPromptUtils.get_date_fields(metadata_fields)

        # Build semantic mapping examples for preamble
        date_mapping_examples = []
        date_mappings = semantic_mapping.get_date_field_mappings(semantic_data)
        if date_mappings:
            for concept, identifier in list(date_mappings.items())[:5]:  # Show top 5 examples
                field_name = semantic_data.get('reverse_mappings', {}).get(identifier, identifier)
                date_mapping_examples.append(f'  - "{concept}" → use "{identifier}" identifier ({field_name} field)')

        # Build preamble with semantic enhancements
        preamble_parts = []
        
        preamble_parts.append(f"""You are an AI assistant specialized in document search using structured metadata filters. 

📅 TODAY'S DATE: {datetime.now().strftime('%Y-%m-%d')}

CRITICAL: Return ONLY a simple JSON object with field identifiers and values.

EXAMPLES:
- "fetch 2D models" → {{"doctype": "2D Model"}}
- "electrical documents" → {{"discipline": "EL - Electrical"}}
- "documents modified today" → {{"registeredQualifier": "TODAY"}}
- "documents on specific date" → {{"registeredQualifier": "ON", "registered1": "YYYY-MM-DD"}}

RULES:
- Use field "identifier" names from metadata (e.g., "doctype", "discipline", "registered")
- NO null values, NO extra structure, NO operators
- For "modified/updated" queries: use registered1 + registeredQualifier
- For "on specific date" queries: use registered only
- Match values to available options in metadata""")

        # Add concept translation and detailed date qualifier/format rules
        preamble_parts.append(f"""
CONCEPT TRANSLATION:
- "modified"/"updated"/"changed" → use "registered" identifier (Date Modified field)
- "created"/"new" → use appropriate creation date identifier
- "approved"/"authorized" → use appropriate approval date identifier

DATE QUALIFIER AND FORMAT RULES:
- ON specific date: {{"<dateId>Qualifier": "ON", "<dateId>1": "YYYY-MM-DD"}}
- BEFORE: {{"<dateId>Qualifier": "BEFORE", "<dateId>1": "YYYY-MM-DD"}}
- AFTER: {{"<dateId>Qualifier": "AFTER", "<dateId>1": "YYYY-MM-DD"}}
- BETWEEN: {{"<dateId>Qualifier": "BETWEEN", "<dateId>1": "YYYY-MM-DDT05:30:00.000Z", "<dateId>2": "YYYY-MM-DDT05:30:00.000Z"}}
- RELATIVE WINDOWS (no dates): TODAY, TOMORROW, YESTERDAY, LAST7DAYS, LAST30DAYS, LAST90DAYS, NEXT7DAYS, NEXT30DAYS, NEXT90DAYS

OUTPUT FORMAT RULES:
- Use only metadata identifier names (e.g., "doctype", "discipline", "registered").
- For fields ending with "Values" (e.g., vdrCodeValues, categoryValues, attribute1Values...), return arrays of strings.
- For other fields, use a single string; if multiple are needed, use a comma-separated string.
- Return multiple values only if explicitly requested or genuinely ambiguous.
- Return just the JSON object: no comments, markdown, or extra text.

EXAMPLE RESPONSE FORMATS:
- Basic: {{"doctype": "2D Model", "discipline": "EL - Electrical,Electrical,electrical"}}
- Date ON: {{"revisiondateQualifier": "ON", "revisiondate1": "2025-08-27"}}
- Date BEFORE: {{"revisiondateQualifier": "BEFORE", "revisiondate1": "2025-08-30"}}
- Date AFTER: {{"revisiondateQualifier": "AFTER", "revisiondate1": "2025-08-25"}}
- Date BETWEEN: {{"revisiondateQualifier": "BETWEEN", "revisiondate1": "2025-08-27T05:30:00.000Z", "revisiondate2": "2025-08-30T05:30:00.000Z"}}
- Date RELATIVE: {{"revisiondateQualifier": "LAST7DAYS"}}
""")

        # Add simplified date mapping if needed
        if date_mapping_examples:
            preamble_parts.append(f"""
DATE FIELDS:
{chr(10).join(date_mapping_examples)}  

DATE RULES:
- "modified today" = {{"registered1": "{datetime.now().strftime('%Y-%m-%d')}", "registeredQualifier": "ON"}}
- "on specific date" = {{"registered": "YYYY-MM-DD"}}""")







        # Add essential fields only
        preamble_parts.append(f"""
Available Fields:
{json.dumps(essential_fields, indent=2)}""")

        # Add available date fields if relevant
        if date_fields:
            preamble_parts.append(f"""
Available Date Fields:
{json.dumps(date_fields, indent=2)}""")



        # Add user query
        preamble_parts.append(f"""
User Query: "{query}"

Return ONLY JSON, no explanation.""")

        # Join all parts
        preamble = "\n".join(preamble_parts)
        
        logger.debug("\nBuilt Enhanced Preamble with Semantic Mapping")
        logger.debug(f"Date intent detected: {has_date_intent}")
        logger.debug(f"Date concepts found: {date_concepts}")
        logger.debug(f"Semantic mappings available: {len(date_mappings)} date mappings")
        
        return preamble
        
    except Exception as e:
        logger.error(f"Error building enhanced preamble: {str(e)}")
        # Fallback to basic preamble if semantic enhancement fails
        return build_basic_preamble(metadata_fields, query)

def build_basic_preamble(metadata_fields: dict, query: str) -> str:
    """Fallback basic preamble builder if semantic enhancement fails."""
    # This is essentially the old build_preamble function as a fallback
    essential_fields = []
    for field in metadata_fields.get("searchSchema", {}).get("singleValueFields", []):
        if field.get("searchable", False) and field.get("dataType") != "DATE":
            field_info = {
                "fieldName": field.get("fieldName"),
                "identifier": field.get("identifier"),
                "dataType": field.get("dataType"),
                "selectValues": [v["value"] for v in field.get("selectValues", [])] if field.get("selectValues") else None
            }
            essential_fields.append(field_info)

    for field in metadata_fields.get("searchSchema", {}).get("multiValueFields", []):
        if field.get("searchable", False) and field.get("dataType") != "DATE":
            field_info = {
                "fieldName": field.get("fieldName"),
                "identifier": field.get("identifier"),
                "dataType": field.get("dataType"),
                "selectValues": [v["value"] for v in field.get("selectValues", [])] if field.get("selectValues") else None
            }
            essential_fields.append(field_info)

    return f"""
You are an AI assistant specialized in document search using structured metadata filters.

CRITICAL FIELD NAMING RULE:
- You MUST use the "identifier" field name, NOT the "fieldName" when creating your JSON response.

Available Fields:
{json.dumps(essential_fields, indent=2)}

User Query: "{query}"
"""

@app.get("/environment")
def get_environment_info():
    """Get information about the current environment configuration."""
    return {
        "status": "success",
        "environment": env_config.get_environment_info(),
        "available_environments": env_config.list_environments()
    }

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
        
        # 2. Preprocess query with date handling
        logger.info("\n2. Preprocessing Query...")
        processed_query = preprocess_query_with_dates(req.query.strip())
        logger.info(f"Original Query: {req.query}")
        logger.info(f"Processed Query: {processed_query}")
        
        # 3. Build preamble
        logger.info("\n3. Building Preamble...")
        preamble = build_preamble(metadata, processed_query)
        logger.info("Preamble built successfully")
        
        # 4. Call LLM with improved retry logic
        logger.info("\n4. Calling LLM...")
        max_retries = 3
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
                    time.sleep(2 ** retry_count)  # Exponential backoff: 2s, 4s, 8s
            except Exception as e:
                error_str = str(e)
                logger.error(f"LLM call failed: {error_str}")
                
                # Check if this is a timeout or service error
                is_timeout = "ReadTimeoutError" in error_str or "timeout" in error_str.lower()
                is_service_error = "500" in error_str or "503" in error_str or "502" in error_str
                
                retry_count += 1
                if retry_count < max_retries and (is_timeout or is_service_error):
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.warning(f"Retrying after {wait_time}s... (Attempt {retry_count + 1}/{max_retries})")
                    logger.warning(f"Retry reason: {'Timeout' if is_timeout else 'Service error'}")
                    time.sleep(wait_time)
                else:
                    # Don't retry for other types of errors, or if max retries reached
                    if retry_count >= max_retries:
                        logger.error("Max retries reached - Oracle Cloud LLM service may be overloaded")
                        raise HTTPException(
                            status_code=503,
                            detail={
                                "message": "LLM service temporarily unavailable",
                                "error": "Oracle Cloud service timeout/overload",
                                "retry_after": "Please try again in a few minutes"
                            }
                        )
                    else:
                        raise
        
        # 5. Parse LLM response
        logger.info("\n5. Parsing LLM Response...")
        try:
            search_params = json.loads(llm_response)
            logger.debug(f"Raw Search Parameters: {json.dumps(search_params, indent=2)}")
            
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
        
        # 6. Validate LLM response
        logger.info("\n6. Validating Search Parameters...")
        validation_result = validation.validate_llm_response(search_params, metadata, verbose=True)
        
        if not validation_result["valid"]:
            logger.warning("Validation failed for search parameters")
            logger.debug(f"Invalid fields: {validation_result['invalid_fields']}")
            logger.debug(f"Invalid values: {validation_result['invalid_values']}")
            
            # Handle validation errors with suggestions
            error_details = handle_validation_errors(validation_result, metadata)
            
            raise HTTPException(
                status_code=422,
                detail={
                    "classification": "validation_error",
                    "message": "Search parameters failed validation",
                    "invalid_fields": validation_result["invalid_fields"],
                    "invalid_values": validation_result["invalid_values"],
                    "suggestions": error_details.get("suggestions", []),
                    "closest_matches": error_details.get("closest_matches", {})
                }
            )
        
        # Use cleaned output from validation
        search_params = validation_result["cleaned_output"]
        logger.debug(f"Validated Search Parameters: {json.dumps(search_params, indent=2)}")
        
        # 7. Normalize search parameters
        logger.info("\n7. Normalizing Search Parameters...")
        normalized_params = normalization.normalize_fields(search_params, metadata, query=processed_query, verbose=True)
        logger.debug(f"Normalized Search Parameters: {json.dumps(normalized_params, indent=2)}")
        
        # 8. Transform parameters for API
        logger.info("\n8. Transforming Parameters for API...")
        transformed_params = transform_parameters_for_api(normalized_params, metadata)
        logger.debug(f"Transformed Search Parameters: {json.dumps(transformed_params, indent=2)}")
        
        # 9. Call search API
        logger.info("\n9. Calling Search API...")
        logger.debug(f"Transformed params for API call: {json.dumps(transformed_params, indent=2)}")
        response = call_search_api(transformed_params, headers, metadata)
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
        llm_chat_request.max_tokens = 200  # Reduced for faster response
        llm_chat_request.is_force_single_step = True

        chat_detail = ChatDetails()
        chat_detail.serving_mode = OnDemandServingMode(model_id=LLM_MODEL)
        chat_detail.compartment_id = COMPARTMENT_ID
        chat_detail.chat_request = llm_chat_request

        logger.info("\nSending request to LLM...")
        chat_response = llm_client.chat(chat_detail)
        response_text = chat_response.data.chat_response.text
        
        logger.debug(f"\nLLM Response Summary: {len(response_text)} chars")
        logger.debug(f"\nLLM Response Summary: {response_text} ")

        if not response_text:
            logger.warning("Warning: LLM returned empty response")
            return "{}"
            
        return response_text
        
    except Exception as e:
        logger.error(f"Error in call_llm: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
