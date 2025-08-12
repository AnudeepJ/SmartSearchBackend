#!/usr/bin/env python3
"""
Semantic Mapping Module for Smart Search

This module handles the semantic relationship between user natural language 
and technical field identifiers. It builds mappings from metadata and provides
translation functions for user concepts.

Key functionality:
- Extract fieldName → identifier mappings from metadata
- Map user concepts (modified, created, approved) to field identifiers  
- Handle document type semantic mappings
- Provide date field semantic translations
"""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class SemanticMappingError(Exception):
    """Raised when semantic field mapping fails"""
    pass

def build_semantic_mappings(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract comprehensive fieldName → identifier mappings from metadata.
    
    Args:
        metadata: Complete metadata structure from API
        
    Returns:
        Dict containing various semantic mappings:
        {
            'field_mappings': {fieldName.lower(): identifier},
            'date_mappings': {concept: identifier},
            'reverse_mappings': {identifier: fieldName},
            'date_fields': [list of date field identifiers]
        }
    """
    try:
        semantic_data = {
            'field_mappings': {},      # "date modified" → "registered"
            'date_mappings': {},       # "modified" → "registered"  
            'reverse_mappings': {},    # "registered" → "Date Modified"
            'date_fields': [],         # ["registered", "creationdate", ...]
            'searchable_fields': {}    # only searchable fields
        }
        
        # Process all fields from metadata
        all_fields = []
        all_fields.extend(metadata.get('searchSchema', {}).get('singleValueFields', []))
        all_fields.extend(metadata.get('searchSchema', {}).get('multiValueFields', []))
        
        logger.debug(f"Processing {len(all_fields)} total fields for semantic mapping")
        
        for field in all_fields:
            field_name = field.get('fieldName', '').strip()
            identifier = field.get('identifier', '').strip()
            data_type = field.get('dataType', '')
            searchable = field.get('searchable', False)
            
            if not field_name or not identifier:
                continue
                
            # Build core mappings
            field_name_key = field_name.lower()
            semantic_data['field_mappings'][field_name_key] = identifier
            semantic_data['reverse_mappings'][identifier] = field_name
            
            # Track searchable fields only
            if searchable:
                semantic_data['searchable_fields'][field_name_key] = {
                    'identifier': identifier,
                    'dataType': data_type,
                    'fieldName': field_name
                }
            
            # Special handling for date fields
            if data_type == 'DATE' and searchable:
                semantic_data['date_fields'].append(identifier)
                
                # Build semantic date mappings based on field names
                field_name_lower = field_name.lower()
                
                # Date Modified / registered mapping (confirmed by user)
                if 'modified' in field_name_lower or identifier == 'registered':
                    semantic_data['date_mappings']['modified'] = identifier
                    semantic_data['date_mappings']['updated'] = identifier
                    semantic_data['date_mappings']['changed'] = identifier
                    semantic_data['date_mappings']['edited'] = identifier
                    semantic_data['date_mappings']['revised'] = identifier
                
                # Creation date mappings
                if 'creat' in field_name_lower or 'creat' in identifier.lower():
                    semantic_data['date_mappings']['created'] = identifier
                    semantic_data['date_mappings']['added'] = identifier
                    semantic_data['date_mappings']['new'] = identifier
                    semantic_data['date_mappings']['uploaded'] = identifier
                    
                # Approval date mappings  
                if 'approv' in field_name_lower or 'approv' in identifier.lower():
                    semantic_data['date_mappings']['approved'] = identifier
                    semantic_data['date_mappings']['authorized'] = identifier
                    semantic_data['date_mappings']['signed off'] = identifier
                    semantic_data['date_mappings']['accepted'] = identifier
                    
                # Test date mappings
                if 'test' in field_name_lower:
                    semantic_data['date_mappings']['tested'] = identifier
                    semantic_data['date_mappings']['testing'] = identifier
                    
                # Effective date mappings
                if 'effective' in field_name_lower:
                    semantic_data['date_mappings']['effective'] = identifier
                    semantic_data['date_mappings']['valid from'] = identifier
        
        logger.info(f"Built semantic mappings: {len(semantic_data['field_mappings'])} field mappings, "
                   f"{len(semantic_data['date_mappings'])} date concept mappings, "
                   f"{len(semantic_data['date_fields'])} date fields")
        
        return semantic_data
        
    except Exception as e:
        logger.error(f"Error building semantic mappings: {str(e)}")
        raise SemanticMappingError(f"Failed to build semantic mappings: {str(e)}")

def get_semantic_field(user_concept: str, semantic_data: Dict[str, Any]) -> Optional[str]:
    """
    Map user concepts to field identifiers using semantic understanding.
    
    Args:
        user_concept: User's natural language concept (e.g., "modified", "created")
        semantic_data: Semantic mappings from build_semantic_mappings()
        
    Returns:
        Field identifier if found, None otherwise
    """
    try:
        concept_lower = user_concept.lower().strip()
        
        # First check direct date concept mappings
        if concept_lower in semantic_data.get('date_mappings', {}):
            identifier = semantic_data['date_mappings'][concept_lower]
            logger.debug(f"Mapped user concept '{user_concept}' to date field '{identifier}'")
            return identifier
            
        # Then check general field mappings
        if concept_lower in semantic_data.get('field_mappings', {}):
            identifier = semantic_data['field_mappings'][concept_lower]
            logger.debug(f"Mapped user concept '{user_concept}' to field '{identifier}'")
            return identifier
            
        # Check partial matches for field names
        for field_name, identifier in semantic_data.get('field_mappings', {}).items():
            if concept_lower in field_name or field_name in concept_lower:
                logger.debug(f"Partial match: '{user_concept}' matched field '{field_name}' → '{identifier}'")
                return identifier
        
        logger.debug(f"No semantic mapping found for concept: '{user_concept}'")
        return None
        
    except Exception as e:
        logger.error(f"Error in semantic field mapping for '{user_concept}': {str(e)}")
        return None

def get_date_field_mappings(semantic_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Get all date field semantic mappings.
    
    Args:
        semantic_data: Semantic mappings from build_semantic_mappings()
        
    Returns:
        Dictionary of user concepts → date field identifiers
    """
    return semantic_data.get('date_mappings', {})

def get_primary_date_field(semantic_data: Dict[str, Any]) -> str:
    """
    Get the primary date field for general date queries.
    Prioritizes 'registered' (Date Modified) as it's most commonly used.
    
    Args:
        semantic_data: Semantic mappings from build_semantic_mappings()
        
    Returns:
        Primary date field identifier (defaults to 'registered')
    """
    date_fields = semantic_data.get('date_fields', [])
    
    # Prioritize 'registered' as confirmed by user
    if 'registered' in date_fields:
        return 'registered'
        
    # Fallback to creation date
    for field in date_fields:
        if 'creat' in field.lower():
            return field
            
    # Last resort: return first available date field
    if date_fields:
        return date_fields[0]
        
    # Default fallback
    return 'registered'

def suggest_semantic_corrections(user_input: str, semantic_data: Dict[str, Any]) -> List[str]:
    """
    Suggest corrections for unrecognized user concepts.
    
    Args:
        user_input: User's input that couldn't be mapped
        semantic_data: Semantic mappings from build_semantic_mappings()
        
    Returns:
        List of suggested field names/concepts
    """
    suggestions = []
    user_lower = user_input.lower()
    
    # Check for close matches in field names
    for field_name in semantic_data.get('field_mappings', {}).keys():
        # Simple fuzzy matching - could be enhanced with more sophisticated algorithms
        if any(word in field_name for word in user_lower.split()):
            original_name = semantic_data['reverse_mappings'].get(
                semantic_data['field_mappings'][field_name], field_name
            )
            suggestions.append(original_name)
    
    # Add common date concepts if it seems like a date query
    date_words = ['time', 'when', 'date', 'day', 'week', 'month', 'year']
    if any(word in user_lower for word in date_words):
        suggestions.extend(['modified', 'created', 'approved'])
    
    return list(set(suggestions))[:5]  # Return top 5 unique suggestions

def get_document_type_mappings() -> Dict[str, List[str]]:
    """
    Get semantic mappings for document types.
    These map user concepts to possible doctype values.
    
    Returns:
        Dictionary of concepts → possible doctype values
    """
    return {
        'drawings': ['2D Model', '3D Model', 'Drawing', 'Plan', 'Blueprint'],
        'plans': ['Plan', '2D Model', 'Drawing', 'Blueprint'],
        'blueprints': ['Blueprint', 'Drawing', '2D Model', 'Plan'],
        'schematics': ['Schematic', 'Drawing', '2D Model'],
        'models': ['2D Model', '3D Model', '3D Combined Model', '2D Combined Model'],
        '2d': ['2D Model', '2D Combined Model'],
        '3d': ['3D Model', '3D Combined Model'],
        'specifications': ['Specification', 'Technical Data', 'Requirements'],
        'specs': ['Specification', 'Technical Data'],
        'requirements': ['Requirements', 'Specification'],
        'documents': ['Document'],  # Generic fallback
        'reports': ['Report', 'Document'],
        'procedures': ['Procedure', 'Document'],
        'manuals': ['Manual', 'Document']
    }

def debug_semantic_mappings(semantic_data: Dict[str, Any]) -> str:
    """
    Generate debug information about current semantic mappings.
    Useful for troubleshooting and understanding the mapping state.
    
    Args:
        semantic_data: Semantic mappings from build_semantic_mappings()
        
    Returns:
        Formatted debug string
    """
    debug_info = []
    debug_info.append("=== SEMANTIC MAPPING DEBUG INFO ===\n")
    
    debug_info.append(f"📊 SUMMARY:")
    debug_info.append(f"  - Total field mappings: {len(semantic_data.get('field_mappings', {}))}")
    debug_info.append(f"  - Date concept mappings: {len(semantic_data.get('date_mappings', {}))}")
    debug_info.append(f"  - Searchable fields: {len(semantic_data.get('searchable_fields', {}))}")
    debug_info.append(f"  - Date fields: {len(semantic_data.get('date_fields', []))}")
    debug_info.append("")
    
    debug_info.append("📅 DATE CONCEPT MAPPINGS:")
    for concept, identifier in semantic_data.get('date_mappings', {}).items():
        field_name = semantic_data.get('reverse_mappings', {}).get(identifier, identifier)
        debug_info.append(f"  '{concept}' → '{identifier}' ({field_name})")
    debug_info.append("")
    
    debug_info.append("🔍 DATE FIELDS AVAILABLE:")
    for field_id in semantic_data.get('date_fields', []):
        field_name = semantic_data.get('reverse_mappings', {}).get(field_id, field_id)
        debug_info.append(f"  '{field_id}' ({field_name})")
    
    return "\n".join(debug_info) 