"""
Utility functions for LLM calls and XML extraction following the Anthropic cookbook pattern
but supporting both OpenAI and Azure OpenAI as backends.
"""

import openai
import os
import re
import json
from typing import Dict, Any, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)  # Force reload environment variables


def safe_json_loads(field_value, default=None):
    """Safely parse JSON with error handling"""
    if not field_value:
        return default if default is not None else []
    try:
        return json.loads(field_value)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Problematic value: {repr(field_value)}")
        return default if default is not None else []


# Initialize LLM client based on provider configuration
def _initialize_llm_client():
    """Initialize the appropriate OpenAI client based on configuration"""
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    
    if provider == 'azure':
        # Azure OpenAI configuration
        api_key = os.getenv('AZURE_OPENAI_API_KEY')
        endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        
        if not api_key or not endpoint:
            raise ValueError(
                "Azure OpenAI configuration incomplete. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT"
            )
        
        print(f"🔧 Initializing Azure OpenAI client - Endpoint: {endpoint}")
        return openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )
    else:
        # Standard OpenAI configuration
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable")
        
        print(f"🔧 Initializing OpenAI client")
        return openai.OpenAI(api_key=api_key)

# Initialize the client
client = _initialize_llm_client()

def get_model_name() -> str:
    """Get the model name based on provider configuration"""
    provider = os.getenv('LLM_PROVIDER', 'openai').lower()
    
    if provider == 'azure':
        # For Azure, use the deployment name
        deployment_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4.1')
        return deployment_name
    else:
        # For OpenAI, use the model name directly
        model_name = os.getenv('LLM_MODEL', 'gpt-4.1')
        return model_name


def llm_call(messages: list, model: str = None, temperature: float = 0.1) -> str:
    """
    Make a call to OpenAI's API (or Azure OpenAI) with consistent error handling.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        model: Model/deployment name to use (if None, uses configured default)
        temperature: Sampling temperature
        
    Returns:
        str: The response content from the LLM
        
    Raises:
        Exception: If the API call fails (to be handled by Huey retries)
    """
    try:
        # Use configured model if none specified
        if model is None:
            model = get_model_name()
        
        provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        provider = os.getenv('LLM_PROVIDER', 'openai').lower()
        provider_name = "Azure OpenAI" if provider == 'azure' else "OpenAI"
        # Let Huey handle retries - just re-raise the exception
        raise Exception(f"{provider_name} API call failed: {str(e)}")


def extract_xml(text: str, tag: str) -> Optional[str]:
    """
    Extract content from XML tags in the response text.
    
    Args:
        text: The text containing XML tags
        tag: The XML tag name to extract (without < >)
        
    Returns:
        str: The content within the XML tags, or None if not found
        
    Raises:
        Exception: If XML parsing fails
    """
    try:
        # Try to find the XML content using regex first (more robust)
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            content = match.group(1).strip()
            return content
        
        # If regex fails, try XML parsing
        try:
            # Wrap in root element if not already wrapped
            if not text.strip().startswith('<'):
                wrapped_text = f"<root>{text}</root>"
            else:
                wrapped_text = text
                
            root = ET.fromstring(wrapped_text)
            element = root.find(f".//{tag}")
            if element is not None:
                return element.text.strip() if element.text else ""
                
        except ET.ParseError:
            pass
            
        # Return None if tag not found
        return None
        
    except Exception as e:
        raise Exception(f"XML extraction failed for tag '{tag}': {str(e)}")


def parse_json_from_xml(xml_content: str) -> Dict[str, Any]:
    """
    Parse JSON content from XML extracted text.
    
    Args:
        xml_content: The content extracted from XML tags
        
    Returns:
        dict: Parsed JSON as dictionary
        
    Raises:
        Exception: If JSON parsing fails
    """
    try:
        # Clean up the content - remove any extra whitespace or formatting
        cleaned_content = xml_content.strip()
        
        # Try to parse as JSON
        return json.loads(cleaned_content)
        
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON from XML content: {str(e)}\nContent: {xml_content[:200]}...")
    except Exception as e:
        raise Exception(f"Unexpected error parsing JSON: {str(e)}")


def format_xml_response(content: str, tag: str) -> str:
    """
    Format content within XML tags for consistent responses.
    
    Args:
        content: The content to wrap
        tag: The XML tag name
        
    Returns:
        str: Content wrapped in XML tags
    """
    return f"<{tag}>\n{content}\n</{tag}>"


def validate_risk_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean up risk analysis results.
    
    Args:
        analysis: Risk analysis dictionary
        
    Returns:
        dict: Validated and cleaned analysis
        
    Raises:
        Exception: If validation fails
    """
    required_fields = [
        'primary_risk_category', 'severity_level', 'confidence_score', 
        'impact_score', 'summary', 'description'
    ]
    
    try:
        # Check required fields
        for field in required_fields:
            if field not in analysis:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate severity level
        valid_severities = ['Critical', 'High', 'Medium', 'Low']
        if analysis['severity_level'] not in valid_severities:
            raise ValueError(f"Invalid severity level: {analysis['severity_level']}")
        
        # Validate urgency level
        valid_urgency_levels = ['Critical', 'High', 'Medium', 'Low']
        if 'urgency_level' in analysis and analysis['urgency_level'] not in valid_urgency_levels:
            print(f"⚠️ Warning: Invalid urgency level '{analysis['urgency_level']}', defaulting to 'Medium'")
            analysis['urgency_level'] = 'Medium'
        
        # Validate temporal impact
        valid_temporal_impacts = ['Immediate', 'Short-term', 'Medium-term', 'Long-term']
        if 'temporal_impact' in analysis and analysis['temporal_impact'] not in valid_temporal_impacts:
            print(f"⚠️ Warning: Invalid temporal impact '{analysis['temporal_impact']}', defaulting to 'Medium-term'")
            analysis['temporal_impact'] = 'Medium-term'
        
        # Validate primary risk category
        valid_categories = [
            'market_risk', 'credit_risk', 'operational_risk', 'liquidity_risk',
            'cybersecurity_risk', 'regulatory_risk', 'systemic_risk', 'reputational_risk'
        ]
        primary_category = analysis['primary_risk_category']
        
        # If multiple categories are provided (separated by |), take the first one and move others to secondary
        if '|' in primary_category or ' and ' in primary_category:
            # Split on various separators
            parts = re.split(r'\s*[|&,]\s*|\s+and\s+', primary_category.lower())
            primary_category = parts[0].strip().replace(' ', '_')
            
            # Move additional categories to secondary if not already there
            if 'secondary_risk_categories' not in analysis:
                analysis['secondary_risk_categories'] = []
            
            for part in parts[1:]:
                clean_part = part.strip().replace(' ', '_')
                if clean_part in valid_categories and clean_part not in analysis['secondary_risk_categories']:
                    analysis['secondary_risk_categories'].append(clean_part)
        
        # Normalize the primary category
        primary_category = primary_category.lower().replace(' ', '_').replace('-', '_')
        
        if primary_category not in valid_categories:
            print(f"⚠️ Warning: Invalid primary risk category '{primary_category}', defaulting to 'market_risk'")
            analysis['primary_risk_category'] = 'market_risk'
        else:
            analysis['primary_risk_category'] = primary_category
        
        # Validate secondary categories
        if 'secondary_risk_categories' in analysis:
            valid_secondary = []
            for cat in analysis['secondary_risk_categories']:
                normalized_cat = cat.lower().replace(' ', '_').replace('-', '_')
                if normalized_cat in valid_categories and normalized_cat != analysis['primary_risk_category']:
                    valid_secondary.append(normalized_cat)
            analysis['secondary_risk_categories'] = valid_secondary
        
        # Validate confidence score
        if 'confidence_score' in analysis:
            try:
                confidence = float(analysis['confidence_score'])
                if confidence < 0 or confidence > 100:
                    print(f"⚠️ Warning: Confidence score {confidence} out of range, clamping to 0-100")
                    confidence = max(0, min(100, confidence))
                analysis['confidence_score'] = int(confidence)
            except (ValueError, TypeError):
                print(f"⚠️ Warning: Invalid confidence score '{analysis['confidence_score']}', defaulting to 50")
                analysis['confidence_score'] = 50
        
        # Validate impact score
        if 'impact_score' in analysis:
            try:
                impact = float(analysis['impact_score'])
                if impact < 0 or impact > 100:
                    print(f"⚠️ Warning: Impact score {impact} out of range, clamping to 0-100")
                    impact = max(0, min(100, impact))
                analysis['impact_score'] = int(impact)
            except (ValueError, TypeError):
                print(f"⚠️ Warning: Invalid impact score '{analysis['impact_score']}', defaulting to 50")
                analysis['impact_score'] = 50
        
        # Validate sentiment score
        if 'sentiment_score' in analysis:
            try:
                sentiment = float(analysis['sentiment_score'])
                if sentiment < -1 or sentiment > 1:
                    print(f"⚠️ Warning: Sentiment score {sentiment} out of range, clamping to -1 to 1")
                    sentiment = max(-1, min(1, sentiment))
                analysis['sentiment_score'] = round(sentiment, 3)
            except (ValueError, TypeError):
                print(f"⚠️ Warning: Invalid sentiment score '{analysis['sentiment_score']}', defaulting to 0")
                analysis['sentiment_score'] = 0.0
        
        # Ensure boolean fields are actually boolean
        boolean_fields = ['is_market_moving', 'is_breaking_news', 'is_regulatory', 'requires_action']
        for field in boolean_fields:
            if field in analysis:
                if isinstance(analysis[field], str):
                    analysis[field] = analysis[field].lower() in ['true', 'yes', '1']
                else:
                    analysis[field] = bool(analysis[field])
        
        # Ensure list fields are actually lists
        list_fields = ['secondary_risk_categories', 'risk_subcategories', 'geographic_regions', 
                      'industry_sectors', 'countries', 'affected_markets', 'keywords', 'entities']
        for field in list_fields:
            if field in analysis and not isinstance(analysis[field], list):
                if isinstance(analysis[field], str):
                    # Try to parse as comma-separated values
                    analysis[field] = [item.strip() for item in analysis[field].split(',') if item.strip()]
                else:
                    analysis[field] = []
        
        return analysis
        
    except Exception as e:
        raise Exception(f"Risk analysis validation failed: {str(e)}")