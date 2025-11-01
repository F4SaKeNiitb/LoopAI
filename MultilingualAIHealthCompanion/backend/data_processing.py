import re
from typing import List, Dict, Any
from dataclasses import dataclass
import logging
import os
from dotenv import load_dotenv
load_dotenv()

# Set up logging for this module
logger = logging.getLogger(__name__)

# Use absolute import instead of relative import to work in background tasks
import lab_data
LAB_TEST_MAPPINGS = lab_data.LAB_TEST_MAPPINGS


@dataclass
class LabResult:
    field: str
    value: float
    units: str
    reference_range: str
    is_critical: bool = False
    is_abnormal: bool = False


def extract_parameters(text: str) -> List[Dict[str, Any]]:
    """
    Extract parameters using Gemini AI with a specialized prompt.
    """
    logger.info(f"Starting parameter extraction, input text length: {len(text) if text else 0}")
    
    # If no GEMINI_API_KEY is available, we'll still try with the default mapping approach
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info("No GEMINI_API_KEY found, using default mapping approach")
        return _extract_known_parameters_fallback(text)
    
    # Use Gemini to extract parameters with a specialized prompt
    return _extract_parameters_with_gemini(text)


def extract_patient_info(text: str) -> Dict[str, Any]:
    """
    Extract patient information using Gemini AI with a specialized prompt.
    """
    logger.info(f"Starting patient info extraction, input text length: {len(text) if text else 0}")
    
    # If no GEMINI_API_KEY is available, return empty patient info
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info("No GEMINI_API_KEY found, skipping patient info extraction")
        return {}
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Create a comprehensive prompt for extracting patient information
        prompt = f"""
        Analyze the following health report text and extract patient information.
        
        Extract the following information if present:
        - Patient name
        - Age
        - Gender/Sex
        - Date of report
        - Report type/investigation
        - Medical record number
        - Any other relevant patient details
        
        Format the results as a JSON object:
        {{
          "name": "patient name if present",
          "age": "age if present",
          "gender": "gender if present",
          "date": "report date if present",
          "report_type": "type of report if present",
          "medical_record_number": "medical record number if present",
          "other_details": "any other relevant information"
        }}
        
        The health report text to analyze:
        {text}
        
        Return ONLY the JSON object with no other text.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean the response to extract JSON
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove ```
        
        import json
        patient_info = json.loads(response_text)
        
        logger.info(f"Patient info extraction completed, found: {list(patient_info.keys())}")
        return patient_info
        
    except Exception as e:
        logger.error(f"Error during Gemini patient info extraction: {str(e)}")
        return {}


def _extract_parameters_with_gemini(text: str) -> List[Dict[str, Any]]:
    """
    Use Gemini to identify all parameters in the health report.
    """
    # Only proceed if we have a GEMINI_API_KEY
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info("No GEMINI_API_KEY found, using fallback method")
        return _extract_known_parameters_fallback(text)
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Create a comprehensive prompt for extracting health parameters
        prompt = f"""
        Analyze the following health report text and extract all laboratory test results, vital signs, 
        and health measurements. Identify the test name, value, and unit for each parameter.
        
        Format the results as a JSON array with the following structure:
        [
          {{
            "field": "test_name",
            "value": numeric_value,
            "units": "unit_of_measurement",
            "reference_range": "normal_range_for_test",
            "is_critical": boolean,  # true if value is critically high or low
            "is_abnormal": boolean   # true if value is outside normal range
          }}
        ]
        
        For each parameter found:
        1. Identify the test name (e.g., "Hemoglobin", "Cholesterol", "Glucose")
        2. Extract the numeric value
        3. Extract the unit of measurement (e.g., "g/dL", "mg/dL", "mmol/L", "U/L")
        4. Determine the normal reference range (if mentioned in text, else provide common range)
        5. Mark as abnormal if outside normal range
        6. Mark as critical if significantly outside normal range
        
        The health report text to analyze:
        {text}
        
        Return ONLY the JSON array with no other text. If no parameters are found, return an empty array.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        logger.debug(f"Gemini response: {response_text}")
        
        # Clean the response to extract JSON
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove ```
        
        import json
        params = json.loads(response_text)
        
        results = []
        for param in params:
            field = param.get('field', 'Unknown Test')
            value = float(param.get('value', 0))
            units = param.get('units', 'N/A')
            ref_range = param.get('reference_range', 'See lab reference')
            is_critical = param.get('is_critical', False)
            is_abnormal = param.get('is_abnormal', False)
            
            result = LabResult(
                field=field,
                value=value,
                units=units,
                reference_range=ref_range,
                is_critical=is_critical,
                is_abnormal=is_abnormal
            )
            
            results.append({
                "field": result.field,
                "value": result.value,
                "units": result.units,
                "reference_range": result.reference_range,
                "is_critical": result.is_critical,
                "is_abnormal": result.is_abnormal
            })
        
        logger.info(f"Gemini found {len(results)} parameters")
        return results
        
    except Exception as e:
        logger.error(f"Error during Gemini parameter extraction: {str(e)}")
        # Fallback to the original method if Gemini fails
        return _extract_known_parameters_fallback(text)


def _extract_known_parameters_fallback(text: str) -> List[Dict[str, Any]]:
    """
    Fallback method using regex matching when Gemini is not available.
    """
    logger.info("Using fallback parameter extraction method")
    results = []
    cleaned_text = text.lower()
    
    # For each test in our mapping, search for its presence in the text
    for test_name, test_info in LAB_TEST_MAPPINGS.items():
        # Look for all synonyms of this test
        for synonym in test_info['synonyms']:
            # Pattern to match value followed by units
            # Example: "hemoglobin 12.5 g/dl" or "hb 12.5 g/dl"
            pattern = rf"({synonym})\s*[:=]?\s*([0-9]+\.?[0-9]*)\s*({test_info['unit']})"
            matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
            
            for match in matches:
                value = float(match[1])
                units = match[2]
                
                # Determine if value is abnormal or critical
                is_abnormal = False
                is_critical = False
                
                if test_info.get('reference_range'):
                    ref_range = test_info['reference_range']
                    # For now, simple comparison (could be enhanced for age/sex specific ranges)
                    if isinstance(ref_range, dict):
                        min_val, max_val = ref_range['min'], ref_range['max']
                    else:
                        # Simple range like "12-16"
                        range_parts = ref_range.split('-')
                        min_val, max_val = float(range_parts[0]), float(range_parts[1])
                    
                    if value < min_val or value > max_val:
                        is_abnormal = True
                    
                    # Check critical ranges
                    critical_range = test_info.get('critical_range')
                    if critical_range:
                        c_min, c_max = critical_range['min'], critical_range['max']
                        if value < c_min or value > c_max:
                            is_critical = True
                
                result = LabResult(
                    field=test_name,
                    value=value,
                    units=units,
                    reference_range=test_info['reference_range'],
                    is_critical=is_critical,
                    is_abnormal=is_abnormal
                )
                
                results.append({
                    "field": result.field,
                    "value": result.value,
                    "units": result.units,
                    "reference_range": result.reference_range,
                    "is_critical": result.is_critical,
                    "is_abnormal": result.is_abnormal
                })
    
    logger.info(f"Fallback method found {len(results)} parameters")
    return results


def normalize_units(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert between different units where possible.
    This is a simplified version; a complete implementation would have more conversions.
    """
    logger.debug(f"Normalizing units: {value} {from_unit} to {to_unit}")
    # Placeholder implementation
    # In a real system, you'd have comprehensive unit conversion logic
    if from_unit == to_unit:
        return value
    
    # Add unit conversion logic here as needed
    # Example: convert mg/dL to mmol/L for glucose
    if from_unit == "mg/dL" and to_unit == "mmol/L":
        return value / 18.018
    
    if from_unit == "mmol/L" and to_unit == "mg/dL":
        return value * 18.018
    
    # Default: return value as is if no conversion is known
    return value