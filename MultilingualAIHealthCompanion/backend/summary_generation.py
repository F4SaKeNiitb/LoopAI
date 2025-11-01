import google.generativeai as genai
from typing import List, Dict, Any
import os
import logging
import re
from dotenv import load_dotenv
load_dotenv()

# Set up logging for this module
logger = logging.getLogger(__name__)

# Use absolute import instead of relative import to work in background tasks
import lab_data
LAB_TEST_MAPPINGS = lab_data.LAB_TEST_MAPPINGS


def generate_summaries(parameters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Two-stage generation process:
    1. Rule-based engine to generate explanations
    2. LLM-based paraphrasing for fluent, conversational summaries
    """
    logger.info(f"Starting summary generation for {len(parameters)} parameters")
    # Separate normal and abnormal results
    abnormal_results = [p for p in parameters if p.get('is_abnormal', False)]
    normal_results = [p for p in parameters if not p.get('is_abnormal', False)]
    
    # Generate rule-based explanations for abnormal results
    rule_based_explanations = []
    critical_warnings = []
    
    for param in abnormal_results:
        field = param['field']
        value = param['value']
        units = param['units']
        ref_range = param['reference_range']
        
        # Get test description
        test_info = LAB_TEST_MAPPINGS.get(field, {})
        description = test_info.get('description', '')
        
        # Determine if high or low by parsing the reference range
        is_low = False
        is_high = False
        
        if isinstance(ref_range, dict):
            # For dict format like {"min": 12, "max": 16}
            min_val, max_val = ref_range['min'], ref_range['max']
            if value < min_val:
                is_low = True
            elif value > max_val:
                is_high = True
        elif isinstance(ref_range, str):
            # Handle string formats like "12-16", "<120", ">5.0", "70-99", "90-120 mg/dL", etc.
            # First, remove units if present (anything after the numbers)
            ref_range_clean = ref_range.strip()
            # Remove any units by taking only the part before any space that contains numbers
            # This regex will match patterns like "12-16", "<120", ">5.0", "90-120 mg/dL", etc.
            import re
            # Extract just the range part (numbers, -, <, >, .)
            range_match = re.search(r'^([<>=]?\s*\d+\.?\d*\s*[-]\s*\d+\.?\d*|[<>=]?\s*\d+\.?\d*)', ref_range_clean)
            if range_match:
                range_part = range_match.group(1).strip()
                range_part_clean = range_part.replace(" ", "")
                
                if '-' in range_part_clean:
                    # Format like "12-16" or "70-99"
                    range_parts = range_part_clean.replace(">", "").replace("<", "").replace("=", "").split('-')
                    if len(range_parts) == 2:
                        try:
                            min_val, max_val = float(range_parts[0]), float(range_parts[1])
                            if value < min_val:
                                is_low = True
                            elif value > max_val:
                                is_high = True
                        except ValueError:
                            # If conversion fails, we can't determine normal/abnormal
                            logger.warning(f"Could not parse range '{ref_range}' for {field}")
                elif range_part_clean.startswith('<'):
                    # Format like "<120" - value should be less than
                    try:
                        threshold = float(range_part_clean[1:])
                        if value >= threshold:
                            is_high = True
                    except ValueError:
                        logger.warning(f"Could not parse range '{ref_range}' for {field}")
                elif range_part_clean.startswith('>'):
                    # Format like ">5.0" - value should be greater than  
                    try:
                        threshold = float(range_part_clean[1:])
                        if value <= threshold:
                            is_low = True
                    except ValueError:
                        logger.warning(f"Could not parse range '{ref_range}' for {field}")
                elif range_part_clean.startswith('='):
                    # Format like "=5.0" - value should equal
                    try:
                        target_val = float(range_part_clean[1:])
                        if value != target_val:
                            is_abnormal = True  # All non-matching values are abnormal
                    except ValueError:
                        logger.warning(f"Could not parse range '{ref_range}' for {field}")
                else:
                    # Handle case where just a number is provided (e.g., single threshold)
                    try:
                        # If there's no comparison operator, it may be a single value threshold
                        logger.warning(f"Unclear range format '{ref_range}' for {field}, skipping comparison")
                    except ValueError:
                        logger.warning(f"Could not parse range '{ref_range}' for {field}")
            else:
                logger.warning(f"Could not parse range '{ref_range}' for {field}")
        # For cases where ref_range is not a dict or str, skip comparison
        
        # Determine status text
        if is_low:
            status = "low"
        elif is_high:
            status = "high"
        else:
            # We know it's abnormal, so if it's not low or high based on range parsing,
            # it might be due to special conditions Gemini detected
            status = "outside normal range"
        
        explanation = f"Your {field} is {value} {units}, which is {status} compared to the normal range of {ref_range} {units}. {description}"
        
        # Check if critical
        if param.get('is_critical', False):
            critical_warnings.append(f"URGENT: Your {field} result of {value} {units} is critically {status} and requires immediate medical attention.")
        
        rule_based_explanations.append(explanation)
    
    # Generate plain language summary
    plain_language = generate_plain_language_summary(rule_based_explanations, normal_results)
    
    # Generate doctor-style summary
    doctor_summary = generate_doctor_summary(abnormal_results)
    
    # Generate suggested questions
    questions = generate_suggested_questions(abnormal_results)
    
    # Generate warnings
    warnings = critical_warnings
    if not critical_warnings and abnormal_results:
        warnings.append("This report contains abnormal results that should be reviewed by a healthcare professional.")
    
    logger.info("Summary generation completed successfully")
    return {
        "plain_language": plain_language,
        "doctor_summary": doctor_summary,
        "questions": questions,
        "warnings": warnings
    }


def generate_plain_language_summary(abnormal_explanations: List[str], normal_results: List[Dict[str, Any]]) -> str:
    """
    Create a user-friendly summary from rule-based explanations.
    """
    logger.debug(f"Generating plain language summary for {len(abnormal_explanations)} abnormal and {len(normal_results)} normal results")
    if not abnormal_explanations:
        return "All your test results are within normal ranges. This is great news!"
    
    # Combine all abnormal explanations
    summary = "Here's what your lab results show:\n\n"
    for i, explanation in enumerate(abnormal_explanations, 1):
        summary += f"{i}. {explanation}\n"
    
    if normal_results:
        summary += f"\n{len(normal_results)} additional test results were within normal ranges."
    
    # Use LLM to make it more conversational if available
    try:
        # Placeholder for LLM integration
        llm_summary = llm_paraphrase(summary)
        logger.debug("LLM paraphrasing applied to plain language summary")
        return llm_summary
    except Exception:
        # Fallback to rule-based summary if LLM is not available
        logger.warning("LLM paraphrasing failed, returning rule-based summary")
        return summary


def generate_doctor_summary(abnormal_results: List[Dict[str, Any]]) -> str:
    """
    Generate a concise, doctor-style summary listing only abnormal findings.
    """
    logger.debug(f"Generating doctor summary for {len(abnormal_results)} abnormal results")
    if not abnormal_results:
        return "All lab values within normal limits."
    
    summary = "Abnormal findings:\n"
    for result in abnormal_results:
        field = result['field']
        value = result['value']
        units = result['units']
        ref_range = result['reference_range']
        
        # Determine if high or low by parsing the reference range
        direction = "outside normal range"  # Default
        
        if isinstance(ref_range, dict):
            # For dict format like {"min": 12, "max": 16}
            min_val, max_val = ref_range['min'], ref_range['max']
            if value < min_val:
                direction = "low"
            elif value > max_val:
                direction = "high"
        elif isinstance(ref_range, str):
            # Handle string formats like "12-16", "<120", ">5.0", "70-99 mg/dL", etc.
            # First, remove units if present (anything after the numbers)
            ref_range_clean = ref_range.strip()
            # Remove any units by taking only the part before any space that contains numbers
            import re
            # Extract just the range part (numbers, -, <, >, .)
            range_match = re.search(r'^([<>=]?\s*\d+\.?\d*\s*[-]\s*\d+\.?\d*|[<>=]?\s*\d+\.?\d*)', ref_range_clean)
            if range_match:
                range_part = range_match.group(1).strip()
                range_part_clean = range_part.replace(" ", "")
                
                if '-' in range_part_clean:
                    # Format like "12-16" or "70-99"
                    range_parts = range_part_clean.replace(">", "").replace("<", "").replace("=", "").split('-')
                    if len(range_parts) == 2:
                        try:
                            min_val, max_val = float(range_parts[0]), float(range_parts[1])
                            if value < min_val:
                                direction = "low"
                            elif value > max_val:
                                direction = "high"
                        except ValueError:
                            # If conversion fails, use default "outside normal range"
                            pass
                elif range_part_clean.startswith('<'):
                    # Format like "<120" - value should be less than
                    try:
                        threshold = float(range_part_clean[1:])
                        if value >= threshold:
                            direction = "high"
                    except ValueError:
                        # If conversion fails, use default
                        pass
                elif range_part_clean.startswith('>'):
                    # Format like ">5.0" - value should be greater than
                    try:
                        threshold = float(range_part_clean[1:])
                        if value <= threshold:
                            direction = "low"
                    except ValueError:
                        # If conversion fails, use default
                        pass
                elif range_part_clean.startswith('='):
                    # Format like "=5.0" - value should equal
                    try:
                        target_val = float(range_part_clean[1:])
                        if value != target_val:
                            direction = "outside normal range"  # Not equal to target
                    except ValueError:
                        # If conversion fails, use default
                        pass
                else:
                    # Handle case where just a number is provided
                    try:
                        # If there's no comparison operator, it may be a single value threshold
                        logger.debug(f"Unclear range format '{ref_range}' for {field}")
                    except ValueError:
                        pass
            # For other formats, keep default "outside normal range"
        
        summary += f"- {field}: {value} {units} ({direction} vs normal range {ref_range} {units})\n"
    
    logger.debug("Doctor summary generated")
    return summary


def generate_suggested_questions(abnormal_results: List[Dict[str, Any]]) -> List[str]:
    """
    Generate 3-5 suggested questions for the user to ask their doctor.
    """
    logger.debug(f"Generating suggested questions for {len(abnormal_results)} abnormal results")
    if not abnormal_results:
        return [
            "What can I do to maintain my healthy results?",
            "When should I have these tests repeated?",
            "Are there any lifestyle changes you recommend based on these results?"
        ]
    
    questions = []
    for result in abnormal_results[:3]:  # Limit to 3 most abnormal results
        field = result['field']
        
        if field == "Hemoglobin":
            questions.extend([
                "Could my low hemoglobin indicate anemia?",
                "Should I consider iron supplements or dietary changes?",
                "Could this be related to my diet or other health conditions?"
            ])
        elif field == "Cholesterol":
            questions.extend([
                "What dietary changes should I make to lower my cholesterol?",
                "Would you recommend medication to manage my cholesterol levels?",
                "How often should I have my cholesterol checked?"
            ])
        elif field == "Glucose":
            questions.extend([
                "Could these results indicate pre-diabetes or diabetes?",
                "What steps should I take to manage my blood sugar levels?",
                "Should I consult with an endocrinologist?"
            ])
        else:
            questions.extend([
                f"What does this abnormal {field} result indicate?",
                f"What treatment options are available for this condition?",
                f"How soon should I follow up with these results?"
            ])
        
        # Limit to 5 questions total
        if len(questions) >= 5:
            break
    
    # Add additional generic questions if needed
    while len(questions) < 3:
        questions.append("What lifestyle changes do you recommend based on these results?")
    
    # Remove duplicates while preserving order
    unique_questions = []
    for q in questions:
        if q not in unique_questions:
            unique_questions.append(q)
    
    result = unique_questions[:5]  # Limit to 5 questions
    logger.debug(f"Generated {len(result)} suggested questions")
    return result


def llm_paraphrase(text: str) -> str:
    """
    Placeholder function for LLM-based paraphrasing using Google Gemini API.
    """
    logger.debug("Attempting LLM paraphrasing")
    # In a real implementation, this would call the Gemini API
    # For now, return the original text with minimal modification
    api_key = os.getenv("GEMINI_API_KEY")  # Retrieve from environment
    if not api_key:
        logger.info("No GEMINI_API_KEY found, returning original text")
        return text  # Return original if no API key available
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        prompt = f"""
        Please rewrite the following medical lab results summary in a more conversational, 
        easy-to-understand format. Keep the information accurate but make it more accessible 
        to a patient. Do not provide medical advice, just explain what the results mean.
        
        Original text: {text}
        """
        
        response = model.generate_content(prompt)
        logger.info("LLM paraphrasing completed successfully")
        return response.text
    except Exception as e:
        logger.error(f"Error with LLM paraphrasing: {str(e)}")
        return text  # Return original if LLM fails