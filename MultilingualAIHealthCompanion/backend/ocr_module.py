import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import asyncio
from typing import Union
from fastapi import UploadFile


import logging

# Set up logging for this module
logger = logging.getLogger(__name__)

async def extract_text_from_file_bytes(file_content: bytes, filename: str) -> str:
    """
    Extract text from file content (bytes) using OCR.
    """
    logger.info(f"Starting OCR processing for file: {filename}")
    file_extension = filename.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        logger.info("Processing PDF file")
        # Try to extract text from PDF first
        try:
            import PyPDF2
            from io import BytesIO
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # If the text is very short, the PDF might be image-based and need OCR
            if len(text.strip()) < 50:  # Less than 50 characters suggests image-based PDF
                logger.info(f"PDF has very little text ({len(text)} chars), attempting OCR on pages")
                # Try converting PDF to images and running OCR
                text = await extract_text_from_pdf_with_ocr(file_content)
            
            logger.info(f"PDF processing completed successfully, extracted {len(text)} characters")
            return text
        except ImportError:
            logger.warning("PyPDF2 not available, using fallback for PDF")
            # Fallback to a cloud OCR service for PDFs
            return await fallback_cloud_ocr(file_content)
        except Exception as e:
            logger.error(f"Error during PDF processing: {str(e)}", exc_info=True)
            # In case of PDF processing error, try fallback
            return await fallback_cloud_ocr(file_content)
    elif file_extension in ['jpg', 'jpeg', 'png']:
        logger.info(f"Processing image file with extension: {file_extension}")
        try:
            # Process image file
            image = Image.open(io.BytesIO(file_content))
            # Convert to OpenCV format
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Preprocess image for better OCR
            processed_image = preprocess_image(image)
            
            # Perform OCR
            text = pytesseract.image_to_string(processed_image)
            logger.info(f"OCR completed successfully, extracted {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"Error during OCR processing: {str(e)}", exc_info=True)
            raise
    else:
        logger.info(f"Unsupported file extension: {file_extension}, using fallback OCR")
        # Fallback to cloud OCR if needed
        return await fallback_cloud_ocr(file_content)


def preprocess_image(image):
    """
    Preprocess image to enhance quality before OCR.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to get image with only black and white
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to remove noise
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return opening


async def extract_text_from_pdf_with_ocr(file_content: bytes) -> str:
    """
    Extract text from PDF by converting pages to images and running OCR on them.
    This handles image-based PDFs where text extraction won't work.
    """
    try:
        # Use pdf2image to convert PDF to images
        from pdf2image import convert_from_bytes
        import tempfile
        import os
        
        # Convert PDF to list of PIL images
        # Use poppler_path parameter if needed for better memory management
        pages = convert_from_bytes(file_content, thread_count=1)  # Limit to 1 thread to save memory
        
        text = ""
        for i, page in enumerate(pages):
            logger.info(f"Processing page {i+1}/{len(pages)} with OCR")
            # Convert PIL image to OpenCV format
            opencv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            
            # Preprocess the image for better OCR
            processed_image = preprocess_image(opencv_image)
            
            # Extract text from the image
            page_text = pytesseract.image_to_string(processed_image)
            text += page_text + "\n"
            
            # Explicitly delete the page image to free memory
            del page
            del opencv_image
            del processed_image
        
        # Also delete the pages list to free memory
        del pages
        
        return text
    except ImportError:
        logger.error("pdf2image not installed, cannot perform OCR on PDF. Install with: pip install pdf2image")
        return "Text extraction failed: pdf2image library not available"
    except Exception as e:
        logger.error(f"Error during PDF OCR processing: {str(e)}", exc_info=True)
        return f"Text extraction failed: {str(e)}"


async def fallback_cloud_ocr(image_content: bytes) -> str:
    """
    Placeholder function for fallback to cloud OCR service like Google Vision API.
    """
    # This is a placeholder - would integrate with Google Vision API in real implementation
    print("Using fallback cloud OCR (Google Vision API placeholder)")
    
    # Simulate API call delay
    await asyncio.sleep(1)
    
    # Return placeholder text
    return "Text extracted via cloud OCR service"