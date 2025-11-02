import sys
import os
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union
import asyncio
import uuid
from datetime import datetime
import json
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

# Add the rag_chatbot module to the path to import it
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_chatbot'))

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track running background tasks to prevent resource leaks
background_tasks = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for proper startup and shutdown."""
    logger.info("Starting up application...")
    # Startup
    logger.info("Starting RAG service initialization...")
    initialize_rag()
    logger.info(f"RAG service initialization completed. Initialized: {rag_chatbot is not None and rag_chatbot.is_initialized()}")
    yield
    # Shutdown
    logger.info("Shutting down application, cleaning up background tasks...")
    
    # Wait for background tasks to complete with a timeout
    if background_tasks:
        # Create a list of tasks to avoid modifying the set during iteration
        tasks_to_wait = list(background_tasks)
        if tasks_to_wait:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks_to_wait, return_exceptions=True),
                    timeout=10.0  # 10 second timeout
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timed out waiting for {len(tasks_to_wait)} background tasks to complete")
    
    logger.info("Application shutdown complete")

app = FastAPI(
    title="Multilingual AI Health Companion API", 
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount audio files directory to serve static audio files
audio_dir = os.path.join(os.path.dirname(__file__), "audio_files")
if os.path.exists(audio_dir):
    app.mount("/audio", StaticFiles(directory=audio_dir), name="audio")
    logger.info(f"Mounted audio directory: {audio_dir}")
else:
    logger.warning(f"Audio directory does not exist: {audio_dir}")

# Store for job results (in production, use a proper database)
job_store: Dict[str, Dict[str, Any]] = {}

# Initialize the RAG chatbot
rag_chatbot = None

class UploadRequest(BaseModel):
    text: Optional[str] = None
    language: Optional[str] = "en"

from typing import Union

@app.post("/upload")
async def upload_file_or_text(
    text: str = Form(default=""), 
    files: Union[UploadFile, List[UploadFile], None] = File(default=None),
    language: str = Form(default="en")
):
    """
    Accept multiple file uploads (PDFs, images) or raw text from SMS messages.
    Process data asynchronously, return job_id immediately.
    """
    job_id = str(uuid.uuid4())
    
    # Debug information
    logger.info(f"Received request:")
    logger.info(f"  text: {text is not None and text != ''}")
    # Handle both single file and list of files
    if files is None:
        file_list = []
        has_files = False
    elif isinstance(files, list):
        file_list = files
        has_files = len(file_list) > 0
    else:
        file_list = [files]  # Single file, wrap in list
        has_files = True
    
    logger.info(f"  files: {has_files}")
    if has_files:
        logger.info(f"  number of files: {len(file_list)}")
        for i, file in enumerate(file_list):
            logger.info(f"    file {i}: {file.filename}, {file.content_type}")
    logger.info(f"  language: {language}")
    
    # Store job with initial status
    job_store[job_id] = {
        "status": "PENDING",
        "result": None,
        "created_at": datetime.utcnow().isoformat(),
        "language": language
    }
    
    # Start processing in background - but first read all file contents to avoid I/O issues
    file_contents = []
    
    if file_list:
        for file in file_list:
            if file.filename:  # Only process if file exists and has a name
                # Read file content immediately to preserve it
                file_content = await file.read()
                filename = file.filename
                content_type = file.content_type
                # Reset file pointer to avoid issues
                await file.seek(0)
                
                file_contents.append({
                    "content": file_content,
                    "filename": filename,
                    "content_type": content_type
                })
    
    # Start processing in background with the contents already read
    asyncio.create_task(process_job(job_id, text if text != "" else None, file_contents))
    
    return {"job_id": job_id}


@app.get("/report/{job_id}")
async def get_report(job_id: str):
    """
    Return the status of the processing job.
    Once status is COMPLETED, return full analysis in structured JSON format.
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] == "COMPLETED":
        return {
            "job_id": job_id,
            "status": job["status"],
            "result": job["result"],
            "language": job["language"]
        }
    else:
        return {
            "job_id": job_id,
            "status": job["status"],
            "message": "Processing in progress..."
        }

# Initialize RAG chatbot after dependencies are loaded
def initialize_rag():
    global rag_chatbot
    try:
        # Try multiple import strategies to ensure the module is found
        import importlib
        import sys
        import os
        
        # Ensure the rag_chatbot directory is in the Python path
        rag_chatbot_path = os.path.join(os.path.dirname(__file__), 'rag_chatbot')
        if rag_chatbot_path not in sys.path:
            sys.path.insert(0, rag_chatbot_path)
        
        # Import using importlib to ensure module is loaded properly
        rag_service_module = importlib.import_module('rag_chatbot.rag_service')
        rag_service = getattr(rag_service_module, 'rag_service')
        
        success = rag_service.initialize()
        if success:
            rag_chatbot = rag_service  # Use the service instance
            logger.info("RAG service initialized successfully")
        else:
            rag_chatbot = None
            logger.error("Failed to initialize RAG service")
    except ImportError as e:
        logger.error(f"Failed to import RAG service: {e}")
        logger.error(f"Available modules in rag_chatbot: {os.listdir(os.path.join(os.path.dirname(__file__), 'rag_chatbot'))}")
        rag_chatbot = None
    except Exception as e:
        logger.error(f"Error initializing RAG service: {e}")
        logger.error(f"Exception details: {str(e)}", exc_info=True)  # Add full traceback
        rag_chatbot = None

# Initialize the RAG chatbot when the application starts
logger.info("Starting RAG service initialization...")
initialize_rag()
logger.info(f"RAG service initialization completed. Initialized: {rag_chatbot is not None and rag_chatbot.is_initialized()}")

# Track running background tasks to prevent resource leaks
background_tasks = set()

# Add the current directory to the Python path to ensure modules can be found in background tasks
async def process_job(job_id: str, text: Optional[str], file_contents: Optional[List[Dict[str, Any]]]):
    """
    Background task to process multiple uploaded files or text.
    Implements the complete processing pipeline with individual analysis and final summary.
    """
    task = asyncio.current_task()
    if task:
        background_tasks.add(task)
        task.add_done_callback(lambda t: background_tasks.discard(t))
    
    try:
        logger.info(f"Starting processing job {job_id}")
        job_store[job_id]["status"] = "PROCESSING"
        
        # Log the environment and path information for debugging
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python path: {sys.path[:3]}...")  # Show first 3 paths
        logger.info(f"Available files in current directory: {os.listdir('.')}")
        
        # Import modules at the beginning to avoid scope issues
        import importlib
        data_processing = importlib.import_module('data_processing')
        extract_params_func = getattr(data_processing, 'extract_parameters')
        extract_patient_info_func = getattr(data_processing, 'extract_patient_info')
        ocr_module = importlib.import_module('ocr_module')
        extract_text_func = getattr(ocr_module, 'extract_text_from_file_bytes')
        summary_generation = importlib.import_module('summary_generation')
        generate_summaries_func = getattr(summary_generation, 'generate_summaries')
        translation_tts = importlib.import_module('translation_tts')
        generate_translations_func = getattr(translation_tts, 'generate_translations_and_tts')
        
        # Step 1: Extract text from files or use provided text
        all_extracted_texts = []
        all_extracted_parameters = []
        all_patient_info = []
        all_individual_summaries = []
        
        # Process each file individually
        if file_contents:
            for file_data in file_contents:
                file_content = file_data["content"]
                filename = file_data["filename"]
                content_type = file_data["content_type"]
                
                if content_type in ["application/pdf", "image/jpeg", "image/png", "image/jpg"]:
                    extracted_text = await extract_text_func(file_content, filename)
                    all_extracted_texts.append(extracted_text)
                    
                    # Extract parameters for this document
                    parameters = extract_params_func(extracted_text)
                    all_extracted_parameters.extend(parameters)
                    
                    # Extract patient info for this document
                    patient_info = extract_patient_info_func(extracted_text)
                    if patient_info:
                        all_patient_info.append(patient_info)
                else:
                    logger.warning(f"Unsupported content type: {content_type}, skipping")
        
        # Add any text that was provided directly
        if text:
            all_extracted_texts.append(text)
            # Extract parameters from the text
            parameters = extract_params_func(text)
            all_extracted_parameters.extend(parameters)
            
            patient_info = extract_patient_info_func(text)
            if patient_info:
                all_patient_info.append(patient_info)
        
        # Step 2: Generate individual summaries for each document
        for text in all_extracted_texts:
            parameters = extract_params_func(text)  # Extract parameters for this specific text
            individual_summary = generate_summaries_func(parameters)
            all_individual_summaries.append(individual_summary)
        
        # Step 3: Generate a final combined summary using Gemini
        final_summary = await _generate_final_summary(all_individual_summaries, all_extracted_texts)
        
        # Step 4: Generate translations and TTS for the final summary
        translated_content = await generate_translations_func(final_summary, job_store[job_id]["language"])
        
        # Step 5: Compile final result
        result = {
            "original_texts": all_extracted_texts,  # All original extracted texts
            "patient_info": all_patient_info,  # All patient information
            "extracted_parameters": all_extracted_parameters,  # All parameters from all documents
            "individual_summaries": all_individual_summaries,  # Individual summaries for each document
            "summary": final_summary["plain_language"],  # Final combined summary
            "doctor_summary": final_summary["doctor_summary"],
            "questions": final_summary["questions"],
            "warnings": final_summary["warnings"],
            "translated_audio_urls": translated_content,
            "disclaimer": "This is an AI-generated summary and not a substitute for professional medical advice. Please consult your doctor."
        }
        
        job_store[job_id]["result"] = result
        job_store[job_id]["status"] = "COMPLETED"
        logger.info(f"Successfully completed job {job_id}")
        
        # Add the processed report to the RAG system
        if rag_chatbot:
            try:
                # Add the report data to RAG for future queries
                report_data = {
                    "job_id": job_id,
                    "original_texts": all_extracted_texts,
                    "patient_info": all_patient_info,
                    "extracted_parameters": all_extracted_parameters,
                    "individual_summaries": all_individual_summaries,
                    "summary": final_summary["plain_language"],
                    "doctor_summary": final_summary["doctor_summary"],
                    "questions": final_summary["questions"],
                    "warnings": final_summary["warnings"],
                    "created_at": job_store[job_id]["created_at"],
                    "language": job_store[job_id]["language"]
                }
                success = rag_chatbot.add_health_report(report_data)
                if success:
                    logger.info(f"Added job {job_id} to RAG system")
                else:
                    logger.error(f"Failed to add job {job_id} to RAG system")
            except Exception as e:
                logger.error(f"Error adding report to RAG system: {e}")
        
        # Update the job store with the result
        job_store[job_id]["result"] = result
        job_store[job_id]["status"] = "COMPLETED"
        logger.info(f"Successfully completed job {job_id}")
        
    except Exception as e:
        job_store[job_id]["status"] = "FAILED"
        job_store[job_id]["error"] = str(e)
        logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)  # Include traceback
    finally:
        # Clean up task reference when done
        current_task = asyncio.current_task()
        if current_task in background_tasks:
            background_tasks.discard(current_task)
        
        # Force garbage collection to free memory
        import gc
        gc.collect()
        
        # If using PyTorch, clear cache to free GPU memory if it was used
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass  # torch not available


async def _generate_final_summary(individual_summaries: List[Dict[str, Any]], all_texts: List[str]) -> Dict[str, Any]:
    """
    Generate a final combined summary from all individual summaries using Gemini.
    """
    import os
    import google.generativeai as genai
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # If no API key, return a basic combination of summaries
        combined_summary = " ".join([summary.get("plain_language", "") for summary in individual_summaries])
        return {
            "plain_language": combined_summary,
            "doctor_summary": "Combined summary of multiple reports",
            "questions": ["Please consult with your doctor for interpretation of multiple reports"],
            "warnings": []
        }
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Create a summary from all individual summaries
        combined_text = "\n\n".join(all_texts)
        
        prompt = f"""
        You have been provided with multiple health reports from the same patient. 
        Please analyze all reports together and create a comprehensive summary that:
        1. Combines all significant findings from all reports
        2. Highlights any trends or changes between reports
        3. Identifies all abnormal results across all reports
        4. Provides a unified set of recommendations
        
        Individual summaries:
        {str(individual_summaries)}
        
        Full texts of all reports:
        {combined_text}
        
        Please return the result in the following JSON format:
        {{
          "plain_language": "comprehensive summary in plain language",
          "doctor_summary": "summary for healthcare provider",
          "questions": ["list", "of", "suggested", "questions"],
          "warnings": ["list", "of", "warnings", "if", "any"]
        }}
        
        Return ONLY the JSON object.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean the response to extract JSON
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove ```
        
        import json
        final_summary = json.loads(response_text)
        return final_summary
        
    except Exception as e:
        logger.error(f"Error during final summary generation: {str(e)}")
        # Fallback to simple combination
        combined_summary = " ".join([summary.get("plain_language", "") for summary in individual_summaries])
        return {
            "plain_language": combined_summary,
            "doctor_summary": "Combined summary of multiple reports",
            "questions": ["Please consult with your doctor for interpretation of multiple reports"],
            "warnings": []
        }


# Admin/clinician review endpoints
@app.get("/admin/reports")
async def list_reports():
    """Lists all processed reports"""
    completed_jobs = [
        {
            "job_id": job_id,
            "status": job_data["status"],
            "created_at": job_data["created_at"],
            "language": job_data.get("language", "en")
        }
        for job_id, job_data in job_store.items()
        if job_data["status"] == "COMPLETED"
    ]
    return {"reports": completed_jobs}


@app.get("/admin/report/{job_id}")
async def get_admin_report(job_id: str):
    """Retrieves a report's original data, extracted values, and AI-generated summary"""
    job = job_store.get(job_id)
    if not job or job["status"] != "COMPLETED":
        raise HTTPException(status_code=404, detail="Report not found or not completed")
    
    # Handle both old and new data structures
    result = job["result"]
    if "original_texts" in result:  # New structure with multiple files
        original_data = result["original_texts"]
    else:  # Old structure with single file
        original_data = result["original_text"]
    
    return {
        "job_id": job_id,
        "original_data": original_data,
        "extracted_values": result["extracted_parameters"],
        "ai_summary": result["summary"]
    }


@app.put("/admin/report/{job_id}")
async def update_admin_report(job_id: str, updated_summary: Dict[str, Any]):
    """Allows an authenticated user to edit and save the generated summary"""
    job = job_store.get(job_id)
    if not job or job["status"] != "COMPLETED":
        raise HTTPException(status_code=404, detail="Report not found or not completed")
    
    # Update the summary with the clinician's edits
    job["result"]["summary"] = updated_summary.get("summary", job["result"]["summary"])
    job["result"]["doctor_summary"] = updated_summary.get("doctor_summary", job["result"]["doctor_summary"])
    job["result"]["questions"] = updated_summary.get("questions", job["result"]["questions"])
    
    return {"message": "Report updated successfully", "job_id": job_id}


# RAG Chatbot endpoints
class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    job_ids: Optional[List[str]] = None  # Optional: only search specific reports


@app.post("/chat/query")
async def chat_query(chat_request: ChatRequest):
    """
    Query the RAG chatbot with a question about the health reports
    """
    if not rag_chatbot or not rag_chatbot.is_initialized():
        raise HTTPException(status_code=500, detail="RAG chatbot is not initialized")
    
    try:
        response = await rag_chatbot.answer_query(
            query=chat_request.query,
            top_k=chat_request.top_k
        )
        
        return response
    except Exception as e:
        logger.error(f"Error in chat query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/chat/health")
async def chat_health():
    """
    Health check for the RAG chatbot
    """
    if rag_chatbot and rag_chatbot.is_initialized():
        status = rag_chatbot.get_health_status()
        status["status"] = "healthy"
        return status
    else:
        return {
            "status": "unhealthy",
            "rag_initialized": False,
            "total_documents": 0,
            "total_chunks": 0
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)