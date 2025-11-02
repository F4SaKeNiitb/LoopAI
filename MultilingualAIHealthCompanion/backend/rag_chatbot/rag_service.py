"""
RAG Service for managing the medical RAG chatbot lifecycle
"""
import logging
from typing import Optional, Dict, Any, List
import gc  # Import garbage collection
import os
import google.generativeai as genai
import re


logger = logging.getLogger(__name__)


class RAGService:
    """
    Memory-optimized RAG service that stores documents in simple format
    to reduce memory usage on M1 Mac with 8GB RAM
    """
    
    def __init__(self):
        # Store only minimal data to reduce memory usage
        self.documents = []  # Store as simple list of dictionaries
        self.health_reports = {}  # Job ID to report data mapping
        self.processed_texts = {}  # Job ID to processed text chunks mapping
        self._initialized = True  # Simplified approach - no heavy models to initialize
    
    def initialize(self) -> bool:
        """
        Initialize RAG service with minimal memory footprint
        """
        try:
            logger.info("Initializing minimal RAG service (no heavy models)")
            self.documents = []
            self.health_reports = {}
            self._initialized = True
            logger.info("Minimal RAG service initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize minimal RAG service: {e}", exc_info=True)
            self._initialized = False
            return False
    
    def is_initialized(self) -> bool:
        """Check if the RAG service is properly initialized"""
        return self._initialized
    
    def add_health_report(self, report_data: Dict[str, Any]) -> bool:
        """
        Add a health report to the RAG system (in memory only)
        Process and chunk the original texts for better RAG retrieval
        """
        if not self.is_initialized():
            logger.error("RAG service not initialized")
            return False
        
        try:
            job_id = report_data.get('job_id', None)
            if job_id:
                self.health_reports[job_id] = report_data
                
                # Process the original OCR text for better retrieval
                # Chunk the original texts to enable semantic search
                chunks = self._chunk_text(
                    report_data.get('original_texts', []), 
                    report_data.get('extracted_parameters', [])
                )
                
                # Store chunks for this job
                self.processed_texts[job_id] = chunks
                logger.info(f"Added health report for job {job_id} with {len(chunks)} text chunks")
                return True
            else:
                logger.warning("No job_id found in report data")
                return False
        except Exception as e:
            logger.error(f"Error adding health report: {e}", exc_info=True)
            return False
    
    def _chunk_text(self, original_texts: List[str], extracted_parameters: List[Dict[str, Any]], chunk_size: int = 500) -> List[str]:
        """
        Chunk the original OCR text into smaller pieces and include parameter information for better retrieval
        """
        chunks = []
        
        # Process original OCR texts
        for original_text in original_texts:
            if not original_text or not isinstance(original_text, str):
                continue
                
            # Split text into chunks based on sentences to maintain context
            sentences = re.split(r'(?<=[.!?]) +', original_text)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
            
            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        # Add parameter-based chunks for better medical term retrieval
        param_chunks = self._create_parameter_chunks(extracted_parameters)
        chunks.extend(param_chunks)
        
        return chunks
    
    def _create_parameter_chunks(self, parameters: List[Dict[str, Any]]) -> List[str]:
        """
        Create text chunks from extracted parameters for better medical term retrieval
        """
        chunks = []
        for param in parameters:
            chunk_text = f"{param.get('field', '')}: {param.get('value', '')} {param.get('units', '')} "
            chunk_text += f"(Reference range: {param.get('reference_range', '')}) "
            chunk_text += f"Status: {'Abnormal' if param.get('is_abnormal', False) else 'Normal'} "
            if param.get('is_critical', False):
                chunk_text += "CRITICAL "
            chunks.append(chunk_text.strip())
        return chunks
    
    async def answer_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a query by searching through stored health reports
        This is a simplified implementation for memory efficiency
        Now searches through OCR chunks for better retrieval
        """
        if not self.is_initialized():
            raise Exception("RAG service not initialized")
        
        # Search through text chunks for better semantic matching
        results = []
        query_lower = query.lower()
        
        # Define medical synonyms/related terms to improve search
        medical_synonyms = {
            'basophil': ['basophils', 'baso'],
            'eosinophil': ['eosinophils', 'eosi'],
            'neutrophil': ['neutrophils', 'neutro'],
            'lymphocyte': ['lymphocytes', 'lymp'],
            'monocyte': ['monocytes', 'mono'],
            'hemoglobin': ['hb', 'hgb'],
            'creatinine': ['crea'],
            'bilirubin': ['bili'],
            'cholesterol': ['chol'],
            'glucose': ['gluc', 'sugar'],
            'potassium': ['k+', 'k'],
            'sodium': ['na+', 'na'],
            'calcium': ['ca++', 'ca'],
            'mchc': ['mch c'],
            'mch': ['mch '],
            'mcv': ['mcv '],
            'rdw': ['rdw '],
            'platelet': ['platelets'],
            'wbc': ['white blood cell'],
            'rbc': ['red blood cell'],
            'hematocrit': ['hct', 'hct'],
        }
        
        # Find matching synonyms to expand search terms
        all_search_terms = [query_lower]
        for med_term, synonyms in medical_synonyms.items():
            if med_term in query_lower or any(syn in query_lower for syn in synonyms):
                all_search_terms.extend([med_term] + synonyms)
        
        # Search through all processed text chunks from all jobs
        for job_id, chunks in self.processed_texts.items():
            report_data = self.health_reports.get(job_id, {})
            
            # Search in each chunk
            for i, chunk in enumerate(chunks):
                chunk_lower = chunk.lower()
                
                # Calculate relevance score based on all search terms
                score = 0
                matched_terms = []
                for term in all_search_terms:
                    term_count = chunk_lower.count(term.lower())
                    if term_count > 0:
                        score += term_count
                        matched_terms.append(term)
                
                if score > 0:
                    results.append({
                        'job_id': job_id,
                        'report_data': report_data,
                        'chunk_index': i,
                        'chunk_text': chunk,
                        'score': score,
                        'matched_terms': matched_terms
                    })
        
        # Sort by score and take top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:top_k]
        
        # Generate a contextual answer using Gemini based on retrieved context
        if results:
            # Prepare context from retrieved chunks
            context_parts = []
            for result in results:
                if result['matched_terms']:
                    context_parts.append(f"Report ID: {result['job_id']} (matched terms: {', '.join(result['matched_terms'])})\nChunk: {result['chunk_text']}")
                else:
                    context_parts.append(f"Report ID: {result['job_id']}\nChunk: {result['chunk_text']}")
            
            context_text = "\n\n".join(context_parts)
            
            # Use Gemini to generate a contextual answer based on retrieved context
            gemini_answer = self._generate_gemini_answer(query, context_text)
            
            # If Gemini generation fails, fall back to the previous approach
            if gemini_answer:
                answer = gemini_answer
            else:
                # Combine the best matches
                answer_parts = []
                for result in results:
                    report_summary = result['report_data'].get('summary', '')
                    if report_summary:
                        answer_parts.append(f"From report {result['job_id']}: {report_summary}")
                
                answer = " ".join(answer_parts) if answer_parts else f"Found {len(results)} relevant chunks containing information about '{query}'."
        else:
            # Even if no results found, try to generate a response with Gemini using the medical context
            gemini_answer = self._generate_medical_fallback_answer(query)
            if gemini_answer:
                answer = gemini_answer
            else:
                answer = f"Could not find specific information about '{query}' in stored health reports. The system currently has {len(self.health_reports)} reports available."
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_reports": [r['report_data'] for r in results],
            "retrieved_chunks": [r['chunk_text'] for r in results],  # Include the actual chunks retrieved
            "processing_time": 0.1,  # Placeholder
            "citations": [r['job_id'] for r in results],
            "total_sources": len(results),
            "avg_confidence": 0.8 if results else 0.3,  # Lower confidence if no direct matches
            "query_transformations_applied": len(all_search_terms) - 1  # Number of synonyms used
        }
    
    def _generate_medical_fallback_answer(self, query: str) -> str:
        """
        Generate a fallback answer using Gemini even when no direct matches are found.
        This helps provide more helpful responses for medical queries.
        """
        try:
            # Get Gemini API key from environment
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY not found, skipping Gemini fallback")
                return None
            
            # Configure the API
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Create a prompt that asks Gemini to provide a response even without direct matches
            prompt = f"""
            You are a medical AI assistant. The user asked '{query}' but no direct matches were found in the available health reports.
            The system has health reports available but the specific query wasn't found in them.
            
            Based on your medical knowledge, provide a helpful response that:
            1. Explains what the user might be looking for
            2. Suggests how they might find the information in their health reports
            3. Advises on the importance of consulting with their doctor for medical information
            
            Be concise but informative.
            """
            
            # Generate the response
            response = model.generate_content(prompt)
            
            if response.text:
                return response.text.strip()
            else:
                logger.warning("Gemini returned empty response for fallback")
                return None
                
        except Exception as e:
            logger.error(f"Error generating fallback answer with Gemini: {str(e)}")
            return None
    
    def _generate_gemini_answer(self, query: str, context: str) -> str:
        """
        Generate a contextual answer using Gemini based on the provided context.
        """
        try:
            # Get Gemini API key from environment
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("GEMINI_API_KEY not found, skipping Gemini enhancement")
                return None
            
            # Configure the API
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            # Create a prompt that asks Gemini to answer the query based on the context
            prompt = f"""
            You are a medical AI assistant. Answer the following question based on the provided health report context.
            
            Question: {query}
            
            Context from health reports:
            {context}
            
            Please provide a clear, concise, and medically relevant answer based on the information in the context.
            If the context doesn't contain the specific information requested, please state that explicitly.
            Always prioritize accuracy and clarity over providing a generic response.
            """
            
            # Generate the response
            response = model.generate_content(prompt)
            
            if response.text:
                return response.text.strip()
            else:
                logger.warning("Gemini returned empty response")
                return None
                
        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {str(e)}")
            return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the RAG system
        """
        return {
            "initialized": self._initialized,
            "total_reports": len(self.health_reports),
            "memory_usage_estimate": "low",  # We're using minimal memory
            "features_active": ["storage", "keyword_search"]  # Only basic features
        }
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Basic search functionality
        """
        if not self.is_initialized():
            logger.error("RAG service not initialized")
            return []
        
        # Simple keyword-based search
        results = []
        query_lower = query.lower()
        
        for job_id, report_data in self.health_reports.items():
            search_text = ""
            
            # Combine various textual fields for searching
            for key, value in report_data.items():
                if isinstance(value, str):
                    search_text += " " + value
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            search_text += " " + item
                        elif isinstance(item, dict):
                            search_text += " " + " ".join([str(v) for v in item.values() if isinstance(v, str)])
                elif isinstance(value, dict):
                    search_text += " " + " ".join([str(v) for v in value.values() if isinstance(v, str)])
            
            # Simple keyword matching
            search_text_lower = search_text.lower()
            if query_lower in search_text_lower:
                score = search_text_lower.count(query_lower)
                results.append({
                    "job_id": job_id,
                    "text": search_text[:300],  # Limit length
                    "score": score,
                    "metadata": {"source": job_id}
                })
        
        # Sort by score and return top k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]


# Global instance of the RAG service
rag_service = RAGService()