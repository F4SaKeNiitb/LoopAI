"""
RAG Service for managing the medical RAG chatbot lifecycle
"""
import logging
from typing import Optional, Dict, Any, List
import gc  # Import garbage collection


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
        """
        if not self.is_initialized():
            logger.error("RAG service not initialized")
            return False
        
        try:
            job_id = report_data.get('job_id', None)
            if job_id:
                self.health_reports[job_id] = report_data
                logger.info(f"Added health report for job {job_id}")
                return True
            else:
                logger.warning("No job_id found in report data")
                return False
        except Exception as e:
            logger.error(f"Error adding health report: {e}", exc_info=True)
            return False
    
    async def answer_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a query by searching through stored health reports
        This is a simplified implementation for memory efficiency
        """
        if not self.is_initialized():
            raise Exception("RAG service not initialized")
        
        # Simple search through stored reports
        results = []
        query_lower = query.lower()
        
        for job_id, report_data in self.health_reports.items():
            # Search in various fields of the report
            search_text = ""
            
            # Combine various textual fields for searching
            if 'summary' in report_data:
                search_text += " " + str(report_data['summary'])
            if 'doctor_summary' in report_data:
                search_text += " " + str(report_data['doctor_summary'])
            if 'patient_info' in report_data:
                search_text += " " + str(report_data['patient_info'])
            if 'questions' in report_data:
                search_text += " " + " ".join(report_data['questions'])
            if 'warnings' in report_data:
                search_text += " " + " ".join(report_data['warnings'])
            if 'original_texts' in report_data:
                search_text += " " + " ".join([str(text) for text in report_data['original_texts']])
            if 'extracted_parameters' in report_data:
                # Add parameter information
                for param in report_data['extracted_parameters']:
                    search_text += f" {param.get('field', '')} {param.get('value', '')} {param.get('units', '')}"
            
            # Simple keyword matching
            search_text_lower = search_text.lower()
            if query_lower in search_text_lower:
                # Calculate a simple relevance score
                score = search_text_lower.count(query_lower)
                
                results.append({
                    'job_id': job_id,
                    'report_data': report_data,
                    'score': score,
                    'matched_text': search_text[:500]  # Limit length
                })
        
        # Sort by score and take top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:top_k]
        
        # Generate a simple answer based on matches
        if results:
            # Combine the best matches
            answer_parts = []
            for result in results:
                report_summary = result['report_data'].get('summary', '')
                if report_summary:
                    answer_parts.append(f"From report {result['job_id']}: {report_summary}")
            
            answer = " ".join(answer_parts) if answer_parts else f"Found {len(results)} relevant reports containing information about '{query}'."
        else:
            answer = f"Could not find specific information about '{query}' in stored health reports. The system currently has {len(self.health_reports)} reports available."
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_reports": [r['report_data'] for r in results],
            "processing_time": 0.1,  # Placeholder
            "citations": [r['job_id'] for r in results],
            "total_sources": len(results),
            "avg_confidence": 0.7,  # Placeholder confidence
            "query_transformations_applied": 0
        }
    
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