"""
RAG Service for managing the medical RAG chatbot lifecycle
"""
import logging
from typing import Optional, Dict, Any, List
from .medical_rag import HealthReportRAG
from .advanced_rag_techniques import HyDEGenerator, QueryTransformer, CrossEncoderReranker, HybridSearch, ParentChildChunker


logger = logging.getLogger(__name__)


class RAGService:
    """
    Service class for managing the RAG chatbot
    Handles initialization, health checks, and operations
    """
    
    def __init__(self):
        self.rag_system: Optional[HealthReportRAG] = None
        self.hyde_generator: Optional[HyDEGenerator] = None
        self.query_transformer: Optional[QueryTransformer] = None
        self.cross_encoder: Optional[CrossEncoderReranker] = None
        self.hybrid_search: Optional[HybridSearch] = None
        self.parent_child_chunker: Optional[ParentChildChunker] = None
        
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize all RAG components
        """
        try:
            logger.info("Initializing RAG service components...")
            
            # Initialize the main RAG system
            self.rag_system = HealthReportRAG()
            
            # Initialize advanced RAG techniques
            self.hyde_generator = HyDEGenerator()
            self.query_transformer = QueryTransformer()
            self.cross_encoder = CrossEncoderReranker()
            self.hybrid_search = HybridSearch()
            self.parent_child_chunker = ParentChildChunker()
            
            self._initialized = True
            logger.info("RAG service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {e}", exc_info=True)
            self._initialized = False
            return False
    
    def is_initialized(self) -> bool:
        """Check if the RAG service is properly initialized"""
        return self._initialized and self.rag_system is not None
    
    def add_health_report(self, report_data: Dict[str, Any]) -> bool:
        """
        Add a health report to the RAG system
        """
        if not self.is_initialized():
            logger.error("RAG service not initialized")
            return False
        
        try:
            self.rag_system.add_health_report(report_data)
            return True
        except Exception as e:
            logger.error(f"Error adding health report: {e}", exc_info=True)
            return False
    
    async def answer_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a query using the RAG system
        """
        if not self.is_initialized():
            raise Exception("RAG service not initialized")
        
        return await self.rag_system.answer_query(query, top_k=top_k)
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the RAG system
        """
        status = {
            "initialized": self._initialized,
            "rag_system_active": self.rag_system is not None,
            "hyde_active": self.hyde_generator is not None,
            "query_transformer_active": self.query_transformer is not None,
            "cross_encoder_active": self.cross_encoder is not None,
            "hybrid_search_active": self.hybrid_search is not None,
            "parent_child_chunker_active": self.parent_child_chunker is not None,
        }
        
        # Add counts if system is active
        if self.rag_system:
            status["document_count"] = len(self.rag_system.documents) if self.rag_system.documents else 0
            status["chunk_count"] = len(self.rag_system.chunks) if self.rag_system.chunks else 0
        else:
            status["document_count"] = 0
            status["chunk_count"] = 0
        
        return status
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Basic search functionality
        """
        if not self.is_initialized():
            logger.error("RAG service not initialized")
            return []
        
        try:
            # Use the retrieve method from the RAG system
            retrieved_chunks = self.rag_system.retrieve(query, top_k=top_k)
            
            results = []
            for chunk in retrieved_chunks:
                results.append({
                    "text": chunk.text,
                    "score": chunk.score,
                    "confidence": chunk.confidence,
                    "source_document": chunk.source_document,
                    "metadata": chunk.metadata
                })
            
            return results
        except Exception as e:
            logger.error(f"Error in document search: {e}", exc_info=True)
            return []


# Global instance of the RAG service
rag_service = RAGService()