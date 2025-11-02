"""
Medical RAG Chatbot Implementation
Implements HyDE, Query Transformation, Hybrid Search, and other advanced RAG techniques
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import uuid
import asyncio
from datetime import datetime
from dataclasses import dataclass
import re


logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with metadata"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    confidence: float
    source_document: str
    page_number: Optional[int] = None


class MedicalRAGChatbot:
    """
    Medical RAG Chatbot with advanced retrieval techniques:
    - HyDE (Hypothetical Document Embeddings)
    - Query Transformation
    - Hybrid Search (Dense + Sparse)
    - Cross-Encoder Reranking
    - Parent-Child Chunking
    """
    
    def __init__(self):
        self.dense_retriever = None
        self.sparse_retriever = None
        self.reranker = None
        self.documents = []  # Store original documents
        self.chunks = []     # Store chunked documents
        self.index = None    # FAISS index for dense retrieval
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.parent_child_mapping = {}  # Maps child chunks to parent documents
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Initialize medical domain-specific models for embeddings and reranking
        """
        # Use domain-specific medical embedding model
        try:
            # Attempt to use a medical-specific model
            self.dense_retriever = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')
            logger.info("Loaded S-PubMedBert-MS-MARCO model for medical embeddings")
        except Exception:
            # Fallback to general model if medical-specific not available
            try:
                self.dense_retriever = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded all-MiniLM-L6-v2 as fallback model")
            except Exception as e:
                logger.error(f"Could not initialize sentence transformer: {e}")
                self.dense_retriever = None
        
        # For sparse retrieval, use TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            lowercase=True,
            stop_words=None,
            max_features=10000
        )
        
        # For reranking, we'll use a cross-encoder model
        try:
            from sentence_transformers import CrossEncoder
            # Using a model trained for re-ranking
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Loaded cross-encoder model for reranking")
        except Exception as e:
            logger.error(f"Could not initialize cross-encoder: {e}")
            self.reranker = None
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Add documents to the RAG system with metadata
        Documents should be in format: {id, text, metadata}
        """
        self.documents = documents
        # Create chunks from documents
        self._chunk_documents()
        # Build indices
        self._build_indices()
    
    def _chunk_documents(self, chunk_size: int = 512, overlap: int = 50):
        """
        Split documents into overlapping chunks for better retrieval
        Implements parent-child chunking strategy
        """
        from .advanced_rag_techniques import ParentChildChunker
        
        # Use the ParentChildChunker for more sophisticated chunking
        chunker = ParentChildChunker(child_chunk_size=128, parent_chunk_size=chunk_size, overlap=overlap)
        
        # Prepare documents in the format expected by ParentChildChunker
        formatted_docs = []
        for doc in self.documents:
            formatted_doc = {
                'id': doc.get('id', str(uuid.uuid4())),
                'text': doc['text'],
                'metadata': doc.get('metadata', {})
            }
            formatted_docs.append(formatted_doc)
        
        # Generate parent-child chunks
        all_chunks = chunker.create_parent_child_chunks(formatted_docs)
        
        # Process the chunks for our RAG system
        self.chunks = []
        self.parent_child_mapping = {}
        
        for chunk in all_chunks:
            # Handle both parent and child chunks
            processed_chunk = {
                'id': chunk['id'],
                'text': chunk['text'],
                'parent_id': chunk.get('parent_id', chunk.get('id', '').replace('_parent_', '_')),  # Default to self if no parent
                'metadata': chunk.get('metadata', {}),
                'doc_id': chunk.get('metadata', {}).get('id', 'unknown'),
                'chunk_type': chunk.get('chunk_type', 'regular')
            }
            
            self.chunks.append(processed_chunk)
            self.parent_child_mapping[chunk['id']] = processed_chunk['parent_id']
    
    def _build_indices(self):
        """
        Build FAISS index for dense retrieval and TF-IDF matrix for sparse retrieval
        """
        if not self.chunks:
            logger.warning("No chunks to build indices from")
            return
        
        # Build dense index with FAISS
        if self.dense_retriever:
            logger.info("Building dense index...")
            chunk_texts = [chunk['text'] for chunk in self.chunks]
            embeddings = self.dense_retriever.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings.astype('float32')
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity after normalization)
            self.index.add(embeddings)
            logger.info(f"Built dense index with {len(chunk_texts)} chunks")
        
        # Build sparse index with TF-IDF
        logger.info("Building sparse index...")
        chunk_texts = [chunk['text'] for chunk in self.chunks]
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(chunk_texts)
        logger.info(f"Built sparse index with {len(chunk_texts)} chunks")
    
    def _generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that answers the query.
        This helps bridge the gap between patient language and medical terminology.
        """
        # This would normally use an LLM to generate the hypothetical document
        # For now, we'll create a simple template-based approach
        # In a production system, this would use a medical LLM
        hypothetical_doc = f"Based on the query '{query}', here are the relevant medical findings and information: "
        
        # Add medical terminology related to the query
        # This is a simplified version - in practice, you'd use an LLM to generate a detailed response
        query_lower = query.lower()
        
        # Some basic transformations to medical terms
        if 'glucose' in query_lower or 'blood sugar' in query_lower:
            hypothetical_doc += "Patient shows glucose levels that may indicate diabetes or hyperglycemia. Recommended tests include HbA1c and fasting glucose. Normal glucose levels range from 70-99 mg/dL fasting."
        elif 'cholesterol' in query_lower or 'lipid' in query_lower:
            hypothetical_doc += "Lipid panel results show cholesterol levels including LDL, HDL, and triglycerides. High cholesterol may indicate cardiovascular risk. Normal total cholesterol should be below 200 mg/dL."
        elif 'blood pressure' in query_lower or 'hypertension' in query_lower:
            hypothetical_doc += "Blood pressure readings indicate systolic and diastolic pressures. Hypertension is defined as readings consistently above 140/90 mmHg. Requires monitoring and possible medication management."
        else:
            # Generate a more generic response based on the query
            hypothetical_doc += f"This document addresses the medical concern of '{query}'. It contains relevant test results, reference ranges, and clinical interpretations."
        
        return hypothetical_doc
    
    def _transform_query(self, query: str) -> List[str]:
        """
        Transform the original query into multiple variations using:
        - Multi-query generation
        - Step-back prompting
        - Query decomposition
        """
        transformed_queries = [query]  # Start with original query
        
        # Multi-query generation - create several variations
        variations = self._generate_query_variations(query)
        transformed_queries.extend(variations)
        
        # Step-back prompting - create broader questions
        step_back_queries = self._generate_step_back_queries(query)
        transformed_queries.extend(step_back_queries)
        
        # Query decomposition - break complex queries into sub-questions
        decomposed_queries = self._decompose_query(query)
        transformed_queries.extend(decomposed_queries)
        
        # Remove duplicates while preserving order
        unique_queries = []
        for q in transformed_queries:
            if q not in unique_queries:
                unique_queries.append(q)
        
        return unique_queries
    
    def _generate_query_variations(self, query: str) -> List[str]:
        """Generate multiple variations of the same query"""
        variations = []
        
        # Synonym-based variations (simplified)
        medical_synonyms = {
            'diabetes': ['diabetic condition', 'sugar diabetes', 'sugar level high'],
            'cholesterol': ['lipid levels', 'blood lipids', 'fat levels'],
            'heart': ['cardiac', 'cardiovascular', 'cardiac health'],
            'liver': ['hepatic function', 'liver function', 'hepatic'],
            'kidney': ['renal function', 'kidney function', 'renal'],
            'blood pressure': ['bp', 'hypertension', 'pressure levels'],
            'glucose': ['blood sugar', 'sugar levels', 'gluc'],
            'hemoglobin': ['hgb', 'hb', 'blood count'],
            'creatinine': ['kidney function', 'renal function'],
            'ast': ['aspartate aminotransferase', 'liver enzyme'],
            'alt': ['alanine aminotransferase', 'liver enzyme']
        }
        
        query_lower = query.lower()
        for med_term, synonyms in medical_synonyms.items():
            if med_term in query_lower:
                for synonym in synonyms:
                    new_query = query_lower.replace(med_term, synonym)
                    variations.append(new_query.capitalize())
        
        return variations
    
    def _generate_step_back_queries(self, query: str) -> List[str]:
        """Generate broader questions related to the original query"""
        step_back_queries = []
        
        # Common patterns for step-back questions
        patterns = [
            f"What does {query} mean in medical terms?",
            f"What conditions are related to {query}?",
            f"How is {query} connected to overall health?",
            f"What are the causes of {query}?",
            f"What should I know about {query}?",
            f"How is {query} treated?",
            f"What tests are related to {query}?"
        ]
        
        step_back_queries.extend(patterns)
        return step_back_queries
    
    def _decompose_query(self, query: str) -> List[str]:
        """Break complex queries into simpler sub-questions"""
        sub_questions = []
        
        # Look for conjunctions that might indicate multiple queries
        and_pattern = re.search(r'(.+?)\s+(and|with|&)\s+(.+)', query, re.IGNORECASE)
        if and_pattern:
            part1 = and_pattern.group(1).strip()
            part2 = and_pattern.group(3).strip()
            sub_questions.append(part1)
            sub_questions.append(part2)
        
        # Look for comparison patterns
        compare_pattern = re.search(r'(.+?)\s+(vs|versus|compared to)\s+(.+)', query, re.IGNORECASE)
        if compare_pattern:
            part1 = compare_pattern.group(1).strip()
            part2 = compare_pattern.group(3).strip()
            sub_questions.append(f"What is {part1}?")
            sub_questions.append(f"What is {part2}?")
            sub_questions.append(f"How does {part1} compare to {part2}?")
        
        return sub_questions
    
    def _hybrid_search(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        """
        Perform hybrid search combining dense and sparse retrieval
        Uses the advanced techniques from advanced_rag_techniques module
        """
        from .advanced_rag_techniques import HyDEGenerator, HybridSearch
        
        dense_results = []
        sparse_results = []
        
        # Create HyDE generator to improve query representation
        hyde_gen = HyDEGenerator()
        
        # Dense retrieval with HyDE enhancement
        if self.dense_retriever and self.index:
            # Use original query
            query_embedding = self.dense_retriever.encode([query])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Generate hypothetical document and use its embedding
            hypo_doc = hyde_gen.generate_hypothetical_document(query)
            hypo_embedding = self.dense_retriever.encode([hypo_doc])
            hypo_embedding = hypo_embedding.astype('float32')
            faiss.normalize_L2(hypo_embedding)
            
            # Combine original query and hypothetical document embeddings
            combined_embedding = (query_embedding + hypo_embedding) / 2
            faiss.normalize_L2(combined_embedding)
            
            # Search with combined embedding
            scores, indices = self.index.search(combined_embedding, top_k * 2)  # Get more results for combination
            dense_results = [(idx, score) for idx, score in zip(indices[0], scores[0]) if idx != -1]
        
        # Sparse retrieval with TF-IDF
        if self.tfidf_vectorizer and self.tfidf_matrix is not None:
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            sparse_indices = similarities.argsort()[-(top_k * 2):][::-1]
            sparse_results = [(idx, similarities[idx]) for idx in sparse_indices]
        
        # Use HybridSearch class to combine results
        hybrid_search = HybridSearch()
        combined_results = hybrid_search.combine_scores(dense_results, sparse_results, top_k)
        
        # Convert to RetrievedChunk objects
        retrieved_chunks = []
        for doc_idx, score in combined_results:
            if doc_idx < len(self.chunks):
                chunk = self.chunks[doc_idx]
                retrieved_chunks.append(
                    RetrievedChunk(
                        id=chunk['id'],
                        text=chunk['text'],
                        score=score,
                        metadata=chunk.get('metadata', {}),
                        confidence=min(score * 10, 1.0),  # Normalize confidence score
                        source_document=chunk.get('doc_id', 'unknown')
                    )
                )
        
        return retrieved_chunks
    
    def _rerank_results(self, query: str, chunks: List[RetrievedChunk], top_k: int = 5) -> List[RetrievedChunk]:
        """
        Rerank the retrieved chunks using a cross-encoder model
        """
        from .advanced_rag_techniques import CrossEncoderReranker
        
        if len(chunks) == 0:
            return chunks[:top_k]
        
        # Create cross-encoder reranker
        cross_encoder_reranker = CrossEncoderReranker()
        
        # Extract texts for reranking
        chunk_texts = [chunk.text for chunk in chunks]
        
        # Rerank using cross-encoder
        reranked_pairs = cross_encoder_reranker.rerank(query, chunk_texts, top_k=len(chunks))
        
        # Create new chunk list with reranked scores
        reranked_chunks = []
        for text, score in reranked_pairs:
            # Find the original chunk that matches this text
            original_chunk = next((chunk for chunk in chunks if chunk.text == text), None)
            if original_chunk:
                new_chunk = RetrievedChunk(
                    id=original_chunk.id,
                    text=original_chunk.text,
                    score=score,
                    metadata=original_chunk.metadata,
                    confidence=min(score / 10.0, 1.0),  # Normalize confidence
                    source_document=original_chunk.source_document
                )
                reranked_chunks.append(new_chunk)
        
        # If for some reason we couldn't match all reranked items, return what we have
        return reranked_chunks[:top_k]
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """
        Main retrieval method that combines all techniques:
        1. Query transformation
        2. HyDE
        3. Hybrid search
        4. Reranking
        """
        logger.info(f"Processing query: {query}")
        
        # Transform the query into multiple variations
        transformed_queries = self._transform_query(query)
        logger.info(f"Generated {len(transformed_queries)} query variations")
        
        # Retrieve using hybrid search for each transformed query
        all_retrieved_chunks = []
        
        for trans_query in transformed_queries:
            chunks = self._hybrid_search(trans_query, top_k=top_k)
            all_retrieved_chunks.extend(chunks)
        
        # Remove duplicates based on text content
        unique_chunks = []
        seen_texts = set()
        seen_ids = set()  # Also prevent duplicate IDs
        for chunk in all_retrieved_chunks:
            if chunk.text not in seen_texts and chunk.id not in seen_ids:
                unique_chunks.append(chunk)
                seen_texts.add(chunk.text)
                seen_ids.add(chunk.id)
        
        # Rerank the results
        reranked_chunks = self._rerank_results(query, unique_chunks, top_k=top_k)
        
        logger.info(f"Retrieved and reranked {len(reranked_chunks)} chunks")
        return reranked_chunks
    
    async def answer_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Answer a query using the RAG approach
        """
        start_time = datetime.now()
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(query, top_k=top_k)
        
        # Generate answer using the retrieved chunks and a language model
        # For now, we'll return the retrieved chunks with basic formatting
        # In a real implementation, this would call an LLM to generate a comprehensive answer
        
        # Create context from retrieved chunks
        context = "\n\n".join([f"Document {i+1}: {chunk.text}" for i, chunk in enumerate(retrieved_chunks)])
        
        # Prepare the response with detailed citation information
        response = {
            "query": query,
            "answer": self._generate_answer_from_context(query, context, retrieved_chunks),
            "retrieved_chunks": [
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "score": chunk.score,
                    "confidence": float(chunk.confidence),
                    "source_document": chunk.source_document,
                    "metadata": chunk.metadata,
                    "page_number": chunk.page_number
                }
                for chunk in retrieved_chunks
            ],
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "citations": [
                {
                    "id": chunk.id,
                    "source_document": chunk.source_document,
                    "confidence": float(chunk.confidence),
                    "text_preview": chunk.text[:100] + "..."
                }
                for chunk in retrieved_chunks
            ],
            "total_sources": len(retrieved_chunks),
            "avg_confidence": sum(c.confidence for c in retrieved_chunks) / len(retrieved_chunks) if retrieved_chunks else 0,
            "query_transformations_applied": len(transformed_queries) if 'transformed_queries' in locals() else 0
        }
        
        return response
    
    def _generate_answer_from_context(self, query: str, context: str, chunks: List[RetrievedChunk]) -> str:
        """
        Generate an answer based on the query and retrieved context
        This is a simplified version - in practice, you'd use a medical LLM
        """
        if not context.strip():
            return "I couldn't find relevant information to answer your query. Please try rephrasing or provide more details."
        
        # Simple answer generation based on context
        # In a real implementation, this would use an LLM like GPT or a medical-specific model
        answer = f"Based on the medical documents I found:\n\n{context[:500]}..."
        
        # Add confidence indicator
        if chunks:
            avg_confidence = sum(c.confidence for c in chunks) / len(chunks)
            if avg_confidence > 0.7:
                confidence_text = "high"
            elif avg_confidence > 0.4:
                confidence_text = "moderate"
            else:
                confidence_text = "low"
            
            answer += f"\n\nThe information has {confidence_text} confidence based on the retrieved documents."
        
        # Add citation information
        if chunks:
            answer += f"\n\nCitations ({len(chunks)} sources):"
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 citations
                source_id = chunk.source_document
                if source_id.startswith('report_'):
                    source_desc = f"Health report {source_id.replace('report_', '')}"
                else:
                    source_desc = f"Document {source_id}"
                
                answer += f"\n  {i+1}. {source_desc} (confidence: {chunk.confidence:.2f})"
            
            if len(chunks) > 3:
                answer += f"\n  ... and {len(chunks) - 3} more sources"
        
        # Add disclaimer about medical information
        answer += f"\n\nImportant: This is an AI-generated response based on health records. It is not a substitute for professional medical advice. Please consult with your doctor for medical decisions."
        
        return answer


# Health report specific RAG class
class HealthReportRAG(MedicalRAGChatbot):
    """
    Specialized RAG chatbot for health reports with additional features:
    - Patient data integration
    - Lab value understanding
    - Medical terminology translation
    """
    
    def __init__(self):
        super().__init__()
        
    def add_health_report(self, report_data: Dict[str, Any]):
        """
        Add a health report to the RAG system
        report_data should include: original_text, extracted_parameters, patient_info, etc.
        """
        # Create document from health report
        doc_id = report_data.get('job_id', str(uuid.uuid4()))
        
        # Combine different parts of the report
        original_texts = report_data.get('original_texts', [])
        if isinstance(original_texts, str):
            original_texts = [original_texts]
        
        full_text = " ".join(original_texts)
        
        # Add lab parameters to the text for better retrieval
        parameters = report_data.get('extracted_parameters', [])
        if parameters:
            params_text = " ".join([
                f"{param.get('field', '')} value {param.get('value', '')} {param.get('units', '')}"
                for param in parameters
            ])
            full_text += f" Lab parameters: {params_text}"
        
        # Add patient info
        patient_info = report_data.get('patient_info', {})
        if patient_info:
            patient_text = f"Patient information: {json.dumps(patient_info)}"
            full_text += f" {patient_text}"
        
        # Create metadata
        metadata = {
            'job_id': doc_id,
            'report_date': report_data.get('created_at', datetime.now().isoformat()),
            'language': report_data.get('language', 'en'),
            'parameters': parameters,
            'patient_info': patient_info
        }
        
        # Add to RAG system
        doc = {
            'id': doc_id,
            'text': full_text,
            'metadata': metadata
        }
        
        self.add_documents([doc])