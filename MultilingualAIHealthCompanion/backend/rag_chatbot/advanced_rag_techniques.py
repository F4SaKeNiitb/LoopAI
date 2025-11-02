"""
Advanced RAG Techniques Implementation
Contains HyDE, Query Transformation, Cross-Encoder Reranking, and other advanced methods
"""
import logging
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
import re


logger = logging.getLogger(__name__)


class HyDEGenerator:
    """
    Hypothetical Document Embeddings Generator
    Generates hypothetical answers to queries before retrieval to improve matching
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
        try:
            from sentence_transformers import SentenceTransformer
            # Use CPU for memory efficiency on M1 Mac
            self.sentence_transformer = SentenceTransformer(model_name, device='cpu')
        except Exception as e:
            logger.warning(f"Could not load {model_name}, using fallback: {e}")
            self.sentence_transformer = None
    
    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that answers the query
        """
        # Create a template-based hypothetical document
        # In a production system, this would use an LLM to generate a detailed answer
        template = f"""
        Question: {query}
        
        Answer:
        """
        
        # For now, we'll create a structured response based on common medical query patterns
        query_lower = query.lower()
        
        # Medical-specific transformations
        if any(term in query_lower for term in ['glucose', 'diabetes', 'blood sugar']):
            return f"The patient's glucose levels show values related to diabetes management. Normal fasting glucose is 70-99 mg/dL. The patient's results indicate {query.lower()} and may require dietary changes or medication."
        elif any(term in query_lower for term in ['cholesterol', 'lipid', 'cardio']):
            return f"Lipid panel results for cholesterol assessment. Total cholesterol, LDL, HDL, and triglycerides are evaluated. High cholesterol levels ({query.lower()}) may indicate cardiovascular risk factors."
        elif any(term in query_lower for term in ['blood pressure', 'hypertension']):
            return f"Blood pressure measurements showing systolic and diastolic values. Hypertension is defined as readings consistently above 140/90 mmHg. The patient's readings of {query.lower()} require monitoring and potential intervention."
        elif any(term in query_lower for term in ['liver', 'ast', 'alt', 'bilirubin']):
            return f"Liver function tests showing values for AST, ALT, and bilirubin. The results for {query.lower()} indicate hepatic function status and potential liver pathology."
        elif any(term in query_lower for term in ['kidney', 'creatinine', 'bun']):
            return f"Renal function assessment with creatinine and BUN values. The results for {query.lower()} indicate kidney function and potential renal impairment."
        else:
            # General pattern for unknown queries
            return f"Medical report containing information related to '{query}'. This document addresses clinical findings, test values, and reference ranges relevant to the mentioned condition."
    
    def embed_hypothetical_document(self, query: str) -> np.ndarray:
        """
        Generate embedding for the hypothetical document
        """
        if self.sentence_transformer:
            hypo_doc = self.generate_hypothetical_document(query)
            embedding = self.sentence_transformer.encode([hypo_doc])
            return embedding[0]  # Return first embedding as numpy array
        else:
            # Fallback: return simple embedding based on query
            return np.random.rand(384).astype(np.float32)  # Typical embedding size


class QueryTransformer:
    """
    Query transformation techniques:
    - Multi-query generation
    - Step-back prompting
    - Query decomposition
    """
    
    def __init__(self):
        self.medical_synonyms = {
            'glucose': ['blood sugar', 'sugar levels', 'gluc'],
            'cholesterol': ['lipid panel', 'lipid levels', 'blood lipids'],
            'diabetes': ['diabetic condition', 'sugar diabetes'],
            'blood pressure': ['bp', 'hypertension', 'pressure levels'],
            'hemoglobin': ['hgb', 'hb', 'blood count'],
            'creatinine': ['kidney function', 'renal function'],
            'ast': ['aspartate aminotransferase', 'liver enzyme'],
            'alt': ['alanine aminotransferase', 'liver enzyme'],
            'bilirubin': ['liver function', 'jaundice'],
            'bun': ['blood urea nitrogen', 'kidney function'],
            'albumin': ['protein levels', 'liver function'],
            'calcium': ['ca', 'mineral levels'],
            'sodium': ['na', 'electrolyte'],
            'potassium': ['k', 'electrolyte'],
            'chloride': ['cl', 'electrolyte'],
            'co2': ['carbon dioxide', 'bicarbonate'],
            'phosphorus': ['phos', 'mineral levels'],
            'magnesium': ['mg', 'mineral levels'],
            'heart': ['cardiac', 'cardiovascular'],
            'liver': ['hepatic', 'hepatic function'],
            'kidney': ['renal', 'renal function'],
            'inflammation': ['inflammatory markers', 'crp', 'esr'],
            'infection': ['infectious disease', 'wbc', 'white blood cells']
        }
    
    def generate_multi_queries(self, query: str) -> List[str]:
        """
        Generate multiple variations of the query using medical synonyms
        """
        queries = [query]
        
        query_lower = query.lower()
        
        # Generate synonym-based queries
        for med_term, synonyms in self.medical_synonyms.items():
            if med_term in query_lower:
                for synonym in synonyms:
                    new_query = query_lower.replace(med_term, synonym)
                    queries.append(new_query.capitalize())
        
        # Remove duplicates while preserving order
        unique_queries = []
        for q in queries:
            if q not in unique_queries:
                unique_queries.append(q)
        
        return unique_queries
    
    def generate_step_back_queries(self, query: str) -> List[str]:
        """
        Generate broader, more general questions related to the original query
        """
        step_back_queries = [
            f"What is the medical significance of {query}?",
            f"How does {query} relate to overall health?",
            f"What conditions are associated with {query}?",
            f"What does {query} indicate about patient health?",
            f"Why would a doctor check for {query}?",
            f"What are the causes of {query}?",
            f"What are the treatments for {query}?",
            f"How is {query} diagnosed?",
            f"What tests are related to {query}?",
            f"What should patients know about {query}?"
        ]
        
        return step_back_queries
    
    def decompose_query(self, query: str) -> List[str]:
        """
        Break complex queries into simpler sub-questions
        """
        sub_questions = []
        
        # Pattern: "What is A and B?" -> "What is A?", "What is B?"
        and_pattern = re.search(r'(what is|what are|explain)\s+(.+?)\s+(and|with|&)\s+(.+)', query, re.IGNORECASE)
        if and_pattern:
            phrase = and_pattern.group(1).capitalize()
            part1 = and_pattern.group(2).strip()
            part2 = and_pattern.group(4).strip()
            sub_questions.append(f"{phrase} {part1}?")
            sub_questions.append(f"{phrase} {part2}?")
        
        # Pattern: "Compare A vs B" -> "What is A?", "What is B?", "How do A and B differ?"
        compare_pattern = re.search(r'(.+?)\s+(vs|versus|compared to|vs\.)\s+(.+)', query, re.IGNORECASE)
        if compare_pattern:
            part1 = compare_pattern.group(1).strip()
            part2 = compare_pattern.group(3).strip()
            sub_questions.append(f"What is {part1}?")
            sub_questions.append(f"What is {part2}?")
            sub_questions.append(f"How does {part1} compare to {part2}?")
        
        # Pattern: "A and B levels" -> "What are A levels?", "What are B levels?"
        levels_pattern = re.findall(r'(\w+)\s+(levels|values|readings)', query, re.IGNORECASE)
        if len(levels_pattern) > 1:
            for term, _ in levels_pattern:
                sub_questions.append(f"What are {term} {levels_pattern[0][1]}?")
        
        # If original query is long, also try to break it into multiple concepts
        if len(query.split()) > 5:
            words = query.split()
            mid_point = len(words) // 2
            first_half = " ".join(words[:mid_point])
            second_half = " ".join(words[mid_point:])
            
            if len(first_half) > 3:  # Avoid very short fragments
                sub_questions.append(f"What is {first_half}?")
            if len(second_half) > 3:
                sub_questions.append(f"What about {second_half}?")
        
        return sub_questions
    
    def transform_query(self, query: str) -> List[str]:
        """
        Apply all transformation techniques to the query
        """
        all_queries = [query]
        
        # Add multi-queries
        multi_queries = self.generate_multi_queries(query)
        all_queries.extend(multi_queries)
        
        # Add step-back queries
        step_back_queries = self.generate_step_back_queries(query)
        all_queries.extend(step_back_queries)
        
        # Add decomposed queries
        decomposed_queries = self.decompose_query(query)
        all_queries.extend(decomposed_queries)
        
        # Remove duplicates while preserving order
        unique_queries = []
        for q in all_queries:
            if q not in unique_queries and q != query:  # Don't include original query twice
                unique_queries.append(q)
        
        # Limit to reasonable number to avoid too many queries
        return unique_queries[:20]  # Limit to 20 transformed queries


class CrossEncoderReranker:
    """
    Cross-encoder for re-ranking retrieved documents
    """
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.model_name = model_name
        self.cross_encoder = None
        
        try:
            # Use CPU for memory efficiency on M1 Mac
            self.cross_encoder = CrossEncoder(model_name, device='cpu')
            logger.info(f"Loaded cross-encoder model: {model_name}")
        except Exception as e:
            logger.error(f"Could not load cross-encoder model {model_name}: {e}")
            # Fallback to not using cross-encoder
            self.cross_encoder = None
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Rerank documents based on query relevance using cross-encoder
        Returns list of (document, score) tuples sorted by score
        """
        if not self.cross_encoder or not documents:
            # Fallback: return documents with basic scores
            return [(doc, 0.5) for doc in documents[:top_k]]
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get scores from cross-encoder
        scores = self.cross_encoder.predict(pairs)
        
        # Convert to list of (document, score) tuples and sort by score
        results = [(doc, float(score)) for doc, score in zip(documents, scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


class HybridSearch:
    """
    Combines dense and sparse search methods for better retrieval
    """
    
    def __init__(self):
        self.dense_weight = 0.7  # Weight for dense retrieval
        self.sparse_weight = 0.3  # Weight for sparse retrieval
    
    def combine_scores(self, dense_scores: List[Tuple[int, float]], 
                      sparse_scores: List[Tuple[int, float]], 
                      k: int = 10) -> List[Tuple[int, float]]:
        """
        Combine dense and sparse scores using weighted combination
        Uses Reciprocal Rank Fusion (RRF) method
        """
        # Create a dictionary to accumulate scores for each document
        combined_scores = {}
        
        # Process dense scores
        for rank, (doc_id, score) in enumerate(dense_scores):
            # Use RRF: score = 1 / (rank + 1), where rank starts from 1
            rrf_score = 1.0 / (rank + 1)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + self.dense_weight * rrf_score
        
        # Process sparse scores
        for rank, (doc_id, score) in enumerate(sparse_scores):
            # Use RRF: score = 1 / (rank + 1), where rank starts from 1
            rrf_score = 1.0 / (rank + 1)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + self.sparse_weight * rrf_score
        
        # Sort by combined score and return top k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]


class ParentChildChunker:
    """
    Implements parent-child chunking for better context retrieval
    """
    
    def __init__(self, child_chunk_size: int = 128, parent_chunk_size: int = 512, overlap: int = 30):
        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size
        self.overlap = overlap
    
    def create_parent_child_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create both parent and child chunks from documents
        Returns list of chunks with parent-child relationships
        """
        all_chunks = []
        
        for doc in documents:
            doc_id = doc.get('id', 'unknown')
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            
            # First, create parent chunks
            parent_chunks = self._create_parent_chunks(text, doc_id)
            
            for parent_chunk in parent_chunks:
                # Create child chunks from each parent chunk
                child_chunks = self._create_child_chunks(parent_chunk['text'], parent_chunk['id'])
                
                # Add parent chunk with children info
                parent_chunk['children'] = child_chunks
                parent_chunk['chunk_type'] = 'parent'
                parent_chunk['metadata'] = metadata
                all_chunks.append(parent_chunk)
                
                # Add child chunks
                for child_chunk in child_chunks:
                    child_chunk['parent_id'] = parent_chunk['id']
                    child_chunk['chunk_type'] = 'child'
                    child_chunk['metadata'] = metadata
                    all_chunks.append(child_chunk)
        
        return all_chunks
    
    def _create_parent_chunks(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Create parent-level chunks"""
        words = text.split()
        chunks = []
        start_idx = 0
        chunk_idx = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.parent_chunk_size, len(words))
            chunk_text = ' '.join(words[start_idx:end_idx])
            
            chunk = {
                'id': f"{doc_id}_parent_{chunk_idx}",
                'text': chunk_text,
                'start_idx': start_idx,
                'end_idx': end_idx
            }
            
            chunks.append(chunk)
            start_idx = end_idx - self.overlap
            chunk_idx += 1
        
        return chunks
    
    def _create_child_chunks(self, parent_text: str, parent_id: str) -> List[Dict[str, Any]]:
        """Create child-level chunks from parent text"""
        words = parent_text.split()
        chunks = []
        start_idx = 0
        chunk_idx = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.child_chunk_size, len(words))
            chunk_text = ' '.join(words[start_idx:end_idx])
            
            chunk = {
                'id': f"{parent_id}_child_{chunk_idx}",
                'text': chunk_text,
                'start_idx': start_idx,
                'end_idx': end_idx
            }
            
            chunks.append(chunk)
            start_idx = end_idx - self.overlap
            chunk_idx += 1
        
        return chunks
    
    def retrieve_with_context(self, child_results: List[Dict[str, Any]], 
                            all_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        For each retrieved child chunk, also retrieve its parent for context
        """
        enhanced_results = []
        
        # Create a map of chunk id to chunk for quick lookup
        chunk_map = {chunk['id']: chunk for chunk in all_chunks}
        
        for child_result in child_results:
            child_chunk = child_result.get('chunk', child_result)
            
            # Find the parent of this child
            parent_id = child_chunk.get('parent_id')
            if parent_id:
                parent_chunk = chunk_map.get(parent_id)
                if parent_chunk:
                    # Add parent context to the result
                    enhanced_result = {
                        'child_chunk': child_chunk,
                        'parent_chunk': parent_chunk,
                        'text_with_context': parent_chunk['text'],  # Use parent for broader context
                        'original_score': child_result.get('score', 0),
                        'metadata': child_chunk.get('metadata', {})
                    }
                    enhanced_results.append(enhanced_result)
            else:
                # If no parent, use the chunk as is
                enhanced_results.append({
                    'child_chunk': child_chunk,
                    'parent_chunk': None,
                    'text_with_context': child_chunk['text'],
                    'original_score': child_result.get('score', 0),
                    'metadata': child_chunk.get('metadata', {})
                })
        
        return enhanced_results