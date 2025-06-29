"""
Embedding and retrieval service for OrgGPT.

This module handles text embeddings, similarity search, and document retrieval
using locally downloaded Hugging Face models.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle
from pathlib import Path
from tqdm import tqdm

from ..models.schemas import DocumentChunk, DocumentSearchResult
from ..core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for text embeddings and similarity search using local models."""
    
    def __init__(self):
        """Initialize the embedding service with local models."""
        self.embedding_model_name = settings.EMBEDDING_MODEL
        self.reranker_model_name = settings.RERANKER_MODEL
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self._initialize_models()
        
        # In-memory storage for embeddings (in production, use a vector database)
        self.embeddings_store: Dict[str, np.ndarray] = {}
        self.chunks_store: Dict[str, DocumentChunk] = {}
        
        # Cache for embeddings to avoid recomputation
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Load cached embeddings if available
        self._load_cache()
    
    def _initialize_models(self):
        """Initialize the embedding and reranking models locally."""
        try:
            # Initialize embedding model using sentence-transformers
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=self.device)
            
            # Initialize reranker model
            self.reranker_model = CrossEncoder(self.reranker_model_name, device=self.device)
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            # Fallback to basic models if specified models fail
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
                self.reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)
            except Exception as fallback_error:
                logger.error(f"Fallback model loading failed: {str(fallback_error)}")
                raise RuntimeError("Failed to load both primary and fallback models")
    
    def _load_cache(self):
        """Load cached embeddings from disk."""
        cache_path = Path("cache/embeddings_cache.pkl")
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load embedding cache: {str(e)}")
                self.embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / "embeddings_cache.pkl"
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {str(e)}")
    
    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for a list of texts using local model.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            embeddings = []
            texts_to_embed = []
            cached_embeddings = []
            
            # Check cache for existing embeddings
            for text in texts:
                text_hash = str(hash(text))
                if text_hash in self.embedding_cache:
                    cached_embeddings.append((len(embeddings), self.embedding_cache[text_hash]))
                    embeddings.append(None)  # Placeholder
                else:
                    texts_to_embed.append((len(embeddings), text))
                    embeddings.append(None)  # Placeholder
            
            # Generate embeddings for non-cached texts
            if texts_to_embed:
                texts_only = [text for _, text in texts_to_embed]
                
                # Generate embeddings using sentence-transformers
                new_embeddings = self.embedding_model.encode(
                    texts_only,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                # Store new embeddings in cache and result list
                for (idx, text), embedding in zip(texts_to_embed, new_embeddings):
                    text_hash = str(hash(text))
                    self.embedding_cache[text_hash] = embedding
                    embeddings[idx] = embedding
            
            # Fill in cached embeddings
            for idx, cached_embedding in cached_embeddings:
                embeddings[idx] = cached_embedding
            
            # Save cache periodically
            if len(texts_to_embed) > 0:
                self._save_cache()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            # Return zero vectors as fallback
            embedding_dim = getattr(self.embedding_model, 'get_sentence_embedding_dimension', lambda: 384)()
            return [np.zeros(embedding_dim) for _ in texts]
    
    def store_document_embeddings(self, chunks: List[DocumentChunk]):
        """
        Generate and store embeddings for document chunks.
        
        Args:
            chunks: List of document chunks to embed
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return
        
        try:
            # Extract text content
            texts = [chunk.content for chunk in chunks]
            
            # Get embeddings
            embeddings = self.get_embeddings(texts)
            
            # Store embeddings and chunks
            for chunk, embedding in tqdm(zip(chunks, embeddings)):
                if embedding is not None:
                    self.embeddings_store[chunk.id] = embedding
                    self.chunks_store[chunk.id] = chunk
                else:
                    logger.warning(f"Failed to generate embedding for chunk {chunk.id}")
                
        except Exception as e:
            logger.error(f"Error storing document embeddings: {str(e)}")
            raise
    
    def search_similar_chunks(
        self, 
        query: str, 
        session_id: str, 
        top_k: int = None
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """
        Search for similar chunks to a query.
        
        Args:
            query: Search query
            session_id: Session ID to filter chunks
            top_k: Number of top results to return
            
        Returns:
            Tuple[List[DocumentChunk], List[float]]: Similar chunks and their scores
        """
        if top_k is None:
            top_k = getattr(settings, 'TOP_K_RETRIEVAL', 5)
        
        if not query.strip():
            logger.warning("Empty query provided")
            return [], []
        
        try:
            # Get query embedding
            query_embeddings = self.get_embeddings([query])
            if not query_embeddings or query_embeddings[0] is None:
                logger.error("Failed to generate query embedding")
                return [], []
            
            query_embedding = query_embeddings[0]
            
            # Filter chunks by session
            session_doc_ids = self._get_session_documents(session_id)
            session_chunks = [
                (chunk_id, chunk) 
                for chunk_id, chunk in self.chunks_store.items() 
                if chunk.document_id in session_doc_ids and chunk_id in self.embeddings_store
            ]
            
            if not session_chunks:
                return [], []
            
            
            # Calculate similarities
            similarities = []
            chunks = []
            
            for chunk_id, chunk in session_chunks:
                chunk_embedding = self.embeddings_store[chunk_id]
                
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    chunk_embedding.reshape(1, -1)
                )[0][0]
                
                similarities.append(float(similarity))
                chunks.append(chunk)
            
            # Sort by similarity and get top-k
            sorted_pairs = sorted(
                zip(chunks, similarities), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            top_chunks = [pair[0] for pair in sorted_pairs[:top_k]]
            top_scores = [pair[1] for pair in sorted_pairs[:top_k]]
            
            return top_chunks, top_scores
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return [], []
    
    def rerank_chunks(
        self, 
        query: str, 
        chunks: List[DocumentChunk], 
        scores: List[float]
    ) -> Tuple[List[DocumentChunk], List[float]]:
        """
        Rerank chunks using a local reranker model.
        
        Args:
            query: Search query
            chunks: List of chunks to rerank
            scores: Original similarity scores
            
        Returns:
            Tuple[List[DocumentChunk], List[float]]: Reranked chunks and scores (normalized 0-1)
        """
        try:
            if not chunks or not query.strip():
                return chunks, scores
            
            # Prepare pairs for reranking
            pairs = [[query, chunk.content] for chunk in chunks]
            
            # Get rerank scores using local model
            rerank_scores = self.reranker_model.predict(pairs)
            
            # Convert to list if numpy array
            if hasattr(rerank_scores, 'tolist'):
                rerank_scores = rerank_scores.tolist()
            
            # Ensure scores are floats
            rerank_scores = [float(score) for score in rerank_scores]
            
            # Normalize scores to 0-1 range using sigmoid function
            # This converts unbounded logits to probabilities
            import math
            normalized_scores = [1 / (1 + math.exp(-score)) for score in rerank_scores]
            
            # Sort by normalized rerank scores
            sorted_pairs = sorted(
                zip(chunks, normalized_scores), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            reranked_chunks = [pair[0] for pair in sorted_pairs]
            reranked_scores = [pair[1] for pair in sorted_pairs]

            return reranked_chunks, reranked_scores
            
        except Exception as e:
            logger.error(f"Error reranking chunks: {str(e)}")
            # Return original order on error
            return chunks, scores
    
    def search_documents(
        self, 
        query: str, 
        session_id: str, 
        limit: int = 10
    ) -> List[DocumentSearchResult]:
        """
        Search for documents containing the query.
        
        Args:
            query: Search query
            session_id: Session ID to filter documents
            limit: Maximum number of results
            
        Returns:
            List[DocumentSearchResult]: Search results
        """
        try:
            # Get similar chunks
            chunks, scores = self.search_similar_chunks(query, session_id, limit)
            
            if not chunks:
                return []
            
            # Optional: Rerank chunks for better results
            if len(chunks) > 1:
                chunks, scores = self.rerank_chunks(query, chunks, scores)
            
            # Convert to search results
            results = []
            for chunk, score in zip(chunks, scores):
                # Get document metadata
                document_filename = self._get_document_filename(chunk.document_id)
                
                # Truncate content for preview
                content_preview = chunk.content
                if len(content_preview) > 500:
                    content_preview = content_preview[:500] + "..."
                
                result = DocumentSearchResult(
                    document_id=chunk.document_id,
                    filename=document_filename,
                    chunk_content=content_preview,
                    similarity_score=float(score),
                    chunk_index=chunk.chunk_index
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def delete_session_embeddings(self, session_id: str):
        """
        Delete all embeddings for a session.
        
        Args:
            session_id: Session ID
        """
        try:
            # Get document IDs for the session
            session_doc_ids = self._get_session_documents(session_id)
            
            # Remove embeddings and chunks for session documents
            chunk_ids_to_remove = [
                chunk_id for chunk_id, chunk in self.chunks_store.items()
                if chunk.document_id in session_doc_ids
            ]
            
            removed_count = 0
            for chunk_id in chunk_ids_to_remove:
                if chunk_id in self.embeddings_store:
                    self.embeddings_store.pop(chunk_id, None)
                    removed_count += 1
                self.chunks_store.pop(chunk_id, None)
                        
        except Exception as e:
            logger.error(f"Error deleting session embeddings: {str(e)}")
    
    def _get_session_documents(self, session_id: str) -> List[str]:
        """
        Get document IDs for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List[str]: List of document IDs
        """
        try:
            # This would query the database in production
            # For now, return all documents that have chunks
            # You might want to implement proper session-document mapping
            session_doc_ids = set()
            for chunk in self.chunks_store.values():
                # Assuming document_id contains session info or we have a mapping
                # You might need to modify this logic based on your document ID structure
                session_doc_ids.add(chunk.document_id)
            
            return list(session_doc_ids)
        except Exception as e:
            logger.error(f"Error getting session documents: {str(e)}")
            return []
    
    def _get_document_filename(self, document_id: str) -> str:
        """
        Get filename for a document ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            str: Document filename
        """
        try:
            # This would query the database in production
            # For now, return a more descriptive placeholder
            return f"document_{document_id[:8]}.pdf"
        except Exception as e:
            logger.error(f"Error getting document filename: {str(e)}")
            return "unknown_document.pdf"
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored embeddings.
        
        Returns:
            Dict[str, Any]: Statistics about embeddings
        """
        return {
            "total_embeddings": len(self.embeddings_store),
            "total_chunks": len(self.chunks_store),
            "cached_embeddings": len(self.embedding_cache),
            "embedding_dimension": self.embeddings_store[list(self.embeddings_store.keys())[0]].shape[0] if self.embeddings_store else 0,
            "device": str(self.device),
            "embedding_model": self.embedding_model_name,
            "reranker_model": self.reranker_model_name
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def __del__(self):
        """Cleanup method to save cache on destruction."""
        try:
            if hasattr(self, 'embedding_cache') and self.embedding_cache:
                self._save_cache()
        except Exception as e:
            logger.error(f"Error saving cache during cleanup: {str(e)}")