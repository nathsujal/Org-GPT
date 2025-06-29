"""
API routes for OrgGPT application.

This module defines all the REST API endpoints for the chatbot functionality.
"""

import uuid
import logging
import numpy as np
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse

from ..models.schemas import (
    QueryRequest, ChatResponse, SessionInfo, FileUploadResponse,
    DocumentSearchRequest, DocumentSearchResult, ErrorResponse,
    ChatMessage, QueryMetadata, DocumentMetadata
)
from ..services.session_service import SessionService
from ..services.document_service import DocumentService
from ..services.embedding_service import EmbeddingService
from ..services.llm_service import LLMService
from ..core.config import settings

# Initialize services
session_service = SessionService()
document_service = DocumentService()
embedding_service = EmbeddingService()
llm_service = LLMService()

# Create router
router = APIRouter()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.post("/sessions", response_model=SessionInfo)
async def create_session():
    """
    Create a new chat session.
    
    Returns:
        SessionInfo: New session information
    """
    try:
        session_id = session_service.create_session()
        session_info = session_service.get_session(session_id)
        return session_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """
    Get session information.
    
    Args:
        session_id: Session ID
        
    Returns:
        SessionInfo: Session information
    """
    if not session_service.is_session_valid(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    session_info = session_service.get_session(session_id)
    return session_info


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and all associated data.
    
    Args:
        session_id: Session ID
    """
    try:
        # Clean up embeddings
        embedding_service.delete_session_embeddings(session_id)
        
        # Clean up files
        document_service.delete_document_files(session_id)
        
        # Delete session
        session_service.delete_session(session_id)
        
        return {"message": "Session deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@router.post("/sessions/{session_id}/upload", response_model=FileUploadResponse)
async def upload_file(
    session_id: str,
    file: UploadFile = File(...),
    url: Optional[str] = Form(None)
):
    """
    Upload a file or process a URL for the session.
    
    Args:
        session_id: Session ID
        file: Uploaded file (optional if URL provided)
        url: URL to process (optional if file provided)
        
    Returns:
        FileUploadResponse: Upload result
    """
    if not session_service.is_session_valid(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if not session_service.can_upload_file(session_id):
        raise HTTPException(
            status_code=400, 
            detail=f"Maximum number of files ({settings.MAX_FILES_PER_SESSION}) reached for this session"
        )
    
    try:
        if file and file.filename:
            # Process uploaded file
            file_content = await file.read()
            file_size = len(file_content)
            
            # Validate file
            if not document_service.validate_file(file.filename, file_size):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid file type or size exceeds limit"
                )
            
            # Save file
            file_path = document_service.save_uploaded_file(
                file_content, file.filename, session_id
            )
            
            # Extract text
            doc_type = document_service.get_document_type(file.filename)
            chunks = document_service.extract_text_from_file(file_path, doc_type)
            
            # Create document metadata
            doc_metadata = document_service.create_document_metadata(
                file.filename, file_size, session_id, len(chunks)
            )
            
            filename = file.filename
            
        elif url:
            # Process URL
            chunks = document_service.process_url(url)
            
            # Create document metadata for URL
            doc_metadata = document_service.create_document_metadata(
                url, 0, session_id, len(chunks)
            )
            doc_metadata.file_type = "url"
            
            filename = url
            
        else:
            raise HTTPException(status_code=400, detail="Either file or URL must be provided")
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text content could be extracted")
        
        # Create document chunks
        chunk_objects = document_service.create_document_chunks(doc_metadata.id, chunks)
        
        # Store in session
        session_service.add_document(session_id, doc_metadata)
        session_service.add_document_chunks(session_id, chunk_objects)
        
        # Generate and store embeddings
        embedding_service.store_document_embeddings(chunk_objects)
        
        return FileUploadResponse(
            document_id=doc_metadata.id,
            filename=filename,
            file_type=doc_metadata.file_type,
            file_size=doc_metadata.file_size,
            chunk_count=len(chunks),
            message="File uploaded and processed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@router.get("/sessions/{session_id}/documents", response_model=List[DocumentMetadata])
async def get_session_documents(session_id: str):
    """
    Get all documents for a session.
    
    Args:
        session_id: Session ID
        
    Returns:
        List[DocumentMetadata]: List of document metadata
    """
    if not session_service.is_session_valid(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    documents = session_service.get_session_documents(session_id)
    return documents


@router.delete("/sessions/{session_id}/documents/{document_id}")
async def delete_document(session_id: str, document_id: str):
    """
    Delete a document from the session.
    
    Args:
        session_id: Session ID
        document_id: Document ID to delete
    """
    if not session_service.is_session_valid(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    try:
        # Remove from session
        removed = session_service.remove_document(session_id, document_id)
        
        if not removed:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Clean up embeddings for this document
        # (This would be more sophisticated in production)
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@router.post("/sessions/{session_id}/search", response_model=List[DocumentSearchResult])
async def search_documents(session_id: str, request: DocumentSearchRequest):
    """
    Search documents in the session.
    
    Args:
        session_id: Session ID
        request: Search request
        
    Returns:
        List[DocumentSearchResult]: Search results
    """
    if not session_service.is_session_valid(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    try:
        results = embedding_service.search_documents(
            request.query, session_id, request.limit
        )
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, request: QueryRequest):
    """
    Process a chat query and generate response.
    
    Args:
        session_id: Session ID
        request: Chat query request
        
    Returns:
        ChatResponse: Generated response with metadata
    """
    if not session_service.is_session_valid(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    try:
        # Handle regeneration
        if request.regenerate and request.message_id:
            # Get stored metadata for regeneration
            stored_metadata = session_service.get_message_metadata(request.message_id)
            if not stored_metadata:
                raise HTTPException(status_code=404, detail="Message metadata not found")
            
            # Use stored metadata to regenerate
            if stored_metadata.response_type == "RAG":
                response_content = llm_service.generate_rag_response(
                    stored_metadata.query,
                    stored_metadata.rewritten_query,
                    stored_metadata.context_chunks,
                    stored_metadata.chat_history,
                    stored_metadata.intent
                )
            else:
                response_content = llm_service.generate_general_response(
                    stored_metadata.query,
                    stored_metadata.chat_history,
                    stored_metadata.intent
                )
            
            # Create new message
            message_id = str(uuid.uuid4())
            
            # Update timestamp in metadata
            stored_metadata.timestamp = datetime.utcnow()
            
            return ChatResponse(
                message_id=message_id,
                content=response_content,
                citations=[],  # Would extract citations in production
                metadata=stored_metadata
            )
        
        # Normal query processing
        # Get chat history
        chat_history = session_service.get_chat_history(session_id, limit=10)
        
        # Add user message to history
        user_message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role="user",
            content=request.query,
            timestamp=datetime.utcnow()
        )
        session_service.add_message(session_id, user_message)
        
        # Rewrite query
        rewritten_query = llm_service.rewrite_query(request.query, chat_history)
        
        # Retrieve similar chunks
        similar_chunks, similarity_scores = embedding_service.search_similar_chunks(
            rewritten_query, session_id
        )

        # Rerank chunks
        if similar_chunks:
            reranked_chunks, reranked_scores = embedding_service.rerank_chunks(
                rewritten_query, similar_chunks, similarity_scores
            )
        else:
            reranked_chunks, reranked_scores = [], []

        context_chunks = [chunk.content for chunk in reranked_chunks[:3]]

        # Classify Intent
        intent, intent_score = llm_service.classify_intent(
            rewritten_query,
            context_chunks,
            avg_relevance=np.mean(np.array(reranked_scores)) if reranked_scores else 0.0
        )

        # Determine response type
        max_similarity = max(reranked_scores) if reranked_scores else 0.0
        use_rag = llm_service.should_use_rag(intent)

        
        # Generate response
        if use_rag and reranked_chunks:
            context_chunks = [chunk.content for chunk in reranked_chunks[:3]]
            response_content = llm_service.generate_rag_response(
                request.query, rewritten_query, context_chunks, chat_history, intent
            )
            response_type = "RAG"
        else:
            context_chunks = []
            response_content = llm_service.generate_general_response(
                request.query, chat_history, intent
            )
            response_type = "General"
        
        # Create metadata
        metadata = QueryMetadata(
            query=request.query,
            rewritten_query=rewritten_query,
            intent=intent,
            intent_score=intent_score,
            similarity_score=max_similarity,
            context_chunks=context_chunks,
            chat_history=[{"role": msg.role, "content": msg.content} for msg in chat_history[-3:]],
            response_type=response_type,
            timestamp=datetime.utcnow()
        )
        
        # Create assistant message
        message_id = str(uuid.uuid4())
        assistant_message = ChatMessage(
            id=message_id,
            session_id=session_id,
            role="assistant",
            content=response_content,
            metadata=metadata,
            timestamp=datetime.utcnow()
        )
        session_service.add_message(session_id, assistant_message)
        
        # Extract citations (simplified)
        citations = []
        if use_rag and reranked_chunks:
            for i, chunk in enumerate(reranked_chunks[:3]):
                citations.append({
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "similarity_score": reranked_scores[i] if i < len(reranked_scores) else 0.0
                })
        
        return ChatResponse(
            message_id=message_id,
            content=response_content,
            citations=citations,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@router.get("/sessions/{session_id}/history", response_model=List[ChatMessage])
async def get_chat_history(session_id: str, limit: Optional[int] = None):
    """
    Get chat history for a session.
    
    Args:
        session_id: Session ID
        limit: Optional limit on number of messages
        
    Returns:
        List[ChatMessage]: Chat history
    """
    if not session_service.is_session_valid(session_id):
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    history = session_service.get_chat_history(session_id, limit)
    return history


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "OrgGPT Backend"}

