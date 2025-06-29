"""
Data schemas for OrgGPT application.

This module defines Pydantic models for request/response validation
and data serialization throughout the application.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class IntentType(str, Enum):
    """Enumeration of supported intent types."""
    
    # RAG-Required Intents - Use when query requires retrieving information from uploaded documents
    DOCUMENT_SEARCH = "document_search"
    CONTENT_LOOKUP = "content_lookup"
    DOCUMENT_SUMMARY = "document_summary"
    CROSS_REFERENCE = "cross_reference"
    INFORMATION_EXTRACTION = "information_extraction"
    FACT_VERIFICATION = "fact_verification"
    SECTION_RETRIEVAL = "section_retrieval"
    KEYWORD_SEARCH = "keyword_search"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    TREND_ANALYSIS = "trend_analysis"
    EVIDENCE_FINDING = "evidence_finding"
    CITATION_REQUEST = "citation_request"
    QUOTE_EXTRACTION = "quote_extraction"
    DATA_RETRIEVAL = "data_retrieval"
    SOURCE_ATTRIBUTION = "source_attribution"
    
    # General Knowledge Intents - Use when query can be answered using general knowledge
    GREETING = "greeting"
    CHITCHAT = "chitchat"
    FAREWELL = "farewell"
    GRATITUDE = "gratitude"
    DEFINITIONS = "definitions"
    EXPLANATIONS = "explanations"
    MATH_CALCULATIONS = "math_calculations"
    SCIENCE_FACTS = "science_facts"
    HISTORY_GENERAL = "history_general"
    GEOGRAPHY = "geography"
    TECHNOLOGY_GENERAL = "technology_general"
    PROGRAMMING_HELP = "programming_help"
    LANGUAGE_QUESTIONS = "language_questions"
    CREATIVE_WRITING = "creative_writing"
    BRAINSTORMING = "brainstorming"
    GENERAL_ADVICE = "general_advice"
    TRIVIA = "trivia"
    CURRENT_EVENTS = "current_events"
    COOKING_RECIPES = "cooking_recipes"
    HEALTH_GENERAL = "health_general"
    ENTERTAINMENT = "entertainment"
    TRAVEL_GENERAL = "travel_general"
    HOBBY_INTERESTS = "hobby_interests"
    PHILOSOPHY = "philosophy"
    ETHICS = "ethics"
    
    # Hybrid Intents - Use when query requires both uploaded document content AND general knowledge
    DOCUMENT_EXPLANATION = "document_explanation"
    CONTEXT_ENHANCEMENT = "context_enhancement"
    COMPARISON_EXTERNAL = "comparison_external"
    DOCUMENT_VALIDATION = "document_validation"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    EDUCATIONAL_SUPPORT = "educational_support"
    RESEARCH_ASSISTANCE = "research_assistance"
    ANALYSIS_ENHANCEMENT = "analysis_enhancement"
    RECOMMENDATION_INFORMED = "recommendation_informed"



class DocumentType(str, Enum):
    """Supported document types for ingestion."""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    TXT = "txt"
    MD = "md"
    PPTX = "pptx"
    ZIP = "zip"
    URL = "url"


class DocumentMetadata(BaseModel):
    """Metadata for uploaded documents."""
    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: DocumentType = Field(..., description="Document type")
    file_size: int = Field(..., description="File size in bytes")
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    chunk_count: int = Field(default=0, description="Number of text chunks extracted")
    session_id: str = Field(..., description="Session ID this document belongs to")


class DocumentChunk(BaseModel):
    """Text chunk extracted from a document."""
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Text content of the chunk")
    chunk_index: int = Field(..., description="Index of chunk within document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional chunk metadata")


class QueryRequest(BaseModel):
    """Request model for chat queries."""
    query: str = Field(..., description="User query text")
    session_id: str = Field(..., description="Session identifier")
    regenerate: bool = Field(default=False, description="Whether to regenerate previous response")
    message_id: Optional[str] = Field(None, description="Message ID for regeneration")


class QueryMetadata(BaseModel):
    """Metadata stored for each query and response."""
    query: str = Field(..., description="Original user query")
    rewritten_query: str = Field(..., description="Rewritten/optimized query")
    intent: IntentType = Field(..., description="Classified intent")
    intent_score: float = Field(..., description="Intent classification confidence")
    similarity_score: float = Field(..., description="Highest similarity score from retrieval")
    context_chunks: List[str] = Field(default_factory=list, description="Retrieved context chunks")
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description="Recent chat history")
    response_type: str = Field(..., description="Type of response generated (RAG/General)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatMessage(BaseModel):
    """Chat message model."""
    id: str = Field(..., description="Unique message identifier")
    session_id: str = Field(..., description="Session identifier")
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    metadata: Optional[QueryMetadata] = Field(None, description="Query metadata for assistant messages")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatResponse(BaseModel):
    """Response model for chat queries."""
    message_id: str = Field(..., description="Unique message identifier")
    content: str = Field(..., description="Response content")
    citations: List[Dict[str, Any]] = Field(default_factory=list, description="Source citations")
    metadata: QueryMetadata = Field(..., description="Query processing metadata")


class SessionInfo(BaseModel):
    """Session information model."""
    session_id: str = Field(..., description="Unique session identifier")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    document_count: int = Field(default=0, description="Number of uploaded documents")
    message_count: int = Field(default=0, description="Number of chat messages")


class FileUploadResponse(BaseModel):
    """Response model for file uploads."""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: DocumentType = Field(..., description="Document type")
    file_size: int = Field(..., description="File size in bytes")
    chunk_count: int = Field(..., description="Number of text chunks extracted")
    message: str = Field(..., description="Upload status message")


class DocumentSearchRequest(BaseModel):
    """Request model for document search."""
    query: str = Field(..., description="Search query")
    session_id: str = Field(..., description="Session identifier")
    limit: int = Field(default=10, description="Maximum number of results")


class DocumentSearchResult(BaseModel):
    """Search result for document content."""
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Document filename")
    chunk_content: str = Field(..., description="Matching chunk content")
    similarity_score: float = Field(..., description="Similarity score")
    chunk_index: int = Field(..., description="Chunk index within document")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

