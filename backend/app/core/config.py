"""
Configuration settings for OrgGPT application.

This module contains all configuration variables and settings
used throughout the application.
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application settings
    APP_NAME: str = "OrgGPT"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API settings
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # File upload settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    MAX_FILES_PER_SESSION: int = 10
    UPLOAD_DIR: str = "uploads"
    ALLOWED_EXTENSIONS: List[str] = [
        "pdf", "docx", "xlsx", "csv", "txt", "md", "pptx", "zip"
    ]
    
    # Session settings
    SESSION_EXPIRE_HOURS: int = 24
    
    # LLM settings
    GROQ_API_KEY: str = "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"
    GROQ_MODEL: str = "llama3-8b-8192"
    
    # Hugging Face settings
    HF_API_KEY: str = "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"
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    RERANKER_MODEL: str = "BAAI/bge-reranker-large"
    
    # RAG settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RETRIEVAL: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    INTENT_THRESHOLD: float = 0.5
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./orggpt.db"
    REDIS_URL: str = "redis://localhost:6379"
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# RAG-REQUIRED INTENTS - Use when query requires retrieving information from uploaded documents
RAG_INTENTS = [
    "document_search",         # Finding specific documents or files in uploaded collection
    "content_lookup",          # Looking up specific information within uploaded documents
    "document_summary",        # Summarizing uploaded documents or sections
    "cross_reference",         # Finding information across multiple uploaded documents
    "information_extraction",  # Extracting specific data points from uploaded documents
    "fact_verification",       # Verifying claims against uploaded document content
    "section_retrieval",       # Getting specific sections from uploaded documents
    "keyword_search",          # Searching for terms/phrases in uploaded documents
    "comparative_analysis",    # Comparing information between uploaded documents
    "trend_analysis",          # Identifying patterns in uploaded document data
    "evidence_finding",        # Finding supporting evidence from uploaded sources
    "citation_request",        # Getting sources/references from uploaded documents
    "quote_extraction",        # Extracting specific quotes or passages from documents
    "data_retrieval",          # Retrieving specific data points or statistics from documents
    "source_attribution"       # Identifying which document contains specific information
]

# GENERAL-KNOWLEDGE INTENTS - Use when query can be answered using general knowledge (no document retrieval needed)
GENERAL_INTENTS = [
    "greeting",               # Greetings, hellos, how are you
    "chitchat",              # Casual conversation, small talk, personal questions
    "farewell",              # Goodbyes, see you later, closing conversations
    "gratitude",             # Thank you, appreciation, acknowledgments
    "definitions",           # General definitions or explanations of concepts
    "explanations",          # General explanations not requiring specific documents
    "math_calculations",     # Mathematical problems or calculations
    "science_facts",         # General science knowledge and facts
    "history_general",       # General historical information
    "geography",             # Information about places, countries, cities
    "technology_general",    # General technology concepts and trends
    "programming_help",      # General coding questions and programming concepts
    "language_questions",    # Grammar, translation, language learning
    "creative_writing",      # Story writing, poetry, creative content generation
    "brainstorming",         # Idea generation, creative thinking exercises
    "general_advice",        # Life advice, general recommendations
    "trivia",                # Fun facts, quiz questions, general knowledge trivia
    "current_events",        # General news, current affairs (from training data)
    "cooking_recipes",       # General cooking and recipe information
    "health_general",        # General health and wellness information
    "entertainment",         # Movies, books, music, games recommendations
    "travel_general",        # General travel advice and destination information
    "hobby_interests",       # General hobby and interest discussions
    "philosophy",            # Philosophical questions and discussions
    "ethics"                 # Ethical discussions and moral questions
]

# HYBRID INTENTS - Use when query requires both uploaded document content AND general knowledge
HYBRID_INTENTS = [
    "document_explanation",   # Explaining concepts found in documents using general knowledge
    "context_enhancement",    # Adding general context to document-specific information
    "comparison_external",    # Comparing document content with general knowledge
    "document_validation",    # Validating document claims against general knowledge
    "knowledge_synthesis",    # Combining document facts with general understanding
    "educational_support",    # Using documents for learning with general explanations
    "research_assistance",    # Helping with research using both sources and general knowledge
    "analysis_enhancement",   # Enhancing document analysis with broader knowledge
    "recommendation_informed" # Making recommendations based on both document data and general principles
]

# Create global settings instance
settings = Settings()

