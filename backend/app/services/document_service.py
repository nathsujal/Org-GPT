"""
Document processing service for OrgGPT.

This module handles document ingestion, processing, and management
including file uploads, text extraction, and chunking.
"""

import os
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
import requests
from urllib.parse import urlparse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader, 
    UnstructuredExcelLoader, CSVLoader, TextLoader,
    UnstructuredMarkdownLoader, UnstructuredPowerPointLoader
)

from ..models.schemas import DocumentMetadata, DocumentChunk, DocumentType
from ..core.config import settings


class DocumentService:
    """Service for handling document operations."""
    
    def __init__(self):
        """Initialize the document service."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
        )
        
        # Ensure upload directory exists
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    def validate_file(self, filename: str, file_size: int) -> bool:
        """
        Validate uploaded file.
        
        Args:
            filename: Name of the uploaded file
            file_size: Size of the file in bytes
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        # Check file extension
        file_ext = filename.lower().split('.')[-1]
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            return False
            
        # Check file size
        if file_size > settings.MAX_FILE_SIZE:
            return False
            
        return True
    
    def get_document_type(self, filename: str) -> DocumentType:
        """
        Determine document type from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            DocumentType: The type of the document
        """
        file_ext = filename.lower().split('.')[-1]
        type_mapping = {
            'pdf': DocumentType.PDF,
            'docx': DocumentType.DOCX,
            'xlsx': DocumentType.XLSX,
            'csv': DocumentType.CSV,
            'txt': DocumentType.TXT,
            'md': DocumentType.MD,
            'pptx': DocumentType.PPTX,
            'zip': DocumentType.ZIP
        }
        return type_mapping.get(file_ext, DocumentType.TXT)
    
    def save_uploaded_file(self, file_content: bytes, filename: str, session_id: str) -> str:
        """
        Save uploaded file to disk.
        
        Args:
            file_content: Binary content of the file
            filename: Original filename
            session_id: Session ID
            
        Returns:
            str: Path to the saved file
        """
        # Create session directory
        session_dir = os.path.join(settings.UPLOAD_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Generate unique filename to avoid conflicts
        file_id = str(uuid.uuid4())
        file_ext = filename.split('.')[-1] if '.' in filename else ''
        safe_filename = f"{file_id}.{file_ext}" if file_ext else file_id
        
        file_path = os.path.join(session_dir, safe_filename)
        
        with open(file_path, 'wb') as f:
            f.write(file_content)
            
        return file_path
    
    def extract_text_from_file(self, file_path: str, doc_type: DocumentType) -> List[str]:
        """
        Extract text content from a file.
        
        Args:
            file_path: Path to the file
            doc_type: Type of the document
            
        Returns:
            List[str]: List of text chunks extracted from the document
        """
        try:
            # Select appropriate loader based on document type
            if doc_type == DocumentType.PDF:
                loader = PyPDFLoader(file_path)
            elif doc_type == DocumentType.DOCX:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif doc_type == DocumentType.XLSX:
                loader = UnstructuredExcelLoader(file_path)
            elif doc_type == DocumentType.CSV:
                loader = CSVLoader(file_path)
            elif doc_type == DocumentType.TXT:
                loader = TextLoader(file_path)
            elif doc_type == DocumentType.MD:
                loader = UnstructuredMarkdownLoader(file_path)
            elif doc_type == DocumentType.PPTX:
                loader = UnstructuredPowerPointLoader(file_path)
            else:
                # Fallback to text loader
                loader = TextLoader(file_path)
            
            # Load and split documents
            documents = loader.load()
            
            # Combine all document content
            full_text = "\n\n".join([doc.page_content for doc in documents])
            
            # Split into chunks
            chunks = self.text_splitter.split_text(full_text)
            
            return chunks
            
        except Exception as e:
            print(f"Error extracting text from {file_path}: {str(e)}")
            return []
    
    def process_url(self, url: str) -> List[str]:
        """
        Process a URL and extract text content.
        
        Args:
            url: URL to process
            
        Returns:
            List[str]: List of text chunks extracted from the URL
        """
        try:
            # Basic URL validation
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid URL format")
            
            # Fetch content (basic implementation)
            # response = requests.get(url, timeout=30)
            response = requests.get(url)
            response.raise_for_status()
            
            # For now, just extract text content
            # In a production system, you'd want more sophisticated parsing
            content = response.text
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            
            return chunks
            
        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")
            return []
    
    def create_document_metadata(
        self, 
        filename: str, 
        file_size: int, 
        session_id: str,
        chunk_count: int
    ) -> DocumentMetadata:
        """
        Create document metadata.
        
        Args:
            filename: Original filename
            file_size: Size of the file in bytes
            session_id: Session ID
            chunk_count: Number of chunks extracted
            
        Returns:
            DocumentMetadata: Document metadata object
        """
        doc_id = str(uuid.uuid4())
        doc_type = self.get_document_type(filename)
        
        return DocumentMetadata(
            id=doc_id,
            filename=filename,
            file_type=doc_type,
            file_size=file_size,
            chunk_count=chunk_count,
            session_id=session_id
        )
    
    def create_document_chunks(
        self, 
        document_id: str, 
        chunks: List[str]
    ) -> List[DocumentChunk]:
        """
        Create document chunk objects.
        
        Args:
            document_id: ID of the parent document
            chunks: List of text chunks
            
        Returns:
            List[DocumentChunk]: List of document chunk objects
        """
        chunk_objects = []
        
        for i, chunk_content in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk = DocumentChunk(
                id=chunk_id,
                document_id=document_id,
                content=chunk_content,
                chunk_index=i,
                metadata={"length": len(chunk_content)}
            )
            chunk_objects.append(chunk)
        
        return chunk_objects
    
    def delete_document_files(self, session_id: str, document_id: Optional[str] = None):
        """
        Delete document files from disk.
        
        Args:
            session_id: Session ID
            document_id: Optional document ID. If None, deletes all files for the session
        """
        session_dir = os.path.join(settings.UPLOAD_DIR, session_id)
        
        if not os.path.exists(session_dir):
            return
        
        if document_id:
            # Delete specific document file
            # This would require tracking file paths in metadata
            pass
        else:
            # Delete entire session directory
            import shutil
            shutil.rmtree(session_dir, ignore_errors=True)