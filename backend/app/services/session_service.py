"""
Session management service for OrgGPT.

This module handles user sessions, chat history, and session-based data management.
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from ..models.schemas import (
    SessionInfo, ChatMessage, QueryMetadata, 
    DocumentMetadata, DocumentChunk
)
from ..core.config import settings

class SessionService:
    """Service for managing user sessions and chat history."""
    
    def __init__(self):
        """Initialize the session service."""
        # In-memory storage (in production, use Redis or database)
        self.sessions: Dict[str, SessionInfo] = {}
        self.chat_history: Dict[str, List[ChatMessage]] = {}
        self.session_documents: Dict[str, List[DocumentMetadata]] = {}
        self.session_chunks: Dict[str, List[DocumentChunk]] = {}
        self.message_metadata: Dict[str, QueryMetadata] = {}
    
    def create_session(self) -> str:
        """
        Create a new session.
        
        Returns:
            str: New session ID
        """
        session_id = str(uuid.uuid4())
        
        session_info = SessionInfo(
            session_id=session_id,
            created_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            document_count=0,
            message_count=0
        )
        
        self.sessions[session_id] = session_info
        self.chat_history[session_id] = []
        self.session_documents[session_id] = []
        self.session_chunks[session_id] = []
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """
        Get session information.
        
        Args:
            session_id: Session ID
            
        Returns:
            Optional[SessionInfo]: Session information or None if not found
        """
        return self.sessions.get(session_id)
    
    def update_session_activity(self, session_id: str):
        """
        Update session last activity timestamp.
        
        Args:
            session_id: Session ID
        """
        if session_id in self.sessions:
            self.sessions[session_id].last_activity = datetime.utcnow()
    
    def is_session_valid(self, session_id: str) -> bool:
        """
        Check if a session is valid and not expired.
        
        Args:
            session_id: Session ID
            
        Returns:
            bool: True if session is valid, False otherwise
        """
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        expiry_time = session.last_activity + timedelta(hours=settings.SESSION_EXPIRE_HOURS)
        
        return datetime.utcnow() < expiry_time
    
    def add_message(self, session_id: str, message: ChatMessage):
        """
        Add a message to chat history.
        
        Args:
            session_id: Session ID
            message: Chat message to add
        """
        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
        
        self.chat_history[session_id].append(message)
        
        # Update session message count
        if session_id in self.sessions:
            self.sessions[session_id].message_count += 1
        
        # Store metadata if it's an assistant message
        if message.metadata:
            self.message_metadata[message.id] = message.metadata
        
        self.update_session_activity(session_id)
    
    def get_chat_history(self, session_id: str, limit: Optional[int] = None) -> List[ChatMessage]:
        """
        Get chat history for a session.
        
        Args:
            session_id: Session ID
            limit: Optional limit on number of messages to return
            
        Returns:
            List[ChatMessage]: Chat history
        """
        if session_id not in self.chat_history:
            return []
        
        messages = self.chat_history[session_id]
        
        if limit:
            return messages[-limit:]
        
        return messages
    
    def get_message_metadata(self, message_id: str) -> Optional[QueryMetadata]:
        """
        Get metadata for a specific message.
        
        Args:
            message_id: Message ID
            
        Returns:
            Optional[QueryMetadata]: Message metadata or None if not found
        """
        return self.message_metadata.get(message_id)
    
    def add_document(self, session_id: str, document: DocumentMetadata):
        """
        Add a document to the session.
        
        Args:
            session_id: Session ID
            document: Document metadata
        """
        if session_id not in self.session_documents:
            self.session_documents[session_id] = []
        
        self.session_documents[session_id].append(document)
        
        # Update session document count
        if session_id in self.sessions:
            self.sessions[session_id].document_count += 1
        
        self.update_session_activity(session_id)
    
    def get_session_documents(self, session_id: str) -> List[DocumentMetadata]:
        """
        Get all documents for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List[DocumentMetadata]: List of document metadata
        """
        return self.session_documents.get(session_id, [])
    
    def remove_document(self, session_id: str, document_id: str) -> bool:
        """
        Remove a document from the session.
        
        Args:
            session_id: Session ID
            document_id: Document ID to remove
            
        Returns:
            bool: True if document was removed, False if not found
        """
        if session_id not in self.session_documents:
            return False
        
        documents = self.session_documents[session_id]
        
        for i, doc in enumerate(documents):
            if doc.id == document_id:
                documents.pop(i)
                
                # Update session document count
                if session_id in self.sessions:
                    self.sessions[session_id].document_count -= 1
                
                # Remove associated chunks
                self.remove_document_chunks(session_id, document_id)
                
                self.update_session_activity(session_id)
                return True
        
        return False
    
    def add_document_chunks(self, session_id: str, chunks: List[DocumentChunk]):
        """
        Add document chunks to the session.
        
        Args:
            session_id: Session ID
            chunks: List of document chunks
        """
        if session_id not in self.session_chunks:
            self.session_chunks[session_id] = []
        
        self.session_chunks[session_id].extend(chunks)
        self.update_session_activity(session_id)
    
    def get_session_chunks(self, session_id: str) -> List[DocumentChunk]:
        """
        Get all chunks for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            List[DocumentChunk]: List of document chunks
        """
        return self.session_chunks.get(session_id, [])
    
    def remove_document_chunks(self, session_id: str, document_id: str):
        """
        Remove all chunks for a specific document.
        
        Args:
            session_id: Session ID
            document_id: Document ID
        """
        if session_id not in self.session_chunks:
            return
        
        chunks = self.session_chunks[session_id]
        self.session_chunks[session_id] = [
            chunk for chunk in chunks 
            if chunk.document_id != document_id
        ]
    
    def can_upload_file(self, session_id: str) -> bool:
        """
        Check if the session can upload more files.
        
        Args:
            session_id: Session ID
            
        Returns:
            bool: True if can upload, False if limit reached
        """
        if session_id not in self.sessions:
            return False
        
        return self.sessions[session_id].document_count < settings.MAX_FILES_PER_SESSION
    
    def delete_session(self, session_id: str):
        """
        Delete a session and all associated data.
        
        Args:
            session_id: Session ID to delete
        """
        # Remove session info
        self.sessions.pop(session_id, None)
        
        # Remove chat history
        self.chat_history.pop(session_id, None)
        
        # Remove documents
        self.session_documents.pop(session_id, None)
        
        # Remove chunks
        self.session_chunks.pop(session_id, None)
        
        # Remove message metadata for this session
        session_message_ids = []
        if session_id in self.chat_history:
            session_message_ids = [msg.id for msg in self.chat_history[session_id]]
        
        for msg_id in session_message_ids:
            self.message_metadata.pop(msg_id, None)
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session_info in self.sessions.items():
            expiry_time = session_info.last_activity + timedelta(hours=settings.SESSION_EXPIRE_HOURS)
            if current_time >= expiry_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dict[str, Any]: Session statistics
        """
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        
        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "document_count": session.document_count,
            "message_count": session.message_count,
            "total_chunks": len(self.session_chunks.get(session_id, [])),
            "can_upload_more": self.can_upload_file(session_id)
        }

