/**
 * API service for OrgGPT frontend.
 * 
 * This module handles all communication with the backend API,
 * including session management, file uploads, and chat functionality.
 */

import axios from 'axios';
import Cookies from 'js-cookie';

// API base URL - adjust for your environment
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '/api/v1' 
  : 'http://localhost:8000/api/v1';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  // timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Session management
const SESSION_COOKIE_NAME = 'orggpt_session_id';

/**
 * Get current session ID from cookies
 */
export const getSessionId = () => {
  return Cookies.get(SESSION_COOKIE_NAME);
};

/**
 * Set session ID in cookies
 */
export const setSessionId = (sessionId) => {
  Cookies.set(SESSION_COOKIE_NAME, sessionId, { 
    expires: 1, // 1 day
    sameSite: 'strict'
  });
};

/**
 * Clear session ID from cookies
 */
export const clearSessionId = () => {
  Cookies.remove(SESSION_COOKIE_NAME);
};

/**
 * Create a new session
 */
export const createSession = async () => {
  try {
    const response = await api.post('/sessions');
    const sessionId = response.data.session_id;
    setSessionId(sessionId);
    return response.data;
  } catch (error) {
    console.error('Failed to create session:', error);
    throw error;
  }
};

/**
 * Get session information
 */
export const getSession = async (sessionId) => {
  try {
    const response = await api.get(`/sessions/${sessionId}`);
    return response.data;
  } catch (error) {
    console.error('Failed to get session:', error);
    throw error;
  }
};

/**
 * Delete session
 */
export const deleteSession = async (sessionId) => {
  try {
    await api.delete(`/sessions/${sessionId}`);
    clearSessionId();
  } catch (error) {
    console.error('Failed to delete session:', error);
    throw error;
  }
};

/**
 * Upload a file to the session
 */
export const uploadFile = async (sessionId, file) => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post(`/sessions/${sessionId}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Failed to upload file:', error);
    throw error;
  }
};

/**
 * Process a URL for the session
 */
export const processUrl = async (sessionId, url) => {
  try {
    const formData = new FormData();
    formData.append('url', url);

    const response = await api.post(`/sessions/${sessionId}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Failed to process URL:', error);
    throw error;
  }
};

/**
 * Get documents for a session
 */
export const getSessionDocuments = async (sessionId) => {
  try {
    const response = await api.get(`/sessions/${sessionId}/documents`);
    return response.data;
  } catch (error) {
    console.error('Failed to get session documents:', error);
    throw error;
  }
};

/**
 * Delete a document from the session
 */
export const deleteDocument = async (sessionId, documentId) => {
  try {
    await api.delete(`/sessions/${sessionId}/documents/${documentId}`);
  } catch (error) {
    console.error('Failed to delete document:', error);
    throw error;
  }
};

/**
 * Search documents in the session
 */
export const searchDocuments = async (sessionId, query, limit = 10) => {
  try {
    const response = await api.post(`/sessions/${sessionId}/search`, {
      query,
      session_id: sessionId,
      limit,
    });
    return response.data;
  } catch (error) {
    console.error('Failed to search documents:', error);
    throw error;
  }
};

/**
 * Send a chat message
 */
export const sendChatMessage = async (sessionId, query, regenerate = false, messageId = null) => {
  try {
    const response = await api.post(`/sessions/${sessionId}/chat`, {
      query,
      session_id: sessionId,
      regenerate,
      message_id: messageId,
    });
    return response.data;
  } catch (error) {
    console.error('Failed to send chat message:', error);
    throw error;
  }
};

/**
 * Get chat history for a session
 */
export const getChatHistory = async (sessionId, limit = null) => {
  try {
    const params = limit ? { limit } : {};
    const response = await api.get(`/sessions/${sessionId}/history`, { params });
    return response.data;
  } catch (error) {
    console.error('Failed to get chat history:', error);
    throw error;
  }
};

/**
 * Health check
 */
export const healthCheck = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Health check failed:', error);
    throw error;
  }
};

// Initialize session on app load
export const initializeSession = async () => {
  let sessionId = getSessionId();
  
  if (!sessionId) {
    // Create new session if none exists
    const sessionData = await createSession();
    return sessionData.session_id;
  }
  
  try {
    // Validate existing session
    await getSession(sessionId);
    return sessionId;
  } catch (error) {
    // Session invalid, create new one
    const sessionData = await createSession();
    return sessionData.session_id;
  }
};

