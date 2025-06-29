/**
 * Main App component for OrgGPT.
 * 
 * This is the root component that manages the overall application state,
 * session management, and coordinates between different UI components.
 */

import React, { useState, useEffect, useRef } from 'react';
import { Loader2, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';

// Import custom components
import Sidebar from './components/Sidebar';
import ChatMessage from './components/ChatMessage';
import ChatInput from './components/ChatInput';
import FileUpload from './components/FileUpload';
import DocumentList from './components/DocumentList';

// Import API services
import {
  initializeSession,
  sendChatMessage,
  getChatHistory,
  uploadFile,
  processUrl,
  getSessionDocuments,
  deleteDocument,
  deleteSession,
  createSession
} from './services/api';

import './App.css';

function App() {
  // State management
  const [sessionId, setSessionId] = useState(null);
  const [activeTab, setActiveTab] = useState('chat');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  
  // Chat state
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);
  
  // Documents state
  const [documents, setDocuments] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  
  // UI state
  const [error, setError] = useState('');
  const [isInitializing, setIsInitializing] = useState(true);
  
  // Refs
  const chatContainerRef = useRef(null);

  // Initialize session on app load
  useEffect(() => {
    const initApp = async () => {
      try {
        setIsInitializing(true);
        const sessionId = await initializeSession();
        setSessionId(sessionId);
        
        // Load existing data
        await Promise.all([
          loadChatHistory(sessionId),
          loadDocuments(sessionId)
        ]);
      } catch (error) {
        console.error('Failed to initialize app:', error);
        setError('Failed to initialize session. Please refresh the page.');
      } finally {
        setIsInitializing(false);
      }
    };

    initApp();
  }, []);

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Load chat history
  const loadChatHistory = async (sessionId) => {
    try {
      const history = await getChatHistory(sessionId);
      setMessages(history);
    } catch (error) {
      console.error('Failed to load chat history:', error);
    }
  };

  // Load documents
  const loadDocuments = async (sessionId) => {
    try {
      const docs = await getSessionDocuments(sessionId);
      setDocuments(docs);
    } catch (error) {
      console.error('Failed to load documents:', error);
    }
  };

  // Handle sending chat messages
  const handleSendMessage = async (message) => {
    if (!sessionId || isLoading) return;

    setIsLoading(true);
    setError('');

    try {
      // Add user message immediately
      const userMessage = {
        id: Date.now().toString(),
        role: 'user',
        content: message,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, userMessage]);

      // Send to backend
      const response = await sendChatMessage(sessionId, message);
      
      // Add assistant response
      const assistantMessage = {
        id: response.message_id,
        role: 'assistant',
        content: response.content,
        metadata: response.metadata,
        citations: response.citations,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      console.error('Failed to send message:', error);
      setError(error.message || 'Failed to send message');
    } finally {
      setIsLoading(false);
    }
  };

  // Handle message regeneration
  const handleRegenerate = async (messageId) => {
    if (!sessionId || isRegenerating) return;

    setIsRegenerating(true);
    setError('');

    try {
      const response = await sendChatMessage(sessionId, '', true, messageId);
      
      // Update the message in place
      setMessages(prev => prev.map(msg => 
        msg.id === messageId 
          ? {
              ...msg,
              content: response.content,
              metadata: response.metadata,
              citations: response.citations,
              timestamp: new Date().toISOString()
            }
          : msg
      ));

    } catch (error) {
      console.error('Failed to regenerate message:', error);
      setError(error.message || 'Failed to regenerate message');
    } finally {
      setIsRegenerating(false);
    }
  };

  // Handle file upload
  const handleFileUpload = async (file) => {
    if (!sessionId) return;

    setIsUploading(true);
    try {
      await uploadFile(sessionId, file);
      await loadDocuments(sessionId);
    } catch (error) {
      console.error('Failed to upload file:', error);
      throw error;
    } finally {
      setIsUploading(false);
    }
  };

  // Handle URL processing
  const handleUrlProcess = async (url) => {
    if (!sessionId) return;

    setIsUploading(true);
    try {
      await processUrl(sessionId, url);
      await loadDocuments(sessionId);
    } catch (error) {
      console.error('Failed to process URL:', error);
      throw error;
    } finally {
      setIsUploading(false);
    }
  };

  // Handle document deletion
  const handleDeleteDocument = async (documentId) => {
    if (!sessionId) return;

    try {
      await deleteDocument(sessionId, documentId);
      await loadDocuments(sessionId);
    } catch (error) {
      console.error('Failed to delete document:', error);
      throw error;
    }
  };

  // Handle new session
  const handleNewSession = async () => {
    try {
      const newSessionData = await createSession();
      setSessionId(newSessionData.session_id);
      setMessages([]);
      setDocuments([]);
      setActiveTab('chat');
    } catch (error) {
      console.error('Failed to create new session:', error);
      setError('Failed to create new session');
    }
  };

  // Handle clear session
  const handleClearSession = async () => {
    if (!sessionId) return;

    try {
      await deleteSession(sessionId);
      await handleNewSession();
    } catch (error) {
      console.error('Failed to clear session:', error);
      setError('Failed to clear session');
    }
  };

  // Loading screen
  if (isInitializing) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin mx-auto mb-4" />
          <p className="text-muted-foreground">Initializing OrgGPT...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen flex bg-background">
      {/* Sidebar */}
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        documentCount={documents.length}
        messageCount={messages.length}
        onNewSession={handleNewSession}
        onClearSession={handleClearSession}
        isCollapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Error Alert */}
        {error && (
          <Alert variant="destructive" className="m-4 mb-0">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="flex items-center justify-between">
              {error}
              <Button 
                variant="outline" 
                size="sm" 
                onClick={() => setError('')}
                className="ml-2"
              >
                Dismiss
              </Button>
            </AlertDescription>
          </Alert>
        )}

        {/* Chat Tab */}
        {activeTab === 'chat' && (
          <>
            {/* Chat Messages */}
            <div 
              ref={chatContainerRef}
              className="flex-1 overflow-y-auto"
            >
              {messages.length === 0 ? (
                <div className="h-full flex items-center justify-center">
                  <div className="text-center max-w-md">
                    <h2 className="text-2xl font-bold mb-4">Welcome to OrgGPT</h2>
                    <p className="text-muted-foreground mb-6">
                      Upload documents or start chatting to get intelligent responses 
                      powered by advanced RAG technology.
                    </p>
                    <Button 
                      variant="outline" 
                      onClick={() => setActiveTab('documents')}
                    >
                      Upload Documents
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-0">
                  {messages.map((message) => (
                    <ChatMessage
                      key={message.id}
                      message={message}
                      onRegenerate={handleRegenerate}
                      isRegenerating={isRegenerating}
                      showRegenerate={message.role === 'assistant'}
                    />
                  ))}
                  {isLoading && (
                    <div className="flex justify-start p-4">
                      <div className="flex items-center gap-2 bg-card border rounded-lg p-4">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span className="text-sm text-muted-foreground">
                          Thinking...
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Chat Input */}
            <ChatInput
              onSendMessage={handleSendMessage}
              isLoading={isLoading}
              placeholder={
                documents.length > 0 
                  ? "Ask me anything about your documents..." 
                  : "Ask me anything or upload documents for context..."
              }
            />
          </>
        )}

        {/* Documents Tab */}
        {activeTab === 'documents' && (
          <div className="flex-1 overflow-y-auto p-6">
            <div className="max-w-4xl mx-auto space-y-6">
              <div>
                <h2 className="text-2xl font-bold mb-2">Document Management</h2>
                <p className="text-muted-foreground">
                  Upload documents or add URLs to enhance your chat experience with contextual information.
                </p>
              </div>

              <FileUpload
                onFileUpload={handleFileUpload}
                onUrlProcess={handleUrlProcess}
                isUploading={isUploading}
              />

              <DocumentList
                documents={documents}
                onDelete={handleDeleteDocument}
                isLoading={false}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;

