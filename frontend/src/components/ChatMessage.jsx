/**
 * ChatMessage component for displaying chat messages with markdown support.
 * 
 * This component renders both user and assistant messages with proper formatting,
 * including markdown rendering for assistant responses and citation support.
 */

import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { User, Bot, RefreshCw, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useState } from 'react';

// Import highlight.js styles
import 'prismjs/themes/prism-tomorrow.css';

const ChatMessage = ({ 
  message, 
  onRegenerate, 
  isRegenerating = false,
  showRegenerate = false 
}) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';
  const isAssistant = message.role === 'assistant';

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  const handleRegenerate = () => {
    if (onRegenerate && message.id) {
      onRegenerate(message.id);
    }
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  return (
    <div className={`flex gap-4 p-4 ${isUser ? 'justify-end' : 'justify-start'}`}>
      {/* Avatar */}
      {!isUser && (
        <div className="flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
            <Bot className="w-5 h-5 text-primary" />
          </div>
        </div>
      )}

      {/* Message Content */}
      <div className={`flex flex-col max-w-[80%] ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Message Bubble */}
        <Card className={`p-4 ${
          isUser 
            ? 'bg-primary text-primary-foreground ml-auto' 
            : 'bg-card border'
        }`}>
          {isUser ? (
            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeHighlight]}
                components={{
                  // Custom components for better styling
                  code: ({ node, inline, className, children, ...props }) => {
                    const match = /language-(\w+)/.exec(className || '');
                    return !inline && match ? (
                      <pre className="bg-muted p-3 rounded-md overflow-x-auto">
                        <code className={className} {...props}>
                          {children}
                        </code>
                      </pre>
                    ) : (
                      <code className="bg-muted px-1 py-0.5 rounded text-sm" {...props}>
                        {children}
                      </code>
                    );
                  },
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-4 border-primary pl-4 italic">
                      {children}
                    </blockquote>
                  ),
                  table: ({ children }) => (
                    <div className="overflow-x-auto">
                      <table className="min-w-full border-collapse border border-border">
                        {children}
                      </table>
                    </div>
                  ),
                  th: ({ children }) => (
                    <th className="border border-border px-3 py-2 bg-muted font-semibold text-left">
                      {children}
                    </th>
                  ),
                  td: ({ children }) => (
                    <td className="border border-border px-3 py-2">
                      {children}
                    </td>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </Card>

        {/* Message Metadata */}
        <div className="flex items-center gap-2 mt-2 text-xs text-muted-foreground">
          <span>{formatTimestamp(message.timestamp)}</span>
          
          {/* Response Type Badge for Assistant Messages */}
          {isAssistant && message.metadata && (
            <Badge variant="outline" className="text-xs">
              {message.metadata.response_type}
            </Badge>
          )}

          {/* Intent Badge for Assistant Messages */}
          {isAssistant && message.metadata && (
            <Badge variant="secondary" className="text-xs">
              {message.metadata.intent.replace('_', ' ')}
            </Badge>
          )}
        </div>

        {/* Citations */}
        {/* {isAssistant && message.citations && message.citations.length > 0 && (
          <div className="mt-3 space-y-2">
            <p className="text-xs font-medium text-muted-foreground">Sources:</p>
            <div className="flex flex-wrap gap-2">
              {message.citations.map((citation, index) => (
                <Badge 
                  key={index} 
                  variant="outline" 
                  className="text-xs cursor-pointer hover:bg-accent"
                  title={`Similarity: ${(citation.similarity_score * 100).toFixed(1)}%`}
                >
                  Doc {citation.document_id.slice(0, 8)}...
                </Badge>
              ))}
            </div>
          </div>
        )} */}

        {/* Action Buttons */}
        {isAssistant && (
          <div className="flex items-center gap-2 mt-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              className="h-8 px-2"
            >
              {copied ? (
                <Check className="w-3 h-3" />
              ) : (
                <Copy className="w-3 h-3" />
              )}
            </Button>
            
            {showRegenerate && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleRegenerate}
                disabled={isRegenerating}
                className="h-8 px-2"
              >
                <RefreshCw className={`w-3 h-3 ${isRegenerating ? 'animate-spin' : ''}`} />
              </Button>
            )}
          </div>
        )}
      </div>

      {/* User Avatar */}
      {isUser && (
        <div className="flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
            <User className="w-5 h-5 text-secondary-foreground" />
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatMessage;

