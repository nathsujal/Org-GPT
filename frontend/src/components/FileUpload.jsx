/**
 * FileUpload component for handling file uploads and URL processing.
 * 
 * This component provides a drag-and-drop interface for file uploads
 * and URL input functionality with progress tracking and validation.
 */

import React, { useState, useRef } from 'react';
import { Upload, Link, X, File, AlertCircle, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';

const FileUpload = ({ 
  onFileUpload, 
  onUrlProcess, 
  isUploading = false,
  maxFiles = 10,
  maxFileSize = 10 * 1024 * 1024, // 10MB
  allowedTypes = ['pdf', 'docx', 'xlsx', 'csv', 'txt', 'md', 'pptx', 'zip']
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [url, setUrl] = useState('');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const fileInputRef = useRef(null);

  const validateFile = (file) => {
    // Check file size
    if (file.size > maxFileSize) {
      return `File size exceeds ${Math.round(maxFileSize / (1024 * 1024))}MB limit`;
    }

    // Check file type
    const fileExtension = file.name.split('.').pop()?.toLowerCase();
    if (!allowedTypes.includes(fileExtension)) {
      return `File type .${fileExtension} is not supported`;
    }

    return null;
  };

  const handleFiles = async (files) => {
    setError('');
    setSuccess('');

    const fileArray = Array.from(files);
    
    // Validate each file
    for (const file of fileArray) {
      const validationError = validateFile(file);
      if (validationError) {
        setError(validationError);
        return;
      }
    }

    // Check max files limit
    if (fileArray.length > maxFiles) {
      setError(`Maximum ${maxFiles} files allowed`);
      return;
    }

    try {
      setUploadProgress(0);
      
      for (let i = 0; i < fileArray.length; i++) {
        const file = fileArray[i];
        setUploadProgress(((i + 1) / fileArray.length) * 100);
        
        await onFileUpload(file);
      }
      
      setSuccess(`Successfully uploaded ${fileArray.length} file(s)`);
      setTimeout(() => setSuccess(''), 3000);
      
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (error) {
      setError(error.message || 'Upload failed');
    } finally {
      setUploadProgress(0);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFiles(e.target.files);
    }
  };

  const handleUrlSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim()) return;

    setError('');
    setSuccess('');

    try {
      await onUrlProcess(url.trim());
      setSuccess('URL processed successfully');
      setUrl('');
      setTimeout(() => setSuccess(''), 3000);
    } catch (error) {
      setError(error.message || 'URL processing failed');
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-4">
      {/* File Upload Area */}
      <Card 
        className={`border-2 border-dashed transition-colors ${
          dragActive 
            ? 'border-primary bg-primary/5' 
            : 'border-muted-foreground/25 hover:border-muted-foreground/50'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <div className="p-8 text-center">
          <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <h3 className="text-lg font-semibold mb-2">Upload Documents</h3>
          <p className="text-sm text-muted-foreground mb-4">
            Drag and drop files here, or click to select files
          </p>
          
          <Button 
            variant="outline" 
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
          >
            <File className="w-4 h-4 mr-2" />
            Choose Files
          </Button>
          
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={allowedTypes.map(type => `.${type}`).join(',')}
            onChange={handleFileSelect}
            className="hidden"
          />
          
          <div className="mt-4 text-xs text-muted-foreground">
            <p>Supported formats: {allowedTypes.join(', ').toUpperCase()}</p>
            <p>Max file size: {formatFileSize(maxFileSize)} | Max files: {maxFiles}</p>
          </div>
        </div>
      </Card>

      {/* URL Input */}
      <Card className="p-4">
        <form onSubmit={handleUrlSubmit} className="space-y-3">
          <div className="flex items-center gap-2">
            <Link className="w-4 h-4 text-muted-foreground" />
            <h4 className="font-medium">Process URL</h4>
          </div>
          
          <div className="flex gap-2">
            <Input
              type="url"
              placeholder="https://example.com/document"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              disabled={isUploading}
              className="flex-1"
            />
            <Button 
              type="submit" 
              disabled={isUploading || !url.trim()}
              variant="outline"
            >
              Process
            </Button>
          </div>
        </form>
      </Card>

      {/* Upload Progress */}
      {isUploading && uploadProgress > 0 && (
        <Card className="p-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span>Uploading...</span>
              <span>{Math.round(uploadProgress)}%</span>
            </div>
            <Progress value={uploadProgress} className="w-full" />
          </div>
        </Card>
      )}

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Success Alert */}
      {success && (
        <Alert className="border-green-200 bg-green-50 text-green-800 dark:border-green-800 dark:bg-green-950 dark:text-green-200">
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>{success}</AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default FileUpload;

