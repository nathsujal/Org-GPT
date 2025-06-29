/**
 * DocumentList component for displaying and managing uploaded documents.
 * 
 * This component shows a list of uploaded documents with metadata,
 * search functionality, and delete capabilities.
 */

import React, { useState } from 'react';
import { 
  File, 
  FileText, 
  FileSpreadsheet, 
  FileImage, 
  Link, 
  Trash2, 
  Search,
  Calendar,
  HardDrive
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog';

const DocumentList = ({ 
  documents = [], 
  onDelete, 
  onSearch,
  isLoading = false 
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [deleteDocId, setDeleteDocId] = useState(null);

  const getFileIcon = (fileType) => {
    switch (fileType) {
      case 'pdf':
        return <FileText className="w-4 h-4 text-red-500" />;
      case 'docx':
        return <FileText className="w-4 h-4 text-blue-500" />;
      case 'xlsx':
        return <FileSpreadsheet className="w-4 h-4 text-green-500" />;
      case 'csv':
        return <FileSpreadsheet className="w-4 h-4 text-green-600" />;
      case 'pptx':
        return <FileImage className="w-4 h-4 text-orange-500" />;
      case 'url':
        return <Link className="w-4 h-4 text-purple-500" />;
      default:
        return <File className="w-4 h-4 text-gray-500" />;
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString([], {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleSearch = (e) => {
    e.preventDefault();
    if (onSearch && searchQuery.trim()) {
      onSearch(searchQuery.trim());
    }
  };

  const handleDelete = async (documentId) => {
    try {
      await onDelete(documentId);
      setDeleteDocId(null);
    } catch (error) {
      console.error('Failed to delete document:', error);
    }
  };

  const filteredDocuments = documents.filter(doc =>
    doc.filename.toLowerCase().includes(searchQuery.toLowerCase())
  );

  if (documents.length === 0) {
    return (
      <Card className="p-8 text-center">
        <File className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
        <h3 className="text-lg font-semibold mb-2">No Documents</h3>
        <p className="text-sm text-muted-foreground">
          Upload documents or add URLs to get started
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      {/* Search Bar */}
      <form onSubmit={handleSearch} className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search documents..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
        <Button type="submit" variant="outline" disabled={!searchQuery.trim()}>
          Search
        </Button>
      </form>

      {/* Document List */}
      <div className="space-y-2">
        {filteredDocuments.map((document) => (
          <Card key={document.id} className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3 flex-1 min-w-0">
                {/* File Icon */}
                <div className="flex-shrink-0">
                  {getFileIcon(document.file_type)}
                </div>

                {/* Document Info */}
                <div className="flex-1 min-w-0">
                  <h4 className="font-medium truncate" title={document.filename}>
                    {document.filename}
                  </h4>
                  
                  <div className="flex items-center gap-4 mt-1 text-xs text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <HardDrive className="w-3 h-3" />
                      {document.file_type === 'url' ? 'URL' : formatFileSize(document.file_size)}
                    </div>
                    
                    <div className="flex items-center gap-1">
                      <Calendar className="w-3 h-3" />
                      {formatDate(document.upload_timestamp)}
                    </div>
                    
                    <Badge variant="secondary" className="text-xs">
                      {document.chunk_count} chunks
                    </Badge>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs">
                  {document.file_type.toUpperCase()}
                </Badge>
                
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-8 w-8 p-0 text-destructive hover:text-destructive"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>Delete Document</AlertDialogTitle>
                      <AlertDialogDescription>
                        Are you sure you want to delete "{document.filename}"? 
                        This action cannot be undone and will remove the document 
                        from your chat context.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>Cancel</AlertDialogCancel>
                      <AlertDialogAction
                        onClick={() => handleDelete(document.id)}
                        className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                      >
                        Delete
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Summary */}
      <div className="text-xs text-muted-foreground text-center pt-2 border-t">
        {documents.length} document{documents.length !== 1 ? 's' : ''} uploaded
        {searchQuery && filteredDocuments.length !== documents.length && 
          ` • ${filteredDocuments.length} matching search`
        }
      </div>
    </div>
  );
};

export default DocumentList;

