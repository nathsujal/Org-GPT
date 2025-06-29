/**
 * Sidebar component for navigation and document management.
 * 
 * This component provides navigation between chat and documents,
 * session management, and quick access to key features.
 */

import React, { useState } from 'react';
import { 
  MessageSquare, 
  FileText, 
  Settings, 
  Trash2, 
  Plus,
  Menu,
  ChevronLeft,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
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

const Sidebar = ({ 
  activeTab, 
  onTabChange, 
  documentCount = 0, 
  messageCount = 0,
  onNewSession,
  onClearSession,
  isCollapsed = false,
  onToggleCollapse
}) => {
  const [showClearDialog, setShowClearDialog] = useState(false);

  const menuItems = [
    {
      id: 'chat',
      label: 'Chat',
      icon: MessageSquare,
      badge: messageCount > 0 ? messageCount : null
    },
    {
      id: 'documents',
      label: 'Documents',
      icon: FileText,
      badge: documentCount > 0 ? documentCount : null
    }
  ];

  const handleClearSession = () => {
    onClearSession();
    setShowClearDialog(false);
  };

  if (isCollapsed) {
    return (
      <div className="w-16 bg-card border-r flex flex-col">
        {/* Toggle Button */}
        <div className="p-4 border-b">
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleCollapse}
            className="w-8 h-8 p-0"
          >
            <Menu className="w-4 h-4" />
          </Button>
        </div>

        {/* Menu Items */}
        <div className="flex-1 p-2 space-y-2">
          {menuItems.map((item) => (
            <Button
              key={item.id}
              variant={activeTab === item.id ? "default" : "ghost"}
              size="sm"
              onClick={() => onTabChange(item.id)}
              className="w-full h-12 p-0 relative"
              title={item.label}
            >
              <item.icon className="w-5 h-5" />
              {item.badge && (
                <Badge 
                  variant="destructive" 
                  className="absolute -top-1 -right-1 h-5 w-5 p-0 text-xs flex items-center justify-center"
                >
                  {item.badge > 99 ? '99+' : item.badge}
                </Badge>
              )}
            </Button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="w-64 bg-card border-r flex flex-col">
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-12 h-12 rounded-full overflow-hidden">
              <img src="org-gpt.png" alt="Logo" className="w-full h-full object-cover" />
            </div>
            <div>
              <h1 className="font-bold text-lg">OrgGPT</h1>
              <p className="text-xs text-muted-foreground">Advanced RAG Chatbot</p>
            </div>
          </div>
          
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleCollapse}
            className="w-8 h-8 p-0"
          >
            <ChevronLeft className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Navigation */}
      <div className="flex-1 p-4">
        <div className="space-y-2">
          {menuItems.map((item) => (
            <Button
              key={item.id}
              variant={activeTab === item.id ? "default" : "ghost"}
              className="w-full justify-start"
              onClick={() => onTabChange(item.id)}
            >
              <item.icon className="w-4 h-4 mr-3" />
              {item.label}
              {item.badge && (
                <Badge variant="secondary" className="ml-auto">
                  {item.badge > 99 ? '99+' : item.badge}
                </Badge>
              )}
            </Button>
          ))}
        </div>

        {/* Session Actions */}
        <div className="mt-8 space-y-2">
          <Button
            variant="outline"
            className="w-full justify-start"
            onClick={onNewSession}
          >
            <Plus className="w-4 h-4 mr-3" />
            New Session
          </Button>

          <AlertDialog open={showClearDialog} onOpenChange={setShowClearDialog}>
            <AlertDialogTrigger asChild>
              <Button
                variant="outline"
                className="w-full justify-start text-destructive hover:text-destructive"
              >
                <Trash2 className="w-4 h-4 mr-3" />
                Clear Session
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Clear Session</AlertDialogTitle>
                <AlertDialogDescription>
                  This will delete all messages and documents in the current session. 
                  This action cannot be undone.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  onClick={handleClearSession}
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                >
                  Clear Session
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t">
        <div className="text-xs text-muted-foreground space-y-1">
          <div className="flex justify-between">
            <span>Documents:</span>
            <span>{documentCount}/10</span>
          </div>
          <div className="flex justify-between">
            <span>Messages:</span>
            <span>{messageCount}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;

