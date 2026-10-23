import React from 'react';
import { ChatSession, DocumentItem } from '../../types';
import { DocumentUpload } from './DocumentUpload';
import { DocumentList } from './DocumentList';
import { MessageSquare, Plus, BookOpen, Layers } from 'lucide-react';

interface SidebarProps {
  sessions: ChatSession[];
  activeSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  documents: DocumentItem[];
  onUploadSuccess: (newDoc: DocumentItem) => void;
  onDeleteDocument: (id: string) => Promise<void>;
}

export const Sidebar: React.FC<SidebarProps> = ({
  sessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  documents,
  onUploadSuccess,
  onDeleteDocument,
}) => {
  return (
    <aside className="w-80 h-full border-r border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 flex flex-col shrink-0">
      {/* Brand & New Chat */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center text-white shadow-md shadow-indigo-500/20">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h1 className="font-bold text-base tracking-tight text-gray-900 dark:text-white leading-tight">
              Modular RAG
            </h1>
            <span className="text-[10px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-400">
              Enterprise AI
            </span>
          </div>
        </div>

        <button
          onClick={onNewSession}
          title="Nova Conversa"
          className="p-2 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 hover:bg-indigo-100 dark:hover:bg-indigo-900/60 transition-colors shadow-sm"
        >
          <Plus className="w-4 h-4" />
        </button>
      </div>

      {/* Sessions List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-1">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-gray-400 px-2 py-1.5">
          Conversas ({sessions.length})
        </div>
        {sessions.length === 0 ? (
          <div className="p-4 text-center text-xs text-gray-400">
            Nenhuma conversa ativa
          </div>
        ) : (
          sessions.map((sess) => {
            const isActive = activeSessionId === sess.id;
            return (
              <button
                key={sess.id}
                onClick={() => onSelectSession(sess.id)}
                className={`w-full flex items-center space-x-2.5 px-3 py-2.5 rounded-xl text-xs font-medium text-left transition-all ${
                  isActive
                    ? 'bg-indigo-50 dark:bg-indigo-950/70 text-indigo-600 dark:text-indigo-400 font-semibold shadow-sm border border-indigo-100 dark:border-indigo-900/40'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800/60'
                }`}
              >
                <MessageSquare className={`w-4 h-4 shrink-0 ${isActive ? 'text-indigo-600 dark:text-indigo-400' : 'text-gray-400'}`} />
                <span className="truncate flex-1">{sess.title}</span>
              </button>
            );
          })
        )}
      </div>

      {/* Documents Section */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-800 bg-gray-50/50 dark:bg-gray-900/50 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-1.5 text-xs font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400">
            <BookOpen className="w-3.5 h-3.5 text-indigo-500" />
            <span>Documentos ({documents.length})</span>
          </div>
        </div>

        <DocumentUpload onUploadSuccess={onUploadSuccess} />

        <DocumentList documents={documents} onDelete={onDeleteDocument} />
      </div>
    </aside>
  );
};
