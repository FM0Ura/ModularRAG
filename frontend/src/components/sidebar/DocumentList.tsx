import React, { useState } from 'react';
import { DocumentItem } from '../../types';
import { FileText, Trash2, Loader2, BookOpen } from 'lucide-react';

interface DocumentListProps {
  documents: DocumentItem[];
  onDelete: (id: string) => Promise<void>;
}

export const DocumentList: React.FC<DocumentListProps> = ({ documents, onDelete }) => {
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const formatFileSize = (bytes: number): string => {
    if (!bytes || bytes <= 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
  };

  const handleDelete = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (deletingId) return;
    setDeletingId(id);
    try {
      await onDelete(id);
    } finally {
      setDeletingId(null);
    }
  };

  if (documents.length === 0) {
    return (
      <div className="py-6 px-3 text-center rounded-xl bg-gray-50/50 dark:bg-gray-800/20 border border-dashed border-gray-200 dark:border-gray-800">
        <BookOpen className="w-6 h-6 mx-auto text-gray-400 mb-2 stroke-[1.5]" />
        <p className="text-xs font-medium text-gray-600 dark:text-gray-400">
          Nenhum documento indexado
        </p>
        <p className="text-[11px] text-gray-400 dark:text-gray-500 mt-1">
          Faça upload para alimentar o banco vetorial
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
      {documents.map((doc) => {
        const isDeleting = deletingId === doc.id;
        return (
          <div
            key={doc.id}
            className="group flex items-center justify-between p-2.5 rounded-xl bg-gray-50 dark:bg-gray-800/60 border border-gray-100 dark:border-gray-800 hover:border-gray-300 dark:hover:border-gray-700 transition-all text-xs"
          >
            <div className="flex items-center space-x-2.5 min-w-0 flex-1 pr-2">
              <div className="p-1.5 rounded-lg bg-indigo-50 dark:bg-indigo-950/70 text-indigo-600 dark:text-indigo-400 shrink-0">
                <FileText className="w-3.5 h-3.5" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="font-medium text-gray-800 dark:text-gray-200 truncate" title={doc.filename}>
                  {doc.filename}
                </p>
                <div className="flex items-center space-x-2 text-[11px] text-gray-500 dark:text-gray-400 mt-0.5">
                  <span>{doc.total_pages} {doc.total_pages === 1 ? 'página' : 'páginas'}</span>
                  <span>•</span>
                  <span>{formatFileSize(doc.file_size_bytes)}</span>
                </div>
              </div>
            </div>

            <button
              onClick={(e) => handleDelete(doc.id, e)}
              disabled={isDeleting}
              title="Excluir documento"
              className="p-1.5 rounded-lg text-gray-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-950/50 transition-colors shrink-0 disabled:opacity-50"
            >
              {isDeleting ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin text-red-500" />
              ) : (
                <Trash2 className="w-3.5 h-3.5" />
              )}
            </button>
          </div>
        );
      })}
    </div>
  );
};
