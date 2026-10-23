import React from 'react';
import { Citation } from '../../types';
import { X, FileText, CheckCircle2, Bookmark } from 'lucide-react';

interface SourceDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  citation: Citation | null;
}

export const SourceDrawer: React.FC<SourceDrawerProps> = ({ isOpen, onClose, citation }) => {
  if (!isOpen || !citation) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/30 backdrop-blur-sm z-40 transition-opacity"
        onClick={onClose}
      />

      {/* Flyout Drawer */}
      <aside className="fixed inset-y-0 right-0 w-full sm:w-[420px] bg-white dark:bg-gray-900 shadow-2xl border-l border-gray-200 dark:border-gray-800 p-6 z-50 flex flex-col transition-all duration-300 ease-in-out">
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-gray-200 dark:border-gray-800">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 flex items-center justify-center font-bold text-xs">
              #{citation.id}
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-white text-base">
                Fonte Citada [{citation.id}]
              </h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                Evidência recuperada do banco vetorial
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Fechar gaveta de fonte"
            className="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body */}
        <div className="mt-5 space-y-4 flex-1 overflow-y-auto pr-1">
          {/* Metadata Card */}
          <div className="bg-gray-50 dark:bg-gray-800/50 p-4 rounded-xl border border-gray-200/80 dark:border-gray-800 space-y-3">
            <div>
              <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                Documento
              </span>
              <p className="text-sm font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-1.5 mt-0.5 break-all">
                <FileText className="w-4 h-4 text-indigo-500 shrink-0" />
                {citation.filename}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3 pt-2 border-t border-gray-200/60 dark:border-gray-700/60">
              <div>
                <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                  Página / Seção
                </span>
                <p className="text-sm font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-1 mt-0.5">
                  <Bookmark className="w-3.5 h-3.5 text-indigo-400" />
                  Página {citation.page_number}
                </p>
              </div>

              <div>
                <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                  Relevância Reranker
                </span>
                <p className="text-sm font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1 mt-0.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  {(citation.score * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>

          {/* Snippet Card */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-bold text-gray-400 uppercase tracking-wider">
                Trecho Original do Documento
              </span>
              <span className="text-[11px] text-indigo-500 font-medium">Texto Literal</span>
            </div>
            <div className="p-4 rounded-xl bg-indigo-50/50 dark:bg-indigo-950/20 border border-indigo-100 dark:border-indigo-900/50 text-xs sm:text-sm text-gray-700 dark:text-gray-300 leading-relaxed font-mono whitespace-pre-wrap selection:bg-indigo-200 dark:selection:bg-indigo-900">
              "{citation.snippet}"
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};
