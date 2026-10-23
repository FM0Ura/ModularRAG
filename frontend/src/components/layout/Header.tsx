import React from 'react';
import { SystemHealth } from '../../types';
import { Cpu, Database, Activity, Sparkles } from 'lucide-react';

interface HeaderProps {
  systemHealth: SystemHealth | null;
}

export const Header: React.FC<HeaderProps> = ({ systemHealth }) => {
  return (
    <header className="h-16 border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md px-6 flex items-center justify-between z-10">
      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-2">
          <span className="text-sm font-semibold tracking-wide text-gray-900 dark:text-gray-100">
            Painel do Sistema
          </span>
        </div>
      </div>

      <div className="flex items-center space-x-4 text-xs font-medium">
        {systemHealth ? (
          <>
            <div className="flex items-center space-x-2 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-400 px-3 py-1.5 rounded-full border border-emerald-200 dark:border-emerald-900/60 shadow-sm">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="font-semibold uppercase tracking-wider text-[11px]">Sistema Operacional</span>
            </div>

            <div className="hidden sm:flex items-center space-x-1.5 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700">
              <Sparkles className="w-3.5 h-3.5 text-indigo-500" />
              <span className="text-gray-500 dark:text-gray-400">LLM:</span>
              <span className="font-semibold text-gray-900 dark:text-gray-100">{systemHealth.llm_provider}</span>
            </div>

            <div className="hidden md:flex items-center space-x-1.5 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700">
              <Database className="w-3.5 h-3.5 text-blue-500" />
              <span className="text-gray-500 dark:text-gray-400">Embeddings:</span>
              <span className="font-semibold text-gray-900 dark:text-gray-100">{systemHealth.embedding_provider}</span>
            </div>

            <div className="hidden lg:flex items-center space-x-1.5 bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700">
              <Cpu className="w-3.5 h-3.5 text-purple-500" />
              <span className="text-gray-500 dark:text-gray-400">Reranker:</span>
              <span className="font-semibold text-gray-900 dark:text-gray-100">
                FlashRank ({((systemHealth.reranker_threshold || 0.4) * 100).toFixed(0)}%)
              </span>
            </div>
          </>
        ) : (
          <div className="flex items-center space-x-2 text-gray-400">
            <Activity className="w-4 h-4 animate-spin text-gray-400" />
            <span>Conectando ao backend...</span>
          </div>
        )}
      </div>
    </header>
  );
};
