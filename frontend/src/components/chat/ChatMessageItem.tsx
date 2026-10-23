import React from 'react';
import ReactMarkdown from 'react-markdown';
import { ChatMessage, Citation } from '../../types';
import { Bot, User, Bookmark } from 'lucide-react';

interface ChatMessageItemProps {
  message: ChatMessage;
  onCitationClick: (citation: Citation) => void;
}

export const ChatMessageItem: React.FC<ChatMessageItemProps> = ({
  message,
  onCitationClick,
}) => {
  const isUser = message.role === 'user';

  return (
    <div className={`flex w-full ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`flex max-w-[85%] items-start gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {/* Avatar */}
        <div
          className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 shadow-sm ${
            isUser
              ? 'bg-indigo-600 text-white'
              : 'bg-white dark:bg-gray-800 text-indigo-600 dark:text-indigo-400 border border-gray-200 dark:border-gray-700'
          }`}
        >
          {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
        </div>

        {/* Message Bubble */}
        <div
          className={`rounded-2xl px-5 py-3.5 text-sm shadow-sm transition-all ${
            isUser
              ? 'bg-gradient-to-r from-indigo-600 to-indigo-700 text-white rounded-tr-none'
              : 'bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 text-gray-800 dark:text-gray-200 rounded-tl-none'
          }`}
        >
          {/* Content */}
          <div className="prose dark:prose-invert max-w-none text-sm leading-relaxed break-words">
            <ReactMarkdown
              components={{
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                ul: ({ children }) => <ul className="list-disc pl-4 mb-2 space-y-1">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal pl-4 mb-2 space-y-1">{children}</ol>,
                code: ({ children }) => (
                  <code className="bg-gray-100 dark:bg-gray-800 text-indigo-600 dark:text-indigo-400 px-1.5 py-0.5 rounded text-xs font-mono">
                    {children}
                  </code>
                ),
                pre: ({ children }) => (
                  <pre className="bg-gray-900 text-gray-100 p-3 rounded-xl overflow-x-auto my-2 text-xs font-mono">
                    {children}
                  </pre>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>

          {/* Interactive Citation Badges [1], [2] */}
          {message.citations && message.citations.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-800 flex flex-wrap items-center gap-1.5">
              <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider flex items-center gap-1 mr-1">
                <Bookmark className="w-3 h-3 text-indigo-500" />
                Fontes:
              </span>
              {message.citations.map((c) => (
                <button
                  key={c.id}
                  onClick={() => onCitationClick(c)}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-indigo-50 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300 text-xs font-medium hover:bg-indigo-100 dark:hover:bg-indigo-900/60 border border-indigo-200/60 dark:border-indigo-800/60 transition-all shadow-2xs hover:scale-105 active:scale-95"
                >
                  <span className="font-bold text-indigo-600 dark:text-indigo-400">[{c.id}]</span>
                  <span className="truncate max-w-[150px]">{c.filename}</span>
                  <span className="text-gray-400 dark:text-gray-500 text-[10px]">p.{c.page_number}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
