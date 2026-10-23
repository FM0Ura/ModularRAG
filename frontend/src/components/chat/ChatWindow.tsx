import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage, Citation } from '../../types';
import { ChatMessageItem } from './ChatMessageItem';
import { Send, Bot, Sparkles, Loader2, ArrowDown } from 'lucide-react';

interface ChatWindowProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  onSendMessage: (message: string) => void;
  onCitationClick: (citation: Citation) => void;
  sessionTitle?: string;
}

const SAMPLE_QUESTIONS = [
  'Qual é o tema principal abordado nos documentos indexados?',
  'Resuma as diretrizes e regras apresentadas.',
  'Quais são os requisitos técnicos especificados?',
];

export const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  isStreaming,
  onSendMessage,
  onCitationClick,
  sessionTitle,
}) => {
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [showScrollBottom, setShowScrollBottom] = useState(false);

  const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  };

  useEffect(() => {
    scrollToBottom('smooth');
  }, [messages, isStreaming]);

  const handleScroll = () => {
    if (!scrollContainerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const isFarUp = scrollHeight - scrollTop - clientHeight > 150;
    setShowScrollBottom(isFarUp);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isStreaming) return;
    onSendMessage(inputText.trim());
    setInputText('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-50/50 dark:bg-gray-950 relative overflow-hidden">
      {/* Session Title Header */}
      {sessionTitle && (
        <div className="h-12 border-b border-gray-200 dark:border-gray-800 bg-white/50 dark:bg-gray-900/50 px-6 flex items-center justify-between shrink-0">
          <div className="flex items-center space-x-2">
            <span className="w-2 h-2 rounded-full bg-indigo-500" />
            <h2 className="text-xs font-semibold text-gray-700 dark:text-gray-300 truncate max-w-md">
              {sessionTitle}
            </h2>
          </div>
          {isStreaming && (
            <div className="flex items-center space-x-1.5 text-xs text-indigo-600 dark:text-indigo-400 font-medium">
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span>Gerando resposta...</span>
            </div>
          )}
        </div>
      )}

      {/* Messages Scroll Area */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4 max-w-4xl w-full mx-auto"
      >
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-8 space-y-6">
            <div className="w-16 h-16 rounded-3xl bg-gradient-to-tr from-indigo-500 to-violet-600 flex items-center justify-center text-white shadow-xl shadow-indigo-500/20">
              <Bot className="w-8 h-8" />
            </div>

            <div className="max-w-md space-y-2">
              <h3 className="text-lg font-bold text-gray-900 dark:text-white">
                Como posso ajudar hoje?
              </h3>
              <p className="text-xs sm:text-sm text-gray-500 dark:text-gray-400">
                Faça perguntas sobre os documentos indexados. Suas respostas incluirão citações com pontuação de relevância.
              </p>
            </div>

            <div className="w-full max-w-md space-y-2 pt-2">
              <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider block">
                Sugestões de perguntas:
              </span>
              <div className="space-y-2">
                {SAMPLE_QUESTIONS.map((q, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setInputText(q);
                    }}
                    className="w-full p-2.5 rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 text-left text-xs text-gray-700 dark:text-gray-300 hover:border-indigo-300 dark:hover:border-indigo-700 hover:text-indigo-600 dark:hover:text-indigo-400 transition-all flex items-center justify-between group shadow-2xs"
                  >
                    <span>{q}</span>
                    <Sparkles className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 text-indigo-500 transition-opacity" />
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <ChatMessageItem
                key={message.id}
                message={message}
                onCitationClick={onCitationClick}
              />
            ))}

            {/* Pulsing indicator when assistant is waiting for first token */}
            {isStreaming &&
              messages.length > 0 &&
              messages[messages.length - 1].role === 'assistant' &&
              !messages[messages.length - 1].content && (
                <div className="flex items-center space-x-2 text-xs text-gray-400 italic pl-12 py-2">
                  <div className="w-2 h-2 rounded-full bg-indigo-500 animate-ping" />
                  <span>Consultando fontes e sintetizando...</span>
                </div>
              )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* Floating Scroll to Bottom Button */}
      {showScrollBottom && (
        <button
          onClick={() => scrollToBottom('smooth')}
          aria-label="Rolar para o final"
          className="absolute bottom-24 right-8 p-2 rounded-full bg-indigo-600 text-white shadow-lg hover:bg-indigo-700 transition-all z-10"
        >
          <ArrowDown className="w-4 h-4" />
        </button>
      )}

      {/* Input Form */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 shrink-0">
        <form
          onSubmit={handleSubmit}
          className="max-w-4xl mx-auto flex items-end gap-2 relative"
        >
          <div className="flex-1 relative">
            <textarea
              rows={1}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Faça uma pergunta sobre seus documentos... (Enter para enviar)"
              disabled={isStreaming}
              className="w-full resize-none rounded-2xl border border-gray-300 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-800/50 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 dark:focus:ring-indigo-400 focus:bg-white dark:focus:bg-gray-900 text-gray-900 dark:text-gray-100 disabled:opacity-60 transition-all placeholder:text-gray-400"
            />
          </div>

          <button
            type="submit"
            disabled={isStreaming || !inputText.trim()}
            aria-label="Enviar mensagem"
            className="p-3 rounded-2xl bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-40 disabled:hover:bg-indigo-600 transition-colors shadow-sm flex items-center justify-center shrink-0"
          >
            {isStreaming ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </form>
      </div>
    </div>
  );
};
