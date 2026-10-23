import { useState, useEffect, useCallback } from 'react';
import { api } from './services/api';
import { DocumentItem, ChatSession, ChatMessage, Citation, SystemHealth } from './types';
import { Header } from './components/layout/Header';
import { Sidebar } from './components/sidebar/Sidebar';
import { ChatWindow } from './components/chat/ChatWindow';
import { SourceDrawer } from './components/citations/SourceDrawer';

export default function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);

  const loadInitialData = useCallback(async () => {
    try {
      const [docs, sessList, health] = await Promise.all([
        api.listDocuments().catch(() => []),
        api.listSessions().catch(() => []),
        api.getHealth().catch(() => null),
      ]);

      setDocuments(docs);
      setSessions(sessList);
      setSystemHealth(health);

      if (sessList.length > 0) {
        setActiveSessionId(sessList[0].id);
      } else {
        const newSess = await api.createSession('Primeira Conversa');
        setSessions([newSess]);
        setActiveSessionId(newSess.id);
      }
    } catch (err) {
      console.error('Erro ao carregar dados iniciais:', err);
    }
  }, []);

  const loadSessionMessages = useCallback(async (sessionId: string) => {
    try {
      const msgs = await api.getMessages(sessionId);
      setMessages(msgs);
    } catch (err) {
      console.error('Erro ao carregar mensagens da sessão:', err);
      setMessages([]);
    }
  }, []);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  useEffect(() => {
    if (activeSessionId) {
      loadSessionMessages(activeSessionId);
    }
  }, [activeSessionId, loadSessionMessages]);

  const handleNewSession = async () => {
    try {
      const newTitle = `Conversa ${sessions.length + 1}`;
      const sess = await api.createSession(newTitle);
      setSessions((prev) => [sess, ...prev]);
      setActiveSessionId(sess.id);
      setMessages([]);
    } catch (err) {
      console.error('Falha ao criar nova conversa:', err);
    }
  };

  const handleUploadSuccess = (newDoc: DocumentItem) => {
    setDocuments((prev) => [newDoc, ...prev]);
  };

  const handleDeleteDocument = async (id: string) => {
    try {
      await api.deleteDocument(id);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
    } catch (err: any) {
      alert(`Erro ao excluir documento: ${err.message}`);
    }
  };

  const handleSendMessage = (userText: string) => {
    if (!activeSessionId || isStreaming) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: userText,
      created_at: new Date().toISOString(),
    };

    const tempAssistantId = `assistant-${Date.now()}`;
    const initialAssistantMessage: ChatMessage = {
      id: tempAssistantId,
      role: 'assistant',
      content: '',
      citations: [],
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage, initialAssistantMessage]);
    setIsStreaming(true);

    api.streamChat(
      activeSessionId,
      userText,
      (citations) => {
        setMessages((prev) =>
          prev.map((msg) => (msg.id === tempAssistantId ? { ...msg, citations } : msg))
        );
      },
      (token) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === tempAssistantId ? { ...msg, content: msg.content + token } : msg
          )
        );
      },
      () => {
        setIsStreaming(false);
      },
      (err) => {
        setIsStreaming(false);
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === tempAssistantId
              ? {
                  ...msg,
                  content: msg.content
                    ? `${msg.content}\n\n⚠️ *[Erro de streaming: ${err}]*`
                    : `⚠️ *Não foi possível obter resposta: ${err}*`,
                }
              : msg
          )
        );
      }
    );
  };

  const activeSession = sessions.find((s) => s.id === activeSessionId);

  return (
    <div className="flex h-screen w-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100 font-sans overflow-hidden">
      {/* Sidebar: Sessions & Documents */}
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={setActiveSessionId}
        onNewSession={handleNewSession}
        documents={documents}
        onUploadSuccess={handleUploadSuccess}
        onDeleteDocument={handleDeleteDocument}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full min-w-0">
        {/* Header: System status bar */}
        <Header systemHealth={systemHealth} />

        {/* Chat Window */}
        <ChatWindow
          messages={messages}
          isStreaming={isStreaming}
          onSendMessage={handleSendMessage}
          onCitationClick={setActiveCitation}
          sessionTitle={activeSession?.title}
        />
      </div>

      {/* Citations Flyout Drawer */}
      <SourceDrawer
        isOpen={Boolean(activeCitation)}
        onClose={() => setActiveCitation(null)}
        citation={activeCitation}
      />
    </div>
  );
}
