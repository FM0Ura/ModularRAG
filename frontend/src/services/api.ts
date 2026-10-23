import { DocumentItem, ChatSession, ChatMessage, Citation, SystemHealth } from '../types';

const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export const api = {
  async getHealth(): Promise<SystemHealth> {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Erro ao obter status do sistema');
    return res.json();
  },

  async listDocuments(): Promise<DocumentItem[]> {
    const res = await fetch(`${API_BASE}/documents`);
    if (!res.ok) throw new Error('Erro ao listar documentos');
    return res.json();
  },

  async uploadDocument(file: File): Promise<DocumentItem> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/documents/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      let detail = 'Falha no upload';
      try {
        const err = await res.json();
        detail = err.detail || detail;
      } catch {
        // use default detail
      }
      throw new Error(detail);
    }
    return res.json();
  },

  async deleteDocument(id: string): Promise<void> {
    const res = await fetch(`${API_BASE}/documents/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Erro ao deletar documento');
  },

  async listSessions(): Promise<ChatSession[]> {
    const res = await fetch(`${API_BASE}/chat/sessions`);
    if (!res.ok) throw new Error('Erro ao listar conversas');
    return res.json();
  },

  async createSession(title?: string): Promise<ChatSession> {
    const res = await fetch(`${API_BASE}/chat/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    if (!res.ok) throw new Error('Erro ao criar conversa');
    return res.json();
  },

  async getMessages(sessionId: string): Promise<ChatMessage[]> {
    const res = await fetch(`${API_BASE}/chat/sessions/${sessionId}/messages`);
    if (!res.ok) throw new Error('Erro ao carregar mensagens');
    return res.json();
  },

  streamChat(
    sessionId: string,
    question: string,
    onSources: (citations: Citation[]) => void,
    onToken: (token: string) => void,
    onDone: () => void,
    onError: (err: string) => void
  ): () => void {
    const controller = new AbortController();

    fetch(`${API_BASE}/chat/${sessionId}/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Falha ao iniciar streaming (${response.status})`);
        }
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        if (!reader) {
          throw new Error('Corpo da resposta indisponível para streaming');
        }

        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split(/\r?\n\r?\n/);
          buffer = blocks.pop() || '';

          for (const block of blocks) {
            if (!block.trim()) continue;

            let eventType = 'message';
            let dataStr = '';

            for (const line of block.split(/\r?\n/)) {
              if (line.startsWith('event:')) {
                eventType = line.slice(6).trim();
              } else if (line.startsWith('data:')) {
                dataStr += (dataStr ? '\n' : '') + line.slice(5).trim();
              }
            }

            if (dataStr) {
              try {
                const data = JSON.parse(dataStr);
                if (eventType === 'sources') {
                  onSources(data.citations || []);
                } else if (eventType === 'token') {
                  onToken(data.delta || '');
                } else if (eventType === 'done') {
                  onDone();
                } else if (eventType === 'error') {
                  onError(data.error || 'Erro desconhecido durante geração');
                }
              } catch (e) {
                console.error('Erro de parse SSE:', e, dataStr);
              }
            }
          }
        }
        onDone();
      })
      .catch((err) => {
        if (err.name !== 'AbortError') {
          onError(err.message || 'Erro de conexão no streaming');
        }
      });

    return () => controller.abort();
  },
};
