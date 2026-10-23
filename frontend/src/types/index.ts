export interface DocumentItem {
  id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  total_pages: number;
  created_at: string;
}

export interface Citation {
  id: number;
  filename: string;
  page_number: number;
  score: number;
  snippet: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  created_at?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
}

export interface SystemHealth {
  status: string;
  llm_provider: string;
  embedding_provider: string;
  reranker_threshold: number;
}
