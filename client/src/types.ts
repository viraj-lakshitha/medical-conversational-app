export interface User {
  username: string;
}

export interface Message {
  sender: "user" | "bot";
  text: string;
  timestamp: string;
}

export interface Conversation {
  _id: string;
  latest: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface SignupRequest {
  username: string;
  password: string;
}

export interface ChatRequest {
  query: string;
  conversation_id?: string;
}

export interface MedicalEntity {
  concept_name?: string;
  display_name?: string;
  text: string;
  concept_type: 'DISEASE' | 'SYMPTOM';
  confidence_score: number;
}

export interface ChatResponse {
  response: string;
  conversation_id: string;
  metadata?: {
    entities_found: number;
    confidence: number;
    processing_time: number;
    warnings: string[];
    llm_method?: string;
    llm_confidence?: number;
    safety_flags?: number;
  };
}

export interface SystemHealth {
  api_status: string;
  components_loaded: Record<string, boolean>;
  advanced_features: Record<string, boolean>;
  configuration: Record<string, boolean>;
  connection_urls: Record<string, string>;
}

export interface PerformanceMetrics {
  performance: {
    total_requests: number;
    success_rate: number;
    avg_response_time: number;
    median_response_time: number;
    p95_response_time: number;
  };
  medical_accuracy: {
    total_queries: number;
    avg_confidence: number;
    avg_uncertainty: number;
    high_confidence_queries: number;
    low_confidence_queries: number;
  };
  errors: {
    total_errors: number;
    errors_by_component: Record<string, number>;
    recent_errors: Array<any>;
  };
  cache_stats: {
    hit_rate: number;
    total_requests: number;
  };
}

export interface LLMInfo {
  is_loaded: boolean;
  load_error?: string;
  model_path: string;
  model_type: string;
  config: Record<string, any>;
}

export interface ConversationContext {
  context_summary: string;
  conversation_length: number;
  entity_timeline: Record<string, any>;
  topic_tracking: Record<string, any>;
}

export interface ConversationsResponse {
  conversations: Conversation[];
}

export interface ConversationResponse {
  messages: Message[];
}
