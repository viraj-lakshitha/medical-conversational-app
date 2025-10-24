import api from './axios';
import {
  SystemHealth,
  PerformanceMetrics,
  LLMInfo,
  ConversationContext,
  MedicalEntity
} from '../types';

export const medicalApi = {
  // System health and monitoring
  async getSystemHealth(): Promise<SystemHealth> {
    const response = await api.get<SystemHealth>('/api/v1/health');
    return response.data;
  },

  async getMetrics(hours: number = 1): Promise<PerformanceMetrics> {
    const response = await api.get<PerformanceMetrics>(`/api/v1/metrics?hours=${hours}`);
    return response.data;
  },

  async getDashboardData(): Promise<any> {
    const response = await api.get('/api/v1/analytics/dashboard');
    return response.data;
  },

  // Entity extraction
  async extractEntities(text: string): Promise<{ entities: MedicalEntity[]; count: number }> {
    const response = await api.post('/api/v1/entities', { text });
    return response.data;
  },

  // Document search
  async searchDocuments(query: string, maxResults: number = 5): Promise<{ documents: any[]; count: number }> {
    const response = await api.post('/api/v1/search', { query, max_results: maxResults });
    return response.data;
  },

  // Disease prediction
  async predictDiseases(symptoms: string[]): Promise<{ predictions: any; symptoms_analyzed: string[] }> {
    const response = await api.post('/api/v1/predict', { symptoms });
    return response.data;
  },

  // LLM information
  async getLLMInfo(): Promise<LLMInfo> {
    const response = await api.get<LLMInfo>('/api/v1/llm/info');
    return response.data;
  },

  // Conversation context
  async getConversationContext(): Promise<ConversationContext> {
    const response = await api.get<ConversationContext>('/api/v1/conversation/context');
    return response.data;
  },

  // Cache management
  async invalidateCache(component?: string, pattern?: string): Promise<{ message: string; deleted_count: number }> {
    const response = await api.post('/api/v1/cache/invalidate', { component, pattern });
    return response.data;
  }
};