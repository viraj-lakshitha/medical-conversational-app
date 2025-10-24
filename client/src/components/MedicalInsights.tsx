import React, { useState, useEffect } from 'react';
import type { FC } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Tag,
  Progress,
  Alert,
  Divider,
  Tooltip,
  Badge,
  Space
} from 'antd';
import {
  HeartOutlined,
  ExperimentOutlined,
  SafetyOutlined,
  ThunderboltOutlined,
  RobotOutlined,
  DatabaseOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import { ChatResponse, MedicalEntity, SystemHealth, PerformanceMetrics } from '../types';
import { medicalApi } from '../api/medicalApi';

interface MedicalInsightsProps {
  lastResponse?: ChatResponse;
  onEntityClick?: (entity: MedicalEntity) => void;
}

const MedicalInsights: React.FC<MedicalInsightsProps> = ({ lastResponse, onEntityClick }) => {
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [entities, setEntities] = useState<MedicalEntity[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadSystemData();
    const interval = setInterval(loadSystemData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (lastResponse?.metadata) {
      extractEntitiesFromLastMessage();
    }
  }, [lastResponse]);

  const loadSystemData = async () => {
    try {
      const [healthData, metricsData] = await Promise.all([
        medicalApi.getSystemHealth(),
        medicalApi.getMetrics(1)
      ]);
      setSystemHealth(healthData);
      setMetrics(metricsData);
    } catch (error) {
      console.error('Failed to load system data:', error);
    }
  };

  const extractEntitiesFromLastMessage = async () => {
    if (!lastResponse?.response) return;
    
    setLoading(true);
    try {
      const result = await medicalApi.extractEntities(lastResponse.response);
      setEntities(result.entities);
    } catch (error) {
      console.error('Failed to extract entities:', error);
    } finally {
      setLoading(false);
    }
  };

  const getHealthStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return '#52c41a';
      case 'degraded': return '#faad14';
      case 'unhealthy': return '#ff4d4f';
      default: return '#d9d9d9';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return '#52c41a';
    if (confidence >= 0.6) return '#faad14';
    return '#ff4d4f';
  };

  const renderSystemHealth = () => {
    if (!systemHealth) return null;

    const componentsLoaded = Object.entries(systemHealth.components_loaded || {});
    const advancedFeatures = Object.entries(systemHealth.advanced_features || {});

    return (
      <Card 
        title={
          <Space>
            <HeartOutlined />
            System Health
            <Badge 
              status={systemHealth.api_status === 'healthy' ? 'success' : 'warning'} 
              text={systemHealth.api_status}
            />
          </Space>
        } 
        size="small"
      >
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Card size="small" title="Core Components">
              {componentsLoaded.map(([component, loaded]) => (
                <div key={component} style={{ marginBottom: 8 }}>
                  <Space>
                    {loaded ? (
                      <CheckCircleOutlined style={{ color: '#52c41a' }} />
                    ) : (
                      <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
                    )}
                    <span style={{ textTransform: 'capitalize' }}>{component}</span>
                  </Space>
                </div>
              ))}
            </Card>
          </Col>
          <Col span={12}>
            <Card size="small" title="Advanced Features">
              {advancedFeatures.map(([feature, enabled]) => (
                <div key={feature} style={{ marginBottom: 8 }}>
                  <Space>
                    {enabled ? (
                      <CheckCircleOutlined style={{ color: '#52c41a' }} />
                    ) : (
                      <ExclamationCircleOutlined style={{ color: '#d9d9d9' }} />
                    )}
                    <span style={{ textTransform: 'capitalize' }}>
                      {feature.replace('_', ' ')}
                    </span>
                  </Space>
                </div>
              ))}
            </Card>
          </Col>
        </Row>
      </Card>
    );
  };

  const renderPerformanceMetrics = () => {
    if (!metrics) return null;

    return (
      <Card 
        title={
          <Space>
            <ThunderboltOutlined />
            Performance Metrics
          </Space>
        } 
        size="small"
      >
        <Row gutter={[16, 8]}>
          <Col span={8}>
            <Statistic
              title="Success Rate"
              value={metrics.performance?.success_rate || 0}
              precision={1}
              suffix="%"
              valueStyle={{ color: getHealthStatusColor('healthy') }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Avg Response Time"
              value={metrics.performance?.avg_response_time || 0}
              precision={2}
              suffix="s"
              valueStyle={{ 
                color: (metrics.performance?.avg_response_time || 0) < 2 ? '#52c41a' : '#faad14' 
              }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Cache Hit Rate"
              value={metrics.cache_stats?.hit_rate || 0}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
        </Row>
        
        <Divider />
        
        <Row gutter={[16, 8]}>
          <Col span={8}>
            <Statistic
              title="Medical Queries"
              value={metrics.medical_accuracy?.total_queries || 0}
              prefix={<ExperimentOutlined />}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Avg Confidence"
              value={metrics.medical_accuracy?.avg_confidence || 0}
              precision={1}
              suffix="%"
              valueStyle={{ 
                color: getConfidenceColor((metrics.medical_accuracy?.avg_confidence || 0) / 100) 
              }}
            />
          </Col>
          <Col span={8}>
            <Statistic
              title="Total Errors"
              value={metrics.errors?.total_errors || 0}
              valueStyle={{ color: (metrics.errors?.total_errors || 0) > 0 ? '#ff4d4f' : '#52c41a' }}
            />
          </Col>
        </Row>
      </Card>
    );
  };

  const renderLastResponseAnalysis = () => {
    if (!lastResponse?.metadata) return null;

    const { metadata } = lastResponse;

    return (
      <Card 
        title={
          <Space>
            <RobotOutlined />
            Last Response Analysis
          </Space>
        } 
        size="small"
      >
        <Row gutter={[16, 8]}>
          <Col span={6}>
            <Statistic
              title="Entities Found"
              value={metadata.entities_found}
              prefix={<DatabaseOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Confidence"
              value={metadata.confidence * 100}
              precision={1}
              suffix="%"
              valueStyle={{ color: getConfidenceColor(metadata.confidence) }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Processing Time"
              value={metadata.processing_time}
              precision={2}
              suffix="s"
              prefix={<ClockCircleOutlined />}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="Safety Flags"
              value={metadata.safety_flags || 0}
              valueStyle={{ 
                color: (metadata.safety_flags || 0) > 0 ? '#faad14' : '#52c41a' 
              }}
              prefix={<SafetyOutlined />}
            />
          </Col>
        </Row>

        {metadata.llm_method && (
          <>
            <Divider />
            <Row gutter={[16, 8]}>
              <Col span={12}>
                <div>
                  <strong>LLM Method:</strong>{' '}
                  <Tag color={metadata.llm_method === 'llm_generated' ? 'green' : 'orange'}>
                    {metadata.llm_method}
                  </Tag>
                </div>
              </Col>
              {metadata.llm_confidence && (
                <Col span={12}>
                  <div>
                    <strong>LLM Confidence:</strong>{' '}
                    <Progress 
                      percent={metadata.llm_confidence * 100} 
                      size="small" 
                      status={metadata.llm_confidence > 0.7 ? 'success' : 'normal'}
                    />
                  </div>
                </Col>
              )}
            </Row>
          </>
        )}

        {metadata.warnings && metadata.warnings.length > 0 && (
          <>
            <Divider />
            <Alert
              message="Medical Warnings"
              description={
                <ul style={{ margin: 0, paddingLeft: 20 }}>
                  {metadata.warnings.map((warning, index) => (
                    <li key={index}>{warning}</li>
                  ))}
                </ul>
              }
              type="warning"
              showIcon
            />
          </>
        )}
      </Card>
    );
  };

  const renderExtractedEntities = () => {
    if (!entities.length) return null;

    const diseases = entities.filter(e => e.concept_type === 'DISEASE');
    const symptoms = entities.filter(e => e.concept_type === 'SYMPTOM');

    return (
      <Card 
        title={
          <Space>
            <ExperimentOutlined />
            Extracted Medical Entities
          </Space>
        } 
        size="small"
        loading={loading}
      >
        {diseases.length > 0 && (
          <div style={{ marginBottom: 16 }}>
            <h4>Diseases:</h4>
            <Space wrap>
              {diseases.map((entity, index) => (
                <Tooltip key={index} title={`Confidence: ${(entity.confidence_score * 100).toFixed(1)}%`}>
                  <Tag 
                    color="red" 
                    style={{ cursor: 'pointer' }}
                    onClick={() => onEntityClick?.(entity)}
                  >
                    {entity.display_name || entity.text}
                  </Tag>
                </Tooltip>
              ))}
            </Space>
          </div>
        )}

        {symptoms.length > 0 && (
          <div>
            <h4>Symptoms:</h4>
            <Space wrap>
              {symptoms.map((entity, index) => (
                <Tooltip key={index} title={`Confidence: ${(entity.confidence_score * 100).toFixed(1)}%`}>
                  <Tag 
                    color="blue" 
                    style={{ cursor: 'pointer' }}
                    onClick={() => onEntityClick?.(entity)}
                  >
                    {entity.display_name || entity.text}
                  </Tag>
                </Tooltip>
              ))}
            </Space>
          </div>
        )}
      </Card>
    );
  };

  return (
    <div style={{ padding: '16px 0' }}>
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        {renderSystemHealth()}
        {renderPerformanceMetrics()}
        {renderLastResponseAnalysis()}
        {renderExtractedEntities()}
      </Space>
    </div>
  );
};

export default MedicalInsights;