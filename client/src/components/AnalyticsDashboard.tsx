import React, { useState, useEffect } from 'react';
import type { FC } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Button,
  Space,
  Alert,
  Tabs,
  Progress,
  Tag,
  message,
  Modal,
  Select
} from 'antd';
import {
  ReloadOutlined,
  DashboardOutlined,
  DeleteOutlined,
  InfoCircleOutlined,
  BarChartOutlined,
  SettingOutlined
} from '@ant-design/icons';
import { medicalApi } from '../api/medicalApi';
import { PerformanceMetrics, LLMInfo, ConversationContext } from '../types';

const { TabPane } = Tabs;
const { Option } = Select;

const AnalyticsDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [llmInfo, setLLMInfo] = useState<LLMInfo | null>(null);
  const [conversationContext, setConversationContext] = useState<ConversationContext | null>(null);
  const [loading, setLoading] = useState(false);
  const [metricsHours, setMetricsHours] = useState(1);
  const [cacheModalVisible, setCacheModalVisible] = useState(false);

  useEffect(() => {
    loadAllData();
  }, [metricsHours]);

  const loadAllData = async () => {
    setLoading(true);
    try {
      const [metricsData, llmData, contextData] = await Promise.all([
        medicalApi.getMetrics(metricsHours),
        medicalApi.getLLMInfo().catch(() => null),
        medicalApi.getConversationContext().catch(() => null)
      ]);
      
      setMetrics(metricsData);
      setLLMInfo(llmData);
      setConversationContext(contextData);
    } catch (error) {
      console.error('Failed to load analytics data:', error);
      message.error('Failed to load analytics data');
    } finally {
      setLoading(false);
    }
  };

  const handleCacheInvalidation = async (component?: string) => {
    try {
      const result = await medicalApi.invalidateCache(component);
      message.success(`${result.message}`);
      setCacheModalVisible(false);
      loadAllData(); // Refresh metrics
    } catch (error) {
      message.error('Failed to invalidate cache');
    }
  };

  const renderOverviewTab = () => {
    if (!metrics) return null;

    const performanceScore = (metrics.performance?.success_rate || 0);
    const confidenceScore = (metrics.medical_accuracy?.avg_confidence || 0) * 100;
    const cacheEfficiency = (metrics.cache_stats?.hit_rate || 0) * 100;

    return (
      <div>
        <Row gutter={[24, 24]}>
          {/* Key Performance Indicators */}
          <Col span={24}>
            <Card title="Key Performance Indicators">
              <Row gutter={16}>
                <Col span={6}>
                  <Statistic
                    title="System Performance"
                    value={performanceScore}
                    precision={1}
                    suffix="%"
                    valueStyle={{ 
                      color: performanceScore > 95 ? '#52c41a' : performanceScore > 85 ? '#faad14' : '#ff4d4f' 
                    }}
                  />
                  <Progress 
                    percent={performanceScore} 
                    size="small" 
                    status={performanceScore > 95 ? 'success' : 'normal'}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Medical Confidence"
                    value={confidenceScore}
                    precision={1}
                    suffix="%"
                    valueStyle={{ 
                      color: confidenceScore > 80 ? '#52c41a' : confidenceScore > 60 ? '#faad14' : '#ff4d4f' 
                    }}
                  />
                  <Progress 
                    percent={confidenceScore} 
                    size="small" 
                    status={confidenceScore > 80 ? 'success' : 'normal'}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Cache Efficiency"
                    value={cacheEfficiency}
                    precision={1}
                    suffix="%"
                    valueStyle={{ color: '#1890ff' }}
                  />
                  <Progress 
                    percent={cacheEfficiency} 
                    size="small" 
                    strokeColor="#1890ff"
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="Total Queries"
                    value={metrics.medical_accuracy?.total_queries || 0}
                    valueStyle={{ color: '#722ed1' }}
                  />
                </Col>
              </Row>
            </Card>
          </Col>

          {/* Performance Details */}
          <Col span={12}>
            <Card title="Response Time Analysis">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Statistic
                    title="Average"
                    value={metrics.performance?.avg_response_time || 0}
                    precision={2}
                    suffix="s"
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Median"
                    value={metrics.performance?.median_response_time || 0}
                    precision={2}
                    suffix="s"
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="95th Percentile"
                    value={metrics.performance?.p95_response_time || 0}
                    precision={2}
                    suffix="s"
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Success Rate"
                    value={metrics.performance?.success_rate || 0}
                    precision={1}
                    suffix="%"
                  />
                </Col>
              </Row>
            </Card>
          </Col>

          {/* Medical Accuracy */}
          <Col span={12}>
            <Card title="Medical Analysis Quality">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Statistic
                    title="High Confidence"
                    value={metrics.medical_accuracy?.high_confidence_queries || 0}
                    suffix="queries"
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="Low Confidence"
                    value={metrics.medical_accuracy?.low_confidence_queries || 0}
                    suffix="queries"
                    valueStyle={{ color: '#ff4d4f' }}
                  />
                </Col>
                <Col span={24}>
                  <div style={{ marginTop: 16 }}>
                    <strong>Confidence Distribution:</strong>
                    <Progress 
                      percent={
                        (metrics.medical_accuracy?.high_confidence_queries || 0) / 
                        (metrics.medical_accuracy?.total_queries || 1) * 100
                      }
                      size="small"
                      format={(percent) => `${percent?.toFixed(0)}% High Confidence`}
                    />
                  </div>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>

        {/* Error Analysis */}
        {metrics.errors && metrics.errors.total_errors > 0 && (
          <Card title="Error Analysis" style={{ marginTop: 24 }}>
            <Row gutter={16}>
              <Col span={8}>
                <Statistic
                  title="Total Errors"
                  value={metrics.errors.total_errors}
                  valueStyle={{ color: '#ff4d4f' }}
                />
              </Col>
              <Col span={16}>
                <div>
                  <strong>Errors by Component:</strong>
                  <div style={{ marginTop: 8 }}>
                    {Object.entries(metrics.errors.errors_by_component || {}).map(([component, count]) => (
                      <Tag key={component} color="red" style={{ margin: '2px' }}>
                        {component}: {count}
                      </Tag>
                    ))}
                  </div>
                </div>
              </Col>
            </Row>
          </Card>
        )}
      </div>
    );
  };

  const renderLLMTab = () => {
    if (!llmInfo) {
      return (
        <Alert
          message="LLM Information Unavailable"
          description="Unable to load LLM status information."
          type="warning"
          showIcon
        />
      );
    }

    return (
      <div>
        <Card title="Language Model Status">
          <Row gutter={[16, 16]}>
            <Col span={8}>
              <div>
                <strong>Status:</strong>{' '}
                <Tag color={llmInfo.is_loaded ? 'green' : 'red'}>
                  {llmInfo.is_loaded ? 'Loaded' : 'Not Loaded'}
                </Tag>
              </div>
            </Col>
            <Col span={8}>
              <div>
                <strong>Model Type:</strong> {llmInfo.model_type}
              </div>
            </Col>
            <Col span={8}>
              <div>
                <strong>Model Path:</strong>{' '}
                <code style={{ fontSize: '12px' }}>
                  {llmInfo.model_path || 'Not specified'}
                </code>
              </div>
            </Col>
          </Row>

          {llmInfo.load_error && (
            <Alert
              message="Model Load Error"
              description={llmInfo.load_error}
              type="error"
              showIcon
              style={{ marginTop: 16 }}
            />
          )}

          {llmInfo.is_loaded && llmInfo.config && (
            <Card title="Model Configuration" size="small" style={{ marginTop: 16 }}>
              <Row gutter={[16, 8]}>
                {Object.entries(llmInfo.config).map(([key, value]) => (
                  <Col span={8} key={key}>
                    <div>
                      <strong>{key}:</strong> {String(value)}
                    </div>
                  </Col>
                ))}
              </Row>
            </Card>
          )}
        </Card>
      </div>
    );
  };

  const renderConversationTab = () => {
    if (!conversationContext) {
      return (
        <Alert
          message="Conversation Context Unavailable"
          description="No conversation context data available."
          type="info"
          showIcon
        />
      );
    }

    return (
      <div>
        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Card title="Conversation Summary">
              <Statistic
                title="Conversation Length"
                value={conversationContext.conversation_length}
                suffix="turns"
              />
              <div style={{ marginTop: 16 }}>
                <strong>Summary:</strong>
                <p style={{ marginTop: 8, padding: 12, background: '#f5f5f5', borderRadius: 4 }}>
                  {conversationContext.context_summary}
                </p>
              </div>
            </Card>
          </Col>
          <Col span={12}>
            <Card title="Topic Tracking">
              {Object.entries(conversationContext.topic_tracking || {}).length > 0 ? (
                <div>
                  {Object.entries(conversationContext.topic_tracking).map(([topic, mentions]) => (
                    <div key={topic} style={{ marginBottom: 8 }}>
                      <Tag color="blue">{topic}</Tag>
                      <span>mentioned in {Array.isArray(mentions) ? mentions.length : mentions} turns</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ color: '#8c8c8c' }}>No topics tracked yet</div>
              )}
            </Card>
          </Col>
          <Col span={24}>
            <Card title="Entity Timeline">
              {Object.entries(conversationContext.entity_timeline || {}).length > 0 ? (
                <Table
                  dataSource={Object.entries(conversationContext.entity_timeline).map(([entity, data], index) => ({
                    key: index,
                    entity,
                    mentions: Array.isArray(data) ? data.length : 1,
                    lastMention: Array.isArray(data) ? Math.max(...data.map((d: any) => d.turn_id || 0)) : 0
                  }))}
                  columns={[
                    { title: 'Entity', dataIndex: 'entity', key: 'entity' },
                    { title: 'Mentions', dataIndex: 'mentions', key: 'mentions' },
                    { title: 'Last Mention (Turn)', dataIndex: 'lastMention', key: 'lastMention' }
                  ]}
                  size="small"
                  pagination={{ pageSize: 10 }}
                />
              ) : (
                <div style={{ color: '#8c8c8c' }}>No entities tracked yet</div>
              )}
            </Card>
          </Col>
        </Row>
      </div>
    );
  };

  const renderCacheModal = () => (
    <Modal
      title="Cache Management"
      visible={cacheModalVisible}
      onCancel={() => setCacheModalVisible(false)}
      footer={null}
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        <Alert
          message="Cache Invalidation"
          description="Clear cached data to force fresh processing. This may temporarily slow down responses."
          type="info"
          showIcon
        />
        
        <Button 
          danger 
          block 
          onClick={() => handleCacheInvalidation()}
          icon={<DeleteOutlined />}
        >
          Clear All Cache
        </Button>
        
        <Button 
          block 
          onClick={() => handleCacheInvalidation('ned')}
        >
          Clear NED Cache
        </Button>
        
        <Button 
          block 
          onClick={() => handleCacheInvalidation('rag')}
        >
          Clear RAG Cache
        </Button>
        
        <Button 
          block 
          onClick={() => handleCacheInvalidation('neuro_symbolic')}
        >
          Clear Neuro-Symbolic Cache
        </Button>
      </Space>
    </Modal>
  );

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2>
          <DashboardOutlined /> Medical AI Analytics Dashboard
        </h2>
        <Space>
          <Select
            value={metricsHours}
            onChange={setMetricsHours}
            style={{ width: 120 }}
          >
            <Option value={1}>Last 1 hour</Option>
            <Option value={6}>Last 6 hours</Option>
            <Option value={24}>Last 24 hours</Option>
          </Select>
          <Button 
            icon={<SettingOutlined />} 
            onClick={() => setCacheModalVisible(true)}
          >
            Cache Management
          </Button>
          <Button 
            type="primary" 
            icon={<ReloadOutlined />} 
            onClick={loadAllData}
            loading={loading}
          >
            Refresh
          </Button>
        </Space>
      </div>

      <Tabs defaultActiveKey="overview">
        <TabPane tab={<span><BarChartOutlined />Overview</span>} key="overview">
          {renderOverviewTab()}
        </TabPane>
        <TabPane tab={<span><InfoCircleOutlined />LLM Status</span>} key="llm">
          {renderLLMTab()}
        </TabPane>
        <TabPane tab="Conversation Context" key="conversation">
          {renderConversationTab()}
        </TabPane>
      </Tabs>

      {renderCacheModal()}
    </div>
  );
};

export default AnalyticsDashboard;