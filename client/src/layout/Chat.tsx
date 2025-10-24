import React, { useState, useEffect, useRef } from "react";
import { Button, Input, message, Spin, Drawer, Badge, Tooltip } from "antd";
import {
  SendOutlined,
  PlusOutlined,
  LogoutOutlined,
  MessageOutlined,
  RobotOutlined,
  UserOutlined,
  BarChartOutlined,
  ExperimentOutlined,
} from "@ant-design/icons";
import { useNavigate } from "react-router-dom";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import api from "../api/axios";
import {
  Message,
  Conversation,
  ChatRequest,
  ChatResponse,
  ConversationsResponse,
  ConversationResponse,
  MedicalEntity,
} from "../types";
import MedicalInsights from "../components/MedicalInsights";
import AnalyticsDashboard from "../components/AnalyticsDashboard";
import { medicalApi } from "../api/medicalApi";

dayjs.extend(relativeTime);

const Chat: React.FC = () => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversation, setCurrentConversation] = useState<string | null>(
    null
  );
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const [conversationsLoading, setConversationsLoading] = useState(true);
  const [lastResponse, setLastResponse] = useState<ChatResponse | null>(null);
  const [insightsDrawerVisible, setInsightsDrawerVisible] = useState(false);
  const [analyticsDrawerVisible, setAnalyticsDrawerVisible] = useState(false);
  const navigate = useNavigate();
  const chatFeedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadConversations();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (chatFeedRef.current) {
      chatFeedRef.current.scrollTop = chatFeedRef.current.scrollHeight;
    }
  }, [messages]);

  const loadConversations = async () => {
    try {
      const response = await api.get<ConversationsResponse>("/conversations");
      setConversations(response.data.conversations);
    } catch (error: any) {
      console.error("Error loading conversations:", error);
      if (error.response?.status === 401) {
        message.error("Session expired. Please login again.");
        localStorage.removeItem("token");
        navigate("/");
      } else {
        message.error("Failed to load conversations");
      }
    } finally {
      setConversationsLoading(false);
    }
  };

  const loadConversation = async (id: string) => {
    try {
      const response = await api.get<ConversationResponse>(
        `/conversation/${id}`
      );
      setMessages(response.data.messages);
      setCurrentConversation(id);
    } catch (error: any) {
      console.error("Error loading conversation:", error);
      if (error.response?.status === 401) {
        message.error("Session expired. Please login again.");
        localStorage.removeItem("token");
        navigate("/");
      } else {
        message.error("Failed to load conversation");
      }
    }
  };

  const sendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      sender: "user",
      text: inputValue,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setLoading(true);

    const chatRequest: ChatRequest = {
      query: inputValue,
      conversation_id: currentConversation || undefined,
    };

    setInputValue("");

    try {
      const response = await api.post<ChatResponse>(
        "/api/v1/chat",
        chatRequest
      );

      const botMessage: Message = {
        sender: "bot",
        text: response.data.response,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, botMessage]);
      setLastResponse(response.data);

      // Update current conversation ID if it's a new conversation
      if (!currentConversation) {
        setCurrentConversation(response.data.conversation_id);
        loadConversations(); // Refresh conversations list
      }
    } catch (error) {
      message.error("Failed to send message");
      // Remove the user message if sending failed
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  const startNewChat = () => {
    setCurrentConversation(null);
    setMessages([]);
    setLastResponse(null);
  };

  const handleEntityClick = async (entity: MedicalEntity) => {
    try {
      // Predict diseases based on the clicked entity if it's a symptom
      if (entity.concept_type === 'SYMPTOM') {
        const predictions = await medicalApi.predictDiseases([entity.concept_name || entity.text]);
        message.info(`Found ${Object.keys(predictions.predictions || {}).length} related conditions`);
      }
      // Add entity to input for further exploration
      setInputValue(`Tell me more about ${entity.display_name || entity.text}`);
    } catch (error) {
      console.error('Failed to process entity click:', error);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    navigate("/");
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return dayjs(timestamp).fromNow();
  };

  return (
    <div className="chat-layout">
      {/* Sidebar */}
      <div className="chat-sidebar">
        <div className="sidebar-header">
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={startNewChat}
            block
          >
            New Chat
          </Button>
        </div>

        <div className="sidebar-content">
          {conversationsLoading ? (
            <div style={{ padding: "2rem", textAlign: "center" }}>
              <Spin />
            </div>
          ) : conversations.length === 0 ? (
            <div
              style={{ padding: "2rem", textAlign: "center", color: "#8c8c8c" }}
            >
              No conversations yet
            </div>
          ) : (
            conversations.map((conv) => (
              <div
                key={conv._id}
                className={`conversation-item ${
                  currentConversation === conv._id ? "active" : ""
                }`}
                onClick={() => loadConversation(conv._id)}
              >
                <div className="conversation-text">
                  <MessageOutlined style={{ marginRight: "0.5rem" }} />
                  {conv.latest}
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="chat-main">
        {/* Header */}
        <div className="chat-header">
          <div className="chat-title">Medical Assistant</div>
          <div style={{ display: 'flex', gap: '8px' }}>
            <Tooltip title="Medical Insights">
              <Badge dot={lastResponse?.metadata?.entities_found ? lastResponse.metadata.entities_found > 0 : false}>
                <Button
                  type="text"
                  icon={<ExperimentOutlined />}
                  onClick={() => setInsightsDrawerVisible(true)}
                >
                  Insights
                </Button>
              </Badge>
            </Tooltip>
            <Tooltip title="Analytics Dashboard">
              <Button
                type="text"
                icon={<BarChartOutlined />}
                onClick={() => setAnalyticsDrawerVisible(true)}
              >
                Analytics
              </Button>
            </Tooltip>
            <Button
              type="text"
              icon={<LogoutOutlined />}
              onClick={handleLogout}
              style={{ color: "#ff4d4f" }}
            >
              Logout
            </Button>
          </div>
        </div>

        {/* Chat Feed */}
        <div className="chat-feed" ref={chatFeedRef}>
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">
                <RobotOutlined />
              </div>
              <h3>Welcome to Medical Assistant</h3>
              <p>Start a conversation by typing your medical question below.</p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender}`}>
                <div className="message-content">
                  {msg.sender === "user" ? (
                    <UserOutlined style={{ marginRight: "0.5rem" }} />
                  ) : (
                    <RobotOutlined style={{ marginRight: "0.5rem" }} />
                  )}
                  {msg.text}
                </div>
                <div className="message-timestamp">
                  {formatTimestamp(msg.timestamp)}
                </div>
              </div>
            ))
          )}

          {loading && (
            <div className="message bot">
              <div className="message-content">
                <div className="loading-message">
                  <Spin size="small" />
                  <span>Thinking...</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="chat-input">
          <div className="input-wrapper">
            <Input.TextArea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a medical question..."
              autoSize={{ minRows: 1, maxRows: 4 }}
              disabled={loading}
            />
            <Button
              type="primary"
              icon={<SendOutlined />}
              onClick={sendMessage}
              loading={loading}
              disabled={!inputValue.trim()}
            >
              Send
            </Button>
          </div>
        </div>
      </div>

      {/* Medical Insights Drawer */}
      <Drawer
        title="Medical Insights"
        placement="right"
        width={600}
        onClose={() => setInsightsDrawerVisible(false)}
        visible={insightsDrawerVisible}
        destroyOnClose
      >
        <MedicalInsights 
          lastResponse={lastResponse || undefined} 
          onEntityClick={handleEntityClick}
        />
      </Drawer>

      {/* Analytics Dashboard Drawer */}
      <Drawer
        title="Analytics Dashboard"
        placement="right"
        width={900}
        onClose={() => setAnalyticsDrawerVisible(false)}
        visible={analyticsDrawerVisible}
        destroyOnClose
      >
        <AnalyticsDashboard />
      </Drawer>
    </div>
  );
};

export default Chat;
