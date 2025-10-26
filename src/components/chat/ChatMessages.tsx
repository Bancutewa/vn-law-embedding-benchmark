import React, { useEffect, useRef } from 'react';
import { Avatar, Spin, Space, Divider } from 'antd';
import { RobotOutlined, UserOutlined } from '@ant-design/icons';
import PropertyGrid from './PropertyGrid';
import PropertyMap from './PropertyMap';
import PriceChart from './PriceChart';
import type { Message, Property } from './ChatInterface';

interface ChatMessagesProps {
  messages: Message[];
  isLoading: boolean;
  onViewDetails?: (property: Property) => void;
  onBookViewing?: (property: Property) => void;
  onCompareProperties?: (properties: Property[]) => void;
}

const ChatMessages: React.FC<ChatMessagesProps> = ({
  messages,
  isLoading,
  onViewDetails,
  onBookViewing,
  onCompareProperties,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('vi-VN', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div
      ref={scrollRef}
      style={{
        flex: 1,
        padding: '16px',
        overflowY: 'auto',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}
    >
      {messages.map((message) => (
        <div key={message.id}>
          <div style={{
            display: 'flex',
            gap: '12px',
            flexDirection: message.role === 'user' ? 'row-reverse' : 'row',
            alignItems: 'flex-start'
          }}>
            <Avatar
              icon={message.role === 'assistant' ? <RobotOutlined /> : <UserOutlined />}
              style={{
                backgroundColor: message.role === 'assistant' ? '#1677ff' : '#f0f0f0',
                color: message.role === 'assistant' ? '#fff' : '#000',
              }}
            />

            <div style={{
              maxWidth: '70%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: message.role === 'user' ? 'flex-end' : 'flex-start'
            }}>
              <div style={{
                background: message.role === 'assistant' ? '#f0f0f0' : '#1677ff',
                color: message.role === 'assistant' ? '#000' : '#fff',
                padding: '8px 12px',
                borderRadius: '8px',
                whiteSpace: 'pre-line',
                fontSize: '14px',
                lineHeight: '1.5'
              }}>
                {message.content}
              </div>
              <div style={{
                fontSize: '12px',
                color: '#999',
                marginTop: '4px'
              }}>
                {formatTime(message.timestamp)}
              </div>
            </div>
          </div>

          {/* Render rich content for assistant messages */}
          {message.role === 'assistant' && message.metadata && (
            <div style={{ marginLeft: '48px', marginTop: '8px' }}>
              {message.metadata.properties && message.metadata.properties.length > 0 && (
                <>
                  <PropertyGrid
                    properties={message.metadata.properties}
                    onViewDetails={onViewDetails}
                    onBookViewing={onBookViewing}
                  />
                  {message.metadata.properties.length > 1 && (
                    <div style={{ marginTop: '8px' }}>
                      <span
                        style={{
                          color: '#1677ff',
                          cursor: 'pointer',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '4px'
                        }}
                        onClick={() => onCompareProperties?.(message.metadata!.properties!)}
                      >
                        So sánh {message.metadata.properties.length} bất động sản
                      </span>
                    </div>
                  )}
                </>
              )}

              {message.metadata.mapData && (
                <PropertyMap
                  properties={message.metadata.mapData.properties}
                  center={message.metadata.mapData.center}
                />
              )}

              {message.metadata.chartData && (
                <PriceChart
                  data={message.metadata.chartData.data}
                  title={message.metadata.chartData.title}
                  description={message.metadata.chartData.description}
                />
              )}
            </div>
          )}
        </div>
      ))}

      {isLoading && (
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <Avatar
            icon={<RobotOutlined />}
            style={{ backgroundColor: '#1677ff', color: '#fff' }}
          />
          <div style={{
            background: '#f0f0f0',
            padding: '8px 12px',
            borderRadius: '8px',
            display: 'flex',
            alignItems: 'center'
          }}>
            <Spin size="small" />
            <span style={{ marginLeft: '8px', fontSize: '14px' }}>Đang suy nghĩ...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatMessages;
