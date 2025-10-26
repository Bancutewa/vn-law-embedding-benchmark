import React, { useState, useEffect, useRef } from 'react';
import { Card, Avatar, Input, Button, Spin, Space } from 'antd';
import {
  RobotOutlined,
  UserOutlined,
  SendOutlined,
  SwapOutlined
} from '@ant-design/icons';
import ChatMessages from './ChatMessages';
import PropertyGrid from './PropertyGrid';
import PropertyMap from './PropertyMap';
import PriceChart from './PriceChart';
import ComparisonModal from './ComparisonModal';
import BookingModal from './BookingModal';

const { TextArea } = Input;

export interface Property {
  id: string;
  title: string;
  price: number;
  location: string;
  area: number;
  bedrooms: number;
  bathrooms: number;
  type: string;
  image: string;
  status: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  metadata?: {
    properties?: Property[];
    mapData?: {
      properties: Array<{
        id: string;
        title: string;
        location: string;
        lat: number;
        lng: number;
      }>;
      center?: { lat: number; lng: number };
    };
    chartData?: {
      data: Array<{ month: string; price: number }>;
      title: string;
      description?: string;
    };
  };
}

// Mock data
const mockProperties: Property[] = [
  {
    id: '1',
    title: 'Căn hộ cao cấp Vinhomes Central Park',
    price: 5500000000,
    location: 'Bình Thạnh, TP. Hồ Chí Minh',
    area: 85,
    bedrooms: 2,
    bathrooms: 2,
    type: 'Căn hộ',
    image: '/modern-apartment-building.png',
    status: 'available',
  },
  {
    id: '2',
    title: 'Nhà phố hiện đại Thảo Điền',
    price: 12000000000,
    location: 'Quận 2, TP. Hồ Chí Minh',
    area: 120,
    bedrooms: 4,
    bathrooms: 3,
    type: 'Nhà phố',
    image: '/modern-townhouse.png',
    status: 'available',
  },
];

const mockChartData = [
  { month: 'T1', price: 4.5 },
  { month: 'T2', price: 4.7 },
  { month: 'T3', price: 4.8 },
  { month: 'T4', price: 5.0 },
  { month: 'T5', price: 5.2 },
  { month: 'T6', price: 5.5 },
];

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: 'Xin chào! Tôi là trợ lý AI của RealEstate. Tôi có thể giúp bạn tìm kiếm bất động sản, so sánh giá cả, và đặt lịch xem nhà. Bạn đang tìm loại bất động sản nào?',
      timestamp: new Date(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [input, setInput] = useState('');
  const [comparisonProperties, setComparisonProperties] = useState<Property[]>([]);
  const [showComparison, setShowComparison] = useState(false);
  const [bookingProperty, setBookingProperty] = useState<Property | null>(null);
  const [showBooking, setShowBooking] = useState(false);

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Simulate AI response
    setTimeout(() => {
      const lowerContent = input.toLowerCase();
      let aiMessage: Message;

      if (lowerContent.includes('tìm') || lowerContent.includes('căn hộ') || lowerContent.includes('nhà')) {
        aiMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: 'Tôi đã tìm thấy 2 bất động sản phù hợp với yêu cầu của bạn. Dưới đây là danh sách:',
          timestamp: new Date(),
          metadata: {
            properties: mockProperties,
            mapData: {
              properties: mockProperties.map(p => ({
                id: p.id,
                title: p.title,
                location: p.location,
                lat: 10.8231 + Math.random() * 0.1,
                lng: 106.6297 + Math.random() * 0.1,
              })),
            },
          },
        };
      } else if (lowerContent.includes('giá') || lowerContent.includes('biểu đồ')) {
        aiMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: 'Đây là biểu đồ giá trung bình của căn hộ tại khu vực Bình Thạnh trong 6 tháng qua:',
          timestamp: new Date(),
          metadata: {
            chartData: {
              data: mockChartData,
              title: 'Giá trung bình căn hộ - Bình Thạnh',
              description: 'Đơn vị: tỷ VND',
            },
          },
        };
      } else {
        aiMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: 'Tôi có thể giúp bạn:\n• Tìm kiếm bất động sản theo yêu cầu\n• Xem biểu đồ giá thị trường\n• So sánh các bất động sản\n• Đặt lịch xem nhà\n\nBạn muốn làm gì?',
          timestamp: new Date(),
        };
      }

      setMessages(prev => [...prev, aiMessage]);
      setIsLoading(false);
    }, 1500);
  };

  const handleViewDetails = (property: Property) => {
    console.log('View details:', property);
  };

  const handleBookViewing = (property: Property) => {
    setBookingProperty(property);
    setShowBooking(true);
  };

  const handleCompareProperties = (properties: Property[]) => {
    setComparisonProperties(properties);
    setShowComparison(true);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <>
      <Card
        style={{
          height: 'calc(100vh - 200px)',
          display: 'flex',
          flexDirection: 'column'
        }}
        bodyStyle={{ padding: 0, flex: 1, display: 'flex', flexDirection: 'column' }}
      >
        <ChatMessages
          messages={messages}
          isLoading={isLoading}
          onViewDetails={handleViewDetails}
          onBookViewing={handleBookViewing}
          onCompareProperties={handleCompareProperties}
        />

        <div style={{
          borderTop: '1px solid #f0f0f0',
          padding: '16px',
          display: 'flex',
          gap: '8px'
        }}>
          <TextArea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Nhập câu hỏi của bạn... (Enter để gửi, Shift+Enter để xuống dòng)"
            autoSize={{ minRows: 1, maxRows: 4 }}
            style={{ flex: 1 }}
            disabled={isLoading}
          />
          <Button
            type="primary"
            icon={<SendOutlined />}
            onClick={handleSendMessage}
            disabled={!input.trim() || isLoading}
            style={{ height: 'auto' }}
          >
            Gửi
          </Button>
        </div>
      </Card>

      <ComparisonModal
        open={showComparison}
        onClose={() => setShowComparison(false)}
        properties={comparisonProperties}
      />

      <BookingModal
        open={showBooking}
        onClose={() => setShowBooking(false)}
        property={bookingProperty}
      />
    </>
  );
};

export default ChatInterface;
