import React from 'react';
import { Row, Col, Typography } from 'antd';
import ChatInterface from '../components/chat/ChatInterface';

const { Title, Paragraph } = Typography;

const HomePage: React.FC = () => {
  console.log('HomePage rendered');
  return (
    <div style={{ padding: '32px', maxWidth: '1280px', margin: '0 auto' }}>
      <Row justify="center">
        <Col xs={24} lg={20} xl={16}>
          <div style={{ textAlign: 'center', marginBottom: '32px' }}>
            <Title level={1} style={{ marginBottom: '16px' }}>
              Tư vấn bất động sản thông minh
            </Title>
            <Paragraph style={{ fontSize: '18px', color: '#666' }}>
              Hỏi AI về bất động sản, nhận tư vấn chi tiết và đặt lịch xem nhà ngay
            </Paragraph>
          </div>

          <ChatInterface />
        </Col>
      </Row>
    </div>
  );
};

export default HomePage;
