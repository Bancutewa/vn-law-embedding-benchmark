import React from 'react';
import { Card, Row, Col, Statistic, Progress, Table, Tag } from 'antd';
import { RobotOutlined, MessageOutlined, AlertOutlined, CheckCircleOutlined } from '@ant-design/icons';

const AIMonitoring: React.FC = () => {
  console.log('AIMonitoring component rendered');
  const stats = [
    {
      title: 'Tổng cuộc trò chuyện',
      value: 1247,
      icon: <MessageOutlined />,
      color: '#1890ff',
    },
    {
      title: 'Trả lời chính xác',
      value: '94.2%',
      icon: <CheckCircleOutlined />,
      color: '#52c41a',
    },
    {
      title: 'Thời gian phản hồi',
      value: '1.2s',
      icon: <RobotOutlined />,
      color: '#722ed1',
    },
    {
      title: 'Lỗi hệ thống',
      value: 3,
      icon: <AlertOutlined />,
      color: '#faad14',
    },
  ];

  const columns = [
    {
      title: 'Thời gian',
      dataIndex: 'time',
      key: 'time',
    },
    {
      title: 'Loại tương tác',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={type === 'question' ? 'blue' : 'green'}>
          {type === 'question' ? 'Câu hỏi' : 'Đặt lịch'}
        </Tag>
      ),
    },
    {
      title: 'Độ chính xác',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy: number) => (
        <Progress percent={accuracy} size="small" status={accuracy > 80 ? 'success' : 'normal'} />
      ),
    },
    {
      title: 'Thời gian xử lý',
      dataIndex: 'processingTime',
      key: 'processingTime',
    },
  ];

  const mockData = [
    { time: '14:30:25', type: 'question', accuracy: 95, processingTime: '0.8s' },
    { time: '14:28:12', type: 'booking', accuracy: 100, processingTime: '1.1s' },
    { time: '14:25:45', type: 'question', accuracy: 88, processingTime: '1.5s' },
    { time: '14:22:33', type: 'question', accuracy: 92, processingTime: '0.9s' },
  ];

  return (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        {stats.map((stat, index) => (
          <Col xs={24} sm={12} md={6} key={index}>
            <Card>
              <Statistic
                title={stat.title}
                value={stat.value}
                prefix={<span style={{ color: stat.color, fontSize: '24px' }}>{stat.icon}</span>}
              />
            </Card>
          </Col>
        ))}
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24} md={12}>
          <Card title="Hiệu suất AI" style={{ height: '100%' }}>
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span>Độ chính xác trả lời</span>
                <span>94.2%</span>
              </div>
              <Progress percent={94.2} status="success" />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span>Tỷ lệ hài lòng</span>
                <span>87.5%</span>
              </div>
              <Progress percent={87.5} strokeColor="#52c41a" />
            </div>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span>Tỷ lệ chuyển đổi</span>
                <span>23.1%</span>
              </div>
              <Progress percent={23.1} strokeColor="#722ed1" />
            </div>
          </Card>
        </Col>
        <Col xs={24} md={12}>
          <Card title="Tương tác gần đây" style={{ height: '100%' }}>
            <Table
              columns={columns}
              dataSource={mockData}
              pagination={false}
              size="small"
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default AIMonitoring;
