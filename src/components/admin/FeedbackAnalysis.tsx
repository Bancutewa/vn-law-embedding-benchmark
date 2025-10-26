import React from 'react';
import { Card, Row, Col, Statistic, Progress, Table, Tag, Avatar } from 'antd';
import { MessageOutlined, LikeOutlined, DislikeOutlined, StarOutlined } from '@ant-design/icons';

const FeedbackAnalysis: React.FC = () => {
  const stats = [
    {
      title: 'Tổng phản hồi',
      value: 1234,
      icon: <MessageOutlined />,
      color: '#1890ff',
    },
    {
      title: 'Đánh giá tích cực',
      value: '87.5%',
      icon: <LikeOutlined />,
      color: '#52c41a',
    },
    {
      title: 'Đánh giá tiêu cực',
      value: '12.5%',
      icon: <DislikeOutlined />,
      color: '#ff4d4f',
    },
    {
      title: 'Điểm trung bình',
      value: 4.2,
      icon: <StarOutlined />,
      color: '#faad14',
    },
  ];

  const columns = [
    {
      title: 'Người dùng',
      dataIndex: 'user',
      key: 'user',
      render: (user: { name: string; avatar?: string }) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Avatar size="small">{user.name.charAt(0)}</Avatar>
          <span>{user.name}</span>
        </div>
      ),
    },
    {
      title: 'Đánh giá',
      dataIndex: 'rating',
      key: 'rating',
      render: (rating: number) => (
        <div>
          {[...Array(5)].map((_, i) => (
            <StarOutlined
              key={i}
              style={{
                color: i < rating ? '#faad14' : '#d9d9d9',
                fontSize: '14px'
              }}
            />
          ))}
          <span style={{ marginLeft: '8px' }}>{rating}/5</span>
        </div>
      ),
    },
    {
      title: 'Phản hồi',
      dataIndex: 'feedback',
      key: 'feedback',
      ellipsis: true,
    },
    {
      title: 'Thời gian',
      dataIndex: 'timestamp',
      key: 'timestamp',
    },
    {
      title: 'Trạng thái',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'reviewed' ? 'green' : 'orange'}>
          {status === 'reviewed' ? 'Đã xem' : 'Chưa xem'}
        </Tag>
      ),
    },
  ];

  const mockFeedback = [
    {
      user: { name: 'Nguyễn Văn A' },
      rating: 5,
      feedback: 'Giao diện đẹp, AI trả lời nhanh và chính xác. Rất hài lòng!',
      timestamp: '2024-01-15 14:30',
      status: 'reviewed'
    },
    {
      user: { name: 'Trần Thị B' },
      rating: 4,
      feedback: 'Tìm được căn hộ ưng ý nhờ AI tư vấn. Chỉ cần cải thiện thêm về tốc độ.',
      timestamp: '2024-01-15 13:45',
      status: 'reviewed'
    },
    {
      user: { name: 'Lê Văn C' },
      rating: 2,
      feedback: 'AI không hiểu đúng yêu cầu của tôi. Cần cải thiện khả năng hiểu ngôn ngữ.',
      timestamp: '2024-01-15 12:20',
      status: 'pending'
    },
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
          <Card title="Phân tích đánh giá">
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span>Rất hài lòng (5 sao)</span>
                <span>45.2%</span>
              </div>
              <Progress percent={45.2} strokeColor="#52c41a" />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span>Hài lòng (4 sao)</span>
                <span>32.3%</span>
              </div>
              <Progress percent={32.3} strokeColor="#1890ff" />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span>Bình thường (3 sao)</span>
                <span>10.0%</span>
              </div>
              <Progress percent={10.0} strokeColor="#faad14" />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span>Không hài lòng (1-2 sao)</span>
                <span>12.5%</span>
              </div>
              <Progress percent={12.5} strokeColor="#ff4d4f" />
            </div>
          </Card>
        </Col>
        <Col xs={24} md={12}>
          <Card title="Phản hồi gần đây">
            <Table
              columns={columns}
              dataSource={mockFeedback}
              pagination={false}
              size="small"
              scroll={{ y: 300 }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default FeedbackAnalysis;
