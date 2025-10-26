import React from 'react';
import { Card, Row, Col, Statistic } from 'antd';
import { DollarOutlined, HomeOutlined, UserOutlined, CalendarOutlined } from '@ant-design/icons';

const AgencyAnalytics: React.FC = () => {
  const stats = [
    {
      title: 'Tổng bất động sản',
      value: 45,
      icon: <HomeOutlined />,
    },
    {
      title: 'Khách hàng tiềm năng',
      value: 123,
      icon: <UserOutlined />,
    },
    {
      title: 'Lịch hẹn tháng này',
      value: 28,
      icon: <CalendarOutlined />,
    },
    {
      title: 'Doanh thu tháng này',
      value: '2.5 tỷ',
      icon: <DollarOutlined />,
    },
  ];

  return (
    <div>
      <Row gutter={[16, 16]}>
        {stats.map((stat, index) => (
          <Col xs={24} sm={12} md={6} key={index}>
            <Card>
              <Statistic
                title={stat.title}
                value={stat.value}
                prefix={<span style={{ color: '#1890ff', fontSize: '24px' }}>{stat.icon}</span>}
              />
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
};

export default AgencyAnalytics;
