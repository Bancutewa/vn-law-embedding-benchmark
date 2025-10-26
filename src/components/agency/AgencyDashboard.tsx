import React, { useState } from 'react';
import { Tabs, Typography, Row, Col } from 'antd';
import {
  AppstoreOutlined,
  BarChartOutlined,
  UserOutlined,
  CalendarOutlined
} from '@ant-design/icons';
import PropertyManagement from './PropertyManagement';
import CustomerManagement from './CustomerManagement';
import AgencyAnalytics from './AgencyAnalytics';
import AppointmentCalendar from './AppointmentCalendar';

const { Title, Paragraph } = Typography;

const AgencyDashboard: React.FC = () => {
  console.log('AgencyDashboard rendered');
  const [activeTab, setActiveTab] = useState('properties');

  const tabItems = [
    {
      key: 'properties',
      label: (
        <span>
          <AppstoreOutlined />
          Bất động sản
        </span>
      ),
      children: <PropertyManagement />,
    },
    {
      key: 'customers',
      label: (
        <span>
          <UserOutlined />
          Khách hàng
        </span>
      ),
      children: <CustomerManagement />,
    },
    {
      key: 'analytics',
      label: (
        <span>
          <BarChartOutlined />
          Thống kê
        </span>
      ),
      children: <AgencyAnalytics />,
    },
    {
      key: 'appointments',
      label: (
        <span>
          <CalendarOutlined />
          Lịch hẹn
        </span>
      ),
      children: <AppointmentCalendar />,
    },
  ];

  return (
    <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ marginBottom: '32px' }}>
        <Title level={2} style={{ marginBottom: '8px' }}>
          Dashboard Agency
        </Title>
        <Paragraph style={{ color: '#666' }}>
          Quản lý bất động sản và khách hàng của bạn
        </Paragraph>
      </div>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        type="card"
        size="large"
        items={tabItems}
      />
    </div>
  );
};

export default AgencyDashboard;
