import React, { useState } from 'react';
import { Tabs, Typography, Row, Col } from 'antd';
import {
  UserOutlined,
  RobotOutlined,
  DatabaseOutlined,
  SafetyOutlined,
  MessageOutlined,
  AuditOutlined
} from '@ant-design/icons';
import UserManagement from './UserManagement';
import AIMonitoring from './AIMonitoring';
import DataManagement from './DataManagement';
import PermissionsManagement from './PermissionsManagement';
import ActivityLogs from './ActivityLogs';
import FeedbackAnalysis from './FeedbackAnalysis';

const { Title, Paragraph } = Typography;

const AdminDashboard: React.FC = () => {
  console.log('AdminDashboard rendered');
  const [activeTab, setActiveTab] = useState('users');

  const tabItems = [
    {
      key: 'users',
      label: (
        <span>
          <UserOutlined />
          Người dùng
        </span>
      ),
      children: <UserManagement />,
    },
    {
      key: 'ai',
      label: (
        <span>
          <RobotOutlined />
          AI
        </span>
      ),
      children: <AIMonitoring />,
    },
    {
      key: 'data',
      label: (
        <span>
          <DatabaseOutlined />
          Dữ liệu
        </span>
      ),
      children: <DataManagement />,
    },
    {
      key: 'permissions',
      label: (
        <span>
          <SafetyOutlined />
          Phân quyền
        </span>
      ),
      children: <PermissionsManagement />,
    },
    {
      key: 'activity',
      label: (
        <span>
          <AuditOutlined />
          Hoạt động
        </span>
      ),
      children: <ActivityLogs />,
    },
    {
      key: 'feedback',
      label: (
        <span>
          <MessageOutlined />
          Phản hồi
        </span>
      ),
      children: <FeedbackAnalysis />,
    },
  ];

  return (
    <div style={{ padding: '24px', maxWidth: '1400px', margin: '0 auto' }}>
      <div style={{ marginBottom: '32px' }}>
        <Title level={2} style={{ marginBottom: '8px' }}>
          Dashboard Quản trị
        </Title>
        <Paragraph style={{ color: '#666' }}>
          Quản lý hệ thống, người dùng và giám sát AI
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

export default AdminDashboard;
