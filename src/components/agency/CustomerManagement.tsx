import React from 'react';
import { Card, Table, Space, Tag, Avatar } from 'antd';

const CustomerManagement: React.FC = () => {
  const columns = [
    {
      title: 'Khách hàng',
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => (
        <Space>
          <Avatar>{name.charAt(0)}</Avatar>
          <span>{name}</span>
        </Space>
      ),
    },
    {
      title: 'Email',
      dataIndex: 'email',
      key: 'email',
    },
    {
      title: 'Số điện thoại',
      dataIndex: 'phone',
      key: 'phone',
    },
    {
      title: 'Trạng thái',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'active' ? 'green' : 'orange'}>
          {status === 'active' ? 'Đang quan tâm' : 'Đã mua'}
        </Tag>
      ),
    },
  ];

  const mockData = [
    { name: 'Nguyễn Văn A', email: 'nguyenvana@example.com', phone: '0123456789', status: 'active' },
    { name: 'Trần Thị B', email: 'tranthib@example.com', phone: '0987654321', status: 'completed' },
  ];

  return (
    <Card title="Quản lý khách hàng">
      <Table columns={columns} dataSource={mockData} rowKey="email" />
    </Card>
  );
};

export default CustomerManagement;
