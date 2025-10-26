import React from 'react';
import { Card, Button, Table, Space, Tag } from 'antd';
import { PlusOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons';

const PropertyManagement: React.FC = () => {
  const columns = [
    {
      title: 'Tên bất động sản',
      dataIndex: 'title',
      key: 'title',
    },
    {
      title: 'Giá',
      dataIndex: 'price',
      key: 'price',
      render: (price: number) => `${(price / 1000000000).toFixed(1)} tỷ`,
    },
    {
      title: 'Vị trí',
      dataIndex: 'location',
      key: 'location',
    },
    {
      title: 'Trạng thái',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'available' ? 'green' : 'orange'}>
          {status === 'available' ? 'Có sẵn' : 'Đã bán'}
        </Tag>
      ),
    },
    {
      title: 'Thao tác',
      key: 'action',
      render: () => (
        <Space size="small">
          <Button type="link" size="small" icon={<EditOutlined />}>Sửa</Button>
          <Button type="link" size="small" danger icon={<DeleteOutlined />}>Xóa</Button>
        </Space>
      ),
    },
  ];

  const mockData = [
    { title: 'Căn hộ Vinhomes Central Park', price: 5500000000, location: 'Bình Thạnh', status: 'available' },
    { title: 'Nhà phố Thảo Điền', price: 12000000000, location: 'Quận 2', status: 'available' },
  ];

  const simpleColumns = [
    {
      title: 'Tên bất động sản',
      dataIndex: 'title',
      key: 'title',
    },
    {
      title: 'Vị trí',
      dataIndex: 'location',
      key: 'location',
    },
    {
      title: 'Giá',
      dataIndex: 'price',
      key: 'price',
    },
    {
      title: 'Trạng thái',
      dataIndex: 'status',
      key: 'status',
    },
  ];

  return (
    <div>
      <h2>Quản lý bất động sản</h2>
      <p>Tổng số bất động sản: {mockData.length}</p>
      <Table
        columns={simpleColumns}
        dataSource={mockData}
        rowKey="title"
        pagination={false}
      />
    </div>
  );
};

export default PropertyManagement;
