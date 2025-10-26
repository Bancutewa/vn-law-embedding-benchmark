import React from 'react';
import { Card, Row, Col, Statistic, Button, Table, Progress, Space } from 'antd';
import { DatabaseOutlined, CloudUploadOutlined, DownloadOutlined, SyncOutlined } from '@ant-design/icons';

const DataManagement: React.FC = () => {
  const stats = [
    {
      title: 'Tổng bất động sản',
      value: 2847,
      icon: <DatabaseOutlined />,
    },
    {
      title: 'Đã đồng bộ',
      value: '96.8%',
      icon: <SyncOutlined />,
    },
    {
      title: 'Dung lượng DB',
      value: '2.4GB',
      icon: <CloudUploadOutlined />,
    },
    {
      title: 'Backup gần nhất',
      value: '2 giờ trước',
      icon: <DownloadOutlined />,
    },
  ];

  const columns = [
    {
      title: 'Tên bảng',
      dataIndex: 'table',
      key: 'table',
    },
    {
      title: 'Số bản ghi',
      dataIndex: 'records',
      key: 'records',
    },
    {
      title: 'Dung lượng',
      dataIndex: 'size',
      key: 'size',
    },
    {
      title: 'Cập nhật cuối',
      dataIndex: 'lastUpdate',
      key: 'lastUpdate',
    },
    {
      title: 'Trạng thái',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Progress
          percent={status === 'completed' ? 100 : 75}
          size="small"
          status={status === 'completed' ? 'success' : 'active'}
        />
      ),
    },
  ];

  const mockData = [
    { table: 'properties', records: '2,847', size: '1.2GB', lastUpdate: '14:30:25', status: 'completed' },
    { table: 'users', records: '1,234', size: '256MB', lastUpdate: '14:25:12', status: 'completed' },
    { table: 'conversations', records: '15,678', size: '890MB', lastUpdate: '14:20:45', status: 'active' },
    { table: 'bookings', records: '456', size: '45MB', lastUpdate: '13:15:30', status: 'completed' },
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
                prefix={<span style={{ color: '#1890ff', fontSize: '24px' }}>{stat.icon}</span>}
              />
            </Card>
          </Col>
        ))}
      </Row>

      <Row gutter={[16, 16]}>
        <Col xs={24}>
          <Card
            title="Quản lý dữ liệu"
            extra={
              <Space>
                <Button icon={<SyncOutlined />}>Đồng bộ</Button>
                <Button icon={<CloudUploadOutlined />}>Import</Button>
                <Button icon={<DownloadOutlined />}>Export</Button>
              </Space>
            }
          >
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

export default DataManagement;
