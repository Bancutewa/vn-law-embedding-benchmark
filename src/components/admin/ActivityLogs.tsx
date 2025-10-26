import React, { useState } from 'react';
import { Card, Table, Select, DatePicker, Space, Tag, Avatar } from 'antd';
import { FilterOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/lib/table';
import dayjs from 'dayjs';

const { RangePicker } = DatePicker;
const { Option } = Select;

interface ActivityLog {
  id: string;
  user: {
    name: string;
    avatar?: string;
  };
  action: string;
  resource: string;
  timestamp: string;
  ip: string;
  status: 'success' | 'error' | 'warning';
}

const mockLogs: ActivityLog[] = [
  {
    id: '1',
    user: { name: 'Nguyễn Văn A', avatar: '' },
    action: 'Đăng nhập',
    resource: 'Hệ thống',
    timestamp: '2024-01-15 14:30:25',
    ip: '192.168.1.100',
    status: 'success',
  },
  {
    id: '2',
    user: { name: 'Trần Thị B', avatar: '' },
    action: 'Thêm bất động sản',
    resource: 'Properties',
    timestamp: '2024-01-15 14:25:12',
    ip: '192.168.1.101',
    status: 'success',
  },
  {
    id: '3',
    user: { name: 'Lê Văn C', avatar: '' },
    action: 'Đặt lịch xem nhà',
    resource: 'Bookings',
    timestamp: '2024-01-15 14:20:45',
    ip: '192.168.1.102',
    status: 'success',
  },
  {
    id: '4',
    user: { name: 'Admin', avatar: '' },
    action: 'Xóa người dùng',
    resource: 'Users',
    timestamp: '2024-01-15 14:15:33',
    ip: '192.168.1.1',
    status: 'warning',
  },
];

const ActivityLogs: React.FC = () => {
  const [logs] = useState<ActivityLog[]>(mockLogs);
  const [actionFilter, setActionFilter] = useState<string>('all');

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'success';
      case 'error': return 'error';
      case 'warning': return 'warning';
      default: return 'default';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'success': return 'Thành công';
      case 'error': return 'Lỗi';
      case 'warning': return 'Cảnh báo';
      default: return status;
    }
  };

  const columns: ColumnsType<ActivityLog> = [
    {
      title: 'Người dùng',
      dataIndex: 'user',
      key: 'user',
      render: (user: { name: string; avatar?: string }) => (
        <Space>
          <Avatar size="small">{user.name.charAt(0)}</Avatar>
          <span>{user.name}</span>
        </Space>
      ),
    },
    {
      title: 'Hành động',
      dataIndex: 'action',
      key: 'action',
    },
    {
      title: 'Tài nguyên',
      dataIndex: 'resource',
      key: 'resource',
    },
    {
      title: 'Thời gian',
      dataIndex: 'timestamp',
      key: 'timestamp',
    },
    {
      title: 'IP',
      dataIndex: 'ip',
      key: 'ip',
    },
    {
      title: 'Trạng thái',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>{getStatusText(status)}</Tag>
      ),
    },
  ];

  const filteredLogs = logs.filter(log => {
    return actionFilter === 'all' || log.action.toLowerCase().includes(actionFilter.toLowerCase());
  });

  return (
    <div>
      <Card title="Nhật ký hoạt động">
        <Space style={{ marginBottom: '16px', display: 'flex', flexWrap: 'wrap' }}>
          <Select
            value={actionFilter}
            onChange={setActionFilter}
            style={{ width: 150 }}
            suffixIcon={<FilterOutlined />}
          >
            <Option value="all">Tất cả</Option>
            <Option value="đăng nhập">Đăng nhập</Option>
            <Option value="thêm">Thêm</Option>
            <Option value="sửa">Sửa</Option>
            <Option value="xóa">Xóa</Option>
          </Select>
          <RangePicker
            defaultValue={[dayjs().subtract(7, 'day'), dayjs()]}
            format="YYYY-MM-DD"
          />
        </Space>

        <Table
          columns={columns}
          dataSource={filteredLogs}
          rowKey="id"
          pagination={{
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `${range[0]}-${range[1]} của ${total} hoạt động`,
          }}
        />
      </Card>
    </div>
  );
};

export default ActivityLogs;
