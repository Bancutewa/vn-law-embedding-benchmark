import React, { useState } from 'react';
import { Table, Button, Space, Tag, Avatar, Input, Select, Card, Statistic } from 'antd';
import { PlusOutlined, SearchOutlined, FilterOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/lib/table';

const { Option } = Select;

interface User {
  id: string;
  name: string;
  email: string;
  role: 'admin' | 'agency' | 'user';
  status: 'active' | 'inactive' | 'suspended';
  lastLogin: string;
  joinDate: string;
}

const mockUsers: User[] = [
  {
    id: '1',
    name: 'Nguyễn Văn A',
    email: 'nguyenvana@example.com',
    role: 'admin',
    status: 'active',
    lastLogin: '2024-01-15 10:30',
    joinDate: '2023-06-01',
  },
  {
    id: '2',
    name: 'Trần Thị B',
    email: 'tranthib@example.com',
    role: 'agency',
    status: 'active',
    lastLogin: '2024-01-14 15:45',
    joinDate: '2023-08-15',
  },
  {
    id: '3',
    name: 'Lê Văn C',
    email: 'levanc@example.com',
    role: 'user',
    status: 'inactive',
    lastLogin: '2024-01-10 09:20',
    joinDate: '2023-12-01',
  },
];

const UserManagement: React.FC = () => {
  console.log('UserManagement component rendered');
  const [users] = useState<User[]>(mockUsers);
  const [searchText, setSearchText] = useState('');
  const [roleFilter, setRoleFilter] = useState<string>('all');

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'admin': return 'red';
      case 'agency': return 'blue';
      case 'user': return 'green';
      default: return 'default';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'inactive': return 'default';
      case 'suspended': return 'error';
      default: return 'default';
    }
  };

  const getRoleText = (role: string) => {
    switch (role) {
      case 'admin': return 'Quản trị';
      case 'agency': return 'Agency';
      case 'user': return 'Người dùng';
      default: return role;
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'active': return 'Hoạt động';
      case 'inactive': return 'Không hoạt động';
      case 'suspended': return 'Đã khóa';
      default: return status;
    }
  };

  const columns: ColumnsType<User> = [
    {
      title: 'Người dùng',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: User) => (
        <Space>
          <Avatar>{name.charAt(0)}</Avatar>
          <div>
            <div style={{ fontWeight: 'bold' }}>{name}</div>
            <div style={{ color: '#666', fontSize: '12px' }}>{record.email}</div>
          </div>
        </Space>
      ),
    },
    {
      title: 'Vai trò',
      dataIndex: 'role',
      key: 'role',
      render: (role: string) => (
        <Tag color={getRoleColor(role)}>{getRoleText(role)}</Tag>
      ),
    },
    {
      title: 'Trạng thái',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>{getStatusText(status)}</Tag>
      ),
    },
    {
      title: 'Đăng nhập cuối',
      dataIndex: 'lastLogin',
      key: 'lastLogin',
    },
    {
      title: 'Ngày tham gia',
      dataIndex: 'joinDate',
      key: 'joinDate',
    },
    {
      title: 'Thao tác',
      key: 'action',
      render: (_: any, record: User) => (
        <Space size="small">
          <Button type="link" size="small">Chỉnh sửa</Button>
          <Button type="link" size="small" danger>
            {record.status === 'suspended' ? 'Mở khóa' : 'Khóa'}
          </Button>
        </Space>
      ),
    },
  ];

  const filteredUsers = users.filter(user => {
    const matchesSearch = user.name.toLowerCase().includes(searchText.toLowerCase()) ||
                         user.email.toLowerCase().includes(searchText.toLowerCase());
    const matchesRole = roleFilter === 'all' || user.role === roleFilter;
    return matchesSearch && matchesRole;
  });

  const stats = {
    total: users.length,
    active: users.filter(u => u.status === 'active').length,
    agencies: users.filter(u => u.role === 'agency').length,
    admins: users.filter(u => u.role === 'admin').length,
  };

  const simpleColumns = [
    {
      title: 'Tên',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Email',
      dataIndex: 'email',
      key: 'email',
    },
    {
      title: 'Vai trò',
      dataIndex: 'role',
      key: 'role',
    },
    {
      title: 'Trạng thái',
      dataIndex: 'status',
      key: 'status',
    },
  ];

  return (
    <div>
      <h2>Quản lý người dùng</h2>
      <p>Tổng số người dùng: {users.length}</p>
      <Table
        columns={simpleColumns}
        dataSource={users}
        rowKey="id"
        pagination={false}
      />
    </div>
  );
};

export default UserManagement;
