import React, { useState } from 'react';
import { Card, Table, Button, Space, Tag, Switch, Modal, Form, Input, Select, message } from 'antd';
import { PlusOutlined, EditOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/lib/table';

const { Option } = Select;

interface Role {
  id: string;
  name: string;
  description: string;
  permissions: string[];
  userCount: number;
  status: 'active' | 'inactive';
}

const mockRoles: Role[] = [
  {
    id: '1',
    name: 'Super Admin',
    description: 'Quyền truy cập đầy đủ hệ thống',
    permissions: ['all'],
    userCount: 2,
    status: 'active',
  },
  {
    id: '2',
    name: 'Agency Manager',
    description: 'Quản lý bất động sản và khách hàng',
    permissions: ['properties.manage', 'customers.manage', 'analytics.view'],
    userCount: 15,
    status: 'active',
  },
  {
    id: '3',
    name: 'User',
    description: 'Người dùng thông thường',
    permissions: ['chat.use', 'bookings.create'],
    userCount: 1245,
    status: 'active',
  },
];

const PermissionsManagement: React.FC = () => {
  const [roles, setRoles] = useState<Role[]>(mockRoles);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [editingRole, setEditingRole] = useState<Role | null>(null);
  const [form] = Form.useForm();

  const availablePermissions = [
    { value: 'all', label: 'Tất cả quyền' },
    { value: 'properties.manage', label: 'Quản lý bất động sản' },
    { value: 'customers.manage', label: 'Quản lý khách hàng' },
    { value: 'analytics.view', label: 'Xem thống kê' },
    { value: 'users.manage', label: 'Quản lý người dùng' },
    { value: 'chat.use', label: 'Sử dụng chat' },
    { value: 'bookings.create', label: 'Tạo đặt lịch' },
    { value: 'bookings.manage', label: 'Quản lý đặt lịch' },
  ];

  const columns: ColumnsType<Role> = [
    {
      title: 'Vai trò',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: Role) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{name}</div>
          <div style={{ color: '#666', fontSize: '12px' }}>{record.description}</div>
        </div>
      ),
    },
    {
      title: 'Quyền hạn',
      dataIndex: 'permissions',
      key: 'permissions',
      render: (permissions: string[]) => (
        <Space wrap>
          {permissions.includes('all') ? (
            <Tag color="red">Tất cả</Tag>
          ) : (
            permissions.slice(0, 3).map(perm => {
              const permObj = availablePermissions.find(p => p.value === perm);
              return <Tag key={perm} color="blue">{permObj?.label}</Tag>;
            })
          )}
          {permissions.length > 3 && <Tag>+{permissions.length - 3} quyền</Tag>}
        </Space>
      ),
    },
    {
      title: 'Số người dùng',
      dataIndex: 'userCount',
      key: 'userCount',
      align: 'center',
    },
    {
      title: 'Trạng thái',
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record: Role) => (
        <Switch
          checked={status === 'active'}
          onChange={(checked) => {
            const newRoles = roles.map(role =>
              role.id === record.id
                ? { ...role, status: (checked ? 'active' : 'inactive') as Role['status'] }
                : role
            );
            setRoles(newRoles);
          }}
        />
      ),
    },
    {
      title: 'Thao tác',
      key: 'action',
      render: (_: any, record: Role) => (
        <Space size="small">
          <Button
            type="link"
            size="small"
            icon={<EditOutlined />}
            onClick={() => handleEdit(record)}
          >
            Sửa
          </Button>
        </Space>
      ),
    },
  ];

  const handleEdit = (role: Role) => {
    setEditingRole(role);
    form.setFieldsValue({
      name: role.name,
      description: role.description,
      permissions: role.permissions,
    });
    setIsModalVisible(true);
  };

  const handleAdd = () => {
    setEditingRole(null);
    form.resetFields();
    setIsModalVisible(true);
  };

  const handleModalOk = () => {
    form.validateFields().then(values => {
      if (editingRole) {
        // Update existing role
        const newRoles = roles.map(role =>
          role.id === editingRole.id ? { ...role, ...values } : role
        );
        setRoles(newRoles);
        message.success('Cập nhật vai trò thành công');
      } else {
        // Add new role
        const newRole: Role = {
          id: Date.now().toString(),
          ...values,
          userCount: 0,
          status: 'active',
        };
        setRoles([...roles, newRole]);
        message.success('Thêm vai trò mới thành công');
      }
      setIsModalVisible(false);
    });
  };

  return (
    <div>
      <Card
        title="Quản lý phân quyền"
        extra={<Button type="primary" icon={<PlusOutlined />} onClick={handleAdd}>Thêm vai trò</Button>}
      >
        <Table
          columns={columns}
          dataSource={roles}
          rowKey="id"
          pagination={false}
        />
      </Card>

      <Modal
        title={editingRole ? 'Chỉnh sửa vai trò' : 'Thêm vai trò mới'}
        open={isModalVisible}
        onOk={handleModalOk}
        onCancel={() => setIsModalVisible(false)}
        width={600}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="name"
            label="Tên vai trò"
            rules={[{ required: true, message: 'Vui lòng nhập tên vai trò' }]}
          >
            <Input placeholder="Nhập tên vai trò" />
          </Form.Item>

          <Form.Item
            name="description"
            label="Mô tả"
            rules={[{ required: true, message: 'Vui lòng nhập mô tả' }]}
          >
            <Input.TextArea placeholder="Nhập mô tả vai trò" rows={3} />
          </Form.Item>

          <Form.Item
            name="permissions"
            label="Quyền hạn"
            rules={[{ required: true, message: 'Vui lòng chọn quyền hạn' }]}
          >
            <Select
              mode="multiple"
              placeholder="Chọn quyền hạn"
              style={{ width: '100%' }}
            >
              {availablePermissions.map(perm => (
                <Option key={perm.value} value={perm.value}>
                  {perm.label}
                </Option>
              ))}
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default PermissionsManagement;
