import React from 'react';
import { Layout, Menu, Button, Avatar, Dropdown } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  HomeOutlined,
  MessageOutlined,
  AppstoreOutlined,
  SettingOutlined,
  LoginOutlined,
  UserOutlined,
  HomeFilled
} from '@ant-design/icons';
import type { MenuProps } from 'antd';

const { Header } = Layout;

interface LayoutWrapperProps {
  children: React.ReactNode;
}

const LayoutWrapper: React.FC<LayoutWrapperProps> = ({ children }) => {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      key: '/',
      icon: <MessageOutlined />,
      label: 'Tư vấn AI',
    },
    {
      key: '/agency',
      icon: <HomeOutlined />,
      label: 'Quản lý BĐS',
    },
    {
      key: '/admin',
      icon: <AppstoreOutlined />,
      label: 'Quản trị',
    },
  ];

  const userMenuItems: MenuProps['items'] = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: 'Hồ sơ',
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: 'Cài đặt',
    },
    {
      type: 'divider',
    },
    {
      key: 'logout',
      icon: <LoginOutlined />,
      label: 'Đăng xuất',
    },
  ];

  const handleMenuClick = (key: string) => {
    console.log('Navigating to:', key);
    navigate(key);
  };

  const handleUserMenuClick: MenuProps['onClick'] = (e) => {
    console.log('User menu click:', e.key);
  };

  const currentKey = menuItems.find(item => item.key === location.pathname)?.key || '/';

  return (
    <Layout>
      <Header style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: '#fff',
        boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        padding: '0 24px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              cursor: 'pointer',
              marginRight: '32px'
            }}
            onClick={() => navigate('/')}
          >
            <HomeFilled style={{ fontSize: '24px', color: '#1677ff', marginRight: '8px' }} />
            <span style={{ fontSize: '18px', fontWeight: 'bold' }}>RealEstate AI</span>
          </div>

          <Menu
            mode="horizontal"
            selectedKeys={[currentKey]}
            items={menuItems}
            onClick={({ key }) => handleMenuClick(key)}
            style={{ border: 'none', background: 'transparent' }}
          />
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <Button icon={<SettingOutlined />} type="text" />
          <Dropdown menu={{ items: userMenuItems, onClick: handleUserMenuClick }} placement="bottomRight">
            <Avatar style={{ cursor: 'pointer' }} icon={<UserOutlined />} />
          </Dropdown>
          <Button type="primary">Đăng nhập</Button>
        </div>
      </Header>

      {children}
    </Layout>
  );
};

export default LayoutWrapper;
