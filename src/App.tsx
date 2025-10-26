import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Layout } from 'antd';
import LayoutWrapper from './components/layout/LayoutWrapper';
import HomePage from './pages/HomePage';
import AdminPage from './pages/AdminPage';
import AgencyPage from './pages/AgencyPage';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';

const { Content } = Layout;

function App() {
  return (
    <LayoutWrapper>
      <Content style={{ minHeight: 'calc(100vh - 64px)' }}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/admin" element={<AdminPage />} />
          <Route path="/agency" element={<AgencyPage />} />
          <Route path="/auth/login" element={<LoginPage />} />
          <Route path="/auth/signup" element={<SignupPage />} />
        </Routes>
      </Content>
    </LayoutWrapper>
  );
}

export default App;
