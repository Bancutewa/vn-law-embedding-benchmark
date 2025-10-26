import React from 'react';
import { Card, Calendar, List, Avatar, Space, Tag, Row, Col } from 'antd';
import type { Dayjs } from 'dayjs';

const AppointmentCalendar: React.FC = () => {
  const mockAppointments = [
    {
      date: '2024-01-15',
      time: '14:00',
      customer: 'Nguyễn Văn A',
      property: 'Căn hộ Vinhomes Central Park',
      status: 'confirmed',
    },
    {
      date: '2024-01-16',
      time: '10:30',
      customer: 'Trần Thị B',
      property: 'Nhà phố Thảo Điền',
      status: 'pending',
    },
  ];

  const dateCellRender = (value: Dayjs) => {
    const dateStr = value.format('YYYY-MM-DD');
    const dayAppointments = mockAppointments.filter(app => app.date === dateStr);

    return (
      <div>
        {dayAppointments.map((app, index) => (
          <div key={index} style={{ fontSize: '12px', marginBottom: '2px' }}>
            {app.time} - {app.customer}
          </div>
        ))}
      </div>
    );
  };

  return (
    <Row gutter={[16, 16]}>
      <Col xs={24} lg={16}>
        <Card title="Lịch hẹn">
          <Calendar dateCellRender={dateCellRender} />
        </Card>
      </Col>
      <Col xs={24} lg={8}>
        <Card title="Lịch hẹn hôm nay">
          <List
            dataSource={mockAppointments}
            renderItem={item => (
              <List.Item>
                <List.Item.Meta
                  avatar={<Avatar>{item.customer.charAt(0)}</Avatar>}
                  title={`${item.time} - ${item.customer}`}
                  description={
                    <Space direction="vertical" size="small">
                      <div>{item.property}</div>
                      <Tag color={item.status === 'confirmed' ? 'green' : 'orange'}>
                        {item.status === 'confirmed' ? 'Đã xác nhận' : 'Chờ xác nhận'}
                      </Tag>
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        </Card>
      </Col>
    </Row>
  );
};

export default AppointmentCalendar;
