import React, { useState } from 'react';
import { Modal, Form, Input, DatePicker, TimePicker, Button, message } from 'antd';
import { UserOutlined, PhoneOutlined, MailOutlined } from '@ant-design/icons';
import type { Property } from './ChatInterface';
import dayjs from 'dayjs';

interface BookingModalProps {
  open: boolean;
  onClose: () => void;
  property: Property | null;
}

const BookingModal: React.FC<BookingModalProps> = ({
  open,
  onClose,
  property,
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (values: any) => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));

      console.log('Booking data:', {
        property: property?.title,
        ...values,
        date: values.date.format('YYYY-MM-DD'),
        time: values.time.format('HH:mm'),
      });

      message.success('Đặt lịch xem nhà thành công! Chúng tôi sẽ liên hệ với bạn sớm.');
      form.resetFields();
      onClose();
    } catch (error) {
      message.error('Có lỗi xảy ra. Vui lòng thử lại.');
    } finally {
      setLoading(false);
    }
  };

  const formatPrice = (price: number) => {
    if (price >= 1000000000) {
      return `${(price / 1000000000).toFixed(1)} tỷ`;
    }
    return `${(price / 1000000).toFixed(0)} triệu`;
  };

  return (
    <Modal
      title="Đặt lịch xem nhà"
      open={open}
      onCancel={onClose}
      footer={null}
      width={600}
      centered
    >
      {property && (
        <div style={{ marginBottom: '24px', padding: '16px', background: '#f5f5f5', borderRadius: '8px' }}>
          <h3 style={{ marginBottom: '8px' }}>{property.title}</h3>
          <p style={{ marginBottom: '4px', color: '#1677ff', fontWeight: 'bold' }}>
            {formatPrice(property.price)}
          </p>
          <p style={{ marginBottom: '4px', color: '#666' }}>
            📍 {property.location}
          </p>
          <p style={{ color: '#666' }}>
            {property.area}m² • {property.bedrooms} PN • {property.bathrooms} WC
          </p>
        </div>
      )}

      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={{
          date: dayjs().add(1, 'day'),
          time: dayjs().hour(9).minute(0),
        }}
      >
        <Form.Item
          name="name"
          label="Họ và tên"
          rules={[{ required: true, message: 'Vui lòng nhập họ và tên' }]}
        >
          <Input prefix={<UserOutlined />} placeholder="Nhập họ và tên" />
        </Form.Item>

        <Form.Item
          name="phone"
          label="Số điện thoại"
          rules={[
            { required: true, message: 'Vui lòng nhập số điện thoại' },
            { pattern: /^[0-9]{10,11}$/, message: 'Số điện thoại không hợp lệ' }
          ]}
        >
          <Input prefix={<PhoneOutlined />} placeholder="Nhập số điện thoại" />
        </Form.Item>

        <Form.Item
          name="email"
          label="Email"
          rules={[
            { required: true, message: 'Vui lòng nhập email' },
            { type: 'email', message: 'Email không hợp lệ' }
          ]}
        >
          <Input prefix={<MailOutlined />} placeholder="Nhập email" />
        </Form.Item>

        <Form.Item
          name="date"
          label="Ngày xem nhà"
          rules={[{ required: true, message: 'Vui lòng chọn ngày' }]}
        >
          <DatePicker
            style={{ width: '100%' }}
            disabledDate={(current) => current && current < dayjs().startOf('day')}
            placeholder="Chọn ngày"
          />
        </Form.Item>

        <Form.Item
          name="time"
          label="Thời gian"
          rules={[{ required: true, message: 'Vui lòng chọn thời gian' }]}
        >
          <TimePicker
            style={{ width: '100%' }}
            format="HH:mm"
            minuteStep={30}
            placeholder="Chọn thời gian"
          />
        </Form.Item>

        <Form.Item name="note" label="Ghi chú">
          <Input.TextArea
            placeholder="Nhập ghi chú nếu có"
            rows={3}
          />
        </Form.Item>

        <Form.Item style={{ marginBottom: 0, textAlign: 'right' }}>
          <Button onClick={onClose} style={{ marginRight: '8px' }}>
            Hủy
          </Button>
          <Button type="primary" htmlType="submit" loading={loading}>
            Đặt lịch
          </Button>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default BookingModal;
