import React from 'react';
import { Modal, Table, Tag, Space } from 'antd';
import type { Property } from './ChatInterface';

interface ComparisonModalProps {
  open: boolean;
  onClose: () => void;
  properties: Property[];
}

const ComparisonModal: React.FC<ComparisonModalProps> = ({
  open,
  onClose,
  properties,
}) => {
  const formatPrice = (price: number) => {
    if (price >= 1000000000) {
      return `${(price / 1000000000).toFixed(1)} tỷ`;
    }
    return `${(price / 1000000).toFixed(0)} triệu`;
  };

  const columns = [
    {
      title: 'Thuộc tính',
      dataIndex: 'attribute',
      key: 'attribute',
      fixed: 'left' as const,
      width: 120,
    },
    ...properties.map((property, index) => ({
      title: property.title.length > 20 ? `${property.title.substring(0, 20)}...` : property.title,
      dataIndex: `property_${index}`,
      key: `property_${index}`,
      width: 150,
      render: (value: any) => {
        if (typeof value === 'number' && value > 1000000) {
          return formatPrice(value);
        }
        return value;
      },
    })),
  ];

  const data = [
    {
      key: 'price',
      attribute: 'Giá',
      ...properties.reduce((acc, property, index) => ({
        ...acc,
        [`property_${index}`]: property.price,
      }), {}),
    },
    {
      key: 'location',
      attribute: 'Vị trí',
      ...properties.reduce((acc, property, index) => ({
        ...acc,
        [`property_${index}`]: property.location,
      }), {}),
    },
    {
      key: 'area',
      attribute: 'Diện tích',
      ...properties.reduce((acc, property, index) => ({
        ...acc,
        [`property_${index}`]: `${property.area}m²`,
      }), {}),
    },
    {
      key: 'bedrooms',
      attribute: 'Phòng ngủ',
      ...properties.reduce((acc, property, index) => ({
        ...acc,
        [`property_${index}`]: property.bedrooms,
      }), {}),
    },
    {
      key: 'bathrooms',
      attribute: 'Phòng tắm',
      ...properties.reduce((acc, property, index) => ({
        ...acc,
        [`property_${index}`]: property.bathrooms,
      }), {}),
    },
    {
      key: 'type',
      attribute: 'Loại',
      ...properties.reduce((acc, property, index) => ({
        ...acc,
        [`property_${index}`]: property.type,
      }), {}),
    },
    {
      key: 'status',
      attribute: 'Trạng thái',
      ...properties.reduce((acc, property, index) => ({
        ...acc,
        [`property_${index}`]: (
          <Tag color={property.status === 'available' ? 'green' : 'orange'}>
            {property.status === 'available' ? 'Có sẵn' : 'Đã bán'}
          </Tag>
        ),
      }), {}),
    },
  ];

  return (
    <Modal
      title="So sánh bất động sản"
      open={open}
      onCancel={onClose}
      width={1200}
      footer={null}
      centered
    >
      <Table
        columns={columns}
        dataSource={data}
        pagination={false}
        scroll={{ x: 'max-content' }}
        size="small"
      />
    </Modal>
  );
};

export default ComparisonModal;
