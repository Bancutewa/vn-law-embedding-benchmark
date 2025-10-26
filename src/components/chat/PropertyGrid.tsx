import React from 'react';
import { Card, Row, Col, Button, Tag, Space } from 'antd';
import { EyeOutlined, CalendarOutlined } from '@ant-design/icons';
import type { Property } from './ChatInterface';

const { Meta } = Card;

interface PropertyGridProps {
  properties: Property[];
  onViewDetails?: (property: Property) => void;
  onBookViewing?: (property: Property) => void;
}

const PropertyGrid: React.FC<PropertyGridProps> = ({
  properties,
  onViewDetails,
  onBookViewing,
}) => {
  const formatPrice = (price: number) => {
    if (price >= 1000000000) {
      return `${(price / 1000000000).toFixed(1)} tỷ`;
    }
    return `${(price / 1000000).toFixed(0)} triệu`;
  };

  return (
    <div style={{ marginTop: '16px' }}>
      <Row gutter={[16, 16]}>
        {properties.map((property) => (
          <Col xs={24} sm={12} key={property.id}>
            <Card
              hoverable
              cover={
                <img
                  alt={property.title}
                  src={property.image}
                  style={{ height: '200px', objectFit: 'cover' }}
                />
              }
              actions={[
                <Button
                  key="view"
                  type="text"
                  icon={<EyeOutlined />}
                  onClick={() => onViewDetails?.(property)}
                >
                  Chi tiết
                </Button>,
                <Button
                  key="book"
                  type="primary"
                  icon={<CalendarOutlined />}
                  onClick={() => onBookViewing?.(property)}
                >
                  Đặt lịch
                </Button>,
              ]}
            >
              <Meta
                title={
                  <div>
                    <div style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '4px' }}>
                      {property.title}
                    </div>
                    <div style={{
                      fontSize: '18px',
                      fontWeight: 'bold',
                      color: '#1677ff',
                      marginBottom: '8px'
                    }}>
                      {formatPrice(property.price)}
                    </div>
                  </div>
                }
                description={
                  <Space direction="vertical" size="small" style={{ width: '100%' }}>
                    <div style={{ color: '#666' }}>
                      📍 {property.location}
                    </div>
                    <Space wrap>
                      <Tag color="blue">{property.type}</Tag>
                      <Tag>{property.area}m²</Tag>
                      <Tag>{property.bedrooms} PN</Tag>
                      <Tag>{property.bathrooms} WC</Tag>
                    </Space>
                    <Tag
                      color={property.status === 'available' ? 'green' : 'orange'}
                      style={{ alignSelf: 'flex-start' }}
                    >
                      {property.status === 'available' ? 'Có sẵn' : 'Đã bán'}
                    </Tag>
                  </Space>
                }
              />
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
};

export default PropertyGrid;
