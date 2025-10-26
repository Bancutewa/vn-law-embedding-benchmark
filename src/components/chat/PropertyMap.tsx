import React from 'react';
import { Card, Alert } from 'antd';

interface Property {
  id: string;
  title: string;
  location: string;
  lat: number;
  lng: number;
}

interface PropertyMapProps {
  properties: Property[];
  center?: { lat: number; lng: number };
}

const PropertyMap: React.FC<PropertyMapProps> = ({ properties, center }) => {
  // Mock map component - in real app, integrate with Google Maps or similar
  return (
    <Card
      style={{ marginTop: '16px' }}
      bodyStyle={{
        padding: '16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '300px',
        background: '#f5f5f5'
      }}
    >
      <Alert
        message="Bản đồ vị trí bất động sản"
        description={`Hiển thị ${properties.length} bất động sản trên bản đồ`}
        type="info"
        showIcon
      />
      <div style={{ marginTop: '16px', textAlign: 'center' }}>
        <p style={{ color: '#666', marginBottom: '8px' }}>
          Vị trí trung tâm: {center ? `${center.lat.toFixed(4)}, ${center.lng.toFixed(4)}` : 'Tự động'}
        </p>
        {properties.map((property) => (
          <div key={property.id} style={{ marginBottom: '4px' }}>
            <strong>{property.title}</strong> - {property.location}
          </div>
        ))}
      </div>
    </Card>
  );
};

export default PropertyMap;
