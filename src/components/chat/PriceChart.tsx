import React from 'react';
import { Card } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface PriceChartProps {
  data: Array<{ month: string; price: number }>;
  title: string;
  description?: string;
}

const PriceChart: React.FC<PriceChartProps> = ({ data, title, description }) => {
  return (
    <Card
      title={title}
      style={{ marginTop: '16px' }}
      bodyStyle={{ padding: '16px' }}
    >
      {description && (
        <p style={{ color: '#666', marginBottom: '16px' }}>{description}</p>
      )}
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="month" />
          <YAxis />
          <Tooltip
            formatter={(value: number) => [`${value} tỷ`, 'Giá']}
            labelFormatter={(label) => `Tháng: ${label}`}
          />
          <Line
            type="monotone"
            dataKey="price"
            stroke="#1677ff"
            strokeWidth={2}
            dot={{ fill: '#1677ff', strokeWidth: 2, r: 4 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );
};

export default PriceChart;
