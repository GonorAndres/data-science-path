'use client'

import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts'
import { LoadingSpinner } from '../ui/LoadingSpinner'

interface ShapPoint {
  x: number
  shap_value: number
}

interface ShapDependenceProps {
  data: {
    feature: string
    points: ShapPoint[]
  }
  isLoading: boolean
}

export function ShapDependence({ data, isLoading }: ShapDependenceProps) {
  if (isLoading) return <LoadingSpinner />

  const points = data?.points ?? []
  const featureName = data?.feature ?? 'Variable'

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
        <CartesianGrid stroke="#E5E4DF" strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="x"
          tick={{ fontSize: 12, fill: '#6b6b6b' }}
          label={{ value: featureName, position: 'insideBottom', offset: -4, fontSize: 12 }}
        />
        <YAxis
          type="number"
          dataKey="shap_value"
          tick={{ fontSize: 12, fill: '#6b6b6b' }}
          label={{ value: 'Valor SHAP', angle: -90, position: 'insideLeft', fontSize: 12 }}
        />
        <Tooltip
          contentStyle={{ fontSize: 12, borderRadius: 4, border: '1px solid #E5E4DF' }}
          formatter={(value: number, name: string) => [
            value.toFixed(4),
            name === 'shap_value' ? 'Valor SHAP' : featureName,
          ]}
          labelFormatter={() => ''}
        />
        <ReferenceLine
          y={0}
          stroke="#C4841D"
          strokeDasharray="6 3"
        />
        <Scatter
          name={featureName}
          data={points}
          fill="#2D6B45"
          fillOpacity={0.5}
          r={3}
        />
      </ScatterChart>
    </ResponsiveContainer>
  )
}
