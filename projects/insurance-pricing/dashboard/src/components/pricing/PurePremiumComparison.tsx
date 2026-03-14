'use client'

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts'
import { LoadingSpinner } from '../ui/LoadingSpinner'

interface SegmentRow {
  segment: string
  actual_pp: number
  glm_pp: number
  xgb_pp: number
}

interface PurePremiumComparisonProps {
  data: {
    segments: SegmentRow[]
  }
  isLoading: boolean
}

const formatCurrency = (value: number) =>
  `EUR ${value.toLocaleString('es-ES', { minimumFractionDigits: 0, maximumFractionDigits: 0 })}`

export function PurePremiumComparison({ data, isLoading }: PurePremiumComparisonProps) {
  if (isLoading) return <LoadingSpinner />

  // Support both {segments: [...]} and {by_area: [...]}
  const d = data as any
  const rawSegments = d?.segments ?? d?.by_area ?? []
  const segments = rawSegments.map((s: any) => ({
    segment: s.segment ?? s.label ?? `Area ${s.area ?? '?'}`,
    actual_pp: s.actual_pp ?? s.observed_pure_premium ?? 0,
    glm_pp: s.glm_pp ?? s.glm_predicted ?? 0,
    gbm_pp: s.xgb_pp ?? s.lightgbm_predicted ?? s.xgboost_predicted ?? 0,
  }))

  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={segments} margin={{ top: 8, right: 24, bottom: 8, left: 16 }}>
        <CartesianGrid stroke="#E5E4DF" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="segment"
          tick={{ fontSize: 12, fill: '#6b6b6b' }}
          label={{ value: 'Segmento', position: 'insideBottom', offset: -4, fontSize: 12 }}
        />
        <YAxis
          tick={{ fontSize: 12, fill: '#6b6b6b' }}
          tickFormatter={(v: number) => `EUR ${v}`}
          label={{ value: 'Prima pura', angle: -90, position: 'insideLeft', offset: 4, fontSize: 12 }}
        />
        <Tooltip
          contentStyle={{ fontSize: 12, borderRadius: 4, border: '1px solid #E5E4DF' }}
          formatter={(value: number, name: string) => [formatCurrency(value), name]}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="actual_pp" name="Observado" fill="#C4841D" radius={[3, 3, 0, 0]} />
        <Bar dataKey="glm_pp" name="Poisson GLM" fill="#1E3A5F" radius={[3, 3, 0, 0]} />
        <Bar dataKey="gbm_pp" name="GBM" fill="#2D6B45" radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
