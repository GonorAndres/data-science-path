'use client'

import { useMemo } from 'react'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts'
import { LoadingSpinner } from '../ui/LoadingSpinner'

interface NumericData {
  type: 'numeric'
  bins: string[]
  counts: number[]
  mean?: number
  median?: number
}

interface CategoricalData {
  type: 'categorical'
  categories: string[]
  counts: number[]
}

interface FeatureDistributionProps {
  data: NumericData | CategoricalData
  featureName: string
  isLoading: boolean
}

export function FeatureDistribution({ data, featureName, isLoading }: FeatureDistributionProps) {
  const chartData = useMemo(() => {
    if (!data) return []
    // Support both {bins, counts} and {histogram: {bin_edges, counts}} and {categories, counts} / {value_counts: {values, counts}}
    let labels: (string | number)[]
    let counts: number[]
    if (data.type === 'numeric') {
      const hist = (data as any).histogram
      labels = hist?.bin_edges ?? (data as any).bins ?? []
      counts = hist?.counts ?? data.counts ?? []
    } else {
      const vc = (data as any).value_counts
      labels = vc?.values ?? (data as any).categories ?? []
      counts = vc?.counts ?? data.counts ?? []
    }
    return labels.map((label, i) => ({
      label: String(label),
      count: counts[i] ?? 0,
    }))
  }, [data])

  if (isLoading) return <LoadingSpinner />

  const isNumeric = data?.type === 'numeric'
  const stats = isNumeric ? ((data as any).statistics ?? data) : null
  const meanVal = stats?.mean ?? (data as any)?.mean
  const medianVal = stats?.median ?? (data as any)?.median

  return (
    <ResponsiveContainer width="100%" height={350}>
      <BarChart data={chartData} margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
        <CartesianGrid stroke="#E5E4DF" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="label"
          tick={{ fontSize: 11, fill: '#6b6b6b' }}
          label={{ value: featureName, position: 'insideBottom', offset: -4, fontSize: 12 }}
        />
        <YAxis
          tick={{ fontSize: 12, fill: '#6b6b6b' }}
          label={{ value: 'Frecuencia', angle: -90, position: 'insideLeft', fontSize: 12 }}
        />
        <Tooltip
          contentStyle={{ fontSize: 12, borderRadius: 4, border: '1px solid #E5E4DF' }}
          formatter={(value: number) => [value.toLocaleString('es-ES'), 'Conteo']}
          labelFormatter={(label: string) => `${featureName}: ${label}`}
        />
        {isNumeric && meanVal != null && (
          <ReferenceLine
            x={String(meanVal)}
            stroke="#1E3A5F"
            strokeDasharray="6 3"
            label={{ value: `Media: ${Number(meanVal).toFixed(1)}`, fontSize: 11, fill: '#1E3A5F' }}
          />
        )}
        {isNumeric && medianVal != null && (
          <ReferenceLine
            x={String(medianVal)}
            stroke="#1A7A6D"
            strokeDasharray="6 3"
            label={{ value: `Mediana: ${Number(medianVal).toFixed(1)}`, fontSize: 11, fill: '#1A7A6D' }}
          />
        )}
        <Bar dataKey="count" fill="#C4841D" radius={[3, 3, 0, 0]} name="Conteo" />
      </BarChart>
    </ResponsiveContainer>
  )
}
