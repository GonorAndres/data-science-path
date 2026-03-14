'use client'

import { useMemo } from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts'
import { LoadingSpinner } from '../ui/LoadingSpinner'

const MODEL_COLORS: Record<string, string> = {
  'Poisson GLM': '#1E3A5F',
  'XGBoost': '#2D6B45',
  'LightGBM': '#1A7A6D',
}

interface BinEntry {
  bin: string
  lift: number
}

interface LiftChartProps {
  data: Record<string, { bins: BinEntry[] }>
  isLoading: boolean
}

export function LiftChart({ data, isLoading }: LiftChartProps) {
  const merged = useMemo(() => {
    if (!data) return []
    // Support both {models: {name: {deciles}}} and {name: {bins}} structures
    const models: any = data.models ?? data
    const modelNames = Object.keys(models)
    if (modelNames.length === 0) return []

    const firstModel = modelNames[0]
    const entries = models[firstModel].deciles ?? models[firstModel].bins ?? []
    return entries.map((entry: any, i: number) => {
      const row: Record<string, string | number> = { bin: entry.decile ?? entry.bin ?? `D${i + 1}` }
      for (const model of modelNames) {
        const mEntries = models[model].deciles ?? models[model].bins ?? []
        row[model] = mEntries[i]?.lift ?? 0
      }
      return row
    })
  }, [data])

  if (isLoading) return <LoadingSpinner />

  const models: any = data?.models ?? data
  const modelNames = models ? Object.keys(models).filter(k => k !== 'description' && k !== 'overall_mean_frequency' && k !== 'n_test') : []

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={merged} margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
        <CartesianGrid stroke="#E5E4DF" strokeDasharray="3 3" />
        <XAxis
          dataKey="bin"
          tick={{ fontSize: 12, fill: '#6b6b6b' }}
          label={{ value: 'Decil', position: 'insideBottom', offset: -4, fontSize: 12 }}
        />
        <YAxis
          tick={{ fontSize: 12, fill: '#6b6b6b' }}
          label={{ value: 'Lift', angle: -90, position: 'insideLeft', fontSize: 12 }}
        />
        <Tooltip
          contentStyle={{ fontSize: 12, borderRadius: 4, border: '1px solid #E5E4DF' }}
          formatter={(value: number) => value.toFixed(3)}
        />
        <Legend
          wrapperStyle={{ fontSize: 12, cursor: 'pointer' }}
        />
        <ReferenceLine y={1} stroke="#C4841D" strokeDasharray="6 3" label={{ value: 'Lift = 1.0', fontSize: 11, fill: '#C4841D' }} />
        {modelNames.map((model) => (
          <Line
            key={model}
            type="monotone"
            dataKey={model}
            stroke={MODEL_COLORS[model] ?? '#999'}
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
