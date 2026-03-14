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
} from 'recharts'
import { LoadingSpinner } from '../ui/LoadingSpinner'

const MODEL_COLORS: Record<string, string> = {
  xgboost: '#2D6B45',
  lightgbm: '#1A7A6D',
}

interface ShapEntry {
  feature: string
  mean_abs_shap: number
  rank: number
}

interface ShapSummaryProps {
  data: Record<string, ShapEntry[]>
  selectedModel: string
  isLoading: boolean
}

export function ShapSummary({ data, selectedModel, isLoading }: ShapSummaryProps) {
  const chartData = useMemo(() => {
    if (!data) return []
    // Support both {xgboost: [...]} and {models: {"XGBoost Poisson": {global_importance: [...]}}}
    const models: any = data.models ?? data
    // Try direct key match, then partial match
    let entries = models[selectedModel]
    if (!entries) {
      const key = Object.keys(models).find(k => k.toLowerCase().includes(selectedModel.toLowerCase()))
      if (key) entries = models[key]
    }
    // Handle nested global_importance
    const items = Array.isArray(entries) ? entries : entries?.global_importance ?? []
    return [...items]
      .sort((a: any, b: any) => (a.rank ?? 0) - (b.rank ?? 0))
      .slice(0, 15)
      .reverse()
  }, [data, selectedModel])

  if (isLoading) return <LoadingSpinner />

  const barColor = MODEL_COLORS[selectedModel] ?? '#2D6B45'

  return (
    <ResponsiveContainer width="100%" height={Math.max(400, chartData.length * 28)}>
      <BarChart data={chartData} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: 120 }}>
        <CartesianGrid stroke="#E5E4DF" strokeDasharray="3 3" horizontal={false} />
        <XAxis
          type="number"
          tick={{ fontSize: 12, fill: '#6b6b6b' }}
          label={{ value: 'Media |SHAP|', position: 'insideBottom', offset: -4, fontSize: 12 }}
        />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fontSize: 11, fill: '#6b6b6b' }}
          width={110}
        />
        <Tooltip
          contentStyle={{ fontSize: 12, borderRadius: 4, border: '1px solid #E5E4DF' }}
          formatter={(value: number) => value.toFixed(4)}
          labelFormatter={(label: string) => `Variable: ${label}`}
        />
        <Bar dataKey="mean_abs_shap" fill={barColor} radius={[0, 3, 3, 0]} name="Media |SHAP|" />
      </BarChart>
    </ResponsiveContainer>
  )
}
