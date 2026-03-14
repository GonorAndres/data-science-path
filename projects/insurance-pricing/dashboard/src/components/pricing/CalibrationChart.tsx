'use client'

import { useMemo } from 'react'
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
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

interface CalibrationBin {
  avg_predicted: number
  avg_actual: number
}

interface CalibrationChartProps {
  data: Record<string, { bins: CalibrationBin[] }>
  isLoading: boolean
}

export function CalibrationChart({ data, isLoading }: CalibrationChartProps) {
  // Normalize data: support both {models: {name: {bins}}} and flat structure
  const models: any = data?.models ?? data
  const modelNames = useMemo(() => {
    if (!models) return []
    return Object.keys(models).filter(k => typeof models[k] === 'object' && models[k]?.bins)
  }, [models])

  const domain = useMemo(() => {
    if (!models) return [0, 0.1]
    let max = 0
    for (const name of modelNames) {
      for (const bin of (models[name].bins ?? [])) {
        const pred = bin.predicted_mean ?? bin.avg_predicted ?? 0
        const obs = bin.observed_mean ?? bin.avg_actual ?? 0
        max = Math.max(max, pred, obs)
      }
    }
    return [0, Math.ceil(max * 1000) / 1000]
  }, [models, modelNames])

  if (isLoading) return <LoadingSpinner />

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
        <CartesianGrid stroke="#E5E4DF" strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="x"
          domain={domain}
          tick={{ fontSize: 12, fill: '#6b6b6b' }}
          label={{ value: 'Predicho promedio', position: 'insideBottom', offset: -4, fontSize: 12 }}
        />
        <YAxis
          type="number"
          dataKey="y"
          domain={domain}
          tick={{ fontSize: 12, fill: '#6b6b6b' }}
          label={{ value: 'Observado promedio', angle: -90, position: 'insideLeft', fontSize: 12 }}
        />
        <Tooltip
          contentStyle={{ fontSize: 12, borderRadius: 4, border: '1px solid #E5E4DF' }}
          formatter={(value: number) => value.toFixed(4)}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <ReferenceLine
          segment={[
            { x: domain[0], y: domain[0] },
            { x: domain[1], y: domain[1] },
          ]}
          stroke="#C4841D"
          strokeDasharray="6 3"
          label={{ value: 'Calibracion perfecta', fontSize: 11, fill: '#C4841D' }}
        />
        {modelNames.map((model) => (
          <Scatter
            key={model}
            name={model}
            data={(models[model].bins ?? []).map((b: any) => ({
              x: b.predicted_mean ?? b.avg_predicted ?? 0,
              y: b.observed_mean ?? b.avg_actual ?? 0,
            }))}
            fill={MODEL_COLORS[model] ?? '#999'}
          />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  )
}
