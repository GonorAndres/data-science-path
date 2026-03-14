'use client'

import { useMemo } from 'react'
import { DataTable } from '../ui/DataTable'
import { LoadingSpinner } from '../ui/LoadingSpinner'

interface Coefficient {
  feature: string
  coefficient: number
  relativity: number
  p_value: number
  significant: boolean
}

interface CoefficientTableProps {
  data: {
    frequency: Coefficient[]
    severity: Coefficient[]
  }
  activeType: 'frequency' | 'severity'
  isLoading: boolean
}

const columns = [
  { key: 'feature', label: 'Variable' },
  {
    key: 'coefficient',
    label: 'Coeficiente',
    format: (v: number | string) => Number(v).toFixed(4),
  },
  {
    key: 'relativity',
    label: 'Relatividad',
    format: (v: number | string) => Number(v).toFixed(3),
  },
  {
    key: 'p_value',
    label: 'Valor p',
    format: (v: number | string) => Number(v) < 0.001 ? '< 0.001' : Number(v).toFixed(4),
  },
  {
    key: 'significant',
    label: 'Significativo',
    format: (v: number | string) => (String(v) === 'true' || v === 1 ? 'Si' : 'No'),
  },
]

export function CoefficientTable({ data, activeType, isLoading }: CoefficientTableProps) {
  const rows = useMemo(() => {
    if (!data) return []
    // Support both {frequency: [...]} and {frequency_model: {coefficients: [...]}}
    const d = data as any
    let source: any[]
    if (activeType === 'frequency') {
      source = d.frequency ?? d.frequency_model?.coefficients ?? []
    } else {
      source = d.severity ?? d.severity_model?.coefficients ?? []
    }
    return (source ?? []).map((row: any) => ({
      ...row,
      relativity: row.relativity ?? row.relativity_per_unit ?? Math.exp(row.coefficient ?? 0),
      significant: row.significant ?? (row.p_value != null && row.p_value < 0.05),
      _rowClassName: (row.significant ?? (row.p_value < 0.05)) ? 'bg-[#2D6B45]/5' : '',
      _coeffColor: row.coefficient < 0 ? 'text-red-600' : 'text-[#2D6B45]',
    }))
  }, [data, activeType])

  if (isLoading) return <LoadingSpinner />

  return (
    <div className="w-full overflow-x-auto">
      <table className="w-full text-sm font-sans">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                className="text-left text-xs uppercase tracking-widest text-muted border-b border-border dark:border-[#2a2a2a] py-2 px-3"
              >
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              className={`border-b border-border/50 dark:border-[#2a2a2a]/50 transition-colors ${
                row.significant ? 'bg-[#2D6B45]/5' : ''
              }`}
            >
              <td className="py-2 px-3 text-ink dark:text-[#F0EFEB]">{row.feature}</td>
              <td className={`py-2 px-3 ${row.coefficient < 0 ? 'text-red-600' : 'text-[#2D6B45]'}`}>
                {columns[1].format!(row.coefficient)}
              </td>
              <td className="py-2 px-3 text-ink dark:text-[#F0EFEB]">
                {columns[2].format!(row.relativity)}
              </td>
              <td className="py-2 px-3 text-ink dark:text-[#F0EFEB]">
                {columns[3].format!(row.p_value)}
              </td>
              <td className="py-2 px-3 text-ink dark:text-[#F0EFEB]">
                {columns[4].format!(row.significant as unknown as number | string)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
