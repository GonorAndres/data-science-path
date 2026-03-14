'use client'
import { useState, useMemo } from 'react'

interface Column {
  key: string
  label: string
  format?: (value: number | string) => string
}

interface DataTableProps {
  columns: Column[]
  data: Record<string, number | string>[]
  sortable?: boolean
}

export function DataTable({ columns, data, sortable = false }: DataTableProps) {
  const [sortKey, setSortKey] = useState<string | null>(null)
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc')

  const handleSort = (key: string) => {
    if (!sortable) return
    if (sortKey === key) {
      setSortDir((prev) => (prev === 'asc' ? 'desc' : 'asc'))
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
  }

  const sorted = useMemo(() => {
    if (!sortKey) return data
    return [...data].sort((a, b) => {
      const av = a[sortKey]
      const bv = b[sortKey]
      if (typeof av === 'number' && typeof bv === 'number') {
        return sortDir === 'asc' ? av - bv : bv - av
      }
      const as = String(av)
      const bs = String(bv)
      return sortDir === 'asc' ? as.localeCompare(bs) : bs.localeCompare(as)
    })
  }, [data, sortKey, sortDir])

  return (
    <div className="w-full overflow-x-auto">
      <table className="w-full text-sm font-sans">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                className={`text-left text-xs uppercase tracking-widest text-muted border-b border-border dark:border-[#2a2a2a] py-2 px-3 ${
                  sortable ? 'cursor-pointer select-none hover:text-ink dark:hover:text-[#F0EFEB]' : ''
                }`}
              >
                {col.label}
                {sortKey === col.key && (
                  <span className="ml-1">{sortDir === 'asc' ? '\u2191' : '\u2193'}</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => (
            <tr
              key={i}
              className="border-b border-border/50 dark:border-[#2a2a2a]/50 hover:bg-surface/50 dark:hover:bg-[#2a2a2a]/50 transition-colors"
            >
              {columns.map((col) => (
                <td key={col.key} className="py-2 px-3 text-ink dark:text-[#F0EFEB]">
                  {col.format ? col.format(row[col.key]) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
