'use client'
import { ChevronDown } from 'lucide-react'

interface FilterBarProps {
  label: string
  options: string[]
  value: string
  onChange: (value: string) => void
  labels?: string[]
}

export function FilterBar({ label, options, value, onChange, labels }: FilterBarProps) {
  return (
    <div className="flex items-center gap-3">
      <label className="font-sans text-xs uppercase tracking-widest text-muted">
        {label}
      </label>
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="select-styled font-sans text-sm border border-border dark:border-[#2a2a2a] rounded-lg bg-surface dark:bg-[#1e1e1e] text-ink dark:text-[#F0EFEB] pl-4 pr-9 py-2.5 shadow-sm outline-none focus:ring-2 focus:ring-glm-blue/30 dark:focus:ring-[#F0EFEB]/20 transition-shadow cursor-pointer"
        >
          {options.map((opt, i) => (
            <option key={opt} value={opt}>
              {labels?.[i] ?? opt}
            </option>
          ))}
        </select>
        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted pointer-events-none" />
      </div>
    </div>
  )
}
