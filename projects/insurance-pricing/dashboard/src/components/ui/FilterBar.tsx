'use client'

interface FilterBarProps {
  label: string
  options: string[]
  value: string
  onChange: (value: string) => void
}

export function FilterBar({ label, options, value, onChange }: FilterBarProps) {
  return (
    <div className="flex items-center gap-3">
      <label className="font-sans text-xs uppercase tracking-widest text-muted">
        {label}
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="font-sans text-sm border border-border dark:border-[#2a2a2a] rounded bg-paper dark:bg-[#1a1a1a] text-ink dark:text-[#F0EFEB] px-3 py-1.5 outline-none focus:ring-1 focus:ring-ink dark:focus:ring-[#F0EFEB] transition-shadow"
      >
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  )
}
