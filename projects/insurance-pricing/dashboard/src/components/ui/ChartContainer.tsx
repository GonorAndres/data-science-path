'use client'

interface ChartContainerProps {
  title?: string
  subtitle?: React.ReactNode
  insight: string
  children: React.ReactNode
  className?: string
}

export function ChartContainer({ title, subtitle, insight, children, className }: ChartContainerProps) {
  return (
    <section className={`mb-8 ${className ?? ''}`}>
      <div className="bg-white dark:bg-[#1a1a1a] rounded-xl p-6 shadow-sm border border-border/50 dark:border-[#2a2a2a]/50 hover:shadow-md transition-shadow">
        {title && (
          <h2 className="font-serif text-2xl md:text-3xl tracking-tight text-ink dark:text-[#F0EFEB]">
            {title}
          </h2>
        )}
        {subtitle && (
          <p className="font-sans text-sm text-muted mb-6 max-w-2xl">{subtitle}</p>
        )}
        <p className="font-sans text-base text-ink dark:text-[#F0EFEB] max-w-2xl mb-8">
          {insight}
        </p>
        <div className="w-full">{children}</div>
      </div>
    </section>
  )
}
