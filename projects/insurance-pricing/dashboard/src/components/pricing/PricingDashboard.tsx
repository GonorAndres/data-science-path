'use client'

import { PricingFilterProvider, usePricingFilter } from '@/context/PricingFilterContext'
import { TabNav } from '@/components/ui/TabNav'
import { KPICard } from '@/components/ui/KPICard'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { ContextPanel } from '@/components/pricing/ContextPanel'
import { EdaPanel } from '@/components/pricing/EdaPanel'
import { ModelComparisonPanel } from '@/components/pricing/ModelComparisonPanel'
import { TechnicalPanel } from '@/components/pricing/TechnicalPanel'
import { useOverview } from '@/hooks/usePricingAPI'

const TABS = [
  { id: 'contexto', label: 'Contexto' },
  { id: 'eda', label: 'EDA Explorer' },
  { id: 'modelos', label: 'Comparacion de Modelos' },
  { id: 'tecnico', label: 'Proceso Tecnico' },
]

function DashboardInner() {
  const { activeTab, setActiveTab } = usePricingFilter()
  const { data: overview, isLoading: overviewLoading } = useOverview()

  return (
    <div className="min-h-screen bg-paper dark:bg-[#141414] text-ink dark:text-[#F0EFEB]">
      {/* Header */}
      <header className="border-b border-border dark:border-[#2a2a2a] px-6 py-3 flex items-center justify-between max-w-5xl mx-auto">
        <span className="font-sans text-xs tracking-widest uppercase text-muted">
          Data Science Path
        </span>
      </header>

      <main className="max-w-5xl mx-auto px-6">
        {/* Title section */}
        <section className="pt-16 pb-12 border-b border-border dark:border-[#2a2a2a]">
          <p className="font-sans text-xs tracking-widest uppercase text-muted mb-4">
            Proyecto de Pricing Actuarial -- 2026
          </p>
          <h1 className="font-serif text-5xl md:text-7xl leading-none tracking-tight mb-6">
            Seguro Auto<br />GLM vs ML
          </h1>
          <p className="font-sans text-base text-muted max-w-xl leading-relaxed">
            Lecciones de la ciencia de datos actuarial europea para el mercado
            mexicano de seguros de auto. Modelos de frecuencia-severidad sobre
            freMTPL2 -- el benchmark global en pricing de seguros.
          </p>
        </section>

        {/* KPI bar */}
        <section className="py-12 border-b border-border dark:border-[#2a2a2a]">
          {overviewLoading ? (
            <LoadingSpinner />
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 md:gap-0 md:divide-x md:divide-border dark:md:divide-[#2a2a2a]">
              <div className="md:pr-8">
                <KPICard
                  label="Polizas"
                  value={overview?.portfolio_summary?.n_policies ?? 0}
                  delay={0}
                />
              </div>
              <div className="md:px-8">
                <KPICard
                  label="Tasa de siniestro"
                  value={(overview?.portfolio_summary?.claim_rate ?? 0) * 100}
                  suffix="%"
                  decimals={1}
                  delay={0.1}
                />
              </div>
              <div className="md:px-8">
                <KPICard
                  label="Siniestro promedio"
                  value={overview?.portfolio_summary?.avg_claim_amount_eur ?? 0}
                  prefix="EUR "
                  decimals={0}
                  delay={0.2}
                />
              </div>
              <div className="md:pl-8">
                <KPICard
                  label="Mejor Gini (freq)"
                  value={overview?.models?.frequency?.best_gini ?? 0}
                  decimals={4}
                  delay={0.3}
                />
              </div>
            </div>
          )}
        </section>

        {/* Tab navigation */}
        <TabNav tabs={TABS} activeTab={activeTab} onTabChange={setActiveTab} />

        {/* Tab content */}
        <div className="min-h-[600px]">
          {activeTab === 'contexto' && <ContextPanel />}
          {activeTab === 'eda' && <EdaPanel />}
          {activeTab === 'modelos' && <ModelComparisonPanel />}
          {activeTab === 'tecnico' && <TechnicalPanel />}
        </div>

        {/* Footer */}
        <footer className="py-10 border-t border-border dark:border-[#2a2a2a] mt-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <p className="font-sans text-xs text-muted">
            Andres Gonzalez Ortega -- UNAM Actuaria, 2026
          </p>
          <p className="font-sans text-xs text-muted">
            Datos: freMTPL2 (OpenML) | Stack: Python + Next.js + FastAPI
          </p>
        </footer>
      </main>
    </div>
  )
}

export function PricingDashboard() {
  return (
    <PricingFilterProvider>
      <DashboardInner />
    </PricingFilterProvider>
  )
}
