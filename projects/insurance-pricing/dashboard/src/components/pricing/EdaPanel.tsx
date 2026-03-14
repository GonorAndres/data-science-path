'use client'

import { usePricingFilter } from '@/context/PricingFilterContext'
import { FilterBar } from '@/components/ui/FilterBar'
import { ChartContainer } from '@/components/ui/ChartContainer'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { FeatureDistribution } from '@/components/pricing/FeatureDistribution'
import { ClaimFrequencyHeatmap } from '@/components/pricing/ClaimFrequencyHeatmap'
import {
  useDistributions,
  useExposure,
  useClaimFrequency,
} from '@/hooks/usePricingAPI'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts'

const FEATURES = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density', 'VehBrand', 'VehGas', 'Area', 'Region']

export function EdaPanel() {
  const { selectedFeature, setSelectedFeature } = usePricingFilter()
  const { data: allDistributions, isLoading: distLoading } = useDistributions()
  const { data: exposure, isLoading: expLoading } = useExposure()
  const { data: claimFreq, isLoading: freqLoading } = useClaimFrequency()

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const currentDist = (allDistributions as any)?.[selectedFeature]

  return (
    <div className="py-6">
      {/* Feature selector */}
      <div className="py-4">
        <FilterBar
          label="Variable"
          options={FEATURES}
          value={selectedFeature}
          onChange={setSelectedFeature}
        />
      </div>

      {/* Feature distribution */}
      <ChartContainer
        title="Distribucion de variables"
        subtitle={`Distribucion de ${selectedFeature} en 677,991 polizas del portafolio frances.`}
        insight={getFeatureInsight(selectedFeature)}
      >
        {distLoading ? (
          <LoadingSpinner />
        ) : currentDist ? (
          <FeatureDistribution
            data={currentDist}
            featureName={selectedFeature}
            isLoading={false}
          />
        ) : (
          <p className="text-sm text-muted">No hay datos para esta variable.</p>
        )}
      </ChartContainer>

      {/* Exposure by Area */}
      <ChartContainer
        title="Exposicion por area"
        subtitle="Total de exposicion (anos-poliza) y tasa de siniestro por zona de densidad poblacional."
        insight="Las zonas urbanas (D-F) concentran mas exposicion y mayor frecuencia de siniestros. En Mexico, la concentracion en CDMX/Monterrey/Guadalajara seria aun mas pronunciada."
      >
        {expLoading ? (
          <LoadingSpinner />
        ) : exposure?.by_area ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={exposure.by_area}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E4DF" />
              <XAxis dataKey={exposure.by_area[0]?.Area ? 'Area' : 'area'} tick={{ fontSize: 12 }} />
              <YAxis yAxisId="left" tick={{ fontSize: 12 }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar yAxisId="left" dataKey="total_exposure" fill="#1E3A5F" name="Exposicion total" />
              <Bar yAxisId="right" dataKey={exposure.by_area[0]?.claim_rate != null ? 'claim_rate' : 'claim_frequency'} fill="#C4841D" name="Tasa siniestro" />
            </BarChart>
          </ResponsiveContainer>
        ) : null}
      </ChartContainer>

      {/* Claim frequency heatmap */}
      <ChartContainer
        title="Frecuencia de siniestros: edad del conductor x edad del vehiculo"
        subtitle="Tasa de siniestros promedio por combinacion de edad del conductor y antiguedad del vehiculo."
        insight="Los conductores jovenes (18-26) con vehiculos viejos tienen la frecuencia mas alta -- patron clasico en pricing actuarial. El sistema bonus-malus penaliza este perfil, pero el GLM podria no capturar la interaccion completa."
      >
        {freqLoading ? (
          <LoadingSpinner />
        ) : claimFreq ? (
          <ClaimFrequencyHeatmap data={(claimFreq?.heatmap ?? claimFreq) as any} isLoading={false} />
        ) : null}
      </ChartContainer>
    </div>
  )
}

function getFeatureInsight(feature: string): string {
  const insights: Record<string, string> = {
    VehPower: 'La mayoria de vehiculos tienen potencia 5-7. En Mexico, la distribucion seria similar pero con mayor concentracion en vehiculos compactos (Nissan March, VW Vento).',
    VehAge: 'Los vehiculos nuevos (0-2 anos) dominan el portafolio. En Mexico, la edad promedio del parque vehicular es mayor (~12 anos), lo que cambiaria esta distribucion significativamente.',
    DrivAge: 'Distribucion normal centrada en 40-50 anos. Los conductores jovenes (<26) son pocos pero con mayor frecuencia de siniestros -- clave para el bonus-malus.',
    BonusMalus: 'Fuertemente sesgado a la izquierda: la mayoria de asegurados tienen bonus 50-60 (los mejores). Esta es la variable mas predictiva de frecuencia. Mexico usa un sistema similar.',
    Density: 'Extremadamente sesgada a la derecha. Unas pocas polizas en zonas muy densas (Paris). Log-transformar es esencial para modelar esta variable.',
    VehBrand: 'Unas pocas marcas dominan (Renault, Peugeot en Francia). En Mexico serian Nissan, VW, Chevrolet, Toyota.',
    VehGas: 'Aproximadamente 60% gasolina, 40% diesel en Francia. En Mexico seria >90% gasolina con crecimiento de electricos.',
    Area: 'Seis zonas de densidad (A=rural a F=urbana). Clave para pricing pero potencial proxy de nivel socioeconomico.',
    Region: 'Regiones administrativas francesas. Las 10 principales cubren ~80% del portafolio.',
  }
  return insights[feature] ?? 'Selecciona una variable para ver su interpretacion.'
}
