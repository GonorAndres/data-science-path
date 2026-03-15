'use client'

import { usePricingFilter } from '@/context/PricingFilterContext'
import { FilterBar } from '@/components/ui/FilterBar'
import { ChartContainer } from '@/components/ui/ChartContainer'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { Tooltip } from '@/components/ui/Tooltip'
import { FeatureDistribution } from '@/components/pricing/FeatureDistribution'
import { ClaimFrequencyHeatmap } from '@/components/pricing/ClaimFrequencyHeatmap'
import {
  useDistributions,
  useExposure,
  useClaimFrequency,
} from '@/hooks/usePricingAPI'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip,
  ResponsiveContainer,
} from 'recharts'

const FEATURES = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density', 'VehBrand', 'VehGas', 'Area', 'Region']
const FEATURE_LABELS = [
  'Potencia del motor (VehPower)',
  'Antigüedad del vehículo (VehAge)',
  'Edad del conductor (DrivAge)',
  'Historial de siniestros (BonusMalus)',
  'Densidad poblacional (Density)',
  'Marca del vehículo (VehBrand)',
  'Tipo de combustible (VehGas)',
  'Zona urbana/rural (Area)',
  'Región administrativa (Region)',
]

const FEATURE_DISPLAY: Record<string, string> = {
  VehPower: 'Potencia del motor',
  VehAge: 'Antigüedad del vehículo',
  DrivAge: 'Edad del conductor',
  BonusMalus: 'Historial de siniestros',
  Density: 'Densidad poblacional',
  VehBrand: 'Marca del vehículo',
  VehGas: 'Tipo de combustible',
  Area: 'Zona urbana/rural',
  Region: 'Región administrativa',
}

export function EdaPanel() {
  const { selectedFeature, setSelectedFeature } = usePricingFilter()
  const { data: allDistributions, isLoading: distLoading } = useDistributions()
  const { data: exposure, isLoading: expLoading } = useExposure()
  const { data: claimFreq, isLoading: freqLoading } = useClaimFrequency()

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const currentDist = (allDistributions as any)?.[selectedFeature]
  const displayName = FEATURE_DISPLAY[selectedFeature] ?? selectedFeature

  return (
    <div className="py-6">
      {/* Feature selector */}
      <div className="py-4">
        <FilterBar
          label="Variable"
          options={FEATURES}
          labels={FEATURE_LABELS}
          value={selectedFeature}
          onChange={setSelectedFeature}
        />
      </div>

      {/* Feature distribution */}
      <ChartContainer
        title={`Distribución: ${displayName}`}
        subtitle={`Distribución de ${displayName} en 677,991 pólizas del portafolio francés.`}
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
        title="Tiempo asegurado por zona"
        subtitle={<>Total de <Tooltip term="exposición">Tiempo que cada póliza estuvo activa, medido en años. Ej: 0.5 = 6 meses asegurado.</Tooltip> (años-póliza) y porcentaje de pólizas con siniestro por zona de densidad poblacional.</>}
        insight="Las zonas urbanas (D-F) concentran más tiempo asegurado y mayor frecuencia de siniestros. En México, la concentración en CDMX/Monterrey/Guadalajara sería aún más pronunciada."
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
              <RTooltip />
              <Bar yAxisId="left" dataKey="total_exposure" fill="#1E3A5F" name="Exposición total" />
              <Bar yAxisId="right" dataKey={exposure.by_area[0]?.claim_rate != null ? 'claim_rate' : 'claim_frequency'} fill="#C4841D" name="Tasa siniestro" />
            </BarChart>
          </ResponsiveContainer>
        ) : null}
      </ChartContainer>

      {/* Claim frequency heatmap */}
      <ChartContainer
        title="Frecuencia de siniestros: edad del conductor x antigüedad del vehículo"
        subtitle="Porcentaje promedio de pólizas con siniestro, por combinación de edad del conductor y antigüedad del vehículo."
        insight="Los conductores jóvenes (18-26) con vehículos viejos tienen la frecuencia más alta. El sistema bonus-malus penaliza este perfil, pero el modelo tradicional (GLM) podría no capturar la interacción completa."
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
    VehPower: 'La mayoría de vehículos tienen potencia 5-7. En México, la distribución sería similar pero con mayor concentración en vehículos compactos (Nissan March, VW Vento).',
    VehAge: 'Los vehículos nuevos (0-2 años) dominan el portafolio. En México, la edad promedio del parque vehicular es mayor (~12 años), lo que cambiaría esta distribución significativamente.',
    DrivAge: 'Concentrada en 40-50 años. Los conductores jóvenes (<26) son pocos pero tienen más siniestros; por eso el sistema bonus-malus los penaliza.',
    BonusMalus: 'La gran mayoría de conductores tienen buen historial (50-60, la mejor calificación). Esta es la variable más predictiva de frecuencia. México usa un sistema similar.',
    Density: 'Muy concentrada en valores bajos, con unas pocas pólizas en zonas muy densas (París). Los valores van de 1 a 27,000; se comprime la escala para que los modelos funcionen bien.',
    VehBrand: 'Unas pocas marcas dominan (Renault, Peugeot en Francia). En México serían Nissan, VW, Chevrolet, Toyota.',
    VehGas: 'Aproximadamente 60% gasolina, 40% diésel en Francia. En México sería >90% gasolina con crecimiento de eléctricos.',
    Area: 'Seis zonas de densidad (A=rural a F=urbana). Clave para pricing, pero puede reflejar indirectamente el nivel socioeconómico, lo que requiere cuidado ético.',
    Region: 'Regiones administrativas francesas. Las 10 principales cubren ~80% del portafolio.',
  }
  return insights[feature] ?? 'Selecciona una variable para ver su interpretación.'
}
