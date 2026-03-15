'use client'

import { usePricingFilter } from '@/context/PricingFilterContext'
import { FilterBar } from '@/components/ui/FilterBar'
import { ChartContainer } from '@/components/ui/ChartContainer'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { Tooltip } from '@/components/ui/Tooltip'
import { LiftChart } from '@/components/pricing/LiftChart'
import { CalibrationChart } from '@/components/pricing/CalibrationChart'
import { ShapSummary } from '@/components/pricing/ShapSummary'
import { ShapDependence } from '@/components/pricing/ShapDependence'
import { CoefficientTable } from '@/components/pricing/CoefficientTable'
import { PurePremiumComparison } from '@/components/pricing/PurePremiumComparison'
import {
  useModelComparison,
  useLiftChart,
  useCalibration,
  useCoefficients,
  useShapSummary,
  useShapDependence,
  usePurePremium,
  useDoubleLift,
  useFairnessAudit,
} from '@/hooks/usePricingAPI'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RTooltip,
  ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts'

const MODEL_OPTIONS = ['xgboost', 'lightgbm']
const SHAP_FEATURES = ['BonusMalus', 'DrivAge', 'VehAge', 'LogDensity', 'VehPower', 'Area']
const SHAP_FEATURE_LABELS = [
  'Historial de siniestros (BonusMalus)',
  'Edad del conductor (DrivAge)',
  'Antigüedad del vehículo (VehAge)',
  'Densidad poblacional (LogDensity)',
  'Potencia del motor (VehPower)',
  'Zona urbana/rural (Area)',
]

const FEATURE_DISPLAY: Record<string, string> = {
  BonusMalus: 'Historial de siniestros',
  DrivAge: 'Edad del conductor',
  VehAge: 'Antigüedad del vehículo',
  LogDensity: 'Densidad poblacional',
  VehPower: 'Potencia del motor',
  Area: 'Zona urbana/rural',
}

/** Mini progress bar for metric visualization */
function MetricBar({ value, max, inverted = false }: { value: number; max: number; inverted?: boolean }) {
  const pct = Math.min(Math.max((value / max) * 100, 0), 100)
  const color = inverted
    ? `hsl(${Math.min(pct * 1.2, 120)}, 60%, 40%)`
    : `hsl(${Math.max(120 - pct * 1.2, 0)}, 60%, 40%)`
  return (
    <div className="h-1.5 w-full bg-surface dark:bg-[#2a2a2a] rounded-full mt-1 overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-700"
        style={{ width: `${pct}%`, backgroundColor: color }}
      />
    </div>
  )
}

export function ModelComparisonPanel() {
  const { selectedModel, setSelectedModel, selectedFeature, setSelectedFeature } = usePricingFilter()
  const { data: comparison, isLoading: compLoading } = useModelComparison()
  const { data: liftData, isLoading: liftLoading } = useLiftChart()
  const { data: calibData, isLoading: calibLoading } = useCalibration()
  const { data: coeffData, isLoading: coeffLoading } = useCoefficients()
  const { data: shapData, isLoading: shapLoading } = useShapSummary()
  const { data: shapDep, isLoading: shapDepLoading } = useShapDependence(selectedFeature, selectedModel === 'lightgbm' ? 'lgb' : 'xgb')
  const { data: ppData, isLoading: ppLoading } = usePurePremium()
  const { data: dlData, isLoading: dlLoading } = useDoubleLift()
  const { data: fairData, isLoading: fairLoading } = useFairnessAudit()

  const freqModels = comparison?.frequency ?? comparison?.models ?? []

  return (
    <div className="py-6">
      {/* Model KPI cards */}
      <section className="py-8 border-b border-border dark:border-[#2a2a2a]">
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-6">
          Métricas de frecuencia
        </h2>
        {compLoading ? (
          <LoadingSpinner />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {freqModels.map((m: any, i: number) => {
              const gini = m.gini ?? m.metrics?.gini
              const d2 = m.d_squared ?? m.metrics?.deviance_score_d2
              const dev = m.poisson_deviance ?? m.metrics?.mean_poisson_deviance
              const topColors = ['border-t-glm-blue', 'border-t-gbm-green', 'border-t-lgbm-teal']
              return (
                <div
                  key={m.model ?? m.name}
                  className={`p-4 rounded-lg border border-border dark:border-[#2a2a2a] bg-white dark:bg-[#1a1a1a] shadow-sm border-t-[3px] ${topColors[i % topColors.length]}`}
                >
                  <p className="font-sans text-xs uppercase tracking-widest text-muted mb-3">
                    {m.model ?? m.name}
                  </p>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between">
                        <Tooltip term="Gini">Capacidad de ranking: 0 = no distingue riesgo, 1 = perfecto.</Tooltip>
                        <span className="font-serif text-lg tabular-nums">{gini?.toFixed(4)}</span>
                      </div>
                      {gini != null && <MetricBar value={gini} max={1} />}
                    </div>
                    <div>
                      <div className="flex justify-between">
                        <Tooltip term="D-squared">Porcentaje de varianza del riesgo que el modelo explica.</Tooltip>
                        <span className="font-serif text-lg tabular-nums">{d2?.toFixed(4)}</span>
                      </div>
                      {d2 != null && <MetricBar value={d2} max={1} />}
                    </div>
                    <div>
                      <div className="flex justify-between">
                        <Tooltip term="Error Poisson">Error de predicción del modelo. Menor es mejor.</Tooltip>
                        <span className="font-serif text-lg tabular-nums">{dev?.toFixed(4)}</span>
                      </div>
                      {dev != null && <MetricBar value={dev} max={1} inverted />}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </section>

      {/* Lift chart */}
      <ChartContainer
        title="Curva de lift"
        subtitle="Los asegurados se ordenan por predicción del modelo en 10 grupos. Un buen modelo separa claramente alto vs bajo riesgo."
        insight="Los modelos GBM logran mayor separación en los extremos (grupos 1 y 10), demostrando mejor capacidad para separar alto vs bajo riesgo que el GLM. La diferencia es más notoria en el grupo de mayor riesgo."
      >
        <LiftChart data={liftData as any} isLoading={liftLoading} />
      </ChartContainer>

      {/* Calibration */}
      <ChartContainer
        title="Calibración"
        subtitle="Predicción promedio vs siniestralidad real por grupo. Un modelo bien calibrado: las predicciones coinciden con la realidad."
        insight="El GLM tiende a estar mejor calibrado en los grupos medios (su diseño lo fuerza). Los GBMs pueden sobre o sub-estimar en los extremos, pero su ranking de riesgo es superior."
      >
        <CalibrationChart data={calibData as any} isLoading={calibLoading} />
      </ChartContainer>

      {/* GLM Coefficients */}
      <ChartContainer
        title="Coeficientes GLM"
        subtitle={<>Coeficientes del modelo tradicional con <Tooltip term="relatividades">Multiplicadores del riesgo base. Una relatividad de 1.25 significa 25% más riesgo.</Tooltip>. Una relatividad mayor a 1 indica mayor frecuencia de siniestros.</>}
        insight="El historial de siniestros (BonusMalus) es el predictor más fuerte: valores altos reflejan un mal historial. Los conductores jóvenes y zonas urbanas también tienen relatividades altas."
      >
        <CoefficientTable
          data={coeffData as any}
          activeType="frequency"
          isLoading={coeffLoading}
        />
      </ChartContainer>

      {/* SHAP analysis */}
      <section className="py-6">
        <div className="flex gap-4 mb-4">
          <FilterBar
            label="Modelo"
            options={MODEL_OPTIONS}
            value={selectedModel}
            onChange={setSelectedModel}
          />
          <FilterBar
            label="Variable"
            options={SHAP_FEATURES}
            labels={SHAP_FEATURE_LABELS}
            value={selectedFeature}
            onChange={setSelectedFeature}
          />
        </div>

        <ChartContainer
          title="Importancia de variables"
          subtitle={<>Cuánto influye cada variable en la predicción de {selectedModel === 'xgboost' ? 'XGBoost' : 'LightGBM'}, medido con <Tooltip term="SHAP">Herramienta que descompone cada predicción en la contribución de cada variable.</Tooltip>.</>}
          insight="El historial de siniestros (BonusMalus) domina, seguido de edad del conductor y vehículo. Esto valida la práctica actuarial: el historial de siniestros es el mejor predictor de riesgo futuro."
        >
          <ShapSummary
            data={shapData as any}
            selectedModel={selectedModel === 'lightgbm' ? 'lightgbm' : 'xgboost'}
            isLoading={shapLoading}
          />
        </ChartContainer>

        <ChartContainer
          title={`Efecto de: ${FEATURE_DISPLAY[selectedFeature] ?? selectedFeature}`}
          subtitle={`Cómo cambia la predicción de frecuencia a medida que cambia ${FEATURE_DISPLAY[selectedFeature] ?? selectedFeature}.`}
          insight={getShapInsight(selectedFeature)}
        >
          <ShapDependence data={shapDep as any} isLoading={shapDepLoading} />
        </ChartContainer>
      </section>

      {/* Pure premium comparison */}
      <ChartContainer
        title="Prima pura por segmento"
        subtitle="Prima pura = frecuencia esperada x costo esperado. Comparación GLM vs XGBoost por zona de densidad."
        insight="Las mayores diferencias entre GLM y GBM aparecen en las zonas extremas (rural y urbano denso). El GBM captura mejor los patrones curvos de la densidad poblacional."
      >
        <PurePremiumComparison data={ppData as any} isLoading={ppLoading} />
      </ChartContainer>

      {/* Double lift: where GLM fails */}
      <ChartContainer
        title="¿Dónde falla el GLM?"
        subtitle="¿Qué segmentos está tarificando mal el modelo tradicional? Comparamos las predicciones del GLM contra el GBM para encontrar dónde cobra de más o de menos."
        insight="El GBM detecta subsidio cruzado: en los grupos donde el GLM predice bajo riesgo, el GBM predice más alto (riesgo subestimado), y viceversa. Este análisis es clave para mejorar la tarifa."
      >
        {dlLoading ? (
          <LoadingSpinner />
        ) : dlData?.bins ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={dlData.bins}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E4DF" />
              <XAxis dataKey="bin" tick={{ fontSize: 12 }} label={{ value: 'Grupo GLM', position: 'insideBottom', offset: -5, fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} domain={['auto', 'auto']} />
              <RTooltip />
              <ReferenceLine y={1} stroke="#6B6B6B" strokeDasharray="3 3" />
              <Bar dataKey={dlData.bins[0]?.gbm_to_glm_ratio != null ? 'gbm_to_glm_ratio' : 'mean_ratio'} fill="#2D6B45" name="Ratio GBM/GLM" />
            </BarChart>
          </ResponsiveContainer>
        ) : null}
      </ChartContainer>

      {/* Fairness audit */}
      <ChartContainer
        title="Auditoría de equidad: discriminación indirecta"
        subtitle="Comparación de predicciones por zona de densidad (Area A-F). Las zonas urbanas densas podrían actuar como proxy de nivel socioeconómico."
        insight="Si el modelo predice sistemáticamente más alto en zonas de alta densidad (urbanas), podría discriminar indirectamente por nivel socioeconómico. En México, esto es especialmente relevante dada la desigualdad entre CDMX y zonas rurales."
      >
        {fairLoading ? (
          <LoadingSpinner />
        ) : (fairData?.area_audit?.groups ?? fairData?.area_audit?.segments) ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={(fairData.area_audit.groups ?? fairData.area_audit.segments).map((s: any) => ({
              group: s.group ?? s.area ?? s.label,
              observed: s.actual_ratio_to_mean ?? s.observed_freq_ratio ?? (s.avg_observed_freq && s.avg_observed_freq / 0.039),
              glm: s['Poisson GLM_ratio_to_mean'] ?? s.glm_ratio ?? (s.avg_predicted_freq_glm && s.avg_predicted_freq_glm / 0.039),
              gbm: s['LightGBM_ratio_to_mean'] ?? s['XGBoost_ratio_to_mean'] ?? s.lgbm_ratio ?? (s.avg_predicted_freq_lgbm && s.avg_predicted_freq_lgbm / 0.039),
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E4DF" />
              <XAxis dataKey="group" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <RTooltip />
              <Legend />
              <Bar dataKey="observed" fill="#6B6B6B" name="Ratio real" />
              <Bar dataKey="glm" fill="#1E3A5F" name="GLM ratio" />
              <Bar dataKey="gbm" fill="#2D6B45" name="GBM ratio" />
            </BarChart>
          </ResponsiveContainer>
        ) : null}
      </ChartContainer>
    </div>
  )
}

function getShapInsight(feature: string): string {
  const insights: Record<string, string> = {
    BonusMalus: 'A peor historial, más riesgo predicho (relación directa). El efecto se acelera después de 100 (historial malo). Este patrón curvo explica parte de la ventaja del GBM sobre el GLM.',
    DrivAge: 'Jóvenes y mayores tienen más siniestros (forma de U). El punto más seguro está alrededor de 45-55 años. El GLM con categorías fijas no captura bien esta curva.',
    VehAge: 'Vehículos muy nuevos (0-1) y muy viejos (>15) tienen mayor riesgo. Los nuevos por su alto valor de reposición, los viejos por mantenimiento deficiente.',
    LogDensity: 'A mayor densidad, mayor frecuencia. Pero la relación se aplana en densidades muy altas, un patrón curvo que el modelo tradicional no captura bien.',
    VehPower: 'Vehículos de alta potencia (>10) muestran mayor riesgo predicho. El efecto es moderado comparado con el historial de siniestros.',
    Area: 'Las áreas urbanas (D-F) consistentemente muestran mayor riesgo predicho. Este es el patrón de discriminación indirecta a auditar.',
  }
  return insights[feature] ?? 'Selecciona una variable para ver cómo afecta la predicción del modelo.'
}
