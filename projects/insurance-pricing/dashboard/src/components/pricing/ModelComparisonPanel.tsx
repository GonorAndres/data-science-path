'use client'

import { usePricingFilter } from '@/context/PricingFilterContext'
import { FilterBar } from '@/components/ui/FilterBar'
import { ChartContainer } from '@/components/ui/ChartContainer'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
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
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts'

const MODEL_OPTIONS = ['xgboost', 'lightgbm']
const SHAP_FEATURES = ['BonusMalus', 'DrivAge', 'VehAge', 'LogDensity', 'VehPower', 'Area']

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

  const freqModels = comparison?.models ?? []

  return (
    <div className="py-6">
      {/* Model KPI cards */}
      <section className="py-8 border-b border-border dark:border-[#2a2a2a]">
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-6">
          Metricas de frecuencia
        </h2>
        {compLoading ? (
          <LoadingSpinner />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {freqModels.map((m: any) => (
              <div
                key={m.name}
                className="p-4 rounded border border-border dark:border-[#2a2a2a] bg-surface/30 dark:bg-[#1a1a1a]"
              >
                <p className="font-sans text-xs uppercase tracking-widest text-muted mb-3">
                  {m.name}
                </p>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted">Gini</span>
                    <span className="font-serif text-lg tabular-nums">{m.metrics?.gini?.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted">D-squared</span>
                    <span className="font-serif text-lg tabular-nums">{m.metrics?.deviance_score_d2?.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted">Deviance Poisson</span>
                    <span className="font-serif text-lg tabular-nums">{m.metrics?.mean_poisson_deviance?.toFixed(4)}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Lift chart */}
      <ChartContainer
        title="Curva de lift"
        subtitle="Los asegurados se ordenan por prediccion del modelo en deciles. Un buen modelo separa claramente alto vs bajo riesgo."
        insight="Los modelos GBM logran mayor separacion en los extremos (deciles 1 y 10), demostrando mejor capacidad discriminatoria que el GLM. La diferencia es mas notoria en el decil de mayor riesgo."
      >
        <LiftChart data={liftData as any} isLoading={liftLoading} />
      </ChartContainer>

      {/* Calibration */}
      <ChartContainer
        title="Calibracion"
        subtitle="Prediccion promedio vs siniestralidad real por decil. Un modelo bien calibrado sigue la diagonal."
        insight="El GLM tiende a estar mejor calibrado en los deciles medios (su diseno lo fuerza). Los GBMs pueden sobre o sub-estimar en los extremos, pero su ranking es superior."
      >
        <CalibrationChart data={calibData as any} isLoading={calibLoading} />
      </ChartContainer>

      {/* GLM Coefficients */}
      <ChartContainer
        title="Coeficientes GLM"
        subtitle="Coeficientes del Poisson GLM con relatividades (exp(coef)). Una relatividad >1 indica mayor frecuencia."
        insight="BonusMalus es el predictor mas fuerte: un coeficiente alto en bins superiores refleja el historial de siniestros. Los conductores jovenes y zonas urbanas tambien tienen relatividades altas."
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
            label="Modelo SHAP"
            options={MODEL_OPTIONS}
            value={selectedModel}
            onChange={setSelectedModel}
          />
          <FilterBar
            label="Variable dependencia"
            options={SHAP_FEATURES}
            value={selectedFeature}
            onChange={setSelectedFeature}
          />
        </div>

        <ChartContainer
          title="SHAP -- Importancia de variables"
          subtitle={`Mean |SHAP value| para ${selectedModel === 'xgboost' ? 'XGBoost' : 'LightGBM'}. Mide cuanto contribuye cada variable a la prediccion.`}
          insight="BonusMalus domina la importancia SHAP, seguido de edad del conductor y vehiculo. Esto valida la practica actuarial: el historial de siniestros es el mejor predictor de riesgo futuro."
        >
          <ShapSummary
            data={shapData as any}
            selectedModel={selectedModel === 'lightgbm' ? 'lightgbm' : 'xgboost'}
            isLoading={shapLoading}
          />
        </ChartContainer>

        <ChartContainer
          title={`SHAP Dependencia -- ${selectedFeature}`}
          subtitle={`Efecto marginal de ${selectedFeature} sobre la prediccion de frecuencia.`}
          insight={getShapInsight(selectedFeature)}
        >
          <ShapDependence data={shapDep as any} isLoading={shapDepLoading} />
        </ChartContainer>
      </section>

      {/* Pure premium comparison */}
      <ChartContainer
        title="Prima pura por segmento"
        subtitle="Prima pura = E[frecuencia] x E[severidad]. Comparacion GLM vs XGBoost por zona de densidad."
        insight="Las mayores diferencias entre GLM y GBM aparecen en las zonas extremas (rural y urbano denso). El GBM captura mejor la no-linealidad de la densidad poblacional."
      >
        <PurePremiumComparison data={ppData as any} isLoading={ppLoading} />
      </ChartContainer>

      {/* Double lift -- where GLM fails */}
      <ChartContainer
        title="Donde falla el GLM?"
        subtitle="Double-lift: ratio de prediccion GBM/GLM por decil del GLM. Valores lejos de 1.0 indican segmentos donde el GLM no captura el patron."
        insight="El GBM detecta subsidio cruzado: en los deciles mas bajos del GLM, el GBM predice mas alto (riesgo subestimado), y viceversa en deciles altos. Este analisis es clave para mejorar la tarifa."
      >
        {dlLoading ? (
          <LoadingSpinner />
        ) : dlData?.bins ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={dlData.bins}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E4DF" />
              <XAxis dataKey="bin" tick={{ fontSize: 12 }} label={{ value: 'Decil GLM', position: 'insideBottom', offset: -5, fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} domain={['auto', 'auto']} />
              <Tooltip />
              <ReferenceLine y={1} stroke="#6B6B6B" strokeDasharray="3 3" />
              <Bar dataKey={dlData.bins[0]?.gbm_to_glm_ratio != null ? 'gbm_to_glm_ratio' : 'mean_ratio'} fill="#2D6B45" name="Ratio GBM/GLM" />
            </BarChart>
          </ResponsiveContainer>
        ) : null}
      </ChartContainer>

      {/* Fairness audit */}
      <ChartContainer
        title="Auditoria de fairness -- proxy discrimination"
        subtitle="Comparacion de predicciones por zona de densidad (Area A-F). Areas urbanas densas podrian actuar como proxy de nivel socioeconomico."
        insight="Si el modelo predice sistematicamente mas alto en zonas de alta densidad (urbanas), podria discriminar indirectamente por nivel socioeconomico. En Mexico, esto es especialmente relevante dada la desigualdad entre CDMX y zonas rurales."
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
              <Tooltip />
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
    BonusMalus: 'Efecto monotono creciente: a mayor bonus-malus, mayor frecuencia predicha. El efecto se acelera despues de 100 (historial malo). La relacion no-lineal explica parte de la ventaja del GBM.',
    DrivAge: 'Efecto en U: jovenes (<25) y mayores (>70) tienen SHAP positivo (mayor riesgo). El minimo esta alrededor de 45-55 anos. El GLM con bins lineales no captura bien esta curva.',
    VehAge: 'Vehiculos muy nuevos (0-1) y muy viejos (>15) tienen mayor riesgo. Los nuevos por valor de reposicion, los viejos por mantenimiento deficiente.',
    LogDensity: 'A mayor densidad, mayor frecuencia. Pero la relacion se aplana en densidades muy altas -- efecto que el GLM lineal no captura.',
    VehPower: 'Vehiculos de alta potencia (>10) tienen SHAP positivo. El efecto es moderado comparado con BonusMalus.',
    Area: 'Las areas urbanas (D-F) tienen SHAP positivo consistente. Esto es el proxy de discriminacion a auditar.',
  }
  return insights[feature] ?? 'Selecciona una variable para ver la interpretacion del efecto SHAP.'
}
