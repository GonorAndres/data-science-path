'use client'

import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Shield, User, Car, Gauge, MapPin, Clock, ChevronRight } from 'lucide-react'
import { Tooltip } from '@/components/ui/Tooltip'

const fadeIn = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 },
}

const DATASET_FEATURES = [
  {
    icon: Shield,
    code: 'BonusMalus',
    label: 'Historial del conductor',
    short: 'Calificación basada en siniestros pasados',
    detail: 'Como un score crediticio, pero para seguros. 50 = sin siniestros por años (el mejor), 150+ = historial malo.',
    scale: { min: 50, max: 230, goodLabel: '50 (sin siniestros)', badLabel: '230 (historial malo)', colors: ['#2D6B45', '#C4841D', '#B91C1C'] },
    highlight: 'La variable más predictiva del dataset',
  },
  {
    icon: User,
    code: 'DrivAge',
    label: 'Edad del conductor',
    short: '18 a 97 años',
    detail: 'Los más jóvenes y los más mayores tienen más siniestros (forma de U). El punto más seguro: 45-55 años.',
    scale: { min: 18, max: 97, goodLabel: '45-55 (menos riesgo)', badLabel: '18-25 / 70+ (más riesgo)', colors: ['#C4841D', '#2D6B45', '#C4841D'] },
    highlight: 'Patrón en U: jóvenes y mayores = más riesgo',
  },
  {
    icon: Car,
    code: 'VehAge',
    label: 'Antigüedad del vehículo',
    short: '0 a 20+ años',
    detail: 'Vehículos muy nuevos (alto valor) y muy viejos (mal mantenimiento) tienen más siniestros.',
    scale: null,
    highlight: 'La mayoría del portafolio: 0-6 años',
  },
  {
    icon: Gauge,
    code: 'VehPower',
    label: 'Potencia del motor',
    short: 'Escala 1-15',
    detail: 'Proxy del valor y velocidad del vehículo. La mayoría se concentra en potencia 5-7.',
    scale: null,
    highlight: 'Efecto moderado vs BonusMalus',
  },
  {
    icon: MapPin,
    code: 'Area / Density',
    label: 'Zona y densidad',
    short: 'A (rural) a F (urbana densa)',
    detail: 'Densidad poblacional de la zona. Más densidad = más tráfico = más siniestros. París es el extremo.',
    scale: { min: 0, max: 5, goodLabel: 'A rural', badLabel: 'F urbana', colors: ['#2D6B45', '#1E3A5F', '#C4841D'] },
    highlight: 'Requiere cuidado ético: puede ser proxy socioeconómico',
  },
  {
    icon: Clock,
    code: 'Exposure',
    label: 'Tiempo asegurado',
    short: '0 a 1 año',
    detail: 'Fracción del año que la póliza estuvo activa. Ej: 0.5 = 6 meses. Se usa para ajustar las predicciones.',
    scale: null,
    highlight: 'Corrige por pólizas parciales',
  },
]

function FeatureCard({ feature }: { feature: typeof DATASET_FEATURES[0] }) {
  const [expanded, setExpanded] = useState(false)
  const Icon = feature.icon

  return (
    <motion.div
      layout
      onClick={() => setExpanded((v) => !v)}
      className="bg-white dark:bg-[#1e1e1e] rounded-lg border border-border/50 dark:border-[#2a2a2a]/50 p-4 cursor-pointer hover:shadow-md transition-shadow"
    >
      <div className="flex items-start gap-3">
        <div className="w-8 h-8 rounded-full bg-glm-blue/10 dark:bg-glm-blue/20 flex items-center justify-center shrink-0">
          <Icon className="w-4 h-4 text-glm-blue" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <p className="font-sans text-sm font-medium text-ink dark:text-[#F0EFEB]">
              {feature.label}
            </p>
            <span className="font-mono text-[10px] text-muted bg-surface dark:bg-[#2a2a2a] px-1.5 py-0.5 rounded">
              {feature.code}
            </span>
          </div>
          <p className="font-sans text-xs text-muted mt-0.5">{feature.short}</p>
        </div>
        <ChevronRight className={`w-4 h-4 text-muted shrink-0 transition-transform ${expanded ? 'rotate-90' : ''}`} />
      </div>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="pt-3 mt-3 border-t border-border/30 dark:border-[#2a2a2a]/30">
              <p className="font-sans text-sm text-ink dark:text-[#F0EFEB] leading-relaxed mb-2">
                {feature.detail}
              </p>
              {feature.scale && (
                <div className="mb-2">
                  <div className="h-2 rounded-full overflow-hidden"
                    style={{ background: `linear-gradient(to right, ${feature.scale.colors.join(', ')})` }}
                  />
                  <div className="flex justify-between mt-1">
                    <span className="font-sans text-[10px] text-muted">{feature.scale.goodLabel}</span>
                    <span className="font-sans text-[10px] text-muted">{feature.scale.badLabel}</span>
                  </div>
                </div>
              )}
              <p className="font-sans text-xs text-accent-amber font-medium">
                {feature.highlight}
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

function FrequencySeverityVisual() {
  return (
    <div className="flex flex-col sm:flex-row items-center gap-4 sm:gap-6 py-6 px-4 bg-white dark:bg-[#1e1e1e] rounded-lg border border-border/50 dark:border-[#2a2a2a]/50">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
        className="flex flex-col items-center gap-1"
      >
        <div className="w-16 h-16 rounded-xl bg-glm-blue/10 flex items-center justify-center">
          <span className="text-2xl">&#x1D453;</span>
        </div>
        <p className="font-sans text-xs font-medium text-ink dark:text-[#F0EFEB]">Frecuencia</p>
        <p className="font-sans text-[10px] text-muted text-center">Qué tan seguido<br/>ocurren siniestros</p>
      </motion.div>

      <span className="font-serif text-2xl text-muted">x</span>

      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2 }}
        className="flex flex-col items-center gap-1"
      >
        <div className="w-16 h-16 rounded-xl bg-gbm-green/10 flex items-center justify-center">
          <span className="text-2xl">$</span>
        </div>
        <p className="font-sans text-xs font-medium text-ink dark:text-[#F0EFEB]">Severidad</p>
        <p className="font-sans text-[10px] text-muted text-center">Cuánto cuestan<br/>cuando ocurren</p>
      </motion.div>

      <span className="font-serif text-2xl text-muted">=</span>

      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3 }}
        className="flex flex-col items-center gap-1"
      >
        <div className="w-16 h-16 rounded-xl bg-accent-amber/10 flex items-center justify-center">
          <span className="text-2xl">P</span>
        </div>
        <p className="font-sans text-xs font-medium text-ink dark:text-[#F0EFEB]">Prima pura</p>
        <p className="font-sans text-[10px] text-muted text-center">Lo que debería<br/>costar la póliza</p>
      </motion.div>
    </div>
  )
}

export function ContextPanel() {
  return (
    <div className="py-10 space-y-12">
      {/* El problema */}
      <motion.section {...fadeIn}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          El problema
        </h2>
        <p className="font-sans text-base text-muted max-w-2xl leading-relaxed mb-4">
          México es el único país de la OCDE sin seguro de auto obligatorio a nivel
          federal. Solo el 30-33% de los vehículos están asegurados, frente al 95%+
          en Europa. El pricing tradicional (tablas manuales, pocas variables) deja
          primas que no reflejan el riesgo real, un subsidio cruzado que encarece el
          seguro para los buenos conductores y desincentiva la compra.
        </p>
        <p className="font-sans text-base text-ink dark:text-[#F0EFEB] max-w-2xl leading-relaxed">
          La ciencia de datos actuarial está transformando el pricing en Europa y EE.UU.
          Este proyecto explora qué pueden aprender las aseguradoras mexicanas (Qualitas,
          GNP, Crabi) de esa revolución, usando el benchmark global freMTPL2.
        </p>
      </motion.section>

      {/* Los datos */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.05 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-2">
          Los datos: freMTPL2
        </h2>
        <p className="font-sans text-sm text-muted max-w-2xl mb-6">
          677,991 pólizas reales de seguro de auto en Francia. El dataset de referencia mundial
          en pricing actuarial. Toca una tarjeta para ver detalles.
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-8">
          {DATASET_FEATURES.map((f) => (
            <FeatureCard key={f.code} feature={f} />
          ))}
        </div>

        <h3 className="font-serif text-xl tracking-tight mb-3">
          Cómo se calcula la prima
        </h3>
        <FrequencySeverityVisual />
      </motion.section>

      {/* Contexto del mercado */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.1 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          Contexto del mercado mexicano
        </h2>
        <div className="grid md:grid-cols-2 gap-6 max-w-3xl">
          {[
            { stat: '30-33%', desc: 'de vehículos asegurados en México (vs 95%+ en Europa)' },
            { stat: '15-20%', desc: 'de aseguradoras mexicanas usan AI/ML en pricing' },
            { stat: '33%', desc: 'cuota de mercado de Qualitas (métodos tradicionales)' },
            { stat: '68', desc: 'insurtechs en México (2do en LatAm), solo 1 con licencia auto' },
            { stat: '96.5%', desc: 'penetración móvil: oportunidad para telemática' },
            { stat: '136K+', desc: 'robos de vehículos/año, riesgo clave para pricing' },
          ].map((item) => (
            <div key={item.stat} className="flex gap-4 items-start">
              <span className="font-serif text-3xl text-glm-blue dark:text-[#4A8CC7] leading-none shrink-0 w-20 text-right">
                {item.stat}
              </span>
              <p className="font-sans text-sm text-muted leading-relaxed">{item.desc}</p>
            </div>
          ))}
        </div>
      </motion.section>

      {/* El debate GLM vs ML */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.2 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          El debate: GLM vs Machine Learning
        </h2>
        <div className="space-y-4 max-w-2xl">
          <div className="border-l-2 border-glm-blue pl-4">
            <p className="font-sans text-sm font-medium text-glm-blue dark:text-[#4A8CC7] mb-1">
              <Tooltip term="GLMs">Modelos estadísticos clásicos que los reguladores conocen y confían. Predecibles y auditables.</Tooltip> (Modelos Lineales Generalizados)
            </p>
            <p className="font-sans text-sm text-muted leading-relaxed">
              Estándar global desde los 90s. Transparentes, auditables por reguladores,
              respaldados por teoría actuarial. Poisson para frecuencia, Gamma para severidad.
              Limitación: no capturan interacciones complejas ni patrones curvos.
            </p>
          </div>
          <div className="border-l-2 border-gbm-green pl-4">
            <p className="font-sans text-sm font-medium text-gbm-green dark:text-[#5AAF7A] mb-1">
              Gradient Boosting (XGBoost, LightGBM)
            </p>
            <p className="font-sans text-sm text-muted leading-relaxed">
              Noll et al. (2020) demostraron que GBM supera a GLM en <Tooltip term="error de predicción">Medida técnica llamada &quot;deviance&quot;: cuánto se equivoca el modelo. Menor es mejor.</Tooltip> sobre
              freMTPL2. Capturan interacciones automáticamente. Pero son &quot;cajas negras&quot;
              para reguladores; <Tooltip term="SHAP">Herramienta que muestra cuánto influye cada variable en la predicción del modelo.</Tooltip> y gráficas de efecto parcial son necesarios para explicabilidad.
            </p>
          </div>
          <div className="border-l-2 border-accent-amber pl-4">
            <p className="font-sans text-sm font-medium text-accent-amber mb-1">
              Enfoque híbrido (la respuesta emergente)
            </p>
            <p className="font-sans text-sm text-muted leading-relaxed">
              GBM audita los residuos del GLM: identifica dónde falla el modelo tradicional.
              SHAP explica las decisiones del GBM. Auditorías de <Tooltip term="discriminación indirecta">Cuando una variable aparentemente neutral (como zona) termina discriminando por nivel socioeconómico.</Tooltip> detectan sesgos ocultos.
              MDPI (2024): GLM+ANN híbrido supera a todos los modelos individuales.
            </p>
          </div>
        </div>
      </motion.section>

      {/* Regulación comparada */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.3 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          Regulación comparada
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm font-sans max-w-3xl">
            <thead>
              <tr className="border-b border-border dark:border-[#2a2a2a]">
                <th className="text-left py-2 pr-4 text-xs uppercase tracking-widest text-muted">Jurisdicción</th>
                <th className="text-left py-2 pr-4 text-xs uppercase tracking-widest text-muted">Marco</th>
                <th className="text-left py-2 text-xs uppercase tracking-widest text-muted">ML en pricing</th>
              </tr>
            </thead>
            <tbody className="text-muted">
              <tr className="border-b border-border/50 dark:border-[#2a2a2a]/50">
                <td className="py-2 pr-4 text-ink dark:text-[#F0EFEB]">México (CNSF/LISF)</td>
                <td className="py-2 pr-4">Solvency II adaptado. Nota técnica requerida, sin aprobación previa de tarifas.</td>
                <td className="py-2">Sin regulación específica de AI. Ley general esperada 2026.</td>
              </tr>
              <tr className="border-b border-border/50 dark:border-[#2a2a2a]/50">
                <td className="py-2 pr-4 text-ink dark:text-[#F0EFEB]">Unión Europea</td>
                <td className="py-2 pr-4">AI Act (2024): seguros clasificados como alto riesgo.</td>
                <td className="py-2">Requiere explicabilidad, auditoría de sesgo, supervisión humana.</td>
              </tr>
              <tr className="border-b border-border/50 dark:border-[#2a2a2a]/50">
                <td className="py-2 pr-4 text-ink dark:text-[#F0EFEB]">EE.UU. (NAIC)</td>
                <td className="py-2 pr-4">FACTS Model Law. Colorado AI Act (Feb 2026).</td>
                <td className="py-2">Estado por estado. Creciente escrutinio de discriminación indirecta.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </motion.section>

      {/* Metodología */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.4 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          Metodología
        </h2>
        <div className="space-y-3 max-w-2xl font-sans text-sm text-muted leading-relaxed">
          <p>
            <span className="text-ink dark:text-[#F0EFEB] font-medium">1. Descomposición frecuencia-severidad:</span>{' '}
            Estándar actuarial. Prima = frecuencia esperada x costo esperado. Modelamos cada
            componente por separado con distribuciones apropiadas (Poisson, Gamma).
          </p>
          <p>
            <span className="text-ink dark:text-[#F0EFEB] font-medium">2. Benchmark en freMTPL2:</span>{' '}
            677,991 pólizas reales de un asegurador francés. El dataset más citado en
            pricing actuarial. Usado por scikit-learn, Noll et al., CAS E-Forum, DAV.
          </p>
          <p>
            <span className="text-ink dark:text-[#F0EFEB] font-medium">3. Comparación GLM vs GBM:</span>{' '}
            Poisson GLM (baseline regulatorio) vs XGBoost y LightGBM con tuning Optuna.
            Métricas: <Tooltip term="Gini">Capacidad de ranking: 0 = aleatorio, 1 = perfecto.</Tooltip>, error de predicción, calibración, lift.
          </p>
          <p>
            <span className="text-ink dark:text-[#F0EFEB] font-medium">4. Explicabilidad y fairness:</span>{' '}
            <Tooltip term="SHAP">Herramienta que descompone cada predicción en la contribución de cada variable.</Tooltip> para interpretar GBM. Auditoría de discriminación indirecta
            sobre Area/Densidad. Análisis de dónde falla el GLM que el GBM captura.
          </p>
        </div>
      </motion.section>

      {/* Referencias */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.5 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          Fundamento académico
        </h2>
        <div className="space-y-2 max-w-2xl">
          {[
            { authors: 'Noll, Salzmann & Wüthrich', year: 2020, title: 'Case Study: French Motor Third-Party Liability Claims', finding: 'GBM y redes neuronales superan GLM en frecuencia de siniestros' },
            { authors: 'Colella & Jones (CAS E-Forum)', year: 2023, title: 'ML and Ratemaking', finding: 'No existe modelo óptimo universal; la evaluación comparativa es clave' },
            { authors: 'MDPI Risks', year: 2024, title: 'GLM vs ML in Motor Insurance', finding: 'Modelo híbrido GLM+ANN supera a todos los modelos individuales' },
            { authors: 'Kuo & Lupton (Variance)', year: 2023, title: 'Explainability in Insurance Pricing', finding: 'SHAP + gráficas de efecto parcial necesarios para revisión regulatoria' },
          ].map((ref) => (
            <div key={ref.title} className="flex gap-3 items-start font-sans text-sm">
              <span className="text-xs text-muted shrink-0 w-10 text-right tabular-nums">{ref.year}</span>
              <div>
                <p className="text-ink dark:text-[#F0EFEB]">{ref.authors}</p>
                <p className="text-muted italic">{ref.title}</p>
                <p className="text-muted text-xs">{ref.finding}</p>
              </div>
            </div>
          ))}
        </div>
      </motion.section>
    </div>
  )
}
