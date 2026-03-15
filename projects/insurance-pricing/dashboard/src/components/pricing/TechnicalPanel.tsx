'use client'

import { motion } from 'framer-motion'
import { useMlflowSummary } from '@/hooks/usePricingAPI'
import { LoadingSpinner } from '@/components/ui/LoadingSpinner'
import { ChartContainer } from '@/components/ui/ChartContainer'

const fadeIn = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 },
}

export function TechnicalPanel() {
  const { data: mlflow, isLoading: mlLoading } = useMlflowSummary()

  return (
    <div className="py-10 space-y-12">
      {/* Architecture */}
      <motion.section {...fadeIn}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          Arquitectura del proyecto
        </h2>
        <div className="font-mono text-xs text-muted bg-surface dark:bg-[#1a1a1a] p-4 rounded max-w-2xl leading-relaxed overflow-x-auto">
          <pre>{`projects/insurance-pricing/
  src/                        # Pipeline ML
    feature_engineering.py    # Binning, encoding, log-transform
    glm_models.py             # Poisson + Gamma GLM (statsmodels)
    gbm_models.py             # XGBoost + LightGBM (Optuna + MLflow)
    shap_analysis.py          # TreeSHAP en subsample
    fairness.py               # Proxy discrimination audit
    train.py                  # Orquestador: genera JSON
  backend/                    # FastAPI (pre-computed JSON)
    pricing_backend/
      main.py                 # CORS, routers, /health
      data_loader.py          # Carga JSON al inicio
      routers/                # overview, eda, models
  dashboard/                  # Next.js 14 + Tailwind + Recharts
    src/components/pricing/   # 4 paneles + charts
    src/hooks/                # SWR hooks
scripts/utils/                # Utilidades compartidas
  data_loader.py              # freMTPL2 de OpenML
  evaluation.py               # Gini, Lorenz, lift, deviance`}</pre>
        </div>
      </motion.section>

      {/* Data flow */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.1 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          Flujo de datos
        </h2>
        <div className="flex flex-col md:flex-row items-start md:items-center gap-4 max-w-3xl">
          {[
            { step: '1', label: 'train.py', desc: 'Entrena GLM + GBM, exporta ~15 JSON' },
            { step: '2', label: 'FastAPI', desc: 'Carga JSON al inicio, sirve vía GET' },
            { step: '3', label: 'Next.js', desc: 'SWR hooks fetchean, Recharts renderiza' },
          ].map((s, i) => (
            <div key={s.step} className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-glm-blue text-white flex items-center justify-center font-serif text-sm shrink-0">
                {s.step}
              </div>
              <div>
                <p className="font-sans text-sm font-medium text-ink dark:text-[#F0EFEB]">{s.label}</p>
                <p className="font-sans text-xs text-muted">{s.desc}</p>
              </div>
              {i < 2 && <span className="hidden md:inline text-muted text-xl mx-2">&#8594;</span>}
            </div>
          ))}
        </div>
      </motion.section>

      {/* MLflow summary */}
      <ChartContainer
        title="Resumen de experimentación (MLflow)"
        subtitle="Mejores hiperparámetros encontrados por Optuna tras búsqueda bayesiana."
        insight="Los hiperparámetros reflejan la naturaleza del problema: árboles poco profundos (max_depth 4-6) y learning rates bajos (0.03-0.08) funcionan mejor para datos tabulares de seguros donde el 96% de las pólizas no tienen siniestros."
      >
        {mlLoading ? (
          <LoadingSpinner />
        ) : mlflow ? (
          <div className="grid md:grid-cols-2 gap-6 max-w-3xl">
            {Object.entries(mlflow).filter(([k]) => k.endsWith('_params')).map(([key, params]: [string, any]) => (
              <div key={key} className="border border-border dark:border-[#2a2a2a] rounded p-4">
                <p className="font-sans text-xs uppercase tracking-widest text-muted mb-3">
                  {key.replace('_params', '').replace('_', ' ').toUpperCase()}
                </p>
                <div className="space-y-1">
                  {Object.entries(params).slice(0, 8).map(([k, v]: [string, any]) => (
                    <div key={k} className="flex justify-between text-sm font-sans">
                      <span className="text-muted">{k}</span>
                      <span className="tabular-nums text-ink dark:text-[#F0EFEB]">
                        {typeof v === 'number' ? v.toFixed(v < 1 ? 4 : 2) : String(v)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : null}
      </ChartContainer>

      {/* Tech stack */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.2 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          Stack tecnológico
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-3xl">
          {[
            { cat: 'Modelado', tools: 'statsmodels, XGBoost, LightGBM' },
            { cat: 'Tuning', tools: 'Optuna (bayesiano), MLflow (tracking)' },
            { cat: 'Explicabilidad', tools: 'SHAP (TreeExplainer)' },
            { cat: 'Backend', tools: 'FastAPI + Uvicorn' },
            { cat: 'Frontend', tools: 'Next.js 14, Recharts, SWR' },
            { cat: 'Estilos', tools: 'Tailwind CSS, Framer Motion' },
            { cat: 'Datos', tools: 'pandas, scikit-learn, OpenML' },
            { cat: 'Testing', tools: 'pytest' },
          ].map((item) => (
            <div key={item.cat}>
              <p className="font-sans text-xs uppercase tracking-widest text-muted mb-1">{item.cat}</p>
              <p className="font-sans text-sm text-ink dark:text-[#F0EFEB]">{item.tools}</p>
            </div>
          ))}
        </div>
      </motion.section>

      {/* Implications for Mexico */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.3 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          Implicaciones para México
        </h2>
        <div className="space-y-4 max-w-2xl font-sans text-sm leading-relaxed">
          <div className="border-l-2 border-glm-blue pl-4">
            <p className="font-medium text-ink dark:text-[#F0EFEB] mb-1">Regulación (CNSF)</p>
            <p className="text-muted">
              México necesita un marco para ML en pricing. La nota técnica actual no requiere
              explicabilidad algorítmica. Con la Ley de AI esperada en 2026, CNSF debería
              emitir lineamientos tipo EU AI Act pero adaptados al contexto mexicano.
            </p>
          </div>
          <div className="border-l-2 border-gbm-green pl-4">
            <p className="font-medium text-ink dark:text-[#F0EFEB] mb-1">Infraestructura de datos</p>
            <p className="text-muted">
              A diferencia de Francia (datos centralizados), México no tiene una base de
              siniestros compartida. AMIS podría liderar la creación de un repositorio
              anonimizado tipo freMTPL2 mexicano.
            </p>
          </div>
          <div className="border-l-2 border-lgbm-teal pl-4">
            <p className="font-medium text-ink dark:text-[#F0EFEB] mb-1">Oportunidad telemática</p>
            <p className="text-muted">
              Con 96.5% de penetración móvil, México tiene la infraestructura para
              seguros basados en uso (telemática). El siguiente paso natural después
              de demostrar el valor de ML pricing con datos tradicionales.
            </p>
          </div>
          <div className="border-l-2 border-accent-amber pl-4">
            <p className="font-medium text-ink dark:text-[#F0EFEB] mb-1">Competencia y accesibilidad</p>
            <p className="text-muted">
              Primas más precisas significan precios más justos. Los buenos conductores
              pagarían menos, incentivando la compra de seguro en un mercado donde solo
              30% está asegurado. First mover advantage para quien adopte ML pricing.
            </p>
          </div>
        </div>
      </motion.section>

      {/* Academic references */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.4 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          Referencias académicas
        </h2>
        <div className="space-y-3 max-w-2xl">
          {[
            { authors: 'Noll, Salzmann & Wüthrich', year: 2020, title: 'Case Study: French Motor TPL Claims' },
            { authors: 'Colella & Jones', year: 2023, title: 'ML and Ratemaking (CAS E-Forum)' },
            { authors: 'MDPI Risks', year: 2024, title: 'GLM vs ML in Motor Insurance' },
            { authors: 'Kuo & Lupton', year: 2023, title: 'Explainability in Insurance Pricing (Variance)' },
            { authors: 'Scandinavian Actuarial Journal', year: 2024, title: 'Enhanced GBM for Insurance Claims' },
            { authors: 'arXiv', year: 2025, title: 'Explainable Boosting Machine for Insurance' },
          ].map((ref) => (
            <div key={ref.title} className="flex gap-3 items-start font-sans text-sm">
              <span className="text-xs text-muted shrink-0 w-10 text-right tabular-nums">{ref.year}</span>
              <div>
                <p className="text-ink dark:text-[#F0EFEB]">{ref.authors}</p>
                <p className="text-muted italic">{ref.title}</p>
              </div>
            </div>
          ))}
        </div>
      </motion.section>

      {/* Retrospective */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.5 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          Retrospectiva
        </h2>
        <div className="grid md:grid-cols-2 gap-6 max-w-3xl">
          <div>
            <p className="font-sans text-xs uppercase tracking-widest text-muted mb-2">Qué funcionó bien</p>
            <ul className="space-y-1 font-sans text-sm text-muted">
              <li>- Descomposición frecuencia-severidad (práctica actuarial estándar)</li>
              <li>- Optuna + MLflow para tracking sistemático de experimentos</li>
              <li>- SHAP para explicar las decisiones del GBM a audiencias no técnicas</li>
              <li>- Framing cross-border (Europa vs México) diferencia el proyecto</li>
            </ul>
          </div>
          <div>
            <p className="font-sans text-xs uppercase tracking-widest text-muted mb-2">Qué mejorar</p>
            <ul className="space-y-1 font-sans text-sm text-muted">
              <li>- Agregar CatBoost y EBM para comparación más completa</li>
              <li>- Implementar Tweedie GLM (modela prima pura directamente)</li>
              <li>- Bootstrapping para intervalos de confianza en métricas</li>
              <li>- Datos mexicanos reales (AMIS/CNSF) cuando estén disponibles</li>
            </ul>
          </div>
        </div>
      </motion.section>
    </div>
  )
}
