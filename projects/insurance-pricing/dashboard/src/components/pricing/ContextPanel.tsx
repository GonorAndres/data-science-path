'use client'

import { motion } from 'framer-motion'

const fadeIn = {
  initial: { opacity: 0, y: 12 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 },
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
          Mexico es el unico pais de la OCDE sin seguro de auto obligatorio a nivel
          federal. Solo el 30-33% de los vehiculos estan asegurados, frente al 95%+
          en Europa. El pricing tradicional (tablas manuales, pocas variables) deja
          primas que no reflejan el riesgo real -- subsidio cruzado que encarece el
          seguro para los buenos conductores y desincentiva la compra.
        </p>
        <p className="font-sans text-base text-ink dark:text-[#F0EFEB] max-w-2xl leading-relaxed">
          La ciencia de datos actuarial esta transformando el pricing en Europa y EE.UU.
          Este proyecto explora que pueden aprender las aseguradoras mexicanas (Qualitas,
          GNP, Crabi) de esa revolucion, usando el benchmark global freMTPL2.
        </p>
      </motion.section>

      {/* Contexto del mercado */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.1 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          Contexto del mercado mexicano
        </h2>
        <div className="grid md:grid-cols-2 gap-6 max-w-3xl">
          {[
            { stat: '30-33%', desc: 'de vehiculos asegurados en Mexico (vs 95%+ en Europa)' },
            { stat: '15-20%', desc: 'de aseguradoras mexicanas usan AI/ML en pricing' },
            { stat: '33%', desc: 'cuota de mercado de Qualitas (metodos tradicionales)' },
            { stat: '68', desc: 'insurtechs en Mexico (2do en LatAm), solo 1 con licencia auto' },
            { stat: '96.5%', desc: 'penetracion movil -- oportunidad para telematica/UBI' },
            { stat: '136K+', desc: 'robos de vehiculos/ano -- riesgo clave para pricing' },
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
              GLMs (Modelos Lineales Generalizados)
            </p>
            <p className="font-sans text-sm text-muted leading-relaxed">
              Estandar global desde los 90s. Transparentes, auditables por reguladores,
              respaldados por teoria actuarial. Poisson para frecuencia, Gamma para severidad.
              Limitacion: no capturan interacciones complejas ni no-linealidades.
            </p>
          </div>
          <div className="border-l-2 border-gbm-green pl-4">
            <p className="font-sans text-sm font-medium text-gbm-green dark:text-[#5AAF7A] mb-1">
              Gradient Boosting (XGBoost, LightGBM)
            </p>
            <p className="font-sans text-sm text-muted leading-relaxed">
              Noll et al. (2020) demostraron que GBM supera a GLM en deviance sobre
              freMTPL2. Capturan interacciones automaticamente. Pero son &quot;cajas negras&quot;
              para reguladores -- SHAP y PDP son necesarios para explicabilidad.
            </p>
          </div>
          <div className="border-l-2 border-accent-amber pl-4">
            <p className="font-sans text-sm font-medium text-accent-amber mb-1">
              Enfoque hibrido (la respuesta emergente)
            </p>
            <p className="font-sans text-sm text-muted leading-relaxed">
              GBM audita los residuos del GLM: identifica donde falla el modelo tradicional.
              SHAP explica las decisiones del GBM. Fairness audits detectan discriminacion
              por proxy. MDPI (2024): GLM+ANN hibrido supera a todos los modelos individuales.
            </p>
          </div>
        </div>
      </motion.section>

      {/* Regulacion comparada */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.3 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          Regulacion comparada
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm font-sans max-w-3xl">
            <thead>
              <tr className="border-b border-border dark:border-[#2a2a2a]">
                <th className="text-left py-2 pr-4 text-xs uppercase tracking-widest text-muted">Jurisdiccion</th>
                <th className="text-left py-2 pr-4 text-xs uppercase tracking-widest text-muted">Marco</th>
                <th className="text-left py-2 text-xs uppercase tracking-widest text-muted">ML en pricing</th>
              </tr>
            </thead>
            <tbody className="text-muted">
              <tr className="border-b border-border/50 dark:border-[#2a2a2a]/50">
                <td className="py-2 pr-4 text-ink dark:text-[#F0EFEB]">Mexico (CNSF/LISF)</td>
                <td className="py-2 pr-4">Solvency II adaptado. Nota tecnica requerida, sin aprobacion previa de tarifas.</td>
                <td className="py-2">Sin regulacion especifica de AI. Ley general esperada 2026.</td>
              </tr>
              <tr className="border-b border-border/50 dark:border-[#2a2a2a]/50">
                <td className="py-2 pr-4 text-ink dark:text-[#F0EFEB]">Union Europea</td>
                <td className="py-2 pr-4">AI Act (2024): seguros clasificados como alto riesgo.</td>
                <td className="py-2">Requiere explicabilidad, auditoria de sesgo, supervision humana.</td>
              </tr>
              <tr className="border-b border-border/50 dark:border-[#2a2a2a]/50">
                <td className="py-2 pr-4 text-ink dark:text-[#F0EFEB]">EE.UU. (NAIC)</td>
                <td className="py-2 pr-4">FACTS Model Law. Colorado AI Act (Feb 2026).</td>
                <td className="py-2">Estado por estado. Creciente escrutinio de proxy discrimination.</td>
              </tr>
            </tbody>
          </table>
        </div>
      </motion.section>

      {/* Metodologia */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.4 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          Metodologia
        </h2>
        <div className="space-y-3 max-w-2xl font-sans text-sm text-muted leading-relaxed">
          <p>
            <span className="text-ink dark:text-[#F0EFEB] font-medium">1. Descomposicion frecuencia-severidad:</span>{' '}
            Estandar actuarial. Prima pura = E[frecuencia] x E[severidad]. Modelamos cada
            componente por separado con distribuciones apropiadas (Poisson, Gamma).
          </p>
          <p>
            <span className="text-ink dark:text-[#F0EFEB] font-medium">2. Benchmark en freMTPL2:</span>{' '}
            677,991 polizas reales de un asegurador frances. El dataset mas citado en
            pricing actuarial. Usado por scikit-learn, Noll et al., CAS E-Forum, DAV.
          </p>
          <p>
            <span className="text-ink dark:text-[#F0EFEB] font-medium">3. Comparacion GLM vs GBM:</span>{' '}
            Poisson GLM (baseline regulatorio) vs XGBoost y LightGBM con tuning Optuna.
            Metricas: Gini, deviance, D-squared, calibracion, lift.
          </p>
          <p>
            <span className="text-ink dark:text-[#F0EFEB] font-medium">4. Explicabilidad y fairness:</span>{' '}
            SHAP (TreeExplainer) para interpretar GBM. Auditoria de proxy discrimination
            sobre Area/Densidad. Double-lift: donde falla el GLM que el GBM captura.
          </p>
        </div>
      </motion.section>

      {/* Referencias */}
      <motion.section {...fadeIn} transition={{ ...fadeIn.transition, delay: 0.5 }}>
        <h2 className="font-serif text-2xl md:text-3xl tracking-tight mb-4">
          Fundamento academico
        </h2>
        <div className="space-y-2 max-w-2xl">
          {[
            { authors: 'Noll, Salzmann & Wuthrich', year: 2020, title: 'Case Study: French Motor Third-Party Liability Claims', finding: 'GBM y redes neuronales superan GLM en frecuencia de siniestros' },
            { authors: 'Colella & Jones (CAS E-Forum)', year: 2023, title: 'ML and Ratemaking', finding: 'No existe modelo optimo universal; la evaluacion comparativa es clave' },
            { authors: 'MDPI Risks', year: 2024, title: 'GLM vs ML in Motor Insurance', finding: 'Modelo hibrido GLM+ANN supera a todos los modelos individuales' },
            { authors: 'Kuo & Lupton (Variance)', year: 2023, title: 'Explainability in Insurance Pricing', finding: 'SHAP + PDP necesarios para satisfacer revision regulatoria' },
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
