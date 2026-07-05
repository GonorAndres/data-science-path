# streamlit-exec-dashboard

## Resumen (ES)

**Problema:** dashboard ejecutivo sobre los marts del proyecto `sql-analytics-ecommerce`. KPIs en tarjetas, grafico de cohorte, anomalias destacadas, disenado para tomadores de decision no tecnicos.

**Metodo:** Streamlit con plotly. Lee el `olist.duckdb` generado por el pipeline SQL. Etiquetas bilingues. Despliegue en Streamlit Cloud.

**Resultado objetivo:** dashboard publico con URL, tiempo de carga inicial menor a 3 segundos, un solo clic por KPI para abrir contexto.

## Summary (EN)

**Problem:** executive dashboard on top of the marts from `sql-analytics-ecommerce`. KPI cards, cohort chart, anomaly callouts, designed for non-technical decision makers.

**Method:** Streamlit with plotly. Reads the `olist.duckdb` produced by the SQL pipeline. Bilingual labels. Deployed on Streamlit Cloud.

**Target result:** public dashboard URL, initial load under 3 seconds, one click per KPI to drill into context.

---

## Why this project

Even the best SQL marts are invisible to stakeholders without a dashboard. This project closes the loop:

- **Inputs.** Marts from `sql-analytics-ecommerce`.
- **Outputs.** Stakeholder-ready visuals, bilingual.
- **Skill signal.** Product communication, not just modeling.

## Structure

```
streamlit-exec-dashboard/
├── README.md
├── theory/
│   ├── 00_problem_framing.md
│   ├── 01_storytelling_for_execs.md
│   ├── 02_kpi_design.md
│   ├── 03_anomaly_visual_cues.md
│   └── 04_dashboard_performance.md
├── app.py                     # main Streamlit app
├── src/
│   ├── __init__.py
│   ├── data.py                # load marts from DuckDB
│   ├── charts.py              # plotly chart builders
│   └── text.py                # bilingual copy blocks
├── .streamlit/
│   └── config.toml            # theme
├── requirements.txt
└── Makefile
```

## Quickstart

```bash
# First, run the SQL pipeline
cd ../sql-analytics-ecommerce
make data && make pipeline

# Then this dashboard
cd ../streamlit-exec-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Pages

1. **Overview.** KPI cards (monthly revenue, active customers, avg order value), revenue trend, YoY delta.
2. **Cohorts.** Heatmap of cohort retention by month.
3. **Products.** Top categories, top products per category, rising stars.
4. **Channels.** LTV by payment channel, acquisition mix.
5. **Anomalies.** Auto-flagged days where revenue deviates more than 2 sigma from the rolling mean.

## Bilingual

Every label has ES and EN versions. Toggle via a language selector in the sidebar. Copy lives in `src/text.py`.

## Deployment

Streamlit Cloud: connect the repo, point to `projects/streamlit-exec-dashboard/app.py`. Free tier is enough for this demo.

## Decisions and trade-offs

| Decision | Alternative | Why |
|----------|-------------|-----|
| Streamlit | Dash, Panel, Gradio | Shortest path to a deployable dashboard; free hosting. |
| plotly | matplotlib, altair | Interactive out of the box; Streamlit integrates cleanly. |
| DuckDB | CSV, SQLite | Reads from the SQL project's output directly. |
| Bilingual toggle | Two separate apps | One deploy; switchable; showcases i18n design. |

## References

- `docs/theory/15_sql_for_data_scientists.md`
- Streamlit docs: docs.streamlit.io
- Few, *Information Dashboard Design*, 2006.
