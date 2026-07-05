"""Bilingual copy for the dashboard. All text in one place."""

COPY = {
    "title": {"en": "Executive Dashboard", "es": "Dashboard Ejecutivo"},
    "page_overview": {"en": "Overview", "es": "General"},
    "page_cohorts": {"en": "Cohorts", "es": "Cohortes"},
    "page_products": {"en": "Products", "es": "Productos"},
    "page_channels": {"en": "Channels", "es": "Canales"},
    "page_anomalies": {"en": "Anomalies", "es": "Anomalias"},
    "kpi_revenue": {"en": "Monthly Revenue", "es": "Ingresos mensuales"},
    "kpi_customers": {"en": "Active Customers", "es": "Clientes activos"},
    "kpi_aov": {"en": "Avg Order Value", "es": "Ticket promedio"},
    "kpi_retention": {"en": "28d Retention", "es": "Retencion 28d"},
    "vs_prev": {"en": "vs previous month", "es": "vs mes anterior"},
    "vs_year": {"en": "vs same month last year", "es": "vs mismo mes ano anterior"},
    "language_label": {"en": "Language", "es": "Idioma"},
}


def t(key: str, lang: str = "en") -> str:
    return COPY.get(key, {}).get(lang, key)
