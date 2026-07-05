"""Executive Dashboard for Olist e-commerce (bilingual)."""

import streamlit as st

from src.charts import (
    anomaly_chart,
    retention_heatmap,
    revenue_trend,
    rolling_revenue,
)
from src.data import (
    load_ltv_by_channel,
    load_monthly_revenue,
    load_product_performance,
    load_retention_curve,
    load_rolling_revenue,
)
from src.text import t


st.set_page_config(page_title="Exec Dashboard", layout="wide")


@st.cache_data
def _monthly():
    return load_monthly_revenue()


@st.cache_data
def _retention():
    return load_retention_curve()


@st.cache_data
def _rolling():
    return load_rolling_revenue()


@st.cache_data
def _products():
    return load_product_performance()


@st.cache_data
def _ltv():
    return load_ltv_by_channel()


def sidebar_controls() -> str:
    with st.sidebar:
        lang = st.radio(
            t("language_label", "en") + " / " + t("language_label", "es"),
            options=["en", "es"],
            format_func=lambda x: "English" if x == "en" else "Espanol",
            horizontal=True,
        )
    return lang


def overview_page(lang: str) -> None:
    st.title(t("title", lang))
    try:
        rev = _monthly()
        roll = _rolling()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    last = rev.iloc[-1]
    prev = rev.iloc[-2] if len(rev) >= 2 else None

    col1, col2, col3 = st.columns(3)
    col1.metric(
        t("kpi_revenue", lang),
        f"R$ {last['revenue']:,.0f}",
        delta=(f"{(last['revenue'] - prev['revenue']) / prev['revenue'] * 100:+.1f}%"
               if prev is not None else None),
    )
    col2.metric(t("kpi_customers", lang), f"{int(last['n_orders']):,}")
    col3.metric(t("kpi_aov", lang), f"R$ {last['avg_order_revenue']:,.0f}")

    st.plotly_chart(revenue_trend(rev, lang), use_container_width=True)
    st.plotly_chart(rolling_revenue(roll, lang), use_container_width=True)


def cohorts_page(lang: str) -> None:
    st.title(t("page_cohorts", lang))
    try:
        df = _retention()
    except FileNotFoundError as e:
        st.error(str(e))
        return
    st.plotly_chart(retention_heatmap(df, lang), use_container_width=True)


def products_page(lang: str) -> None:
    st.title(t("page_products", lang))
    try:
        df = _products()
    except FileNotFoundError as e:
        st.error(str(e))
        return
    st.dataframe(df, use_container_width=True)


def channels_page(lang: str) -> None:
    st.title(t("page_channels", lang))
    try:
        df = _ltv()
    except FileNotFoundError as e:
        st.error(str(e))
        return
    st.dataframe(df, use_container_width=True)


def anomalies_page(lang: str) -> None:
    st.title(t("page_anomalies", lang))
    try:
        df = _rolling()
    except FileNotFoundError as e:
        st.error(str(e))
        return
    st.plotly_chart(anomaly_chart(df, lang), use_container_width=True)


def main():
    lang = sidebar_controls()
    page = st.sidebar.radio(
        "Page",
        [
            t("page_overview", lang),
            t("page_cohorts", lang),
            t("page_products", lang),
            t("page_channels", lang),
            t("page_anomalies", lang),
        ],
    )

    pages = {
        t("page_overview", lang): overview_page,
        t("page_cohorts", lang): cohorts_page,
        t("page_products", lang): products_page,
        t("page_channels", lang): channels_page,
        t("page_anomalies", lang): anomalies_page,
    }
    pages[page](lang)


main()
