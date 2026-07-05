"""Plotly chart builders."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def revenue_trend(df: pd.DataFrame, lang: str = "en") -> go.Figure:
    fig = px.line(df, x="month", y="revenue", markers=True)
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Revenue" if lang == "en" else "Ingresos",
        hovermode="x unified",
        height=360,
    )
    return fig


def retention_heatmap(df: pd.DataFrame, lang: str = "en") -> go.Figure:
    pivot = df.pivot(index="cohort_month", columns="months_since", values="retention_pct")
    fig = px.imshow(
        pivot,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(
            x="Months since signup" if lang == "en" else "Meses desde alta",
            y="Cohort" if lang == "en" else "Cohorte",
            color="Retention" if lang == "en" else "Retencion",
        ),
    )
    fig.update_layout(height=420)
    return fig


def rolling_revenue(df: pd.DataFrame, lang: str = "en") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["day"], y=df["revenue"], name="Daily", opacity=0.4))
    fig.add_trace(go.Scatter(x=df["day"], y=df["revenue_7d"], name="7-day avg"))
    fig.add_trace(go.Scatter(x=df["day"], y=df["revenue_28d"], name="28-day avg"))
    fig.update_layout(hovermode="x unified", height=360)
    return fig


def flag_anomalies(df: pd.DataFrame, window: int = 28, z_thresh: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    df["mean"] = df["revenue"].rolling(window=window, min_periods=7).mean()
    df["std"] = df["revenue"].rolling(window=window, min_periods=7).std()
    df["z"] = (df["revenue"] - df["mean"]) / df["std"]
    df["anomaly"] = df["z"].abs() > z_thresh
    return df


def anomaly_chart(df: pd.DataFrame, lang: str = "en") -> go.Figure:
    df = flag_anomalies(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["day"], y=df["revenue"], name="Daily revenue"))
    anomalies = df[df["anomaly"]]
    fig.add_trace(
        go.Scatter(
            x=anomalies["day"],
            y=anomalies["revenue"],
            mode="markers",
            name="Anomaly" if lang == "en" else "Anomalia",
            marker=dict(color="red", size=10, symbol="x"),
        )
    )
    fig.update_layout(hovermode="x unified", height=360)
    return fig
