"""
SolarSense — Solar Panel Performance Anomaly Detection
Streamlit dashboard

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py

Expects solar_generation.csv (raw) in the same folder. It recomputes the
weather-normalized baseline, anomaly flags, fault classification, and
7-day forecast live, so changing the sliders in the sidebar re-runs the
whole pipeline — no separate notebook step required.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# ----------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------
st.set_page_config(page_title="SolarSense", page_icon="🔆", layout="wide")

st.markdown(
    """
    <style>
    .metric-card {background-color:#f5f7fa; padding:14px 18px; border-radius:10px;
                  border:1px solid #e3e7ee;}
    .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🔆 SolarSense")
st.caption("Weather-normalized solar performance monitoring — catch underperforming sites before the electricity bill does.")


# ----------------------------------------------------------------------
# Data loading (raw generation + weather). Falls back to synthetic
# generation if no file is provided, so the app runs standalone too.
# ----------------------------------------------------------------------
@st.cache_data
def generate_synthetic(seed: int = 42, n_days: int = 365) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sites_cfg = {
        "Site_A": {"capacity_kw": 50, "fault": None},
        "Site_B": {"capacity_kw": 80, "fault": "gradual_soiling"},
        "Site_C": {"capacity_kw": 65, "fault": "sudden_inverter"},
        "Site_D": {"capacity_kw": 40, "fault": None},
        "Site_E": {"capacity_kw": 100, "fault": "partial_shading"},
    }
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    day = np.arange(n_days)
    seasonal = 5.5 + 3.2 * np.sin(2 * np.pi * (day - 80) / 365)
    cloud = np.clip(rng.beta(2, 4, n_days), 0, 1)
    irr = np.clip(seasonal * (1 - 0.75 * cloud) + rng.normal(0, 0.25, n_days), 0.2, None)
    temp = 22 + 10 * np.sin(2 * np.pi * (day - 60) / 365) + rng.normal(0, 2, n_days)

    rows = []
    for site, cfg in sites_cfg.items():
        cap = cfg["capacity_kw"]
        age_derate = 1 - 0.005 * (day / 365)
        temp_derate = 1 - 0.004 * np.clip(temp - 25, 0, None)
        base = cap * irr * 0.85 * age_derate * temp_derate
        gen = np.clip(base + rng.normal(0, base * 0.03), 0, None)

        mult = np.ones(n_days)
        fault = cfg["fault"]
        if fault == "gradual_soiling":
            start = 200
            mult -= np.clip((day - start) / 90, 0, 1) * 0.30
        elif fault == "sudden_inverter":
            mult[260:278] *= 0.35
        elif fault == "partial_shading":
            mult[150:] *= 0.88
        gen = gen * mult

        for i in range(n_days):
            rows.append({
                "date": dates[i], "site": site, "capacity_kw": cap,
                "irradiance": irr[i], "cloud_cover": cloud[i], "temperature": temp[i],
                "generation_kwh": gen[i],
            })
    return pd.DataFrame(rows)


@st.cache_data
def load_raw(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=["date"])
        return df
    try:
        return pd.read_csv("solar_generation.csv", parse_dates=["date"])
    except FileNotFoundError:
        return generate_synthetic()


# ----------------------------------------------------------------------
# Pipeline: baseline model -> expected vs actual -> anomaly flags -> classify
# ----------------------------------------------------------------------
@st.cache_data
def run_pipeline(df: pd.DataFrame, baseline_days: int, gap_threshold: float,
                  min_consec_days: int, rate_per_kwh: float):
    sites = sorted(df.site.unique())
    models, scored_parts = {}, []

    for site in sites:
        sub = df[df.site == site].sort_values("date").reset_index(drop=True)
        train = sub.iloc[:baseline_days]
        X_train = train[["irradiance", "temperature", "cloud_cover"]]
        y_train = train["generation_kwh"]
        model = LinearRegression().fit(X_train, y_train)
        models[site] = model

        X_full = sub[["irradiance", "temperature", "cloud_cover"]]
        expected = model.predict(X_full)
        actual = sub["generation_kwh"].values
        residual = actual - expected
        pct_gap = np.where(expected > 0.5, residual / expected * 100, 0)

        g = sub[["date", "site"]].copy()
        g["expected_kwh"] = expected
        g["actual_kwh"] = actual
        g["residual_kwh"] = residual
        g["pct_gap"] = pct_gap
        g["rolling_gap_7d"] = g["pct_gap"].rolling(7, min_periods=3).mean()

        below = g["rolling_gap_7d"] < gap_threshold
        streak = below.groupby((~below).cumsum()).cumcount() + 1
        g["is_anomaly"] = below & (streak >= min_consec_days)
        scored_parts.append(g)

    scored = pd.concat(scored_parts, ignore_index=True)

    # classify pattern per site
    fault_type = {}
    for site in sites:
        g = scored[scored.site == site].sort_values("date").reset_index(drop=True)
        flagged = g[g.is_anomaly]
        if flagged.empty:
            fault_type[site] = "none"
            continue
        first_idx = flagged.index[0]
        window_start = max(0, first_idx - 5)
        drop_5d = g["rolling_gap_7d"].iloc[first_idx] - g["rolling_gap_7d"].iloc[window_start]
        span_days = (flagged["date"].max() - flagged["date"].min()).days
        if drop_5d < -10:
            fault_type[site] = "sudden_drop"
        elif span_days > 25:
            fault_type[site] = "gradual_decline"
        else:
            fault_type[site] = "moderate_underperformance"

    # site summary
    summary_rows = []
    for site in sites:
        g = scored[scored.site == site]
        shortfall = -g.loc[g.is_anomaly, "residual_kwh"].sum()
        summary_rows.append({
            "site": site,
            "flagged_days": int(g.is_anomaly.sum()),
            "total_shortfall_kwh": float(shortfall),
            "estimated_loss": float(shortfall) * rate_per_kwh,
            "detected_pattern": fault_type[site],
        })
    site_summary = (pd.DataFrame(summary_rows)
                     .sort_values("estimated_loss", ascending=False)
                     .reset_index(drop=True))
    site_summary["priority_rank"] = site_summary.index + 1

    # 7-day forward forecast using the last 7 days' actual weather as a stand-in
    # for a forecast feed
    forecast_rows = []
    for site in sites:
        sub = scored[scored.site == site].tail(7)
        weather_next7 = df[df.site == site].sort_values("date").tail(7)[
            ["irradiance", "temperature", "cloud_cover"]]
        forecast_expected = models[site].predict(weather_next7)
        forecast_rows.append({
            "site": site,
            "forecast_expected_avg_kwh": float(forecast_expected.mean()),
            "recent_actual_avg_kwh": float(sub["actual_kwh"].mean()),
            "recent_pct_gap": float(sub["pct_gap"].mean()),
            "currently_flagged": bool(sub["is_anomaly"].iloc[-1]) if len(sub) else False,
        })
    forecast_df = pd.DataFrame(forecast_rows).sort_values("recent_pct_gap")

    return scored, site_summary, forecast_df


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload generation CSV (optional)", type=["csv"])
    st.caption("Columns needed: date, site, irradiance, temperature, cloud_cover, generation_kwh")

    st.divider()
    baseline_days = st.slider("Baseline (healthy) period — days", 30, 180, 90, step=10)
    gap_threshold = st.slider("Anomaly threshold — % gap vs. expected", -30, -5, -12, step=1)
    min_consec_days = st.slider("Minimum consecutive days to flag", 1, 10, 3)
    rate_per_kwh = st.number_input("Tariff rate (currency/kWh)", min_value=0.0, value=6.0, step=0.5)

raw_df = load_raw(uploaded)
scored, site_summary, forecast_df = run_pipeline(
    raw_df, baseline_days, gap_threshold, min_consec_days, rate_per_kwh
)
sites = sorted(raw_df.site.unique())


# ----------------------------------------------------------------------
# Top KPI row
# ----------------------------------------------------------------------
total_flagged = int(site_summary["flagged_days"].sum())
total_loss = site_summary["estimated_loss"].sum()
sites_at_risk = int((site_summary["flagged_days"] > 0).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sites monitored", len(sites))
c2.metric("Sites currently flagged", sites_at_risk)
c3.metric("Total flagged site-days", total_flagged)
c4.metric("Estimated total loss", f"{total_loss:,.0f}")

st.divider()

tab_dash, tab_site, tab_forecast, tab_data = st.tabs(
    ["📊 Priority Dashboard", "🔍 Site Detail", "🔮 7-Day Forecast", "📄 Raw Data"]
)

# ----------------------------------------------------------------------
# Tab 1: Priority dashboard
# ----------------------------------------------------------------------
with tab_dash:
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Maintenance priority ranking")
        display_summary = site_summary.copy()
        display_summary["estimated_loss"] = display_summary["estimated_loss"].round(0)
        display_summary["total_shortfall_kwh"] = display_summary["total_shortfall_kwh"].round(1)
        st.dataframe(
            display_summary[["priority_rank", "site", "detected_pattern",
                              "flagged_days", "total_shortfall_kwh", "estimated_loss"]],
            hide_index=True, use_container_width=True,
        )

    with right:
        st.subheader("Estimated loss by site")
        colors = ["#d62728" if p != "none" else "#2ca02c" for p in site_summary["detected_pattern"]]
        fig = go.Figure(go.Bar(
            x=site_summary["estimated_loss"], y=site_summary["site"],
            orientation="h", marker_color=colors,
        ))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                           xaxis_title="Estimated loss", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent 7-day performance gap by site")
    colors2 = ["#d62728" if v < gap_threshold else "#2ca02c" for v in forecast_df["recent_pct_gap"]]
    fig2 = go.Figure(go.Bar(
        x=forecast_df["recent_pct_gap"], y=forecast_df["site"],
        orientation="h", marker_color=colors2,
    ))
    fig2.add_vline(x=gap_threshold, line_dash="dash", line_color="gray",
                    annotation_text="Flag threshold")
    fig2.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="% gap vs. expected", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig2, use_container_width=True)

# ----------------------------------------------------------------------
# Tab 2: Per-site detail — expected vs actual, with flagged points
# ----------------------------------------------------------------------
with tab_site:
    site_choice = st.selectbox("Choose a site", sites)
    sub = scored[scored.site == site_choice].sort_values("date")
    pattern = site_summary.loc[site_summary.site == site_choice, "detected_pattern"].iloc[0]

    badge_color = {"none": "green", "gradual_decline": "orange",
                   "sudden_drop": "red", "moderate_underperformance": "orange"}.get(pattern, "gray")
    st.markdown(f"**Detected pattern:** :{badge_color}[{pattern}]")

    fig3 = make_subplots(specs=[[{"secondary_y": False}]])
    fig3.add_trace(go.Scatter(x=sub.date, y=sub.expected_kwh, name="Expected (weather-adjusted)",
                               line=dict(color="#999999", width=1.5)))
    fig3.add_trace(go.Scatter(x=sub.date, y=sub.actual_kwh, name="Actual",
                               line=dict(color="#1f77b4", width=1.5)))
    flagged = sub[sub.is_anomaly]
    if not flagged.empty:
        fig3.add_trace(go.Scatter(x=flagged.date, y=flagged.actual_kwh, mode="markers",
                                   name="Flagged", marker=dict(color="red", size=6)))
    fig3.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                        yaxis_title="kWh/day", legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = go.Figure(go.Scatter(x=sub.date, y=sub.rolling_gap_7d, line=dict(color="#9467bd")))
    fig4.add_hline(y=gap_threshold, line_dash="dash", line_color="red",
                    annotation_text="Anomaly threshold")
    fig4.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                        yaxis_title="7-day rolling % gap")
    st.plotly_chart(fig4, use_container_width=True)

    n_flag = int(sub.is_anomaly.sum())
    st.info(f"{site_choice} has {n_flag} flagged day(s) out of {len(sub)} days monitored.")

# ----------------------------------------------------------------------
# Tab 3: Forecast
# ----------------------------------------------------------------------
with tab_forecast:
    st.subheader("Next-7-day expected generation vs. recent actual performance")
    fc = forecast_df.copy()
    fc[["forecast_expected_avg_kwh", "recent_actual_avg_kwh", "recent_pct_gap"]] = \
        fc[["forecast_expected_avg_kwh", "recent_actual_avg_kwh", "recent_pct_gap"]].round(1)
    st.dataframe(fc, hide_index=True, use_container_width=True)
    st.caption(
        "Forecast expected generation uses each site's regression model applied to the most "
        "recent 7 days of weather as a stand-in for a live forecast feed. In production, swap "
        "in a weather-forecast API for `weather_next7`."
    )

# ----------------------------------------------------------------------
# Tab 4: Raw / scored data
# ----------------------------------------------------------------------
with tab_data:
    st.subheader("Scored dataset")
    st.dataframe(scored, use_container_width=True, height=400)
    st.download_button(
        "Download scored data as CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="scored_generation.csv",
        mime="text/csv",
    )
