# SolarSense — Streamlit Dashboard

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens at http://localhost:8501

## What it does

- Loads `solar_generation.csv` if present in the same folder; otherwise generates
  the same synthetic 5-site dataset used in the notebook, so the app also runs
  completely standalone.
- Recomputes the whole pipeline live: weather-normalized regression baseline →
  expected-vs-actual → rolling-gap anomaly flags → gradual/sudden fault
  classification → 7-day forecast.
- Sidebar sliders let you change the baseline period, anomaly threshold, minimum
  consecutive flagged days, and tariff rate — the dashboard updates instantly.
- You can also upload your own CSV (needs columns: `date, site, irradiance,
  temperature, cloud_cover, generation_kwh`) to run the same analysis on real data.

## Tabs

1. **Priority Dashboard** — ranked sites by estimated revenue loss, plus a
   recent 7-day performance-gap view.
2. **Site Detail** — expected-vs-actual chart per site with flagged points, and
   the rolling % gap trend.
3. **7-Day Forecast** — projected expected generation vs. recent actual
   performance per site.
4. **Raw Data** — the full scored dataset, downloadable as CSV.
