SolarSense is a data analytics prototype that detects underperforming solar panels by comparing actual daily generation against a weather-normalized expected baseline, rather than relying on misleading raw output numbers. Using a per-site linear regression model trained on irradiance, temperature, and cloud cover, it flags sustained performance gaps, classifies whether the cause looks like a gradual issue (soiling, shading) or a sudden fault (inverter failure), forecasts the coming week, and ranks sites by estimated revenue loss so maintenance teams know exactly where to send a technician first — validated end-to-end on a synthetic 5-site, 1-year dataset where it correctly caught all three injected faults while leaving healthy sites unflagged.

1. Problem Statement
Solar installations lose 10–25% of expected output over time due to issues like panel soiling, shading, inverter faults, or wiring degradation — but most small/mid-size solar installers and commercial building owners only discover this months later when a manual inspection happens or an electricity bill looks off. There's no simple system that continuously compares expected generation (given weather conditions) against actual generation and flags the gap early.

2. Key Features & Value Proposition
Weather-adjusted performance baseline — the key differentiator; raw output alone is misleading since a cloudy week naturally lowers generation, so comparing to a weather-normalized expectation is what makes the flag meaningful
Root-cause hints: gradual decline → likely soiling/shading; sudden drop → likely inverter/wiring fault (based on the shape of the anomaly, not just its size)
Maintenance prioritization list: ranks flagged sites by estimated revenue/energy loss, so O&M teams visit the highest-impact sites first
Value prop: "Know which panels are underperforming — and roughly why — before it shows up on the electricity bill."

3. What Makes It Innovative
Most existing monitoring dashboards (from inverter manufacturers like SolarEdge/Enphase) show raw generation numbers, not a weather-normalized expected vs. actual comparison. SolarSense's edge is treating underperformance as a statistical deviation problem (like sensor drift detection in manufacturing) rather than just a data visualization problem — and translating the anomaly shape into an actionable maintenance reason.

4. Basic Business/Revenue Model
Per-site monthly subscription (scales naturally with installer's portfolio size)
Tiered by site count: flat low-cost tier for small installers (5–20 sites), volume pricing for O&M companies managing hundreds of sites
Optional add-on: automated maintenance-ticket generation for flagged sites.

5. Go-to-Market
Partner directly with solar O&M/maintenance providers — they already own the "which sites need a truck roll" problem and would pay to prioritize it.
Offer a free diagnostic on a solar installer's existing historical generation data as a demo — show them real underperforming sites they didn't know about, which sells itself.
Attend/target solar industry trade associations and installer network groups (a well-defined B2B channel, unlike consumer solar).
