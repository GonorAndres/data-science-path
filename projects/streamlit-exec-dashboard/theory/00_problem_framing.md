# Problem Framing: Exec Dashboard

## The audience

Non-technical decision-makers. They have 30 seconds. They want:

- The number that changed.
- Whether it is good or bad.
- What is driving it.
- What they should do about it.

Anything else is noise.

## What makes a dashboard work

- **Hierarchy.** Top of page: the answer. Middle: the breakdown. Bottom: the detail.
- **Contrast.** Clear visual emphasis on what changed or what is outside the band.
- **Context.** Every number needs a baseline (target, last period, last year).
- **Brevity.** 5 to 7 charts per page, maximum.
- **Responsiveness.** Loads in under 3 seconds; interactions feel immediate.

## Common mistakes

- Too many KPIs. 20 numbers of equal weight is no answer.
- Default plotly colors. They do not convey good/bad on their own.
- No baseline. A $1M revenue number without the prior month is meaningless.
- Over-animating. Fancy transitions distract from signal.
- No text. A chart without a one-line narrative next to it is half-done.

## This dashboard's design

- Page 1: headline KPIs on tiles with sparklines and YoY deltas. The "is the business healthy" page.
- Page 2: cohorts. The "are we retaining customers" page.
- Page 3: products. The "what is selling" page.
- Page 4: channels. The "where is revenue coming from" page.
- Page 5: anomalies. The "what needs attention right now" page.

Language switch in the sidebar.

## Cross-links

- `01_storytelling_for_execs.md`
- `02_kpi_design.md`
- `03_anomaly_visual_cues.md`
- `04_dashboard_performance.md`
