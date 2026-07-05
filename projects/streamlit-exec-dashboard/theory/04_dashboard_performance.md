# Dashboard Performance

## The 3-second rule

If a dashboard takes more than 3 seconds to load, people stop trusting it. Performance targets:

- **Initial load.** Under 3 seconds to first meaningful paint.
- **Filter interaction.** Under 500 ms.
- **Chart zoom / hover.** Under 100 ms.

## Streamlit performance tips

### Cache expensive operations

```python
@st.cache_data
def load_data():
    return pd.read_sql(...)
```

`@st.cache_data` caches across reruns; `@st.cache_resource` for database connections and models that should persist.

### Avoid recomputing per slider change

Use `@st.cache_data(ttl=...)` with appropriate TTL. Parameterize the cache key by filter values.

### Small-data pattern

Pre-aggregate in SQL; the dashboard reads small denormalized tables. 10k rows load instantly; 10M rows do not.

### Lazy-load pages

With Streamlit multipage, only the visible page's code runs. Do not import heavy libraries at the top level of app.py if they are only used in page 3.

## Chart performance

- **Plotly.** WebGL mode (`px.scatter_3d`, `go.Scattergl`) for large datasets; canvas mode for smaller.
- **Datapoint count.** A line chart with more than 10,000 points slows browsers. Downsample with LTTB (largest-triangle-three-buckets) or aggregation.
- **Interactions.** Enable only what you need. Fancy crossfilter widgets add memory.

## Database performance

- Keep DuckDB database local to the Streamlit app (same disk). Remote DB calls over the network multiply latency.
- Materialize marts; do not compute them on each dashboard render.

## Streamlit Cloud specifics

- Free tier: 1 GB RAM per app. A multi-GB DuckDB may be too large; consider a Parquet file and lazy-loaded partitions.
- Cold starts: after 7 days of inactivity, apps sleep. First load is slower.
- Persistent storage: limited; treat the app as stateless.

## Measuring

- Streamlit has built-in timing via `st.write(time.time() - t0)`.
- Plotly provides `figure.layout.updatemenus` for filter-change timing.
- Chrome DevTools to measure client-side render.

## Cross-links

- `00_problem_framing.md`
- `streamlit.io/docs`
