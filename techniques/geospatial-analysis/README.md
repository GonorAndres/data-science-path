# Geospatial Analysis

## What it is

Geospatial analysis is the collection of techniques for working with data that has a geographic (location) component -- coordinates, addresses, ZIP codes, counties, territories, or any spatial reference. In practice, it means loading, transforming, visualizing, and modeling data on maps. For a data scientist, this involves working with geographic data structures (points, lines, polygons), performing spatial operations (joins, buffers, intersections), creating map visualizations (choropleths, point maps), and measuring spatial relationships (autocorrelation, clustering). The core Python library for this is **GeoPandas**, which extends pandas DataFrames with geometry columns and spatial operations.

## Why it matters

Insurance is fundamentally a geographic business. Location is one of the strongest predictors of risk:

- **Property insurance**: Flood zones, wildfire risk, hurricane exposure, crime rates, distance to fire stations -- all spatial.
- **Auto insurance**: Traffic density, road conditions, accident hotspots, urban vs. rural driving patterns.
- **Territory rating**: Actuaries group geographic areas into rating territories. Understanding spatial risk patterns is essential for fair and accurate territory definitions.
- **Catastrophe modeling**: Exposure accumulation by geography -- "How much insured value do we have within 50 miles of the coast?"
- **Market analysis**: Where are we growing? Where are competitors strong? Where are underserved markets?
- **Claims analytics**: Mapping claim density to identify fraud rings concentrated in specific areas, or to deploy adjusters effectively after a catastrophe.
- **Regulatory compliance**: Insurance regulators require that territory definitions are actuarially justified and do not unfairly discriminate.

Being able to manipulate, visualize, and model geographic data is a core skill for any actuarial data scientist.

## When to use it

- You have data tied to locations (latitude/longitude, addresses, ZIP codes, counties, states, countries) and want to analyze spatial patterns.
- You need to create **choropleth maps** showing how a metric (loss ratio, claim frequency, premium volume) varies across geographic areas.
- You want to perform **spatial joins**: linking data points to the geographic regions they fall in (e.g., which rating territory does each policy belong to?).
- You need to detect **spatial autocorrelation** -- whether nearby areas have similar risk characteristics (Moran's I).
- You are building or evaluating **territory rating plans** and need to understand spatial risk gradients.
- You want to calculate distances, buffers, or areas (e.g., "all policies within 10 miles of a coastline").
- You need to overlay multiple spatial datasets (e.g., flood zone maps + policy locations + census demographics).

## How it works (intuitive)

### Geographic data types

Geospatial data comes in two flavors:

**Vector data** (what we mostly use):
- **Points**: A single location (latitude, longitude). Example: a policy's address geocoded to coordinates.
- **Lines**: Connected sequences of points. Example: roads, rivers, boundaries.
- **Polygons**: Closed shapes defining an area. Example: ZIP code boundaries, flood zones, rating territories, county outlines.

**Raster data** (grid-based):
- A grid of cells (pixels), each with a value. Example: satellite imagery, elevation maps, weather radar.
- Less common in typical insurance analytics but used in catastrophe modeling and remote sensing.

### Coordinate Reference Systems (CRS)

This is where geospatial analysis gets tricky. The Earth is a 3D sphere (roughly), but maps are 2D. A **Coordinate Reference System** defines how 3D locations are projected onto 2D surfaces.

**Geographic CRS (unprojected)**: Uses latitude and longitude on the sphere. Example: **EPSG:4326** (WGS 84) -- this is what GPS uses and what most latitude/longitude data is in.

**Projected CRS**: Projects the sphere onto a flat surface. Example: **EPSG:3857** (Web Mercator, used by Google Maps) or **EPSG:32618** (UTM Zone 18N, good for the US East Coast).

**Why it matters**: In a geographic CRS, distances and areas are in degrees (not meters or miles). If you calculate the "distance" between two points in EPSG:4326, you get degrees, which is not physically meaningful. To compute real distances or areas, you need a projected CRS appropriate for your region.

**Rule of thumb**: Store data in EPSG:4326. When you need to compute distances or areas, project to a local CRS (UTM or state plane) for the calculation, then project back.

### Spatial operations

**Spatial join**: Link a point (policy location) to a polygon (territory) based on containment. "Which territory does this policy fall in?" This is the spatial equivalent of a merge/join in pandas, but based on geometry instead of a key column.

**Buffer**: Create a zone around a geometry. "Find all policies within 5 miles of a coastline" = buffer the coastline by 5 miles, then spatial join.

**Intersection/overlay**: Find the area where two polygons overlap. "What fraction of territory X falls in a flood zone?"

**Dissolve**: Merge multiple polygons into one based on a shared attribute. "Combine all ZIP codes that belong to territory 7 into a single polygon."

**Centroid**: Find the geometric center of a polygon. Useful for labeling maps or approximating point locations for areal data.

### Choropleth maps

A choropleth is a map where geographic areas are shaded according to a data variable. Example: a map of US counties colored by average claim severity, where dark red = high severity and light yellow = low severity.

**Steps**:
1. Get a shapefile or GeoJSON with polygon boundaries (ZIP codes, counties, territories).
2. Merge your data onto the geographic data by a common key (ZIP code, FIPS code).
3. Choose a color scale and classification method (quantiles, equal intervals, natural breaks).
4. Plot with GeoPandas, Plotly, or Folium.

### Spatial autocorrelation (Moran's I)

**Concept**: Spatial autocorrelation measures whether nearby areas tend to have similar values. Positive spatial autocorrelation: high-risk areas are next to other high-risk areas (clustering). Negative: high-risk areas are next to low-risk areas (checkerboard pattern). Zero: no spatial pattern (random).

**Moran's I** is the most common measure:

$$I = \frac{N}{\sum_{i}\sum_{j} w_{ij}} \cdot \frac{\sum_{i}\sum_{j} w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_{i}(x_i - \bar{x})^2}$$

Where:
- $N$ is the number of spatial units
- $x_i$ is the value at location $i$
- $\bar{x}$ is the mean
- $w_{ij}$ is the spatial weight between locations $i$ and $j$ (typically 1 if neighbors, 0 otherwise)

$I$ ranges from approximately -1 (perfect dispersion) through 0 (random) to +1 (perfect clustering). A statistically significant positive Moran's I means your risk variable shows genuine spatial clustering -- nearby areas have correlated risk, which has implications for territory design.

**Local Moran's I (LISA)**: Computes Moran's I for each individual location, identifying local clusters (hot spots and cold spots) and spatial outliers.

## Key concepts

### The GeoPandas ecosystem

| Library | Purpose |
|---------|---------|
| **geopandas** | Spatial DataFrames with geometry operations |
| **shapely** | Geometric objects (points, polygons) and operations |
| **fiona** | Reading/writing spatial file formats (shapefiles, GeoJSON) |
| **pyproj** | Coordinate transformations and projections |
| **folium** | Interactive Leaflet.js maps in Python |
| **plotly** | Interactive choropleths and scatter maps |
| **libpysal / PySAL** | Spatial statistics (Moran's I, spatial weights, spatial regression) |
| **contextily** | Add basemap tiles (OpenStreetMap, etc.) to static maps |
| **h3-py** | Uber's H3 hexagonal spatial indexing |

### Spatial file formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Shapefile | .shp (+.dbf, .shx, .prj) | Legacy standard. Multiple files per dataset. |
| GeoJSON | .geojson | Text-based, web-friendly, single file. Good for small-medium data. |
| GeoPackage | .gpkg | Modern replacement for shapefiles. Single file, supports multiple layers. |
| Parquet (GeoParquet) | .parquet | Fast, columnar, great for large datasets. Emerging standard. |

### Territory pricing workflow

Territory rating in insurance follows roughly these steps:

1. **Geocode policies**: Convert addresses to latitude/longitude.
2. **Spatial join**: Assign each policy to a geographic unit (ZIP code, census tract, county).
3. **Aggregate loss experience**: Compute loss ratio, frequency, severity by geographic unit.
4. **Credibility weighting**: Small areas may have too few policies for credible statistics. Apply credibility formulas or spatial smoothing.
5. **Spatial smoothing**: Use spatial methods (kriging, spatial regression, or geographic weighted averaging) to borrow strength from neighboring areas.
6. **Territory grouping**: Cluster geographic units with similar risk into territories using contiguity constraints.
7. **Validate**: Check spatial autocorrelation of residuals (if residuals are still spatially correlated, the territories are not capturing all the geographic signal).
8. **Regulatory review**: Ensure territories are actuarially justified and comply with regulations.

### Spatial weights

Spatial weights define the neighborhood structure -- which areas are "neighbors." Common definitions:

- **Queen contiguity**: Two polygons are neighbors if they share any boundary point (including corners). Named after the queen in chess.
- **Rook contiguity**: Two polygons are neighbors if they share an edge (not just a corner). Named after the rook.
- **Distance-based**: All locations within a threshold distance are neighbors.
- **K-nearest neighbors**: The K closest locations are neighbors.

## Python implementation

### Loading and basic operations with GeoPandas

```python
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load a shapefile of US counties
counties = gpd.read_file('us_counties.shp')
print(counties.crs)  # Check CRS -- probably EPSG:4326
print(counties.head())
print(counties.geometry.head())  # Shows polygon geometries

# Load policy data with lat/lon
policies = pd.read_csv('policies.csv')
policies_gdf = gpd.GeoDataFrame(
    policies,
    geometry=gpd.points_from_xy(policies.longitude, policies.latitude),
    crs='EPSG:4326'
)

# Spatial join: assign each policy to a county
policies_in_counties = gpd.sjoin(
    policies_gdf, counties, how='left', predicate='within'
)

# Compute loss ratio by county
county_stats = policies_in_counties.groupby('county_fips').agg(
    total_losses=('incurred_loss', 'sum'),
    total_premium=('earned_premium', 'sum'),
    policy_count=('policy_id', 'count')
).reset_index()
county_stats['loss_ratio'] = county_stats['total_losses'] / county_stats['total_premium']

# Merge back to county geometries for mapping
counties_with_stats = counties.merge(county_stats, on='county_fips')
```

### Choropleth maps

```python
# Static choropleth with GeoPandas + matplotlib
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
counties_with_stats.plot(
    column='loss_ratio',
    cmap='RdYlGn_r',        # Red = high loss ratio, Green = low
    scheme='quantiles',       # Classify by quantiles
    k=5,                      # 5 classes
    legend=True,
    legend_kwds={'title': 'Loss Ratio'},
    edgecolor='gray',
    linewidth=0.2,
    ax=ax
)
ax.set_title('Loss Ratio by County')
ax.set_axis_off()
plt.tight_layout()
plt.show()

# Interactive choropleth with Plotly
import plotly.express as px

fig = px.choropleth(
    county_stats,
    geojson=counties.__geo_interface__,
    locations='county_fips',
    featureidkey='properties.county_fips',
    color='loss_ratio',
    color_continuous_scale='RdYlGn_r',
    title='Loss Ratio by County'
)
fig.update_geos(fitbounds='locations', visible=False)
fig.show()
```

### Distance calculations

```python
# IMPORTANT: project to a metric CRS before computing distances
policies_projected = policies_gdf.to_crs(epsg=3857)  # Web Mercator (meters)

# Distance from each policy to the nearest coastline point
coastline = gpd.read_file('coastline.shp').to_crs(epsg=3857)
policies_projected['dist_to_coast_m'] = policies_projected.geometry.apply(
    lambda point: coastline.distance(point).min()
)
policies_projected['dist_to_coast_miles'] = policies_projected['dist_to_coast_m'] / 1609.34

# Buffer: find policies within 10 miles of coast
coast_buffer = coastline.buffer(10 * 1609.34)  # 10 miles in meters
coastal_policies = gpd.sjoin(policies_projected,
                              gpd.GeoDataFrame(geometry=coast_buffer),
                              predicate='within')
```

### Spatial autocorrelation with PySAL

```python
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local

# Build spatial weights (queen contiguity)
w = Queen.from_dataframe(counties_with_stats)
w.transform = 'r'  # Row-standardize

# Global Moran's I
moran = Moran(counties_with_stats['loss_ratio'], w)
print(f"Moran's I: {moran.I:.4f}")
print(f"P-value: {moran.p_sim:.4f}")
# Significant positive I = spatial clustering of loss ratios

# Local Moran's I (LISA) for hot spot detection
lisa = Moran_Local(counties_with_stats['loss_ratio'], w)
counties_with_stats['lisa_cluster'] = lisa.q  # Quadrant (HH, HL, LH, LL)
counties_with_stats['lisa_significant'] = lisa.p_sim < 0.05

# Map the hot spots
sig = counties_with_stats[counties_with_stats['lisa_significant']]
fig, ax = plt.subplots(figsize=(15, 10))
counties_with_stats.plot(ax=ax, color='lightgray', edgecolor='white')
sig.plot(column='lisa_cluster', categorical=True, ax=ax, legend=True,
         cmap='RdBu', edgecolor='black', linewidth=0.5)
ax.set_title('LISA Clusters: Loss Ratio Hot Spots and Cold Spots')
ax.set_axis_off()
plt.show()
```

### Geocoding

```python
# Geocode addresses to lat/lon using geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

geolocator = Nominatim(user_agent="insurance_analysis")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# For production use, prefer commercial geocoders (Google, HERE, Mapbox)
# They are faster, more accurate, and handle more address formats
```

## R implementation (if applicable)

R has excellent geospatial support through `sf`, `tmap`, and `spdep`:

```r
library(sf)
library(tmap)
library(spdep)

# Load shapefile
counties <- st_read("us_counties.shp")

# Spatial join
policies_sf <- st_as_sf(policies, coords = c("longitude", "latitude"), crs = 4326)
joined <- st_join(policies_sf, counties)

# Choropleth
tm_shape(counties_with_stats) +
  tm_polygons("loss_ratio", palette = "RdYlGn", style = "quantile") +
  tm_layout(title = "Loss Ratio by County")

# Moran's I
nb <- poly2nb(counties_with_stats, queen = TRUE)
lw <- nb2listw(nb, style = "W")
moran.test(counties_with_stats$loss_ratio, lw)
```

Python and R are both strong for geospatial work. R's `sf` + `tmap` combination is arguably more elegant for static cartography, while Python's GeoPandas + Plotly/Folium ecosystem offers better interactive visualization and integration with ML pipelines.

## Gotchas & lessons learned

### CRS mismatches are the #1 source of bugs
If you join two datasets with different CRS, the geometries will not align. Always check CRS with `.crs` and transform with `.to_crs()` before any spatial operation. This will cause silent errors -- spatial joins will return no matches, distances will be wrong, and maps will look empty.

### Never compute distances in EPSG:4326
Distances in a geographic CRS are in degrees, not meters. A degree of longitude at the equator is ~111 km, but at 45 degrees latitude it is ~78 km. Always project to a local metric CRS (UTM zone, state plane) before computing distances, areas, or buffers.

### Shapefile limitations
Shapefiles truncate column names to 10 characters, cannot store null values properly, require multiple files, and have a 2 GB size limit. Use GeoPackage or GeoParquet for new projects.

### Ecological fallacy
Aggregating data to geographic areas (counties, ZIP codes) and then analyzing the aggregates can lead to the ecological fallacy -- relationships observed at the aggregate level may not hold at the individual level. A county with a high average loss ratio does not mean every policyholder there is high-risk.

### Modifiable Areal Unit Problem (MAUP)
The results of spatial analysis can change depending on how you define the geographic units. The same data analyzed at the ZIP code level vs. the county level can produce different patterns and conclusions. Be aware of this and, when possible, analyze at the finest geographic resolution available.

### Large spatial datasets are slow
GeoPandas operations on millions of geometries can be very slow. Strategies:
- Use spatial indexes (R-tree): `gdf.sindex`
- Use GeoParquet for fast I/O
- Consider DuckDB with spatial extensions for large-scale processing
- Simplify geometries when high precision is not needed

### Geocoding quality matters
Address geocoding is imperfect. Rural addresses, PO boxes, and non-standard formats often geocode to the wrong location or fail entirely. Always check your geocoding hit rate and handle failures (fallback to ZIP centroid, flag for manual review).

### Map projections distort something
Every map projection distorts area, shape, distance, or direction (or some combination). Mercator preserves shape but wildly distorts area (Greenland looks as big as Africa). For insurance territory maps showing area-based metrics, use an equal-area projection.

## Resources

- [GeoPandas documentation](https://geopandas.org/) -- Core library for geospatial analysis in Python
- [PySAL documentation](https://pysal.org/) -- Spatial statistics and spatial econometrics
- [Plotly choropleth maps](https://plotly.com/python/choropleth-maps/) -- Interactive choropleths
- [Folium documentation](https://python-visualization.github.io/folium/) -- Interactive Leaflet maps
- [Geographic Data Science with Python (free book)](https://geographicdata.science/book/) -- Excellent modern textbook
- [CRS Explorer](https://epsg.io/) -- Look up EPSG codes and CRS definitions
- [US Census TIGER/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) -- Free boundary files for US geographies
- [Natural Earth](https://www.naturalearthdata.com/) -- Free global boundary and feature data
- [CAS Monograph on Territory Rating](https://www.casact.org/) -- Actuarial perspective on geographic rating
- [Anselin (1995) - Local Indicators of Spatial Association](https://doi.org/10.1111/j.1538-4632.1995.tb00338.x) -- The LISA paper
