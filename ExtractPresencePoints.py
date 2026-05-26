import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from shapely.geometry import Point

# 1. Load CSV and filter points with presence == 1
df = pd.read_csv("Data/pre_pseu_iso_scaled.csv")
df_pos = df[df["presence"] == 1].copy()

# 2. Load raster using rasterio
with rasterio.open("stacked_bioclim.tif") as src:
    meta = src.meta.copy()
    bands = src.count
    width, height = src.width, src.height
    transform = src.transform
    crs = src.crs

    # 3. Convert lat/lon to the raster's CRS if needed
    gdf = gpd.GeoDataFrame(
        df_pos,
        geometry=[Point(xy) for xy in zip(df_pos["Longitude"], df_pos["Latitude"])],
        crs="EPSG:4326"
    )
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    # 4. Get the (x, y) coords
    coords = [(geom.x, geom.y) for geom in gdf.geometry]

    # 5. Sample raster at these points: shape → (N_points, N_bands)
    sampled_vals = list(src.sample(coords))  # list of arrays, one per point
    sampled_vals = np.stack(sampled_vals)    # shape: (n_points, 19)

    # 6. Create empty raster (bands, height, width)
    output = np.full((bands, height, width), src.nodata if src.nodata is not None else np.nan, dtype=src.dtypes[0])

    # 7. Fill only pixels at presence==1 points
    for val, geom in zip(sampled_vals, gdf.geometry):
        row, col = src.index(geom.x, geom.y)
        output[:, row, col] = val

# 8. Write result to a new GeoTIFF
meta.update({"count": bands, "nodata": np.nan})
with rasterio.open("presence_1_points_only.tif", "w", **meta) as dst:
    dst.write(output)
