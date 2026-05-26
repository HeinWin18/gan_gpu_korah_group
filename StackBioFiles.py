# pip install rioxarray xarray rasterio
from pathlib import Path
import numpy as np
import xarray as xr
import rioxarray as rxr

# # 1) List your 19 tiffs in a stable order
# files = sorted(Path("Data/bioclims/californiaclipped").glob("*.tif"))
#
# # 2) Open the first as the reference grid
# ref = rxr.open_rasterio(files[0], masked=True).squeeze("band", drop=True)
#
# # 3) Read each layer; reproject/align to the ref if needed; collect
# layers = []
# for f in files:
#     da = rxr.open_rasterio(f, masked=True).squeeze("band", drop=True)
#
#     # Move stray `_FillValue` out of attrs (or drop it)
#     if "_FillValue" in da.attrs:
#         da.encoding["_FillValue"] = da.attrs.pop("_FillValue")  # preferred
#         # or just drop it:
#         # da.attrs.pop("_FillValue", None)
#
#     if (
#         da.rio.crs != ref.rio.crs
#         or da.shape != ref.shape
#         or da.rio.transform() != ref.rio.transform()
#     ):
#         da = da.rio.reproject_match(ref)
#     layers.append(da)
#
# # 4) Stack into (band, y, x)
# stack_b_y_x = xr.concat(layers, dim="band", compat="override")
# stack_b_y_x = stack_b_y_x.assign_coords(band=np.arange(1, len(files)+1))
#
# # 5) Convert to (H, W, C) = (y, x, band)
# arr_HWC = np.transpose(stack_b_y_x.values, (1, 2, 0))   # shape: (H, W, 19)
#
# # Optional: ensure a single mask (NaN anywhere -> NaN everywhere)
# mask = np.any(np.isnan(stack_b_y_x.values), axis=0)     # (H, W)
# arr_HWC[mask] = np.nan
#
# # 6a) Save as NumPy array for ML
# np.save("bioclim_stack_HWC.npy", arr_HWC)
#
# # 6b) Or write a 19-band GeoTIFF (band-first) that preserves georeferencing
# stack_b_y_x.rio.write_nodata(np.nan, inplace=True)
# stack_b_y_x.rio.to_raster("bioclim_19band.tif")

import rioxarray as rxr
import xarray as xr
import numpy as np
from pathlib import Path

files = sorted(Path("Data/bioclims/californiaclipped").glob("*.tif"))   # 19 files
names = [f.stem for f in files]                         # channel names

# open
darrs = [rxr.open_rasterio(str(f), masked=True).squeeze("band", drop=True) for f in files]

# (optional) reproject/align everything to the first raster to guarantee identical coords
template = darrs[0]
darrs = [da.rio.reproject_match(template) if da.rio.bounds() != template.rio.bounds()
         or da.rio.resolution() != template.rio.resolution()
         else da for da in darrs]

# make sure nodata is NaN to avoid attribute conflicts (_FillValue, etc.)
darrs = [da.where(da != da.rio.nodata).astype("float32") for da in darrs]

# concatenate into channels
stack = xr.concat(darrs, dim="channel").assign_coords(channel=names)

# shape: (channel, y, x) → (y, x, channel) for H×W×C arrays
hwc = stack.transpose("y", "x", "channel")

# if you need a NumPy array for a DL model:
X = np.asarray(hwc)   # shape: (H, W, C), dtype float32

# Transpose to (C, H, W)
chw = np.moveaxis(X, -1, 0)

# Wrap into xarray DataArray
stacked_da = xr.DataArray(
    chw,
    dims=("band", "y", "x"),
    coords={"band": np.arange(1, chw.shape[0] + 1)},
    attrs={"long_name": "Stacked bioclim layers"}
)

# Attach spatial metadata from the template
stacked_da.rio.write_crs(template.rio.crs, inplace=True)
stacked_da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
stacked_da.rio.write_transform(template.rio.transform(), inplace=True)
stacked_da.rio.write_nodata(np.nan, inplace=True)

# Save to GeoTIFF
stacked_da.rio.to_raster("stacked_bioclim.tif")
