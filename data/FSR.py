"""
Created on Tue Feb 27 11:46:27 2024

@author: KVasilakou 
"""

import rasterio
import xarray as xr
import numpy as np

################################################################################
################################# FISH RICHNESS ################################
################################################################################

# Open the GeoTIFF file
with rasterio.open("/data/...tif") as src1:
    fishbase_data = src1.read(1)  
    fishbase_meta = src1.meta
    profile1 = src1.profile  

width = fishbase_meta['width']
height = fishbase_meta['height']
transform = fishbase_meta['transform']

# Calculate latitude and longitude values
min_x = transform[2]
max_y = transform[5]
x_res = transform[0]
y_res = transform[4]
max_x = min_x + width * x_res
min_y = max_y - (height * abs(y_res)) 
lons = np.linspace(min_x + x_res / 2, max_x - x_res / 2, width).round(6)
lats = np.linspace(max_y - abs(y_res) / 2, min_y + abs(y_res) / 2, height).round(6)

# Create DataArray for raster_data
fishbase_array = xr.DataArray(fishbase_data, dims=('latitude', 'longitude'), coords={'latitude': lats, 'longitude': lons})
fishbase_array.data[fishbase_array.data < 0] = np.nan

# Calculate new latitude values covering -90 to 90
new_lats = np.linspace(89.9583, -89.9583, 2160, dtype=np.float32)
insert_above = new_lats < lats[0]
insert_below = new_lats > lats[-1]
insert_above = new_lats > lats[0]
insert_below = new_lats < lats[-1]

above_data = np.full((sum(insert_above), width), np.nan, dtype=np.float32)
below_data = np.full((sum(insert_below), width), np.nan, dtype=np.float32)

above_array = xr.DataArray(above_data, dims=('latitude', 'longitude'), coords={'latitude': new_lats[insert_above], 'longitude': lons})
below_array = xr.DataArray(below_data, dims=('latitude', 'longitude'), coords={'latitude': new_lats[insert_below], 'longitude': lons})

fishbase_array_updated = xr.concat([above_array, fishbase_array, below_array], dim='latitude')
fishbase = xr.Dataset({'fishrichness': fishbase_array_updated})

# Calculate the scaling factor for resampling
scale_factor = 6  # 30 arc-minutes / 5 arc-minutes

fishbase_data = fishbase.coarsen(latitude=scale_factor, longitude=scale_factor, boundary='trim').max()
for lat_idx, lat in enumerate(fishbase_data.latitude):
     for lon_idx, lon in enumerate(fishbase_data.longitude):
         # Define latitude and longitude limits for the current cell
         lat_min = round(new_lats[lat_idx] - scale_factor * 0.25, 6)
         lat_max = round(new_lats[lat_idx] + scale_factor * 0.25, 6)
         lon_min = round(lon.values - scale_factor * 0.25, 6)
         lon_max = round(lon.values + scale_factor * 0.25, 6)
 
         fishbase_30arc = fishbase.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
         
latitude_data_array = fishbase_data['latitude']
rounded_latitude = latitude_data_array.round(decimals=2)
fishbase_data['latitude'] = rounded_latitude 
 
# Save aggregated data to netCDF file
resampled_nc_file = f'Fish_richness_0.5_0.nc'
fishbase_data.to_netcdf(resampled_nc_file)
print(f"Resampled data saved to {resampled_nc_file}") 

temperatures = [1.5, 2.0, 3.2]
for temp in temperatures:
    # File path for PAF datasets
    PAF_tif_file = f'/data/...PAF_no_dispersal_{temp}.tif'
    
    # Open the GeoTIFF file
    with rasterio.open(PAF_tif_file) as src2:
        PAF_data = src2.read(1) 
        
    result_data = np.where((fishbase_array.data >= 0) & (PAF_data >= 0), fishbase_array.data.astype(np.float64) - (fishbase_array.data.astype(np.float64) * PAF_data.astype(np.float64)), np.nan)
        
    # Create a mask to identify non-NaN values in the result_data
    valid_mask = ~np.isnan(result_data)
    result_data_rounded = result_data.copy()  # Make a copy to preserve the original NaN values
    result_data_rounded[valid_mask] = np.round(result_data[valid_mask]).astype(np.int64)
    
    fish_richness_array = xr.DataArray(result_data_rounded, dims=('latitude', 'longitude'), coords={'latitude': lats, 'longitude': lons})
    
    new_lats = np.linspace(89.9583, -89.9583, 2160, dtype=np.float32)
    insert_above = new_lats < lats[0]
    insert_below = new_lats > lats[-1]
    insert_above = new_lats > lats[0]
    insert_below = new_lats < lats[-1]
    
    above_data = np.full((sum(insert_above), width), np.nan, dtype=np.float32)
    below_data = np.full((sum(insert_below), width), np.nan, dtype=np.float32)
    above_array = xr.DataArray(above_data, dims=('latitude', 'longitude'), coords={'latitude': new_lats[insert_above], 'longitude': lons})
    below_array = xr.DataArray(below_data, dims=('latitude', 'longitude'), coords={'latitude': new_lats[insert_below], 'longitude': lons})
    
    # Concatenate the original array with the NaN-filled rows
    fish_richness_array_updated = xr.concat([above_array, fish_richness_array, below_array], dim='latitude')
    fishrichness = xr.Dataset({'fishrichness': fish_richness_array_updated})

    fishrichness_data = fishrichness.coarsen(latitude=scale_factor, longitude=scale_factor, boundary='trim').max()
    for lat_idx, lat in enumerate(fishrichness_data.latitude):
        for lon_idx, lon in enumerate(fishrichness_data.longitude):
            
            lat_min = round(new_lats[lat_idx] - scale_factor * 0.25, 6)
            lat_max = round(new_lats[lat_idx] + scale_factor * 0.25, 6)
            lon_min = round(lon.values - scale_factor * 0.25, 6)
            lon_max = round(lon.values + scale_factor * 0.25, 6)
    
            # Select values from result_data_rounded within the limits
            fishrichness_30arc = fishrichness.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
            
    latitude_data_array = fishrichness_data['latitude']
    rounded_latitude = latitude_data_array.round(decimals=2)
    fishrichness_data['latitude'] = rounded_latitude 
    
    # Save aggregated data to netCDF file
    new_file = f'Fish_richness_{temp}.nc'
    fishrichness_data.to_netcdf(new_file)
    print(f"FSR data saved to {new_file}")
