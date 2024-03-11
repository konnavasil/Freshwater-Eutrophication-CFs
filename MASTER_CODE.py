# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:46:27 2024

@author: KVasilakou
"""
############################### DISCHARGE DATA ################################
import xarray as xr
import numpy as np

# Load the NetCDF file
nc_file = 'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Discharge/discharge_weekAvg_output_gfdl_rcp6p0_2090-01-07_to_2099-12-30.nc'
data = xr.open_dataset(nc_file)

# Calculate the scaling factor for resampling
scale_factor = 6  # 30 arc-minutes / 5 arc-minutes
    
# Iterate over each year
for year in range(2090, 2100):
    # Select data for the current year
    yearly_data = data.sel(time=str(year))

    # Calculate yearly average
    yearly_avg = yearly_data.mean(dim='time')
    
    # Sum over each grid cell
    aggregated_data = yearly_avg.coarsen(latitude=scale_factor, longitude=scale_factor, boundary='trim').sum()
    
    # Find indices where aggregated_data is zero
    zero_indices = np.where(aggregated_data.discharge == 0)

    # Iterate over the zero indices
    for lat_idx, lon_idx in zip(*zero_indices):
        # Get the corresponding lat and lon values
        lat = aggregated_data.latitude[lat_idx].values
        lon = aggregated_data.longitude[lon_idx].values
        
        # Define latitude and longitude limits for the current cell
        lat_min = lat - scale_factor * 0.25
        lat_max = lat + scale_factor * 0.25
        lon_min = lon - scale_factor * 0.25
        lon_max = lon + scale_factor * 0.25
        
        # Select values from yearly dataset within the limits
        yearly_values = yearly_avg.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
        
        # Check if any value is zero in yearly data within the limits
        if np.any(yearly_values.discharge == 0):
            # Assign zero to the corresponding cell in aggregated_data
            aggregated_data.discharge[lat_idx, lon_idx] = 0
        else:
            # Assign NaN to the corresponding cell in aggregated_data
            aggregated_data.discharge[lat_idx, lon_idx] = np.nan

            
    # Save the resampled data to a new NetCDF file
    resampled_nc_file = f'Discharge_0.5_{year}.nc'
    aggregated_data.to_netcdf(resampled_nc_file)
    print(f"Resampled data saved to {resampled_nc_file}")

    
#%%
############################### RIVER VOLUME #################################  
import os
import rasterio
import xarray as xr
import numpy as np

# Add path with raw data
discharge_path = 'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Discharge/RCP60/MIROC'
area_path = 'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/Land cover'

# Read Area data
with rasterio.open(os.path.join(area_path, 'Land_area_0.5.tif')) as src:
     area = src.read(1)
     

for year in range(2021, 2100):
    # Read Discharge data
    discharge_data = xr.open_dataset(os.path.join(discharge_path, f'Discharge_0.5_{year}.nc'))
    discharge = discharge_data['discharge'].values  
    
   
    # Constants
    aw = 5.01e-2  # km^-0.56 y^0.52
    bw = 0.52
    ad = 1.04e-3  # km^-0.11 y^0.37
    bd = 0.37
    sb = 2.12
    
    # Calculate Width, Depth, Length, and River_vol
    Width = aw * ((discharge * 1e-9 / 3.16887646e-8) ** bw)  # km
    Depth = ad * ((discharge * 1e-9 / 3.16887646e-8) ** bd)  # km
    Length = sb * np.sqrt(area)  # m
    
    # Pad the Length array to match the shape (360, 720)
    padded_length = np.pad(Length, ((0, 60), (0, 0)), mode='constant', constant_values=np.nan)
    
    River_vol = Width * 1000 * Depth * 1000 * padded_length  # m^3
    
    # Convert River_vol to xarray DataArray
    river_vol_da = xr.DataArray(River_vol, dims=('latitude', 'longitude'), name='river_vol')
    
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    river_vol_da['latitude'] = discharge_data.latitude
    river_vol_da['longitude'] = discharge_data.longitude
    
    # Save as NetCDF file
    river_vol_nc_file = f'Rivervol_0.5_{year}.nc'
    river_vol_da.to_netcdf(river_vol_nc_file)

    print(f"Data saved to {river_vol_nc_file}")
        
#%%

############################# ADVECTION RATES ################################
import xarray as xr
import numpy as np
import rasterio

# Open the TIF file using rasterio
with rasterio.open('C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/Lakes/HydroLAKES_Volume_0.5.tif') as src:
    # Read the image data
    tif_data = src.read(1)
    # Get the metadata (e.g., coordinate reference system)
    tif_meta = src.meta

# Create x and y coordinates corresponding to the shape of the data
y_coords = np.arange(tif_data.shape[0])
x_coords = np.arange(tif_data.shape[1])

# Convert the TIF data to xarray DataArray
lakesvol_data = xr.DataArray(tif_data, dims=('y', 'x'))

# Add additional rows filled with NaN values to create 360x720
nan_rows = np.full((60, lakesvol_data.sizes['x']), np.nan)
lakesvol_data_padded = xr.concat([lakesvol_data, xr.DataArray(nan_rows, dims=('y', 'x'))], dim='y') #m3

# Iterate over each year
for year in range(2021, 2100):
    
    # Load the NetCDF files
    discharge_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS\Eutrophication CFs/GLOBAL/FUTURE/Discharge/RCP60/MIROC/Discharge_0.5_{year}.nc'
    discharge_data = xr.open_dataset(discharge_file) #m3/s
    
    rivervol_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS\Eutrophication CFs/GLOBAL/FUTURE/River volume/RCP60/MIROC/Rivervol_0.5_{year}.nc'
    rivervol_data = xr.open_dataset(rivervol_file) #m3
    
    # Calculate the advection rate
    
    # Calculate the advection rate
    denominator = lakesvol_data_padded.values + rivervol_data.river_vol
    mask_nan_discharge = np.isnan(discharge_data.discharge)
    # Apply the conditions
    adv_rate = np.where(mask_nan_discharge, np.nan, discharge_data.discharge / denominator) #s-1
    
    # Convert adv_rate to xarray DataArray
    adv_rate_da = xr.DataArray(adv_rate, dims=('latitude', 'longitude'), name='adv_rate')
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    adv_rate_da['latitude'] = discharge_data.latitude
    adv_rate_da['longitude'] = discharge_data.longitude
    
    # Convert the advection rate data to float32
    adv_rate_da_float32 = adv_rate_da.astype(np.float32)
    
    # Save the advection rate as NetCDF file
    adv_nc_file = f'Adv_0.5_{year}.nc'
    adv_rate_da_float32.to_netcdf(adv_nc_file)
    print(f"Data saved to {adv_nc_file}")

#%%

######################## RETENTION RATES #####################################
import xarray as xr
import numpy as np
import rasterio


def read_geotiff_to_xarray(tif_file):
    """Read GeoTIFF file and convert to xarray DataArray"""
    with rasterio.open(tif_file) as src:
        # Read the image data
        tif_data = src.read(1)
        # Get the metadata (e.g., coordinate reference system)
        tif_meta = src.meta

    # Create x and y coordinates corresponding to the shape of the data
    y_coords = np.arange(tif_data.shape[0])
    x_coords = np.arange(tif_data.shape[1])

    # Convert the TIF data to xarray DataArray
    return xr.DataArray(tif_data, dims=('y', 'x'))

# File paths for volume and area datasets
volume_tif_file = 'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/Lakes/HydroLAKES_Volume_0.5.tif'
area_tif_file = 'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/Lakes/HydroLAKES_Area_0.5.tif'

# Read volume and area datasets
lakesvol_data_padded = read_geotiff_to_xarray(volume_tif_file)
lakesarea_data_padded = read_geotiff_to_xarray(area_tif_file)

# Add additional rows filled with NaN values to create 360x720
nan_rows = np.full((60, lakesvol_data_padded.sizes['x']), np.nan)
lakesvol_data_padded = xr.concat([lakesvol_data_padded, xr.DataArray(nan_rows, dims=('y', 'x'))], dim='y') #m3
lakesarea_data_padded = xr.concat([lakesarea_data_padded, xr.DataArray(nan_rows, dims=('y', 'x'))], dim='y') #m2

# Iterate over each year
for year in range(2021, 2100):
    
    # Load the NetCDF files
    discharge_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS\Eutrophication CFs/GLOBAL/FUTURE/Discharge/RCP60/MIROC/Discharge_0.5_{year}.nc'
    discharge_data = xr.open_dataset(discharge_file) #m3/s
    
    rivervol_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS\Eutrophication CFs/GLOBAL/FUTURE/River volume/RCP60/MIROC/Rivervol_0.5_{year}.nc'
    rivervol_data = xr.open_dataset(rivervol_file) #m3
    
    # Define the conditions and corresponding values
    condition_1 = discharge_data.discharge > 14.2
    condition_2 = (discharge_data.discharge <= 14.2) & (discharge_data.discharge > 2.8)
    
    value_1 = 0.012
    value_2 = 0.068
    value_3 = 0.195
    
    # Create a new DataArray based on the conditions
    kret_riv = xr.where(condition_1, value_1, xr.where(condition_2, value_2, value_3))

    #Calculate retention rate
    ret_rate = (1 / (rivervol_data.river_vol + lakesvol_data_padded.values)) * (rivervol_data.river_vol * kret_riv.values + 0.038 * lakesarea_data_padded.values)

    # Convert adv_rate to xarray DataArray
    ret_rate_da = xr.DataArray(ret_rate, dims=('latitude', 'longitude'), name='ret_rate')
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    ret_rate_da['latitude'] = discharge_data.latitude
    ret_rate_da['longitude'] = discharge_data.longitude
    
    # Convert the advection rate data to float32
    ret_rate_da_float32 = ret_rate_da.astype(np.float32)
    
    # Save the advection rate as NetCDF file
    ret_nc_file = f'Ret_0.5_{year}.nc' #d-1
    ret_rate_da_float32.to_netcdf(ret_nc_file)
    print(f"Data saved to {ret_nc_file}")

#%%

############################# RUNOFF DATA ###################################
import xarray as xr
import numpy as np

# Load the NetCDF file
nc_file = 'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Runoff/Raw data/pcr-globwb_miroc5_ewembi_rcp60_2005soc_co2_qtot_global_daily_2091_2099.nc4'
data = xr.open_dataset(nc_file)
# Conversion factor from kg/m²/s to mm/year
conversion_factor = 60 * 60 * 24 * 365  # seconds in a year

    
# Iterate over each year
for year in range(2091, 2100):
    # Select data for the current year
    yearly_data = data.sel(time=str(year))
    
    # Convert data to mm/year and calculate yearly average
    yearly_avg = yearly_data * conversion_factor

    # Calculate yearly average
    yearly_avg = yearly_avg.mean(dim='time')
        
    resampled_nc_file = f'Runoff_0.5_{year}.nc'
    yearly_avg.to_netcdf(resampled_nc_file) #mm/year

    print(f"Data saved to {resampled_nc_file}")


#%%
############################ WATER IRRIGATION ##############################

import pandas as pd
import numpy as np
import xarray as xr
from scipy.interpolate import griddata

# Read the CSV file
df = pd.read_csv("C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Irrigation water use/Raw data/RCP60/MIROC/twdirr_km3permonth.csv")

# Extract latitude and longitude values
latitudes = df["lat"]
longitudes = df["lon"]

# Define the resolution and extent
resolution = 0.5
lon_min, lon_max = -179.75, 179.75
lat_min, lat_max = -89.75, 89.75

# Create regular grid for interpolation
grid_lon, grid_lat = np.meshgrid(np.arange(lon_min, lon_max + resolution, resolution),
                                 np.arange(lat_max, lat_min - resolution, -resolution))  # Reversed latitude range

# Calculate the number of samples based on the resolution
num_lon_samples = int((lon_max - lon_min) / resolution) + 1
num_lat_samples = int((lat_max - lat_min) / resolution) + 1
# Create an array of latitudes and longitudes for the grid
lon_values = np.linspace(lon_min, lon_max, num_lon_samples) 
lat_values = np.linspace(lat_max, lat_min, num_lat_samples)  # Reversed latitude range

# Iterate over each year
for year in range(2021, 2100):
    # Filter columns for the current year
    yearly_data = df.filter(regex=str(year))
    # Exclude the last column 
    yearly_data = yearly_data.iloc[:, :-1]
    #Sum monthly values for the year
    yearly_sum = yearly_data.sum(axis=1)
    
    # Initialize an empty array to hold interpolated values with NaNs
    interpolated_values = np.full_like(grid_lon, np.nan)
    
    # Iterate over each row in the original dataframe
    for index, row in yearly_data.iterrows():
        # Find the closest grid cell for each latitude and longitude in the original dataset
        lat_idx = np.argmin(np.abs(grid_lat[:, 0] - latitudes[index]))
        lon_idx = np.argmin(np.abs(grid_lon[0, :] - longitudes[index]))
        # Assign the value from the original dataset to the corresponding grid cell
        interpolated_values[lat_idx, lon_idx] = row.dropna().sum()  # Summing non-NaN values
    
    # Create an xarray Dataset
    data = xr.Dataset(
        {
            "summed_values": (["lat", "lon"], interpolated_values),
        },
        coords={"lat": lat_values, "lon": lon_values}
    )

    # Set the coordinates' attributes
    data["lon"].attrs["units"] = "degrees_west"
    data["lat"].attrs["units"] = "degrees_south"
    data["summed_values"].attrs["units"] = "km3/y"
    
    

    # Save the data to a NetCDF file
    file_name = f"Irr_0.5_{year}.nc" #km3/y
    data.to_netcdf(file_name)
    print(f"Data for year {year} saved to {file_name}")

#%%

######################## WATER USE RATES #####################################
import xarray as xr
import numpy as np
import rasterio

def read_geotiff_to_xarray(tif_file):
    """Read GeoTIFF file and convert to xarray DataArray"""
    with rasterio.open(tif_file) as src:
        # Read the image data
        tif_data = src.read(1)
        # Get the metadata (e.g., coordinate reference system)
        tif_meta = src.meta
        

    # Create x and y coordinates corresponding to the shape of the data
    y_coords = np.arange(tif_data.shape[0])
    x_coords = np.arange(tif_data.shape[1])

    # Convert the TIF data to xarray DataArray
    return xr.DataArray(tif_data, dims=('y', 'x'))

# File paths for volume and area datasets
volume_tif_file = 'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/Lakes/HydroLAKES_Volume_0.5.tif'

# Read volume and area datasets
lakesvol_data_padded = read_geotiff_to_xarray(volume_tif_file)

# Add additional rows filled with NaN values to create 360x720
nan_rows = np.full((60, lakesvol_data_padded.sizes['x']), np.nan)
lakesvol_data_padded = xr.concat([lakesvol_data_padded, xr.DataArray(nan_rows, dims=('y', 'x'))], dim='y') #m3


# Iterate over each year
for year in range(2021, 2100):
    
    # Load the NetCDF files
    runoff_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Runoff/RCP60/MIROC/Runoff_0.5_{year}.nc'
    adv_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Advection rates/RCP60/MIROC/Adv_0.5_{year}.nc'
    irr_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Irrigation water use/RCP60/MIROC/Irr_0.5_{year}.nc'
    rivervol_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/River volume/RCP60/MIROC/Rivervol_0.5_{year}.nc'
    discharge_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Discharge/RCP60/MIROC/Discharge_0.5_{year}.nc'
    
    runoff_data = xr.open_dataset(runoff_file)['qtot'] / 1000  # Convert to m/y
    adv_data = xr.open_dataset(adv_file)['adv_rate']  # s-1
    irr_data = xr.open_dataset(irr_file)  # km3/y
    rivervol_data = xr.open_dataset(rivervol_file)['river_vol']  # m3
    discharge_data = xr.open_dataset(discharge_file)['discharge']  # m3/s
    
    # Calculate FE values
    edip = 0.29
    edop = 0.01
    adip = 0.85
    adop = 0.95
    bdip = 2.00
    
    FEdip = edip * 1 / (1 + ((runoff_data / adip) ** (-bdip)))
    FEdop = edop * (runoff_data ** adop)
    FEsoil = FEdip + FEdop
    
    # Pad FEsoil array
    #FEsoil_padded = np.pad(FEsoil, ((0, 60), (0, 0)), mode='constant', constant_values=np.nan)
    
    # Create latitude and longitude coordinates from adv_data and rename them to match the specified dimensions
    lat = adv_data['latitude'].rename({'latitude': 'lat'})
    lon = adv_data['longitude'].rename({'longitude': 'lon'})
   
    # Save the FEsoil_padded data to a DataArray with the coordinates from adv_data
    FEsoil_da = xr.DataArray(FEsoil, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})
    
    # Save the FEsoil_padded data to a NetCDF file
    #file_name_FEsoil = f"FEsoil_0.5_{year}.nc"
    #FEsoil_padded_da.to_netcdf(file_name_FEsoil)
    #print(f"FEsoil data for year {year} saved to {file_name_FEsoil}")
    
    # Calculate firr with conditions
    firr = np.where(discharge_data.data == 0,  # Condition: discharge_data.data is zero
                0,  # If True, set firr to zero
                (irr_data.summed_values * 1e9 / (365*24*3600)) / discharge_data.data)  # If False, perform original calculation
    # Set firr to NaN where discharge_data.data is NaN
    firr = np.where(np.isnan(discharge_data.data), np.nan, firr)
    # Set firr to zero where discharge_data.data is less than 1e-04
    firr = np.where(discharge_data.data < 1e-4, 0, firr)
   
    # Save the FEsoil_padded data to a DataArray with the coordinates from adv_data
    firr_da = xr.DataArray(firr, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})
    
    # Save the firr data to a NetCDF file
    #file_name_firr = f"firr_0.5_{year}.nc" 
    #firr_da.to_netcdf(file_name_firr)
    #print(f"firr data for year {year} saved to {file_name_firr}")
    
    kuse = firr_da.data * (1-FEsoil_da.data) * adv_data #s-1
    
    # Save the FEsoil_padded data to a DataArray with the coordinates from adv_data
    kuse_da = xr.DataArray(kuse, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}, name = 'use_rate')
    
    # Save the firr data to a NetCDF file
    file_name_use = f"Use_0.5_{year}.nc" 
    kuse_da.to_netcdf(file_name_use)
    print(f"Use data for year {year} saved to {file_name_use}")

#%%
#################### FISH RICHNESS PRE-PROCESSING #############################

import rasterio
import xarray as xr
import numpy as np

# Open the GeoTIFF file
with rasterio.open("C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/Future/Fish richness/Raw data/Fish_richness_base.tif") as src1:
    # Read the raster data
    fishbase_data = src1.read(1)  # Assuming single band GeoTIFF
    # Read the metadata
    fishbase_meta = src1.meta
    profile1 = src1.profile  # Get the metadata/profile

# Extract necessary metadata
width = fishbase_meta['width']
height = fishbase_meta['height']
transform = fishbase_meta['transform']

# Calculate latitude and longitude values
# Assuming the GeoTIFF is in WGS84 coordinate reference system
min_x = transform[2]
max_y = transform[5]
x_res = transform[0]
y_res = transform[4]

# Calculate the upper-left corner coordinates
max_x = min_x + width * x_res
min_y = max_y - (height * abs(y_res))  # Ensure y resolution is positive

# Calculate the center of the pixels
lons = np.linspace(min_x + x_res / 2, max_x - x_res / 2, width).round(6)
lats = np.linspace(max_y - abs(y_res) / 2, min_y + abs(y_res) / 2, height).round(6)


# Create DataArray for raster_data
fishbase_array = xr.DataArray(fishbase_data, dims=('latitude', 'longitude'), coords={'latitude': lats, 'longitude': lons})

fishbase_array.data[fishbase_array.data < 0] = np.nan



# Calculate new latitude values covering -90 to 90
new_lats = np.linspace(89.9583, -89.9583, 2160, dtype=np.float32)

# Find rows to insert above and below the original latitude range
insert_above = new_lats < lats[0]
insert_below = new_lats > lats[-1]

# Find rows to insert above and below the original latitude range
insert_above = new_lats > lats[0]
insert_below = new_lats < lats[-1]

# Create NaN-filled rows for latitude values outside the original range
above_data = np.full((sum(insert_above), width), np.nan, dtype=np.float32)
below_data = np.full((sum(insert_below), width), np.nan, dtype=np.float32)

# Create DataArrays for the above and below NaN-filled rows
above_array = xr.DataArray(above_data, dims=('latitude', 'longitude'), coords={'latitude': new_lats[insert_above], 'longitude': lons})
below_array = xr.DataArray(below_data, dims=('latitude', 'longitude'), coords={'latitude': new_lats[insert_below], 'longitude': lons})

# Concatenate the original array with the NaN-filled rows
fishbase_array_updated = xr.concat([above_array, fishbase_array, below_array], dim='latitude')
# Create Dataset with the DataArray
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
 
         # Select values from result_data_rounded within the limits
         fishbase_30arc = fishbase.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
         
 
# Accessing the latitude data array
latitude_data_array = fishbase_data['latitude']
 
# Rounding the latitude data to 2 decimals
rounded_latitude = latitude_data_array.round(decimals=2)
 
# Update the latitude data array with rounded values
fishbase_data['latitude'] = rounded_latitude 
 
# Save aggregated data to netCDF file
resampled_nc_file = f'Fish_richness_0.5_0.nc'
fishbase_data.to_netcdf(resampled_nc_file)
print(f"Resampled data saved to {resampled_nc_file}") 


temperatures = [1.5, 2.0, 3.2]

for temp in temperatures:

    # File path for PAF datasets
    PAF_tif_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/Future/Fish richness/Raw data/PAF_no_dispersal_{temp}.tif'
    # Open the GeoTIFF file
    with rasterio.open(PAF_tif_file) as src2:
        # Read the raster data
        PAF_data = src2.read(1)  # Assuming single band GeoTIFF
        
    # Apply the condition to multiply only where both data1 and data2 are >= 0
    result_data = np.where((fishbase_array.data >= 0) & (PAF_data >= 0), fishbase_array.data.astype(np.float64) - (fishbase_array.data.astype(np.float64) * PAF_data.astype(np.float64)), np.nan)
        
    # Create a mask to identify non-NaN values in the result_data
    valid_mask = ~np.isnan(result_data)
        
    # Round the non-NaN values of result_data to integers
    result_data_rounded = result_data.copy()  # Make a copy to preserve the original NaN values
    result_data_rounded[valid_mask] = np.round(result_data[valid_mask]).astype(np.int64)
    
    # Create DataArray for raster_data
    fish_richness_array = xr.DataArray(result_data_rounded, dims=('latitude', 'longitude'), coords={'latitude': lats, 'longitude': lons})
    
    # Calculate new latitude values covering -90 to 90
    new_lats = np.linspace(89.9583, -89.9583, 2160, dtype=np.float32)
    
    # Find rows to insert above and below the original latitude range
    insert_above = new_lats < lats[0]
    insert_below = new_lats > lats[-1]
    
    # Find rows to insert above and below the original latitude range
    insert_above = new_lats > lats[0]
    insert_below = new_lats < lats[-1]
    
    # Create NaN-filled rows for latitude values outside the original range
    above_data = np.full((sum(insert_above), width), np.nan, dtype=np.float32)
    below_data = np.full((sum(insert_below), width), np.nan, dtype=np.float32)
    
    # Create DataArrays for the above and below NaN-filled rows
    above_array = xr.DataArray(above_data, dims=('latitude', 'longitude'), coords={'latitude': new_lats[insert_above], 'longitude': lons})
    below_array = xr.DataArray(below_data, dims=('latitude', 'longitude'), coords={'latitude': new_lats[insert_below], 'longitude': lons})
    
    # Concatenate the original array with the NaN-filled rows
    fish_richness_array_updated = xr.concat([above_array, fish_richness_array, below_array], dim='latitude')
    # Create Dataset with the DataArray
    fishrichness = xr.Dataset({'fishrichness': fish_richness_array_updated})
    # Save to netCDF file
    #fishrichness.to_netcdf(f'Fish_richness_large_{temp}.nc') 


    fishrichness_data = fishrichness.coarsen(latitude=scale_factor, longitude=scale_factor, boundary='trim').max()
    

    for lat_idx, lat in enumerate(fishrichness_data.latitude):
        for lon_idx, lon in enumerate(fishrichness_data.longitude):
            # Define latitude and longitude limits for the current cell
            lat_min = round(new_lats[lat_idx] - scale_factor * 0.25, 6)
            lat_max = round(new_lats[lat_idx] + scale_factor * 0.25, 6)
            lon_min = round(lon.values - scale_factor * 0.25, 6)
            lon_max = round(lon.values + scale_factor * 0.25, 6)
    
            # Select values from result_data_rounded within the limits
            fishrichness_30arc = fishrichness.sel(latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max))
            
    
   # Accessing the latitude data array
    latitude_data_array = fishrichness_data['latitude']
    
    # Rounding the latitude data to 2 decimals
    rounded_latitude = latitude_data_array.round(decimals=2)
    
    # Update the latitude data array with rounded values
    fishrichness_data['latitude'] = rounded_latitude 
    
    # Save aggregated data to netCDF file
    resampled_nc_file = f'Fish_richness_0.5_{temp}.nc'
    fishrichness_data.to_netcdf(resampled_nc_file)
    print(f"Resampled data saved to {resampled_nc_file}")

#%%
###################### FISH RICHNESS DENSITY ##################################
import xarray as xr
import numpy as np
import rasterio

# Open the TIF file using rasterio
with rasterio.open('C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/Lakes/HydroLAKES_Volume_0.5.tif') as src:
    # Read the image data
    tif_data = src.read(1)
    # Get the metadata (e.g., coordinate reference system)
    tif_meta = src.meta

# Create x and y coordinates corresponding to the shape of the data
y_coords = np.arange(tif_data.shape[0])
x_coords = np.arange(tif_data.shape[1])

# Convert the TIF data to xarray DataArray
lakesvol_data = xr.DataArray(tif_data, dims=('y', 'x'))

# Add additional rows filled with NaN values to create 360x720
nan_rows = np.full((60, lakesvol_data.sizes['x']), np.nan)
lakesvol_data_padded = xr.concat([lakesvol_data, xr.DataArray(nan_rows, dims=('y', 'x'))], dim='y') #m3

rcp=26 
gcm='gfdl'
 
if rcp == 26:
     if gcm =='gfdl':
         temp = 1.5
         # Iterate over each year
         for year in range(2021, 2100):
            
            if year >= 2032:
                temp = 2.0
            if year >= 2078:
                temp = 3.2
                
            rivervol_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/River volume/RCP60/MIROC/Rivervol_0.5_{year}.nc'
            rivervol_data = xr.open_dataset(rivervol_file) #m3
            
            # Load the NetCDF files
            fishrichness_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Fish richness/Fish_richness_0.5_{temp}.nc'
            fish_richness = xr.open_dataset(fishrichness_file)
            
            
            # Calculate the FRD
            denominator = lakesvol_data_padded.values + rivervol_data.river_vol
            # Check if any values in the denominator are zero or NaN
            mask_zero_nan = np.logical_or(np.isnan(denominator), denominator < 1e-4)
            
            # Convert boolean mask to array of indices
            indices = np.where(~mask_zero_nan)
            
            # Initialize frd array with NaNs
            frd = np.full_like(denominator, np.nan)
            
            # Perform the division incrementally
            for i, j in zip(*indices):
                if not np.isnan(denominator[i, j]) and denominator[i, j] != 0:
                    frd[i, j] = fish_richness.fishrichness.values[i, j] / denominator[i, j]                    
            
            # Convert frd to xarray DataArray
            frd_da = xr.DataArray(frd, dims=('latitude', 'longitude'), name='frd')
            # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
            frd_da['latitude'] = rivervol_data.latitude
            frd_da['longitude'] = rivervol_data.longitude
            
            # Convert the advection rate data to float32
            frd_da_float32 = frd_da.astype(np.float32)
            
            # Save the advection rate as NetCDF file
            frd_nc_file = f'FRD_0.5_{year}.nc'
            frd_da_float32.to_netcdf(frd_nc_file)
            print(f"Data saved to {frd_nc_file}")

#%%
######################### Effect factor ################################
import xarray as xr
import numpy as np
import rasterio

################### Linear effect factor calculation #########################
# Open the TIF file using rasterio
with rasterio.open('C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/Climate regions/Climate_0.5.tif') as src:
    # Read the image data
    tif_data = src.read(1)
    # Get the metadata (e.g., coordinate reference system)
    tif_meta = src.meta

# Create x and y coordinates corresponding to the shape of the data
y_coords = np.arange(tif_data.shape[0])
x_coords = np.arange(tif_data.shape[1])

# Convert the TIF data to xarray DataArray
climate_data = xr.DataArray(tif_data, dims=('y', 'x'))

# 1: tropical; 2: temperate; 3: cold; 4: xeric
# Add additional rows filled with NaN values to create 360x720
nan_rows = np.full((60, climate_data.sizes['x']), np.nan)
climate_data_padded = xr.concat([climate_data, xr.DataArray(nan_rows, dims=('y', 'x'))], dim='y') 

# Define LEF values based on climate types
LEF_lake = xr.where(climate_data_padded == 1, 13457.67,
            xr.where(climate_data_padded == 2, 1253.05,
            xr.where(climate_data_padded == 3, 18279.74,
            xr.where(climate_data_padded == 4, 13457.67, np.nan))))

LEF_river = xr.where(climate_data_padded == 1, 777.98,
              xr.where(climate_data_padded == 2, 674.48,
              xr.where(climate_data_padded == 3, 647.48,
              xr.where(climate_data_padded == 4, 777.98, np.nan))))

#Define total fish richness in the world
FR_global = 11425 #LIMITATION: CANT ESTIMATE HOW IT CHANGES OVER TIME

# Open the lakes volume TIF file using rasterio
with rasterio.open('C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/Lakes/HydroLAKES_Volume_0.5.tif') as src:
    # Read the image data
    tif_data = src.read(1)
    # Get the metadata (e.g., coordinate reference system)
    tif_meta = src.meta

# Create x and y coordinates corresponding to the shape of the data
y_coords = np.arange(tif_data.shape[0])
x_coords = np.arange(tif_data.shape[1])

# Convert the TIF data to xarray DataArray
lakesvol_data = xr.DataArray(tif_data, dims=('y', 'x'))

# Add additional rows filled with NaN values to create 360x720
nan_rows = np.full((60, lakesvol_data.sizes['x']), np.nan)
lakesvol_data_padded = xr.concat([lakesvol_data, xr.DataArray(nan_rows, dims=('y', 'x'))], dim='y') #m3


# Iterate over each year
for year in range(2021, 2100): 
    # Load the NetCDF files
    FRD_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Fish richness density/RCP60/MIROC/FRD_0.5_{year}.nc'
    FRD = xr.open_dataset(FRD_file)
    
    #Calculate effect factor
    EF_lake = FRD.frd.data * LEF_lake.data / FR_global
    EF_river = FRD.frd.data * LEF_river.data / FR_global
    
    # Convert EF to xarray DataArray
    EF_lake_da = xr.DataArray(EF_lake, dims=('latitude', 'longitude'), name='EF_lake')
    EF_river_da = xr.DataArray(EF_river, dims=('latitude', 'longitude'), name='EF_river')
    
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    EF_lake_da['latitude'] = FRD.latitude
    EF_lake_da['longitude'] = FRD.longitude
    
    EF_river_da['latitude'] = FRD.latitude
    EF_river_da['longitude'] = FRD.longitude
    
    # Convert the effect rates data to float32
    EF_lake_da_float32 = EF_lake_da.astype(np.float32)
    EF_river_da_float32 = EF_river_da.astype(np.float32)
    
    # Save the effect rates as NetCDF file
    #EF_lake_nc_file = f'EF_lake_0.5_{year}.nc'
    #EF_lake_da_float32.to_netcdf(EF_lake_nc_file)
    #print(f"Data saved to {EF_lake_nc_file}")
    
    #EF_river_nc_file = f'EF_river_0.5_{year}.nc'
    #EF_river_da_float32.to_netcdf(EF_river_nc_file)
    #print(f"Data saved to {EF_river_nc_file}")

    
    #Calculate freshwater fraction by type in each cell
    #Load river volume data
    rivervol_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/River volume/RCP60/MIROC/Rivervol_0.5_{year}.nc'
    rivervol_data = xr.open_dataset(rivervol_file) #m3
    
    denominator = lakesvol_data_padded.values + rivervol_data.river_vol
    # Check if any values in the denominator are zero or NaN
    mask_zero_nan = np.logical_or(np.isnan(denominator), denominator < 1e-4)
    
    # Convert boolean mask to array of indices
    indices = np.where(~mask_zero_nan)
    
    # Initialize fraction arrays with NaNs
    fraction_lake = np.full_like(denominator, np.nan)
    fraction_river = np.full_like(denominator, np.nan)

    # Perform the division incrementally
    for i, j in zip(*indices):
        if not np.isnan(denominator[i, j]) and denominator[i, j] != 0:
            fraction_lake[i, j] = lakesvol_data_padded.values[i, j] / denominator[i, j]
            fraction_river[i, j] = 1 - fraction_lake[i, j]
                  
    # Convert frd to xarray DataArray
    fraction_lake_da = xr.DataArray(fraction_lake, dims=('latitude', 'longitude'), name='fraction_lake')
    fraction_river_da = xr.DataArray(fraction_river, dims=('latitude', 'longitude'), name='fraction_river')
    
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    fraction_lake_da['latitude'] = rivervol_data.latitude
    fraction_lake_da['longitude'] = rivervol_data.longitude
    fraction_river_da['latitude'] = rivervol_data.latitude
    fraction_river_da['longitude'] = rivervol_data.longitude
    
    # Convert the data to float32
    fraction_lake_da_float32 = fraction_lake_da.astype(np.float32)
    fraction_river_da_float32 = fraction_river_da.astype(np.float32)
    
    # Save the fractions as NetCDF files
    #fraction_lake_nc_file = f'Fraction_lake_0.5_{year}.nc'
    #fraction_lake_da_float32.to_netcdf(fraction_lake_nc_file)
   # print(f"Data saved to {fraction_lake_nc_file}")
    #fraction_river_nc_file = f'Fraction_river_0.5_{year}.nc'
    #fraction_river_da_float32.to_netcdf(fraction_river_nc_file)
    #print(f"Data saved to {fraction_river_nc_file}")
    
    #Calculate final effect factors
    effect_factor = fraction_lake * EF_lake + fraction_river * EF_river
    # Convert effect_factor to xarray DataArray
    effect_factor_da = xr.DataArray(effect_factor, dims=('latitude', 'longitude'), name='effect_factor')
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    effect_factor_da['latitude'] = rivervol_data.latitude
    effect_factor_da['longitude'] = rivervol_data.longitude
    
    # Convert the advection rate data to float32
    effect_factor_da_float32 = effect_factor_da.astype(np.float32)
    
    # Save the advection rate as NetCDF file
    EF_nc_file = f'EF_0.5_{year}.nc'
    effect_factor_da_float32.to_netcdf(EF_nc_file)
    print(f"Data saved to {EF_nc_file}")

#%%

######################### CHARACTERIZATION FACTORS ############################

import numpy as np
from osgeo import gdal
import rasterio
import xarray as xr


# Open the ASCII grid file
with rasterio.open('C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/Flow direction/g_network.asc') as src:
    # Read the image data
    tif_data = src.read(1)
    # Get the metadata (e.g., coordinate reference system)
    tif_meta = src.meta

# Create x and y coordinates corresponding to the shape of the data
y_coords = np.arange(tif_data.shape[0])
x_coords = np.arange(tif_data.shape[1])

# Convert the TIF data to xarray DataArray
FD_data = xr.DataArray(tif_data, dims=('y', 'x')) 

# Find number of rows i and columns j of the map
num_i, num_j = FD_data.shape

# Iterate over each year
for year in range(2021, 2022): 
    
    
    # Load disharge data
    discharge_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS\Eutrophication CFs/GLOBAL/FUTURE/Discharge/RCP26/GFDL/Discharge_0.5_{year}.nc'
    discharge_data = xr.open_dataset(discharge_file) #m3/s
    
    #Load removal rates
    adv_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Advection rates/RCP26/GFDL/Adv_0.5_{year}.nc'
    adv_rate = xr.open_dataset(adv_file) #s-1
    ret_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Retention rates/RCP26/GFDL/Ret_0.5_{year}.nc'
    ret_rate = xr.open_dataset(ret_file) #d-1
    use_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Water use rates/RCP26/GFDL/Use_0.5_{year}.nc'
    use_rate = xr.open_dataset(use_file) #s-1
  
    # Convert kret from days to seconds
    ret_rate.ret_rate.data /= 86400
    
    #Load effect factors
    EF_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Effect factors/RCP26/GFDL/EF_0.5_{year}.nc'
    EF = xr.open_dataset(EF_file)
    
    # Calculate persistence of P (days)
    tau = 1 / (adv_rate.adv_rate.data + ret_rate.ret_rate.data + use_rate.use_rate.data) * 0.0000115741  # conversion from sec to days
    # Convert effect_factor to xarray DataArray
    #tau_da = xr.DataArray(tau, dims=('latitude', 'longitude'), name='tau')
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    #tau_da['latitude'] = adv_rate.latitude
    #tau_da['longitude'] = adv_rate.longitude
    
    # Save the advection rate as NetCDF file
    #tau_nc_file = f'Tau_0.5_{year}.nc'
    #tau_da.to_netcdf(tau_nc_file)
    #print(f"Data saved to {tau_nc_file}")
    
    # Initialize FF and CF
    FF = np.zeros((num_i, num_j))
    CF = np.zeros((num_i, num_j))

    for i in range(num_i):
        for j in range(num_j):
            # Print i and j after each iteration
            print(f"i: {i}, j: {j}")
            if np.isnan(discharge_data.discharge.data[i, j]):
                FF[i, j] = np.nan
                CF[i, j] = np.nan
            elif np.isnan(EF.effect_factor.data[i, j]):
                FF[i, j] = 0
                CF[i, j] = 0
            else:
                current_direction = FD_data.data[i, j]
                CF_current = 0
                boundary_cell = False
                new_i = i
                new_j = j
                z = -1
                x = np.array([])
                y = np.array([])

                while current_direction != 0 and current_direction != -9999:
                    if current_direction == 1 and new_j < num_j:
                        z += 1
                        x = np.append(x, new_i)
                        y= np.append(y, new_j + 1)
                    elif current_direction == 2 and new_i < num_i and new_j < num_j:
                        z += 1
                        x = np.append(x, new_i + 1)
                        y= np.append(y, new_j + 1)
                    elif current_direction == 4 and new_i < num_i:
                        z += 1
                        x = np.append(x, new_i + 1)
                        y= np.append(y, new_j)
                    elif current_direction == 8 and new_i < num_i and new_j > 1:
                        z += 1
                        x= np.append(x, new_i + 1)
                        y= np.append(y, new_j - 1)
                    elif current_direction == 16 and new_j > 1:
                        z += 1
                        x= np.append(x, new_i)
                        y= np.append(y, new_j - 1)
                    elif current_direction == 32 and new_i > 1 and new_j > 1:
                        z += 1
                        x = np.append(x, new_i - 1)
                        y= np.append(y, new_j - 1)
                    elif current_direction == 64 and new_i > 1:
                        z += 1
                        x= np.append(x, new_i - 1)
                        y = np.append(y, new_j)
                    elif current_direction == 128 and new_i > 1 and new_j < num_j:
                        z += 1
                        x = np.append(x, new_i - 1)
                        y = np.append(y, new_j + 1)
                    else:
                        boundary_cell = True

                    if not boundary_cell:
                        if not np.isnan(EF.effect_factor.data[int(x[z]), int(y[z])]) and not np.isnan(tau[int(x[z]), int(y[z])]):
                            ff_before = 1
                            if z > 1:
                                for p in range(len(x) - 1):
                                    ff_before *= adv_rate.adv_rate.data[int(x[p]), int(y[p])] / (adv_rate.adv_rate.data[int(x[p]), int(y[p])] + ret_rate.ret_rate.data[int(x[p]), int(y[p])] + use_rate.use_rate.data[int(x[p]), int(y[p])])
                            ff = (adv_rate.adv_rate.data[i, j] / (adv_rate.adv_rate.data[i, j] + ret_rate.ret_rate.data[i, j] + use_rate.use_rate.data[i, j])) * ff_before
                            CF_current += ff * tau[int(x[z]), int(y[z])] * EF.effect_factor.data[int(x[z]), int(y[z])]
                        else:
                            CF_current = 0
                            break

                        if (current_direction / FD_data.data[int(x[z]), int(y[z])]) == 16 or (current_direction / FD_data.data[int(x[z]), int(y[z])]) == 0.0625:
                            break
                        else:
                            current_direction = FD_data.data[int(x[z]), int(y[z])]
                            new_i = int(x[z])
                            new_j = int(y[z])
                    else:
                        break

                CF[i, j] = CF_current + tau[i, j] * EF.effect_factor.data[i, j]
    
    # Convert CF from days to years
    CF /= 365
    # Convert effect_factor to xarray DataArray
    CF_da = xr.DataArray(CF, dims=('latitude', 'longitude'), name='CF')
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    CF_da['latitude'] = discharge_data.latitude
    CF_da['longitude'] = discharge_data.longitude
    
    # Save the advection rate as NetCDF file
    CF_nc_file = f'CF_0.5_{year}.nc'
    CF_da.to_netcdf(CF_nc_file)
    print(f"Data saved to {CF_nc_file}")
    
#%%
import scipy.io
import xarray as xr

# Iterate over each year
for year in range(2021, 2100): 
    # Load the .mat file
    mat_contents = scipy.io.loadmat(f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/MATLAB/CF_05_{year}.mat')
    
    # Extract the array from the dictionary
    your_array = mat_contents['CF']
    # Convert effect_factor to xarray DataArray
    CF_da = xr.DataArray(your_array, dims=('latitude', 'longitude'), name='CF')
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    CF_da['latitude'] = discharge_data.latitude
    CF_da['longitude'] = discharge_data.longitude
    
    # Save the advection rate as NetCDF file
    CF_nc_file = f'CF_0.5_{year}.nc'
    CF_da.to_netcdf(CF_nc_file)
    print(f"Data saved to {CF_nc_file}")

#%%
import xarray as xr
import numpy as np

# Define the range of years
start_year_1 = 2021
end_year_1 = 2060
start_year_2 = 2060
end_year_2 = 2099

# Create empty arrays to store data
data_1 = []
data_2 = []
gcm = 'IPSL'
# Iterate over each file
for year in range(start_year_1, end_year_1 + 1):
    filename = f"C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Characterization factors/RCP60/{gcm}/CF_0.5_{year}.nc"
    # Load the file using xarray
    ds = xr.open_dataset(filename)
    # Append the data to the array
    data_1.append(ds)
    ds.close()

for year in range(start_year_2, end_year_2 + 1):
    filename = f"C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Characterization factors/RCP60/{gcm}/CF_0.5_{year}.nc"
    # Load the file using xarray
    ds = xr.open_dataset(filename)
    # Append the data to the array
    data_2.append(ds)
    ds.close()

# Concatenate the data along the time dimension
combined_data_1 = xr.concat(data_1, dim='time')
combined_data_2 = xr.concat(data_2, dim='time')

# Calculate the average along the time dimension
average_data_1 = combined_data_1.median(dim='time')
average_data_2 = combined_data_2.median(dim='time')

# Save the resulting datasets to netCDF files
average_data_1.to_netcdf(f"RCP60_{gcm}_median_2021_to_2060.nc")
average_data_2.to_netcdf(f"RCP60_{gcm}_median_2060_to_2099.nc")
