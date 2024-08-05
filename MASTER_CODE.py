"""
Created on Tue Feb 27 11:46:27 2024

@author: KVasilakou 
"""

import os
import rasterio
import xarray as xr
import numpy as np
from osgeo import gdal

###############################################################################
############################### USER INPUTS ###################################
###############################################################################

#SPECIFY CLIMATE CHANGE SCENARIO!!
#RCP = 26, 60
RCP = 26
#GCM = 'GFDL', 'HADGEM', 'IPSL', 'MIROC'
GCM = 'GFDL'


############################### RIVER VOLUME #################################  
# Add path with raw data
discharge_path = f'/data/Discharge/RCP{RCP}/{GCM}'
area_path = '/data/Land.nc'
area =  xr.open_dataset(area_path)
land_area = area['land'].data

for year in range(2021, 2100):
    # Read Discharge data
    discharge_data = xr.open_dataset(os.path.join(discharge_path, f'Discharge_{RCP}_{GCM}_{year}.nc'))
    discharge = discharge_data['dis'].values  
     
    # Constants
    aw = 5.01e-2  # km^-0.56 y^0.52
    bw = 0.52
    ad = 1.04e-3  # km^-0.11 y^0.37
    bd = 0.37
    sb = 2.12
    
    # Calculate Width, Depth, Length, and River_vol
    Width = aw * ((discharge * 1e-9 / 3.16887646e-8) ** bw)  # km
    Depth = ad * ((discharge * 1e-9 / 3.16887646e-8) ** bd)  # km
    Length = sb * np.sqrt(land_area)  # m
    
    River_vol = Width * 1000 * Depth * 1000 * Length  # m^3
    
    # Convert River_vol to xarray DataArray
    river_vol_da = xr.DataArray(River_vol, dims=('lat', 'lon'), name='river_vol')
    river_vol_da['lat'] = discharge_data['lat']
    river_vol_da['lon'] = discharge_data['lon']
    
    # Save as NetCDF file
    river_vol_nc_file = f'Rivervol_{RCP}_{GCM}_{year}.nc'
    river_vol_da.to_netcdf(river_vol_nc_file)

    print(f"Data saved to {river_vol_nc_file}")

############################# ADVECTION RATES ################################
volume_path = '/data/Lakesvol.nc'
data =  xr.open_dataset(volume_path)
lakesvol = data['lakesvol_data_padded'].data #m3

# Iterate over each year
for year in range(2021, 2100):
    
    # Load the NetCDF files
    discharge_file = f'/data/Discharge/RCP{RCP}/{GCM}/Discharge_{RCP}_{GCM}_{year}.nc'
    discharge_data = xr.open_dataset(discharge_file) #m3/s
    
    rivervol_file = f'/data/Rivervol/RCP{RCP}/{GCM}/Rivervol_{RCP}_{GCM}_{year}.nc'
    rivervol_data = xr.open_dataset(rivervol_file) #m3

    # Calculate the advection rate
    denominator = lakesvol + rivervol_data.river_vol
    mask_nan_discharge = np.isnan(discharge_data.dis)
    # Apply the conditions
    adv_rate = np.where(mask_nan_discharge, np.nan, discharge_data.dis / denominator) #s-1
    
    # Convert adv_rate to xarray DataArray
    adv_rate_da = xr.DataArray(adv_rate, dims=('lat', 'lon'), name='adv_rate')
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    adv_rate_da['lat'] = discharge_data['lat']
    adv_rate_da['lon'] = discharge_data['lon']
    
    # Convert the advection rate data to float32
    adv_rate_da_float32 = adv_rate_da.astype(np.float32)
    
    # Save the advection rate as NetCDF file
    adv_nc_file = f'Adv_{RCP}_{GCM}_{year}.nc'
    adv_rate_da_float32.to_netcdf(adv_nc_file)
    print(f"Data saved to {adv_nc_file}")

#%%

######################## RETENTION RATES #####################################


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
    condition_1 = discharge_data.dis > 14.2
    condition_2 = (discharge_data.dis <= 14.2) & (discharge_data.dis > 2.8)
    
    value_1 = 0.012
    value_2 = 0.068
    value_3 = 0.195
    
    # Create a new DataArray based on the conditions
    kret_riv = xr.where(condition_1, value_1, xr.where(condition_2, value_2, value_3))

    #Calculate retention rate
    ret_rate = (1 / (rivervol_data.river_vol + lakesvol_data_padded.values)) * (rivervol_data.river_vol * kret_riv.values + 0.038 * lakesarea_data_padded.values)

    # Convert adv_rate to xarray DataArray
    ret_rate_da = xr.DataArray(ret_rate, dims=('lat', 'lon'), name='ret_rate')
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    ret_rate_da['lat'] = discharge_data['lat']
    ret_rate_da['lon'] = discharge_data['lon']
    
    # Convert the advection rate data to float32
    ret_rate_da_float32 = ret_rate_da.astype(np.float32)
    
    # Save the advection rate as NetCDF file
    ret_nc_file = f'Ret_0.5_{year}.nc' #d-1
    ret_rate_da_float32.to_netcdf(ret_nc_file)
    print(f"Data saved to {ret_nc_file}")

#%%



#%%



#%%
######################## WATER USE RATES #####################################


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
    runoff_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Runoff/RCP26/IPSL/Runoff_0.5_{year}.nc'
    adv_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Advection rates/RCP26/IPSL/Adv_0.5_{year}.nc'
    irr_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Irrigation water use/RCP26/IPSL/Irr_0.5_{year}.nc'
    rivervol_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/River volume/RCP26/IPSL/Rivervol_0.5_{year}.nc'
    discharge_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Discharge/RCP26/IPSL/Discharge_0.5_{year}.nc'
    
    runoff_data = xr.open_dataset(runoff_file)['qtot'] / 1000  # Convert to m/y
    adv_data = xr.open_dataset(adv_file)['adv_rate']  # s-1
    irr_data = xr.open_dataset(irr_file)  # m3
    rivervol_data = xr.open_dataset(rivervol_file)['river_vol']  # m3
    discharge_data = xr.open_dataset(discharge_file)['dis']  # m3/s
    
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
    lat = adv_data['lat'].rename({'lat': 'lat'})
    lon = adv_data['lon'].rename({'lon': 'lon'})
   
    # Save the FEsoil_padded data to a DataArray with the coordinates from adv_data
    FEsoil_da = xr.DataArray(FEsoil, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon})
    
    # Save the FEsoil_padded data to a NetCDF file
    #file_name_FEsoil = f"FEsoil_0.5_{year}.nc"
    #FEsoil_padded_da.to_netcdf(file_name_FEsoil)
    #print(f"FEsoil data for year {year} saved to {file_name_FEsoil}")
    
    # Calculate firr with conditions
    firr = np.where(discharge_data.data == 0,  # Condition: discharge_data.data is zero
                0,  # If True, set firr to zero
                irr_data.airrww / (discharge_data))  # If False, perform original calculation
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
    output_folder = 'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Water use rates/RCP26/IPSL'
    file_name_use = os.path.join(output_folder,f"Use_0.5_{year}.nc") 
    kuse_da.to_netcdf(file_name_use)
    print(f"Use data for year {year} saved to {file_name_use}")


#%%


#%%
###################### FISH RICHNESS DENSITY ##################################

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
         temp = 0.0
         # Iterate over each year
         for year in range(2021, 2100):
            
            if year >= 2023:
                temp = 1.5
            if year >= 2039:
                temp = 2.0
            if year >= 2073:
                temp = 3.2
                
            rivervol_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/River volume/RCP26/GFDL/Rivervol_0.5_{year}.nc'
            rivervol_data = xr.open_dataset(rivervol_file) #m3
            
            # Load the NetCDF files
            fishrichness_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Fish richness/Fish_richness_0.5_{temp}.nc'
            fish_richness = xr.open_dataset(fishrichness_file)
            
            # Calculate the FRD
            denominator = lakesvol_data_padded.values + rivervol_data.river_vol
            # Check if any values in the denominator are zero or NaN
            mask_zero_nan = np.logical_or(np.isnan(denominator), denominator < 1e-4)
                    
                    
            frd = xr.full_like(denominator, np.nan)
            frd = frd.where(mask_zero_nan, fish_richness.fishrichness.values / denominator)  
            
            # Convert frd to xarray DataArray
            frd_da = xr.DataArray(frd, dims=('lat', 'lon'), name='frd')
            # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
            frd_da['lat'] = rivervol_data.lat
            frd_da['lon'] = rivervol_data.lon
            
            # Convert the advection rate data to float32
            frd_da_float32 = frd_da.astype(np.float32)
            
            # Save the advection rate as NetCDF file
            frd_nc_file = f'FRD_0.5_{year}.nc'
            frd_da_float32.to_netcdf(frd_nc_file)
            print(f"Data saved to {frd_nc_file}")

#%%
######################### Effect factor ################################


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
    EF_lake_da = xr.DataArray(EF_lake, dims=('lat', 'lon'), name='EF_lake')
    EF_river_da = xr.DataArray(EF_river, dims=('lat', 'lon'), name='EF_river')
    
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    EF_lake_da['lat'] = FRD.lat
    EF_lake_da['lon'] = FRD.lon
    
    EF_river_da['lat'] = FRD.lat
    EF_river_da['lon'] = FRD.lon
    
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
    
    # Initialize fraction arrays with NaNs
    fraction_lake = xr.full_like(denominator, np.nan)
    fraction_river = xr.full_like(denominator, np.nan)
    # Perform the division for non-zero and non-NaN values
    valid_indices = ~mask_zero_nan
    
    fraction_lake = fraction_lake.where(mask_zero_nan, lakesvol_data_padded.values / denominator)
    fraction_river = fraction_river.where(mask_zero_nan, 1 - fraction_lake)
                  
    # Convert frd to xarray DataArray
    fraction_lake_da = xr.DataArray(fraction_lake, dims=('lat', 'lon'), name='fraction_lake')
    fraction_river_da = xr.DataArray(fraction_river, dims=('lat', 'lon'), name='fraction_river')
    
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    fraction_lake_da['lat'] = rivervol_data.lat
    fraction_lake_da['lon'] = rivervol_data.lon
    fraction_river_da['lat'] = rivervol_data.lat
    fraction_river_da['lon'] = rivervol_data.lon
    
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
    effect_factor_da = xr.DataArray(effect_factor, dims=('lat', 'lon'), name='effect_factor')
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    effect_factor_da['lat'] = rivervol_data.lat
    effect_factor_da['lon'] = rivervol_data.lon
    
    # Convert the advection rate data to float32
    effect_factor_da_float32 = effect_factor_da.astype(np.float32)
    
    # Save the advection rate as NetCDF file
    EF_nc_file = f'EF_0.5_{year}.nc'
    effect_factor_da_float32.to_netcdf(EF_nc_file)
    print(f"Data saved to {EF_nc_file}")

#%%

######################### CHARACTERIZATION FACTORS ############################

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
for year in range(2041, 2042): 
    
    
    # Load disharge data
    discharge_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS\Eutrophication CFs/GLOBAL/FUTURE/Discharge/RCP60/MIROC/Discharge_0.5_{year}.nc'
    discharge_data = xr.open_dataset(discharge_file) #m3/s
    
    #Load removal rates
    adv_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Advection rates/RCP60/MIROC/Adv_0.5_{year}.nc'
    adv_rate = xr.open_dataset(adv_file) #s-1
    ret_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Retention rates/RCP60/MIROC/Ret_0.5_{year}.nc'
    ret_rate = xr.open_dataset(ret_file) #d-1
    use_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Water use rates/RCP60/MIROC/Use_0.5_{year}.nc'
    use_rate = xr.open_dataset(use_file) #s-1
  
    # Convert kret from days to seconds
    ret_rate.ret_rate.data /= 86400
    
    #Load effect factors
    EF_file = f'C:/Users/KVasilakou/OneDrive - Universiteit Antwerpen/PhD/GIS/Eutrophication CFs/GLOBAL/FUTURE/Effect factors/RCP60/MIROC/EF_0.5_{year}.nc'
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
