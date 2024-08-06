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


# Add path with raw data
land_path = '/data/Land.nc'
land =  xr.open_dataset(land_path)
land_area = land['land'].data

volume_path = '/data/Lakesvol.nc'
volume_data =  xr.open_dataset(volume_path)
lakesvol = volume_data['lakesvol_data_padded'].data #m3

area_path = '/data/Lakesarea.nc'
area_data =  xr.open_dataset(area_path)
lakesarea = area_data['lakesarea'].data #m2

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

for year in range(2021, 2100):
   
    ############################### RIVER VOLUME #################################  
    discharge_file = f'/data/Discharge/RCP{RCP}/{GCM}/Discharge_{RCP}_{GCM}_{year}.nc'
    discharge_data = xr.open_dataset(discharge_file) #m3/s 
     
    # Constants
    aw = 5.01e-2  # km^-0.56 y^0.52
    bw = 0.52
    ad = 1.04e-3  # km^-0.11 y^0.37
    bd = 0.37
    sb = 2.12
    
    # Calculate Width, Depth, Length, and River_vol
    Width = aw * ((discharge_data.dis * 1e-9 / 3.16887646e-8) ** bw)  # km
    Depth = ad * ((discharge_data.dis * 1e-9 / 3.16887646e-8) ** bd)  # km
    Length = sb * np.sqrt(land_area)  # m
    
    River_vol = Width * 1000 * Depth * 1000 * Length  # m^3
    
############################# ADVECTION RATES ################################
    # Calculate the advection rate
    denominator = lakesvol + River_vol
    mask_nan_discharge = np.isnan(discharge_data.dis)
    # Apply the conditions
    adv_rate = np.where(mask_nan_discharge, np.nan, discharge_data.dis / denominator) #s-1
    
######################## RETENTION RATES #####################################        
    # Define the conditions and corresponding values
    condition_1 = discharge_data.dis > 14.2
    condition_2 = (discharge_data.dis <= 14.2) & (discharge_data.dis > 2.8)
    
    value_1 = 0.012
    value_2 = 0.068
    value_3 = 0.195
    
    # Create a new DataArray based on the conditions
    kret_riv = xr.where(condition_1, value_1, xr.where(condition_2, value_2, value_3))

    #Calculate retention rate
    ret_rate = (1 / (River_vol + lakesvol)) * (River_vol * kret_riv.values + 0.038 * lakesarea)

######################## WATER USE RATES #####################################
    # Load the NetCDF files
    runoff_file = f'/data/Runoff/RCP{RCP}/{GCM}/Runoff_{RCP}_{GCM}_{year}.nc'
    irr_file = f'/data/Irrigation/RCP{RCP}/{GCM}/Irr_{RCP}_{GCM}_{year}.nc'
    
    runoff_data = xr.open_dataset(runoff_file)['qtot'] / 1000  # Convert to m/y
    irr_data = xr.open_dataset(irr_file)  # m3
    
    # Calculate FE values
    edip = 0.29
    edop = 0.01
    adip = 0.85
    adop = 0.95
    bdip = 2.00
    
    FEdip = edip * 1 / (1 + ((runoff_data / adip) ** (-bdip)))
    FEdop = edop * (runoff_data ** adop)
    FEsoil = FEdip + FEdop
    
    # Calculate firr with conditions
    firr = np.where(discharge_data.dis == 0,  # Condition: discharge_data.data is zero
                0,  # If True, set firr to zero
                irr_data.airrww / (discharge_data.dis))  # If False, perform original calculation
    firr = np.where(np.isnan(discharge_data.dis), np.nan, firr)
    firr = np.where(discharge_data.dis < 1e-4, 0, firr)
    
    use_rate = firr * (1-FEsoil) * adv_rate #s-1

###################### FISH RICHNESS DENSITY ##################################
    temp = 0.0
    if RCP == 26:
        if GCM =='GFDL':    
                 fishrichness_file = f'/data/Fish_richness_{temp}.nc'
                 fish_richness = xr.open_dataset(fishrichness_file)
                
        elif GCM =='HADGEM':
             if year >= 20238:
                 temp = 1.5
                 fishrichness_file = f'/data/Fish_richness_{temp}.nc'
                 fish_richness = xr.open_dataset(fishrichness_file)
                 
        elif GCM =='IPSL':
              temp = 1.5
              if year >= 2033: 
                 temp = 2.0
                 fishrichness_file = f'/data/Fish_richness_{temp}.nc'
                 fish_richness = xr.open_dataset(fishrichness_file)
                  
        elif GCM =='MIROC':
              temp = 1.5
              if year >= 2035:
                 temp = 2.0
                 fishrichness_file = f'/data/Fish_richness_{temp}.nc'
                 fish_richness = xr.open_dataset(fishrichness_file)
    elif RCP == 60:
        if GCM =='GFDL': 
             if year >= 2052:
                 temp = 1.5
             if year >= 2073:
                 temp = 2.0
                 fishrichness_file = f'/data/Fish_richness_{temp}.nc'
                 fish_richness = xr.open_dataset(fishrichness_file)
                
        elif GCM =='HADGEM':
             if year >= 2034:
                 temp = 1.5
             if year >= 2048:
                 temp = 2.0
             if year >= 2082:
                 temp = 3.2
                 fishrichness_file = f'/data/Fish_richness_{temp}.nc'
                 fish_richness = xr.open_dataset(fishrichness_file)
       
        elif GCM =='IPSL':
              temp = 1.5
              if year >= 2032:
                 temp = 2.0
              if year >= 2078:
                 temp = 3.2
                 fishrichness_file = f'/data/Fish_richness_{temp}.nc'
                 fish_richness = xr.open_dataset(fishrichness_file)
                  
        elif GCM =='MIROC':
              if year >= 2023:
                 temp = 1.5
              if year >= 2039:
                 temp = 2.0
              if year >= 2073:
                 temp = 3.2
                 fishrichness_file = f'/data/Fish_richness_{temp}.nc'
                 fish_richness = xr.open_dataset(fishrichness_file)
           
    # Calculate the FRD
    denominator = lakesvol + River_vol
    # Check if any values in the denominator are zero or NaN
    mask_zero_nan = np.logical_or(np.isnan(denominator), denominator < 1e-4)
                              
    frd = xr.full_like(denominator, np.nan)
    frd = frd.where(mask_zero_nan, fish_richness.fishrichness.values / denominator)             

######################### Effect factor ################################

################### Linear effect factor calculation #########################
   
    #Calculate effect factor
    EF_lake = frd * LEF_lake.data / FR_global
    EF_river = frd * LEF_river.data / FR_global
    
    denominator = lakesvol + River_vol
    mask_zero_nan = np.logical_or(np.isnan(denominator), denominator < 1e-4)
    
    # Initialize fraction arrays with NaNs
    fraction_lake = xr.full_like(denominator, np.nan)
    fraction_river = xr.full_like(denominator, np.nan)
    # Perform the division for non-zero and non-NaN values
    valid_indices = ~mask_zero_nan
    
    fraction_lake = fraction_lake.where(mask_zero_nan, lakesvol / denominator)
    fraction_river = fraction_river.where(mask_zero_nan, 1 - fraction_lake)
                  
    #Calculate final effect factors
    effect_factor = fraction_lake * EF_lake + fraction_river * EF_river
    # Convert effect_factor to xarray DataArray
    effect_factor_da = xr.DataArray(effect_factor, dims=('lat', 'lon'), name='effect_factor')
    # Add coordinate values if needed (assuming Latitude and Longitude coordinates)
    effect_factor_da['lat'] = discharge_data.lat
    effect_factor_da['lon'] = discharge_data.lon
    
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
