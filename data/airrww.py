"""
Created on Tue Feb 27 11:46:27 2024

@author: KVasilakou 
"""

import xarray as xr
import numpy as np

###############################################################################
############################### WATER IRRIGATION ##############################
###############################################################################

# Load the NetCDF file with decode_times=False
nc_file = '../data/...nc4'
data = xr.open_dataset(nc_file, engine='netcdf4', decode_times=False)

area_path = '/data/Land.nc'
area =  xr.open_dataset(area_path)
land_area = area['land'].data

#SPECIFY CLIMATE CHANGE SCENARIO!!
#RCP = 26, 60
RCP = 26
#GCM = 'GFDL', 'HADGEM', 'IPSL', 'MIROC'
GCM = 'GFDL'

# Calculate the start and end indices for each year from 2021 to 2099
start_index = (2021 - 2006) * 12  
end_index = (2099 - 2006 + 1) * 12 

# Define the number of seconds in each month 
days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
seconds_in_month = np.array(days_in_month) * 24 * 60 * 60

# Iterate over each year
for year in range(2021, 2100):
    # Calculate the indices for the current year
    start_month = (year - 2006) * 12  
    end_month = start_month + 12 

    # Select data for the current year
    data_year = data.isel(time=slice(start_month, end_month))

    # Calculate annual average
    yearly_total_kg_m2 = data_year.mean(dim='time')
    
    # Convert from kg/m2/s to m3/s (assuming 998 kg/m3 is the density conversion factor)
    yearly_total_m3 = yearly_total_kg_m2 * land_area / 998

    # Save new .nc file
    new_nc_file = f'Irr_{RCP}_{GCM}_{year}.nc'
    yearly_total_m3.to_netcdf(new_nc_file)
    print(f'Annual water irrigation withdrawal data saved to {new_nc_file}')
