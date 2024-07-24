"""
Created on Tue Feb 27 11:46:27 2024

@author: KVasilakou 
"""

import xarray as xr
import numpy as np

##################################################################################
############################ RIVER DISCHARGE DATA ################################
##################################################################################

# Load the .nc4 file
nc_file = '../data/...nc4'
data = xr.open_dataset(nc_file)

# Iterate over each year
for year in range(2021, 2100):
    # Select data for the current year
    yearly_data = data.sel(time=str(year))
    
    # Calculate annual average
    yearly_avg = yearly_data.mean(dim='time')

    # Save new .nc file
    new_nc_file = f'Discharge_{RCP}_{GCM}_{year}.nc'
    yearly_avg.to_netcdf(new_nc_file) #m3/s
    print(f"Annual river discharge data saved to {new_nc_file}")
