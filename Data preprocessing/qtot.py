"""
Created on Tue Feb 27 11:46:27 2024

@author: KVasilakou 
"""

################################################################################
################################ RUNOFF DATA ###################################
################################################################################

# Load the .nc4 file
nc_file = '../data/...nc4'
data = xr.open_dataset(nc_file)

# Conversion factor from kg/m²/s to mm/year (assuming 1000 kg/m3 density)
conversion_factor = 60 * 60 * 24 * 365  # seconds in a year

# Iterate over each year
for year in range(2021, 2100):
    # Select data for the current year
    yearly_data = data.sel(time=str(year))
    
    # Convert data to mm/year 
    yearly_avg = yearly_data * conversion_factor

    # Calculate annual average
    yearly_avg = yearly_avg.mean(dim='time')

    # Save new .nc file
    new_nc_file = f'Runoff_{RCP}_{GCM}_{year}.nc'
    yearly_avg.to_netcdf(new_nc_file) #mm/year
    print(f"Annual runoff data saved to {new_nc_file}")
