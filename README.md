Created on Tue Feb 27 11:46:27 2024

################################### README ##################################### 

Author: Konstantina Vasilakou
https://orcid.org/0000-0001-8609-9512

Code generates freshwater eutrophication characterization factors, quantified as the potentially disappeared fraction of fish species per amount of phosphorus emitted annually.
Results are provided at a global 30 arc-min resolution, for 8 different RCP-GCM climate change scenarios.

############################## REQUIREMENTS ####################################                             
This code was created using Python 3.11.5 and conda 23.9.0.
Packages required:
xarray
numpy
pandas
rasterio
GDAL
A requirements.txt file is included in the repository.

################################ INPUT DATA ####################################

Orininal input data required for the calculations were taken from:
1. River discharge, Runoff, Irrigation water: https://doi.org/10.48364/ISIMIP.626689
2. Fish richness: https://doi.org/10.1038/s41467-021-21655-w
3. Flow direction: https://wsag.unh.edu/Stn-30/stn-30.html
4. Climate regions: https://koeppen-geiger.vu-wien.ac.at/present.htm
5. Lakes/reservoirs volume/area: https://www.hydrosheds.org/products/hydrolakes

Pre-processing is required for River discharge, Runoff and Irrigation water datasets, in order to calculate annual average.
Fish richness data are reprojected to EPSG:4326, with an extent of -180o,-90o:180o,90o and 30 arc-min resolution.
The relevant Python codes are also provided as separate files at the data folder.

############################ USER-REQUIRED DATA ###############################

Results can be generated for four RCP-GCM scenarios:
1. RCP = 2.6 and GCM = GFDL-ESM2M
2. RCP = 2.6 and GCM = HadGEM2-ES
3. RCP = 2.6 and GCM = IPSL-CM5A-LR
4. RCP = 2.6 and GCM = MIROC5
5. RCP = 6.0 and GCM = GFDL-ESM2M
6. RCP = 6.0 and GCM = HadGEM2-ES
7. RCP = 6.0 and GCM = IPSL-CM5A-LR
8. RCP = 6.0 and GCM = MIROC5

The user should define the scenario at the beginning of the code.
All characterization factors are saved as .nc files.
Intermediate results, such as advection, retantion, water use rates, can also be saved as separate files.
Results are saved as "{Variable}_{RCP}_{GCM}_{year}.nc"
