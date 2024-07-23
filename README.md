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

################################ INPUT DATA ####################################

All input data required for the calculations are already included in the data folder.
Data file includes pre-processed input data (georeferenfed files) required for the calculations. Pre-processing includes reprojection to EPSG:4326, with an extent of -180o,-90o:180o,90o and 30 arc-min resolution. The original data were taken from:
1. River discharge, Runoff, Irrigation water: https://doi.org/10.48364/ISIMIP.626689
2. Fish richness: https://doi.org/10.1038/s41467-021-21655-w
3. Flow direction: https://wsag.unh.edu/Stn-30/stn-30.html
4. Climate regions: https://koeppen-geiger.vu-wien.ac.at/present.htm
5. Lakes/reservoirs volume/area: https://www.hydrosheds.org/products/hydrolakes

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
The user should define the year(s) of analysis. Years range from 2021 till 2099.
All results are saved as .nc files in the results folder.
Intermediate results, such as advection, retantion, water use rates, are also saved as separate files. Final characterization factors are saved in the CFs folder.
Results are saved as "{Result}0.5{RCP}{GCM}{year}.nc".
Important note: calculations are time expensive for each scenario/year due to extensive loops
