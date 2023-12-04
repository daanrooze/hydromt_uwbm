import pandas as pd
import numpy as np
import re
import logging
from typing import Tuple, Union, Optional

logger = logging.getLogger(__name__)


### NOTES
# assuming DateTime format does not have to be provided by user and is provided by 'new' workflow in HydroMT
# can we show the user the options for temporal interpolation?
# Should interpolation of all time series happen in 1 function? Or do we call the function multiple times?

class testclass(object):

    def setup_forcing_from_constant(
        self,
        ts_in: pd.DataFrame,
        key: str,
        interp_method: Optional[str] = 'none'
    ) -> pd.DataFrame:
        """Workflow for temporal interpolation of one or multiple time series. Used for precipitation and evaporation time series.
    
        Parameters
        ----------
        ts_in: pandas.DataFrame
            Imported DataFrame time series containing precipitation and/or evaporation.
        key: str
            Name of key containing forcing data to be interpolated.
        interp_method: str, optional
            Method for temporal interpolation of time series. Options: none (default), zeros (inserting zeros), ffill, bfill, linear (linear interpolation between previous and next value)
    
        Returns
        ----------
        p_out: pandas.DataFrame
            interpolated precipitation and evaporation time series
        """
        logger.info(f"Interpolating {key} using {interp_method}.")
        
        ### step 1: remove all non-int and non-float vlaues and replace with NaN
        ts_in[key] = pd.to_numeric(ts_in[key], errors='coerce')
        
        ### step 2: replace all negative values with 0
        ts_in[key][ts_in[key] < 0] = 0
        
        ### step 3: Interpolate with provided method
        if interp_method == 'none':
            ts_out = ts_in
        elif interp_method == 'linear':
            ts_in[key] = ts_in[key].interpolate(method='linear')
            ts_out = ts_in
        elif interp_method == 'zeros':
            ts_out = ts_in.fillna(0)
        else:
            ts_out = ts_in.fillna(method=interp_method)
        
        return ts_out

#%%

import hydromt
from hydromt.log import setuplog
from hydromt.workflows.forcing import pet_makkink
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import mode
from typing import Tuple, Union, Optional
import datetime

### CONFIGURATION
# selecting the start and end date for sampling
date_start = pd.to_datetime("1990-01-01")
date_end = pd.to_datetime("2023-01-01")

case = "Nainital"
timestep = 3600 #seconds

#%%
### Importing
logger = setuplog("preparing UWBM", log_level=10)
rootdir = r"P:\11208922-edb-uttarakhand\14_scripts"
input_dir = os.path.join(rootdir, "01_input")
output_dir = os.path.join(rootdir, "03_output")

### Import hydromt data catalog
data_catalog = hydromt.DataCatalog(logger=logger, data_libs=["deltares_data"])

### Import project area shp
geom = gpd.read_file(os.path.join(input_dir, 'Nainital_polygon.shp'))

## Import translation tables
#transl_soil = pd.read_excel(os.path.join(input_dir, "soil_translation.xlsx"))
#transl_landuse = pd.read_excel(os.path.join(input_dir, "landuse_translation.xlsx"))

#%%
### Functions

def setup_forcing_from_constant(
    self,
    ts_in: pd.DataFrame,
    key: str,
    interp_method: Optional[str] = 'none'
    ) -> pd.DataFrame:
    """Workflow for temporal interpolation of one or multiple time series. Used for precipitation and evaporation time series.

    Parameters
    ----------
    ts_in: pandas.DataFrame
        Imported DataFrame time series containing precipitation and/or evaporation.
    key: str
        Name of key containing forcing data to be interpolated.
    interp_method: str, optional
        Method for temporal interpolation of time series. Options: none (default), zeros (inserting zeros), ffill, bfill, linear (linear interpolation between previous and next value)

    Returns
    ----------
    p_out: pandas.DataFrame
        interpolated precipitation and evaporation time series
    """
    logger.info(f"Interpolating {key} using {interp_method}.")
    
    ### step 1: remove all non-int and non-float vlaues and replace with NaN
    ts_in[key] = pd.to_numeric(ts_in[key], errors='coerce')
    
            
    ### step 2: replace all negative values with 0
    ts_in[key][ts_in[key] < 0] = 0
    
    ### step 3: Interpolate with provided method
    if interp_method == 'none':
        ts_out = ts_in
    elif interp_method == 'linear':
        ts_in[key] = ts_in[key].interpolate(method='linear')
        ts_out = ts_in
    elif interp_method == 'zeros':
        ts_out = ts_in.fillna(0)
    else:
        ts_out = ts_in.fillna(method=interp_method)
    
    return ts_out


#%%
### Preparations
num_yrs = int(np.round(((date_end-date_start).days)/365.25, 0))
num_hrs = timestep/3600
TimeDelta = datetime.timedelta(seconds=timestep)


#%%
""" PART 1: dynamic data """

### importing dynamic data from data catalog
ds_meteo = data_catalog.get_rasterdataset("era5_hourly_zarr", geom=geom, time_tuple=[date_start, date_end],
                                          variables=['precip', 'temp', 'press_msl', 'kin', 'kout'], 
                                          single_var_as_array=True, buffer=1)


### Calculating PET
ds_meteo['PET'] = pet_makkink(ds_meteo['temp'], ds_meteo['press_msl'], ds_meteo['kin'], timestep=timestep)


### reprojection according to timestep given
ds_meteo = hydromt.workflows.forcing.resample_time(ds_meteo[{'precip', 'temp', 'press_msl', 'kin', 'kout', 'PET'}], freq = TimeDelta, downsampling="sum")


# sample uit ds_meteo, dan PET berekenen
ds_meteo_select = ds_meteo.raster.sample(geom.centroid).to_dataframe()
ds_meteo_select = ds_meteo_select.droplevel(level=1).reset_index()


### Data quality check
ds_meteo_select = setup_forcing_from_constant(object, ds_meteo_select, 'precip', 'zeros')
ds_meteo_select = setup_forcing_from_constant(object, ds_meteo_select, 'PET', 'zeros')


    
### Creating desired df
df_forcing = ds_meteo_select[["time", "precip", "PET"]]


### rename columns
df_forcing = df_forcing.rename(columns={"time":"date", "precip":"P_atm", "PET":"E_pot_OW"})


### calculate crop reference ET
df_forcing["Ref.grass"] = df_forcing["E_pot_OW"] * 0.8982


### reposition columns to "date, P_atm, Ref.grass, E_pot_OW"
df_forcing = df_forcing.loc[:, ["date","P_atm","Ref.grass","E_pot_OW"]]
df_forcing = df_forcing.set_index("date")


### export df to csv and write date into correct format
df_forcing.to_csv(os.path.join(output_dir, f"forcing_{case}_{num_yrs}_years_{int(np.round(num_hrs,0))}_h.csv"),
                  sep=',', date_format="%d-%m-%Y %H:%M")

