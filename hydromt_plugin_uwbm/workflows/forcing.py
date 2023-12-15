import logging
from hydromt.models import VectorModel

from pathlib import Path
import os
from os.path import basename, dirname, isdir, isfile, join
from typing import Optional, Dict, Any, Union, List

from . import DATADIR, workflows

import hydromt
from hydromt import workflows
import numpy as np
import pandas as pd
import geopandas as gpd
import codecs
import toml
import glob
import xarray as xr

logger = logging.getLogger(__name__)




#TODO: move forcing functions to here from main uwbm.py

def forcing_interpolation( #own function or integration with precipitation function?
    self,
    ts_in: pd.DataFrame,
    keys: List,
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
    for key in keys:
        logger.info(f"Interpolating {key} using {interp_method}.")
        
        ### step 1: remove all non-int and non-float vlaues and replace with NaN
        ts_in[key] = pd.to_numeric(ts_in[key], errors='coerce')
        
                
        ### step 2: replace all negative values with 0
        ts_in[key][ts_in[key] < 0] = float(0)
        
        ### step 3: Interpolate with provided method
        if not interp_method == 'none':
            if interp_method == 'linear':
                ts_in[key] = ts_in[key].interpolate(method='linear')
            elif interp_method == 'zeros':
                ts_in = ts_in.fillna(float(0))
            else:
                ts_in = ts_in.fillna(method=interp_method)
        
    return ts_in



