

import geopandas as gpd
from geopandas import GeoSeries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np





def landuse_from_osm(
        road_fn: str = None,
        railway_fn: str = None,
        waterways_fn: str = None,
        buildings_area: str = None,
        water_area: str = None,
):
    """Preparing landuse map from OpenStreetMap.
    
    Parameters
    ----------
    value_name: str
        Name of the attribute that is changed.
    value: float or int
        Value of the attribute that is changed.
    name: str, optional
        Name of new map layer, this is used to overwrite the name of a DataFrame
        or to select a variable from a Dataset.
    
    Returns
    ----------
    response_unit: pd.GeoDataFrame
        Response unit geometry
    """
    
    
    return table, lu_map
    
    
    

