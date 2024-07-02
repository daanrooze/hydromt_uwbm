""" Landuse workflows for UWBM plugin """

import geopandas as gpd
from geopandas import GeoSeries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import logging
logger = logging.getLogger(__name__)

__all__ = ["landuse_from_osm"]

def landuse_from_osm(
        region,
        road_fn: str = None,
        railway_fn: str = None,
        waterways_fn: str = None,
        buildings_area: str = None,
        water_area: str = None,
        landuse_mapping_fn: str = None
):
    """Preparing landuse map from OpenStreetMap.
    
    Parameters
    ----------
    road_fn: str
        Name of road line polygon layer.
    railway_fn: str
        Name of railway line polygon layer.
    waterways_fn: str
        Name of waterway line polygon layer.
    buildings_area: str
        Name of building polygon layer.
    water_area: str
        Name of water polygon layer.
    landuse_mapping_fn: str
        Name of OSM landuse translation table. Default table is taken from DATADIR.
    
    Returns
    ----------
    landuse_map: GeoJSON
        Landuse geometry.
    """
    # Create unpaved base layer from region
    da_unpaved = region
    da_unpaved = da_unpaved.assign(reclass = 'unpaved')
    # Combine all polylines into 1 dataset for translation
    ds_joined = pd.concat([road_fn, railway_fn, waterways_fn])
    # Merge dataframe with translation table on fclass
    ds_joined = ds_joined.merge(landuse_mapping_fn, on='fclass', how='left')
    # Assign land use columns
    da_paved_roof = buildings_area.assign(reclass='paved_roof')
    da_water_area = water_area.assign(reclass='water')
    # Create buffers along lines
    if any(ds_joined['reclass'] == 'closed_paved'):
        da_closed_paved = linestring_buffer(ds_joined, 'closed_paved')
    else:
        da_closed_paved = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry')
    if any(ds_joined['reclass'] == 'open_paved'):
        da_open_paved = linestring_buffer(ds_joined, 'open_paved')
    else:
        da_open_paved = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry')
    if any(ds_joined['reclass'] == 'water'):
        da_water = linestring_buffer(ds_joined, 'water')
    else:
        da_water = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry')
    # Join water areas and waterways
    da_water = pd.concat([da_water_area, da_water])
    # Create combined land use layers
    lu_map = combine_layers(da_unpaved, da_water)
    lu_map = combine_layers(lu_map, da_open_paved)
    lu_map = combine_layers(lu_map, da_closed_paved)
    lu_map = combine_layers(lu_map, da_paved_roof)
    # Dissolve by land use category
    lu_map = lu_map.dissolve(by='reclass', aggfunc='sum')
    # Clip by project area to create neat land use map
    lu_map = gpd.clip(lu_map, region, keep_geom_type=True)
    return lu_map

def landuse_table(
        lu_map
):
    """Preparing landuse table based on provided land use map.
    
    Parameters
    ----------
    lu_map: str
        Name polygon land use map.
    
    Returns
    ----------
    landuse_table: pandas.DataFrame
        Table with landuse areas and percentages of total.
    """
    # Generate dataframe with areas
    lu_table = np.round(lu_map.area.to_frame(),0)
    lu_table = lu_table.rename(columns={0: 'area'}).reset_index()
    tot_area = float(lu_table['area'].sum())
    # Increase water size
    if lu_map.loc['water'].empty==True or lu_map.loc['water'].geometry.area < 0.01*tot_area:
        
        # Add water if not present
        if lu_map.loc['water'].empty==True:
            lu_table.loc[len(lu_table)] = ['water', 0]
        
        area_tot_new = tot_area / 0.99  
        
        lu_table.loc[lu_table['reclass'] == 'water', 'area'] = lu_table.loc[lu_table['reclass'] == 'water', 'area'] + area_tot_new * 0.01
        lu_table['frac'] = np.round(lu_table['area'] / area_tot_new, 3)
    else:
        lu_table['frac'] = np.round(lu_table['area'] / tot_area, 3)
    lu_table = pd.concat([lu_table, pd.DataFrame({'reclass': 'tot_area', 'area': tot_area, 'frac': 1}, index=[len(lu_table)])])
    # Rename index values to model conventions
    lu_table['reclass'] = lu_table['reclass'].replace({
        'open_paved': 'op',
        'water': 'ow',
        'unpaved': 'up',
        'paved_roof': 'pr',
        'closed_paved': 'cp'})
    return lu_table

def linestring_buffer(
        input_ds,
        reclass
):
    """Generating buffers with varying sized depending on land use category.
    
    Parameters
    ----------
    input_ds: pandas.GeoDataFrame
        Pandas GeoDataFrame containing linestring elements.
    reclass: str
        Name of land use category.
    
    Returns
    ----------
    output_ds: pandas.GeoDataFrame
        Pandas GeoDataFrame with buffered polygons.
    """
    input_ds_select = input_ds.loc[input_ds['reclass'] == reclass]
    output_ds = input_ds_select.buffer((input_ds_select['width_t']) / 2, cap_style=2)
    output_ds = gpd.GeoDataFrame(geometry=gpd.GeoSeries(output_ds))
    output_ds = output_ds.assign(reclass = reclass)
    output_ds = output_ds.dissolve(by='reclass', aggfunc='sum')
    output_ds = output_ds.reset_index()      
    return output_ds

def combine_layers(
        ds_base,
        ds_add
):
    """Combining two GeoDataFrame layers into a single GeoDataFrame layer.
    
    Parameters
    ----------
    ds_base: pandas.GeoDataFrame
        Pandas GeoDataFrame containing base layer.
    ds_add: pandas.GeoDataFrame
        Pandas GeoDataFrame containing additional layer
    
    Returns
    ----------
    output_ds: pandas.GeoDataFrame
        Pandas GeoDataFrame with combined layers.
    """
    if not ds_add.empty:
        # Cut out new layer from base layer
        ds_out = gpd.overlay(ds_base, ds_add, how = 'difference')
        # Add new layer to base layer
        ds_out = pd.concat([ds_out, ds_add])
        return ds_out
    else:
        return ds_base

