# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 14:21:29 2022

@author: rooze_dn
"""

# this script converts original land use classes into predefined new land use classes.

import geopandas as gpd
from geopandas import GeoSeries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

case = 'aoi_1_Votris'

rootdir = r"P:\11207698-reachout-project\scripts"
input_dir =  os.path.join(rootdir, "01_input")
output_dir =  os.path.join(rootdir, "03_output")

# Set target CRS
crs = 'EPSG:3857'

# Import conversion table for OSM road buffer
road_buffer_table = pd.read_excel(os.path.join(input_dir, 'OSM_lines_classification_table.xlsx'))

# Import project area
ds_project_area = gpd.read_file(os.path.join(input_dir, 'aoi_1_Votris.shp'))
ds_project_area = ds_project_area.to_crs(crs)


#%%

''' IMPORT LAYERS '''
# Reading base OSM layers: roads, railways, waterways, buildings and water polygons

# Importing lines
ds_roads = gpd.read_file(os.path.join(input_dir, 'gis_osm_roads_free_1.shp'), mask = ds_project_area)
ds_roads = ds_roads.to_crs(crs)

ds_railways = gpd.read_file(os.path.join(input_dir, 'gis_osm_railways_free_1.shp'), mask = ds_project_area)
ds_railways = ds_railways.to_crs(crs)

ds_waterways = gpd.read_file(os.path.join(input_dir, 'gis_osm_waterways_free_1.shp'), mask = ds_project_area)
ds_waterways = ds_waterways.to_crs(crs)

# Combine all polylines into 1 dataset for translation
ds_joined = pd.concat([ds_roads, ds_railways, ds_waterways])
# Merge dataframe with translation table on fclass
ds_joined = ds_joined.merge(road_buffer_table, on='fclass', how='left')
  
    

# Importing polygons
ds_buildings = gpd.read_file(os.path.join(input_dir, 'gis_osm_buildings_a_free_1.shp'), mask = ds_project_area)
ds_buildings = ds_buildings.to_crs(crs)
ds_buildings = ds_buildings.assign(reclass='paved_roof')

ds_water_a = gpd.read_file(os.path.join(input_dir, 'gis_osm_water_a_free_1.shp'), mask = ds_project_area)
ds_water_a = ds_water_a.to_crs(crs)
ds_water_a = ds_water_a.assign(reclass='water')

#%%

''' FUNCTIONS '''

def linestring_buffer(input_ds, reclass, **kwargs):
    # Creating buffers around categories
    print('Buffering: ', reclass)
   
    input_ds_select = input_ds.loc[input_ds['reclass'] == reclass]
    
    output_ds = input_ds_select.buffer((input_ds_select['width_t']) / 2)
    
    output_ds = gpd.GeoDataFrame(geometry=gpd.GeoSeries(output_ds))
    output_ds = output_ds.assign(reclass = reclass)
    output_ds = output_ds.dissolve(by='reclass', aggfunc='sum')
    output_ds = output_ds.reset_index()      

    return output_ds


def combine_layers(ds_base, ds_add, **kwargs):
    # Combine two geodataframe layers
    if not ds_add.empty:
        #ds_add = ds_add.explode()
        # Cut out new layer from base layer
        ds_out = gpd.overlay(ds_base, ds_add, how = 'difference')
        # Add new layer to base layer
        ds_out = pd.concat([ds_out, ds_add])
        
        return ds_out
    
    else:
        return ds_base



#%%

''' GENERATION OF POLYGONS '''

# Create buffers along lines
ds_closed_paved = linestring_buffer(ds_joined, 'closed_paved')
#ds_closed_paved = ds_closed_paved.explode(index_parts=True)
ds_open_paved = linestring_buffer(ds_joined, 'open_paved')
#ds_open_paved = ds_open_paved.explode(index_parts=True)
ds_water = linestring_buffer(ds_joined, 'water')
#ds_water = ds_water.explode(index_parts=True)

# Use buildings as paved roof area
ds_paved_roof = ds_buildings
#ds_paved_roof = ds_paved_roof.explode(index_parts=True)

# Join water areas and waterways
ds_water = pd.concat([ds_water_a, ds_water])
#ds_water = ds_water.explode(index_parts=True)


# Create unpaved base layer
ds_unpaved = ds_project_area.assign(reclass = 'unpaved')

#%%

''' MERGING LAYERS FOR FINAL LAND USE '''

#landuse_final = ds_project_area
#landuse_final = landuse_final.assign(reclass = 'unpaved')

landuse_final = combine_layers(ds_unpaved, ds_water)
landuse_final = combine_layers(landuse_final, ds_open_paved)
landuse_final = combine_layers(landuse_final, ds_closed_paved)
landuse_final = combine_layers(landuse_final, ds_paved_roof)


# Dissolve by reclass
landuse_final = landuse_final.dissolve(by='reclass', aggfunc='sum')
# Clip by project area to create neat land use map
landuse_final = gpd.clip(landuse_final, ds_project_area, keep_geom_type=True)
#landuse_final = landuse_final.reset_index()

#landuse_final = landuse_final_backup

#%%

# plot:
#fig, ax = plt.subplots(figsize = (10,8))
#landuse_final = landuse_final.reset_index()
#landuse_final.plot(ax = ax, column = 'reclass')


#%%

''' Calculate area '''

### make dataframe of areas
df_area = np.round(landuse_final.area.to_frame(),0)
df_area = df_area.rename(columns={0: 'area'}).reset_index()
area_tot = float(df_area['area'].sum())

# Increase water size
if ds_water.empty==True or landuse_final.area.water < 0.01*area_tot:
    
    # Add water if not present
    if ds_water.empty==True:
        df_area.loc[len(df_area)] = ['water', 0]
    
    area_tot_new = area_tot / 0.99  
    
    df_area.loc[df_area['reclass'] == 'water', 'area'] = df_area.loc[df_area['reclass'] == 'water', 'area'] + area_tot_new * 0.01
    df_area['perc'] = np.round(df_area['area'] / area_tot_new, 3)
else:
    df_area['perc'] = np.round(df_area['area'] / area_tot, 3)

#%%

df_area.to_csv(os.path.join(output_dir, f"landuse_areas_{case}.csv"))
landuse_final.to_file(os.path.join(output_dir, f"landuse_reclas_{case}.shp"))
