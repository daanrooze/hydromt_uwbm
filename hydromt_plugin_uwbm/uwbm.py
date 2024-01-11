import logging
from hydromt.models import VectorModel

from pathlib import Path
import os
from os.path import basename, dirname, isdir, isfile, join
from typing import Optional, Dict, Any, Union, List

from . import DATADIR, workflows

import hydromt
#from hydromt import workflows
import numpy as np
import pandas as pd
import geopandas as gpd
import codecs
import toml
import glob
import xarray as xr

__all__ = ["UWBM"]

logger = logging.getLogger(__name__)


class UWBM(VectorModel):
    """This is the uwbm class"""

    # Any global class variables your model should have go here
    _NAME: str = "UWBM"
    _CONF: str = "neighbourhood_params.ini"
    _DATADIR: Path = DATADIR
    _GEOMS = {
        "OSM": "OpenStreetMap"
        }
    _FORCING = {
        "time" : "date",
        "precip": "P_atm",
        "PET":"E_pot_OW"
        }

    # Name of default folders to create in the model directory
    _FOLDERS: List[str] = ["input", "input/landuse", "input/forcing", "input/config", "output", "output/landuse", "output/forcing", "output/config"]

    # Name of defaults catalogs to include when initialising the model
    # For example to include model specific parameter data or mapping
    # These default catalogs can be placed in the _DATADIR folder.
    _CATALOGS: List[str] = []
    # Cli args forwards the region and res arguments to the correct functions
    # Uncomment, check and overwrite if needed
    # _CLI_ARGS = {"region": <your func>, "res": <your func>}

    def __init__(
        self,
        root: Optional[str] = None,
        mode: str = "w+",
        config_fn: Optional[str] = None,
        data_libs: Optional[Union[List[str], str]] = None,
        logger: logging.Logger = logger,
    ):
        """Initialize the uwbm model class UWBM.

        Contains methods to read, write, setup and update uwbm models.

        Parameters
        ----------
        root : str, Path, optional
            Path to the model folder
        mode : {'w', 'w+', 'r', 'r+'}, optional
            Mode to open the model, by default 'w'
        config_fn : str, Path, optional
            Path to the model configuration file, by default None to read
            from template in build mode or default name in update mode.
        data_libs : list of str, optional
            List of data catalogs to use, by default None.
        logger : logging.Logger, optional
            Logger to use, by default logger
        """
        # Add model _CATALOGS to the data_libs
        if self._CATALOGS:
            if isinstance(data_libs, str):
                data_libs = [data_libs]
            if data_libs is None: 
                data_libs = []
            data_libs = data_libs + self._CATALOGS

        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )
        # If your model needs any extra specific initialisation add them here




    # SETUP METHODS
    # Write here specific methods to add or update model data components

    def setup_project(
        self,
        region,
        name: str = None,
        t_start: str = None,
        t_end: str = None,
        ts: int = None,
        crs: Optional[str] = "EPSG:3857"
    ):
        """Setup project geometry from vector"""
        if name == None:
            raise IOError("Provide name of study")
        if region == None:
            raise IOError("Provide path to case study project area")
        if t_start == None:
            raise IOError("Provide start date of time period in format YYYY-MM-DD")
        if t_end == None:
            raise IOError("Provide end date of time period in format YYYY-MM-DD")
        if ts == None or ts not in [3600,86400]:
            raise IOError("Provide timestep in seconds (3600 or 86400 seconds)")
        
        kind, region = hydromt.workflows.parse_region(
            region, data_catalog=self.data_catalog, logger=self.logger,
        )
        
        
        
        if kind in ["geom", "bbox"]:
            self.setup_region(region=region, hydrography_fn=None, basin_index_fn=None)
        else:
            raise IOError("Provide project region as either GeoPandas DataFrame or BoundingBox.")
        
        region = self.geoms["region"].to_crs(crs)
        self.set_geoms(region, name="region")
        
        self.set_config("starttime", pd.to_datetime(t_start))
        self.set_config("endtime", pd.to_datetime(t_end))
        self.set_config("timestepsecs", ts)
        self.set_config("name", name)
    
        
    def setup_precip_forcing(
        self,
        precip_fn: str = "era5_hourly_zarr",
        **kwargs,
    ) -> None:
        """Generate area-averaged, tabular precipitation forcing for geom.

        Adds model layer:

        * **precip**: precipitation [mm]

        Parameters
        ----------
        precip_fn : str, default era5_hourly_zarr
            Precipitation data source.

            * Required variable: ['precip']
        """
        if precip_fn is None:
            return
        starttime = self.get_config("starttime")
        endtime = self.get_config("endtime")
        freq = pd.to_timedelta(self.get_config("timestepsecs"), unit="s")
        geom = self.region
        #geom = self.geoms["project_geom"]

        precip = self.data_catalog.get_rasterdataset(
            precip_fn,
            geom=geom,
            buffer=2,
            time_tuple=(starttime, endtime),
            variables=["precip"],
        )
        
        precip = hydromt.workflows.forcing.resample_time(precip, freq = freq, downsampling="sum")
        
        precip_out = precip.raster.sample(geom.centroid).to_dataframe(name="P_atm")
        precip_out = precip_out.droplevel(level=1).reset_index()

        precip_out.attrs.update({"precip_fn": precip_fn})
        self.set_forcing(precip_out, name="precip") #TODO change to P_atm


    def setup_pet_forcing(
        self,
        temp_pet_fn: str = "era5_hourly_zarr",
        pet_method: str = "debruin",
        press_correction: bool = False,
        temp_correction: bool = False,
        wind_correction: bool = True,
        wind_altitude: int = 10,
        reproj_method: str = "nearest_index",
        dem_forcing_fn: str = "era5_orography",
        **kwargs,
    ) -> None:
        """Generate area-averaged, tabular reference evapotranspiration forcing for geom.

        Adds model layer:

        * **pet**: reference evapotranspiration [mm]

        Parameters
        ----------
        temp_pet_fn : str, optional
            Name or path of data source with variables to calculate temperature
            and reference evapotranspiration, see data/forcing_sources.yml.
            By default 'era5_hourly_zarr'.

            * Required variable for temperature: ['temp']

            * Required variables for De Bruin reference evapotranspiration: \
                ['temp', 'press_msl', 'kin', 'kout']

            * Required variables for Makkink reference evapotranspiration: \
                ['temp', 'press_msl', 'kin']

            * Required variables for daily Penman-Monteith \
                reference evapotranspiration: \
                    either ['temp', 'temp_min', 'temp_max', 'wind', 'rh', 'kin'] \
                    for 'penman-monteith_rh_simple' or ['temp', 'temp_min', 'temp_max', 'temp_dew', \
                    'wind', 'kin', 'press_msl', "wind10_u", "wind10_v"] for 'penman-monteith_tdew' \
                    (these are the variables available in ERA5)
                pet_method : {'debruin', 'makkink', 'penman-monteith_rh_simple', \
                    'penman-monteith_tdew'}, optional
                    Reference evapotranspiration method, by default 'debruin'.
                    If penman-monteith is used, requires the installation of the pyet package.
                press_correction, temp_correction : bool, optional
                    If True pressure, temperature are corrected using elevation lapse rate,
                    by default False.
                dem_forcing_fn : str, default None
                    Elevation data source with coverage of entire meteorological forcing domain.
                    If temp_correction is True and dem_forcing_fn is provided this is used in
                    combination with elevation at model resolution to correct the temperature.
        """
        press_correction = False
        temp_correction = False
        
        if temp_pet_fn is None:
            return
        starttime = self.get_config("starttime")
        endtime = self.get_config("endtime")
        timestep = self.get_config("timestepsecs")
        freq = pd.to_timedelta(timestep, unit="s")
        geom = self.region

        variables = ["temp"]
        if pet_method == "debruin":
            variables += ["press_msl", "kin", "kout"]
        elif pet_method == "makkink":
            variables += ["press_msl", "kin"]
        elif pet_method == "penman-monteith_rh_simple":
            variables += ["temp_min", "temp_max", "wind", "rh", "kin"]
        elif pet_method == "penman-monteith_tdew":
            variables += [
                "temp_min",
                "temp_max",
                "wind10_u",
                "wind10_v",
                "temp_dew",
                "kin",
                "press_msl",
            ]
        else:
            methods = [
                "debruin",
                "makking",
            #    "penman-monteith_rh_simple",
            #    "penman-monteith_tdew",
            ]
            ValueError(f"Unknown pet method {pet_method}, select from {methods}")

        ds = self.data_catalog.get_rasterdataset(
            temp_pet_fn,
            geom=geom,
            buffer=1,
            time_tuple=(starttime, endtime),
            variables=variables,
            single_var_as_array=False,  # always return dataset
        )

        if (
            "penman-monteith" in pet_method
        ):  # also downscaled temp_min and temp_max for Penman needed
        
            dem_forcing = None
            if dem_forcing_fn is not None:
                dem_forcing = self.data_catalog.get_rasterdataset(
                    dem_forcing_fn,
                    geom=ds.raster.box,  # clip dem with forcing bbox for full coverage
                    buffer=2,
                    variables=["elevtn"],
                ).squeeze()

            temp_in = hydromt.workflows.forcing.temp(
                ds["temp"],
                dem_model=None,
                dem_forcing=dem_forcing,
                lapse_correction=temp_correction,
                logger=self.logger,
                freq=None,  # resample time after pet workflow
                **kwargs,
            )
        
            temp_max_in = hydromt.workflows.forcing.temp(
                ds["temp_max"],
                dem_model=self.grid[self._MAPS["elevtn"]],
                dem_forcing=dem_forcing,
                lapse_correction=temp_correction,
                logger=self.logger,
                freq=None,  # resample time after pet workflow
                **kwargs,
            )
            temp_max_in.name = "temp_max"

            temp_min_in = hydromt.workflows.forcing.temp(
                ds["temp_min"],
                dem_model=self.grid[self._MAPS["elevtn"]],
                dem_forcing=dem_forcing,
                lapse_correction=temp_correction,
                logger=self.logger,
                freq=None,  # resample time after pet workflow
                **kwargs,
            )
            temp_min_in.name = "temp_min"

            temp_in = xr.merge([temp_in, temp_max_in, temp_min_in])


        ds_out = ds.raster.sample(geom.centroid)
        
        if pet_method == "debruin":
            pet_out = hydromt.workflows.forcing.pet_debruin(
                ds_out["temp"], 
                ds_out["press_msl"], 
                ds_out["kin"], 
                ds_out["kout"], 
                timestep=timestep, 
                cp=1005.0, 
                beta=20.0, 
                Cs=110.0
            )
        
        elif pet_method == "makkink":
            pet_out = hydromt.workflows.forcing.pet_makkink(
                ds_out["temp"], 
                ds_out["press_msl"], 
                ds_out["kin"], 
                timestep=timestep, 
                cp=1005.0
            )
        
        '''elif "penman-monteith" in pet_method:
            logger.info("Calculating Penman-Monteith ref evaporation")
            # Add wind
            # compute wind from u and v components at 10m (for era5)
            if ("wind10_u" in ds.data_vars) & ("wind10_v" in ds.data_vars):
                ds["wind"] = wind(
                    da_model=dem_model,
                    wind_u=ds["wind10_u"],
                    wind_v=ds["wind10_v"],
                    altitude=wind_altitude,
                    altitude_correction=wind_correction,
                )
            else:
                ds["wind"] = wind(
                    da_model=dem_model,
                    wind=ds["wind"],
                    altitude=wind_altitude,
                    altitude_correction=wind_correction,
                )
            if pet_method == "penman-monteith_rh_simple":
                pet_out = pm_fao56(
                    temp["temp"],
                    temp["temp_max"],
                    temp["temp_min"],
                    ds["press"],
                    ds["kin"],
                    ds["wind"],
                    ds["rh"],
                    dem_model,
                    "rh",
                )
            elif pet_method == "penman-monteith_tdew":
                pet_out = pm_fao56(
                    temp["temp"],
                    temp["temp_max"],
                    temp["temp_min"],
                    ds["press"],
                    ds["kin"],
                    ds["wind"],
                    ds["temp_dew"],
                    dem_model,
                    "temp_dew",
                )
        '''
        #TODO: check pet_out here
        pet_out = hydromt.workflows.forcing.resample_time(pet_out, freq = freq, downsampling="sum") #TODO: downsampling doesn't work for PET
        
        pet_out = pet_out.to_dataframe(name="E_pot_OW") #TODO: how to add name to dataarray in PET calculation?
        pet_out = pet_out.droplevel(level=1).reset_index()
        
        pet_out["Ref.grass"] = pet_out["E_pot_OW"] * 0.8982
    
        # Update meta attributes with setup opt
        opt_attr = {
            "pet_fn": temp_pet_fn,
            "pet_method": pet_method,
        }
        pet_out.attrs.update(opt_attr)
        self.set_forcing(pet_out, name="pet")


    def setup_landuse(
        self,
        source: str = "osm",
        landuse_mapping_fn = None
    ):
        """ Generate landuse map for region based on provided base files.

        Parameters
        ----------
        source: str, optional
            Source of landuse base files. Current default is "osm".
        landuse_mapping_fn: str, optional
            Name of landuse mapping translation table. Current default is "osm_mapping".
        """
        sources = ["osm"]
        if source not in sources:
            raise IOError(f"Provide source of landuse files from {sources}")
        
        if source == "osm":
            table = self.data_catalog.get_dataframe(landuse_mapping_fn) #TODO: how to add fallback option for csv in DATADIR?
            # in _init_ data catalog from yml, provide yml path in _CATALOGS (see wflow)
            self.set_tables(table, name = landuse_mapping_fn) #TODO: don't set to model
            if not all(item in table.columns for item in ['fclass', 'width_t', 'reclass']):
                raise IOError("Provide translation table with columns 'fclass', 'width_t', 'reclass'")
            if not all(item in ['paved_roof', 'closed_paved', 'open_paved', 'unpaved', 'water'] for item in table['reclass']):
                raise IOError("Valid translation classes are 'paved_roof', 'closed_paved', 'open_paved', 'unpaved', 'water'")
            if not table['width_t'].dtypes in ['float64', 'int']:
                raise IOError("Provide total width (width_t) values as float or int'")

            layers = [
                "osm_roads",
                "osm_railways",
                "osm_waterways",
                "osm_buildings",
                "osm_water"
            ]
            for layer in layers:
                try:
                    osm_layer = self.data_catalog.get_geodataframe(layer, geom = self.region)        
                    self.set_geoms(osm_layer, name=layer)
                except:
                    osm_layer = gpd.GeoDataFrame()
                    self.set_geoms(osm_layer, name=layer)
            
            lu_map = workflows.landuse.landuse_from_osm( #TODO change landuse function to take empty layers
                region = self.region,
                road_fn = self.geoms["osm_roads"],
                railway_fn = self.geoms[layers[1]],
                waterways_fn = self.geoms[layers[2]],
                buildings_area = self.geoms[layers[3]],
                water_area = self.geoms[layers[4]],
                landuse_mapping_fn=self._tables["osm_mapping"]
                )
        # Add landuse map to geoms
        self.set_geoms(lu_map, name="landuse_map")
        # Create landuse table from landuse map
        lu_table = workflows.landuse.landuse_table(
            lu_map = self.geoms['landuse_map']
            )
        # Add landuse table to tables
        self.set_tables(lu_table, name="landuse_table")
        # Add landuse categories to config
        df_landuse = self.tables['landuse_table']
        for reclass in df_landuse['reclass']:
            self.set_config("landuse", f"{reclass}", float(df_landuse.loc[df_landuse["reclass"]==reclass, 'area']))




    def setup_model_config(self):
        """Update TOML configuration file based on landuse calculations"""
        neighbourhood_params = self._configread()
        for key in self.get_config("landuse"):
            neighbourhood_params[key] = self.get_config("landuse", f"{key}")
            
        
        #TODO: update 1)config landuse area and 2) config landuse fraction for each land use class.
        
        #TODO: move config write below to separate config_write function
        #self._configwrite(neighbourhood_params = neighbourhood_params)


    # ====================================================================================================================================================
    # I/O METHODS
    # Write here specific methods to read or write model data components or overwrite the ones from HydroMT CORE

    def write(self):
        self.write_forcing()
        self.write_tables()
        self.write_geoms()
        #self.write_config()


    def _configread(self):
        """Read TOML configuration file"""
        directory = join(self.root, "input", "config")
        with codecs.open(join(directory, f"ep_neighbourhood_{self.config['name']}.ini"), "r", encoding="utf-8") as f:
            fdict = toml.load(f)
        #self.set_tables(fdict, name="neighbourhood_parameters")
        return fdict

    def _configwrite(self, neighbourhood_params):
        """Write TOML configuration file"""
        directory = join(self.root, "output", "config")
        with codecs.open(join(directory, f"ep_neighbourhood_{self.config['name']}.ini"), "w", encoding="utf-8") as f:
            toml.dump(neighbourhood_params, f)

 
    
    
    def write_forcing(
        self,
        fn_out: str = None,
        decimals=2
    ):
        """Write forcing at ``fn_out`` in model ready format (.csv).

        Parameters
        ----------
        fn_out: str, Path, optional
            Path to save output csv file. Default folder is output/forcing.
        decimals: int, optional
            Round the ouput data to the given number of decimals.
        """
           
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self.forcing:
            self.logger.info("Write forcing file")
        else:
            pass
               
        df = pd.DataFrame.from_dict(self.forcing)
        df = df[["time", "P_atm", "E_pot_OW", "Ref.grass"]]
        df = df.rename(columns={"time":f"{self._FORCING['time']}"})
        df = df.loc[:, ["date","P_atm","Ref.grass","E_pot_OW"]]
        df = df.set_index("date")
        
        h = int(self.config["timestepsecs"] / 3600)
        num_yrs = int(np.round(((self.config["endtime"]-self.config["starttime"]).days)/365.25, 0))
        
        if fn_out == None:
            path = join(self.root, "output", "forcing", f"Forcing_{self.config['name']}_{num_yrs}y_{h}h.csv")
        else:
            path = fn_out
        
        df.to_csv(path, sep=',', date_format="%d-%m-%Y %H:%M")
    
    
    
    
    def write_landuse(
            self,
            fn_out: str = None
    ):
        """Write landuse at ``fn_out`` in model ready format (.csv and .shp).

        Parameters
        ----------
        fn_out: str, Path, optional
            Path to save output files. Default folder is output/landuse.
        decimals: int, optional
            Round the ouput data to the given number of decimals.
        """
        if fn_out == None:
            path = join(self.root, "output", "landuse")
        else:
            path = fn_out
        # Write landuse table
        fn_lu_table = f"landuse_{self.config['name']}"
        self.tables['landuse_table'].to_csv(join(path, fn_lu_table + ".csv"))
        # Write landuse map
        fn_lu_map = f"landuse_{self.config['name']}"
        self.geoms['landuse_map'].to_file(join(path, fn_lu_map + ".shp"))
    
    
    
    
    # MODEL COMPONENTS AND PROPERTIES
    # Write here specific model properties and components not available in HydroMT CORE
