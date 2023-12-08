import logging
from hydromt.models import VectorModel

from pathlib import Path
import os
from os.path import basename, dirname, isdir, isfile, join
from typing import Optional, Dict, Any, Union, List

from . import DATADIR, workflows

import hydromt
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

    # Name of default folders to create in the model directory
    _FOLDERS: List[str] = ["input"]

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
        mode: str = "w",
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
        self._tables = dict()
        self._geoms = None




    # SETUP METHODS
    # Write here specific methods to add or update model data components
    
    #TODO add function for checking if folders exist, otherwise add folders
    def makedir(
            self,
            path
    ):
        if not os.path.exists(path):
            self.logger.info(f"Directory {path} not found. Creating new directory.")
            os.makedirs(path)
   

    
    def setup_project(
        self,
        name: str = None,
        t_start: str = None,
        t_end: str = None,
        ts: int = None,
    ):
        """Setup project geometry from vector"""
        if name == None:
            raise IOError("Provide name of case study")
        if t_start == None:
            raise IOError("Provide start date of time period in format YYYY-MM-DD")
        if t_end == None:
            raise IOError("Provide end date of time period in format YYYY-MM-DD")
        if ts == None or ts not in [3600,86400]:
            raise IOError("Provide timestep in seconds (3600 or 86400 seconds)")
        
        _dir = "project_area"
        
        #TODO Add check on empty project area folder.
        # if case not in join(self.root, "input_data", _dir):
            #raise IOError(f"Case study region not found in {_dir}")

        self.read_geoms(_dir, name)
        self.set_geoms(self.geoms[name], name="project_geom")
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
        geom = self.geoms["project_geom"]
        #geom = self.geoms[self._GEOMS["project_geom"]]

        precip = self.data_catalog.get_rasterdataset(
            precip_fn,
            geom=geom,
            buffer=2,
            time_tuple=(starttime, endtime),
            variables=["precip"],
        )
        
        precip = hydromt.workflows.forcing.resample_time(precip, freq = freq, downsampling="sum")
        
        precip_out = precip.raster.sample(geom.centroid).to_dataframe()
        precip_out = precip_out.droplevel(level=1).reset_index()    #Data can be xarray.DataArray, xarray.Dataset or pandas.DataFrame.
                                                                    #If pandas.DataFrame, indices should be the DataFrame index and the columns
                                                                    #the variable names.

        precip_out.attrs.update({"precip_fn": precip_fn}) # <- needed or not?
        self.set_forcing(precip_out, name="precip")





    def setup_pet_forcing(
        self,
        temp_pet_fn: str = "era5_hourly_zarr",
        pet_method: str = "penman-monteith_rh_simple",
        press_correction: bool = True,
        temp_correction: bool = True,
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
        if temp_pet_fn is None:
            return
        starttime = self.get_config("starttime")
        endtime = self.get_config("endtime")
        timestep = self.get_config("timestepsecs")
        freq = pd.to_timedelta(timestep, unit="s")
        geom = self.geoms["project_geom"]
        #geom = self.geoms[self._GEOMS["project_geom"]]

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
                "penman-monteith_rh_simple",
                "penman-monteith_tdew",
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
            dem_model=dem_forcing, #TODO 'ValueError: dimension 'time' already exists as a scalar variable'
            dem_forcing=dem_forcing,
            lapse_correction=temp_correction,
            logger=self.logger,
            freq=None,  # resample time after pet workflow
            **kwargs,
        )

        if (
            "penman-monteith" in pet_method
        ):  # also downscaled temp_min and temp_max for Penman needed
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


        #TODO: how to sample ds, temp_in, dem_model before calculating forcing.pet()?
        ds_out = ds.raster.sample(geom.centroid).to_dataframe() 
        ds_out = ds_out.droplevel(level=1).reset_index()    #Data can be xarray.DataArray, xarray.Dataset or pandas.DataFrame.
                                                                    #If pandas.DataFrame, indices should be the DataFrame index and the columns
                                                                    #the variable names.


        pet_out = hydromt.workflows.forcing.pet(
            ds_out[variables[1:]],
            temp=temp_in,
            dem_model=self.grid[self._MAPS["elevtn"]],
            method=pet_method,
            press_correction=press_correction,
            wind_correction=wind_correction,
            wind_altitude=wind_altitude,
            reproj_method=reproj_method,
            freq=freq,
            resample_kwargs=dict(label="right", closed="right"),
            logger=self.logger,
            **kwargs,
        )
        
        pet_out = hydromt.workflows.forcing.resample_time(pet_out[{variables}], freq = freq, downsampling="sum")
        
        # Update meta attributes with setup opt
        opt_attr = {
            "pet_fn": temp_pet_fn,
            "pet_method": pet_method,
        }
        pet_out.attrs.update(opt_attr) # <- needed or not?
        self.set_forcing(pet_out, name="pet") # <- does this automatically add to the existing dataframe in self.forcing?





    
    def setup_landuse(
        self,
        landuse_mapping_fn="osm_mapping",
    ):
        
        self.read_OSM() # of read_geoms
        
        
        osm = self.geoms["OSM"]
        ds_landuse = workflows.landuse(
            osm,
            landuse_mapping_fn=self.tables[self._tables["osm_mapping"]]
            )
        self.set_tables(ds_landuse, name="landuse")
        



    

    def setup_soiltype( # not important for first functionality of plugin
        self,
        value_name: str,
        value: int or float,
        name: Optional[str] = None,
    ):
        """Adding a constant value to a response unit (for instance soilgrids).
        
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
        geom = self.geoms["project_geom"]
        
        ### type checks, copied from set_staticgeoms.
        gtypes = [gpd.GeoDataFrame, gpd.GeoSeries]
        if not np.any([isinstance(geom, t) for t in gtypes]):
            raise ValueError("First parameter map(s) should be geopandas.GeoDataFrame or geopandas.GeoSeries")
        
        ### setting single value:    
        if value_name not in geom:
            raise ValueError(f"Attribute '{value_name}' not found in GeoDataFrame")
        
        if not (isinstance(value, int) or isinstance(value, float)):
            raise ValueError(f"The assigned value for '{value_name}' must be an integer or float")
        else: 
            geom[value_name] = value
        
        self.set_tables(geom, name="soiltype")








    def setup_forcing_interpolation( #own function or integration with precipitation function?
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







    def setup_model_config(self, fn): #TODO: how to bring values from setup_landuse (saved as dF or dict) into the config toml file?
        """Update TOML configuration file based on landuse calculations"""
        self.set_config("total_area", "value") #refer to dictionary created by OSM landuse
        self.set_config("total_area", "value")
        self.set_config("total_area", "value")
        self.set_config("total_area", "value")
        self.set_config("total_area", "value")
        self.set_config("total_area", "value")
        
        
        return


    # ====================================================================================================================================================
    # I/O METHODS
    # Write here specific methods to read or write model data components or overwrite the ones from HydroMT CORE

    def read_geoms(
            self,
            _dir: str,
            geom_fn: str = None,
        ):
            """Read geoms at <root/data/geoms> and parse to geopandas."""
            if not self._write:
                self._geoms = dict()  # fresh start in read-only mode
            dir_default = join(self.root, "input_data", _dir)
       #     dir_mod = dirname(
       #         self.get_config("geoms", abs_path=True, fallback=dir_default)
       #     )
            if geom_fn == None:
                fns = glob.glob(join(dir_default, "*.shp"))
            else:
                fns = glob.glob(join(dir_default, f"{geom_fn}.shp"))
       #     if len(fns) > 1:
       #         self.logger.info("Reading model staticgeom files.")
            for fn in fns:
                if geom_fn == None:
                    name = basename(fn).split(".")[0]
                else:
                    name = geom_fn
                self.set_geoms(gpd.read_file(fn), name=name)


    def read_OSM(
            self,
            osm_fn: str = "osm",
            layers = [
                "gis_osm_roads_free_1",
                "gis_osm_railways_free_1",
                "gis_osm_waterways_free_1",
                "gis_osm_buildings_a_free_1",
                "gis_osm_water_a_free_1"
            ]
    ):
        if not self._write:
            self._geoms = dict()  # fresh start in read-only mode
        dir_default = join(self.root, "input_data")
    #    dir_mod = dirname( #op zoek naar input/landuse
    #        self.get_config("landuse", abs_path=True, fallback=dir_default)
    #    )
        fns = glob.glob(join(dir_default, "input_data", "*.shp") in layers) #TODO: how to only import the relevant layers?
        for fn in fns:
            name = basename(fn).split(".")[0]
            self.set_geoms(gpd.read_file(fn, mask = self.geoms), name="OSM") #TODO: save to single geopackage or different files under _geoms?



    def read_tables(self, **kwargs):
        """Read table files at <root> and parse to dict of dataframes."""
        """READ OSM LANDUSE TRANSLATION TABLE - copied from wflow"""
        if not self._write:
            self._tables = dict()  # start fresh in read-only mode

        self.logger.info("Reading model table files.")
        dir_landuse = join(self.root, "input_data/landuse")
        fns = glob.glob(join(dir_landuse, "*.csv"))
        if len(fns) > 0:
            for fn in fns:
                name = basename(fn).split(".")[0]
                tbl = pd.read_csv(fn)
                self.set_tables(tbl, name=name)
                #error message if more than 1 file for converting landuse






    def _configread(self, fn):
        """Read TOML configuration file"""
        with codecs.open(fn, "r", encoding="utf-8") as f:
            fdict = toml.load(f)
        return fdict #TODO how to add to model?

    def _configwrite(self, fn):
        """Write TOML configuration file"""
        with codecs.open(fn, "w", encoding="utf-8") as f:
            toml.dump(self.config, f)



        
    
    
    
    
    def write_forcing(
        self,
        decimals=2,
        datetime_format="%d-%m-%Y %H:%M",
        **kwargs,
    ):
        """Write forcing at ``fn_out`` in model ready format (.csv).

        Parameters
        ----------
        fn_out: str, Path, optional
            Path to save output csv file.
        decimals: int, optional
            Round the ouput data to the given number of decimals.
        datetime_format: str, optional
            Datetime notation of forcing. By default "%d-%m-%Y %H:%M".
        """
           
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self.forcing:
            self.logger.info("Write forcing file")
        
        #fn_out = # <- e.g. 'Dehradun_30y_1h.csv'
        
        df = pd.DataFrame.from_dict(self.forcing)
        
        ### Selecting desired variables
        df = df[["time", "precip"]]#, "PET"]] #TODO REMOVE HASH ONCE PET IS FIXED
        ### rename columns
        
        df['PET']=df['precip'] #TODO REMOVE FROM FILE ONCE PET IS FIXED
        
        df = df.rename(columns={"time":"date", "precip":"P_atm", "PET":"E_pot_OW"})
        ### calculate crop reference ET
        df["Ref.grass"] = df["E_pot_OW"] * 0.8982
        ### reposition columns to "date, P_atm, Ref.grass, E_pot_OW"
        df = df.loc[:, ["date","P_atm","Ref.grass","E_pot_OW"]]
        df = df.set_index("date")
        
        h = int(self.config["timestepsecs"] / 3600)
        num_yrs = int(np.round(((self.config["endtime"]-self.config["starttime"]).days)/365.25, 0))
        path = join(self.root, "output/forcing", f"Forcing_{self.config['name']}_{num_yrs}y_{h}h.csv")
        
        df.to_csv(path, sep=',', date_format=datetime_format)
    
    
    
    # MODEL COMPONENTS AND PROPERTIES
    # Write here specific model properties and components not available in HydroMT CORE
