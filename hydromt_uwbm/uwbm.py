import logging
from hydromt.models import VectorModel

from pathlib import Path
from os.path import basename, dirname, isdir, isfile, join
from typing import Optional, Dict, Any, Union, List

from . import DATADIR, workflows

import hydromt
import numpy as np
import pandas as pd
import geopandas as gpd
import codecs
import toml

__all__ = ["UWBM"]

logger = logging.getLogger(__name__)

class UWBM(VectorModel):
    """This is the uwbm class"""
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
    _FOLDERS: List[str] = ["input",
                           "input/project_area",
                           "input/landuse",
                           "input/config",
                           "output",
                           "output/forcing",
                           "output/landuse",
                           "output/config"]

    _CATALOGS = [join(_DATADIR, "parameters_data.yml")]
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

    # ====================================================================================================================================================
    # SETUP METHODS
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
        self.set_forcing(precip_out, name="precip")


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
        else:
            methods = [
                "debruin",
                "makking",
            ]
            raise ValueError(
                f"Unknown pet method {pet_method}, select from {methods}"
            )

        ds = self.data_catalog.get_rasterdataset(
            temp_pet_fn,
            geom=geom,
            buffer=1,
            time_tuple=(starttime, endtime),
            variables=variables,
            single_var_as_array=False,
        )

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
        
        pet_out = hydromt.workflows.forcing.resample_time(pet_out, freq = freq, downsampling="mean")
        
        pet_out = pet_out.to_dataframe(name="E_pot_OW")
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
        
        Adds model layer:
        * **lu_map**: polygon layer containing urban land use
        * **lu_table**: table containing urban land use surface areas [m2]
            
        Updates config:
        * **landuse_area**: surface area of the land use clasess [m2]
        * **landuse_frac**: surface area fraction of the land use clasess [-]

        Parameters
        ----------
        source: str, optional
            Source of landuse base files. Current default is "osm".
        landuse_mapping_fn: str, optional
            Name of landuse mapping translation table. Default is "osm_mapping_default".
        """
        self.logger.info("Preparing landuse map.")
        
        sources = ["osm"]
        if source not in sources:
            raise IOError(f"Provide source of landuse files from {sources}")
                
        if source == "osm":
            if landuse_mapping_fn is None:
                self.logger.info(f"No landuse translation table provided. Using default translation table for source {source}.")
                fn_map = f"{source}_mapping_default"
            else:
                fn_map = landuse_mapping_fn
            if not isfile(fn_map) and fn_map not in self.data_catalog:
                raise ValueError(f"LULC mapping file not found: {fn_map}")
            
            table = self.data_catalog.get_dataframe(fn_map)
            if not all(item in table.columns for item in ['fclass', 'width_t', 'reclass']):
                raise IOError("Provide translation table with columns 'fclass', 'width_t', 'reclass'")
            if not all(item in ['paved_roof', 'closed_paved', 'open_paved', 'unpaved', 'water'] for item in table['reclass']):
                raise IOError("Valid translation classes are 'paved_roof', 'closed_paved', 'open_paved', 'unpaved', 'water'")
            if not table['width_t'].dtypes in ['float64', 'int', 'int64']:
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
                    osm_layer = self.data_catalog.get_geodataframe(layer, geom = self.region, crs=self.crs)  
                    osm_layer = osm_layer.to_crs(self.crs)
                    self.set_geoms(osm_layer, name=layer)
                except:
                    osm_layer = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=self.crs)
                    osm_layer = osm_layer.to_crs(self.crs)
                    self.set_geoms(osm_layer, name=layer)
            
            lu_map = workflows.landuse.landuse_from_osm(
                region = self.region,
                road_fn = self.geoms["osm_roads"],
                railway_fn = self.geoms["osm_railways"],
                waterways_fn = self.geoms["osm_waterways"],
                buildings_area = self.geoms["osm_buildings"],
                water_area = self.geoms["osm_water"],
                landuse_mapping_fn = table
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
            self.set_config("landuse_area", f"{reclass}", float(df_landuse.loc[df_landuse["reclass"]==reclass, 'area']))
            self.set_config("landuse_frac", f"{reclass}", float(df_landuse.loc[df_landuse["reclass"]==reclass, 'frac']))


    def setup_model_config(
            self,
            config_fn: str = None
    ):
        """ Update TOML configuration file based on landuse calculations.

        Parameters
        ----------
        config_fn: str, optional
            Path to the config file. Default is self.config['name']
        """
        if config_fn == None:
            config_fn = f"ep_neighbourhood_{self.config['name']}.ini"
        neighbourhood_params = self._configread(config_fn = config_fn)
        keys = ['op', 'ow', 'up', 'pr', 'cp']
        for key in keys:
            neighbourhood_params[f"tot_{key}_area"] = self.get_config("landuse_area", f"{key}")
            neighbourhood_params[f"{key}_frac"] = self.get_config("landuse_frac", f"{key}")
        neighbourhood_params["tot_area"] = self.get_config("landuse_area", "tot_area")
        self._configwrite(neighbourhood_params = neighbourhood_params)

    # ====================================================================================================================================================
    # I/O METHODS

    def write(self):
        "Generic write function for all model workflows"
        self.write_forcing()
        self.write_tables(join(self.root, "output", "landuse", f"landuse_{self.config['name']}.csv"))
        self.write_geoms(join(self.root, "output", "landuse", f"landuse_{self.config['name']}.geojson"))


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
        if len(self.forcing)>0:
        
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
        
        
    def _configread(self, config_fn):
        """Read TOML configuration file.
        This function serves as alternative to the default read_config function to support ini files without headers"""
        path = join(self.root, "input", "config", config_fn)
        with codecs.open(path, "r", encoding="utf-8") as f:
            fdict = toml.load(f)
        return fdict


    def _configwrite(self, neighbourhood_params):
        """Write TOML configuration file"""
        directory = join(self.root, "output", "config")
        with codecs.open(join(directory, f"ep_neighbourhood_{self.config['name']}.ini"), "w", encoding="utf-8") as f:
            toml.dump(neighbourhood_params, f)
    