import logging
from hydromt.models import VectorModel

from pathlib import Path
from os.path import join
from typing import Optional, Dict, Any, Union, List

from . import DATADIR, workflows

import hydromt
import numpy as np
import pandas as pd
import geopandas as gpd
import codecs
import toml
import glob

__all__ = ["UWBM"]

logger = logging.getLogger(__name__)


class UWBM(VectorModel):
    """This is the uwbm class"""

    # Any global class variables your model should have go here
    _NAME: str = "UWBM"
    _CONF: str = "model.yaml"
    _DATADIR: Path = DATADIR

    # Name of default folders to create in the model directory
    _FOLDERS: List[str] = ["staticgeoms"]

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




    # SETUP METHODS
    # Write here specific methods to add or update model data components
    
    """setup forcing steps: 1) get precip, 2) get other vars (era5), 3)merge, 4) sample to centroid , 5) calculate PET"""
    
    def setup_precip_forcing( #FOR ARCHIVE ONLY
        self,
        precip_fn: str = "era5_hourly_zarr",
        chunksize: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Generate area-averaged precipitation forcing for geom.

        Adds model layer:

        * **precip**: precipitation [mm]

        Parameters
        ----------
        precip_fn : str, default era5_hourly_zarr
            Precipitation data source.

            * Required variable: ['precip']
        chunksize: int, optional
            Chunksize on time dimension for processing data (not for saving to disk!).
            If None the data chunksize is used, this can however be optimized for
            large/small catchments. By default None.
        """
        if precip_fn is None:
            return
        starttime = self.get_config("starttime")
        endtime = self.get_config("endtime")
        freq = pd.to_timedelta(self.get_config("timestepsecs"), unit="s")
        mask = self.grid[self._MAPS["basins"]].values > 0 #geom ipv raster

        precip = self.data_catalog.get_rasterdataset(
            precip_fn,
            geom=self.region,
            buffer=2,
            time_tuple=(starttime, endtime),
            variables=["precip"],
        )

        if chunksize is not None:
            precip = precip.chunk({"time": chunksize})

        precip_out = hydromt.workflows.forcing.precip(
            precip=precip,
            da_like=self.grid[self._MAPS["elevtn"]],
            freq=freq,
            resample_kwargs=dict(label="right", closed="right"),
            logger=self.logger,
            **kwargs,
        )

        # Update meta attributes (used for default output filename later)
        precip_out.attrs.update({"precip_fn": precip_fn})
        self.set_forcing(precip_out.where(mask), name="precip")



    def setup_forcing(
        self,
        forcing_fn: str = "era5_hourly_zarr",
        chunksize: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Generate area-averaged forcing for geom.

        Adds model layer:

        * **precip**: precipitation [mm]

        Parameters
        ----------
        forcing_fn : str, default era5_hourly_zarr
            Precipitation data source.

            * Required variable: ['precip']
        chunksize: int, optional
            Chunksize on time dimension for processing data (not for saving to disk!).
            If None the data chunksize is used, this can however be optimized for
            large/small catchments. By default None.
        """
        if forcing_fn is None:
            return
        starttime = self.get_config("starttime")
        endtime = self.get_config("endtime")
        freq = pd.to_timedelta(self.get_config("timestepsecs"), unit="s")
        mask = self.grid[self._MAPS["basins"]].values > 0 #geom ipv raster

        precip = self.data_catalog.get_rasterdataset(
            forcing_fn,
            geom=self.region,
            buffer=2,
            time_tuple=(starttime, endtime),
            variables=["precip"],
        )

        if chunksize is not None:
            precip = precip.chunk({"time": chunksize})

        precip_out = hydromt.workflows.forcing.precip(
            precip=precip,
            da_like=self.grid[self._MAPS["elevtn"]],
            freq=freq,
            resample_kwargs=dict(label="right", closed="right"),
            logger=self.logger,
            **kwargs,
        )

        # Update meta attributes (used for default output filename later)
        precip_out.attrs.update({"forcing_fn": forcing_fn})
        self.set_forcing(precip_out.where(mask), name="precip")








    def setup_constant_pars(
        self,
        response_unit: gpd.GeoDataFrame,
        value_name: str,
        value: int or float,
        name: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """Adding a constant value to a response unit (for instance soilgrids).
        
        Parameters
        ----------
        response_unit: pd.GeoDataFrame
            Response unit geometry
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
        ### type checks, copied from set_staticgeoms.
        gtypes = [gpd.GeoDataFrame, gpd.GeoSeries]
        if not np.any([isinstance(response_unit, t) for t in gtypes]):
            raise ValueError("First parameter map(s) should be geopandas.GeoDataFrame or geopandas.GeoSeries")
        
        ### setting single value:    
        if value_name not in response_unit:
            raise ValueError(f"Attribute '{value_name}' not found in GeoDataFrame")
        
        if not (isinstance(value, int) or isinstance(value, float)):
            raise ValueError(f"The assigned value for '{value_name}' must be an integer or float")
        else: 
            response_unit[value_name] = value
        
        return response_unit








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






    # I/O METHODS
    # Write here specific methods to read or write model data components or overwrite the ones from HydroMT CORE

    def read_tables(self, **kwargs):
        """Read table files at <root> and parse to dict of dataframes."""
        """READ OSM LANDUSE TRANSLATION TABLE - copied from wflow"""
        if not self._write:
            self._tables = dict()  # start fresh in read-only mode

        self.logger.info("Reading model table files.")
        fns = glob.glob(join(self.root, "*.csv"))
        if len(fns) > 0:
            for fn in fns:
                name = basename(fn).split(".")[0]
                tbl = pd.read_csv(fn)
                self.set_tables(tbl, name=name)
                #error message if more than 1 file

    def set_tables(self, df, name):
        """Add table <pandas.DataFrame> to model."""
        """ADD OSM LANDUSE TRANSLATION TABLE TO MODEL - copied from wflow"""
        if not (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)):
            raise ValueError("df type not recognized, should be pandas.DataFrame.")
        if name in self._tables:
            if not self._write:
                raise IOError(f"Cannot overwrite table {name} in read-only mode")
            elif self._read:
                self.logger.warning(f"Overwriting table: {name}")
        self._tables[name] = df




    def _configread(self, fn):
        """Read TOML configuration file"""
        with codecs.open(fn, "r", encoding="utf-8") as f:
            fdict = toml.load(f)
        return fdict

    def _configwrite(self, fn):
        """Write TOML configuration file"""
        with codecs.open(fn, "w", encoding="utf-8") as f:
            toml.dump(self.config, f)


    def setup_model_config(self, fn):
        """Update TOML configuration file based on calculations"""
        self.set_config("old", "new") #refer to dictionary created by OSM landuse
        return
        

    # MODEL COMPONENTS AND PROPERTIES
    # Write here specific model properties and components not available in HydroMT CORE
