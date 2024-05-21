"""HydroMT plugin UWBM: A HydroMT plugin for the Urban Water Balance Model"""

from os.path import dirname, join, abspath

DATADIR = abspath(join(dirname(__file__), "data"))

__version__ = "0.1.0"

from .uwbm import *
