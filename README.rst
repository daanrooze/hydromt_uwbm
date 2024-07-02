# HydroMT UWBM: A HydroMT plugin for the Urban Water Balance Model

# Introduction
## What is HydroMT UWBM?
HydroMT (Hydro Model Tools) is an open-source Python package that facilitates the process of building and analyzing spatial geoscientific models with a focus on water system models. It does so by automating the workflow to go from raw data to a complete model instance which is ready to run and to analyze model results once the simulation has finished. This plugin serves to prepare datasets for the `Urban Water Balance Model (UWBM) <https://publicwiki.deltares.nl/display/AST/Urban+Water+balance+model>`_.

## Installation
A base installation of the HydroMT package is required for the UWBM plugin. For more information about the prerequisites for an installation of the HydroMT package and related dependencies, please visit the documentation of `HydroMT core <https://deltares.github.io/hydromt/latest/>`_.

Follow these steps to install the plugin:
1) In the command prompt, navigate to the package root directory (containing the pyproject.toml file).
2) execute "pip install -e ."

## Example usage
Within the REACHOUT project by the European Commission, the CRCTool (with the Urban Water Balance Model as core model) was configured for Athens, Gdynia and Lillestrom as part of the Triple-A Toolbox.