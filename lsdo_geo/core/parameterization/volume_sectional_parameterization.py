"""
Backward-compatibility shim for volume_sectional_parameterization.
This module has been renamed to sectional_parameterization.
"""
import warnings

warnings.warn(
    "lsdo_geo.core.parameterization.volume_sectional_parameterization is deprecated. "
    "Import from lsdo_geo or lsdo_geo.core.parameterization.sectional_parameterization instead.",
    DeprecationWarning,
    stacklevel=2,
)

from .sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs,
    SectionalParameterization,
    SectionalParameters,
)

__all__ = [
    "VolumeSectionalParameterization",
    "VolumeSectionalParameterizationInputs",
    "SectionalParameterization",
    "SectionalParameters",
]
