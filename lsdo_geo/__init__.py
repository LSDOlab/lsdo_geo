from lsdo_geo.core.geometry.geometry import Geometry
from lsdo_geo.core.geometry.geometry_functions import *
from lsdo_geo.core.parameterization.free_form_deformation_functions import *
from lsdo_geo.core.parameterization.ffd_block import FFDBlock
from lsdo_geo.core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization, VolumeSectionalParameterizationInputs
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables


from pathlib import Path

_REPO_ROOT_FOLDER = Path(__file__).parents[0]
IMPORT_FOLDER = _REPO_ROOT_FOLDER / 'core' / 'stored_files' / 'imports'
REFIT_FOLDER = _REPO_ROOT_FOLDER / 'core' / 'stored_files' / 'refits'
PROJECTIONS_FOLDER = _REPO_ROOT_FOLDER / 'core' / 'stored_files' / 'projections'