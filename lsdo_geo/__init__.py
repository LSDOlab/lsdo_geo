from .core.geometry.geometry import Geometry
from .core.geometry.geometry_functions import *
from .core.geometry.mesh import Mesh
from .core.parameterization.free_form_deformation_functions import *
from .core.parameterization.ffd_block import FFDBlock
from .core.parameterization.volume_sectional_parameterization import VolumeSectionalParameterization, VolumeSectionalParameterizationInputs
from .core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables

from pathlib import Path

_REPO_ROOT_FOLDER = Path(__file__).parents[0]
IMPORT_FOLDER = _REPO_ROOT_FOLDER / 'core' / 'stored_files' / 'imports'
REFIT_FOLDER = _REPO_ROOT_FOLDER / 'core' / 'stored_files' / 'refits'
PROJECTIONS_FOLDER = _REPO_ROOT_FOLDER / 'core' / 'stored_files' / 'projections'