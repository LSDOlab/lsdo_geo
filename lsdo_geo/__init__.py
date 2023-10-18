from lsdo_geo.splines.b_splines.b_spline_set_space import BSplineSetSpace
from lsdo_geo.splines.b_splines.b_spline_space import BSplineSpace
from lsdo_geo.splines.b_splines.b_spline_set import BSplineSet
from lsdo_geo.splines.b_splines.b_spline_space import BSplineSpace

from lsdo_geo.core.geometry.geometry import Geometry
from lsdo_geo.core.geometry.geometry_functions import *
from lsdo_geo.core.parameterization.free_form_deformation_functions import *

from pathlib import Path

_REPO_ROOT_FOLDER = Path(__file__).parents[0]
IMPORT_FOLDER = _REPO_ROOT_FOLDER / 'core' / 'stored_files' / 'imports'
REFIT_FOLDER = _REPO_ROOT_FOLDER / 'core' / 'stored_files' / 'refits'
PROJECTIONS_FOLDER = _REPO_ROOT_FOLDER / 'core' / 'stored_files' / 'projections'