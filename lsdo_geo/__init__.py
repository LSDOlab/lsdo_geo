"""
lsdo_geo: Geometry representation, deformational parameterization,
and sensitivity analysis for Multidisciplinary Design Optimization (MDO).
"""

__version__ = "1.0.0"

from .core.geometry.geometry import Geometry
from .core.geometry.mesh import Mesh
from .core.geometry.geometry_functions import (
    import_geometry,
    rotate,
    apply_quaternion_rotation,
    hamilton_product,
)
from .core.parameterization.ffd_block import FFDBlock
from .core.parameterization.free_form_deformation_functions import (
    construct_ffd_block_around_entities,
    construct_ffd_block_from_corners,
    construct_tight_fit_ffd_block,
)
from .core.parameterization.sectional_parameterization import (
    SectionalParameterization,
    SectionalParameters,
)
from .core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs,
)
from .core.parameterization.parameterization_solver import (
    ParameterizationSolver,
    GeometricVariables,
)
from .optimization import (
    Optimization,
    NewtonOptimizer,
)

__all__ = [
    "__version__",
    # Geometry
    "Geometry",
    "Mesh",
    "import_geometry",
    "rotate",
    "apply_quaternion_rotation",
    "hamilton_product",
    # Parameterization
    "FFDBlock",
    "construct_ffd_block_around_entities",
    "construct_ffd_block_from_corners",
    "construct_tight_fit_ffd_block",
    "SectionalParameterization",
    "SectionalParameters",
    "VolumeSectionalParameterization",
    "VolumeSectionalParameterizationInputs",
    "ParameterizationSolver",
    "GeometricVariables",
    # Optimization
    "Optimization",
    "NewtonOptimizer",
]
