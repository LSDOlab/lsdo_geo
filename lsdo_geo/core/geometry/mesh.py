from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import csdl_alpha as csdl
import lsdo_geo
import lsdo_function_spaces as lfs

from typing import Union, Optional


@dataclass
class Mesh:
    geometry : lsdo_geo.Geometry
    parametric_coordinates: list[tuple[int,npt.NDArray[np.float64]]]
    mesh_counter = 0
    shape : Optional[tuple[int,...]] = None
    value : Optional[csdl.Variable] = None
    name : Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = f'mesh_{Mesh.mesh_counter}'
            Mesh.mesh_counter += 1
        if self.shape is None:
            if isinstance(self.geometry, lfs.FunctionSet):
                num_physical_dimensions = list(self.geometry.functions.values())[0].coefficients.shape[-1]
                self.shape = (len(self.parametric_coordinates),) + tuple(self.parametric_coordinates[0][1].shape[:-1]) + (num_physical_dimensions,)
            elif isinstance(self.geometry, lfs.Function):
                num_physical_dimensions = self.geometry.coefficients.shape[-1]
                self.shape = tuple(self.parametric_coordinates.shape[:-1]) + (num_physical_dimensions,)

    def evaluate(self, geometry:Union[lsdo_geo.Geometry,lfs.FunctionSet], plot:bool=False) -> csdl.Variable:
        '''
        Overload this method with the process to generate the mesh from the parametric coordinates.

        Parameters
        ----------
        geometry : Union[lsdo_geo.Geometry,lfs.FunctionSet,lfs.Function]
            The geometry object that the mesh will be evaluated on.
        plot : bool = False, optional
            If True, the mesh will be plotted. The default is False.

        Returns
        -------
        mesh : csdl.Variable
            The mesh generated from the parametric coordinates.        
        '''
        self.geometry = geometry
        mesh = self.geometry.evaluate(self.parametric_coordinates, plot=plot)
        return mesh
