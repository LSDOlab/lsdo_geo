from dataclasses import dataclass

import numpy as np
import lsdo_geo


@dataclass
class Mesh:
    geometry : lsdo_geo.Geometry
    parametric_coordinates: list[tuple[int,np.ndarray]]
    mesh_counter = 0
    name : str = None

    def __post_init__(self):
        if self.name is None:
            self.name = f'mesh_{Mesh.mesh_counter}'
            Mesh.mesh_counter += 1

    def evaluate(self, geometry:lsdo_geo.Geometry, plot:bool=False):
        '''
        Overload this method with the process to generate the mesh from the parametric coordinates.

        Parameters
        ----------
        geometry : lsdo_geo.Geometry
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
