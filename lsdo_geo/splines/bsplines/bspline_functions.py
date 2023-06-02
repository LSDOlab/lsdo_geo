'''
This file is for functions associated with the B-splines "Package"
'''

import numpy as np
import scipy.sparse as sps

from caddee.primitives.bsplines.bspline_curve import BSplineCurve
from caddee.primitives.bsplines.bspline_surface import BSplineSurface
from caddee.primitives.bsplines.bspline_volume import BSplineVolume

from caddee.cython.basis_matrix_surface_py import get_basis_surface_matrix
from caddee.cython.get_open_uniform_py import get_open_uniform


'''
Solves P = B*C for C 
'''
def fit_bspline(fitting_points:np.ndarray, paramatric_coordinates=None, 
        order:tuple = (4,), num_control_points:tuple = (10,), knot_vectors = None, name:str = None):
    if len(fitting_points.shape[:-1]) == 1:     # If B-spline curve
        raise Exception("Function not implemented yet for B-spline curves.")
    elif len(fitting_points.shape[:-1]) == 2:   # If B-spline surface
        num_points_u = (fitting_points[:,0,:]).shape[0]
        num_points_v = (fitting_points[0,:,:]).shape[0]
        num_points = num_points_u * num_points_v
        space_dimension = fitting_points.shape[-1]
        
        if len(order) == 1:
            order_u = order[0]
            order_v = order[0]
        else:
            order_u = order[0]
            order_v = order[1]

        if len(num_control_points) == 1:
            num_control_points_u = num_control_points[0]
            num_control_points_v = num_control_points[0]
        else:
            num_control_points_u = num_control_points[0]
            num_control_points_v = num_control_points[1]

        if paramatric_coordinates is None:
            u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u), np.ones(num_points_v)).flatten()
            v_vec = np.einsum('i,j->ij', np.ones(num_points_u), np.linspace(0., 1., num_points_v)).flatten()
        else:
            u_vec = paramatric_coordinates[0]
            v_vec = paramatric_coordinates[1]

        if knot_vectors is None:
            knot_vector_u = np.zeros(num_control_points_u+order_u)
            knot_vector_v = np.zeros(num_control_points_v+order_v)
            get_open_uniform(order_u, num_control_points_u, knot_vector_u)
            get_open_uniform(order_v, num_control_points_v, knot_vector_v)
        else:
            knot_vector_u = knot_vectors[0]
            knot_vector_v = knot_vectors[1]

        evaluation_matrix = construct_evaluation_matrix(parametric_coordinates=(u_vec, v_vec),  order=(order_u, order_v),
            num_control_points=(num_control_points_u, num_control_points_v), knot_vectors = knot_vectors)

        flattened_fitting_points_shape = tuple((num_points, space_dimension))
        flattened_fitting_points = fitting_points.reshape(flattened_fitting_points_shape)

        # Perform fitting
        a = ((evaluation_matrix.T).dot(evaluation_matrix)).toarray()
        if np.linalg.det(a) == 0:
            cps_fitted,_,_,_ = np.linalg.lstsq(a, evaluation_matrix.T.dot(flattened_fitting_points), rcond=None)
        else: 
            cps_fitted = np.linalg.solve(a, evaluation_matrix.T.dot(flattened_fitting_points))            

        bspline = BSplineSurface(
            name=name,
            order_u=order_u,
            order_v=order_v,
            shape=np.array([num_control_points_u, num_control_points_v, space_dimension]),
            control_points=np.array(cps_fitted).reshape((num_control_points_u,num_control_points_v,space_dimension)),
            knots_u=np.array(knot_vector_u),
            knots_v=np.array(knot_vector_v))

        return bspline
    
    elif len(fitting_points.shape[:-1]) == 3:     # If B-spline volume
        raise Exception("Function not implemented yet for B-spline volumes.")
    else:
        raise Exception("Function not implemented yet for B-spline hyper-volumes.")


'''
Evaluate a B-spline and refit it with the desired parameters.
'''
def refit_bspline(bspline, order : tuple = (4,), num_control_points : tuple = (10,), fit_resolution : tuple = (30,), name=None):
    # TODO allow it to be curves or volumes too
    if len(bspline.shape[:-1]) == 0:  # is point
        print('fitting points has not been implemented yet for points.')
        pass        #is point
    elif len(bspline.shape[:-1]) == 1:  # is curve
        print('fitting curves has not been implemented yet for B-spline curves.')
        pass 
    elif len(bspline.shape[:-1]) == 2:
        if len(order) == 1:
            order_u = order[0]
            order_v = order[0]
        else:
            order_u = order[0]
            order_v = order[1]

        if len(num_control_points) == 1:
            num_control_points_u = num_control_points[0]
            num_control_points_v = num_control_points[0]
        else:
            num_control_points_u = num_control_points[0]
            num_control_points_v = num_control_points[1]
        
        if len(fit_resolution) == 1:
            num_points_u = fit_resolution[0]
            num_points_v = fit_resolution[0]
        else:
            num_points_u = fit_resolution[0]
            num_points_v = fit_resolution[1]

        num_dimensions = bspline.control_points.shape[-1]   # usually 3 for 3D space

        u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u), np.ones(num_points_v)).flatten()
        v_vec = np.einsum('i,j->ij', np.ones(num_points_u), np.linspace(0., 1., num_points_v)).flatten()

        points_vector = bspline.evaluate_points(u_vec, v_vec)
        points = points_vector.reshape((num_points_u, num_points_v, num_dimensions))
        
        bspline = fit_bspline(fitting_points=points, paramatric_coordinates=(u_vec, v_vec), 
            order=(order_u,order_v), num_control_points=(num_control_points_u,num_control_points_v),
            knot_vectors=None, name=name)

        return bspline
        
        # bspline_entity_surface.starting_geometry_index = self.num_control_points       # TODO figure out indexing
        self.input_bspline_entity_dict[bspline.name] = bspline_entity_surface
        self.bspline_control_points_indices[bspline.name] = np.arange(self.num_control_points, self.num_control_points+np.cumprod(bspline_entity_surface.shape[:-1])[-1])
        self.num_control_points += np.cumprod(bspline_entity_surface.shape)[-2]

    elif len(bspline.shape[:-1]) == 3:  # is volume
        print('fitting BSplineVolume has not been implemented yet for B-spline volumes.')
        pass
    else:
        raise Exception("Function not implemented yet for B-spline hyper-volumes.")
    return


'''
Construct B-spline evaluation matrix
'''
def construct_evaluation_matrix(parametric_coordinates:tuple,  order:tuple, num_control_points:tuple, knot_vectors = None):
    num_points = len(parametric_coordinates[0])

    if len(parametric_coordinates) == 1:
        raise Exception("Function not implemented yet.")
    elif len(parametric_coordinates) == 2:
        order_u = order[0]
        order_v = order[1]
        num_control_points_u = num_control_points[0]
        num_control_points_v = num_control_points[1]
        u_vec = parametric_coordinates[0]
        v_vec = parametric_coordinates[1]

        if knot_vectors is None:
            knot_vector_u = np.zeros(num_control_points_u+order_u)
            knot_vector_v = np.zeros(num_control_points_v+order_v)
            get_open_uniform(order_u, num_control_points_u, knot_vector_u)
            get_open_uniform(order_v, num_control_points_v, knot_vector_v)
        else:
            knot_vector_u = knot_vectors[0]
            knot_vector_v = knot_vectors[1]

        nnz = num_points * order_u * order_v
        data = np.zeros(nnz)
        row_indices = np.zeros(nnz, np.int32)
        col_indices = np.zeros(nnz, np.int32)
        get_basis_surface_matrix(
            order_u, num_control_points_u, 0, u_vec, knot_vector_u,
            order_v, num_control_points_v, 0, v_vec, knot_vector_v,
            num_points, data, row_indices, col_indices,
        )

        basis0 = sps.csc_matrix(
            (data, (row_indices, col_indices)), 
            shape=(num_points, num_control_points_u * num_control_points_v),
        )
        return basis0

    elif len(parametric_coordinates.shape) == 3:
        raise Exception("Function not implemented yet.")
    else:
        raise Exception("Function not implemented yet")


def generate_open_uniform_knot_vector(num_control_points, order):
    knot_vector = np.zeros(num_control_points+order)
    get_open_uniform(order, num_control_points, knot_vector)
    return knot_vector


def create_bspline_from_corners(corners:np.ndarray, order:tuple=(4,), num_control_points:tuple=(10,), knot_vectors:tuple=None, name:str = None):
    num_dimensions = len(corners.shape)-1
    
    if len(order) != num_dimensions:
        order = tuple(np.tile(order, num_dimensions))
    if len(num_control_points) != num_dimensions:
        num_control_points = tuple(np.tile(num_control_points, num_dimensions))
    if knot_vectors is not None:
        if len(knot_vectors) != num_dimensions:
            knot_vectors = tuple(np.tile(knot_vectors, num_dimensions))
    else:
        knot_vectors = tuple()
        for dimension_index in range(num_dimensions):
            knot_vectors = knot_vectors + \
                    tuple(generate_open_uniform_knot_vector(num_control_points[dimension_index], order[dimension_index]),)

    # Build up hyper-volume based on corners given
    previous_dimension_hyper_volume = corners
    dimension_hyper_volumes = corners.copy()
    for dimension_index in np.arange(num_dimensions, 0, -1)-1:
        dimension_hyper_volumes_shape = np.array(previous_dimension_hyper_volume.shape)
        dimension_hyper_volumes_shape[dimension_index] = (dimension_hyper_volumes_shape[dimension_index]-1) * num_control_points[dimension_index]
        dimension_hyper_volumes_shape = tuple(dimension_hyper_volumes_shape)
        dimension_hyper_volumes = np.zeros(dimension_hyper_volumes_shape)

        # Move dimension index to front so we can index the correct dimension
        linspace_index_front = np.moveaxis(dimension_hyper_volumes, dimension_index, 0)
        previous_index_front = np.moveaxis(previous_dimension_hyper_volume, dimension_index, 0)
        include_endpoint = False
        # Perform interpolations
        for dimension_level_index in range(previous_dimension_hyper_volume.shape[dimension_index]-1):
            if dimension_level_index == previous_dimension_hyper_volume.shape[dimension_index]-2:
                include_endpoint = True
            linspace_index_front[dimension_level_index*num_control_points[dimension_index]:(dimension_level_index+1)*num_control_points[dimension_index]] = \
                    np.linspace(previous_index_front[dimension_level_index], previous_index_front[dimension_level_index+1],
                            num_control_points[dimension_index], endpoint=include_endpoint)
        # Move axis back to proper location
        dimension_hyper_volumes = np.moveaxis(linspace_index_front, 0, dimension_index)
        previous_dimension_hyper_volume = dimension_hyper_volumes.copy()

    # return BSpline()    # ndbspline
    if num_dimensions == 1:
        return BSplineCurve(name=name, control_points=dimension_hyper_volumes, order_u=order[0], knots_u=knot_vectors[0])
    elif num_dimensions == 2:
        return BSplineSurface(name=name, order_u=order[0], order_v=order[1], knots_u=knot_vectors[0], knots_v=knot_vectors[1],
                shape=dimension_hyper_volumes.shape, control_points=dimension_hyper_volumes)
    elif num_dimensions == 3:
        return BSplineVolume(name=name, order_u=order[0], order_v=order[1], order_w=order[2],
                knots_u=knot_vectors[0], knots_v=knot_vectors[1], knots_w=knot_vectors[2],
                shape=dimension_hyper_volumes.shape, control_points=dimension_hyper_volumes)

