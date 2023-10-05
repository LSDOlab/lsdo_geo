'''
This file is for functions associated with the B-splines "Package"
'''

import numpy as np
import scipy.sparse as sps
import re
import pandas as pd

from lsdo_geo.splines.b_splines.b_spline import BSpline
from lsdo_geo.splines.b_splines.b_spline_space import BSplineSpace
from lsdo_geo.splines.b_splines.b_spline_set import BSplineSet
from lsdo_geo.splines.b_splines.b_spline_set_space import BSplineSetSpace

from lsdo_geo.cython.basis_matrix_surface_py import get_basis_surface_matrix
from lsdo_geo.cython.get_open_uniform_py import get_open_uniform


# def create_b_spline_space() # It's not that hard to get the control points shape

def create_b_spline_set(name : str, b_splines : dict[str, BSpline]):
    '''
    Creates a B-spline set from a list of B-splines.

    Parameters
    ----------
    b_splines : dict[str, BSpline]
        A dictionary of B-splines where the key is the name of the B-spline.

    Returns
    -------
    b_spline_set : BSplineSet
        The B-spline set.
    '''
    num_physical_dimensions = {}
    num_coefficients = 0
    for b_spline_name, b_spline in b_splines.items():
        num_coefficients += b_spline.num_coefficients
        num_physical_dimensions[b_spline_name] = b_spline.num_physical_dimensions

    spaces = {}
    b_spline_to_space_dict = {}
    for b_spline_name, b_spline in b_splines.items():
        if type(b_spline) is str:
            raise ValueError('B-spline set cannot be created from a list of strings. Must be a list of B-spline objects.')
        spaces[b_spline.space.name] = b_spline.space
        b_spline_to_space_dict[b_spline.name] = b_spline.space.name

    b_spline_set_space = BSplineSetSpace(name=name, spaces=spaces, b_spline_to_space_dict=b_spline_to_space_dict)

    coefficients = np.zeros((num_coefficients,))
    num_coefficients = 0
    for b_spline_name, b_spline in b_splines.items():
        coefficients[num_coefficients:num_coefficients + b_spline.num_coefficients] = b_spline.coefficients
        num_coefficients += b_spline.num_coefficients

    b_spline_set = BSplineSet(name=name, space=b_spline_set_space, coefficients=coefficients, num_physical_dimensions=num_physical_dimensions)

    return b_spline_set


def import_file(file_name):
    ''' Read file '''
    with open(file_name, 'r') as f:
        print('Importing OpenVSP', file_name)
        if 'B_SPLINE_SURFACE_WITH_KNOTS' not in f.read():
            print("No surfaces found!!")
            print("Something is wrong with the file" \
                , "or this reader doesn't work for this file.")
            return

    '''Stage 1: Parse all information and line numbers for each surface and create B-spline objects'''
    b_splines = {}
    b_spline_spaces = {}
    b_splines_to_spaces_dict = {}
    parsed_info_dict = {}
    with open(file_name, 'r') as f:
        b_spline_surf_info = re.findall(r"B_SPLINE_SURFACE_WITH_KNOTS.*\)", f.read())
        num_surf = len(b_spline_surf_info)
        for i, surf in enumerate(b_spline_surf_info):
            #print(surf)
            # Get numbers following hashes in lines with B_SPLINE... These numbers should only be the line numbers of the cntrl_pts
            info_index = 0
            parsed_info = []
            while(info_index < len(surf)):
                if(surf[info_index]=="("):
                    info_index += 1
                    level_1_array = []
                    while(surf[info_index]!=")"):
                        if(surf[info_index]=="("):
                            info_index += 1
                            level_2_array = []

                            while(surf[info_index]!=")"):
                                if(surf[info_index]=="("):
                                    info_index += 1
                                    nest_level3_start_index = info_index
                                    level_3_array = []
                                    while(surf[info_index]!=")"):
                                        info_index += 1
                                    level_3_array = surf[nest_level3_start_index:info_index].split(', ')
                                    level_2_array.append(level_3_array)
                                    info_index += 1
                                else:
                                    level_2_array.append(surf[info_index])
                                    info_index += 1
                            level_1_array.append(level_2_array)
                            info_index += 1
                        elif(surf[info_index]=="'"):
                            info_index += 1
                            level_2_array = []
                            while(surf[info_index]!="'"):
                                level_2_array.append(surf[info_index])
                                info_index += 1
                            level_2_array = ''.join(level_2_array)
                            level_1_array.append(level_2_array)
                            info_index += 1
                        else:
                            level_1_array.append(surf[info_index])
                            info_index += 1
                    info_index += 1
                else:
                    info_index += 1
            info_index = 0
            last_comma = 1
            while(info_index < len(level_1_array)):
                if(level_1_array[info_index]==","):
                    if(((info_index-1) - last_comma) > 1):
                        parsed_info.append(''.join(level_1_array[(last_comma+1):info_index]))
                    else:
                        parsed_info.append(level_1_array[info_index-1])
                    last_comma = info_index
                elif(info_index==(len(level_1_array)-1)):
                    parsed_info.append(''.join(level_1_array[(last_comma+1):(info_index+1)]))
                info_index += 1

            while "," in parsed_info[3]:
                parsed_info[3].remove(',')
            for j in range(4):
                parsed_info[j+8] = re.findall('\d+' , ''.join(parsed_info[j+8]))
                if j <= 1:
                    info_index = 0
                    for ele in parsed_info[j+8]:
                        parsed_info[j+8][info_index] = int(ele)
                        info_index += 1
                else:
                    info_index = 0
                    for ele in parsed_info[j+8]:
                        parsed_info[j+8][info_index] = float(ele)
                        info_index += 1

            parsed_info[0] = parsed_info[0][17:]+f', {i}'   # Hardcoded 17 to remove useless string
            #print(parsed_info[0])
            knots_u = np.array([parsed_info[10]])
            knots_u = np.repeat(knots_u, parsed_info[8])
            knots_u = knots_u/knots_u[-1]
            knots_v = np.array([parsed_info[11]])
            knots_v = np.repeat(knots_v, parsed_info[9])
            knots_v = knots_v/knots_v[-1]

            order_u = int(parsed_info[1])+1
            order_v = int(parsed_info[2])+1
            space_name = 'order_u_' + str(order_u) + 'order_v_' + str(order_v) + 'knots_u_' + str(knots_u) + 'knots_v_' + str(knots_v)
            if space_name in b_spline_spaces:
                b_spline_space = b_spline_spaces[space_name]
            else:
                coefficients_shape = tuple([len(knots_u)-order_u, len(knots_v)-order_v])
                knots = np.hstack((knots_u, knots_v))
                b_spline_space = BSplineSpace(name=space_name, order=(order_u, order_v), knots=knots, 
                                              parametric_coefficients_shape=coefficients_shape)
                b_spline_spaces[space_name] = b_spline_space

            # b_splines[parsed_info[0]] = BSpline(name=parsed_info[0], space=b_spline_space, coefficients=None, num_physical_dimensions=None)
            b_splines_to_spaces_dict[parsed_info[0]] = space_name

            parsed_info_dict[f'surf{i}_name'] = parsed_info[0]
            parsed_info_dict[f'surf{i}_cp_line_nums'] = np.array(parsed_info[3])
            parsed_info_dict[f'surf{i}_u_multiplicities'] = np.array(parsed_info[8])
            parsed_info_dict[f'surf{i}_v_multiplicities'] = np.array(parsed_info[9])

    ''' Stage 2: Replace line numbers of control points with control points arrays'''

    line_numbs_total_array = np.array([])
    for i in range(num_surf):
        line_numbs_total_array = np.append(line_numbs_total_array, parsed_info_dict[f'surf{i}_cp_line_nums'].flatten())
    point_table = pd.read_csv(file_name, sep='=', names=['lines', 'raw_point'])
    filtered_point_table = point_table.loc[point_table["lines"].isin(line_numbs_total_array)]
    point_table = pd.DataFrame(filtered_point_table['raw_point'].str.findall(r"(-?\d+\.\d*E?-?\d*)").to_list(), columns=['x', 'y', 'z'])
    point_table["lines"] = filtered_point_table["lines"].values

    for i in range(num_surf):
        num_rows_of_cps = parsed_info_dict[f'surf{i}_cp_line_nums'].shape[0]
        num_cp_per_row = parsed_info_dict[f'surf{i}_cp_line_nums'].shape[1]
        cntrl_pts = np.zeros((num_rows_of_cps, num_cp_per_row, 3))
        for j in range(num_rows_of_cps):
            col_cntrl_pts = point_table.loc[point_table["lines"].isin(parsed_info_dict[f'surf{i}_cp_line_nums'][j])][['x', 'y', 'z']]
            if ((len(col_cntrl_pts) != num_cp_per_row) and (len(col_cntrl_pts) != 1)):
                for k in range(num_cp_per_row):
                    cntrl_pts[j,k,:] = point_table.loc[point_table["lines"]==parsed_info_dict[f'surf{i}_cp_line_nums'][j][k]][['x', 'y', 'z']]
            else:
                filtered = False
                cntrl_pts[j,:,:] = col_cntrl_pts

        b_spline_name = parsed_info_dict[f'surf{i}_name']
        # cntrl_pts = np.reshape(cntrl_pts, (num_rows_of_cps*num_cp_per_row,3))
        coefficients = cntrl_pts.reshape((-1,))
        b_spline_space = b_spline_spaces[b_splines_to_spaces_dict[b_spline_name]]
        num_physical_dimensions = cntrl_pts.shape[-1]
        b_spline = BSpline(name=b_spline_name, space=b_spline_space, coefficients=coefficients, num_physical_dimensions=num_physical_dimensions)
        b_splines[b_spline_name] = b_spline
        
    print('Complete import')
    return b_splines


def fit_b_spline(fitting_points:np.ndarray, paramatric_coordinates=None, 
        order:tuple = (4,), num_coefficients:tuple = (10,), knot_vectors = None, name:str = None):
    '''
    Solves P = B*C for C 
    '''
    if len(fitting_points.shape[:-1]) == 1:     # If B-spline curve
        raise Exception("Function not implemented yet for B-spline curves.")
    elif len(fitting_points.shape[:-1]) == 2:   # If B-spline surface
        num_points_u = (fitting_points[:,0,:]).shape[0]
        num_points_v = (fitting_points[0,:,:]).shape[0]
        num_points = num_points_u * num_points_v
        num_physical_dimensions = fitting_points.shape[-1]
        
        if len(order) == 1:
            order_u = order[0]
            order_v = order[0]
        else:
            order_u = order[0]
            order_v = order[1]

        if len(num_coefficients) == 1:
            num_coefficients_u = num_coefficients[0]
            num_coefficients_v = num_coefficients[0]
        else:
            num_coefficients_u = num_coefficients[0]
            num_coefficients_v = num_coefficients[1]

        if paramatric_coordinates is None:
            u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u), np.ones(num_points_v)).flatten()
            v_vec = np.einsum('i,j->ij', np.ones(num_points_u), np.linspace(0., 1., num_points_v)).flatten()
        else:
            u_vec = paramatric_coordinates[0]
            v_vec = paramatric_coordinates[1]

        if knot_vectors is None:
            knot_vector_u = np.zeros(num_coefficients_u+order_u)
            knot_vector_v = np.zeros(num_coefficients_v+order_v)
            get_open_uniform(order_u, num_coefficients_u, knot_vector_u)
            get_open_uniform(order_v, num_coefficients_v, knot_vector_v)
        else:
            knot_vector_u = knot_vectors[0]
            knot_vector_v = knot_vectors[1]

        evaluation_matrix = construct_evaluation_matrix(parametric_coordinates=(u_vec, v_vec),  order=(order_u, order_v),
            num_coefficients=(num_coefficients_u, num_coefficients_v), knot_vectors = knot_vectors)

        flattened_fitting_points_shape = tuple((num_points, num_physical_dimensions))
        flattened_fitting_points = fitting_points.reshape(flattened_fitting_points_shape)

        # Perform fitting
        a = ((evaluation_matrix.T).dot(evaluation_matrix)).toarray()
        if np.linalg.det(a) == 0:
            cps_fitted,_,_,_ = np.linalg.lstsq(a, evaluation_matrix.T.dot(flattened_fitting_points), rcond=None)
        else: 
            cps_fitted = np.linalg.solve(a, evaluation_matrix.T.dot(flattened_fitting_points))            

        space_name = 'order_u_' + str(order_u) + 'order_v_' + str(order_v) + 'knots_u_' + str(knot_vector_u) + 'knots_v_' + str(knot_vector_v)

        knots = np.hstack((knot_vector_u, knot_vector_v))
        b_spline_space = BSplineSpace(name=space_name, order=(order_u, order_v), knots=knots, 
            parametric_coefficients_shape=(num_coefficients_u, num_coefficients_v))

        b_spline = BSpline(
            name=name,
            space=b_spline_space,
            coefficients=np.array(cps_fitted).reshape((-1,)),
            num_physical_dimensions=num_physical_dimensions
        )

        return b_spline
    
    elif len(fitting_points.shape[:-1]) == 3:     # If B-spline volume
        raise Exception("Function not implemented yet for B-spline volumes.")
    else:
        raise Exception("Function not implemented yet for B-spline hyper-volumes.")


'''
Evaluate a B-spline and refit it with the desired parameters.
'''
def refit_b_spline(b_spline, order : tuple = (4,), num_coefficients : tuple = (10,), fit_resolution : tuple = (30,), name=None):
    # TODO allow it to be curves or volumes too
    if len(b_spline.shape[:-1]) == 0:  # is point
        print('fitting points has not been implemented yet for points.')
        pass        #is point
    elif len(b_spline.shape[:-1]) == 1:  # is curve
        print('fitting curves has not been implemented yet for B-spline curves.')
        pass 
    elif len(b_spline.shape[:-1]) == 2:
        if len(order) == 1:
            order_u = order[0]
            order_v = order[0]
        else:
            order_u = order[0]
            order_v = order[1]

        if len(num_coefficients) == 1:
            num_coefficients_u = num_coefficients[0]
            num_coefficients_v = num_coefficients[0]
        else:
            num_coefficients_u = num_coefficients[0]
            num_coefficients_v = num_coefficients[1]
        
        if len(fit_resolution) == 1:
            num_points_u = fit_resolution[0]
            num_points_v = fit_resolution[0]
        else:
            num_points_u = fit_resolution[0]
            num_points_v = fit_resolution[1]

        num_dimensions = b_spline.coefficients.shape[-1]   # usually 3 for 3D space

        u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u), np.ones(num_points_v)).flatten()
        v_vec = np.einsum('i,j->ij', np.ones(num_points_u), np.linspace(0., 1., num_points_v)).flatten()

        points_vector = b_spline.evaluate_points(u_vec, v_vec)
        points = points_vector.reshape((num_points_u, num_points_v, num_dimensions))
        
        b_spline = fit_b_spline(fitting_points=points, paramatric_coordinates=(u_vec, v_vec), 
            order=(order_u,order_v), num_coefficients=(num_coefficients_u,num_coefficients_v),
            knot_vectors=None, name=name)

        return b_spline
        
        # b_spline_entity_surface.starting_geometry_index = self.num_coefficients       # TODO figure out indexing
        self.input_b_spline_entity_dict[b_spline.name] = b_spline_entity_surface
        self.b_spline_coefficients_indices[b_spline.name] = np.arange(self.num_coefficients, self.num_coefficients+np.cumprod(b_spline_entity_surface.shape[:-1])[-1])
        self.num_coefficients += np.cumprod(b_spline_entity_surface.shape)[-2]

    elif len(b_spline.shape[:-1]) == 3:  # is volume
        print('fitting BSplineVolume has not been implemented yet for B-spline volumes.')
        pass
    else:
        raise Exception("Function not implemented yet for B-spline hyper-volumes.")
    return


def refit_b_spline_set(b_spline_set:BSplineSet, num_coefficients:tuple=(25,25), fit_resolution:tuple=(50,50), order:tuple=(4,4)):
    '''
    Evaluates a grid over the B-spline set and finds the best set of coefficients/control points at the desired resolution to fit the B-spline set.

    Parameters
    ----------
    num_coefficients : tuple, optional
        The number of coefficients to use in each direction.
    fit_resolution : tuple, optional
        The number of points to evaluate in each direction for each B-spline to fit the B-spline set.
    order : tuple, optional
        The order of the B-splines to use in each direction.
    '''
    b_splines = {}
    for b_spline_name, indices in b_spline_set.coefficient_indices.items():
        if type(num_coefficients) is int:
            num_coefficients = (num_coefficients, num_coefficients)
        elif len(num_coefficients) == 1:
            num_coefficients = (num_coefficients[0], num_coefficients[0])

        if type(fit_resolution) is int:
            fit_resolution = (fit_resolution, fit_resolution)
        elif len(fit_resolution) == 1:
            fit_resolution = (fit_resolution[0], fit_resolution[0])

        if type(order) is int:
            order = (order, order)
        elif len(order) == 1:
            order = (order[0], order[0])

        num_points_u = fit_resolution[0]
        num_points_v = fit_resolution[1]

        num_dimensions = b_spline_set.num_physical_dimensions[b_spline_name]
        u_vec = np.einsum('i,j->ij', np.linspace(0., 1., num_points_u), np.ones(num_points_v)).flatten()
        v_vec = np.einsum('i,j->ij', np.ones(num_points_u), np.linspace(0., 1., num_points_v)).flatten()
        parametric_coordinates = np.zeros((num_points_u*num_points_v, 2))
        parametric_coordinates[:,0] = u_vec.copy()
        parametric_coordinates[:,1] = v_vec.copy()
        points_vector = b_spline_set.evaluate(b_spline_name=b_spline_name, parametric_coordinates=parametric_coordinates)
        points = points_vector.reshape((num_points_u, num_points_v, num_dimensions))

        b_spline = fit_b_spline(fitting_points=points, paramatric_coordinates=(u_vec, v_vec), 
            order=order, num_coefficients=num_coefficients,
            knot_vectors=None, name=b_spline_name)
        
        b_splines[b_spline_name] = b_spline

    new_b_spline_set = create_b_spline_set(name=b_spline_set.name, b_splines=b_splines)
        
    return new_b_spline_set

'''
Construct B-spline evaluation matrix
'''
def construct_evaluation_matrix(parametric_coordinates:tuple,  order:tuple, num_coefficients:tuple, knot_vectors = None):
    num_points = len(parametric_coordinates[0])

    if len(parametric_coordinates) == 1:
        raise Exception("Function not implemented yet.")
    elif len(parametric_coordinates) == 2:
        order_u = order[0]
        order_v = order[1]
        num_coefficients_u = num_coefficients[0]
        num_coefficients_v = num_coefficients[1]
        u_vec = parametric_coordinates[0]
        v_vec = parametric_coordinates[1]

        if knot_vectors is None:
            knot_vector_u = np.zeros(num_coefficients_u+order_u)
            knot_vector_v = np.zeros(num_coefficients_v+order_v)
            get_open_uniform(order_u, num_coefficients_u, knot_vector_u)
            get_open_uniform(order_v, num_coefficients_v, knot_vector_v)
        else:
            knot_vector_u = knot_vectors[0]
            knot_vector_v = knot_vectors[1]

        nnz = num_points * order_u * order_v
        data = np.zeros(nnz)
        row_indices = np.zeros(nnz, np.int32)
        col_indices = np.zeros(nnz, np.int32)
        get_basis_surface_matrix(
            order_u, num_coefficients_u, 0, u_vec, knot_vector_u,
            order_v, num_coefficients_v, 0, v_vec, knot_vector_v,
            num_points, data, row_indices, col_indices,
        )

        basis0 = sps.csc_matrix(
            (data, (row_indices, col_indices)), 
            shape=(num_points, num_coefficients_u * num_coefficients_v),
        )
        return basis0

    elif len(parametric_coordinates.shape) == 3:
        raise Exception("Function not implemented yet.")
    else:
        raise Exception("Function not implemented yet")


def generate_open_uniform_knot_vector(num_coefficients, order):
    knot_vector = np.zeros(num_coefficients+order)
    get_open_uniform(order, num_coefficients, knot_vector)
    return knot_vector


def create_b_spline_from_corners(name:str, corners:np.ndarray, order:tuple=(4,), num_coefficients:tuple=(10,), knot_vectors:tuple=None):
    num_dimensions = len(corners.shape)-1
    
    if len(order) != num_dimensions:
        order = tuple(np.tile(order, num_dimensions))
    if len(num_coefficients) != num_dimensions:
        num_coefficients = tuple(np.tile(num_coefficients, num_dimensions))
    if knot_vectors is not None:
        if len(knot_vectors) != num_dimensions:
            knot_vectors = tuple(np.tile(knot_vectors, num_dimensions))
    else:
        knot_vectors = tuple()
        for dimension_index in range(num_dimensions):
            knot_vectors = knot_vectors + \
                    tuple(generate_open_uniform_knot_vector(num_coefficients[dimension_index], order[dimension_index]),)

    # Build up hyper-volume based on corners given
    previous_dimension_hyper_volume = corners
    dimension_hyper_volumes = corners.copy()
    for dimension_index in np.arange(num_dimensions, 0, -1)-1:
        dimension_hyper_volumes_shape = np.array(previous_dimension_hyper_volume.shape)
        dimension_hyper_volumes_shape[dimension_index] = (dimension_hyper_volumes_shape[dimension_index]-1) * num_coefficients[dimension_index]
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
            linspace_index_front[dimension_level_index*num_coefficients[dimension_index]:(dimension_level_index+1)*num_coefficients[dimension_index]] = \
                    np.linspace(previous_index_front[dimension_level_index], previous_index_front[dimension_level_index+1],
                            num_coefficients[dimension_index], endpoint=include_endpoint)
        # Move axis back to proper location
        dimension_hyper_volumes = np.moveaxis(linspace_index_front, 0, dimension_index)
        previous_dimension_hyper_volume = dimension_hyper_volumes.copy()

    b_spline_space = BSplineSpace(name=name+'_space', order=order, knots=knot_vectors, 
                                  parametric_coefficients_shape=dimension_hyper_volumes.shape[:-1])
    return BSpline(name=name, space=b_spline_space, coefficients=dimension_hyper_volumes, num_physical_dimensions=corners.shape[-1])
    # if num_dimensions == 1:
    #     return BSplineCurve(name=name, coefficients=dimension_hyper_volumes, order_u=order[0], knots_u=knot_vectors[0])
    # elif num_dimensions == 2:
    #     return BSplineSurface(name=name, order_u=order[0], order_v=order[1], knots_u=knot_vectors[0], knots_v=knot_vectors[1],
    #             shape=dimension_hyper_volumes.shape, coefficients=dimension_hyper_volumes)
    # elif num_dimensions == 3:
    #     return BSplineVolume(name=name, order_u=order[0], order_v=order[1], order_w=order[2],
    #             knots_u=knot_vectors[0], knots_v=knot_vectors[1], knots_w=knot_vectors[2],
    #             shape=dimension_hyper_volumes.shape, coefficients=dimension_hyper_volumes)


def create_cartesian_enclosure_block(name:str, points:np.ndarray, num_coefficients:tuple, order:tuple, knot_vectors:tuple=None, 
                                      num_parametric_dimensions:int=3, num_physical_dimensions:int=3) -> BSpline:
    '''
    Creates an nd volume that tightly fits around a set of entities.
    '''
    if type(num_coefficients) is int:
        num_coefficients = (num_coefficients,)*num_parametric_dimensions
    if type(order) is int:
        order = (order,)*num_parametric_dimensions

    if knot_vectors is None:
        knot_vectors=()
        for i in range(len(num_coefficients)):
            axis_knots = generate_open_uniform_knot_vector(num_coefficients[i], order[i])
            knot_vectors = knot_vectors + (axis_knots,)      # append knots

    mins = np.min(points.reshape((-1,num_physical_dimensions)), axis=0).reshape((-1,1))
    maxs = np.max(points.reshape((-1,num_physical_dimensions)), axis=0).reshape((-1,1))

    mins_and_maxs = np.hstack((mins, maxs))

    # if abs(maxs[0]-mins[0]) < 1e-6:
    #     maxs[0] += 1e-6
    # if abs(maxs[1]-mins[1]) < 1e-6:
    #     maxs[1] += 1e-6
    # if abs(maxs[2]-mins[2]) < 1e-6:
    #     maxs[2] += 1e-6


    # Can probably automate this to nd using a for loop
    corners_shape = (2,)*num_parametric_dimensions + (num_physical_dimensions,)
    corners = np.zeros(corners_shape)
    corners_flattened = np.zeros((np.prod(corners_shape),))
    physical_dimension_index = 0
    for i in range(len(corners_flattened)):
        parametric_dimension_counter = int(i/num_physical_dimensions)
        binary_parametric_dimension_counter = bin(parametric_dimension_counter)[2:].zfill(num_physical_dimensions)
        min_or_max = int(binary_parametric_dimension_counter[physical_dimension_index])
        corners_flattened[i] = mins_and_maxs[physical_dimension_index, min_or_max]

        physical_dimension_index += 1
        if physical_dimension_index == num_physical_dimensions:
            physical_dimension_index = 0

    corners = corners_flattened.reshape(corners_shape)

    # points[0,0,0,:] = np.array([mins[0], mins[1], mins[2]])
    # points[0,0,1,:] = np.array([mins[0], mins[1], maxs[2]])
    # points[0,1,0,:] = np.array([mins[0], maxs[1], mins[2]])
    # points[0,1,1,:] = np.array([mins[0], maxs[1], maxs[2]])
    # points[1,0,0,:] = np.array([maxs[0], mins[1], mins[2]])
    # points[1,0,1,:] = np.array([maxs[0], mins[1], maxs[2]])
    # points[1,1,0,:] = np.array([maxs[0], maxs[1], mins[2]])
    # points[1,1,1,:] = np.array([maxs[0], maxs[1], maxs[2]])

    hyper_volume = create_b_spline_from_corners(name=name, corners=corners, order=order, num_coefficients=num_coefficients,
            knot_vectors=knot_vectors)

    return hyper_volume

