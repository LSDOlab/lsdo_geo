import lsdo_geo
import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs


def import_geometry(file_name:str, name:str='geometry', parallelize:bool=True, scale:int=1.0) -> lsdo_geo.Geometry:
    '''
    Imports geometry from a file.

    Parameters
    ----------
    file_name : str
        The name of the file (with path) that containts the geometric information.
    '''
    
    function_set = lfs.import_file(file_name, parallelize=parallelize)
    if scale != 1.0:
        for function in function_set.functions.values():
            function.coefficients = function.coefficients * scale
    geometry = lsdo_geo.Geometry(functions=function_set.functions, function_names=function_set.function_names, name=name, space=function_set.space)
    return geometry


def rotate(points:csdl.Variable, axis_origin:csdl.Variable, axis_vector:csdl.Variable, angles:csdl.Variable, units:str='radians') -> csdl.Variable:
    points_out_shape = None
    if len(points.shape) == 1:
        print("Rotating points is in vector format, so rotation is assuming 3d and reshaping into (-1,3)")
        points = points.reshape((points.size//3,3))
    if len(points.shape) > 2:
        points_out_shape = points.shape
        points = points.reshape((points.size//points.shape[-1], points.shape[-1]))
        
    if type(points) is np.ndarray:
        points = csdl.Variable(shape=points.shape, value=points)

    if type(axis_origin) is np.ndarray:
        axis_origin = csdl.Variable(shape=axis_origin.shape, value=axis_origin)
    
    if isinstance(axis_vector, np.ndarray):
        axis_vector = csdl.Variable(shape=axis_vector.shape, value=axis_vector)

    if isinstance(angles, (float, int)):
        angles = csdl.Variable(shape=(1,), value=angles)
    elif isinstance(angles, np.ndarray):
        angles = csdl.Variable(shape=angles.shape, value=angles)
    if units == 'degrees':
        angles = angles * np.pi / 180

    points_wrt_axis = points - csdl.expand(axis_origin, points.shape, 'i->ji')

    output_shape = (points.shape[0], 4)

    points_wrt_axis_quaternion = csdl.Variable(shape=output_shape, name='points_wrt_axis_quaternion',
                                               value=0.)
    points_wrt_axis_quaternion = points_wrt_axis_quaternion.set(csdl.slice[:,1:], points_wrt_axis)
    points_wrt_axis_quaternion = points_wrt_axis_quaternion.set(csdl.slice[:,0], 0)

    if angles.shape[0] > 1:
        output_shape = (angles.shape[0],) + output_shape
    else:
        output_shape = (1,) + output_shape

    rotated_points_quaternion = csdl.Variable(shape=output_shape, name='rotated_points', value=0.)
    # for i in csdl.frange(angles.shape[0]):
    for i in range(angles.shape[0]):
        angle = angles[i]

        sin_theta_divide_by_2 = csdl.sin(angle / 2)

        quaternion = csdl.Variable(shape=(4,), value=0.)
        quaternion = quaternion.set(csdl.slice[0], csdl.cos(angle / 2))
        quaternion = quaternion.set(csdl.slice[1], sin_theta_divide_by_2 * axis_vector[0])
        quaternion = quaternion.set(csdl.slice[2], sin_theta_divide_by_2 * axis_vector[1])
        quaternion = quaternion.set(csdl.slice[3], sin_theta_divide_by_2 * axis_vector[2])

        quaternion = quaternion / csdl.norm(quaternion)

        quaternion_conjugate = csdl.Variable(shape=(4,), value=0.)
        quaternion_conjugate = quaternion_conjugate.set(csdl.slice[0], quaternion[0])
        quaternion_conjugate = quaternion_conjugate.set(csdl.slice[1], -quaternion[1])
        quaternion_conjugate = quaternion_conjugate.set(csdl.slice[2], -quaternion[2])
        quaternion_conjugate = quaternion_conjugate.set(csdl.slice[3], -quaternion[3])

        # for j in csdl.frange(points_wrt_axis.shape[0]):
        # for j in range(points_wrt_axis.shape[0]):
            # rotated_points_quaternion = rotated_points_quaternion.set(csdl.slice[i,j,:], 
            #                     hamiltonion_product(quaternion, 
            #                     hamiltonion_product(points_wrt_axis_quaternion[j],
            #                                         quaternion_conjugate)))
            
        rotated_points_quaternion = rotated_points_quaternion.set(csdl.slice[i,:,:], 
                            vectorized_hamiltonion_product_2(quaternion, 
                            vectorized_hamiltonion_product_1(points_wrt_axis_quaternion,
                                                quaternion_conjugate)))

    if angles.shape[0] == 1:
        rotated_points_wrt_axis = rotated_points_quaternion[0,:,1:]
        rotated_points = rotated_points_wrt_axis + csdl.expand(axis_origin, points.shape, 'i->ji')
    else:
        rotated_points_wrt_axis = rotated_points_quaternion[:,:,1:]
        output_shape = (angles.shape[0], points.shape[0], points.shape[1])
        rotated_points = rotated_points_wrt_axis + csdl.expand(axis_origin, output_shape, 'i->kij')

    if points_out_shape is not None:
        rotated_points = rotated_points.reshape(points_out_shape)
    return rotated_points


def vectorized_hamiltonion_product_1(q1:csdl.Variable, q2:csdl.Variable) -> csdl.Variable:
    # q1 = q1.reshape((4,))
    # q2 = q2.reshape((4,))

    q1_0 = q1[:,0]
    q1_1 = q1[:,1]
    q1_2 = q1[:,2]
    q1_3 = q1[:,3]

    q2_0 = q2[0]
    q2_1 = q2[1]
    q2_2 = q2[2]
    q2_3 = q2[3]

    q = csdl.Variable(shape=q1.shape, name='quaternion_product', value=0.)
    q = q.set(csdl.slice[:,0], q1_0*q2_0 - q1_1*q2_1 - q1_2*q2_2 - q1_3*q2_3)
    q = q.set(csdl.slice[:,1], q1_0*q2_1 + q1_1*q2_0 + q1_2*q2_3 - q1_3*q2_2)
    q = q.set(csdl.slice[:,2], q1_0*q2_2 - q1_1*q2_3 + q1_2*q2_0 + q1_3*q2_1)
    q = q.set(csdl.slice[:,3], q1_0*q2_3 + q1_1*q2_2 - q1_2*q2_1 + q1_3*q2_0)

    return q

def vectorized_hamiltonion_product_2(q1:csdl.Variable, q2:csdl.Variable) -> csdl.Variable:
    # q1 = q1.reshape((4,))
    # q2 = q2.reshape((4,))

    q1_0 = q1[0]
    q1_1 = q1[1]
    q1_2 = q1[2]
    q1_3 = q1[3]

    q2_0 = q2[:,0]
    q2_1 = q2[:,1]
    q2_2 = q2[:,2]
    q2_3 = q2[:,3]

    q = csdl.Variable(shape=q2.shape, name='quaternion_product', value=0.)
    q = q.set(csdl.slice[:,0], q1_0*q2_0 - q1_1*q2_1 - q1_2*q2_2 - q1_3*q2_3)
    q = q.set(csdl.slice[:,1], q1_0*q2_1 + q1_1*q2_0 + q1_2*q2_3 - q1_3*q2_2)
    q = q.set(csdl.slice[:,2], q1_0*q2_2 - q1_1*q2_3 + q1_2*q2_0 + q1_3*q2_1)
    q = q.set(csdl.slice[:,3], q1_0*q2_3 + q1_1*q2_2 - q1_2*q2_1 + q1_3*q2_0)

    return q





def hamiltonion_product(q1:csdl.Variable, q2:csdl.Variable) -> csdl.Variable:
    q1 = q1.reshape((4,))
    q2 = q2.reshape((4,))

    q1_0 = q1[0]
    q1_1 = q1[1]
    q1_2 = q1[2]
    q1_3 = q1[3]

    q2_0 = q2[0]
    q2_1 = q2[1]
    q2_2 = q2[2]
    q2_3 = q2[3]

    q = csdl.Variable(shape=(4,), name='quaternion_product', value=0.)
    q = q.set(csdl.slice[0], q1_0*q2_0 - q1_1*q2_1 - q1_2*q2_2 - q1_3*q2_3)
    q = q.set(csdl.slice[1], q1_0*q2_1 + q1_1*q2_0 + q1_2*q2_3 - q1_3*q2_2)
    q = q.set(csdl.slice[2], q1_0*q2_2 - q1_1*q2_3 + q1_2*q2_0 + q1_3*q2_1)
    q = q.set(csdl.slice[3], q1_0*q2_3 + q1_1*q2_2 - q1_2*q2_1 + q1_3*q2_0)

    return q
