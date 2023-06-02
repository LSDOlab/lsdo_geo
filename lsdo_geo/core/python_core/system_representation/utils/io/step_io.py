import pandas as pd
import re#
import regex# as re
import numpy as np
import copy
from vedo import Points, Plotter, colors, LegendBox, show

from caddee.primitives.bsplines.bspline_surface import BSplineSurface
import caddee.primitives.bsplines.bspline_functions as bsp

import io

'''
TODO Move these functions to the Bspline package!!
'''

def read_gmsh_stp(geo, file_name):
    ''' Read file '''
    with open(file_name, 'r') as f:
        print('Importing Gmsh', file_name)
        if 'B_SPLINE_SURFACE_WITH_KNOTS' not in f.read():
            print("No knot surfaces found!!")
            print("Something is wrong with the file" \
                , "or this reader doesn't work for this file.")
            return
    '''Stage 1: Parse all information and line numbers for each surface'''
    parsed_info_dict = {}
    with open(file_name, 'r') as f:
        b_spline_surf_info_initial = re.findall(r'B_SPLINE_SURFACE_WITH_KNOTS[\s\S]*?(?=;)', f.read())#"B_SPLINE_SURFACE_WITH_KNOTS(.*)"B_SPLINE_SURFACE_WITH_KNOTS[\s\S]*.\);
        num_surf = len(b_spline_surf_info_initial)
        print('num_surf',num_surf)
        #print(b_spline_surf_info_initial[0])
        b_spline_surf_info = []
        for string in b_spline_surf_info_initial:
            string = string.split()#'\n'
            string = ''.join(string)
            string = string.replace(',#',', #')
            b_spline_surf_info.append(string)
        #print(b_spline_surf_info[0])
        #re.sub(r',#', r',1', b_spline_surf_info[0])
        #print(re.findall(r',#', b_spline_surf_info[0]))
        #print(re.sub(r"\n","",b_spline_surf_info_initial[0]))

        for i, surf in enumerate(b_spline_surf_info):
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
            for j in range(4): #r"\d+\.?\d*" ##r"-?\d+\.\d*E?-?\+?\d*"
                parsed_info[j+8] = re.findall( r"\d+\.?\d*E?-?\+?\d*", ''.join(parsed_info[j+8]))
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

            parsed_info[0] = 'Surface'+ parsed_info[0][17:]+f', {i}'   # Hardcoded 17 to remove useless string

            knots_u = np.array([parsed_info[10]])
            knots_u = np.repeat(knots_u, parsed_info[8])

            knots_u = knots_u/knots_u[-1]
            knots_v = np.array([parsed_info[11]])
            knots_v = np.repeat(knots_v, parsed_info[9])
            knots_v = knots_v/knots_v[-1]

            geo.primitives[parsed_info[0]] = (BSplineSurface(
                name=parsed_info[0],
                order_u=int(parsed_info[1])+1,
                order_v=int(parsed_info[2])+1,
                shape=None,
                control_points=None,
                knots_u=knots_u,
                knots_v=knots_v))

            parsed_info_dict[f'surf{i}_name'] = parsed_info[0]
            parsed_info_dict[f'surf{i}_cp_line_nums'] = np.array(parsed_info[3])
            parsed_info_dict[f'surf{i}_u_multiplicities'] = np.array(parsed_info[8])
            parsed_info_dict[f'surf{i}_v_multiplicities'] = np.array(parsed_info[9])
        #print('finished')
    ''' Stage 2: Replace line numbers of control points with control points arrays'''

    line_numbs_total_array = np.array([])
    for i in range(num_surf):
        line_numbs_total_array = np.append(line_numbs_total_array, parsed_info_dict[f'surf{i}_cp_line_nums'].flatten())
    with open(file_name, 'r') as f:
        point_info_initial = re.findall(r'#[\s\S]*?(?=\);)', f.read())#"B_SPLINE_SURFACE_WITH_KNOTS(.*)"B_SPLINE_SURFACE_WITH_KNOTS[\s\S]*.\);
        point_info = []
        for string in point_info_initial:
            string = string.split()#'\n'
            string = ''.join(string)
            point_info.append(string)
    map(float, point_info)
    point_info_string = "\n".join(point_info)
    point_table = pd.read_csv(io.StringIO(point_info_string), sep='=', names=['lines', 'raw_point'])
    filtered_point_table = point_table.loc[point_table["lines"].isin(line_numbs_total_array)]
    point_table = pd.DataFrame(filtered_point_table['raw_point'].str.findall(r"-?\d+\.\d*E?-?\+?\d*").to_list(), columns=['x', 'y', 'z'])
    point_table["lines"] = filtered_point_table["lines"].values
    geo.initial_input_bspline_entity_dict = copy.deepcopy(geo.primitives)
    initial_surfaces = []
    for i in range(num_surf):
        num_rows_of_cps = parsed_info_dict[f'surf{i}_cp_line_nums'].shape[0]
        num_cp_per_row = parsed_info_dict[f'surf{i}_cp_line_nums'].shape[1]
        #print(i, num_rows_of_cps, num_cp_per_row)
        cntrl_pts = np.zeros((num_rows_of_cps, num_cp_per_row, 3))
        for j in range(num_rows_of_cps):
            col_cntrl_pts = point_table.loc[point_table["lines"].isin(parsed_info_dict[f'surf{i}_cp_line_nums'][j])][['x', 'y', 'z']]
            if ((len(col_cntrl_pts) != num_cp_per_row) and (len(col_cntrl_pts) != 1)):
                #print('SKIPPED SURFACES: ', parsed_info_dict[f'surf{i}_name'])
                # geo.initial_input_bspline_entity_dict.pop(f'surf{i}_name', None)
                # geo.primitives.pop(f'surf{i}_name', None)
                # filtered = True
                # continue
                for k in range(num_cp_per_row):
                    cntrl_pts[j,k,:] = point_table.loc[point_table["lines"]==parsed_info_dict[f'surf{i}_cp_line_nums'][j][k]][['x', 'y', 'z']]
            else:
                filtered = False
                cntrl_pts[j,:,:] = col_cntrl_pts

        # print('Control Points shape: ', cntrl_pts.shape)
        geo.primitives[parsed_info_dict[f'surf{i}_name']].shape = np.array(cntrl_pts.shape)
        geo.initial_input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']].shape = np.array(cntrl_pts.shape)
        # print('NUMBER OF CONTROL POINTS IMPORT: ', num_cp)
        # geo.primitives[parsed_info_dict[f'surf{i}_name']].starting_geometry_index = num_cp
        # geo.initial_input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']].starting_geometry_index = num_cp
        cntrl_pts = np.reshape(cntrl_pts, (num_rows_of_cps*num_cp_per_row,3))     
        geo.primitives[parsed_info_dict[f'surf{i}_name']].control_points = cntrl_pts
        geo.initial_input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']].control_points = cntrl_pts
        # print('Number of rows: ', num_rows_of_cps)
        # print('Number of cp per row: ', num_cp_per_row)
        # num_cp += num_rows_of_cps * num_cp_per_row
        #initial = geo.primitives[parsed_info_dict[f'surf{i}_name']]
        #print(geo.primitives[parsed_info_dict[f'surf{i}_name']])
        #print(parsed_info_dict[f'surf{i}_name'])
        #print(geo.primitives[parsed_info_dict[f'surf{i}_name']])
        #print('CONTROL POINTS SHAPE: ', cntrl_pts.shape)
        #print('INDEXING CONTROL POINTS SHAPE: ', cntrl_pts.shape[0])
        
        if np.sum(parsed_info_dict[f'surf{i}_u_multiplicities'][1:-1]) != len(parsed_info_dict[f'surf{i}_u_multiplicities'][1:-1]) \
            or np.sum(parsed_info_dict[f'surf{i}_v_multiplicities'][1:-1]) != len(parsed_info_dict[f'surf{i}_v_multiplicities'][1:-1])\
            or np.any(cntrl_pts.shape[0] <= 8)\
            or num_rows_of_cps < 3\
            or num_cp_per_row < 3:
            if not filtered:
                geo.remove_multiplicity(geo.primitives[parsed_info_dict[f'surf{i}_name']])
        #exit()
        initial_surfaces.append(np.reshape(geo.primitives[parsed_info_dict[f'surf{i}_name']].control_points, geo.primitives[parsed_info_dict[f'surf{i}_name']].shape))
        #print(i,geo.primitives[parsed_info_dict[f'surf{i}_name']].shape ,np.shape(geo.primitives[parsed_info_dict[f'surf{i}_name']].control_points))
        geo.total_cntrl_pts_vector = np.append(geo.total_cntrl_pts_vector, geo.primitives[parsed_info_dict[f'surf{i}_name']].control_points)
    
    if len(geo.total_cntrl_pts_vector)%3!= 0: 
        print('Warning: Incorrectly imported bspline object')
    geo.total_cntrl_pts_vector = np.reshape(geo.total_cntrl_pts_vector,(len(geo.total_cntrl_pts_vector)//3,3))            
    print('Complete import')

def read_openvsp_stp(file_name):
    ''' Read file '''
    with open(file_name, 'r') as f:
        print('Importing OpenVSP', file_name)
        if 'B_SPLINE_SURFACE_WITH_KNOTS' not in f.read():
            print("No knot surfaces found!!")
            print("Something is wrong with the file" \
                , "or this reader doesn't work for this file.")
            return

    '''Stage 1: Parse all information and line numbers for each surface and create B-spline objects'''
    bsplines = {}
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

            bsplines[parsed_info[0]] = BSplineSurface(
                name=parsed_info[0],
                order_u=int(parsed_info[1])+1,
                order_v=int(parsed_info[2])+1,
                shape=None,
                control_points=None,
                knots_u=knots_u,
                knots_v=knots_v)

            parsed_info_dict[f'surf{i}_name'] = parsed_info[0]
            parsed_info_dict[f'surf{i}_cp_line_nums'] = np.array(parsed_info[3])
            parsed_info_dict[f'surf{i}_u_multiplicities'] = np.array(parsed_info[8])
            parsed_info_dict[f'surf{i}_v_multiplicities'] = np.array(parsed_info[9])

    ''' Stage 2: Replace line numbers of control points with control points arrays'''

    line_numbs_total_array = np.array([])
    for i in range(num_surf):
        line_numbs_total_array = np.append(line_numbs_total_array, parsed_info_dict[f'surf{i}_cp_line_nums'].flatten())
    #print('line_numbs_total_array',line_numbs_total_array)
    point_table = pd.read_csv(file_name, sep='=', names=['lines', 'raw_point'], error_bad_lines=False)
    #print('point_table',point_table)
    filtered_point_table = point_table.loc[point_table["lines"].isin(line_numbs_total_array)]
    #print('filtered_point_table',filtered_point_table)
    point_table = pd.DataFrame(filtered_point_table['raw_point'].str.findall(r"(-?\d+\.\d*E?-?\d*)").to_list(), columns=['x', 'y', 'z'])
    #print('point_table',point_table)
    point_table["lines"] = filtered_point_table["lines"].values
    initial_surfaces = []

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

        bspline_name = parsed_info_dict[f'surf{i}_name']
        bsplines[bspline_name].shape = np.array(cntrl_pts.shape)
        cntrl_pts = np.reshape(cntrl_pts, (num_rows_of_cps*num_cp_per_row,3))     
        bsplines[bspline_name].control_points = cntrl_pts
        
        # if np.sum(parsed_info_dict[f'surf{i}_u_multiplicities'][1:-1]) != len(parsed_info_dict[f'surf{i}_u_multiplicities'][1:-1]) \
        #     or np.sum(parsed_info_dict[f'surf{i}_v_multiplicities'][1:-1]) != len(parsed_info_dict[f'surf{i}_v_multiplicities'][1:-1])\
        #     or np.any(cntrl_pts.shape[0] <= 8):
        #     if not filtered:
        #         bsplines[bspline_name] = bsp.refit_bspline(bspline=bsplines[bspline_name], name=bspline_name, num_control_points=(10,), fit_resolution=(25,))
        #         # bsplines[bspline_name] = bsp.refit_bspline(bspline=bsplines[bspline_name], name=bspline_name, num_control_points=(25,), fit_resolution=(50,))
          
    print('Complete import')
    return bsplines


def write_step(geo, file_name):
    print('Step writer is not implemented yet! Please use the iges writer.')
    pass