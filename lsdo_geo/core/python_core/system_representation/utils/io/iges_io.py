import numpy as np
from vedo import Points, Plotter, colors, LegendBox, show

from caddee.primitives.bsplines.bspline_surface import BSplineSurface

import pandas as pd
import re
import numpy as np
import copy
from vedo import Points, Plotter, colors, LegendBox, show


def read_iges(geo,file_name):
    """
    Read and import a geometry that's in an IGES format.
    Parameters
    ----------
    fileName : str
        File name of iges file. Should have .igs extension.
    """

    # print('The IGES reader still needs to be implemented. Please use the step reader.')
    # f = open(file_name, 'r')
    # Ifile = []
    # for line in f:
    #     line = line.replace(';', ',')  #This is a bit of a hack...
    #     Ifile.append(line)
    # f.close()

    # start_lines   = int((Ifile[-1][1:8]))
    # general_lines = int((Ifile[-1][9:16]))
    # directory_lines = int((Ifile[-1][17:24]))
    # parameter_lines = int((Ifile[-1][25:32]))

    # # Now we know how many lines we have to deal with
    # dir_offset  = start_lines + general_lines
    # para_offset = dir_offset + directory_lines

    # surf_list = []
    # # Directory lines is ALWAYS a multiple of 2
    # for i in range(directory_lines//2):
    #     # 128 is bspline surface type
    #     if int(Ifile[2*i + dir_offset][0:8]) == 128:
    #         start = int(Ifile[2*i + dir_offset][8:16])
    #         num_lines = int(Ifile[2*i + 1 + dir_offset][24:32])
    #         surf_list.append([start, num_lines])

    # nSurf = 1 #

    # print('Found %d surfaces in Iges File.'%(len(surf_list)))

    # surfs = []

    # for isurf in range(nSurf):  # Loop over our patches
    #     print(isurf)
    #     data = []
    #     # Create a list of all data
    #     # -1 is for conversion from 1 based (iges) to python
    #     para_offset = surf_list[isurf][0]+dir_offset+directory_lines-1

    #     for i in range(surf_list[isurf][1]):
    #         aux = Ifile[i+para_offset][0:69].split(',')
    #         for j in range(len(aux)-1):
    #             data.append(float(aux[j]))

    #     # Now we extract what we need
    #     Nctlu = int(data[1]+1)
    #     Nctlv = int(data[2]+1)
    #     ku    = int(data[3]+1)
    #     kv    = int(data[4]+1)

    #     counter = 10
    #     tu = data[counter:counter+Nctlu+ku]
    #     counter += (Nctlu + ku)

    #     tv = data[counter:counter+Nctlv+kv]
    #     counter += (Nctlv + kv)
    #     #print('tu',tu)
    #     #print('tv',tv)
    #     weights = data[counter:counter+Nctlu*Nctlv]
    #     weights = np.array(weights)
    #     if weights.all() != 1:
    #         print('WARNING: Not all weight in B-spline surface are 1. A NURBS surface CANNOT be replicated exactly')
    #     counter += Nctlu*Nctlv

    #     coef = np.zeros([Nctlu, Nctlv, 3])
    #     for j in range(Nctlv):
    #         for i in range(Nctlu):
    #             coef[i, j, :] = data[counter:counter +3]
    #             counter += 3

    #     # Last we need the ranges
    #     prange = np.zeros(4)

    #     prange[0] = data[counter    ]
    #     prange[1] = data[counter + 1]
    #     prange[2] = data[counter + 2]
    #     prange[3] = data[counter + 3]

    #     # Re-scale the knot vectors in case the upper bound is not 1
    #     tu = np.array(tu)
    #     tv = np.array(tv)
    #     '''
    #     if not tu[-1] == 1.0:
    #         tu /= tu[-1]

    #     if not tv[-1] == 1.0:
    #         tv /= tv[-1]
    #     '''
    # #return tu, tv, Nctlu, Nctlv, ku, kv #coef, 
        
    ''' Read file '''
    with open(file_name, 'r') as f:
        print('Importing', file_name)
        if 'B_SPLINE_SURFACE_WITH_KNOTS' not in f.read():
            print("No knot surfaces found!!")
            print("Something is wrong with the file" \
                , "or this reader doesn't work for this file.")
            return

    '''Stage 1: Parse all information and line numbers for each surface'''
    parsed_info_dict = {}
    with open(file_name, 'r') as f:
        b_spline_surf_info = re.findall(r"B_SPLINE_SURFACE_WITH_KNOTS.*\)", f.read())
        num_surf = len(b_spline_surf_info)
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
            knots_u = np.array([parsed_info[10]])
            print('knots_u',knots_u)
            knots_u = np.repeat(knots_u, parsed_info[8])
            print('repeat knots_u', parsed_info[8], 'times',knots_u)
            knots_u = knots_u/knots_u[-1]
            knots_v = np.array([parsed_info[11]])
            print('knots_v',knots_v)
            knots_v = np.repeat(knots_v, parsed_info[9])
            print('repeat knots_v', parsed_info[9], 'times',knots_v)
            knots_v = knots_v/knots_v[-1]
            print('knots_u',knots_u,'knots_v',knots_v)
            exit()
            geo.input_bspline_entity_dict[parsed_info[0]] = (BSplineSurface(
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

    ''' Stage 2: Replace line numbers of control points with control points arrays'''

    line_numbs_total_array = np.array([])
    for i in range(num_surf):
        line_numbs_total_array = np.append(line_numbs_total_array, parsed_info_dict[f'surf{i}_cp_line_nums'].flatten())

    point_table = pd.read_csv(file_name, sep='=', error_bad_lines=False, names=['lines', 'raw_point'])
    filtered_point_table = point_table.loc[point_table["lines"].isin(line_numbs_total_array)]
    point_table = pd.DataFrame(filtered_point_table['raw_point'].str.findall(r"(-?\d+\.\d*E?-?\d*)").to_list(), columns=['x', 'y', 'z'])
    point_table["lines"] = filtered_point_table["lines"].values
    geo.initial_input_bspline_entity_dict = copy.deepcopy(geo.input_bspline_entity_dict)
    initial_surfaces = []
    for i in range(num_surf):
        num_rows_of_cps = parsed_info_dict[f'surf{i}_cp_line_nums'].shape[0]
        num_cp_per_row = parsed_info_dict[f'surf{i}_cp_line_nums'].shape[1]
        cntrl_pts = np.zeros((num_rows_of_cps, num_cp_per_row, 3))
        for j in range(num_rows_of_cps):
            col_cntrl_pts = point_table.loc[point_table["lines"].isin(parsed_info_dict[f'surf{i}_cp_line_nums'][j])][['x', 'y', 'z']]
            if ((len(col_cntrl_pts) != num_cp_per_row) and (len(col_cntrl_pts) != 1)):
                print('SKIPPED SURFACES: ', parsed_info_dict[f'surf{i}_name'])
                # geo.initial_input_bspline_entity_dict.pop(f'surf{i}_name', None)
                # geo.input_bspline_entity_dict.pop(f'surf{i}_name', None)
                # filtered = True
                # continue
                for k in range(num_cp_per_row):
                    cntrl_pts[j,k,:] = point_table.loc[point_table["lines"]==parsed_info_dict[f'surf{i}_cp_line_nums'][j][k]][['x', 'y', 'z']]
            else:
                filtered = False
                cntrl_pts[j,:,:] = col_cntrl_pts

        # print('Control Points shape: ', cntrl_pts.shape)
        geo.input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']].shape = np.array(cntrl_pts.shape)
        geo.initial_input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']].shape = np.array(cntrl_pts.shape)
        # print('NUMBER OF CONTROL POINTS IMPORT: ', num_cp)
        # geo.input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']].starting_geometry_index = num_cp
        # geo.initial_input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']].starting_geometry_index = num_cp
        cntrl_pts = np.reshape(cntrl_pts, (num_rows_of_cps*num_cp_per_row,3))     
        geo.input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']].control_points = cntrl_pts
        geo.initial_input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']].control_points = cntrl_pts
        # print('Number of rows: ', num_rows_of_cps)
        # print('Number of cp per row: ', num_cp_per_row)
        # num_cp += num_rows_of_cps * num_cp_per_row
        #initial = geo.input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']]
        #print(geo.input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']])
        #print(parsed_info_dict[f'surf{i}_name'])
        #print(geo.input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']])
        print('CONTROL POINTS SHAPE: ', cntrl_pts.shape)
        print('INDEXING CONTROL POINTS SHAPE: ', cntrl_pts.shape[0])
        
        if np.sum(parsed_info_dict[f'surf{i}_u_multiplicities'][1:-1]) != len(parsed_info_dict[f'surf{i}_u_multiplicities'][1:-1]) \
            or np.sum(parsed_info_dict[f'surf{i}_v_multiplicities'][1:-1]) != len(parsed_info_dict[f'surf{i}_v_multiplicities'][1:-1])\
            or np.any(cntrl_pts.shape[0] <= 8):
            if not filtered:
                geo.remove_multiplicity(geo.input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']])
        #exit()
        initial_surfaces.append(np.reshape(geo.input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']].control_points, geo.input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']].shape))
        #print(i,geo.input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']].shape ,np.shape(geo.input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']].control_points))
        geo.total_cntrl_pts_vector = np.append(geo.total_cntrl_pts_vector, geo.input_bspline_entity_dict[parsed_info_dict[f'surf{i}_name']].control_points)
        
    # nvert, nedge, ngroup, surf_ptrs, edge_ptrs, surf_group, edge_group = geo.compute_topology(initial_surfaces)
    # size, topo, bspline = geo.compute_indices(initial_surfaces, nvert, nedge, ngroup, surf_ptrs, edge_ptrs, surf_group, edge_group)
    # vec = {
    #     'df_str': None,
    #     'df': None,
    #     'cp': None,
    #     'cp_str': None,
    #     'pt_str': None,
    #     'pt': None,
    # }
    # for vec_type in ['df_str', 'df', 'cp', 'cp_str', 'pt_str', 'pt']:
    #     geo.initialize_vec(vec_type, vec_type, vec, size, topo['surf_group'], bspline, 3) 
    # print('Vector sizes (unique, structured)')
    # print('---------------------------------')
    # print('# free control points:', np.shape(geo.vec['df']), np.shape(geo.vec['df_str']))
    # print('# control points:     ', np.shape(geo.vec['cp']), np.shape(geo.vec['cp_str']))
    # print('# discretized points: ', np.shape(geo.vec['pt']), np.shape(geo.vec['pt_str']))
    
    if len(geo.total_cntrl_pts_vector)%3!= 0: 
        print('Warning: Incorrectly imported bspline object')
    geo.total_cntrl_pts_vector = np.reshape(geo.total_cntrl_pts_vector,(len(geo.total_cntrl_pts_vector)//3,3))            
    print('Complete import')
    pass


def write_iges(geo, file_name, plot = False):
        """
        Write the surface to IGES format
        Parameters
        ----------
        fileName : str
            File name of iges file. Should have .igs extension.
        """
        if plot == True:
            vp_out = Plotter()
            vps = []
            for surf, color in zip(geo.output_bspline_entity_dict.values(), colors.colors.values()):
                vps.append(Points(surf.control_points, r=8, c = color).legend(surf.name))
            #TODO legend
            #lb = LegendBox(vps, nmax=i, width = 0.2, pad = 0, pos = "top-left")
            vp_out.show(vps, 'Control points', axes=1, viewup="z", interactive = False)
        f = open(file_name, 'w')
        print('Exporting', file_name)
        #TODO Change to correct information
        f.write('                                                                        S      1\n')
        f.write('1H,,1H;,7H128-000,11H128-000.IGS,9H{unknown},9H{unknown},16,6,15,13,15, G      1\n')
        f.write('7H128-000,1.,6,1HM,8,0.016,15H19970830.165254, 0.0001,0.,               G      2\n')
        f.write('21Hdennette@wiz-worx.com,23HLegacy PDD AP Committee,11,3,               G      3\n')
        f.write('13H920717.080000,23HMIL-PRF-28000B0,CLASS 1;                            G      4\n')
        Dcount = 1
        Pcount = 1
        for surf in geo.primitives.values():
            surf = surf.geometry_primitive
            paraEntries = 13 + (len(surf.knots_u)) + (len(surf.knots_v)) + surf.shape[0] * surf.shape[1] + 3 * surf.shape[0] * surf.shape[1] + 1
            paraLines = (paraEntries - 10) // 3 + 2
            if np.mod(paraEntries - 10, 3) != 0:
                paraLines += 1
            f.write("     128%8d       0       0       1       0       0       000000001D%7d\n" % (Pcount, Dcount))
            f.write(
            "     128       0       2%8d       0                               0D%7d\n" % (paraLines, Dcount + 1)
            )
            Dcount += 2
            Pcount += paraLines
        Pcount  = 1
        counter = 1
        for surf in geo.primitives.values():
            surf = surf.geometry_primitive
            f.write(
                "%10d,%10d,%10d,%10d,%10d,          %7dP%7d\n"
                % (128, surf.shape[0] - 1, surf.shape[1] - 1, surf.order_u - 1, surf.order_v - 1, Pcount, counter)
            )
            counter += 1
            f.write("%10d,%10d,%10d,%10d,%10d,          %7dP%7d\n" % (0, 0, 1, 0, 0, Pcount, counter))

            counter += 1
            pos_counter = 0

            for i in range(len(surf.knots_u)):
                pos_counter += 1
                f.write("%20.12g," % (np.real(surf.knots_u[i])))
                if np.mod(pos_counter, 3) == 0:
                    f.write("  %7dP%7d\n" % (Pcount, counter))
                    counter += 1
                    pos_counter = 0

            for i in range(len(surf.knots_v)):
                pos_counter += 1
                f.write("%20.12g," % (np.real(surf.knots_v[i])))
                if np.mod(pos_counter, 3) == 0:
                    f.write("  %7dP%7d\n" % (Pcount, counter))
                    counter += 1
                    pos_counter = 0

            for i in range(surf.shape[0] * surf.shape[1]):
                pos_counter += 1
                f.write("%20.12g," % (1.0))
                if np.mod(pos_counter, 3) == 0:
                    f.write("  %7dP%7d\n" % (Pcount, counter))
                    counter += 1
                    pos_counter = 0

            for j in range(surf.shape[1]):
                for i in range(surf.shape[0]):
                    for idim in range(3):
                        pos_counter += 1
                        cntrl_pts = np.reshape(surf.control_points, (surf.shape[0], surf.shape[1],3))
                        f.write("%20.12g," % (np.real(cntrl_pts[i, j, idim])))
                        if np.mod(pos_counter, 3) == 0:
                            f.write("  %7dP%7d\n" % (Pcount, counter))
                            counter += 1
                            pos_counter = 0

            for i in range(4):
                pos_counter += 1
                if i == 0:
                    f.write("%20.12g," % (np.real(surf.knots_u[0])))
                if i == 1:
                    f.write("%20.12g," % (np.real(surf.knots_u[1])))
                if i == 2:
                    f.write("%20.12g," % (np.real(surf.knots_v[0])))
                if i == 3:
                    f.write("%20.12g;" % (np.real(surf.knots_v[1])))
                if np.mod(pos_counter, 3) == 0:
                    f.write("  %7dP%7d\n" % (Pcount, counter))
                    counter += 1
                    pos_counter = 0
                else:  
                    if i == 3:
                        for j in range(3 - pos_counter):
                            f.write("%21s" % (" "))
                        pos_counter = 0
                        f.write("  %7dP%7d\n" % (Pcount, counter))
                        counter += 1

            Pcount += 2 
        f.write('S%7dG%7dD%7dP%7d%40sT%6s1\n'%(1, 4, Dcount-1, counter-1, ' ', ' '))
        f.close()  
        print('Complete export')