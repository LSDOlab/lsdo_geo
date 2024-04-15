from steputils import p21
import meshio


# msh = meshio.read("examples/advanced_examples/meshes/segment0.msh")
# msh = meshio.read("examples/advanced_examples/meshes/test_mesh.msh")
# msh = meshio.read("examples/advanced_examples/meshes/Module_no_pump.msh")
msh = meshio.read("examples/advanced_examples/robotic_fish/meshes/module_v1_fine.msh")
# msh = meshio.read("examples/advanced_examples/robotic_fish/meshes/fish_v1.msh")

# print(msh.points)
# msh = meshio.read("examples/advanced_examples/meshes/couple2_tail.msh")
# msh = meshio.read("lsdo_geo/splines/b_splines/sample_geometries/SERPENT_3modules.msh")
# msh = meshio.read("../../temp/mesh2.msh")
# msh = meshio.read("../../temp/cube_mesh.msh")
# msh = meshio.read("../../temp/pneunet1.msh")
# msh = meshio.read("lsdo_geo/splines/b_splines/sample_geometries/fishy_mesh_from_iges.msh")
# msh = meshio.read("lsdo_geo/splines/b_splines/sample_geometries/simplified_fishy.msh")
# msh = meshio.read("lsdo_geo/splines/b_splines/sample_geometries/SERPENT_3modules1.msh")

# def create_mesh(mesh, cell_type, prune_z=False):
#     cells = mesh.get_cells_type(cell_type)
#     cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
#     points = mesh.points[:, :2] if prune_z else mesh.points
#     out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={
#                            "name_to_read": [cell_data]})
#     return out_mesh

# def create_mesh(mesh, cell_type, prune_z=False):
#     cells = mesh.get_cells_type(cell_type)
#     # cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
#     points = mesh.points[:, :2] if prune_z else mesh.points/1000
#     out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})
#     return out_mesh

# def create_mesh(mesh, cell_type, prune_z=False):
#     cells = mesh.get_cells_type(cell_type)
#     cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
#     points = mesh.points[:, :2] if prune_z else mesh.points
#     out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={
#                            "name_to_read": [cell_data]})
#     return out_mesh

# tetra_mesh = create_mesh(msh, "tetra", prune_z=False)

cells = msh.get_cells_type("tetra")
# meshio.write("examples/advanced_examples/meshes/segment0.xdmf", meshio.Mesh(points=msh.points, cells={"tetra": msh.cells["tetra"]}))
meshio.write("examples/advanced_examples/meshes/module_v1_fine.xdmf", meshio.Mesh(points=msh.points/1000, cells={"tetra": cells}))
# meshio.write("examples/advanced_examples/meshes/segment1_with_left_chamber.xdmf", meshio.Mesh(points=msh.points, cells={"tetra": cells}))
cells = msh.get_cells_type("triangle")
cell_data = msh.get_cell_data("gmsh:physical", "triangle")
# meshio.write("examples/advanced_examples/meshes/left_chamber_inner_surfaces.xdmf", meshio.Mesh(points=msh.points, cells={"triangle": msh.cells_dict["triangle"]},
#                                     cell_data={"triangle": {"left_chamber_inner_surfaces": msh.cell_data_dict["triangle"]["gmsh:physical"]}}))
meshio.write("examples/advanced_examples/meshes/module_v1_fine_left_chamber_inner_surfaces.xdmf", meshio.Mesh(points=msh.points/1000, cells={"triangle": cells},
                                    cell_data={"left_chamber_inner_surfaces": [cell_data]}))
# meshio.write("examples/advanced_examples/meshes/right_chamber_inner_surfaces.xdmf", meshio.Mesh(points=msh.points, cells={"triangle": cells}, 
#                                     cell_data={"triangle": {"right_chamber_inner_surfaces": msh.cell_data["triangle"]["gmsh:physical"]}}))
meshio.write("examples/advanced_examples/meshes/module_v1_fine_right_chamber_inner_surfaces.xdmf", meshio.Mesh(points=msh.points/1000, cells={"triangle": cells}, 
                                    cell_data={"right_chamber_inner_surfaces": [cell_data]}))
# meshio.write("examples/advanced_examples/meshes/segment1_with_left_chamber_left_chamber.xdmf", meshio.Mesh(points=msh.points, cells={"triangle": cells},
#                                     cell_data={"triangle": {"left_chamber_inner_surfaces": cell_data}}))

# from dolfinx import * 
# mesh = Mesh()
# with XDMFFile("examples/advanced_examples/meshes/segment0.xdmf") as infile:
#     infile.read(mesh)
# mvc = MeshValueCollection("size_t", mesh, 2) 
# with XDMFFile("examples/advanced_examples/meshes/segment0.xdmf") as infile:
#     infile.read(mvc, "name_to_read")
# mf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# mvc = MeshValueCollection("size_t", mesh, 3)
# with XDMFFile("examples/advanced_examples/meshes/segment0.xdmf") as infile:
#     infile.read(mvc, "name_to_read")
# cf = cpp.mesh.MeshFunctionSizet(mesh, mvc)

# for cell in msh.cells:
#     if cell.type == "triangle":
#         triangle_cells = cell.data
#     elif  cell.type == "tetra":
#         tetra_cells = cell.data

# for key in msh.cell_data_dict["gmsh:physical"].keys():
#     if key == "triangle":
#         triangle_data = msh.cell_data_dict["gmsh:physical"][key]
#     elif key == "tetra":
#         tetra_data = msh.cell_data_dict["gmsh:physical"][key]
# tetra_mesh = meshio.Mesh(points=msh.points, cells={"tetra": tetra_cells})
# triangle_mesh =meshio.Mesh(points=msh.points,
#                            cells=[("triangle", triangle_cells)],
#                            cell_data={"name_to_read":[triangle_data]})

# meshio.write("examples/advanced_examples/meshes/segment0.xdmf", tetra_mesh)
# meshio.write("lsdo_geo/splines/b_splines/sample_geometries/cube_mesh.xdmf", tetra_mesh)
# meshio.write("lsdo_geo/splines/b_splines/sample_geometries/pneunet1.xdmf", tetra_mesh)
# meshio.write("lsdo_geo/splines/b_splines/sample_geometries/fishy_mesh_from_iges.xdmf", tetra_mesh)
# meshio.write("lsdo_geo/splines/b_splines/sample_geometries/simplified_fishy.xdmf", tetra_mesh)
# meshio.write("lsdo_geo/splines/b_splines/sample_geometries/SERPENT_3modules1.xdmf", tetra_mesh)
# meshio.write("examples/advanced_examples/meshes.segment0_triangles.xdmf", triangle_mesh)

# meshio.write("examples/advanced_examples/meshes.segment0.xdmf", meshio.Mesh(points=msh.points, cells={"tetra": msh.cells["tetra"]}))
# meshio.write("examples/advanced_examples/meshes.segment0.xdmf", meshio.Mesh(points=msh.points, cells={"tetra": msh.cells}))

# my_file = p21.readfile('lsdo_geo/splines/b_splines/sample_geometries/SERPENT_3modules.STEP')
# # my_file = p21.readfile('lsdo_geo/splines/b_splines/sample_geometries/lift_plus_cruise_final.stp')
# print(my_file)

# names_list = {}
# for simple_or_complex_entity in my_file.data[0].instances.values():
#     try:
#         if simple_or_complex_entity.entity.name not in names_list:
#             # names_list.append(simple_or_complex_entity.entity.name)
#             names_list[simple_or_complex_entity.entity.name] = 1
#         else:
#             names_list[simple_or_complex_entity.entity.name] += 1
#     except:
#         for entity in simple_or_complex_entity.entities:
#             if entity.name not in names_list:
#                 # names_list.append(entity.name)
#                 names_list[entity.name] = 1
#             else:
#                 names_list[entity.name] += 1

