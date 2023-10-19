
import sys
import numpy as np

from setuptools import setup, Extension
from Cython.Build import cythonize

compile_args = []

if sys.platform.startswith('darwin'):
    compile_args=['-std=c++17', '-stdlib=libc++']
else:
    compile_args=['-std=c++17']

list_of_pyx_names = [
    ('cython', 'get_open_uniform'),
    ('cython', 'get_open_uniform_py'),
    ('cython', 'basis0'),
    ('cython', 'basis1'),
    ('cython', 'basis2'),
    ('cython', 'basis_matrix_curve'),
    ('cython', 'basis_matrix_curve_py'),
    ('cython', 'basis_matrix_surface'),
    ('cython', 'basis_matrix_surface_py'),
    ('cython', 'basis_matrix_volume'),
    ('cython', 'basis_matrix_volume_py'),
    ('cython', 'surface_projection'),
    ('cython', 'surface_projection_py'),
    ('cython', 'volume_projection'),
    ('cython', 'volume_projection_py'),
    ('cython', 'curve_projection'),
    ('cython', 'curve_projection_py'),
]

ext_modules = []
packages = []
for name_list in list_of_pyx_names:
    ext_name = 'lsdo_geo'
    source_name = 'lsdo_geo'
    packages.append('{}.{}'.format('lsdo_geo', name_list[0]))
    for name_part in name_list:
        ext_name = '{}.{}'.format(ext_name, name_part)
        source_name = '{}/{}'.format(source_name, name_part)
    source_name = source_name + '.pyx'
    ext_modules = ext_modules + cythonize(
        Extension(
            name=ext_name,
            sources=[source_name],
            language='c++',
            extra_compile_args=compile_args,
            include_dirs=[np.get_include()],
        ),
        annotate=True,
        build_dir='build',
    )

setup(
    name="lsdo_geo",
    version="0.1.0",
    description="LSDOGeo",
    packages=["lsdo_geo"],
    install_requires=[],
    ext_modules=ext_modules,
)
