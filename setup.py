from setuptools import setup, find_packages

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


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='lsdo_geo',
    version='0.0.0',
    author='Andrew Fletcher',
    author_email='afletcher168@gmail.com',
    license='LGPLv3+',
    keywords='python project template repository package',
    url='http://github.com/LSDOlab/lsdo_project_template',
    download_url='http://pypi.python.org/pypi/lsdo_project_template',
    description='A template repository/package for LSDOlab projects',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.7',
    platforms=['any'],
    install_requires=[
        'numpy',
        'pytest',
        'myst-nb',
        'sphinx_rtd_theme',
        'sphinx-copybutton',
        'sphinx-autoapi',
        'numpydoc',
        'gitpython',
        #'sphinxcontrib-collections @ git+https://github.com/anugrahjo/sphinx-collections.git', # 'sphinx-collections',
        'sphinxcontrib-bibtex',
        'setuptools',
        'wheel',
        'twine',
    ],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Topic :: Documentation',
        'Topic :: Documentation :: Sphinx',
        'Topic :: Software Development',
        'Topic :: Software Development :: Documentation',
        'Topic :: Software Development :: Testing',
        'Topic :: Software Development :: Libraries',
    ],
    # ext_modules=ext_modules
)
