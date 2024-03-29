import numpy as np

from distutils.core import setup, Extension
from Cython.Build import cythonize

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

setup(ext_modules=cythonize(Extension(
    "points_to_rect",
    ["points_to_rect.pyx"],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include]))
)
