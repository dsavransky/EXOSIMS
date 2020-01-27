'''
Setup file for CyKeplerSTM

Code must be compiled before use: 
> python CyKeplerSTM_setup.py build_ext --inplace
'''

import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension(
    name="CyKeplerSTM",
    sources=["CyKeplerSTM.pyx", "KeplerSTM_C.c"],
    language="c++",
    extra_compile_args=["-stdlib=libc++"],
    extra_link_args= ["-stdlib=libc++"],
    include_dirs = [numpy.get_include()],
    )]

setup(
    name = 'CyKeplerSTM',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
)

