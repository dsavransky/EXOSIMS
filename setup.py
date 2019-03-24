import setuptools
import numpy
import os.path

use_cython = True
try:
    from Cython.Build import cythonize
    extensions = [setuptools.Extension("EXOSIMS.util.KeplerSTM_C.CyKeplerSTM", \
                    [os.path.join("EXOSIMS","util","KeplerSTM_C","CyKeplerSTM.pyx"),\
                    os.path.join("EXOSIMS","util","KeplerSTM_C","KeplerSTM_C.c")],\
                    include_dirs = [numpy.get_include()]) ]
    extensions = cythonize(extensions)
except ImportError:
    use_cython = False
    extensions = []

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EXOSIMS",
    version="2.0.0",
    author="Dmitry Savransky",
    author_email="ds264@cornell.edu",
    description="Two-body orbital propagation tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dsavransky/EXOSIMS",
    packages=setuptools.find_packages(exclude=['tests*']),
    install_requires=['numpy','scipy','astropy','jplephem','h5py','ortools'],
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    ext_modules = extensions
)
