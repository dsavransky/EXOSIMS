import setuptools
import os.path
import re

use_cython = True
try:
    from Cython.Build import cythonize
    import numpy
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

with open(os.path.join("EXOSIMS","__init__.py"), "r") as f:
    version_file = f.read()

version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",\
        version_file, re.M)

if version_match:
    version_string = version_match.group(1)
else:
    raise RuntimeError("Unable to find version string.")

setuptools.setup(
    name="EXOSIMS",
    version=version_string,
    author="Dmitry Savransky",
    author_email="ds264@cornell.edu",
    description="Exoplanet Imaging Mission Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dsavransky/EXOSIMS",
    packages=setuptools.find_packages(exclude=['tests*']),
    include_package_data=True,
    install_requires=['numpy','scipy','astropy','jplephem','h5py','ortools'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    ext_modules = extensions
)
