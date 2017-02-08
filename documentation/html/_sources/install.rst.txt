.. _install:
Installing and Configuring EXOSIMS
####################################

Environment and Package Dependencies
==========================================

``EXOSIMS`` requires Python 2.7.x and the following packages:

* astropy
* numpy
* scipy
* matplotlib (for visualization of results only)

Optional Packages
---------------------
* ``cPickle`` is used preferentially, but ``pickle`` will be loaded if ``cPickle`` is not installed
* ``jplephem`` is used by the Observatory prototype and implementations for calculating positions of solar system bodies and is highly recommended.  It can be installed through pip or from source.  An SPK ephemeris file is needed - the default one can be downloaded from http://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp and should be placed in the Observatory subdirectory of EXOSIMS. Other kernels can be downloaded from http://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
* ``fitting_parameters.h5`` is used by the Forecaster module (an implementation of ``PlanetPhysicalModel``). It can be downloaded from https://github.com/chenjj2/forecaster and must be added to the PlanetPhysicalModel directory.  If using Forecaster, you must also have ``h5py`` installed.


Path Setup
============
EXOSIMS is organized into a folder hierarchy, with a folder for each module type.  All implementations of each module type should be placed in their appropriate subfolder.  There is also a Prototypes directory, which carries all of the module prototypes, as well as a Scripts directory for json scripts.  Certain modules will save intermediate products to their particular module subfolders, and so the entire EXOSIMS folder tree must be user writeable.  The directory containing the ``EXOSIMS`` directory (i.e., ``../EXOSIMS``) must be in your Python path.



