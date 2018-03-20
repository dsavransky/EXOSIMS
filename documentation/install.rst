.. _install:

Installing and Configuring EXOSIMS
####################################

Obtaining EXOSIMS
=========================================

``EXOSIMS`` is hosted on github at https://github.com/dsavransky/EXOSIMS.  The master branch is a development branch and may be frequently changed.  We strongly recommend using a tagged release:

https://github.com/dsavransky/EXOSIMS/releases

Environment and Package Dependencies
==========================================

``EXOSIMS`` requires Python 2.7.8+ and the following packages:

* astropy
* numpy
* scipy
* matplotlib (for visualization of results only)


Optional Packages
---------------------
* ``cPickle`` is used preferentially, but ``pickle`` will be loaded if ``cPickle`` is not installed
* ``jplephem`` is used by the Observatory prototype and implementations for calculating positions of solar system bodies and is highly recommended.  Install ``jplephem`` from source or via pip 
  ::
    pip install jplephem
  An SPK ephemeris file is required - the default one can be downloaded from http://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp and should be placed in the Observatory subdirectory of EXOSIMS. Other kernels can be downloaded from http://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
* The ``Forecaster`` ``PlanetPhysicalModel`` implementation requires module ``h5py``.  Install from source or via pip 
  ::
    pip install h5py
* The ``Forecaster`` ``PlanetPhysicalModel`` implementation requires a data file called ``fitting_parameters.h5``.  Download from https://github.com/chenjj2/forecaster and place in the PlanetPhysicalModel. 
* Several methods have been implemented in both python and Cython (http://cython.org/).  To get the speed benefits of the Cython versions, you will need to install Cython (from source or pip) and compile the Cythonized EXOSIMS modules on your system (see: :ref:`cythonized`).
* The SLSQP module requires ortools which can be installed by following instructions at the following link (https://developers.google.com/optimization/introduction/installing/binary). For installation in a virtualenv (https://virtualenv.pypa.io/en/stable/) on POSIX systems, use these steps:
  :: 
    mkdir python-virtual-env
    cd python-virtual-env
    sudo pip install virtualenv
    virtualenv 
    virtualenv -p /usr/bin/python2.7 myVENV
    source myVENV/bin/activate
    cd myVENV
    pip install --upgrade ortools

  .. note::
    
    You may need to have a C/C++ compiler installed on your system, such as the GNU compiler collection, or the command line tools for XCode on OS X.  You will also need a working installation of SWIG (http://www.swig.org/)


Installation and Path Setup
=============================
EXOSIMS is organized into a folder hierarchy, with a folder for each module type.  All implementations of each module type should be placed in their appropriate subfolder.  There is also a Prototypes directory, which carries all of the module prototypes, as well as a Scripts directory for json scripts.  Certain modules will save intermediate products to their particular module subfolders, and so the entire EXOSIMS folder tree must be user writeable.  

.. _EXOSIMSROOT:

The directory containing the ``EXOSIMS`` code directory, which we will call ``EXOSIMSROOT`` must be in your Python path.  If you clone EXOSIMS directly from github then you will automatically get a directory hierarchy that looks like ``installpath/EXOSIMS/EXOSIMS``.  In this case ``installpath/EXOSIMS`` is the ``EXOSIMSROOT``. On POSIX systems, this is accomplished simply by appending the path to the ``PYTHONPATH`` environment variables.

Setting PYTHONPATH on MacOS/Linux for bash Environments
---------------------------------------------------------

To append the ``EXOSIMS`` directory to your ``PYTHONPATH`` locate your .bashrc file (should be in your home directory) and append the following line to the end:
::
    export PYHTONPATH="$PYTHONPATH:EXOSIMSROOT"

You will need to start a new shell session, or source the bashrc in your current session (``> source ~/.bashrc``). For other shell environments, check the relevant documentation for your environment. To check which shell you are using, execute ``> echo $SHELL``.  To check the current ``PYTHONPATH``, execute ``> echo $PYTHONPATH``.



Setting PYTHONPATH in WINDOWS
-----------------------------
Right click on My Computer and select Properties > Advanced Systems Settings > Environment Variables > then under system variables add a new variable called ``PYTHONPATH`` or append to it if it exists. In this variable you need to have ``C:\\EXOSIMSROOT``.

For more information see: https://docs.python.org/2/using/windows.html#excursus-setting-environment-variables


.. _cythonized:

Compiling Cython Modules
============================

To speed up execution, some EXOSIMS components are implemented as both regular interpreted python and as statically compiled executables via Cython. The code is set to automatically use the compiled versions if they are available, and these (currently) must be manually compiled on each system where the code is installed.  In all cases, compilation is done by executing a python setup script.  The individual components with Cython implementations are listed below.

KeplerSTM
-------------
The ``KeplerSTM`` utility is responsible for orbital propagation in ``EXOSIMS``.  It has a Cython implementation: ``CyKeplerSTM``, which wraps a pure C implementation of the propagation algorithms, called ``KeplerSTM_C``. To compile the Cython implementation, navigate to ``EXOSIMSROOT/EXOSIMS/util/KeplerSTM_C``.  Execute: 
::
    > python CyKeplerSTM_setup.py build_ext --inplace

This will generate a ``.c`` file and compile to a ``.so`` file on MacOS/Linux or a ``.pyd`` file on Windows.  The python ``KeplerSTM`` automatically loads the compiled module if it is present, and uses it by default if successfully loaded.

    
    




