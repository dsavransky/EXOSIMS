.. _install:

Installing and Configuring
####################################

Obtaining EXOSIMS
=========================================

``EXOSIMS`` is hosted on github at https://github.com/dsavransky/EXOSIMS.  The master branch is a development branch and may be frequently changed, including ICD breaking changes (however it should always pass all unit and end-to-end tests).  For publishable/reproducible results, we strongly recommend using the latest tagged release from:

https://github.com/dsavransky/EXOSIMS/releases

Environment and Package Dependencies
==========================================

``EXOSIMS`` requires Python 2.7.8+ or 3.6+ and the following packages:

* astropy
* numpy
* scipy
* matplotlib (for visualization of results only)


Optional Packages
---------------------
* ``cPickle`` is used preferentially in Python 2, but ``pickle`` will be loaded if ``cPickle`` is not installed
* ``jplephem`` is used by the Observatory prototype and implementations for calculating positions of solar system bodies and is highly recommended.  Install ``jplephem`` from source or via pip:
  ::
  
   pip install jplephem

  An SPK ephemeris file is required - the default one can be downloaded from http://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp and should be placed in the ``.EXOSIMS/downloads`` subdirectory as discussed below. EXOSIMS will attempt to automatically fetch the ephemeris file if it is not found. Other kernels can be downloaded from http://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
* The ``Forecaster`` ``PlanetPhysicalModel`` implementation requires module ``h5py``.  Install from source or via pip:
  ::

   pip install h5py
    
* The ``Forecaster`` ``PlanetPhysicalModel`` implementation requires a data file called ``fitting_parameters.h5``.  Download from https://github.com/chenjj2/forecaster and place in the ``.EXOSIMS/downloads`` subdirectory (the code will automatically attempt to download this file if not found at runtime). 
* Several methods have been implemented in both python and Cython (http://cython.org/).  To get the speed benefits of the Cython versions, you will need to install Cython (from source or pip) and compile the Cythonized EXOSIMS modules on your system (see: :ref:`cythonized`).
* The SLSQP module requires ortools which can be installed from source or via pip:
  ::

   pip install ortools

  .. note::
    
    You may need to have a C/C++ compiler installed on your system, such as the GNU compiler collection, or the command line tools for XCode on OS X.  You will also need a working installation of SWIG (http://www.swig.org/)


Installation and Path Setup
=============================
EXOSIMS is organized into a folder hierarchy, with a folder for each module type.  All implementations of each module type should be placed in their appropriate subfolder.  There is also a Prototypes directory, which carries all of the module prototypes, as well as a Scripts directory for json scripts.  Certain modules will save intermediate products to a specified cache directory (or to the default cache directory if unspecified).  

.. _EXOSIMSROOT:

The directory containing the ``EXOSIMS`` code directory, which we will call ``EXOSIMSROOT`` must be in your Python path.  If you clone EXOSIMS directly from github then you will automatically get a directory hierarchy that looks like ``installpath/EXOSIMS/EXOSIMS``.  In this case ``installpath/EXOSIMS`` is the ``EXOSIMSROOT``.  ``EXOSIMSROOT`` will include the package ``setup.py`` file and the ``EXOSIMS`` code directory at the top level.

Option 1: pip install (recommended)
--------------------------------------
The easiest way to get EXOSIMS installed on your system is via the ``pip`` package installer utility.  Simply navigate to ``EXOSIMSROOT`` and execute:
::

   pip install .

.. note::

   This installs EXOSIMS for all users on the system and requires administrative privileges (or write access to the python site-package folder heriarchy).  If you wish only to install EXOSIMS for your own user (or lack the appropriate permissions) add ``--user`` to the ``pip`` command.  See here for more information: https://pip.pypa.io/en/stable/user_guide/#user-installs

If you wish to install EXOSIMS in developer mode (equivalent to setuptools develop mode) use:
::

   pip install -e .

Note that this is the install mode used for all automated unit testing.  Sere here for more details: https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs

.. note::

     Installing EXOSIMS via pip will automatically install all required and optional packages listed above, except for cython and matplotlib.  If cython and numpy are already present on the system, the cythonized components of EXOSIMS will be compiled automatically, otherwise they will be ignored.  Note that this does not represent any loss of functionality, as these components are drop-in replacements for native python implementations.  The only difference in execution will be an increase in total run time. 

Option 2: Manually Add EXOSIMS to PYTHONPATH (not recommended)
----------------------------------------------------------------

On POSIX systems, this is accomplished simply by appending the ``EXOSIMSROOT`` path to the ``PYTHONPATH`` environment variables.

Setting PYTHONPATH on MacOS/Linux for bash Environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To append the ``EXOSIMS`` directory to your ``PYTHONPATH`` locate your .bashrc file (should be in your home directory) and append the following line to the end:
::

   export PYHTONPATH="$PYTHONPATH:EXOSIMSROOT"

You will need to start a new shell session, or source the bashrc in your current session (``> source ~/.bashrc``). For other shell environments, check the relevant documentation for your environment. To check which shell you are using, execute ``> echo $SHELL``.  To check the current ``PYTHONPATH``, execute ``> echo $PYTHONPATH``.


Setting PYTHONPATH in WINDOWS
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Right click on My Computer and select Properties > Advanced Systems Settings > Environment Variables > then under system variables add a new variable called ``PYTHONPATH`` or append to it if it exists. In this variable you need to have ``C:\\EXOSIMSROOT``.

For more information see: https://docs.python.org/2/using/windows.html#excursus-setting-environment-variables

.. _EXOSIMSCACHE:

Cache Directory
===========================

EXOSIMS generates a large number of cached data products during run time.  These are stored in the EXOSIMS cache, which can be controlled via environment variables or on a script-by-script basis.

On POSIX systems, the default cached directory is given by ``/home/user/.EXOSIMS/cache``. On Windows systems, the default cache directory is typically like ``C:/Users/User/.EXOSIMS/cache``. For details on how the home directory is determined, see method ``get_home_dir`` in ``util/get_dirs.py``.  If ``cachedir`` is specified in the input json script, the cache directory will be ``cachedir`` (this path may include environment variables and other expandable elements).  Alternatively, the cache directory may be specified by setting environment variable ``EXOSIMS_CACHE_DIR``.

The order of precedence for determining the cache directory is:

#. JSON input
#. Environment variable
#. Default path

.. _EXOSIMSDOWNLOADS:

Downloads Directory
======================
The downloads directory is where files from outside EXOSIMS are stored (this includes SPK files used by jplephem, Forecaster fitting parameters, etc.). On POSIX systems, the downloads directory is given by ``/home/user/.EXOSIMS/downloads``. On Windows systems, the downloads directory is typically like ``C:/Users/User/.EXOSIMS/downloads``. Just like the :ref:`EXOSIMSCACHE`, the downloads directory may be set via the JSON script (input ``downloadsdir``) or via environment variable ``EXOSIMS_DOWNLOADS_DIR``.  The order of precedence in selecting the directory to use at runtime is the same as for the cache.  


.. _cythonized:

Compiling Cython Modules
============================

  .. note::

     Installing EXOSIMS via pip will automatically compile all of these components if cython *and* numpy are already installed on the system.  You only need to preform this procedure if installing manually, or if you decide to add cython after initial installation. Note, however, that it is still preferable to just rerun the pip installation procedure after installing the cython package. 

To speed up execution, some EXOSIMS components are implemented as both regular interpreted python and as statically compiled executables via Cython. The code is set to automatically use the compiled versions if they are available, and these (currently) must be manually compiled on each system where the code is installed.  In all cases, compilation is done by executing a python setup script.  The individual components with Cython implementations are listed below.

KeplerSTM
-------------
The ``KeplerSTM`` utility is responsible for orbital propagation in ``EXOSIMS``.  It has a Cython implementation: ``CyKeplerSTM``, which wraps a pure C implementation of the propagation algorithms, called ``KeplerSTM_C``. To compile the Cython implementation, navigate to ``EXOSIMSROOT/EXOSIMS/util/KeplerSTM_C``.  Execute: 
::

   > python CyKeplerSTM_setup.py build_ext --inplace

This will generate a ``.c`` file and compile to a ``.so`` file on MacOS/Linux or a ``.pyd`` file on Windows.  The python ``KeplerSTM`` automatically loads the compiled module if it is present, and uses it by default if successfully loaded.
