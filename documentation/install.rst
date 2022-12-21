.. _install:

Installing and Configuring
####################################

Obtaining EXOSIMS
=========================================

EXOSIMS is hosted on github at https://github.com/dsavransky/EXOSIMS.  The master branch is a development branch and may be frequently changed, including ICD-breaking changes (however it should always pass all unit and end-to-end tests).  For publishable/reproducible results, we strongly recommend using the latest tagged release from:

https://github.com/dsavransky/EXOSIMS/releases

Environment and Package Dependencies
==========================================

EXOSIMS requires Python 3.7+ and a number of packages hosted on the `PyPI <https://pypi.org/>`__. For a full list of
current package dependencies, see the ``requirements.txt`` file in the top level of the repository. 


Installation
=============================

If you wish to install EXOSIMS to try it out, or to run existing code, then just grab the current stable version from
PyPI:
::

    pip install EXOSIMS

If, however, you are planning on developing your own module implementations, then it is recommended that you install
in developer mode.  Clone (or download) a copy of the repository and in the top level folder (the one containing ``setup.py`` 
run:
::

   pip install -e .

Note that this is the install mode used for all automated unit testing.  Sere here for more details: https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs

.. note::

     Installing EXOSIMS via pip will automatically install all required and optional packages, except for cython.  If cython and numpy are already present on the system, the cythonized components of EXOSIMS will be compiled automatically, otherwise they will be ignored.  Note that this does not represent any loss of functionality, as these components are drop-in replacements for native python implementations.  The only difference in execution will be an increase in total run time. 


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
