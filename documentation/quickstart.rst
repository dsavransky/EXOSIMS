.. _quickstart:
Quick Start Guide
######################

This is intended as a very brief overview of the steps necessary to get ``EXOSIMS`` running.  To start writing your own modules, please refer to the ICD and detailed as-built documentation of the prototypes.

Setting Up Your Environment
---------------------------

``EXOSIMS`` requires Python 2.7 and multiple packages including numpy and astropy.  All required modules are listed in the ICD and can be obtained from PyPI or other package resources such as Macports.


The directory containing the ``EXOSIMS`` directory must be in your Python path.  On POSIX systems, this is accomplished simply by appending the path to the ``PYTHONPATH`` environment variables.

For Ubuntu users
To append the ``EXOSIMS`` directory to your ``PYTHONPATH``
1. locate your .bashrc file (should be in your home directory /home/username)
2. Append the following lines::
    export PYHTONPATH="$PYTHONPATH:/home/username/PATH TO EXOSIMS PARENT DIRECTORY"
3. Save
4. From command line execute::
    $ source ~/.bashrc

An example: If I pulled ``EXOSIMS`` into the folder /home/user1/TACO such that TACO contains the folders EXOSIMS ICD and documentation, then my PYTHONPATH is::
    export PYTHONPATH="$PYTHONPATH:/home/user1/TACO"


Creating a Mission Simulation Object
-------------------------------------

The entry point to all ``EXOSIMS`` runs is via the survey simulation object, which is created via an input script file specifying the modules to be used and setting various input parameters.  The instantiation of this object, in turn, causes the instantiation of all other module objects.  Here is a quick example using a provided sample script:

.. code-block:: python

    import EXOSIMS,os.path
    scriptfile = os.path.join(EXOSIMS.__path__[0],'Scripts','sampleScript_coron.json')
    import EXOSIMS.MissionSim
    sim = EXOSIMS.MissionSim.MissionSim(scriptfile)

Once instantiated, the MissionSim object contains directly accessible instances of all modules (i.e., ``sim.Observatory``) as well as a dictionary of the modules (``sim.modules``).  A run simulation can be kicked off via the ``run_sim`` method of the SurveySimulation class:

.. code-block:: python
    
    res = sim.SurveySimulation.run_sim()

The terminal will display Observation #'s as well as any detections or characterizations that are made. The output is saved to the DRM variable. Which can be accessed by:

.. code-block:: python
    
    myDRM = sim.SurveySimulation.DRM
    myDRM

The terminal should display all the mission parameters of all observations made. If the last observation output of res = sim.SurveySimulation.run_sim() was #15, then accessing myDRM[14] would display the output of the 15th observation. To find the number of stars observed during my mission that have at least 1 planet detected, we could run:

.. code-block:: python
    
    count = 0
    for i in range(1,len(myDRM)):
        if 1 in myDRM[i]['det_status']:
            count = count+1
    print(count)


Above the basics
----------------

To run with the Forecaster Module, h5py must be installed.
See http://docs.h5py.org/en/latest/build.html
For Ubuntu users::
    $ pip install h5py

You also need to specify "PlanetPhysicalModel": "Forecaster", in the module portion of your .json file.

To run the WFIRSTObservatoryL2 module, you must have jplephem installed. Instructions can be found here https://pypi.python.org/pypi/jplephem
For Ubuntu users::
    $ pip install jplephem


