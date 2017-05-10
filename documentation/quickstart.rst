.. _quickstart:
Quick Start Guide
######################

This is intended as a very brief overview of the steps necessary to get ``EXOSIMS`` running.  To start writing your own modules, please refer to the ICD and detailed as-built documentation of the prototypes.

Setting Up Your Environment
---------------------------

``EXOSIMS`` requires Python 2.7 and multiple packages including numpy and astropy.  All required modules are listed in the ICD and can be obtained from PyPI or other package resources such as Macports.


The directory containing the ``EXOSIMS`` directory must be in your Python path.  On POSIX systems, this is accomplished simply by appending the path to the ``PYTHONPATH`` environment variables.

Setting PYTHONPATH in Ubuntu
----------------------------

To append the ``EXOSIMS`` directory to your ``PYTHONPATH`` locate your .bashrc file (should be in your home directory /home/username) and append the following line to the end.
::
    export PYHTONPATH="$PYTHONPATH:/home/username/PATH TO EXOSIMS PARENT DIRECTORY"

Be sure to execute ``$ source ~/.bashrc``

An example: If ``EXOSIMS`` were pulled from git into the folder /home/user1/TACO such that TACO contains the folders EXOSIMS ICD and documentation, then my PYTHONPATH is
::
    export PYTHONPATH="$PYTHONPATH:/home/user1/TACO"

Setting PYTHONPATH in WINDOWS
-----------------------------
Find My Computer > Properties > Advanced Systems Settings > Environment Variables > then under system variables add a new variable called PYTHONPATH or append to it. In this variable you need to have C:\\PATH TO EXOSIMS PARENT DIRECTORY


Creating a Mission Simulation Object
-------------------------------------

The entry point to all ``EXOSIMS`` runs is via the survey simulation object, which is created via an input script file specifying the modules to be used and setting various input parameters.  The instantiation of this object, in turn, causes the instantiation of all other module objects.  Here is a quick example using a provided sample script:

.. code-block:: python

    import EXOSIMS,EXOSIMS.MissionSim,os.path
    scriptfile = os.path.join(EXOSIMS.__path__[0],'Scripts','sampleScript_coron.json')
    sim = EXOSIMS.MissionSim.MissionSim(scriptfile)

Once instantiated, the MissionSim object contains directly accessible instances of all modules (i.e., ``sim.Observatory``) as well as a dictionary of the modules (``sim.modules``).  A survey simulation can be kicked off via the ``run_sim`` method. The terminal will display Observation #'s as well as any detections or characterizations that are made. The output is saved to the Design Reference Mission (DRM) variable, which contains all the mission parameters of all observations made. 

.. code-block:: python
    
    sim.run_sim()
    DRM = sim.SurveySimulation.DRM

If e.g. 10 observations were made, accessing DRM[4] would display the output of the 5th observation. To find the number of stars observed during my mission that have at least 1 planet detected, we could run:

.. code-block:: python
    
    len([DRM[x]['star_ind'] for x in range(len(DRM)) if 1 in DRM[x]['det_status']])

To run a new simulation using the same input scriptfile, simply reset the simulation and run it again. You can choose to generate new planets and/or rewind their positions. 

.. code-block:: python
    
    sim.reset_sim(genNewPlanets=True, rewindPlanets=True)
    sim.run_sim()

You can also run an ensemble of N simulations, wich produces a list of DRMs. From there, you can find e.g. the number of observations made during each survey.

.. code-block:: python
    
    sim.reset_sim()
    N = 100
    ens = sim.run_ensemble(N, genNewPlanets=True, rewindPlanets=True)
    nb_obs = []
    for i in range(N):
        DRM = ens[i]
        nb_obs.append(len(DRM))

To run with the Forecaster Module, h5py must be installed.
See http://docs.h5py.org/en/latest/build.html
For Ubuntu users::
    $ pip install h5py

You also need to specify "PlanetPhysicalModel": "Forecaster", in the module portion of your .json file.

To run the WFIRSTObservatoryL2 module, you must have jplephem installed. Instructions can be found here https://pypi.python.org/pypi/jplephem
For Ubuntu users::
    $ pip install jplephem

To use Hybrid Lyot Coronagraph (HLC) contrast curves, you first need to download the contrast curves from https://wfirst.ipac.caltech.edu/sims/Coronagraph_public_images.html#CGI_Performance labeled "HLC files". Move these to a folder of your choosing. To specifiy these curves for the HLC-565, it is sufficient to change values of occ_trans, core_thruput, core_mean_intensity, and core_area to the path of the corresponding .fits file in the .json script.


