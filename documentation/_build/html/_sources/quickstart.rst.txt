.. _quickstart:
Quick Start Guide
######################

This is intended as a very brief overview of the steps necessary to get ``EXOSIMS`` running.  To start writing your own modules, please refer to the ICD and detailed as-built documentation of the prototypes.

Environment
--------------

``EXOSIMS`` requires Python 2.7 and multiple packages including numpy and astropy.  All required modules are listed in the ICD and can be obtained from PyPI or other package resources such as Macports.

The directory containing the ``EXOSIMS`` directory must be in your Python path.  On POSIX systems, this is accomplished simply by appending the path to the ``PYTHONPATH`` environment variables.  


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


