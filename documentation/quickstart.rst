.. _quickstart:
Quick Start Guide
######################

This is intended as a very brief overview of the steps necessary to get ``EXOSIMS`` running.  To start writing your own modules, please refer to the ICD and detailed as-built documentation of the prototypes.  Before using this guide, read through the :ref:`install` guide and follow all of the setup steps.


EXOSIMS Workflow
===========================

Creating a MissionSimulation
-------------------------------
The entry point to all ``EXOSIMS`` is via the ``MissionSimulation`` object, which is created via an input script file specifying the modules to be used and setting various input parameters.  The instantiation of this object, in turn, causes the instantiation of all other module objects.  Here is a quick example using a provided sample script:

.. code-block:: python

    import EXOSIMS,EXOSIMS.MissionSim,os.path
    scriptfile = os.path.join(EXOSIMS.__path__[0],'Scripts','sampleScript_coron.json')
    sim = EXOSIMS.MissionSim.MissionSim(scriptfile)

Once instantiated, the MissionSim object contains directly accessible instances of all modules (i.e., ``sim.Observatory``) as well as a dictionary of the modules (``sim.modules``).  At this point you can interrogate the ``sim`` object to see the results of the setup.  In particular, you can check the size of your initial target list (``sim.TargetList.nStars``) as well as their initial (single-visit) completeness values (``sim.TargetList.comp0``).

.. warning::
    The sampleScript_coron.json script uses the prototype Completeness module, which does not actually do any completeness calculations, but simply returns the same value for all stars.  To calculate completeness you must use one of the implementations: ``BrownCompleteness`` or ``GarrettCompleteness``.


You can also get a full list of all parameters (this includes the ones that weren't specified in your input script and were filled in by defaults) via the ``sim.genOutspec()`` method, which returns a dictionary of all simulation parameters.  Setting the ``tofile`` keyword will also write this dictionary out to the specified path.

.. note::
    The python JSON writer supports reading/writing values (such as infinity and nan) that are not in the JSON specification.  This means that output script files may only be parseable by python, or another parser supporting these extensions to the specification.


Running a Simulation and Analyzing Results
---------------------------------------------
The survey simulation is executed via the ``run_sim`` method. When running with default settings (as in ``sampleScript_coron.json``) the simulation details (observation numbers and detections/characterizations) will be printed as it is executed (this can be toggled off via the ``verbose`` script keyword). The full mission timeline is saved to the ``DRM`` variable in the SurveySimulation object, and can be accessed as:

.. code-block:: python
    
    sim.run_sim()
    DRM = sim.SurveySimulation.DRM

The ``DRM`` is a list of dictionaries, each representing one observations, so that a mission simulation of 10 observations will produce a ``DRM`` of length 10.  The dictionaries in ``DRM`` contain all of the details on each observation.  You can look at a full list of dictionary keywords by executing ``sim.SurveySimulation.DRM[0].keys()``. Of particular importance are:

* star_ind - The index of star observed.  You can get information about the star by using this index with the TargetList object.  For example ``sim.TargetList.Name[sim.SurveySimulation.DRM[0]['star_ind']]`` will return the name of the first star observed, and ``sim.TargetList.coords[sim.SurveySimulation.DRM[0]['star_ind']]`` will return its coordinates.
* plan_inds - The indices of all planets belonging to this star. You can get information about these planets by using this index with the SimulatedUniverse object. For example ``sim.SimulatedUniverse.a[sim.SurveySimulation.DRM[0]['plan_inds']]`` returns the semi-major axes of all planets in the system observed in the fist observation, and ``sim.SimulatedUniverse.Rp[sim.SurveySimulation.DRM[0]['plan_inds']]`` will return their radii.
* det_status - This encodes the outcome of the observation for each planet.  0 represents a missed detection, 1 represents a detection, and -1, -2 represent the planet being inside the IWA and outside the OWA, respectively.  ``sim.SurveySimulation.DRM[0]['plan_inds'][sim.SurveySimulation.DRM[0]['det_status'] == 1]`` will return the indices of all planets found in the first observation.

To find the number of stars observed during my mission that have at least 1 planet detected, we could run:

.. code-block:: python
    
    len([DRM[x]['star_ind'] for x in range(len(DRM)) if 1 in DRM[x]['det_status']])

``MissionSim`` also provides utilities for examining the ``DRM``.  The ``DRM2array`` method will return an array of all of the ``DRM`` entries for a specified key for the full ``DRM``.  So, running ``sim.DRM2array('plan_inds')`` will return and array of arrays of all the planet indices encountered (but not necessarily detected) during the full mission.  ``numpy.hstack(sim.DRM2array('plan_inds'))`` will flatten this array into a 1D list of all planet indices encountered.

The ``filter_status`` method will filter a provided key with a given status code.  ``sim.filter_status('plan_inds',0)`` will return all planet indices with missed detection throughout the full mission and ``sim.filter_status('plan_inds',1)`` will return the indices of all detected planets.


Running Additional Simulations
-----------------------------------
To run a new simulation using the same input scriptfile, simply reset the simulation and run it again. You can choose to generate new planets or to rewind the positions of the current set of planets to their initial states.  Setting both of these keywords to ``False`` will result in running a simulation that starts with all planets in their final states from the previous simulation.  

.. code-block:: python
    
    sim.reset_sim(genNewPlanets=True, rewindPlanets=True)
    sim.run_sim()

You can also run an ensemble of N simulations, which produces a list of DRMs. From there, you can find e.g. the number of observations made during each survey.

.. code-block:: python
    
    sim.reset_sim()
    N = 100
    ens = sim.run_ensemble(N, genNewPlanets=True, rewindPlanets=True)
    nb_obs = []
    for i in range(N):
        DRM = ens[i]
        nb_obs.append(len(DRM))

The default ensemble will run in sequence. For more details on ensembles and parallelization see :ref:`parallel`.


Building Your Own Mission
==============================

The only required components of the input specification are:

* The modules dictionary
* The science instruments list
* The starlight suppression systems list.
  
All other values will be filled in with defaults, although this will typically not produce a reasonable mission description, depending on the modules selected.  We begin with an empty set of modules, which would load all of the prototypes, and a single instrument and starlight suppression system, which will define the default observing mode.

.. code-block:: json
    
    {
     "modules": {
     "PlanetPopulation": " ",
     "StarCatalog": " ",
     "OpticalSystem": " ",
     "ZodiacalLight": " ",
     "BackgroundSources": " ",
     "PlanetPhysicalModel": " ",
     "Observatory": " ",
     "TimeKeeping": " ",
     "PostProcessing": " ",
     "Completeness": " ",
     "TargetList": " ",
     "SimulatedUniverse": " ",
     "SurveySimulation": " ",
     "SurveyEnsemble": " "
     },
     "scienceInstruments": [
     { "name": "imager" }
     ],
     "starlightSuppressionSystems": [
     {  "name": "coronagraph" }
     ]
    }

You can create a ``MissionSim`` object with this script, but it won't be particularly useful, since there are no real stars in the prototype ``StarCatalog``.  Now we must decide what kind of universe we will be modeling.  Let's select the EXOCAT-1 input catalog (http://nexsci.caltech.edu/missions/EXEP/EXEPstarlist.html), provided by the ``EXOCAT1`` star catalog and only model Earth-twins in the habitable zone.  We have two suitable ``PlanetPopulation`` implementations - ``EarthTwinHabZone1`` and ``EarthTwinHabZone2``, but we would like to override the defaults and only consider eccentricities between 0 and 0.35.  Our JSON script now becomes:

.. code-block:: json

    {
     "modules": {
     "PlanetPopulation": "EarthTwinHabZone2",
     "StarCatalog": "EXOCAT1",
     "OpticalSystem": " ",
     "ZodiacalLight": " ",
     "BackgroundSources": " ",
     "PlanetPhysicalModel": " ",
     "Observatory": " ",
     "TimeKeeping": " ",
     "PostProcessing": " ",
     "Completeness": " ",
     "TargetList": " ",
     "SimulatedUniverse": " ",
     "SurveySimulation": " ",
     "SurveyEnsemble": " "
     },
     "scienceInstruments": [
     { "name": "imager" }
     ],
     "starlightSuppressionSystems": [
     {  "name": "coronagraph" }
     ],
     "erange": [0, 0.3]
    }

We build a ``MissionSim`` object called ``sim`` using this script and then verify that our ``erange`` has overwritten the default by looking at the contents of ``sim.PlanetPopulation.erange``.  At this point, we should have a large number of stars (because the prototype Completeness isn't calculating the true completeness, and the default instrument settings will result in very low integration times for most stars).  Now we can describe the actual instrument.  We wish to model a 4 meter diameter, unobscured primary.  Our coronagraph will have an inner working angle of 100 mas and an outer working angle of 1 arcsecond, with a constant contrast of :math:`10^{-11}`. The JSON script now looks like this:

.. code-block:: json

    {
     "modules": {
     "PlanetPopulation": "EarthTwinHabZone2",
     "StarCatalog": "EXOCAT1",
     "OpticalSystem": " ",
     "ZodiacalLight": " ",
     "BackgroundSources": " ",
     "PlanetPhysicalModel": " ",
     "Observatory": " ",
     "TimeKeeping": " ",
     "PostProcessing": " ",
     "Completeness": " ",
     "TargetList": " ",
     "SimulatedUniverse": " ",
     "SurveySimulation": " ",
     "SurveyEnsemble": " "
     },
     "scienceInstruments": [
     { "name": "imager" }
     ],
     "starlightSuppressionSystems": [
     {  "name": "coronagraph",
        "IWA": 0.1,
        "OWA": 1.0,
        "core_contrast": 1.0e-11
     }
     ],
     "erange": [0, 0.3],
     "pupilDiam": 4.0,
     "obscurFac": 0.0,
    }


We again build a ``MissionSim`` object called ``sim`` using the updated script and check that our changes have been applied.  Running ``sim.OpticalSystem.starlightSuppressionSystems[0]['core_contrast'](sim.OpticalSystem.starlightSuppressionSystems[0]['lam'],sim.OpticalSystem.starlightSuppressionSystems[0]['IWA'])`` evaluates the contrast at the coronagraph central wavelength and inner working angle.  Running ``(sim.OpticalSystem.pupilDiam/2.0)**2.*np.pi - sim.OpticalSystem.pupilArea`` should return zero, verifying that the aperture is unobscured.

We will now replace the remaining prototype modules which don't perform the specific calculations and only return dummy values with full implementations.  We save this step for last as multiple setup calculations (and especially the completeness) can take a while, and we want everything else to be set before we run these.  We will use Nemati ``OpticalSystem``, the Brown ``Completeness`` (this is the Monte-Carlo version of the calculation) and the Stark ``ZodiacalLight`` modules:

.. code-block:: json

    {
     "modules": {
     "PlanetPopulation": "EarthTwinHabZone2",
     "StarCatalog": "EXOCAT1",
     "OpticalSystem": "Nemati",
     "ZodiacalLight": "Stark",
     "BackgroundSources": " ",
     "PlanetPhysicalModel": "Forecaster",
     "Observatory": " ",
     "TimeKeeping": " ",
     "PostProcessing": " ",
     "Completeness": "BrownCompleteness",
     "TargetList": " ",
     "SimulatedUniverse": " ",
     "SurveySimulation": " ",
     "SurveyEnsemble": " "
     },
     "scienceInstruments": [
     { "name": "imager" }
     ],
     "starlightSuppressionSystems": [
     {  "name": "coronagraph",
        "IWA": 0.1,
        "OWA": 1.0,
        "core_contrast": 1.0e-11
     }
     ],
     "erange": [0, 0.3],
     "pupilDiam": 4.0,
     "obscurFac": 0.0,
    }

Building the ``sim`` object will now take longer as the Monte Carlo completeness calculation executes.  Note that this will only happen once per script, as the completeness is cached on disk.   Looking at the new TargetList, we see that it has relatively few targets.  This is due to the completeness filtering.  This is controlled by two parameters: ``minComp`` and ``dMagLim``.  The former sets the cutoff below which targets are discarded, and the second sets the limiting :math:`\Delta`\mag of the dimmest planets of interest. The default values for these parameters (which can be confirmed either from the code, or by generating an outSpec dictionary, or by querying the parameters in the ``sim`` object) are 0.1 and 25, respectively.  Given that the population of Earth twins is typically dimmer than 25, these settings lead to relatively low completeness values.  On the other hand, setting ``dMagLim`` significantly higher will cause us to hit the noise floor set by our contrast (and all of the other optical system defaults) and thereby calculate infeasible integration times for more stars.  The best solution is to further flesh out the optical system by overriding more of the defaults, but for now we will simply decrease the minimum completeness. 

At the same time, we will make this a five year mission with one year of integration time dedicated to planet finding.   We also wish to only perform detections, and not spend any time on spectral characterizations.  This is controlled by setting the SNR to zero in the characterization observing mode.  Right now, there is only one observing mode that was automatically generated from the single instrument and starlight suppression system, so we will have to define a dummy spectrometer instrument and two modes - one for detection and one for characterization.  Our JSON script now looks like this:

.. code-block:: json

    {
     "modules": {
     "PlanetPopulation": "EarthTwinHabZone2",
     "StarCatalog": "EXOCAT1",
     "OpticalSystem": "Nemati",
     "ZodiacalLight": "Stark",
     "BackgroundSources": " ",
     "PlanetPhysicalModel": "Forecaster",
     "Observatory": " ",
     "TimeKeeping": " ",
     "PostProcessing": " ",
     "Completeness": "BrownCompleteness",
     "TargetList": " ",
     "SimulatedUniverse": " ",
     "SurveySimulation": " ",
     "SurveyEnsemble": " "
     },
     "scienceInstruments": [
     { "name": "imager" },
     { "name": "spectrometer" }
     ],
     "starlightSuppressionSystems": [
     {  "name": "coronagraph",
        "IWA": 0.1,
        "OWA": 1.0,
        "core_contrast": 1.0e-11
     }
     ],
     "erange": [0, 0.3],
     "pupilDiam": 4.0,
     "obscurFac": 0.0,
     "observingModes": [
        { "instName": "imager",
          "systName": "coronagraph",
          "detectionMode": true,
          "SNR": 5
        },
        { "instName": "spectrometer",
          "systName": "coronagraph",
          "SNR": 0
        }
     ],
     "minComp": 0.01,
     "dMagLim": 25,
     "missionLife": 5,
     "missionPortion": 0.2
    }

We are now ready to run our simulation.
