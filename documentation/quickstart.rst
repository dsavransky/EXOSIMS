.. _quickstart:

Quick Start Guide
######################

This is intended as a very brief overview of the steps necessary to get ``EXOSIMS`` running.  To start writing your own modules, please refer to the `ICD <https://cdn.rawgit.com/dsavransky/EXOSIMS/master/ICD/icd.pdf>`_ and detailed as-built documentation of the prototypes.  Before using this guide, read through the :ref:`install` guide and follow all of the setup steps.


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


You can also get a full list of all parameters (this includes the ones that weren't specified in your input script and were filled in by defaults) via the ``sim.genOutSpec()`` method, which returns a dictionary of all simulation parameters.  Setting the ``tofile`` keyword will also write this dictionary out to the specified path.

.. note::
    The python JSON writer supports reading/writing values (such as infinity and nan) that are not in the JSON specification.  This means that output script files may only be parseable by python, or another parser supporting these extensions to the specification.

.. _runsimandanalyze:

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

.. _buildamission:

Building Your Own Mission
==============================

This is a brief guide to iteratively building up a simulation script, with comments and sanity checks along the way.  It touches on only a subset of all possible user settings for the base modules.  A more complete list is available here: :ref:`userparams`.

Step 1
--------

The only required components of the input specification are:

* The modules dictionary
* The science instruments list
* The starlight suppression systems list.
  
All other values will be filled in with defaults, although this will typically not produce a reasonable mission description, depending on the modules selected.  We begin with an empty set of modules, which would load all of the prototypes, and a single instrument and starlight suppression system, which will define the default observing mode. In a directory of your choosing (preferably outside of ``EXOSIMSROOT`` - see :ref:`here <EXOSIMSROOT>`), create a file called ``test.json`` with the following contents:

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

You can create a ``MissionSim`` object with this script, but it won't be particularly useful, since there are no real stars in the prototype ``StarCatalog``.  We'll do it anyway to sanity check that the code is working.  In a python interpreter running in the same directory as your test script run:

.. code-block:: python

    import EXOSIMS.MissionSim
    sim = EXOSIMS.MissionSim.MissionSim('test.json')

You should see outputs showing the modules being loaded as the simulation object is instantiated, along the lines of ::

    Imported SurveyEnsemble (prototype module) from EXOSIMS.Prototypes.SurveyEnsemble
    Imported SurveySimulation (prototype module) from EXOSIMS.Prototypes.SurveySimulation
    Imported SimulatedUniverse (prototype module) from EXOSIMS.Prototypes.SimulatedUniverse
    Imported TargetList (prototype module) from EXOSIMS.Prototypes.TargetList
    Imported StarCatalog (prototype module) from EXOSIMS.Prototypes.StarCatalog
    Imported OpticalSystem (prototype module) from EXOSIMS.Prototypes.OpticalSystem
    Imported ZodiacalLight (prototype module) from EXOSIMS.Prototypes.ZodiacalLight
    Imported PostProcessing (prototype module) from EXOSIMS.Prototypes.PostProcessing
    Imported BackgroundSources (prototype module) from EXOSIMS.Prototypes.BackgroundSources
    Imported Completeness (prototype module) from EXOSIMS.Prototypes.Completeness
    Imported PlanetPopulation (prototype module) from EXOSIMS.Prototypes.PlanetPopulation
    Imported PlanetPhysicalModel (prototype module) from EXOSIMS.Prototypes.PlanetPhysicalModel
    Imported Observatory (prototype module) from EXOSIMS.Prototypes.Observatory
    Imported TimeKeeping (prototype module) from EXOSIMS.Prototypes.TimeKeeping
    Numpy random seed is: 491873991

Printing the contents of ``sim.TargetList.nStars`` and ``sim.SimulatedUniverse.plan2star`` will show that this simulation has one (fake) star with one simulated planet (``plan2star`` is an array of indices mapping planet attributes to stars - in this case it is a single element array mapping to star 0). This planet is generated with properties that ensure that it is detectable with all of the default settings in the other modules.

Step 2
-------

Now we must decide what kind of universe we will be modeling.  Let's select the EXOCAT-1 input catalog (http://nexsci.caltech.edu/missions/EXEP/EXEPstarlist.html), provided by the ``EXOCAT1`` ``StarCatalog`` implementation and only model Earth-twins in the habitable zone.  We have two suitable ``PlanetPopulation`` implementations - ``EarthTwinHabZone1`` and ``EarthTwinHabZone2``, but we would like to override the defaults and only consider eccentricities between 0 and 0.35 so we will use ``EarthTwinHabZone2`` (``EarthTwinHabZone1`` does not allow for overriding orbital parameters).  Our JSON script now becomes:

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

We again build a ``MissionSim`` object called ``sim`` using this script and then verify that our ``erange`` has overwritten the default by looking at the contents of ``sim.PlanetPopulation.erange`` and by printing ``sim.SimulatedUniverse.e.min(), sim.SimulatedUniverse.e.max()``.  The former shows us the range used in sampling by the ``PlanetPopulation`` while the latter shows the range of values actually sampled when creating the simulated universe.

Another important thing to note is that the ``EarthTwinHabZone2`` populations set the ``constrainOrbits`` keyword to ``True`` by default.   This flag forces all orbital radii to be within the semi-major axis range (so that :math:`a(1+e) \le a_\mathrm{max}` and  :math:`a(1-e) \ge a_\mathrm{min}`). At the same time, the ``EarthTwinHabZone`` implementations also set the ``scaleOrbits`` flag to ``True``, which causes the semi-major axes to be scaled by the square root of the stellar luminosities as they are generated in the ``SimulatedUniverse``.  To verify that these things are happening we can execute the following:

.. code-block:: python

    import numpy as np
    Ls = sim.TargetList.L[sim.SimulatedUniverse.plan2star]
    smas = sim.SimulatedUniverse.a/np.sqrt(Ls)
    print(np.all((smas <= sim.PlanetPopulation.arange[1]) & (smas >= sim.PlanetPopulation.arange[0])))
    print(np.all((smas*(1+sim.SimulatedUniverse.e) <= sim.PlanetPopulation.arange[1]) & (smas*(1-sim.SimulatedUniverse.e) >= sim.PlanetPopulation.arange[0])))

The ``plan2star`` attribute maps the simulated planets to their parent stars in the target list object, allowing us to extract the stellar luminosities.  Both of the logical tests should evaluate to ``True`` (both the semi-major axes and extrema of the orbital radii should fall within the semi-major axis range with the default flags).

Another thing to test is that we are generating the proper number of planets.  In this population, this is controlled by the ``eta`` parameter (also settable in the JSON script), which defaults to 0.1, meaning that we expect one planet per ten stars, on average.  As these are generated probabilistically, we will not have an exact occurrence rate of 0.1 in any given simulation, but over many simulations, we should expect to average to this rate.  We can explicitly test this by executing the following:

.. code-block:: python

    rate = 0
    for j in range(100):
        rate += float(len(sim.SimulatedUniverse.plan2star))/sim.TargetList.nStars
        sim.reset_sim()

    print(rate/100.0)

The rate should be very nearly 0.1 (with standard Poisson error).

At this point, we should have a large number of stars in our target list (verify by printing ``sim.TargetList.nStars``) because the prototype Completeness isn't calculating the true completeness, and the default instrument settings will result in very low integration times for most stars, meaning that they won't be filtered out based on your integration time cutoff, encoded in ``sim.OpticalSystem.intCutoff`` with a default value of 50 days, and also settable as ``intCutoff`` in the JSON script.  The filtering works by calculating the minimum necessary integration time (with no zodiacal light contribution) for a planet of ``sim.OpticalSystem.dMag0`` at a working angle of ``sim.OpticalSystem.WA0`` (both of these also settable in the JSON script as ``dMag0`` and ``WA0``, respectively. The default ``dMag0`` is 15 (:math:`10^{-6}` contrast), meaning that the vast majority of targets are retained. 

Step 3
-------

Now we can describe the actual instrument.  We wish to model a 4 meter diameter, unobscured primary.  Our coronagraph will have an inner working angle of 100 mas and an outer working angle of 1 arcsecond, with a constant contrast of :math:`10^{-11}`. We will assume a modest post-processing factor of 0.1 (meaning that we can reduce residual speckle noise by one order of magnitude via post-processing). The JSON script now looks like this:

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
     "ppFact": 0.1
    }


We again build a ``MissionSim`` object called ``sim`` using the updated script and check that our changes have been applied.  Running:

.. code-block:: python
    
    sim.OpticalSystem.starlightSuppressionSystems[0]['core_contrast'](sim.OpticalSystem.starlightSuppressionSystems[0]['lam'],sim.OpticalSystem.starlightSuppressionSystems[0]['IWA'])
    
evaluates the contrast at the coronagraph central wavelength and inner working angle and should return our input constant contrast.  Running:

.. code-block:: python

    sim.OpticalSystem.pupilDiam**2.*sim.OpticalSystem.shapeFac - sim.OpticalSystem.pupilArea

should return zero, verifying that the aperture is unobscured. ``shapeFac`` is another user-settable parameter, and is defined such that its product with the square of the aperture diameter gives the pupil area (it defaults to the value for circular apertures).  

Looking at ``sim.TargetList.nStars``, we see that our target list is now significantly smaller than it was before.  This is directly a consequence of setting an inner and outer working angle for our coronagraph (the default values are zero to infinity).  Due to the limited nature of the selected planet population, and finite IWA/OWA instantly filters out the majority of stars, for which the entire planet population would fall outside of this coronagraph's operating angular separation range.

Step 4
--------

We will now replace the remaining prototype modules which don't perform the specific calculations and only return dummy values with full implementations.  We will use:

* The Nemati ``OpticalSystem`` (integration time calculations are based on the equations found in [Nemati2014]_) 
* The Brown ``Completeness`` (this is the Monte-Carlo version of the calculation, based on [Brown2005]_; alternatively, we have ``GarrettCompletness`` which is a fully analytical implementation based on [Garrett2016]_)
* The Stark ``ZodiacalLight`` module (the local zodi is based on modeling from [Stark2014]_)
* The Forecaster ``PlanetPhysicalModel`` implementation (this uses Forecaster [Chen2016]_ to probabilistically calculate planet densities)

Our JSON script now looks as follows:

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
     "ppFact": 0.1
    }


Building the ``sim`` object will now take considerably longer as the Monte Carlo completeness calculation executes (and the output will include status messages regarding this calculation).  Note that this will only happen once per script, as the completeness is cached on disk.   
Looking at the new TargetList, we see that it has relatively few targets.  This is due to the completeness filtering.  This is controlled by two parameters: ``minComp`` and ``dMagLim``.  The former sets the cutoff below which targets are discarded, and the second sets the limiting :math:`\Delta`\mag of the dimmest planets of interest (the effective instrumental contrast floor used in the completeness calculation). The default values for these parameters (which can be confirmed either from the code, or by generating an outSpec dictionary, or by querying the parameters in the ``sim.Completeness`` object) are 0.1 and 25, respectively.  Given that the population of Earth twins is typically dimmer than 25, these settings lead to relatively low completeness values. 

If we wish to expand our initial target list, we can change ``dMagLim`` or ``minComp`` (or both).  It is important to note that the ``dMagLim`` parameter value serves as the default for the ``dMagint`` parameter in the ``SurveySimulation`` module, which (in the prototype implementation) sets the target planet magnitude used in determining integration times for each target.  Increasing ``dMagLim`` without changing ``dMagInt`` will therefore cause integration times to grow, and may potentially waste a lot of mission time. We therefore allow for independent setting of these two parameters. However, once you select a ``dMagInt`` that is different from the ``dMagLim``, you explicitly decouple the completeness from the execution of the survey (this is not a large consideration, as the two are always fundamentally different, but is important to remember when interpreting results).



Step 5
----------
Finally, we will fill in a few more mission details.  We will make this a five year mission with one year of integration time dedicated to planet finding.   We also wish to only perform detections, and not spend any time on spectral characterizations.  This is achieved by setting the SNR to zero in the characterization observing mode.  Right now, there is only one observing mode that is automatically generated from the single instrument and starlight suppression system (stored in ``sim.OpticalSystem.observingModes``), so we will have to define a dummy spectrometer instrument and two modes - one for detection and one for characterization.  Our JSON script now looks like this:

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
     "ppFact": 0.1,
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
     "dMagLim": 26,
     "missionLife": 5,
     "missionPortion": 0.2
    }

After creating a new ``sim`` object with this script, we are now ready to run our simulation. We execute ``sim.run_sim()`` and the simulation progress is printed as it runs, terminating somewhere near 1826.25 days (the actual mission end time will depend on the specific observations scheduled).

.. note::
    
    It is possible for the mission end time to be greater than the mission lifetime as observations are not interrupted if they extend past the end of the nominal mission life.  However, no new observations will be scheduled after this point.

We can now use the same tools as described in :ref:`runsimandanalyze` to analyze the results.


Creating Synthetic Universes
==============================
In some instances, you may wish to use EXOSIMS's synthetic universe generation capabilities without wanting to set up a full mission simulation (and all of the overhead that goes with it).  You can do so by directly instantiating a ``SimulatedUniverse`` object. This requires only a subset of modules to be instantiated, namely:

#. TargetList
#. StarCatalog
#. PlanetPopulation
#. PlanetPhysicalModel
#. OpticalSystem
#. ZodiacalLight
#. BackgroundSources
#. PostProcessing
#. Completeness
#. SimulatedUniverse

While you probably don't care about several of these, they are needed to build the TargetList, and you can just specify their Prototype implementations.  In particular, the prototype Completeness implementation returns values of 0.2 for every target, and so can be used to retain all targets regardless of their actual completeness values under your selected planet population. You can create a JSON script as in :ref:`buildamission`, and then read it in like so:

.. code-block:: python

    import json
    with open(scriptfile) as ff:
         specs = json.loads(ff.read())

or, alternatively, just define a specs dictionary in your python session.  For example, if we wanted to build a Kepler-like simulated universe based on the EXOCAT-1 catalog, then a minimal specification would look like this:

.. code-block:: python

   specs = {"modules": {
         "PlanetPopulation": "KeplerLike2",
         "StarCatalog": "EXOCAT1",
         "OpticalSystem": "Nemati",
         "ZodiacalLight": "Stark",
         "BackgroundSources": " ",
         "PlanetPhysicalModel": "FortneyMarleyCahoyMix1",
         "PostProcessing": " ",
         "Completeness": " ",
         "TargetList": " ",
         "SimulatedUniverse": "KeplerLikeUniverse" },
         "scienceInstruments": [{ "name": "imager"}],
         "starlightSuppressionSystems": [{ "name": "coronagraph"}],
         "explainFiltering": True}

The ``explainFiltering`` key will cause EXOSIMS to print out how the target list is being filtered based on the other modules.  You can control this behavior by setting other inputs, as described in the documentation for individual modules. Once the specs dictionary is defined, you can instantiate your Simulated Universe as:

.. code-block:: python

   import EXOSIMS.SimulatedUniverse.KeplerLikeUniverse
   SU = EXOSIMS.SimulatedUniverse.KeplerLikeUniverse.KeplerLikeUniverse(**specs)

.. warning::
   The instantiation of this object will modify the ``specs`` dictionary in such a way that you will not be able to instantiate another instance from it.  If you wish to preserve its form, make a copy (not assignment) of ``specs`` prior to running the above code.

You can now interact with the ``SU`` object as usual.  All of the planet properties are stored as numpy arrays as documented in the SimulatedUniverse docstrings and the ICD.


References
============

.. [Nemati2014] Nemati, Bijan (2014) Detector selection for the WFIRST-AFTA coronagraph integral field spectrograph, Proc. SPIE, 91430
.. [Brown2005] Brown, R. A. (2005) Single-visit photometric and obscurational completeness, ApJ 624
.. [Garrett2016] Garett, D. and Savransky, D. (2016) Analytical Formulation of the Single-visit Completeness Joint Probability Density Function, ApJ 828(1)
.. [Stark2014] Stark, C., Roberge, A., Mandell, A., and Robinson, T. D. (2014) Maximizing the ExoEarth Candidate Yield from a Future Direct Imaging Mission, ApJ 795(2)
.. [Chen2016] Chen, J. and Kipping, D. M. (2016) Probabilistic Forecasting of the Masses and Radii of Other Worlds, ApJ 834(1)

