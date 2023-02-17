.. _surveyensemble:

SurveyEnsemble
###################################

Survey ensemble modules facilitate the generation of ensembles of mission simulations.

Prototype
==============

The :py:class:`~EXOSIMS.Prototypes.SurveyEnsemble` prototype does not provide any parallelization capability, but executes simulations in series.  It is intended only to provide the basic interface specification for generating survey ensembles, and can be useful for debugging purposes where you wish the output from each simulation to be displayed as it is executed. 

.. _run-ensemble:

The method for generating an ensemble is ``run_ensemble``, with inputs:

* ``sim`` - A :py:class:`~EXOSIMS.MissionSim` object
* ``nb_runs`` - The number of simulations to execute
* ``run_one`` - A method to execute for each simulation

There are two keywords: ``genNewPlanets`` and ``rewindPlanets``, which are both True by default and are passed to the ``run_one`` method.  All other keywords are passed directly to ``run_one``.


In the prototype implementation, ``run_one`` cannot be overwritten and is always defined as follows:

.. code-block:: python 
    
    def run_one(self, SS, genNewPlanets=True, rewindPlanets=True):
        SS.run_sim()
        res = SS.DRM[:]
        SS.reset_sim(genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets)
        return res
    

IPyParallel
===============

This implementation uses the `ipyparallel package <http://ipyparallel.readthedocs.org/en/latest/>`_ for paralellization.  This requires you to run a cluster, typically activated at the command line by ``ipcluster start -n 10``, where 10 is the number of workers to create (Note: If a worker from a cluster that was not shut down properly persists when the new cluster is created, this can cause run_ipcluster_ensemble.py to hang. It is recommended to terminate all your active python sessions).  Then, from an ipython command prompt, create the :py:class:`~EXOSIMS.MissionSim` object as usual with a script including the :py:class:`~EXOSIMS.SurveyEnsemble.IPClusterEnsemble` module:

.. code-block:: python

    import EXOSIMS, EXOSIMS.MissionSim, os.path
    scriptfile = os.path.join(EXOSIMS.__path__[0],'Scripts','sampleScript_parallel_ensemble.json')
    sim = EXOSIMS.MissionSim.MissionSim(scriptfile)


.. warning::

    It is important that you first run a version of your script with the prototype ``SurveyEnsemble`` to ensure that there are no errors and that all cached products are built on disk before execution on the cluster.  The MissionSim constructor takes a ``nopar`` keyword input.  Building the ``sim`` object with ``nopar=True`` will ignore any non-prototype ``SurveyEnsemble`` in your script file and build with the prototype.


The ``sim`` object will have a ``SurveyEnsemble`` attribute with a :ref:`run_ensemble <run-ensemble>` method, as described above.  This method takes an argument of the ``run_one`` function, which must be defined in the `main intepreter scope <https://docs.python.org/3/library/__main__.html>`_.

.. note::

     Because of the requirement for ``run_one`` to be in the main scope, you cannot import a ``run_one`` method from a file, or use an object method from an instantiated object.  Your easiest options are either to copy and paste your ``run_one`` method directly into the interpreter (preferably using the ipython %paste magic command), or to place the ``run_one`` method by itself in a file on disk, and then use the ``run`` command from the intepreter to load it into the main scope.  In either case, ``run_one`` should be defined in the interpreter session as ``<function __main__.run_one>``.

:py:class:`~EXOSIMS.SurveyEnsemble.IPClusterEnsemble`` imports the following modules to all workers in the cluster:

* EXOSIMS, EXOSIMS.util.get_module
* os, os.path
* time
* random
* cPickle
* traceback

and executes ``SS = EXOSIMS.util.get_module.get_module(specs['modules']['SurveySimulation'], 'SurveySimulation')(**specs)`` on each worker, generating a ``SurveySimulation`` object called ``SS`` on each worker using the parameters of your original input script.  If your particular ``run_one`` requires additional inputs or common pre-simulation commands to be executed, then you must modify the constructor (preferably by implementing a new ``SurveyEnsemble`` implementation that inherits ``IPClusterEnsemble``.

A simple ``run_one`` implementation is provided below:

.. code-block:: python
    
    def run_one(genNewPlanets=True, rewindPlanets=True):

        SS.run_sim()
        res = SS.DRM[:]
        SS.reset_sim(genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets)

        return res

.. warning::

    This version of ``run_one`` returns the full ``DRM`` list, meaning that all outputs will need to be collected in the main scope after the ensemble execution, potentially adding considerable overheads.  A better approach for large ensembles is to write each individual set of results to disk and return only a scalar value (or some other small output) to the main scope.

Once defined, the ``run_one`` method is executed in parallel by running:

.. code-block:: python

    res = sim.run_ensemble(N, run_one=run_one, **kwargs)

where ``kwargs`` are any kewyord arguments, or a dictionary of arguments that are passed to ``run_one``.

run_ipcluster_ensemble
-------------------------

To simplify parallel ensemble execution via ``IPClusterEnsemble``, ``EXOSIMS`` provides a run script called ``run_ipcluster_ensemble.py`` (located in the ``run`` directory - see: :ref:`exosimsdirs`).  This script is intended to be called from the command line, and is executed as:

.. code-block:: shell

    >python run_ipcluster_ensemble scriptname nruns

where ``scriptname`` is the full path to the JSON script to use, and ``nruns`` is the number of simulations to execute.  For full usage information, execute:

.. code-block:: shell

    >python run_ipcluster_ensemble --help

This script saves the results of each individual simulation to disk as a pickle file, containing a dictionary with two keys:

* ``DRM``: The full ``DRM`` list of dictionaries encoding the mission simulation
* ``systems``: A dictionary of planet parameters generated by the ``dump_systems`` method of the ``SimulatedUniverse`` object.

In addition, the script saves the output specification (generated by ``sim.genOutSpec()``) to the same directory as the rest of the results, and saves the traceback of any error generated on any worker during ensemble execution to a ``log.err`` file in the output directory.

read_ipcluster_ensemble
--------------------------

To read in and parse the pickle files generated by ``run_ipcluster_ensemble`` we use :py:class:`EXOSIMS.util.read_ipcluster_ensemble` which provides a :py:meth:`~EXOSIMS.util.read_ipcluster_ensemble.gen_summary` method.  This generates lists of detection and characterization parameters for all missions in an ensemble.


