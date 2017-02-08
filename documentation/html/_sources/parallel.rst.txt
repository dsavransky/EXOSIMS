.. _parallel:
Survey Ensemble Parallelization
###################################

This describes the various ways to parallelize survey ensemble generation.

IPyParallel
--------------

The basic method for parallelization relies heavily on the `ipyparallel package <http://ipyparallel.readthedocs.org/en/latest/>`_.  This requires you to run a cluster, typically activated at the command line by ``ipcluster start``.  Then, from an ipython command prompt, create the ``MissionSim`` object as usual with a script including the ``IPClusterEnsemble`` SurveyEnsemble module:

.. code-block:: python

    import EXOSIMS,os.path
    scriptfile = os.path.join(EXOSIMS.__path__[0],'Scripts','parallel_ensemble_script.json')
    import EXOSIMS.MissionSim
    sim = EXOSIMS.MissionSim.MissionSim(scriptfile)

The ``sim`` object will have a ``SurveyEnsemble`` attribute with a ``run_ensemble`` method.  This method takes an argument of the ``run_one`` function, which must be defined in the ipython session:

.. code-block:: python
    
    def run_one():

        sim.run_sim()
        res = sim.DRM[:]
        sim.DRM = []
        sim.TimeKeeping.__init__(**sim.TimeKeeping._outspec)

        return res

Once defined, the run_one can be parallelized N times by running ``res = sim.SurveyEnsemble.run_ensemble(run_one, N)``.  On return, ``res`` will be a list of lists of the DRM dictionaries. 






