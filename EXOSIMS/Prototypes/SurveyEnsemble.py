from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_dirs import get_cache_dir
import time


class SurveyEnsemble(object):
    """:ref:`SurveyEnsemble` Prototype

    Args:
        cachedir (str, optional):
            Full path to cachedir.
            If None (default) use default (see :ref:`EXOSIMSCACHE`)
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        _outspec (dict):
            :ref:`sec:outspec`
        cachedir (str):
            Path to the EXOSIMS cache directory (see :ref:`EXOSIMSCACHE`)

    .. important::

        The prototype implementation provides no parallelization.
        See :ref:`SurveyEnsemble` for more info.

    """

    _modtype = "SurveyEnsemble"

    def __init__(self, cachedir=None, **specs):

        # start the outspec
        self._outspec = {}

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec["cachedir"] = self.cachedir
        specs["cachedir"] = self.cachedir

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get("verbose", True))

    def run_ensemble(
        self,
        sim,
        nb_run_sim,
        run_one=None,
        genNewPlanets=True,
        rewindPlanets=True,
        kwargs={},
    ):
        """
        Execute simulation ensemble

        Args:
            sim (:py:class:`EXOSIMS.MissionSim`):
                MissionSim object
            nb_run_sim (int):
                number of simulations to run
            run_one (callable):
                method to call for each simulation
            genNewPlanets (bool):
                Generate new planets each for simulation. Defaults True.
            rewindPlanets (bool):
                Reset planets to initial mean anomaly for each simulation.
                Defaults True
            kwargs (dict):
                Keyword arguments to pass onwards (not used in prototype)

        Returns:
            list(dict):
                List of dictionaries of mission results
        """

        SS = sim.SurveySimulation
        t1 = time.time()
        res = []
        for j in range(nb_run_sim):
            print("\nSurvey simulation number %s/%s" % (j + 1, int(nb_run_sim)))
            ar = self.run_one(
                SS, genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets
            )
            res.append(ar)
        t2 = time.time()
        self.vprint(
            "%s survey simulations, completed in %d sec" % (int(nb_run_sim), t2 - t1)
        )

        return res

    def run_one(self, SS, genNewPlanets=True, rewindPlanets=True):
        """
        Args:
            SS (:ref:`SurveySimulation`):
                SurveySimulation object
            genNewPlanets (bool):
                Generate new planets each for simulation. Defaults True.
            rewindPlanets (bool):
                Reset planets to initial mean anomaly for each simulation.
                Defaults True
        Returns:
            list(dict):
                Mission results
        """
        SS.reset_sim(genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets)
        SS.run_sim()
        res = SS.DRM[:]
        return res
