from EXOSIMS.Observatory.SotoStarshade_ContThrust import SotoStarshade_ContThrust
from EXOSIMS.Prototypes.TargetList import TargetList

import numpy as np
import sys
import ipyparallel as ipp


class SotoStarshade_parallel(SotoStarshade_ContThrust):
    """StarShade Observatory class
    This class is implemented at L2 and contains all variables, functions,
    and integrators to calculate occulter dynamics.

    Args:
        orbit_datapath (str, optional):
            Full path to reference orbit file
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        rc (ipyparallel.Client):
            Client
        dview (ipyparallel.Client):
            Direct view

    """

    def __init__(self, orbit_datapath=None, **specs):

        SotoStarshade_ContThrust.__init__(self, **specs)

        self.rc = ipp.Client()
        self.dview = self.rc[:]

        self.dview.execute("import numpy as np")
        self.dview.execute("import EXOSIMS")
        self.dview.execute("import EXOSIMS.Observatory.SotoStarshade_parallel as ens")
        self.dview.execute("import os")
        self.dview.execute("import os.path")
        self.dview.execute("import astropy.units as u")
        self.dview.execute("from astropy.time import Time")
        self.dview.execute("import pickle")
        self.dview.execute("import time")
        self.dview.execute("ensemble = ens.SotoStarshade_parallel()")

        self.dview.execute("from EXOSIMS.Prototypes.TargetList import TargetList")

        self.dview.block = False

        self.engine_ids = self.rc.ids
        self.nEngines = len(self.engine_ids)

    def run_ensemble(self, fun, nStars, tA, dtRange, m0, seed):
        """Execute Ensemble

        Args:
            fun (callable):
                run one method
            nStars (int):
                Number of stars
            tA (~astropy.time.Time):
                Current absolute mission time in MJD
            dtRange (~astropy.time.Time):
                Time range
            m0 (float):
                Initial mass
            seed (int):
                Random seed

        Returns:
            list:
                results

        """

        TL = TargetList(
            **{
                "ntargs": nStars,
                "seed": seed,
                "modules": {
                    "StarCatalog": "FakeCatalog",
                    "TargetList": " ",
                    "OpticalSystem": "Nemati",
                    "ZodiacalLight": "Stark",
                    "PostProcessing": " ",
                    "Completeness": " ",
                    "BackgroundSources": "GalaxiesFaintStars",
                    "PlanetPhysicalModel": " ",
                    "PlanetPopulation": "KeplerLike1",
                },
                "scienceInstruments": [{"name": "imager"}],
                "starlightSuppressionSystems": [{"name": "HLC-565"}],
            }
        )

        tlString = (
            f'TL = TargetList(**{{"ntargs": {int(nStars)}, "seed": {int(seed)}, '
            ' \'modules\': {"StarCatalog": "FakeCatalog" , "TargetList": " ", '
            '"OpticalSystem": "Nemati" , "ZodiacalLight": "Stark" , '
            '"PostProcessing": " ", "Completeness": " ", '
            '"BackgroundSources": "GalaxiesFaintStars" , '
            '"PlanetPhysicalModel": " ", "PlanetPopulation": "KeplerLike1"}, '
            '"scienceInstruments":  [{"name": "imager"}], '
            '"starlightSuppressionSystems": [{ "name": "HLC-565"}]  })'
        )

        self.dview.execute(tlString)

        sInds = np.arange(0, TL.nStars)
        ang = self.star_angularSep(TL, 0, sInds, tA)
        sInd_sorted = np.argsort(ang)
        angles = ang[sInd_sorted].to("deg").value

        self.dview["angles"] = angles
        self.dview["tA"] = tA
        self.dview["dtRange"] = dtRange
        self.dview["m0"] = m0
        self.dview["sInd_sorted"] = sInd_sorted
        self.dview["seed"] = seed
        self.lview = self.rc.load_balanced_view()

        async_res = []
        for j in range(int(TL.nStars)):
            print(sInd_sorted[j])
            ar = self.lview.apply_async(fun, sInd_sorted[j])
            async_res.append(ar)

        ar = self.rc._asyncresult_from_jobs(async_res)
        while not ar.ready():
            ar.wait(60.0)
            if ar.progress > 0:
                timeleft = ar.elapsed / ar.progress * (nStars - ar.progress)
                if timeleft > 3600.0:
                    timeleftstr = "%2.2f hours" % (timeleft / 3600.0)
                elif timeleft > 60.0:
                    timeleftstr = "%2.2f minutes" % (timeleft / 60.0)
                else:
                    timeleftstr = "%2.2f seconds" % timeleft
            else:
                timeleftstr = "who knows"

            print(
                "%4i/%i tasks finished after %4i min. About %s to go."
                % (ar.progress, nStars, ar.elapsed / 60, timeleftstr)
            )
            sys.stdout.flush()

        print("Tasks complete.")
        sys.stdout.flush()
        res = [ar.get() for ar in async_res]

        return res
