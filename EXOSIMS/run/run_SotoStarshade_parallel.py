import numpy as np
import os
import os.path
from astropy.time import Time
import astropy.units as u
import pickle
import time
import SotoStarshade_parallel as ens

ensemble = ens.SotoStarshade_parallel()


def get_dMMap(TL, sInd, angles, tA, dtRange, m0, filename):
    """Retrieving full trajectory from file"""

    dtFlipped = np.flipud(dtRange)

    dMmap = np.zeros(len(dtRange)) * u.kg
    eMap = 2 * np.ones(len(dtRange))

    tic = time.perf_counter()
    process_StartTime = time.localtime()
    for i, t in enumerate(dtFlipped):
        s_coll, t_coll, e_coll, TmaxRange = ens.collocate_Trajectory_minEnergy(
            TL, 0, sInd, tA, t, m0
        )

        # if unsuccessful, reached min time -> move on to next star
        if e_coll == 2 and t.value < 30:
            break

        m = s_coll[6, :]
        dm = m[-1] - m[0]
        dMmap[i] = dm
        eMap[i] = e_coll
        toc = time.perf_counter()
        process_CurrentTime = time.localtime()

        dmPath = os.path.join(ens.cachedir, filename + ".dmmap")
        A = {
            "dMmap": dMmap,
            "eMap": eMap,
            "angles": angles,
            "dtRange": dtRange,
            "time": toc - tic,
            "startTime": process_StartTime,
            "currentTime": process_CurrentTime,
            "mass": ens.mass,
            "tA": tA,
            "m0": m0,
            "ra": TL.coords.ra,
            "dec": TL.coords.dec,
        }
        with open(dmPath, "wb") as f:
            pickle.dump(A, f)

    return process_CurrentTime


if __name__ == "__main__":

    nStars = 50
    nSplits = 30

    dtRange = np.arange(10, 60, 5) * u.d
    m0 = 1
    # find a way to import missionStart time from TimeKeeping
    tA = Time(np.array(60634.0, ndmin=1, dtype=float), format="mjd", scale="tai")

    # run ensemble
    tic = time.perf_counter()
    res = ensemble.run_ensemble(get_dMMap, nStars, nSplits, tA, dtRange, m0)
    toc = time.perf_counter()
