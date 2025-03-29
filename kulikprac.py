import EXOSIMS.Observatory.KulikStarshade as KulikStarshade
import EXOSIMS.Observatory.SotoStarshade as SotoStarshade
import EXOSIMS.Prototypes.TargetList as TargetList
import numpy as np
import astropy.units as u
import time

starshade = KulikStarshade.KulikStarshade(
    mode="energyOptimal",
    dynamics=0,
    exponent=8,
    precompfname="Observatory/haloEnergy",
    starShadeRadius=27 * u.m,
)
starshadeSoto = SotoStarshade.SotoStarshade(f_nStars=101)
targets = TargetList.TargetList(
    modules={
        "StarCatalog": "EXOCAT1",
        "OpticalSystem": " ",
        "ZodiacalLight": "  ",
        "PostProcessing": " ",
        "Completeness": " ",
        "PlanetPopulation": "KeplerLike1",
        "BackgroundSources": " ",
        "PlanetPhysicalModel": " ",
    }
)

cur_time = time.time()
for i in range(targets.nStars):
    dV = starshade.calculate_dV(
        targets, 5, np.array([i]), 10 * np.ones((1, 50)), starshade.equinox
    )
    if i == 20:
        break
    print(dV)
    print(time.time() - cur_time)


# sd = starshade.star_angularSep(targets, 5, np.array([i]), starshade.equinox)
# print(sd)
# sotodV = starshadeSoto.calculate_dV(targets, 5, np.array([i]), sd, 15 *  np.ones((1, 50)) * u.d, starshade.equinox)
# print(sotodV)
