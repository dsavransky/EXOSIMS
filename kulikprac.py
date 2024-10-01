import EXOSIMS.Observatory.KulikStarshade as KulikStarshade
import EXOSIMS.Observatory.SotoStarshade as SotoStarshade
import EXOSIMS.Prototypes.TargetList as TargetList
import numpy as np



starshade = KulikStarshade.KulikStarshade(mode="impulsive", dynamics=0, exponent=8, precompfname="Observatory/haloImpulsive", starShadeRadius = 27)
starshadeSoto = SotoStarshade.SotoStarshade(f_nStars=101)

targets = TargetList.TargetList(modules= {"StarCatalog": "EXOCAT1", "OpticalSystem": " ", "ZodiacalLight": "  ", "PostProcessing": " ", "Completeness": " ", "PlanetPopulation" : "KeplerLike1", "BackgroundSources" : " ", "PlanetPhysicalModel" : " "})

for i in range(targets.nStars):
    print(type(starshade.equinox))
    dV = starshade.calculate_dV(targets, 5, np.array([i]), 10 * np.ones((1, 50)), starshade.equinox)
    print(dV)
   # sd = starshade.star_angularSep(targets, 5, np.array([i]), starshade.equinox)
   # print(sd)
   # sotodV = starshadeSoto.calculate_dV(targets, 5, np.array([i]), sd, 15 *  np.ones((1, 50)) * u.d, starshade.equinox)
   # print(sotodV)