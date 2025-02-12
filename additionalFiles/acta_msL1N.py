import EXOSIMS
from EXOSIMS.Observatory.SotoStarshadeMoon import SotoStarshadeMoon
import os.path
import numpy as np
import astropy.units as u

path_f1 = os.path.normpath(os.path.expandvars("$HOME/Documents/Research/testFilesActa/filesLunarSSL1N.txt"))
f1 = open(path_f1, "r")
fpath = os.path.normpath(os.path.expandvars("$HOME/Documents/Research/testFilesActa/EXOSIMS/EXOSIMS/Observatory/"))

#d = np.array([120000, 100000, 80000, 60000, 40000, 20000, 10000, 7500])
d = np.array([10000, 7500])

Isp = 280*u.s
g = 9.81*u.m/u.s**2
ue = Isp*g
md_0 = 6722*u.kg
mw_0 = np.exp(60/ue.value)*md_0
D_0 = 72*u.m
R_0 = D_0/2
d_0 = 120E3*u.km
tanTheta = (R_0/d_0).decompose()

for jj in f1:
    fpath = jj[0:-1]
    fname = os.path.split(jj[0:-1])[1][0:-4]

    dtMax = float(fname.split('_')[-2])/2
    
    for occSep_p in d:
        print(fname)
        print(str(occSep_p))
        R_p = (occSep_p*u.km*tanTheta).to('m')
        D_p = (R_p*2).value
        
        md_p = (md_0*(R_p/R_0)**2).value
        mw_p = np.exp(60/ue.value)*md_p

        Obs = SotoStarshadeMoon(**{
          "seed": 968390887, \
          "missionLife": 5.0, \
          "missionStart": 64681, \
          "missionPortion": 0.5, \
          "GAPortion_SS": 0.5, \
          "GA_simult_det_fraction": 0.0, \
          "int_inflection": False, \
          "minComp": 0.1, \
          "pupilDiam": 6, \
          "obscurFac": 0.0, \
          "dMagLim": 26.0, \
          "dMagLim_offset": 1, \
          "magEZ": 22.4205, \
          "intCutoff": 60.0, \
          "contrast_floor_notyet": 1E-10, \
          "texp_flag": True, \
          "scaleWAdMag": True, \
          "dMagint": 23, \
          "WAint":0.2, \
          "dMag0":25, \
          "WA0":0.2, \
          "charMargin": 0.01, \
          "thrust": 22*12*1000, \
          "slewIsp": 280, \
          "scMass": mw_p, \
          "dryMass": md_p, \
          "occulterSep": occSep_p, \
          "nSteps":1, \
          "missionStart":64041, \
          "equinox":64041, \
          "nVisitsMax": 10, \
          "occ_max_visits": 1, \
          "revisit_wait": 0.335, \
          "phase1_end": 16, \
          "revisit_weight": 10, \
          "n_det_remove": 2, \
          "n_det_min": 3, \
          "max_successful_chars": 1, \
          "max_successful_dets": 4, \
          "nmax_promo_det": 3, \
          "occ_dtmax":dtMax, \
          "cachedir": "$HOME/.EXOSIMS/cache_lunarSSL1N_12172024_72", \
          "orbit_datapath": fpath, \
          "orbit_filename": fname, \
          "scienceInstruments": [
            { "name": "imager"
            },
            { "name": "spectro"
            }
          ],
          "starlightSuppressionSystems": [
            { "name": "occulter",
                "occulter": True,
                "lam": 650,
                "IWA": 0.0287,
                "OWA": 3.933,
                "ohTime": 0.33,
                "BW": 1.08,
                "optics": 1,
                "core_platescale": 0.1667,
                "occ_trans": "/Users/gracegenszler/Documents/Research/testFilesActa/TV3_occ_trans_asec_650_6m.fits",
                "occulterDiameter": D_p,
                "NocculterDistances" :1,
                "occulterDistances": [
                  {
                  "occulterDistance": 120000,
                  "occulterRedEdge": 1000,
                  "occulterBlueEdge": 300
                  }
              ]
            }
          ],
          "modules": {
            "PlanetPopulation": "DulzPlavchan",
            "StarCatalog": "EXOCAT1",
            "OpticalSystem": "Nemati",
            "ZodiacalLight": "Stark",
            "BackgroundSources": "GalaxiesFaintStars",
            "PlanetPhysicalModel": "Forecaster",
            "Observatory": "SotoStarshadeMoon",
            "TimeKeeping": " ",
            "PostProcessing": " ",
            "Completeness": "BrownCompleteness",
            "TargetList": " ",
            "SimulatedUniverse": "DulzPlavchanUniverse",
            "SurveySimulation": "linearJScheduler",
            "SurveyEnsemble": " "
          },
        "completeness_specs":{
            "modules":{
                "PlanetPopulation": "EarthTwinHabZone1SDET",
                "PlanetPhysicalModel": " "
            }
        }
        }
        )
