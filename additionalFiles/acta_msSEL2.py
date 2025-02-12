import EXOSIMS
from EXOSIMS.Observatory.SotoStarshade import SotoStarshade
import os.path
import numpy as np
import astropy.units as u

Obs = SotoStarshade(**{
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
  "thrust": 12*22*1000, \
  "slewIsp": 280, \
  "scMass": 14500, \
  "dryMass": 6722, \
  "occulterSep": 120E3, \
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
  "occ_dtmax":90, \
  "cachedir": "$HOME/.EXOSIMS/cache_SEL2_10182024", \
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
        "occulterDiameter": 72,
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
    "Observatory": "SotoStarshade",
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
