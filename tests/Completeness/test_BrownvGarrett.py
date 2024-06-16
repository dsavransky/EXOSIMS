import unittest
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams
import EXOSIMS.Completeness.BrownCompleteness
import EXOSIMS.Completeness.GarrettCompleteness
from EXOSIMS.Prototypes.TargetList import TargetList
from EXOSIMS.util.get_module import get_module
import os
import pkgutil
import numpy as np
import astropy.units as u
import json
import copy


class TestBrownvGarrett(unittest.TestCase):
    """

    Compare results of Brown and Garrett completeness modules

    """

    def setUp(self):

        self.dev_null = open(os.devnull, "w")
        self.script = resource_path("test-scripts/template_nemati.json")
        with open(self.script) as f:
            self.spec = json.loads(f.read())

    @unittest.skip("Skipping Garrett vs Brown comparison tests.")
    def test_target_completeness_def(self):
        """
        Compare calculated completenesses for multiple targets under default population
        settings.
        """

        with RedirectStreams(stdout=self.dev_null):
            TL = TargetList(ntargs=100, **copy.deepcopy(self.spec))

            mode = list(
                filter(
                    lambda mode: mode["detectionMode"] == True,
                    TL.OpticalSystem.observingModes,
                )
            )[0]
            IWA = mode["IWA"]
            OWA = mode["OWA"]
            rrange = TL.PlanetPopulation.rrange
            maxd = (rrange[1] / np.tan(IWA)).to(u.pc).value
            mind = (rrange[0] / np.tan(OWA)).to(u.pc).value

            # want distances to span from outer edge below IWA to inner edge above OWA
            TL.dist = (
                np.logspace(np.log10(mind / 10.0), np.log10(maxd * 10.0), TL.nStars)
                * u.pc
            )

        Brown = EXOSIMS.Completeness.BrownCompleteness.BrownCompleteness(
            **copy.deepcopy(self.spec)
        )
        Garrett = EXOSIMS.Completeness.GarrettCompleteness.GarrettCompleteness(
            **copy.deepcopy(self.spec)
        )

        cBrown = Brown.target_completeness(TL)
        cGarrett = Garrett.target_completeness(TL)

        np.testing.assert_allclose(cGarrett, cBrown, rtol=0.1, atol=1e-6)

        # test when scaleOrbits == True
        TL.L = np.exp(
            np.random.uniform(low=np.log(0.1), high=np.log(10.0), size=TL.nStars)
        )
        Brown.PlanetPopulation.scaleOrbits = True
        Garrett.PlanetPopulation.scaleOrbits = True

        cBrown = Brown.target_completeness(TL)
        cGarrett = Garrett.target_completeness(TL)

        cGarrett = cGarrett[cBrown != 0]
        cBrown = cBrown[cBrown != 0]
        meandiff = np.mean(np.abs(cGarrett - cBrown) / cBrown)

        self.assertLessEqual(meandiff, 0.1)

    @unittest.skip("Skipping Garrett vs Brown comparison tests.")
    def test_target_completeness_constrainOrbits(self):
        """
        Compare calculated completenesses for multiple targets with constrain orbits set to true
        """

        with RedirectStreams(stdout=self.dev_null):
            TL = TargetList(
                ntargs=100, constrainOrbits=True, **copy.deepcopy(self.spec)
            )

            mode = list(
                filter(
                    lambda mode: mode["detectionMode"] == True,
                    TL.OpticalSystem.observingModes,
                )
            )[0]
            IWA = mode["IWA"]
            OWA = mode["OWA"]
            rrange = TL.PlanetPopulation.rrange
            maxd = (rrange[1] / np.tan(IWA)).to(u.pc).value
            mind = (rrange[0] / np.tan(OWA)).to(u.pc).value

            # want distances to span from outer edge below IWA to inner edge above OWA
            TL.dist = (
                np.logspace(np.log10(mind / 10.0), np.log10(maxd * 10.0), TL.nStars)
                * u.pc
            )

        Brown = EXOSIMS.Completeness.BrownCompleteness.BrownCompleteness(
            constrainOrbits=True, **copy.deepcopy(self.spec)
        )
        Garrett = EXOSIMS.Completeness.GarrettCompleteness.GarrettCompleteness(
            constrainOrbits=True, **copy.deepcopy(self.spec)
        )

        cBrown = Brown.target_completeness(TL)
        cGarrett = Garrett.target_completeness(TL)

        np.testing.assert_allclose(cGarrett, cBrown, rtol=0.1, atol=1e-6)
