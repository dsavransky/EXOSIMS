import unittest
from EXOSIMS.PlanetPopulation.DulzPlavchan import DulzPlavchan
from tests.TestSupport.Info import resource_path
import numpy as np
import os
from tests.TestSupport.Utilities import RedirectStreams
import copy


class TestDulzPlavchan(unittest.TestCase):
    """

    Tests specifically for DulzPlavchan coverage

    """

    def setUp(self):
        self.dev_null = open(os.devnull, "w")
        self.occDataPath = resource_path("PlanetPopulation/NominalOcc_Mass.csv")
        self.spec = {
            "modules": {"PlanetPhysicalModel": "PlanetPhysicalModel"},
            "occDataPath": self.occDataPath,
        }

    def tearDown(self):
        self.dev_null.close()

    def test_gen_plan_params(self):
        """
        Test that gen_plan_params returns the correct number of samples
        and values are within the min and max values specified.
        """

        # number of samples to generate
        num = 10000
        spec = copy.deepcopy(self.spec)
        with RedirectStreams(stdout=self.dev_null):
            obj = DulzPlavchan(**spec)
        a, e, p, Rp = obj.gen_plan_params(num)
        # check each sampled parameter
        self.assertEqual(
            len(a), num, "Incorrect number of samples generated for DulzPlavchan"
        )
        self.assertTrue(
            np.all(a <= obj.arange[1]), "a high bound failed for DulzPlavchan"
        )
        self.assertTrue(
            np.all(a >= obj.arange[0]), "a low bound failed for DulzPlavchan"
        )
        self.assertEqual(
            len(e), num, "Incorrect number of samples generated for DulzPlavchan"
        )
        self.assertTrue(
            np.all(e <= obj.erange[1]), "e high bound failed for DulzPlavchan"
        )
        self.assertTrue(
            np.all(e >= obj.erange[0]), "e low bound failed for DulzPlavchan"
        )
        self.assertEqual(
            len(p), num, "Incorrect number of samples generated for DulzPlavchan"
        )
        self.assertTrue(
            np.all(p <= obj.prange[1]), "p high bound failed for DulzPlavchan"
        )
        self.assertTrue(
            np.all(p >= obj.prange[0]), "p low bound failed for DulzPlavchan"
        )
        self.assertEqual(
            len(Rp), num, "Incorrect number of samples generated for DulzPlavchan"
        )
        self.assertTrue(
            np.all(Rp <= obj.Rprange[1]), "Rp high bound failed for DulzPlavchan"
        )
        self.assertTrue(
            np.all(Rp >= obj.Rprange[0]), "Rp low bound failed for DulzPlavchan"
        )

    def test_dist_albedo(self):
        """
        Test that albedos outside of the range have zero probability

        """

        spec = copy.deepcopy(self.spec)
        spec["modules"]["PlanetPhysicalModel"] = "FortneyMarleyCahoyMix1"
        with RedirectStreams(stdout=self.dev_null):
            pp = DulzPlavchan(**spec)

        p = np.linspace(pp.prange[0] - 1, pp.prange[1] + 1, 100)

        fp = pp.dist_albedo(p)
        self.assertTrue(
            np.all(fp[p < pp.prange[0]] == 0),
            "dist_albedo high bound failed for DulzPlavchan",
        )
        self.assertTrue(
            np.all(fp[p > pp.prange[1]] == 0),
            "dist_albedo low bound failed for DulzPlavchan",
        )
        self.assertTrue(
            np.all(fp[(p >= pp.prange[0]) & (p <= pp.prange[1])] > 0),
            "dist_albedo generates zero probabilities within range for DulzPlavchan",
        )
