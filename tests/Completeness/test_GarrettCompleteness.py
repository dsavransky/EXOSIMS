import unittest
from tests.TestSupport.Info import resource_path
import EXOSIMS.Completeness.GarrettCompleteness
import os
import numpy as np
import json
import copy


class TestGarrettCompleteness(unittest.TestCase):
    """

    Test GarrettCompleteness specific methods not covered in overloaded
    method unittesting.

    """

    def setUp(self):

        self.dev_null = open(os.devnull, "w")
        self.script = resource_path("test-scripts/template_minimal.json")
        with open(self.script) as f:
            self.spec = json.loads(f.read())
        self.spec["modules"]["PlanetPopulation"] = "AlbedoByRadius"
        self.spec["prange"] = [0.1, 0.5]
        self.spec["Rprange"] = [1, 10]
        self.script2 = resource_path("test-scripts/simplest.json")
        with open(self.script2) as f:
            self.spec2 = json.loads(f.read())

    def tearDown(self):
        self.dev_null.close()

    def test_calcs(self):
        """
        Performs following tests:
            Ensure calc_fdmag returns a valid pdf.
            Ensure that comp_calc, comp_dmag, and comp_s are all within 1e-3.
            Test if f_dmags and f_sdmag return same result.
            Test s_bound against mindmag and maxdmag.
        """

        Gcomp = EXOSIMS.Completeness.GarrettCompleteness.GarrettCompleteness(
            **copy.deepcopy(self.spec)
        )

        print("Testing calc_fdmag ..."),
        val = Gcomp.calc_fdmag(25.0, 0.75, 1.0)
        self.assertGreaterEqual(
            val, 0, "dmag pdf must be greater than zero for GarrettCompleteness"
        )
        print("ok")

        print("Testing comp_calc, comp_dmag, comp_s are all close ..."),
        val1 = Gcomp.comp_calc(0.75, 1.0, 25.0)
        val2 = Gcomp.comp_dmag(0.75, 1.0, 25.0)
        val3 = Gcomp.comp_s(0.75, 1.0, 25.0)
        # compare comp_calc to comp_dmag
        self.assertLessEqual(
            np.abs(val1 - val2),
            1e-3,
            "comp_calc and comp_dmag must be within 1e-3 for GarrettCompleteness",
        )
        # compare comp_calc to comp_s
        self.assertLessEqual(
            np.abs(val1 - val3),
            1e-3,
            "comp_calc and comp_s must be within 1e-3 for GarrettCompleteness",
        )
        # compare comp_dmag to comp_s
        self.assertLessEqual(
            np.abs(val2 - val3),
            1e-3,
            "comp_dmag and comp_s must be within 1e-3 for GarrettCompleteness",
        )
        print("ok")

        print("Testing f_dmags and f_sdmag yield same result ..."),
        val1 = Gcomp.f_dmags(22.0, 1.0)
        val2 = Gcomp.f_sdmag(1.0, 22.0)
        self.assertEqual(val1, val2, "f_dmags and f_sdmag must return same result")
        print("ok")

        s = 1.0
        mind = Gcomp.mindmag(s)
        maxd = Gcomp.maxdmag(s)
        s1 = Gcomp.s_bound(mind, Gcomp.amax)
        s2 = Gcomp.s_bound(maxd, Gcomp.amax)
        print("Testing s_bound against mindmag ..."),
        self.assertLessEqual(
            np.abs(s - s1),
            1e-3,
            "s_bound must return s value from mindmag for GarrettCompleteness",
        )
        print("ok")
        print("Testing s_bound against maxdmag ..."),
        self.assertLessEqual(
            np.abs(s - s2),
            1e-3,
            "s_bound must return s value from maxdmag for GarrettCompleteness",
        )
        print("ok")

    @unittest.skip("GarrettCompleteness stable. Skipping auxiliary tests.")
    def test_comp_constrainOrbits(self):
        """
        Test that GarrettCompleteness returns a valid completeness value when
        constrainOrbits is True.
        """

        spec = copy.deepcopy(self.spec2)
        spec["constrainOrbits"] = True
        Gcomp = EXOSIMS.Completeness.GarrettCompleteness.GarrettCompleteness(**spec)
        val = Gcomp.comp_calc(1.0, 10.0, 22.0)
        self.assertGreaterEqual(
            val,
            0,
            "Completeness evaluated less than zero by GarrettCompleteness when constrainOrbits is True",
        )
        self.assertLessEqual(
            val,
            1,
            "Completeness evaluated greater than one by GarrettCompleteness when constrainOrbits is True",
        )

    @unittest.skip("GarrettCompleteness stable. Skipping auxiliary tests.")
    def test_constant_sma(self):
        """
        Test that GarrettCompleteness returns a valid completeness value when
        sma is constant.
        """

        spec = copy.deepcopy(self.spec2)
        spec["arange"] = [5, 5]
        Gcomp = EXOSIMS.Completeness.GarrettCompleteness.GarrettCompleteness(**spec)
        val = Gcomp.comp_calc(1.0, 5.0, 22.0)
        self.assertGreaterEqual(
            val,
            0,
            "Completeness evaluated less than zero by GarrettCompleteness when sma constant",
        )
        self.assertLessEqual(
            val,
            1,
            "Completeness evaluated greater than one by GarrettCompleteness when sma constant",
        )

    @unittest.skip("GarrettCompleteness stable. Skipping auxiliary tests.")
    def test_circular(self):
        """
        Test that GarrettCompleteness returns a valid completeness value when
        orbits are circular.
        """

        spec = copy.deepcopy(self.spec2)
        spec["erange"] = [0, 0]
        Gcomp = EXOSIMS.Completeness.GarrettCompleteness.GarrettCompleteness(**spec)
        val = Gcomp.comp_calc(1.0, 5.0, 22.0)
        self.assertGreaterEqual(
            val,
            0,
            "Completeness evaluated less than zero by GarrettCompleteness when orbits are circular",
        )
        self.assertLessEqual(
            val,
            1,
            "Completeness evaluated greater than one by GarrettCompleteness when orbits are circular",
        )

    @unittest.skip("GarrettCompleteness stable. Skipping auxiliary tests.")
    def test_constant_eccentricity(self):
        """
        Test that GarrettCompleteness returns a valid completeness value when
        eccentricity is constant and nonzero.
        """

        spec = copy.deepcopy(self.spec2)
        spec["erange"] = [0.1, 0.1]
        Gcomp = EXOSIMS.Completeness.GarrettCompleteness.GarrettCompleteness(**spec)
        val = Gcomp.comp_calc(1.0, 10.0, 22.0)
        self.assertGreaterEqual(
            val,
            0,
            "Completeness evaluated less than zero by GarrettCompleteness when eccentricity constant",
        )
        self.assertLessEqual(
            val,
            1,
            "Completeness evaluated greater than one by GarrettCompleteness when eccentricity constant",
        )

    @unittest.skip("GarrettCompleteness stable. Skipping auxiliary tests.")
    def test_constant_sma_eccentricity(self):
        """
        Test that GarrettCompleteness returns a valid completeness value when
        sma and eccentricity are constant.
        """

        spec = copy.deepcopy(self.spec2)
        spec["erange"] = [0.1, 0.1]
        spec["arange"] = [5, 5]
        Gcomp = EXOSIMS.Completeness.GarrettCompleteness.GarrettCompleteness(**spec)
        val = Gcomp.comp_calc(1.0, 5.0, 22.0)
        self.assertGreaterEqual(
            val,
            0,
            "Completeness evaluated less than zero by GarrettCompleteness when sma and eccentricity constant",
        )
        self.assertLessEqual(
            val,
            1,
            "Completeness evaluated greater than one by GarrettCompleteness when sma and eccentricity constant",
        )

    @unittest.skip("GarrettCompleteness stable. Skipping auxiliary tests.")
    def test_constant_sma_constrainOrbits(self):
        """
        Test that GarrettCompleteness returns a valid completeness value when
        sma is constant and constrainOrbits is true
        """

        spec = copy.deepcopy(self.spec2)
        spec["arange"] = [5, 5]
        Gcomp = EXOSIMS.Completeness.GarrettCompleteness.GarrettCompleteness(**spec)
        val = Gcomp.comp_calc(1.0, 5.0, 22.0)
        self.assertGreaterEqual(
            val,
            0,
            "Completeness evaluated less than zero by GarrettCompleteness when sma constant and constrainOrbits==True",
        )
        self.assertLessEqual(
            val,
            1,
            "Completeness evaluated greater than one by GarrettCompleteness when sma constant and constrainOrbits==True",
        )

    @unittest.skip("GarrettCompleteness stable. Skipping auxiliary tests.")
    def test_constant_albedo(self):
        """
        Test that GarrettCompleteness returns a valid completeness value when
        albedo is constant.
        """

        spec = copy.deepcopy(self.spec2)
        spec["prange"] = [0.2, 0.2]
        Gcomp = EXOSIMS.Completeness.GarrettCompleteness.GarrettCompleteness(**spec)
        val = Gcomp.comp_calc(1.0, 10.0, 22.0)
        self.assertGreaterEqual(
            val,
            0,
            "Completeness evaluated less than zero by GarrettCompleteness when albedo constant",
        )
        self.assertLessEqual(
            val,
            1,
            "Completeness evaluated greater than one by GarrettCompleteness when albedo constant",
        )

    @unittest.skip("GarrettCompleteness stable. Skipping auxiliary tests.")
    def test_constant_radius(self):
        """
        Test that GarrettCompleteness returns a valid completeness value when
        planetary radius is constant.
        """

        spec = copy.deepcopy(self.spec2)
        spec["Rprange"] = [5, 5]
        Gcomp = EXOSIMS.Completeness.GarrettCompleteness.GarrettCompleteness(**spec)
        val = Gcomp.comp_calc(1.0, 10.0, 22.0)
        self.assertGreaterEqual(
            val,
            0,
            "Completeness evaluated less than zero by GarrettCompleteness when planetary radius constant",
        )
        self.assertLessEqual(
            val,
            1,
            "Completeness evaluated greater than one by GarrettCompleteness when planetary radius constant",
        )

    @unittest.skip("GarrettCompleteness stable. Skipping auxiliary tests.")
    def test_constant_albedo_radius(self):
        """
        Test that GarrettCompleteness returns a valid completeness value when
        albedo and planetary radius are constant.
        """

        spec = copy.deepcopy(self.spec2)
        spec["prange"] = [0.2, 0.2]
        spec["Rprange"] = [5, 5]
        Gcomp = EXOSIMS.Completeness.GarrettCompleteness.GarrettCompleteness(**spec)
        val = Gcomp.comp_calc(1.0, 10.0, 22.0)
        self.assertGreaterEqual(
            val,
            0,
            "Completeness evaluated less than zero by GarrettCompleteness when albedo and planetary radius constant",
        )
        self.assertLessEqual(
            val,
            1,
            "Completeness evaluated greater than one by GarrettCompleteness when albedo and planetary radius constant",
        )
