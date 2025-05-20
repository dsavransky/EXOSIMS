import unittest
import os
import EXOSIMS.Prototypes.PlanetPhysicalModel
from EXOSIMS.PlanetPhysicalModel.ForecasterSousa import ForecasterSousa
from EXOSIMS.util.get_module import get_module
import pkgutil
from tests.TestSupport.Utilities import RedirectStreams
import numpy as np
import astropy.units as u
import sys
from io import StringIO


class TestForecasterSousa(unittest.TestCase):
    def setUp(self):
        self.dev_null = open(os.devnull, "w")

        modtype = getattr(
            EXOSIMS.Prototypes.PlanetPhysicalModel.PlanetPhysicalModel, "_modtype"
        )
        pkg = EXOSIMS.PlanetPhysicalModel
        self.mod = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(
            pkg.__path__, pkg.__name__ + "."
        ):
            if not is_pkg:
                mod = get_module(module_name.split(".")[-1], modtype)
                self.assertTrue(
                    mod._modtype is modtype, "_modtype mismatch for %s" % mod.__name__
                )
                if "Sousa" in str(mod):
                    self.mod = mod

    def tearDown(self):
        self.dev_null.close()

    def test_calc_radius_from_mass(self):
        """
        Tests that radius returned has correct value.
        """
        with RedirectStreams(stdout=self.dev_null):
            obj = self.mod()
        Mp_valuetest = np.array([50.0, 159.0, 200.0]) * u.earthMass
        Rp_truth = np.array([7.1, 14.1, 14.1]) * u.earthRad
        Rp_valuetest = obj.calc_radius_from_mass(Mp_valuetest)

        np.testing.assert_allclose(
            Rp_valuetest.value,
            Rp_truth.value,
            rtol=1e-1,
            err_msg="Radius values do not match expected values.",
        )

    def test_calc_mass_from_radius(self):
        """
        Tests that mass returned has correct value.
        """
        with RedirectStreams(stdout=self.dev_null):
            obj = self.mod()
        Rp_valuetest = np.array([5.0, 14.1, 20.0]) * u.earthRad
        Mp_truth = np.array([27.47, 159.245, (u.M_jupiter).to(u.M_earth)]) * u.earthMass
        Mp_valuetest = obj.calc_mass_from_radius(Rp_valuetest)

        np.testing.assert_allclose(
            Mp_valuetest.value,
            Mp_truth.value,
            rtol=1e-1,
            err_msg="Mass values do not match expected values.",
        )
