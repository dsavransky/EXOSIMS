import unittest
import os
import EXOSIMS.Prototypes.PlanetPhysicalModel
import EXOSIMS.PlanetPhysicalModel
from EXOSIMS.util.get_module import get_module
import pkgutil
from tests.TestSupport.Utilities import RedirectStreams
import numpy as np
import astropy.units as u
import sys
from io import StringIO


class TestPlanetPhysicalModel(unittest.TestCase):
    def setUp(self):

        self.dev_null = open(os.devnull, "w")

        modtype = getattr(
            EXOSIMS.Prototypes.PlanetPhysicalModel.PlanetPhysicalModel, "_modtype"
        )
        pkg = EXOSIMS.PlanetPhysicalModel
        self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(
            pkg.__path__, pkg.__name__ + "."
        ):
            if not is_pkg:
                mod = get_module(module_name.split(".")[-1], modtype)
                self.assertTrue(
                    mod._modtype is modtype, "_modtype mismatch for %s" % mod.__name__
                )
                self.allmods.append(mod)

    def tearDown(self):
        self.dev_null.close()

    def test_calc_albedo_from_sma(self):
        """
        Tests that albedos returned are have the correct length, are finite, and >= 0.
        """

        for mod in self.allmods:
            if "calc_albedo_from_sma" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod()
                sma = np.random.uniform(0.1, 10.0, 10) * u.AU
                p = obj.calc_albedo_from_sma(sma, [0.367, 0.367])

                self.assertTrue(
                    len(p) == len(sma),
                    "length of albedo array does not match input sma for %s"
                    % mod.__name__,
                )
                self.assertTrue(
                    np.all(np.isfinite(p)),
                    "infinite albedo value returned for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(p >= 0.0),
                    "negative albedo value returned for %s" % mod.__name__,
                )

    def test_calc_radius_from_mass(self):
        """
        Tests that radius returned has correct length, unit, value, is finite, and > 0.
        """

        for mod in self.allmods:
            if "calc_radius_from_mass" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod()
                Mp = np.random.uniform(0.5, 500.0, 100) * u.earthMass
                Rp = obj.calc_radius_from_mass(Mp)
                self.assertTrue(
                    len(Rp) == len(Mp),
                    "length of radius array does not match input mass array for %s"
                    % mod.__name__,
                )
                self.assertTrue(
                    Rp.unit is u.earthRad,
                    "radius unit is not earthRad for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(np.isfinite(Rp)),
                    "Infinite radius value returned for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(Rp > 0.0),
                    "negative radius value returned for %s" % mod.__name__,
                )

    def test_calc_mass_from_radius(self):
        """
        Tests that mass returned has correct length, unit, value, is finite, and > 0.
        """

        for mod in self.allmods:
            if "calc_mass_from_radius" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod()

                Rp = np.random.uniform(0.5, 11.2, 100) * u.earthRad
                Mp = obj.calc_mass_from_radius(Rp)

                self.assertTrue(
                    len(Mp) == len(Rp),
                    "length of mass array does not match input radius array for %s"
                    % mod.__name__,
                )
                self.assertTrue(
                    Mp.unit is u.earthMass,
                    "mass unit is not earthMass for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(np.isfinite(Mp)),
                    "Infinite mass value returned for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(Mp > 0.0),
                    "negative mass value returned for %s" % mod.__name__,
                )

    def test_calc_Phi(self):
        """
        Tests that phase function returns appropriate values.
        """

        for mod in self.allmods:
            if "calc_Phi" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod()
                betas = np.linspace(0.0, np.pi, 100) * u.rad
                Phi = obj.calc_Phi(betas)

                self.assertTrue(
                    len(Phi) == len(betas),
                    (
                        "length of phase function values returned does not match input "
                        f"phase angles for {mod.__name__}"
                    ),
                )
                self.assertTrue(
                    np.all(np.isfinite(Phi)),
                    "calc_Phi returned infinite value for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(Phi <= 1.0),
                    "calc_Phi returned value > 1 for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(Phi >= 0.0),
                    "calc_Phi returned negative value for %s" % mod.__name__,
                )

    def test_calc_Teff(self):
        """
        Tests that calc_Teff returns values within appropriate ranges.
        """

        for mod in self.allmods:
            if "calc_Teff" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod()
                starL = np.random.uniform(0.7, 1.3, 100)
                p = np.random.uniform(0.0, 1.0, 100)
                d = np.random.uniform(0.2, 10.0, 100) * u.AU
                Teff = obj.calc_Teff(starL, d, p)

                self.assertTrue(
                    len(Teff) == len(starL),
                    "length of Teff returned does not match inputs for %s"
                    % mod.__name__,
                )
                self.assertTrue(
                    np.all(np.isfinite(Teff)),
                    "calc_Teff returned infinite value for %s" % mod.__name__,
                )
                self.assertTrue(
                    np.all(Teff >= 0.0),
                    "calc_Teff returned negative value for %s" % mod.__name__,
                )

    def test_str(self):
        """
        Test __str__ method, for full coverage and check that all modules have
        required attributes.
        """

        for mod in self.allmods:
            if "__str__" not in mod.__dict__:
                continue

            with RedirectStreams(stdout=self.dev_null):
                obj = mod()
            original_stdout = sys.stdout
            sys.stdout = StringIO()
            # call __str__ method
            result = obj.__str__()
            # examine what was printed
            contents = sys.stdout.getvalue()
            self.assertEqual(type(contents), type(""))
            # attributes from ICD
            self.assertIn(
                "_outspec", contents, "_outspec missing for %s" % mod.__name__
            )
            sys.stdout.close()
            # it also returns a string, which is not necessary
            self.assertEqual(type(result), type(""))
            # put stdout back
            sys.stdout = original_stdout
