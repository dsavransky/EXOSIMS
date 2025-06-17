import unittest
from tests.TestSupport.Info import resource_path
import EXOSIMS
import EXOSIMS.Prototypes.Observatory
import EXOSIMS.Observatory
import pkgutil
import os
import json
import sys
import copy
from EXOSIMS.util.get_module import get_module
import numpy as np
from astropy.time import Time
import astropy.units as u
from tests.TestSupport.Utilities import RedirectStreams
from io import StringIO


class TestObservatory(unittest.TestCase):
    """

    Global Observatory tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """

    def setUp(self):

        self.dev_null = open(os.devnull, "w")

        # self.spec = {"modules": {"PlanetPhysicalModel": "PlanetPhysicalModel"}}
        self.script = resource_path("test-scripts/template_minimal.json")
        with open(self.script) as f:
            self.spec = json.loads(f.read())

        modtype = getattr(EXOSIMS.Prototypes.Observatory.Observatory, "_modtype")
        pkg = EXOSIMS.Observatory
        self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(
            pkg.__path__, pkg.__name__ + "."
        ):
            if not (is_pkg) and ("parallel" not in module_name):
                mod = get_module(module_name.split(".")[-1], modtype)
                self.assertTrue(
                    mod._modtype is modtype, "_modtype mismatch for %s" % mod.__name__
                )
                self.allmods.append(mod)

    def tearDown(self):
        self.dev_null.close()

    def test_init(self):
        """
        Test of initialization and __init__.
        """

        req_atts = [
            "koAngles_SolarPanel",
            "ko_dtStep",
            "settlingTime",
            "thrust",
            "slewIsp",
            "scMass",
            "slewMass",
            "skMass",
            "twotanks",
            "dryMass",
            "coMass",
            "occulterSep",
            "skIsp",
            "defburnPortion",
            "checkKeepoutEnd",
            "forceStaticEphem",
            "constTOF",
            "occ_dtmin",
            "occ_dtmax",
            "maxdVpct",
            "dVtot",
            "dVmax",
            "flowRate",
            "havejplephem",
            "slewEff",
            "skEff",
        ]

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                if "SotoStarshade" in mod.__name__:
                    obj = mod(f_nStars=4, **copy.deepcopy(self.spec))
                else:
                    obj = mod(**copy.deepcopy(self.spec))

            # verify that all attributes are there
            for att in req_atts:
                self.assertTrue(
                    hasattr(obj, att),
                    "Missing attribute {} for {}".format(att, mod.__name__),
                )

    def test_orbit(self):
        """
        Test orbit method to ensure proper output
        """

        t_ref = Time(2027.0, format="jyear")
        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                if "SotoStarshade" in mod.__name__:
                    obj = mod(f_nStars=4, **copy.deepcopy(self.spec))
                else:
                    obj = mod(**copy.deepcopy(self.spec))
            r_sc = obj.orbit(t_ref)
            # the r_sc attribute is set and is a 3-tuple of astropy Quantity's
            self.assertEqual(type(r_sc), type(1.0 * u.km))
            self.assertEqual(r_sc.shape, (1, 3))

    def test_log_occulterResults(self):
        """
        Test that log_occulter_results returns proper dictionary with keys
        """

        atts_list = ["slew_time", "slew_angle", "slew_dV", "slew_mass_used", "scMass"]
        for mod in self.allmods:
            if "log_occulterResults" in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    if "SotoStarshade" in mod.__name__:
                        obj = mod(f_nStars=4, **copy.deepcopy(self.spec))
                    else:
                        obj = mod(**copy.deepcopy(self.spec))
                DRM = {}
                slewTimes = np.ones((5,)) * u.day
                sInds = np.arange(5)
                sd = np.ones((5,)) * u.rad
                dV = np.ones((5,)) * u.m / u.s

                DRM = obj.log_occulterResults(DRM, slewTimes, sInds, sd, dV)

                for att in atts_list:
                    self.assertTrue(
                        att in DRM,
                        "Missing key in log_occulterResults for %s" % mod.__name__,
                    )

                obj = mod(
                    skMass=1, slewMass=1, twotanks=True, **copy.deepcopy(self.spec)
                )
                DRM = obj.log_occulterResults(DRM, slewTimes, sInds, sd, dV)
                self.assertTrue(
                    "slewMass" in DRM,
                    "Missing key in log_occulterResults for %s" % mod.__name__,
                )

    def test_str(self):
        """
        Test __str__ method, for full coverage and check that all modules have
        required attributes.
        """

        atts_list = [
            "koAngles_SolarPanel",
            "ko_dtStep",
            "settlingTime",
            "thrust",
            "slewIsp",
            "scMass",
            "slewMass",
            "skMass",
            "twotanks",
            "dryMass",
            "coMass",
            "occulterSep",
            "skIsp",
            "defburnPortion",
            "checkKeepoutEnd",
            "forceStaticEphem",
            "constTOF",
            "occ_dtmin",
            "occ_dtmax",
            "maxdVpct",
            "dVtot",
            "dVmax",
            "flowRate",
            "havejplephem",
            "slewEff",
            "skEff",
        ]

        for mod in self.allmods:
            if "__str__" not in mod.__dict__:
                continue

            with RedirectStreams(stdout=self.dev_null):
                if "SotoStarshade" in mod.__name__:
                    obj = mod(f_nStars=4, **copy.deepcopy(self.spec))
                else:
                    obj = mod(**copy.deepcopy(self.spec))
            original_stdout = sys.stdout
            sys.stdout = StringIO()
            # call __str__ method
            result = obj.__str__()
            # examine what was printed
            contents = sys.stdout.getvalue()
            self.assertEqual(type(contents), type(""))
            # attributes from ICD
            for att in atts_list:
                self.assertIn(
                    att, contents, "{} missing for {}".format(att, mod.__name__)
                )
            sys.stdout.close()
            # it also returns a string, which is not necessary
            self.assertEqual(type(result), type(""))
            # put stdout back
            sys.stdout = original_stdout
