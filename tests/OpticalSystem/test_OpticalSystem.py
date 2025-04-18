import unittest
from tests.TestSupport.Utilities import RedirectStreams
from tests.TestSupport.Info import resource_path
import EXOSIMS.OpticalSystem
from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem
from EXOSIMS.util.get_module import get_module
from EXOSIMS.Prototypes.TargetList import TargetList
import os
import pkgutil
import json
import copy
import astropy.units as u
import numpy as np
import sys
from io import StringIO


class TestOpticalSystem(unittest.TestCase):
    """

    Global OpticalSystem tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """

    def setUp(self):

        self.dev_null = open(os.devnull, "w")
        self.script = resource_path("test-scripts/template_minimal.json")
        with open(self.script) as f:
            self.spec = json.loads(f.read())

        with RedirectStreams(stdout=self.dev_null):
            self.TL = TargetList(ntargs=10, **copy.deepcopy(self.spec))

        modtype = getattr(OpticalSystem, "_modtype")
        whitelist = ["Nemati_2019"]
        pkg = EXOSIMS.OpticalSystem
        self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(
            pkg.__path__, pkg.__name__ + "."
        ):
            if (not is_pkg) and (module_name.split(".")[-1] not in whitelist):
                mod = get_module(module_name.split(".")[-1], modtype)
                self.assertTrue(
                    mod._modtype is modtype, "_modtype mismatch for %s" % mod.__name__
                )
                self.allmods.append(mod)

    def tearDown(self):
        self.dev_null.close()

    def test_Cp_Cb_Csp(self):
        """
        Consistency check Cp_Cb_Csp calculations.
        """

        for mod in self.allmods:
            if "Cp_Cb_Csp" not in mod.__dict__:
                continue

            obj = mod(**copy.deepcopy(self.spec))

            # first check, infinite dMag should give zero C_p
            C_p, C_b, C_sp = obj.Cp_Cb_Csp(
                self.TL,
                np.arange(self.TL.nStars),
                np.array([0] * self.TL.nStars) / (u.arcsec**2.0),
                np.array([0] * self.TL.nStars) * u.ph / u.s / u.m**2 / u.arcsec**2,
                np.ones(self.TL.nStars) * np.inf,
                np.array(self.TL.int_WA.value) * self.TL.int_WA.unit,
                obj.observingModes[0],
            )
            self.assertEqual(len(C_p), len(C_b))
            self.assertEqual(len(C_b), len(C_sp))
            self.assertTrue(np.all(C_p.value == 0))

            # second check, outside OWA, C_p and C_sp should be all zero
            # (C_b may be non-zero due to read/dark noise)
            C_p, C_b, C_sp = obj.Cp_Cb_Csp(
                self.TL,
                np.arange(self.TL.nStars),
                np.array([0] * self.TL.nStars) / (u.arcsec**2.0),
                np.array([0] * self.TL.nStars) * u.ph / u.s / u.m**2 / u.arcsec**2,
                np.ones(self.TL.nStars) * self.TL.int_dMag,
                np.array([obj.observingModes[0]["OWA"].value * 2.0] * self.TL.nStars)
                * obj.observingModes[0]["OWA"].unit,
                obj.observingModes[0],
            )
            self.assertTrue(np.all(C_p.value == 0))
            self.assertTrue(np.all(C_sp.value == 0))

            # third check, inside IWA, C_p and C_sp should be all zero
            # (C_b may be non-zero due to read/dark noise)
            C_p, C_b, C_sp = obj.Cp_Cb_Csp(
                self.TL,
                np.arange(self.TL.nStars),
                np.array([0] * self.TL.nStars) / (u.arcsec**2.0),
                np.array([0] * self.TL.nStars) * u.ph / u.s / u.m**2 / u.arcsec**2,
                np.ones(self.TL.nStars) * self.TL.int_dMag,
                np.array([obj.observingModes[0]["IWA"].value / 2.0] * self.TL.nStars)
                * obj.observingModes[0]["IWA"].unit,
                obj.observingModes[0],
            )
            self.assertTrue(np.all(C_p.value == 0))
            self.assertTrue(np.all(C_sp.value == 0))

    def test_calc_intTime(self):
        """
        Check calc_intTime i/o and basic consistency
        """

        for mod in self.allmods:
            if "calc_intTime" not in mod.__dict__:
                continue

            obj = mod(**copy.deepcopy(self.spec))

            # inttime for 0 dMag
            intTime0 = obj.calc_intTime(
                self.TL,
                np.arange(self.TL.nStars),
                np.array([0] * self.TL.nStars) / (u.arcsec**2.0),
                np.array([0] * self.TL.nStars) * u.ph / u.s / u.m**2 / u.arcsec**2,
                np.zeros(self.TL.nStars),
                np.array(self.TL.int_WA.value) * self.TL.int_WA.unit,
                obj.observingModes[0],
            )
            self.assertEqual(len(intTime0), self.TL.nStars)

            # inttime for 1 dMag
            intTime1 = obj.calc_intTime(
                self.TL,
                np.arange(self.TL.nStars),
                np.array([0] * self.TL.nStars) / (u.arcsec**2.0),
                np.array([0] * self.TL.nStars) * u.ph / u.s / u.m**2 / u.arcsec**2,
                np.ones(self.TL.nStars),
                np.array(self.TL.int_WA.value) * self.TL.int_WA.unit,
                obj.observingModes[0],
            )
            self.assertTrue(
                np.all(intTime1 >= intTime0),
                "dMag=1 produces shorter integration times than dMag=0",
            )

    @unittest.skip("Redundant with test_intTime_dMag_roundtrip")
    def test_calc_dMag_per_intTime(self):
        """
        Check calc_dMag_per_intTime i/o
        """

        exclude_mods = []

        for mod in self.allmods:
            if mod.__name__ in exclude_mods:
                continue

            if "calc_dMag_per_intTime" not in mod.__dict__:
                continue
            obj = mod(**copy.deepcopy(self.spec))

            dMag = obj.calc_dMag_per_intTime(
                np.ones(self.TL.nStars) * u.day,
                self.TL,
                np.arange(self.TL.nStars),
                np.array([0] * self.TL.nStars) / (u.arcsec**2.0),
                np.array([0] * self.TL.nStars) * u.ph / u.s / u.m**2 / u.arcsec**2,
                np.array(self.TL.int_WA.value) * self.TL.int_WA.unit,
                obj.observingModes[0],
            )

            self.assertEqual(dMag.shape, np.arange(self.TL.nStars).shape)

    def test_intTime_dMag_roundtrip(self):
        """
        Check calc_intTime to calc_dMag_per_intTime to calc_intTime to
        calc_dMag_per_intTime give equivalent results
        """

        # modules which do not calculate dMag from intTime
        whitelist = ["KasdinBraems"]

        # set up values
        fZ = np.array([self.TL.ZodiacalLight.fZ0.value] * self.TL.nStars) / (
            u.arcsec**2
        )
        JEZ = self.TL.JEZ0[self.TL.OpticalSystem.observingModes[0]["hex"]]
        # JEZ = self.TL.ZodiacalLight.calc_JEZ0(self.TL.MV, np.array([90])*u.deg, np.array([1]), 10*u.nm)

        for mod in self.allmods:
            if mod.__name__ in whitelist:
                continue
            # Because int_dMag depends on the OpticalSystem module we need to
            # recalculate it with the current module
            tmpspec = copy.deepcopy(self.spec)
            tmpspec["modules"]["OpticalSystem"] = mod.__name__
            TL = TargetList(ntargs=10, skipSaturationCalcs=False, **tmpspec)

            obj = TL.OpticalSystem
            # make sure all dMags are lower than the saturation dMag
            dMags1 = TL.saturation_dMag - np.random.rand(TL.nStars) * 2

            WA = np.array(self.TL.int_WA.value) * self.TL.int_WA.unit
            # integration times from dMags1
            intTime1 = obj.calc_intTime(
                TL, np.arange(TL.nStars), fZ, JEZ, dMags1, WA, obj.observingModes[0]
            )

            # dMags from intTime1
            dMags2 = obj.calc_dMag_per_intTime(
                intTime1, TL, np.arange(TL.nStars), fZ, JEZ, WA, obj.observingModes[0]
            )

            # check that all times were feasible
            useful_inds = ~np.isnan(intTime1)
            self.assertTrue(np.all(useful_inds))

            # ensure dMags match up roundtrip
            self.assertTrue(np.allclose(dMags1[useful_inds], dMags2[useful_inds]))

    def test_ddMag_dt(self):
        """
        Check ddMag_dt i/o
        """

        for mod in self.allmods:

            if "ddMag_dt" not in mod.__dict__:
                continue
            obj = mod(**copy.deepcopy(self.spec))

            ddMag = obj.ddMag_dt(
                np.ones(self.TL.nStars) * u.day,
                self.TL,
                np.arange(self.TL.nStars),
                np.array([0] * self.TL.nStars) / (u.arcsec**2.0),
                np.array([0] * self.TL.nStars) * u.ph / u.s / u.m**2 / u.arcsec**2,
                np.array(self.TL.int_WA.value) * self.TL.int_WA.unit,
                obj.observingModes[0],
            )

            self.assertEqual(ddMag.shape, np.arange(self.TL.nStars).shape)

    def calc_saturation_dMag(self):
        """
        Check calc_saturation_dMag i/o and basic consistency
        """

        for mod in self.allmods:
            if "calc_saturation_dMag" not in mod.__dict__:
                continue

            obj = mod(**copy.deepcopy(self.spec))

            sat_dMag = obj.calc_saturation_dMag(
                self.TL,
                np.arange(self.TL.nStars),
                np.array([0] * self.TL.nStars) / (u.arcsec**2.0),
                np.array([0] * self.TL.nStars) * u.ph / u.s / u.m**2 / u.arcsec**2,
                np.array(self.TL.int_WA.value) * self.TL.int_WA.unit,
                obj.observingModes[0],
            )
            self.assertEqual(len(sat_dMag), self.TL.nStars)

            # inttime at saturation
            intTimesat = obj.calc_intTime(
                self.TL,
                np.arange(self.TL.nStars),
                np.array([0] * self.TL.nStars) / (u.arcsec**2.0),
                np.array([0] * self.TL.nStars) * u.ph / u.s / u.m**2 / u.arcsec**2,
                sat_dMag,
                np.array(self.TL.int_WA.value) * self.TL.int_WA.unit,
                obj.observingModes[0],
            )
            self.assertTrue(
                np.all(np.isnan(intTimesat)),
                "saturation dMag leads to finite integration time",
            )

            # inttime at 0.9*saturation
            intTimenonsat = obj.calc_intTime(
                self.TL,
                np.arange(self.TL.nStars),
                np.array([0] * self.TL.nStars) / (u.arcsec**2.0),
                np.array([0] * self.TL.nStars) * u.ph / u.s / u.m**2 / u.arcsec**2,
                sat_dMag * 0.9,
                np.array(self.TL.int_WA.value) * self.TL.int_WA.unit,
                obj.observingModes[0],
            )
            self.assertTrue(
                np.all(~np.isnan(intTimenonsat)),
                "< saturation dMag leads to NaN integration time",
            )

    def test_str(self):
        """
        Test __str__ method, for full coverage and check that all modules have
        required attributes.
        """

        atts_list = [
            "obscurFac",
            "shapeFac",
            "pupilDiam",
            "intCutoff",
            "pupilArea",
            "haveOcculter",
            "IWA",
            "OWA",
            "koAngles_Sun",
            "koAngles_Earth",
            "koAngles_Moon",
            "koAngles_Small",
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
