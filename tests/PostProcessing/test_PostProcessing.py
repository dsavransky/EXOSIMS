import unittest
import EXOSIMS
import EXOSIMS.Prototypes.PostProcessing
from EXOSIMS.Prototypes.TargetList import TargetList
import EXOSIMS.PostProcessing
import pkgutil
from EXOSIMS.util.get_module import get_module
import numpy as np
import os
import json
from tests.TestSupport.Utilities import RedirectStreams
from tests.TestSupport.Info import resource_path
import astropy.units as u
import inspect
import sys
from io import StringIO


class TestPostProcessing(unittest.TestCase):
    """

    Global PostProcessing tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """

    def setUp(self):

        self.dev_null = open(os.devnull, "w")
        self.specs = {"modules": {"BackgroundSources": " "}}
        script = resource_path("test-scripts/template_minimal.json")
        with open(script) as f:
            spec = json.loads(f.read())
        with RedirectStreams(stdout=self.dev_null):
            self.TL = TargetList(**spec)

        modtype = getattr(EXOSIMS.Prototypes.PostProcessing.PostProcessing, "_modtype")
        pkg = EXOSIMS.PostProcessing
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

    # Testing some limiting cases below
    def test_zeroFAP(self):
        """
        Test case where false alarm probability is 0 and missed det prob is 0
        All observations above SNR limit should be detected
        """

        SNRin = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.1, 6.0])
        expected_MDresult = [True, True, True, True, False, False, False]

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(FAP=0.0, MDP=0.0, **self.specs)

            FA, MD = obj.det_occur(
                SNRin, self.TL.OpticalSystem.observingModes[0], self.TL, 0, 1 * u.d
            )
            for x, y in zip(MD, expected_MDresult):
                self.assertEqual(x, y)
            self.assertEqual(FA, False)

    # another limiting case
    def test_oneFAP(self):
        """
        Test case where false alarm probability is 1 and missed det prob is 0
        """

        SNRin = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.1, 6.0])
        expected_MDresult = [True, True, True, True, False, False, False]

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(FAP=1.0, MDP=0.0, **self.specs)

            FA, MD = obj.det_occur(
                SNRin, self.TL.OpticalSystem.observingModes[0], self.TL, 0, 1 * u.d
            )
            for x, y in zip(MD, expected_MDresult):
                self.assertEqual(x, y)
            self.assertEqual(FA, True)

    def test_ppFact_fits(self):
        # get fits file path for ppFact test
        classpath = os.path.split(inspect.getfile(self.__class__))[0]
        ppFactPath = os.path.join(classpath, "test_PostProcessing_ppFact.fits")

        # fits file has values for WA in [0.1,0.2]
        testWA = np.linspace(0.1, 0.2, 100) * u.arcsec

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(ppFact=ppFactPath, **self.specs)

            vals = obj.ppFact(testWA)

            self.assertTrue(
                np.all(vals > 0), "negative value of ppFact for %s" % mod.__name__
            )
            self.assertTrue(np.all(vals <= 1), "ppFact > 1 for %s" % mod.__name__)

    def test_FAdMag0_fits(self):
        # get fits file path for FAdMag0 test
        classpath = os.path.split(inspect.getfile(self.__class__))[0]
        FAdMag0Path = os.path.join(classpath, "test_PostProcessing_FAdMag0.fits")

        # fits file has values for WA in [0.1, 0.2] and FAdMag0 in [10, 20]
        testWA = np.linspace(0.1, 0.2, 100) * u.arcsec

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(FAdMag0=FAdMag0Path, **self.specs)

            vals = obj.FAdMag0(testWA)

            self.assertTrue(
                np.all(vals >= 10), "value below range of FAdMag0 for %s" % mod.__name__
            )
            self.assertTrue(
                np.all(vals <= 20), "value above range of FAdMag0 for %s" % mod.__name__
            )

    def test_str(self):
        """
        Test __str__ method, for full coverage and check that all modules have
        required attributes.
        """
        atts_list = ["BackgroundSources", "FAP", "MDP"]
        for mod in self.allmods:
            if "__str__" not in mod.__dict__:
                continue

            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**self.specs)
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
