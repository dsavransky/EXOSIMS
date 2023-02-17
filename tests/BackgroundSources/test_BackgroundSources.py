import unittest
import os
import EXOSIMS
import EXOSIMS.Prototypes.BackgroundSources
import EXOSIMS.BackgroundSources
from EXOSIMS.util.get_module import get_module
import pkgutil
from tests.TestSupport.Info import resource_path
import json
from tests.TestSupport.Utilities import RedirectStreams
from EXOSIMS.Prototypes.TargetList import TargetList
import numpy as np
import astropy.units as u
import sys
from io import StringIO
from astropy.coordinates import SkyCoord


class TestBackgroundSources(unittest.TestCase):
    """

    Global BackgroundSources tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """

    def setUp(self):
        self.dev_null = open(os.devnull, "w")
        modtype = getattr(
            EXOSIMS.Prototypes.BackgroundSources.BackgroundSources, "_modtype"
        )
        pkg = EXOSIMS.BackgroundSources
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

        # need a TargetList object for testing
        # script = resource_path("test-scripts/template_prototype_testing.json")
        script = resource_path("test-scripts/template_minimal.json")
        with open(script) as f:
            spec = json.loads(f.read())
        spec["ntargs"] = 10  # generate fake targets list with 10 stars
        with RedirectStreams(stdout=self.dev_null):
            self.TL = TargetList(**spec)

        # assign different coordinates
        self.TL.coords = SkyCoord(
            ra=np.random.uniform(low=0, high=180, size=self.TL.nStars) * u.deg,
            dec=np.random.uniform(low=-90, high=90, size=self.TL.nStars) * u.deg,
            distance=np.random.uniform(low=1, high=10, size=self.TL.nStars),
        )

    def tearDown(self):
        self.dev_null.close()

    def test_dNbackground(self):
        """
        Test to ensure that dN returned has correct length, units, and is >= 0.
        """
        coords = self.TL.coords
        intDepths = np.random.uniform(15.0, 25.0, len(coords))

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod()
            dN = obj.dNbackground(coords, intDepths)

            self.assertTrue(
                len(dN) == len(intDepths),
                "dNbackground returns different length than input for %s"
                % mod.__name__,
            )
            self.assertTrue(
                dN.unit == 1 / u.arcmin**2,
                "dNbackground does not return 1/arcmin**2 for %s" % mod.__name__,
            )
            self.assertTrue(
                np.all(dN >= 0.0),
                "dNbackground returns negative values for %s" % mod.__name__,
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
            # sys.stdout = StringIO.StringIO()
            sys.stdout = StringIO()
            # call __str__ method
            result = obj.__str__()
            # examine what was printed
            contents = sys.stdout.getvalue()
            self.assertEqual(type(contents), type(""))
            sys.stdout.close()
            # it also returns a string, which is not necessary
            self.assertEqual(type(result), type(""))
            # put stdout back
            sys.stdout = original_stdout
