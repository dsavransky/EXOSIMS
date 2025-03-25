import unittest
from tests.TestSupport.Utilities import RedirectStreams
import EXOSIMS.StarCatalog
from EXOSIMS.Prototypes.StarCatalog import StarCatalog
from EXOSIMS.util.get_module import get_module
import os
import sys
import pkgutil
from io import StringIO


class TestStarCatalog(unittest.TestCase):
    """

    Global StarCatalog tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """

    def setUp(self):

        self.dev_null = open(os.devnull, "w")

        modtype = getattr(StarCatalog, "_modtype")
        pkg = EXOSIMS.StarCatalog
        self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(
            pkg.__path__, pkg.__name__ + "."
        ):
            if (
                ("Gaia" not in module_name)
                and ("HIPfromSimbad" not in module_name)
                and ("plandbcat" not in module_name)
                and not (is_pkg)
            ):
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

        Only checks for existence and uniform size of basic catalog parameters
        """

        req_atts = [
            "Name",
            "Spec",
            "parx",
            "Umag",
            "Bmag",
            "Vmag",
            "Rmag",
            "Imag",
            "Jmag",
            "Hmag",
            "Kmag",
            "dist",
            "BV",
            "MV",
            "BC",
            "L",
            "coords",
            "pmra",
            "pmdec",
            "rv",
            "Binary_Cut",
        ]

        for mod in self.allmods:
            if "__init__" not in mod.__dict__:
                continue

            with RedirectStreams(stdout=self.dev_null):
                obj = mod()

            self.assertTrue(hasattr(obj, "ntargs"))
            for att in req_atts:
                self.assertTrue(hasattr(obj, att))
                self.assertEqual(len(getattr(obj, att)), obj.ntargs)

    def test_str(self):
        """
        Test __str__ method, for full coverage and check that all modules have
        required attributes.
        """
        atts_list = [
            "Name",
            "Spec",
            "parx",
            "Umag",
            "Bmag",
            "Vmag",
            "Rmag",
            "Imag",
            "Jmag",
            "Hmag",
            "Kmag",
            "dist",
            "BV",
            "MV",
            "BC",
            "L",
            "coords",
            "pmra",
            "pmdec",
            "rv",
            "Binary_Cut",
        ]

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
            for att in atts_list:
                self.assertIn(
                    att, contents, "{} missing for {}".format(att, mod.__name__)
                )
            sys.stdout.close()
            # it also returns a string, which is not necessary
            self.assertEqual(type(result), type(""))
            # put stdout back
            sys.stdout = original_stdout
