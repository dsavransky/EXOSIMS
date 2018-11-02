import unittest
from tests.TestSupport.Utilities import RedirectStreams
import EXOSIMS.StarCatalog
from EXOSIMS.Prototypes.StarCatalog import StarCatalog
from EXOSIMS.util.get_module import get_module
import os
import pkgutil

class TestStarCatalog(unittest.TestCase):
    """ 

    Global StarCatalog tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """
    
    def setUp(self):

        self.dev_null = open(os.devnull, 'w')
        
        modtype = getattr(StarCatalog,'_modtype')
        pkg = EXOSIMS.StarCatalog
        self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__+'.'):
            if (not 'Gaia' in module_name) and \
            not is_pkg:
                mod = get_module(module_name.split('.')[-1],modtype)
                self.assertTrue(mod._modtype is modtype,'_modtype mismatch for %s'%mod.__name__)
                self.allmods.append(mod)


    def test_init(self):
        """
        Test of initialization and __init__.

        Only checks for existence and uniform size of basic catalog parameters
        """

        req_atts = ['Name', 'Spec', 'parx', 'Umag', 'Bmag', 'Vmag', 'Rmag', 
                    'Imag', 'Jmag', 'Hmag', 'Kmag', 'dist', 'BV', 'MV', 'BC', 'L', 
                    'coords', 'pmra', 'pmdec', 'rv', 'Binary_Cut']

        for mod in self.allmods:
            if '__init__' not in mod.__dict__:
                continue

            with RedirectStreams(stdout=self.dev_null):
                obj = mod()

            self.assertTrue(hasattr(obj,'ntargs'))
            for att in req_atts:
                self.assertTrue(hasattr(obj,att))
                self.assertEqual(len(getattr(obj,att)),obj.ntargs)

