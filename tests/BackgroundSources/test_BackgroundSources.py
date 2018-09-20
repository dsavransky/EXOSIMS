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

class TestBackgroundSources(unittest.TestCase):
    """

    Global BackgroundSources tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """

    def setUp(self):

        self.dev_null = open(os.devnull, 'w')
        modtype = getattr(EXOSIMS.Prototypes.BackgroundSources.BackgroundSources,'_modtype')
        pkg = EXOSIMS.BackgroundSources
        self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__+'.'):
            if not is_pkg:
                mod = get_module(module_name.split('.')[-1],modtype)
                self.assertTrue(mod._modtype is modtype,'_modtype mismatch for %s'%mod.__name__)
                self.allmods.append(mod)
        # need a TargetList object for testing
        script = resource_path('test-scripts/template_prototype_testing.json')
        spec = json.loads(open(script).read())
        with RedirectStreams(stdout=self.dev_null):
            self.TL = TargetList(**spec)


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

            self.assertTrue(len(dN) == len(intDepths),'dNbackground returns different length than input for %s'%mod.__name__)
            self.assertTrue(dN.unit == 1/u.arcmin**2,'dNbackground does not return 1/arcmin**2 for %s'%mod.__name__)
            self.assertTrue(np.all(dN >= 0.0),'dNbackground returns negative values for %s'%mod.__name__)

