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

class TestPostProcessing(unittest.TestCase):
    """ 

    Global PostProcessing tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """
    
    def setUp(self):

        self.dev_null = open(os.devnull, 'w')
        self.specs = {'modules':{'BackgroundSources':' '}}
        script = resource_path('test-scripts/template_minimal.json')
        spec = json.loads(open(script).read())     
        with RedirectStreams(stdout=self.dev_null):
            self.TL = TargetList(**spec)

        modtype = getattr(EXOSIMS.Prototypes.PostProcessing.PostProcessing,'_modtype')
        pkg = EXOSIMS.PostProcessing
        self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__+'.'):
            if not is_pkg:
                mod = get_module(module_name.split('.')[-1],modtype)
                self.assertTrue(mod._modtype is modtype,'_modtype mismatch for %s'%mod.__name__)
                self.allmods.append(mod)


    #Testing some limiting cases below
    def test_zeroFAP(self):
        """
        Test case where false alarm probability is 0 and missed det prob is 0
        All observations above SNR limit should be detected
        """

        SNRin = np.array([1.0,2.0,3.0,4.0,5.0,5.1,6.0])
        expected_MDresult = [True,True,True,True,False,False,False] 

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(FAP=0.0,MDP=0.0,**self.specs)

            FA, MD = obj.det_occur(SNRin,self.TL.OpticalSystem.observingModes[0],self.TL,0,1*u.d)     
            for x,y in zip(MD,expected_MDresult):
                self.assertEqual( x,y )
            self.assertEqual( FA, False)

    #another limiting case
    def test_oneFAP(self):
        """
        Test case where false alarm probability is 1 and missed det prob is 0
        """

        SNRin = np.array([1.0,2.0,3.0,4.0,5.0,5.1,6.0])
        expected_MDresult = [True,True,True,True,False,False,False] 

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(FAP=1.0,MDP=0.0,**self.specs)

            FA, MD = obj.det_occur(SNRin,self.TL.OpticalSystem.observingModes[0],self.TL,0,1*u.d)     
            for x,y in zip(MD,expected_MDresult):
                self.assertEqual( x,y )
            self.assertEqual( FA, True)


