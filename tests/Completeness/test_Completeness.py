import unittest
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams
import EXOSIMS.Completeness
from EXOSIMS.Prototypes.Completeness import Completeness
from EXOSIMS.Prototypes.TargetList import TargetList
from EXOSIMS.util.get_module import get_module
import os
import pkgutil
import numpy as np
import astropy.units as u
import json
import copy

class TestCompleteness(unittest.TestCase):
    """ 

    Global Completeness tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """
    
    def setUp(self):

        self.dev_null = open(os.devnull, 'w')
        self.script = resource_path('test-scripts/template_minimal.json')
        self.spec = json.loads(open(self.script).read())
        
        with RedirectStreams(stdout=self.dev_null):
            self.TL = TargetList(ntargs=10,**copy.deepcopy(self.spec))
        self.TL.dist = np.random.uniform(low=0,high=100,size=self.TL.nStars)*u.pc
        
        modtype = getattr(Completeness,'_modtype')
        pkg = EXOSIMS.Completeness
        self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__+'.'):
            if (not 'starkAYO' in module_name) and not is_pkg:
                mod = get_module(module_name.split('.')[-1],modtype)
                self.assertTrue(mod._modtype is modtype,'_modtype mismatch for %s'%mod.__name__)
                self.allmods.append(mod)

    def test_init(self):
        """
        Test of initialization and __init__.
        """

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**copy.deepcopy(self.spec))

            self.assertTrue(hasattr(obj,'dMagLim'))
            self.assertTrue(hasattr(obj,'minComp'))

            with RedirectStreams(stdout=self.dev_null):
                obj = mod(dMagLim=5, minComp=0.5, **copy.deepcopy(self.spec))
            self.assertEqual(obj.dMagLim,5)
            self.assertEqual(obj.minComp,0.5)


    def test_target_completeness(self):
        """
        Ensure that target completenesses are generated with proper dimension and bounds
        """
    
        for mod in self.allmods:
            if 'target_completeness' not in mod.__dict__:
                continue

            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**copy.deepcopy(self.spec))
                comp = obj.target_completeness(self.TL)

            self.assertEqual(len(comp),self.TL.nStars,"Incorrect number of completeness values returned for %s"%mod.__name__)
            for c in comp:
                self.assertGreaterEqual(c,0,"Completeness less than zero for %s"%mod.__name__)
                self.assertLessEqual(c,1,"Completeness greater than one for %s"%mod.__name__)

    def test_gen_update(self):
        """
        Ensure that target completeness updates are generated with proper dimension and bounds
        """
    
        for mod in self.allmods:
            if 'gen_update' not in mod.__dict__:
                continue
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**copy.deepcopy(self.spec))
                obj.gen_update(self.TL)
            
            self.assertTrue(hasattr(obj,'updates'),"Updates array not created for  %s"%mod.__name__)
            self.assertTrue(obj.updates.shape == (self.TL.nStars,5),"Updates array improperly sized for %s"%mod.__name__)
            for c in obj.updates.flatten():
                self.assertGreaterEqual(c,0,"Completeness less than zero for %s"%mod.__name__)
                self.assertLessEqual(c,1,"Completeness greater than one for %s"%mod.__name__)



    def test_completeness_update(self):
        """
        Ensure that target completeness updates are properly sized
        """
    
        for mod in self.allmods:
            if 'completeness_update' not in mod.__dict__:
                continue

            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**copy.deepcopy(self.spec))
                obj.gen_update(self.TL)
                dcomp = obj.completeness_update(self.TL, 0, np.array([1]), np.array([100])*u.d)

            self.assertEqual(len(np.array(dcomp,ndmin=1)),1,"Dynamic completeness incorrectly sized for  %s"%mod.__name__)

    def test_revise_update(self):
        """
        Ensure that target completeness update revisions are appropriately sized
        """
    
        for mod in self.allmods:
            if 'revise_updates' not in mod.__dict__:
                continue

            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**copy.deepcopy(self.spec))
                obj.gen_update(self.TL)

            ind = np.sort(np.random.choice(self.TL.nStars,size=int(self.TL.nStars/2.0),replace=False))
            obj.revise_updates(ind)

            self.assertTrue(obj.updates.shape == (len(ind),5),"Updates array improperly resized for %s"%mod.__name__)

    def test_comp_per_intTime(self):

        for mod in self.allmods:
            if 'comp_per_intTime' not in mod.__dict__:
                continue

            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**copy.deepcopy(self.spec))
                comp0 = obj.target_completeness(self.TL)

            comp = obj.comp_per_intTime(np.array([1]*self.TL.nStars)*u.d, self.TL, np.arange(self.TL.nStars),np.array([0])/u.arcsec**2., 
                    np.array([0])/u.arcsec**2., self.TL.OpticalSystem.WA0, self.TL.OpticalSystem.observingModes[0])

            self.assertEqual(len(comp),self.TL.nStars)
            #for c in comp:
                #self.assertGreaterEqual(c,0,"Completeness less than zero for %s"%mod.__name__)
                #self.assertLessEqual(c,1,"Completeness greater than one for %s"%mod.__name__)

    def test_dcomp_dt(self):

        for mod in self.allmods:
            if 'dcomp_dt' not in mod.__dict__:
                continue

            with RedirectStreams(stdout=self.dev_null):
                obj = mod(**copy.deepcopy(self.spec))
                comp0 = obj.target_completeness(self.TL)

            dcomp = obj.dcomp_dt(np.array([1]*self.TL.nStars)*u.d, self.TL, np.arange(self.TL.nStars),np.array([0])/u.arcsec**2., 
                    np.array([0])/u.arcsec**2., self.TL.OpticalSystem.WA0, self.TL.OpticalSystem.observingModes[0])

            self.assertEqual(len(dcomp),self.TL.nStars)

