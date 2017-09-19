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

class TestOpticalSystem(unittest.TestCase):
    """ 

    Global OpticalSystem tests.
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
        
        modtype = getattr(OpticalSystem,'_modtype')
        pkg = EXOSIMS.OpticalSystem
        self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__+'.'):
            if not is_pkg:
                mod = get_module(module_name.split('.')[-1],modtype)
                self.assertTrue(mod._modtype is modtype,'_modtype mismatch for %s'%mod.__name__)
                self.allmods.append(mod)


    def test_Cp_Cb_Csp(self):
        """
        Sanity check Cp_Cb_Csp calculations.
        """

        for mod in self.allmods:
            if 'Cp_Cb_Csp' not in mod.__dict__:
                continue

            obj = mod(**copy.deepcopy(self.spec))

            #first check, infinite dMag should give zero C_p
            C_p,C_b,C_sp = obj.Cp_Cb_Csp(self.TL, np.arange(self.TL.nStars), np.array([0]*self.TL.nStars)/(u.arcsec**2.),
                    np.array([0]*self.TL.nStars)/(u.arcsec**2.),np.ones(self.TL.nStars)*np.inf,
                    np.array([obj.WA0.value]*self.TL.nStars)*obj.WA0.unit,obj.observingModes[0])
            self.assertEqual(len(C_p),len(C_b))
            self.assertEqual(len(C_b),len(C_sp))
            self.assertTrue(np.all(C_p.value == 0))


            #second check, outside OWA, C_p and C_sp should be all zero (C_b may be non-zero due to read/dark noise)
            C_p,C_b,C_sp = obj.Cp_Cb_Csp(self.TL, np.arange(self.TL.nStars), np.array([0]*self.TL.nStars)/(u.arcsec**2.),
                    np.array([0]*self.TL.nStars)/(u.arcsec**2.),np.ones(self.TL.nStars)*obj.dMag0,
                    np.array([obj.observingModes[0]['OWA'].value*2.]*self.TL.nStars)*obj.observingModes[0]['OWA'].unit,obj.observingModes[0])
            self.assertTrue(np.all(C_p.value == 0))
            self.assertTrue(np.all(C_sp.value == 0))

            
            #third check, inside IWA, C_p and C_sp should be all zero (C_b may be non-zero due to read/dark noise)
            C_p,C_b,C_sp = obj.Cp_Cb_Csp(self.TL, np.arange(self.TL.nStars), np.array([0]*self.TL.nStars)/(u.arcsec**2.),
                    np.array([0]*self.TL.nStars)/(u.arcsec**2.),np.ones(self.TL.nStars)*obj.dMag0,
                    np.array([obj.observingModes[0]['IWA'].value/2.]*self.TL.nStars)*obj.observingModes[0]['IWA'].unit,obj.observingModes[0])
            self.assertTrue(np.all(C_p.value == 0))
            self.assertTrue(np.all(C_sp.value == 0))


    def test_calc_intTime(self):
        """
        Check calc_intTime i/o only
        """

        for mod in self.allmods:
            if 'calc_intTime' not in mod.__dict__:
                continue

            obj = mod(**copy.deepcopy(self.spec))

            #first check, infinite dMag should give zero C_p
            intTime = obj.calc_intTime(self.TL, np.arange(self.TL.nStars), np.array([0]*self.TL.nStars)/(u.arcsec**2.),
                    np.array([0]*self.TL.nStars)/(u.arcsec**2.),np.ones(self.TL.nStars)*obj.dMag0,
                    np.array([obj.WA0.value]*self.TL.nStars)*obj.WA0.unit,obj.observingModes[0])

            
            self.assertEqual(len(intTime),self.TL.nStars)


    def test_calc_minintTime(self):
        """
        Check calc_minintTime i/o and sanity check against intTime
        """

        for mod in self.allmods:
            if 'calc_intTime' not in mod.__dict__:
                continue
            obj = mod(**copy.deepcopy(self.spec))

            #first check, infinite dMag should give zero C_p
            intTime = obj.calc_intTime(self.TL, np.arange(self.TL.nStars), np.array([0]*self.TL.nStars)/(u.arcsec**2.),
                    np.array([0]*self.TL.nStars)/(u.arcsec**2.),np.ones(self.TL.nStars)*obj.dMag0*1.5,
                    np.array([obj.WA0.value]*self.TL.nStars)*obj.WA0.unit,obj.observingModes[0])

            minTime = obj.calc_minintTime(self.TL)
            for x,y in zip(minTime,intTime):
                self.assertLessEqual(x,y)


    def test_calc_dMag_per_intTime(self):
        """
        Check calc_dMag_per_intTime i/o 
        """

        for mod in self.allmods:
            if 'calc_dMag_per_intTime' not in mod.__dict__:
                continue
            obj = mod(**copy.deepcopy(self.spec))
            
            dMag = obj.calc_dMag_per_intTime(np.ones(self.TL.nStars)*u.day,
                    self.TL, np.arange(self.TL.nStars), 
                    np.array([0]*self.TL.nStars)/(u.arcsec**2.),np.array([0]*self.TL.nStars)/(u.arcsec**2.),
                    np.array([obj.WA0.value]*self.TL.nStars)*obj.WA0.unit,obj.observingModes[0])
        
            self.assertEqual(dMag.shape,(self.TL.nStars,self.TL.nStars))
            
            

    def test_ddMag_dt(self):
        """
        Check ddMag_dt i/o 
        """

        for mod in self.allmods:
            if 'ddMag_dt' not in mod.__dict__:
                continue
            obj = mod(**copy.deepcopy(self.spec))

            ddMag = obj.ddMag_dt(np.ones(self.TL.nStars)*u.day,
                    self.TL, np.arange(self.TL.nStars), 
                    np.array([0]*self.TL.nStars)/(u.arcsec**2.),np.array([0]*self.TL.nStars)/(u.arcsec**2.),
                    np.array([obj.WA0.value]*self.TL.nStars)*obj.WA0.unit,obj.observingModes[0])
            
            self.assertEqual(ddMag.shape,(self.TL.nStars,self.TL.nStars))


