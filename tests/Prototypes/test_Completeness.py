import unittest
from EXOSIMS.Prototypes import Completeness
import numpy as np
import astropy.units as u


class Test_Completeness_Prototype(unittest.TestCase):
    """
    The prototype implementation of completeness returns only dummy outputs
    so all these tests are only to ensure that input/output is maintained.

    """

    def setUp(self):
        self.spec = {'modules':{'PlanetPhysicalModel':' ','PlanetPopulation':' '}}
        self.fixture = Completeness.Completeness(**self.spec)

        self.nStars = 100  
    
    def tearDown(self):
        pass

    
    def test_target_completeness(self):
        
        int_comp = self.fixture.target_completeness(self)

        self.assertEqual(len(int_comp),self.nStars)

    def test_gen_update(self):

        self.fixture.gen_update(self)
        self.assertEqual(self.fixture.updates.shape[0],self.nStars)

    def test_completeness_update(self):

        self.int_comp = self.fixture.target_completeness(self)
        int_comp = self.fixture.completeness_update(self,np.arange(self.nStars),np.zeros(self.nStars),1*u.d)

        self.assertTrue(np.all(self.int_comp == int_comp))

    def test_revise_updates(self):

        self.fixture.gen_update(self)
        self.fixture.revise_updates(0)
        self.assertEqual(self.fixture.updates.shape[0],5)

    def tests_comp_per_intTime(self):

        comp = self.fixture.comp_per_intTime(np.ones(self.nStars)*u.d,self,np.arange(self.nStars),0/(u.arcsec**2),0/(u.arcsec**2),0*u.arcsec,{})
        
        self.assertTrue(len(comp),self.nStars)

        with self.assertRaises(AssertionError):
            comp = self.fixture.comp_per_intTime(np.ones(self.nStars-1)*u.d,self,np.arange(self.nStars),0/(u.arcsec**2),0/(u.arcsec**2),0*u.arcsec,{})

        with self.assertRaises(AssertionError):
            comp = self.fixture.comp_per_intTime(np.ones(self.nStars)*u.d,self,np.arange(self.nStars),np.zeros(2)/(u.arcsec**2),0/(u.arcsec**2),0*u.arcsec,{})

        with self.assertRaises(AssertionError):
            comp = self.fixture.comp_per_intTime(np.ones(self.nStars)*u.d,self,np.arange(self.nStars),0/(u.arcsec**2),np.zeros(2)/(u.arcsec**2),0*u.arcsec,{})

        with self.assertRaises(AssertionError):
            comp = self.fixture.comp_per_intTime(np.ones(self.nStars-1)*u.d,self,np.arange(self.nStars),0/(u.arcsec**2),0/(u.arcsec**2),np.zeros(2)*u.arcsec,{})

    def test_dcomp_dt(self):

        dt = self.fixture.dcomp_dt(np.ones(self.nStars)*u.d,self,np.arange(self.nStars),0/(u.arcsec**2),0/(u.arcsec**2),0*u.arcsec,{})
        
        self.assertTrue(len(dt),self.nStars)

        with self.assertRaises(AssertionError):
            dt = self.fixture.dcomp_dt(np.ones(self.nStars-1)*u.d,self,np.arange(self.nStars),0/(u.arcsec**2),0/(u.arcsec**2),0*u.arcsec,{})

        with self.assertRaises(AssertionError):
            dt = self.fixture.dcomp_dt(np.ones(self.nStars)*u.d,self,np.arange(self.nStars),np.zeros(2)/(u.arcsec**2),0/(u.arcsec**2),0*u.arcsec,{})

        with self.assertRaises(AssertionError):
            dt = self.fixture.dcomp_dt(np.ones(self.nStars)*u.d,self,np.arange(self.nStars),0/(u.arcsec**2),np.zeros(2)/(u.arcsec**2),0*u.arcsec,{})

        with self.assertRaises(AssertionError):
            dt = self.fixture.dcomp_dt(np.ones(self.nStars-1)*u.d,self,np.arange(self.nStars),0/(u.arcsec**2),0/(u.arcsec**2),np.zeros(2)*u.arcsec,{})

    def test_completeness_specs(self):
        spec = {'completeness_specs': {}, 'modules': {'PlanetPopulation': ' ', 'PlanetPhysicalModel': ' '}}
        Comp = Completeness.Completeness(**spec)
        
        self.assertTrue(Comp.PlanetPopulation.__class__.__name__=='PlanetPopulation',"empty completeness_specs did not load prototype PlanetPopulation")
        self.assertTrue(Comp.PlanetPhysicalModel.__class__.__name__=='PlanetPhysicalModel',"empty completeness_specs did not load prototype PlanetPhysicalModel")
