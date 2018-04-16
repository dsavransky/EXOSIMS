import unittest
import numpy as np
import os
import EXOSIMS
from  EXOSIMS import MissionSim
from EXOSIMS.Prototypes import TargetList
import numpy as np
from astropy import units as u
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams
import json
import copy

r"""TargetList module unit tests

Paul Nunez, JPL, Aug. 2016
"""

scriptfile = resource_path('test-scripts/template_prototype_testing.json')

class Test_TargetList_prototype(unittest.TestCase):
    dev_null = open(os.devnull, 'w')

    def setUp(self):
        self.spec = json.loads(open(scriptfile).read())
        
        # quiet the chatter at initialization
        with RedirectStreams(stdout=self.dev_null):
            sim = MissionSim.MissionSim(**self.spec)
        self.targetlist = sim.TargetList
        self.opticalsystem = sim.OpticalSystem
        self.planetpop = sim.PlanetPopulation
    
    def test_nan_filter(self):
        #Introduce a few nans to see if they are filtered out
        n0 = len(self.targetlist.Umag)
        self.targetlist.nan_filter()

        self.assertEqual(len(self.targetlist.Name), n0)
        self.targetlist.Umag[2]=float('nan')
        self.targetlist.nan_filter()

        self.assertEqual(len(self.targetlist.Name), n0-1)

        #insert another nan for testing
        self.targetlist.dist[10]=float('nan')
        self.targetlist.nan_filter()

        self.assertEqual( len(self.targetlist.Name), n0-2 ) 
        

    def test_binary_filter(self):
        n0 = self.targetlist.nStars
        #adding 3 binaries by hand
        self.targetlist.Binary_Cut[1] = True
        self.targetlist.Binary_Cut[3] = True
        self.targetlist.Binary_Cut[10] = True
        self.targetlist.binary_filter()
        n1 = self.targetlist.nStars
        #3 binaries should be removed
        self.assertEqual( n1, n0-3 )

    def test_outside_IWA_filter(self):
        n0 = self.targetlist.nStars
        #Test default IWA = 0
        self.targetlist.outside_IWA_filter()
        n1 = self.targetlist.nStars

        self.assertEqual( n0, n1 )

        #Test particular case
        self.opticalsystem.IWA = 10 * u.arcsec
        self.targetlist.outside_IWA_filter()
        #assert self.targetlist.nStars == 417 #not a useful test
        n1 = self.targetlist.nStars #reference 
        #introduce two stars with planet below 10 arcsec. should be removed
        self.targetlist.dist[10] = 21 * u.pc #rrange is 1e-3 to 200au, so this planet is below the IWA of 10 arcsec 
        self.targetlist.dist[12] = 22 * u.pc
        self.targetlist.outside_IWA_filter()

        self.assertEqual( self.targetlist.nStars , n1 - 2 )

        #Test limiting case of IWA = PI/2
        self.opticalsystem.IWA = 3.14/2 * u.rad
        with self.assertRaises(IndexError):
            self.targetlist.outside_IWA_filter()
        #self.assertEqual(targetlist.nStars, 0) #Note that nStars is now zero so I can no longer filter out stars. This is why the limiting case of dMagLim = 0 should be done last

    def test_vis_mag_filter(self):
        n0 = self.targetlist.nStars
        #Test limiting case
        vmax = np.inf
        self.targetlist.vis_mag_filter(vmax)

        self.assertEqual( self.targetlist.nStars , n0 )

        vmax = 8.0
        self.targetlist.vis_mag_filter(vmax)
        n1 = self.targetlist.nStars #reference
        #assert self.targetlist.nStars == 710 #not a useful test
        #two stars with vmax = 8.1 should be filtered out
        self.targetlist.Vmag[10] = 8.1
        self.targetlist.Vmag[12] = 9.1
        self.targetlist.vis_mag_filter(vmax)

        self.assertEqual( self.targetlist.nStars , n1 - 2 )

        # Test another limiting case
        vmax = -10
        with self.assertRaises(IndexError):
            self.targetlist.vis_mag_filter(vmax)
        #self.assertEqual( self.targetlist.nStars , 0 )

    def test_dmag_filter(self):
        n0 = self.targetlist.nStars
        #Test default with IWA = 0  , dMagLim = 22.5
        self.targetlist.max_dmag_filter()
        n1 = self.targetlist.nStars
        self.assertEqual( n0 , n1 )
        #Test limiting case of dMagLim = inf
        self.targetlist.Completeness.dMagLim = np.inf
        self.targetlist.max_dmag_filter()
        self.assertEqual( self.targetlist.nStars , n0)
        #Test limiting case of dMagLim = 0
        self.targetlist.Completeness.dMagLim = 0.0
        with self.assertRaises(IndexError):
            self.targetlist.max_dmag_filter()
        #self.assertEqual( self.targetlist.nStars , 0 ) #Note that nStars is now zero so I can no longer filter out stars. 

    def test1_dmag_filter(self):
        #Test limiting case that distance to a star is (effectively) infinite
        # turmon: changed from inf to 1e8 because inf causes a confusing RuntimeWarning
        self.planetpop.rrange = np.array([1e8,1e8])*u.AU
        with self.assertRaises(IndexError):
            self.targetlist.max_dmag_filter()      
        #self.assertEqual( self.targetlist.nStars , 0)
        

    def test_int_cutoff_filter(self):
        n0 = self.targetlist.nStars
        #Test default 
        self.targetlist.int_cutoff_filter()
        self.assertEqual( self.targetlist.nStars , n0)
        #Test limiting case of infinite max integration time
        self.opticalsystem.intCutoff = np.array([np.inf]) * u.day
        self.targetlist.int_cutoff_filter()
        self.assertEqual( self.targetlist.nStars , n0)
        #Test limiting case of zero max integration time
        self.opticalsystem.intCutoff = np.array([0]) * u.day
        with self.assertRaises(IndexError):
            self.targetlist.int_cutoff_filter()
        #self.assertEqual( self.targetlist.nStars , 0)

    def test_completeness_filter(self):
        n0 = self.targetlist.nStars
        self.targetlist.completeness_filter()
        self.assertEqual( self.targetlist.nStars , n0)
        #Test limiting case of minComp = 1.0
        self.targetlist.Completeness.minComp = 1.0
        with self.assertRaises(IndexError):
            self.targetlist.completeness_filter()
        #self.assertEqual(self.targetlist.nStars , 0)

    def test_life_expectancy_filter(self):
        #test default removal of BV < 0.3 (hard-coded)
        self.targetlist.life_expectancy_filter()
        self.assertEqual( np.any(self.targetlist.BV<0.3) , False)

    def test_main_sequence_filter(self):
        n0 = self.targetlist.nStars
        self.targetlist.main_sequence_filter()
        #print self.targetlist.nStars
        #Check that no stars fall outside main sequence strip
        self.assertEqual( np.any((self.targetlist.BV < 0.74) & (self.targetlist.MV > 6*self.targetlist.BV+1.8)) , False)
        self.assertEqual( np.any((self.targetlist.BV >= 0.74) & (self.targetlist.BV < 1.37) & (self.targetlist.MV > 4.3*self.targetlist.BV+3.05)) , False)
        self.assertEqual( np.any((self.targetlist.BV >= 1.37) & (self.targetlist.MV > 18*self.targetlist.BV-15.7)) , False)
        self.assertEqual( np.any((self.targetlist.BV < 0.87) & (self.targetlist.MV < -8*(self.targetlist.BV-1.35)**2+7.01)) , False)
        self.assertEqual( np.any((self.targetlist.BV >= 0.87) & (self.targetlist.BV < 1.45) & (self.targetlist.MV > 5*self.targetlist.BV+0.81)) , False)
        self.assertEqual( np.any((self.targetlist.BV >= 1.45) & (self.targetlist.MV < 18*self.targetlist.BV-18.04)) , False)
        #check that filtered target list does not have repeating elements
        import collections
        compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
        self.assertEqual( compare(list(set(self.targetlist.Name)), list(self.targetlist.Name)) , True)

    
    def test_stellar_mass(self):
        #Test with absolute magnitue of the sun
        self.targetlist.MV = np.array([4.83])
        self.targetlist.stellar_mass()
        #Should give 1 solar mass approximately
        np.testing.assert_allclose(self.targetlist.MsEst[0], 1.05865*u.solMass, rtol=1e-5, atol=0)
        #Relative tolerance is 0.07 
        np.testing.assert_allclose(self.targetlist.MsTrue[0], 1.05865*u.solMass, rtol=0.07, atol=0)

    def test_fgk_filter(self):
        self.targetlist.fgk_filter()
        #check that there are no other spectral types besides FGK
        #Had to rewrite fgk_filter
        for i in range(len(self.targetlist.Spec)):
            self.assertNotEqual( self.targetlist.Spec[i][0]  , 'O' )
            self.assertNotEqual( self.targetlist.Spec[i][0]  , 'B' )
            self.assertNotEqual( self.targetlist.Spec[i][0]  , 'A' )
            self.assertNotEqual( self.targetlist.Spec[i][0]  , 'M')
            assert (self.targetlist.Spec[i][0] == 'F' or self.targetlist.Spec[i][0] == 'G' or self.targetlist.Spec[i][0] == 'K')


    def test_revise_lists(self):
        #Check that passing all indices does not change list
        #and that coordinates are in degrees
        i0 = range(len(self.targetlist.Name))
        self.targetlist.revise_lists(i0)        
        self.assertEqual( len(i0) , len(self.targetlist.Name))
        #Check to see that only 3 elements are retained
        i1=np.array([1,5,10])
        self.targetlist.revise_lists(i1)        
        self.assertEqual( len(i1) , len(self.targetlist.Name))
        #Check to see that passing no indices yields an emply list
        i2=[]
        with self.assertRaises(IndexError):
            self.targetlist.revise_lists(i2)        
        #self.assertEqual( len(self.targetlist.Name) , 0)
       
    def test_fillPhotometry(self):
        """
        Filling in photometry should result in larger or equal sized target list
        """

        with RedirectStreams(stdout=self.dev_null):
            sim = MissionSim.MissionSim(scriptfile,fillPhotometry=True)

        self.assertTrue(sim.TargetList.fillPhotometry)
        self.assertGreaterEqual(sim.TargetList.nStars, self.targetlist.nStars)
    
    def test_completeness_specs(self):
        """
        Test completeness_specs logic
        """
        
        # test case where no completeness specs given
        self.assertEqual(self.targetlist.PlanetPopulation.__class__.__name__,self.targetlist.Completeness.PlanetPopulation.__class__.__name__)
        
        # test case where completeness specs given
        spec2 = json.loads(open(scriptfile).read())
        spec2['completeness_specs'] = {'modules': {"PlanetPopulation": "EarthTwinHabZone1", \
             "PlanetPhysicalModel": "PlanetPhysicalModel"}}
        spec2['explainFiltering'] = True
        spec2['scaleOrbits'] = True
        
        tl = TargetList.TargetList(**spec2)
        self.assertNotEqual(tl.PlanetPopulation.__class__.__name__,tl.Completeness.PlanetPopulation.__class__.__name__)



if __name__ == '__main__':
    unittest.main()

