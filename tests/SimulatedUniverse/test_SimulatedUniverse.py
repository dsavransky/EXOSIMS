import unittest
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams
import EXOSIMS.SimulatedUniverse
from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
from EXOSIMS.Prototypes.TargetList import TargetList
from EXOSIMS.util.get_module import get_module
import os
import pkgutil
import numpy as np
import astropy.units as u
import astropy.constants as const
import json
import copy

class TestSimulatedUniverse(unittest.TestCase):
    """ 

    Global SimulatedUniverse tests.
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
        
        modtype = getattr(SimulatedUniverse,'_modtype')
        pkg = EXOSIMS.SimulatedUniverse
        self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__+'.'):
            if (not 'starkAYO' in module_name) and not is_pkg:
                mod = get_module(module_name.split('.')[-1],modtype)
                self.assertTrue(mod._modtype is modtype,'_modtype mismatch for %s'%mod.__name__)
                self.allmods.append(mod)

    def test_init(self):
        """
        Test of initialization and __init__.

        Because some implementations depend on a specific planet population,
        there needs to be additional logic in the setup
        """

        req_atts = ['plan2star', 'a', 'e', 'I', 'O', 'w', 'Min', 'M0', 'Rp', 'Mp', 'p',
                    'r', 'v', 'd', 's', 'phi', 'fEZ', 'dMag', 'WA']

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                spec = copy.deepcopy(self.spec)
                spec['modules']['PlanetPhysicalModel']='FortneyMarleyCahoyMix1'
                spec['modules']['StarCatalog']='EXOCAT1'
                if 'Kepler' in mod.__name__:
                    spec['modules']['PlanetPopulation']='KeplerLike1'
                elif 'KnownRV' in mod.__name__:
                    spec['modules']['PlanetPopulation']='KnownRVPlanets'
                    spec['modules']['TargetList']='KnownRVPlanetsTargetList'
                elif 'KnownRV' in mod.__name__:
                    spec['modules']['PlanetPopulation']='SAG13'

                obj = mod(**spec)

            #verify that all attributes are there
            for att in req_atts:
                self.assertTrue(hasattr(obj,att))

            #planet properties must all be the same size
            self.assertTrue(len(obj.a) == len(obj.e) == len(obj.I) == len(obj.O) == len(obj.w) ==
                    len(obj.M0) == len(obj.Rp) == len(obj.Mp) == len(obj.p) == len(obj.d) == len(obj.s)
                    == len(obj.phi) == len(obj.fEZ) == len(obj.dMag) == len(obj.WA) == obj.nPlans,
                    "Planet parameters do not have all same lengths in %s"%mod.__name__)
        
            # r and v must be nx3
            self.assertEqual(obj.r.shape,(obj.nPlans,3),"r has incorrect shape in %s"%mod.__name__)
            self.assertEqual(obj.v.shape,(obj.nPlans,3),"v has incorrect shape in %s"%mod.__name__)

            #basic sanity checks
            self.assertTrue(np.all(np.linalg.norm(obj.r,axis=1) == obj.d.value),"r and d do not match magnitudes in %s"%mod.__name__)
            self.assertTrue(np.all(obj.s <= obj.d),"Projected separation exceeds orbital radius in %s"%mod.__name__)
            #self.assertTrue(np.all(obj.d <= obj.a*(1+obj.e)),"Orbital radius exceeds sma*(1+e) in %s"%mod.__name__)
            #self.assertTrue(np.all(obj.d >= obj.a*(1-obj.e)),"Orbital radius exceeds sma*(1-e) in %s"%mod.__name__)

            #if module has its own propagator, spin first planet forward by one period and check that it returns to starting position
            if 'propag_system' in mod.__dict__:
                sInd = obj.plan2star[0]
                pInds = np.where(obj.plan2star == sInd)[0]
                pInd = pInds[0]
                
                Ms = obj.TargetList.MsTrue[[sInd]]
                Mp = obj.Mp[pInd]
                mu = (const.G*(Mp + Ms)).to('AU3/day2')
                dt = np.sqrt(4*np.pi**2.*obj.a[pInd]**3.0/mu).to(u.day)
                
                r0 = obj.r[pInd].copy()
                v0 = obj.v[pInd].copy()
                
                obj.propag_system(sInd,dt)
                np.testing.assert_allclose(r0, obj.r[pInd],err_msg="propagated r mismatch in %s"%mod.__name__)
                np.testing.assert_allclose(v0, obj.v[pInd],err_msg="propagated r mismatch in %s"%mod.__name__)


    def test_honor_scaleOrbit(self):
        """
        Test that scaleOrbit flag is honored

        Because some implementations depend on a specific planet population,
        there needs to be additional logic in the setup
        """

        whitelist = ['KeplerLikeUniverse','KnownRVPlanetsUniverse','SAG13Universe']
        for mod in self.allmods:
            if mod.__name__ in whitelist:
                continue
            with RedirectStreams(stdout=self.dev_null):
                spec = copy.deepcopy(self.spec)
                spec['modules']['PlanetPhysicalModel']='FortneyMarleyCahoyMix1'
                spec['modules']['StarCatalog']='EXOCAT1'
                if 'Kepler' in mod.__name__:
                    spec['modules']['PlanetPopulation']='KeplerLike1'
                elif 'KnownRV' in mod.__name__:
                    spec['modules']['PlanetPopulation']='KnownRVPlanets'
                    spec['modules']['TargetList']='KnownRVPlanetsTargetList'
                elif 'KnownRV' in mod.__name__:
                    spec['modules']['PlanetPopulation']='SAG13'
                elif 'SAG13' in mod.__name__:
                    spec['modules']['PlanetPopulation']='SAG13'

                obj = mod(scaleOrbits=True,**spec)

            self.assertTrue(obj.PlanetPopulation.scaleOrbits,"scaleOrbits not set in %s"%mod.__name__)

            aeff = obj.a/np.sqrt(obj.TargetList.L[obj.plan2star])
            self.assertTrue(np.all(aeff <= obj.PlanetPopulation.arange[1]),"scaled sma out of bounds in %s"%mod.__name__)
            self.assertTrue(np.all(aeff >= obj.PlanetPopulation.arange[0]),"scaled sma out of bounds in %s"%mod.__name__)

    def test_Honor_fixedPlanPerStar(self):
        """
        Test that fixed PlanPerStar flag passes through integers and None
        """

        spec = json.loads(open(self.script).read())
        #If fixedPlanPerStar is not Defined
        SU = SimulatedUniverse(**spec)
        self.assertTrue(SU.fixedPlanPerStar==None)

        #For 1 star
        del SU
        script = resource_path('test-scripts/template_minimal.json')
        spec = json.loads(open(self.script).read())
        #If fixedPlanPerStar is defined
        spec['fixedPlanPerStar'] = 1
        SU = SimulatedUniverse(**spec)
        self.assertTrue(SU.fixedPlanPerStar==1)
        self.assertTrue(SU.plan2star == np.unique(SU.plan2star))
        self.assertTrue(SU.TargetList.nStars*SU.fixedPlanPerStar == SU.nPlans)  
        self.assertTrue(SU.nPlans == 1)#for this specific test instance

        #For a random integer of stars
        del SU
        script = resource_path('test-scripts/template_minimal.json')
        spec = json.loads(open(self.script).read())
        #If fixedPlanPerStar is defined
        n = np.random.randint(0,100)

        spec['fixedPlanPerStar'] = n
        SU = SimulatedUniverse(**spec)
        SU.TargetList.nStars = np.random.randint(0,100)#randomly generate a number of stars in nStars
        SU.TargetList.Name[0] = 'TACO47'#Needs to be anything but prototype to ensure self attributes are not reset
        SU.gen_physical_properties(**spec)#update parameters in gen_physical_properties
        self.assertTrue(SU.fixedPlanPerStar==n)
        self.assertTrue(SU.nPlans == SU.TargetList.nStars*SU.fixedPlanPerStar)
        self.assertTrue(len(SU.plan2star) == SU.TargetList.nStars*SU.fixedPlanPerStar)
        
    def test_honor_Min(self):
        """
        Test that gen_M0 assigns constant or random value to mean anomaly
        
        Because some implementations depend on a specific planet population,
        there needs to be additional logic in the setup
        """
        
        whitelist = ['KnownRVPlanetsUniverse']
        # Test Min = None first
        for mod in self.allmods:
            if mod.__name__ in whitelist:
                continue
            with RedirectStreams(stdout=self.dev_null):
                spec = copy.deepcopy(self.spec)
                spec['modules']['PlanetPhysicalModel']='FortneyMarleyCahoyMix1'
                spec['modules']['StarCatalog']='EXOCAT1'
                if 'Kepler' in mod.__name__:
                    spec['modules']['PlanetPopulation']='KeplerLike1'
                elif 'SAG13' in mod.__name__:
                    spec['modules']['PlanetPopulation']='SAG13'
                    spec['Rprange'] = [1,10]
                    
                obj = mod(**spec)
            
            self.assertTrue(obj.M0[0] != obj.M0[1],"Initial M0 must be randomly set")
        
        # Test Min = 20
        for mod in self.allmods:
            if mod.__name__ in whitelist:
                continue
            with RedirectStreams(stdout=self.dev_null):
                spec = copy.deepcopy(self.spec)
                spec['modules']['PlanetPhysicalModel']='FortneyMarleyCahoyMix1'
                spec['modules']['StarCatalog']='EXOCAT1'
                if 'Kepler' in mod.__name__:
                    spec['modules']['PlanetPopulation']='KeplerLike1'
                elif 'SAG13' in mod.__name__:
                    spec['modules']['PlanetPopulation']='SAG13'
                    spec['Rprange'] = [1,10]
                spec['Min'] = 20    
                obj = mod(**spec)

            self.assertTrue(np.all(obj.M0.to('deg').value == 20),"Initial M0 must be 20")
        



