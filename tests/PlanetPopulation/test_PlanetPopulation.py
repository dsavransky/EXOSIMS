import unittest
import EXOSIMS
import EXOSIMS.Prototypes.PlanetPopulation 
import EXOSIMS.PlanetPopulation
import pkgutil
from EXOSIMS.util.get_module import get_module
import numpy as np
import os
from tests.TestSupport.Utilities import RedirectStreams

class TestPlanetPopulation(unittest.TestCase):
    """ 

    Global PlanetPopulation tests.
    Applied to all implementations, for overloaded methods only.

    Any implementation-specific methods, or to test specific new
    method functionality, separate tests are needed.

    """
    
    def setUp(self):

        self.dev_null = open(os.devnull, 'w')

        self.spec = {"modules":{"PlanetPhysicalModel" : "PlanetPhysicalModel"}}
    
        modtype = getattr(EXOSIMS.Prototypes.PlanetPopulation.PlanetPopulation,'_modtype')
        pkg = EXOSIMS.PlanetPopulation
        self.allmods = [get_module(modtype)]
        for loader, module_name, is_pkg in pkgutil.walk_packages(pkg.__path__, pkg.__name__+'.'):
            if not is_pkg:
                mod = get_module(module_name.split('.')[-1],modtype)
                self.assertTrue(mod._modtype is modtype,'_modtype mismatch for %s'%mod.__name__)
                self.allmods.append(mod)


    def test_honor_arange(self):
        """
        Tests that the input range for semi-major axis is properly set 
        and is used when generating sma samples.
        """

        exclude_setrange = ['EarthTwinHabZone1','EarthTwinHabZone2', 'JupiterTwin']
        exclude_checkrange = ['KeplerLike1']

        arangein = np.sort(np.random.rand(2)*10.0)
        
        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(arange = arangein,**self.spec)

            #test that the input arange is used
            if mod.__name__ not in exclude_setrange:
                self.assertTrue(obj.arange[0].value == arangein[0],'sma low bound set failed for %s'%mod.__name__)
                self.assertTrue(obj.arange[1].value == arangein[1],'sma high bound set failed for %s'%mod.__name__)


            #test that generated values honor range
            #ignore any cases where gen_plan_params is inherited
            if (mod.__name__ not in exclude_checkrange) and ('gen_plan_params' in mod.__dict__):
                x = 10000
                a, e, p, Rp = obj.gen_plan_params(x)
                self.assertEqual(len(a),x,'Incorrect number of samples generated for %s'%mod.__name__)
                self.assertTrue(np.all(a.value <= obj.arange[1].value),'sma high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(a.value >= obj.arange[0].value),'sma low bound failed for %s'%mod.__name__)


    def test_honor_erange(self):
        """
        Tests that the input range for eccentricity is properly set 
        and is used when generating e samples.
        """

        exclude_setrange = ['EarthTwinHabZone1']
        exclude_checkrange = [ ]

        tmp = np.random.rand(1)*0.5
        erangein = np.hstack((tmp,np.random.rand(1)*0.5+0.5))
        
        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(erange = erangein,**self.spec)

            #test that the input arange is used
            if mod.__name__ not in exclude_setrange:
                self.assertTrue(obj.erange[0] == erangein[0],'e low bound set failed for %s'%mod.__name__)
                self.assertTrue(obj.erange[1] == erangein[1],'e high bound set failed for %s'%mod.__name__)


            #test thatgenerated values honor arange
            #ignore any cases where gen_plan_params is inherited
            if (mod.__name__ not in exclude_checkrange) and ('gen_plan_params' in mod.__dict__):
                x = 10000
                a, e, p, Rp = obj.gen_plan_params(x)
                self.assertEqual(len(e),x,'Incorrect number of samples generated for %s'%mod.__name__)
                self.assertTrue(np.all(e <= obj.erange[1]),'e high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(e >= obj.erange[0]),'e low bound failed for %s'%mod.__name__)


    def test_honor_constrainOrbits(self):
        """
        Test that constrainOrbits is consistently applied

        Generated orbital radii must be within the rrange, which 
        is the original arange.
        """

        exclude_check = [ ]

        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(constrainOrbits=True,**self.spec)
            self.assertTrue(obj.constrainOrbits,'constrainOrbits not set for %s'%mod.__name__)
            self.assertTrue(np.all(obj.arange == obj.rrange),'arange and rrange do not match with constrainOrbits set for %s'%mod.__name__)

            #ignore any cases where gen_plan_params is inherited
            if (mod.__name__ not in exclude_check) and ('gen_plan_params' in mod.__dict__):
                x = 10000
                a, e, p, Rp = obj.gen_plan_params(x)
                self.assertTrue(np.all(a*(1+e) <= obj.rrange[1]),'constrainOribts high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(a*(1-e) >= obj.rrange[0]),'constrainOribts low bound failed for %s'%mod.__name__)


    def test_honor_prange(self):
        """
        Tests that the input range for albedois properly set 
        and is used when generating p samples.
        """

        exclude_setrange = ['EarthTwinHabZone1','EarthTwinHabZone2','JupiterTwin','AlbedoByRadius']
        exclude_checkrange = ['KeplerLike1','SAG13']

        tmp = np.random.rand(1)*0.5
        prangein = np.hstack((tmp,np.random.rand(1)*0.5+0.5))
        
        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(prange = prangein,**self.spec)

            #test that the input arange is used
            if mod.__name__ not in exclude_setrange:
                self.assertTrue(obj.prange[0] == prangein[0],'p low bound set failed for %s'%mod.__name__)
                self.assertTrue(obj.prange[1] == prangein[1],'p high bound set failed for %s'%mod.__name__)


            #test thatgenerated values honor arange
            #ignore any cases where gen_plan_params is inherited
            if (mod.__name__ not in exclude_checkrange) and ('gen_plan_params' in mod.__dict__):
                x = 10000
                a, e, p, Rp = obj.gen_plan_params(x)
                self.assertEqual(len(p),x,'Incorrect number of samples generated for %s'%mod.__name__)
                self.assertTrue(np.all(p <= obj.prange[1]),'p high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(p >= obj.prange[0]),'p low bound failed for %s'%mod.__name__)


    def test_honor_Rprange(self):
        """
        Tests that the input range for planet radius is properly set 
        and is used when generating radius samples.
        """

        exclude_setrange = ['EarthTwinHabZone1','EarthTwinHabZone2','JupiterTwin']
        exclude_checkrange = ['KeplerLike1']

        Rprangein = np.sort(np.random.rand(2)*10.0)
        
        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(Rprange = Rprangein,**self.spec)

            #test that the input arange is used
            if mod.__name__ not in exclude_setrange:
                self.assertTrue(obj.Rprange[0].value == Rprangein[0],'Rp low bound set failed for %s'%mod.__name__)
                self.assertTrue(obj.Rprange[1].value == Rprangein[1],'Rp high bound set failed for %s'%mod.__name__)


            #test that generated values honor range
            #ignore any cases where gen_plan_params is inherited
            if (mod.__name__ not in exclude_checkrange) and ('gen_plan_params' in mod.__dict__):
                x = 10000
                a, e, p, Rp = obj.gen_plan_params(x)
                self.assertEqual(len(Rp),x,'Incorrect number of samples generated for %s'%mod.__name__)
                self.assertTrue(np.all(Rp.value <= obj.Rprange[1].value),'Rp high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(Rp.value >= obj.Rprange[0].value),'Rp low bound failed for %s'%mod.__name__)

    def test_honor_Mprange(self):
        """
        Tests that the input range for planet mass is properly set 
        and is used when generating mass samples.
        """

        exclude_setrange = ['EarthTwinHabZone1','EarthTwinHabZone2','JupiterTwin']
        exclude_checkrange = ['KeplerLike1']

        Mprangein = np.sort(np.random.rand(2)*10.0)
        
        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(Mprange = Mprangein,**self.spec)

            #test that the input arange is used
            if mod.__name__ not in exclude_setrange:
                self.assertTrue(obj.Mprange[0].value == Mprangein[0],'Mp low bound set failed for %s'%mod.__name__)
                self.assertTrue(obj.Mprange[1].value == Mprangein[1],'Mp high bound set failed for %s'%mod.__name__)


            #test that generated values honor range
            #ignore any cases where gen_plan_params is inherited
            if (mod.__name__ not in exclude_checkrange) and ('gen_mass' in mod.__dict__):
                x = 10000
                Mp = obj.gen_mass(x)
                self.assertEqual(len(Mp),x,'Incorrect number of samples generated for %s'%mod.__name__)
                self.assertTrue(np.all(Mp.value <= obj.Mprange[1].value),'Mp high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(Mp.value >= obj.Mprange[0].value),'Mp low bound failed for %s'%mod.__name__)

    def test_honor_wrange(self):
        """
        Tests that the input range for arg or periapse is properly set 
        and is used when generating w samples.
        """

        exclude_setrange = [ ]
        exclude_checkrange = [ ]

        tmp = np.random.rand(1)*180
        wrangein = np.hstack((tmp,np.random.rand(1)*180+180))
        
        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(wrange = wrangein,**self.spec)

            #test that the input arange is used
            if mod.__name__ not in exclude_setrange:
                self.assertTrue(obj.wrange[0].value == wrangein[0],'w low bound set failed for %s'%mod.__name__)
                self.assertTrue(obj.wrange[1].value == wrangein[1],'w high bound set failed for %s'%mod.__name__)


            #test thatgenerated values honor arange
            #ignore any cases where gen_plan_params is inherited
            if (mod.__name__ not in exclude_checkrange) and ('gen_angles' in mod.__dict__):
                x = 10000
                I, O, w = obj.gen_angles(x)
                self.assertEqual(len(w),x,'Incorrect number of samples generated for %s'%mod.__name__)
                self.assertTrue(np.all(w <= obj.wrange[1]),'w high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(w >= obj.wrange[0]),'w low bound failed for %s'%mod.__name__)

    def test_honor_Irange(self):
        """
        Tests that the input range for inclination is properly set 
        and is used when generating I samples.
        """

        exclude_setrange = [ ]
        exclude_checkrange = [ ]

        tmp = np.random.rand(1)*90
        Irangein = np.hstack((tmp,np.random.rand(1)*90+90))
        
        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(Irange = Irangein,**self.spec)

            #test that the input arange is used
            if mod.__name__ not in exclude_setrange:
                self.assertTrue(obj.Irange[0].value == Irangein[0],'I low bound set failed for %s'%mod.__name__)
                self.assertTrue(obj.Irange[1].value == Irangein[1],'I high bound set failed for %s'%mod.__name__)


            #test thatgenerated values honor arange
            #ignore any cases where gen_plan_params is inherited
            if (mod.__name__ not in exclude_checkrange) and ('gen_angles' in mod.__dict__):
                x = 10000
                I, O, w = obj.gen_angles(x)
                self.assertEqual(len(I),x,'Incorrect number of samples generated for %s'%mod.__name__)
                self.assertTrue(np.all(I <= obj.Irange[1]),'I high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(I >= obj.Irange[0]),'I low bound failed for %s'%mod.__name__)


    def test_honor_Orange(self):
        """
        Tests that the input range for long. of ascending node is properly set 
        and is used when generating O samples.
        """

        exclude_setrange = [ ]
        exclude_checkrange = [ ]

        tmp = np.random.rand(1)*180
        Orangein = np.hstack((tmp,np.random.rand(1)*180+180))
        
        for mod in self.allmods:
            with RedirectStreams(stdout=self.dev_null):
                obj = mod(Orange = Orangein,**self.spec)

            #test that the input arange is used
            if mod.__name__ not in exclude_setrange:
                self.assertTrue(obj.Orange[0].value == Orangein[0],'O low bound set failed for %s'%mod.__name__)
                self.assertTrue(obj.Orange[1].value == Orangein[1],'O high bound set failed for %s'%mod.__name__)


            #test thatgenerated values honor arange
            #ignore any cases where gen_plan_params is inherited
            if (mod.__name__ not in exclude_checkrange) and ('gen_angles' in mod.__dict__):
                x = 10000
                I, O, w = obj.gen_angles(x)
                self.assertEqual(len(O),x,'Incorrect number of samples generated for %s'%mod.__name__)
                self.assertTrue(np.all(O <= obj.Orange[1]),'O high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(O >= obj.Orange[0]),'O low bound failed for %s'%mod.__name__)


    def test_dist_eccen_from_sma(self):
        """
        Test that eccentricities generating radii outside of arange
        have zero probability.
        """
        
        for mod in self.allmods:
            if 'dist_eccen_from_sma' in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    obj = mod(**self.spec)
                    
                x = 10000
                a, e, p, Rp = obj.gen_plan_params(x)

                f = obj.dist_eccen_from_sma(e,a)
                self.assertTrue(np.all(f[a*(1+e) > obj.rrange[1]] == 0),'dist_eccen_from_sma low bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(f[a*(1-e) < obj.rrange[0]] == 0),'dist_eccen_from_sma high bound failed for %s'%mod.__name__)

    def test_dist_sma(self):
        """
        Test that smas outside of the range have zero probability

        """

        for mod in self.allmods:
            if 'dist_sma' in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    pp = mod(**self.spec)

                a = np.logspace(np.log10(pp.arange[0].value/10.),np.log10(pp.arange[1].value*100.),100) 

                fa = pp.dist_sma(a)
                self.assertTrue(np.all(fa[a < pp.arange[0].value] == 0),'dist_sma high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(fa[a > pp.arange[1].value] == 0),'dist_sma low bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(fa[(a >= pp.arange[0].value) & (a <= pp.arange[1].value)] > 0),'dist_sma generates zero probabilities within range for %s'%mod.__name__)

    def test_dist_eccen(self):
        """
        Test that eccentricities outside of the range have zero probability

        """
        for mod in self.allmods:
            if 'dist_eccen' in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    pp = mod(**self.spec)

                e = np.linspace(pp.erange[0]-1,pp.erange[1]+1,100)

                fe = pp.dist_eccen(e)
                self.assertTrue(np.all(fe[e < pp.erange[0]] == 0),'dist_eccen high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(fe[e > pp.erange[1]] == 0),'dist_eccen low bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(fe[(e >= pp.erange[0]) & (e <= pp.erange[1])] > 0),'dist_eccen generates zero probabilities within range for %s'%mod.__name__)


    def test_dist_albedo(self):
        """
        Test that albedos outside of the range have zero probability

        """

        exclude_mods = ['KeplerLike1']
        for mod in self.allmods:
            if (mod.__name__ not in exclude_mods) and ('dist_albedo' in mod.__dict__):
                with RedirectStreams(stdout=self.dev_null):
                    pp = mod(**self.spec)

                p = np.linspace(pp.prange[0]-1,pp.prange[1]+1,100)

                fp = pp.dist_albedo(p)
                self.assertTrue(np.all(fp[p < pp.prange[0]] == 0),'dist_albedo high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(fp[p > pp.prange[1]] == 0),'dist_albedo low bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(fp[(p >= pp.prange[0]) & (p <= pp.prange[1])] > 0),'dist_albedo generates zero probabilities within range for %s'%mod.__name__)


    def test_dist_radius(self):
        """
        Test that radii outside of the range have zero probability

        """
        for mod in self.allmods:
            if 'dist_radius' in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    pp = mod(**self.spec)

                Rp = np.logspace(np.log10(pp.Rprange[0].value/10.),np.log10(pp.Rprange[1].value*100.),100) 

                fr = pp.dist_radius(Rp)
                self.assertTrue(np.all(fr[Rp < pp.Rprange[0].value] == 0),'dist_radius high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(fr[Rp > pp.Rprange[1].value] == 0),'dist_radius high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(fr[(Rp >= pp.Rprange[0].value) & (Rp <= pp.Rprange[1].value)] > 0),'dist_radius generates zero probabilities within range for %s'%mod.__name__)



    def test_dist_mass(self):
        """
        Test that masses outside of the range have zero probability

        """

        for mod in self.allmods:
            if 'dist_mass' in mod.__dict__:
                with RedirectStreams(stdout=self.dev_null):
                    pp = mod(**self.spec)

                Mp = np.logspace(np.log10(pp.Mprange[0].value/10.),np.log10(pp.Mprange[1].value*100.),100) 

                fr = pp.dist_mass(Mp)
                self.assertTrue(np.all(fr[Mp < pp.Mprange[0].value] == 0),'dist_mass high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(fr[Mp > pp.Mprange[1].value] == 0),'dist_mass high bound failed for %s'%mod.__name__)
                self.assertTrue(np.all(fr[(Mp >= pp.Mprange[0].value) & (Mp <= pp.Mprange[1].value)] > 0))


