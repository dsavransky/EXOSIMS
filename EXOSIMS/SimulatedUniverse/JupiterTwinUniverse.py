from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.keplerSTM import planSys
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.eccanom import eccanom
from EXOSIMS.util.deltaMag import deltaMag
import numpy as np
import astropy.units as u
import astropy.constants as const
from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse

class JupiterTwinUniverse(SimulatedUniverse):
    """Simulated Universe class template
    
    This class contains all variables and functions necessary to perform 
    Simulated Universe Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs:
            user specified values
    
    Attributes:
        StarCatalog (StarCatalog module):
            StarCatalog class object (only retained if keepStarCatalog is True)
        PlanetPopulation (PlanetPopulation module):
            PlanetPopulation class object
        PlanetPhysicalModel (PlanetPhysicalModel module):
            PlanetPhysicalModel class object
        OpticalSystem (OpticalSystem module):
            OpticalSystem class object
        ZodiacalLight (ZodiacalLight module):
            ZodiacalLight class object
        BackgroundSources (BackgroundSources module):
            BackgroundSources class object
        PostProcessing (BackgroundSources module):
            PostProcessing class object
        Completeness (Completeness module):
            Completeness class object
        TargetList (TargetList module):
            TargetList class object
        nPlans (integer):
            Total number of planets
        plan2star (integer ndarray):
            Indices mapping planets to target stars in TargetList
        sInds (integer ndarray):
            Unique indices of stars with planets in TargetList
        a (astropy Quantity array):
            Planet semi-major axis in units of AU
        e (float ndarray):
            Planet eccentricity
        I (astropy Quantity array):
            Planet inclination in units of deg
        O (astropy Quantity array):
            Planet right ascension of the ascending node in units of deg
        w (astropy Quantity array):
            Planet argument of perigee in units of deg
        M0 (astropy Quantity array):
            Initial mean anomaly in units of deg
        p (float ndarray):
            Planet albedo
        Rp (astropy Quantity array):
            Planet radius in units of km
        Mp (astropy Quantity array):
            Planet mass in units of kg
        r (astropy Quantity nx3 array):
            Planet position vector in units of AU
        v (astropy Quantity nx3 array):
            Planet velocity vector in units of AU/day
        d (astropy Quantity array):
            Planet-star distances in units of AU
        s (astropy Quantity array):
            Planet-star apparent separations in units of AU
        phi (float ndarray):
            Planet phase function, given its phase angle
        fEZ (astropy Quantity array):
            Surface brightness of exozodiacal light in units of 1/arcsec2
        dMag (float ndarray):
            Differences in magnitude between planets and their host star
        WA (astropy Quantity array)
            Working angles of the planets of interest in units of arcsec
    
    Notes:
        PlanetPopulation.eta is treated as the rate parameter of a Poisson distribution.
        Each target's number of planets is a Poisson random variable sampled with \lambda=\eta.
    
    """

    #_modtype = 'SimulatedUniverse'
    #_outspec = {}
    
    def __init__(self, **specs):

        SimulatedUniverse.__init__(self, **specs)


        # # load the vprint function (same line in all prototype module constructors)
        # self.vprint = vprint(specs.get('verbose', True))
       
        # # import TargetList class
        # self.TargetList = get_module(specs['modules']['TargetList'],
        #         'TargetList')(**specs)
        
        # # bring inherited class objects to top level of Simulated Universe
        # TL = self.TargetList
        # self.StarCatalog = TL.StarCatalog
        # self.PlanetPopulation = TL.PlanetPopulation
        # self.PlanetPhysicalModel = TL.PlanetPhysicalModel
        # self.OpticalSystem = TL.OpticalSystem
        # self.ZodiacalLight = TL.ZodiacalLight
        # self.BackgroundSources = TL.BackgroundSources
        # self.PostProcessing = TL.PostProcessing
        # self.Completeness = TL.Completeness
        
        # # list of possible planet attributes
        # self.planet_atts = ['plan2star', 'a', 'e', 'I', 'O', 'w', 'M0', 'Rp', 'Mp', 'p',
        #         'r', 'v', 'd', 's', 'phi', 'fEZ', 'dMag', 'WA']
        
        # # generate orbital elements, albedos, radii, and masses
        # self.gen_physical_properties(**specs)
        
        # # find initial position-related parameters: position, velocity, planet-star 
        # # distance, apparent separation, surface brightness of exo-zodiacal light
        # self.init_systems()

    # def __str__(self):
    #     """String representation of Simulated Universe object
        
    #     When the command 'print' is used on the Simulated Universe object, 
    #     this method will return the values contained in the object
        
    #     """
        
    #     for att in self.__dict__.keys():
    #         print('%s: %r' % (att, getattr(self, att)))
        
    #     return 'Simulated Universe class object attributes'

    def gen_physical_properties(self, **specs):
        """Generates the planetary systems' physical properties. 
        
        Populates arrays of the orbital elements, albedos, masses and radii 
        of all planets, and generates indices that map from planet to parent star.
        
        """
        
        PPop = self.PlanetPopulation
        TL = self.TargetList
        
        # treat eta as the rate parameter of a Poisson distribution
        #targetSystems = np.random.poisson(lam=PPop.eta, size=TL.nStars)
        plan2star = []
        #for j,n in enumerate(targetSystems):
        #    plan2star = np.hstack((plan2star, [j]*n))
        plan2star = np.arange(0,TL.nStars)
        #print(saltyburrito)
        self.plan2star = plan2star.astype(int)
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(self.plan2star)
        
        # sample all of the orbital and physical parameters
        self.I, self.O, self.w = PPop.gen_angles(self.nPlans)
        self.a, self.e, self.p, self.Rp = PPop.gen_plan_params(self.nPlans)
        if PPop.scaleOrbits:
            self.a *= np.sqrt(TL.L[self.plan2star])
        self.M0 = np.random.uniform(360, size=self.nPlans)*u.deg # initial mean anomaly
        self.Mp = PPop.gen_mass(self.nPlans)                     # mass
        
        # The prototype StarCatalog module is made of one single G star at 1pc. 
        # In that case, the SimulatedUniverse prototype generates one Jupiter 
        # at 5 AU to allow for characterization testing.
        # Also generates at least one Jupiter if no planet was generated.
        if TL.Name[0] == 'Prototype' or self.nPlans == 0:
            self.plan2star = np.array([0], dtype=int)
            self.sInds = np.unique(self.plan2star)
            self.nPlans = len(self.plan2star)
            self.a = np.array([5.])*u.AU
            self.e = np.array([0.])
            self.I = np.array([0.])*u.deg # face-on
            self.O = np.array([0.])*u.deg
            self.w = np.array([0.])*u.deg
            self.M0 = np.array([0.])*u.deg
            self.Rp = np.array([10.])*u.earthRad
            self.Mp = np.array([300.])*u.earthMass
            self.p = np.array([0.6])
