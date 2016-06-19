import numpy as np
import astropy.units as u
import astropy.constants as const
from EXOSIMS.util.keplerSTM import planSys
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.eccanom import eccanom
import EXOSIMS.util.statsFun as statsFun 
import numbers

class SimulatedUniverse(object):
    """Simulated Universe class template
    
    This class contains all variables and functions necessary to perform 
    Simulated Universe Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs:
            user specified values
    
    Attributes:
        PlanetPopulation (object):
            PlanetPopulation class object
        PlanetPhysicalModel (object):
            PlanetPhysicalModel class object
        OpticalSystem (object):
            OpticalSystem class object
        ZodiacalLight (object):
            ZodiacalLight class object
        BackgroundSources (object):
            BackgroundSources class object
        PostProcessing (object):
            PostProcessing class object
        Completeness (object):
            Completeness class object
        TargetList (object):
            TargetList class object
        eta (float)
            Global occurrence rate defined as expected number of planets 
            per star in a given universe
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
        p (float ndarray):
            Planet albedo
        Rp (astropy Quantity array):
            Planet radius in units of km
        Mp (astropy Quantity array):
            Planet mass in units of kg
        r (astropy Quantity nx3 array):
            Planet position vector in units of km
        v (astropy Quantity nx3 array):
            Planet velocity vector in units of km/s
        s (astropy Quantity array):
            Planet-star apparent separations in units of km
        d (astropy Quantity array):
            Planet-star distances in units of km
        fEZ (astropy Quantity array):
            Surface brightness of exozodiacal light in units of 1/arcsec2
    
    """

    _modtype = 'SimulatedUniverse'
    _outspec = {}
    
    def __init__(self, eta=0.1, **specs):
       
        #check inputs
        assert isinstance(eta,numbers.Number) and (eta > 0),\
                "eta must be a positive number."
        
        #global occurrence rate defined as expected number of planets per 
        #star in a given universe
        self.eta = eta
        
        # import TargetList class
        self.TargetList = get_module(specs['modules']['TargetList'],'TargetList')(**specs)
        
        # bring inherited class objects to top level of Simulated Universe
        TL = self.TargetList
        self.OpticalSystem = TL.OpticalSystem 
        self.PlanetPopulation = TL.PlanetPopulation 
        self.PlanetPhysicalModel = TL.PlanetPhysicalModel
        self.ZodiacalLight = TL.ZodiacalLight 
        self.BackgroundSources = TL.BackgroundSources
        self.Completeness = TL.Completeness 
        self.PostProcessing = TL.PostProcessing
        
        self.gen_planetary_systems(**specs)

    def __str__(self):
        """String representation of Simulated Universe object
        
        When the command 'print' is used on the Simulated Universe object, 
        this method will return the values contained in the object"""
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Simulated Universe class object attributes'

    def gen_planetary_systems(self,**specs):
        """
        Generate the planetary systems for the current simulated universe.
        This routine populates arrays of the orbital elements and physical 
        characteristics of all planets, and generates indexes that map from 
        planet to parent star.
        """
        
        TL = self.TargetList
        PPop = self.PlanetPopulation
        
        # Map planets to target stars
        probs = np.random.uniform(size=TL.nStars)
        self.plan2star = np.where(probs > self.eta)[0]
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(self.plan2star)
        
        self.a = PPop.gen_sma(self.nPlans)                  # semi-major axis
        self.e = PPop.gen_eccen_from_sma(self.nPlans,self.a) if PPop.constrainOrbits \
                else PPop.gen_eccen(self.nPlans)            # eccentricity
        self.I = PPop.gen_I(self.nPlans)                    # inclination
        self.O = PPop.gen_O(self.nPlans)                    # longitude of ascending node
        self.w = PPop.gen_w(self.nPlans)                    # argument of periapsis
        self.Rp = PPop.gen_radius(self.nPlans)              # radius
        self.Mp = PPop.gen_mass(self.nPlans)                # mass
        self.p = PPop.gen_albedo(self.nPlans)               # albedo
        self.r, self.v = self.planet_pos_vel()              # initial position
        self.d = np.sqrt(np.sum(self.r**2, axis=1))         # planet-star distance
        self.s = np.sqrt(np.sum(self.r[:,0:2]**2, axis=1))  # apparent separation
        
        # exo-zodi levels for systems with planets
        self.fEZ = self.ZodiacalLight.fEZ(TL,self.plan2star,self.I)

    def planet_pos_vel(self,M=None):
        """Assigns each planet an initial position (km) and velocity (km/s)
        
        This method makes us of the planet orbital elements (a, e, I, O, w, E), 
        and the planet and star masses.

        Inputs:
            M (ndarray)
                Initial Mean anomaly in radians.  If None (default) will be randomly
                generated in U([0, 2\pi))
                
        Returns:
            r (astropy Quantity nx3 array):
                Initial position vector of each planet in units of km
            v (astropy Quantity nx3 array):
                Initial velocity vector of each planet in units of km/s
        
        """
        
        # planet orbital elements
        a = self.a                              # semi-major axis
        e = self.e                              # eccentricity
        I = self.I.to('rad').value              # inclinations
        Omega = self.O.to('rad').value          # right ascension of the ascending node
        w = self.w.to('rad').value              # argument of perigee

        if M is None:
            # generate random mean anomaly
            M = np.random.uniform(high=2*np.pi,size=self.nPlans) 
        
        E = eccanom(M,e)                        # eccentric anomaly
        # planet and star masses
        Mp = self.Mp
        Ms = self.TargetList.MsTrue[self.plan2star]
        
        
        a1 = (a*(np.cos(Omega)*np.cos(w) - np.sin(Omega)*np.cos(I)*np.sin(w))).to('AU')
        a2 = (a*(np.sin(Omega)*np.cos(w) + np.cos(Omega)*np.cos(I)*np.sin(w))).to('AU')
        a3 = (a*np.sin(I)*np.sin(w)).to('AU')
        A = np.vstack((a1,a2,a3)).T.value*u.AU
        
        b1 = (-a*np.sqrt(1.-e**2)*(np.cos(Omega)*np.sin(w) + np.sin(Omega)\
                *np.cos(I)*np.cos(w))).to('AU')
        b2 = (a*np.sqrt(1.-e**2)*(-np.sin(Omega)*np.sin(w) + np.cos(Omega)\
                *np.cos(I)*np.cos(w))).to('AU')
        b3 = (a*np.sqrt(1.-e**2)*np.sin(I)*np.cos(w)).to('AU')
        B = np.vstack((b1,b2,b3)).T.value*u.AU
        
        r1 = np.cos(E) - e
        r1 = np.array([r1]*3).T
        r2 = np.sin(E)
        r2 = np.array([r2]*3).T
        r = A*r1 + B*r2
        
        mu = const.G*(Mp + Ms*const.M_sun)
        v1 = (np.sqrt(mu/a**3)/(1. - e*np.cos(E))).to('/s').value
        v1 = np.hstack((v1.reshape(len(v1),1), v1.reshape(len(v1),1), v1.reshape(len(v1),1)))/u.s
        v2 = np.cos(E)
        v2 = np.hstack((v2.reshape(len(v2),1), v2.reshape(len(v2),1), v2.reshape(len(v2),1)))
        v = v1*(-A*r2 + B*v2)
        
        return r.to('km'), v.to('km/s')

    def prop_system(self, r, v, Mp, Ms, dt):
        """Propagates planet state vectors (position and velocity) 
        
        This method uses the Kepler state transition matrix to propagate a 
        planet's state (position and velocity vectors) forward in time using 
        the Kepler state transition matrix.
        
        Args:
            r (astropy Quantity nx3 array):
                Initial position vector of each planet in units of km
            v (astropy Quantity nx3 array):
                Initial velocity vector of each planet in units of km/s
            Mp (astropy Quantity array):
                Planet masses in units of kg
            Ms (float ndarray):
                Target star masses in M_sun
            dt (astropy Quantity):
                Time increment to propagate the system in units of day
        
        Returns:
            rnew (astropy Quantity nx3 array):
                Propagated position vectors in units of km
            vnew (astropy Quantity nx3 array):
                Propagated velocity vectors in units of km/s
            snew (astropy Quantity array):
                Propagated apparent separations in units of km
            dnew (astropy Quantity array):
                Propagated planet-star distances in units of km
        
        """
        
        # stack dimensionless positions and velocities
        x0 = np.array([])
        for i in xrange(r.shape[0]):
            x0 = np.hstack((x0, r[i].to('km').value, v[i].to('km/day').value))
                
        # calculate vector of gravitational parameter
        mu = (const.G*(Mp + Ms*const.M_sun)).to('km3/day2').value
        if np.isscalar(mu):
            mu = np.array([mu])
        # use keplerSTM.py to propagate the system
        prop = planSys(x0, mu)
        try:
            prop.takeStep(dt.to('day').value)
        except ValueError:
            raise ValueError('planSys error')
            
        # split off position and velocity vectors
        x1 = np.array(np.hsplit(prop.x0, 2*len(r)))
        rind = np.array(range(0,len(x1),2)) # even indices
        vind = np.array(range(1,len(x1),2)) # odd indices
        
        # assign new position, velocity, apparent separation, and planet-star distance
        rnew = x1[rind]*u.km
        vnew = x1[vind]*u.km/u.day
        snew = np.sqrt(np.sum(rnew[:,0:2]**2, axis=1))
        dnew = np.sqrt(np.sum(rnew**2, axis=1))
        
        return rnew.to('km'), vnew.to('km/s'), snew.to('km'), dnew.to('km')

    def get_current_WA(self,pInds):
        """Calculate the current working angles for planets specified by the 
        given indices.
        
        Args:
            pInds (integer ndarray):
                Integer indices of the planets of interest
        
        Returns:
            WA (astropy Quantity array):
                Working angles in units of arcsec
        
        """
        
        starDists = self.TargetList.dist[self.plan2star[pInds]] # distance to star
        WA = np.arctan(self.s[pInds]/starDists).to('arcsec')
        
        return WA

