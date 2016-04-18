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
        TargetList (TargetList):
            TargetList class object
        PlanetPhysicalModel (PlanetPhysicalModel):
            PlanetPhysicalModel class object
        OpticalSystem (OpticalSystem):
            OpticalSystem class object
        PlanetPopulation (PlanetPopulation):
            PlanetPopulation class object
        ZodiacalLight (ZodiacalLight):
            ZodiacalLight class object
        BackgroundSources (BackgroundSources):
            BackgroundSources class object
        PostProcessing (PostProcessing):
            PostProcessing class object
        Completeness (Completeness):
            Completeness class object
        nPlans (int):
            total number of planets
        plan2star (ndarray):
            1D numpy ndarray of indices mapping planets to target stars in
            TargetList
        sInds (ndarray):
            indicies of target stars in TargetList with planets
        a (Quantity):
            1D numpy ndarray containing semi-major axis for each planet 
            (default units of AU)
        e (ndarray):
            1D numpy ndarray containing eccentricity for each planet
        w (Quantity):
            1D numpy ndarray containing argument of perigee in degrees
        O (Quantity):
            1D numpy ndarray containing right ascension of the ascending node 
            in degrees
        Mp (Quantity):
            1D numpy ndarray containing mass for each planet (default units of 
            kg)
        Rp (Quantity):
            1D numpy ndarray containing radius for each planet (default units 
            of km)
        r (Quantity):
            numpy ndarray containing position vector for each planet (default
            units of km)
        v (Quantity):
            numpy ndarray containing velocity vector for each planet (default 
            units of km/s)
        I (Quantity):
            1D numpy ndarray containing inclination in degrees for each planet
        p (ndarray):
            1D numpy ndarray containing albedo for each planet
        fEZ (ndarray):
            1D numpy ndarray containing exozodi level for each planet
    
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

        atts = self.__dict__.keys()
        
        for att in atts:
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
        probs = np.random.uniform(TL.nStars)
        self.plan2star = np.where(probs > self.eta)[0]
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(self.plan2star)

        self.a = PPop.gen_sma(self.nPlans)                  # semi-major axis
        self.e = PPop.gen_eccentricity_from_sma(self.nPlans,self.a) if PPop.constrainOrbits \
                else PPop.gen_eccentricity(self.nPlans)     # eccentricity
        self.w = PPop.gen_w(self.nPlans)                    # argument of periapsis
        self.O = PPop.gen_O(self.nPlans)                    # longitude of ascending node
        self.I = PPop.gen_I(self.nPlans)                    # inclination
        self.Rp = PPop.gen_radius(self.nPlans)              # radius
        self.Mp = PPop.gen_mass(self.nPlans)                # mass
        self.p = PPop.gen_albedo(self.nPlans)               # albedo
        self.r, self.v = self.planet_pos_vel()              # initial position
        self.d = np.sqrt(np.sum(self.r**2, axis=1))         # planet-star distance
        self.s = np.sqrt(np.sum(self.r[:,0:2]**2, axis=1))  # apparent separation

        # exo-zodi levels for systems with planets
        self.fEZ = self.ZodiacalLight.fEZ(self.TargetList,self.plan2star,self.I)


    def planet_pos_vel(self):
        """Assigns each planet an initial position (km) and velocity (km/s)
        
        This defines the data type expected, specific SimulatedUniverse class
        objects will populate these values.
        
        This method has access to the following:
            self.a:
                planet semi-major axis
            self.e:
                planet eccentricity
            self.Mp:
                planet masses
            self.I:
                planet inclinations in degrees
            self.w:
                planet argument of perigee in degrees
            self.O:
                planet right ascension of the ascending node in degrees
                
        Returns:
            r, v (Quantity, Quantity):
                numpy ndarray containing initial position vector of each planet 
                (units of km), numpy ndarray containing initial velocity vector 
                of each planet (units of km/s)
                
        """
        
        Omega = self.O.to('rad').value
        omega = self.w.to('rad').value
        I = self.I.to('rad').value
        a = self.a
        e = self.e

        #generate random mean anomlay and calculate eccentric anomaly
        M = np.random.uniform(high=2.*np.pi,size=self.nPlans)
        E = eccanom(M,e)

        a1 = (a*(np.cos(Omega)*np.cos(omega) - np.sin(Omega)*np.cos(I)*np.sin(omega))).to('AU')
        a2 = (a*(np.sin(Omega)*np.cos(omega) + np.cos(Omega)*np.cos(I)*np.sin(omega))).to('AU')
        a3 = (a*np.sin(I)*np.sin(omega)).to('AU')
        A = np.vstack((a1,a2,a3)).T.value*u.AU
        
        b1 = (-a*np.sqrt(1.-e**2)*(np.cos(Omega)*np.sin(omega) + np.sin(Omega)\
                *np.cos(I)*np.cos(omega))).to('AU')
        b2 = (a*np.sqrt(1.-e**2)*(-np.sin(Omega)*np.sin(omega) + np.cos(Omega)\
                *np.cos(I)*np.cos(omega))).to('AU')
        b3 = (a*np.sqrt(1.-e**2)*np.sin(I)*np.cos(omega)).to('AU')
        B = np.vstack((b1,b2,b3)).T.value*u.AU
        
        Mu = const.G*(self.Mp+self.TargetList.MsTrue[self.plan2star]*const.M_sun)
        
        r1 = np.cos(E) - e
        r1 = np.hstack((r1.reshape(len(r1),1), r1.reshape(len(r1),1), r1.reshape(len(r1),1)))
        r2 = np.sin(E)
        r2 = np.hstack((r2.reshape(len(r2),1), r2.reshape(len(r2),1), r2.reshape(len(r2),1)))
        r = A*r1 + B*r2
        
        v1 = (np.sqrt(Mu/a**3)/(1. - e*np.cos(E))).to('/s').value
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
            r (Quantity):
                numpy ndarray containing planet position vectors relative to 
                host stars (units of distance)
            v (Quantity): 
                numpy ndarray containing planet velocity vectors relative to 
                host stars (units of velocity)
            Mp (Quantity):
                1D numpy ndarray containing planet masses (units of mass)
            Ms (ndarray):
                1D numpy ndarray containing target star mass (in M_sun)
            dt (Quantity):
                time increment to propagate the system (units of time)
        
        Returns:
            rnew, vnew (Quantity, Quantity):
                numpy ndarray of propagated position vectors (units of km),
                numpy ndarray of propagated velocity vectors (units of km/day)
        
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
                Numpy ndarray containing integer indices of the planets of interest
        
        Returns:
            wa (Quantity):
                numpy ndarray of working angles (units of arcsecons)
        """

        starDists = self.TargetList.dist[self.plan2star[pInds]] # distance to star
        wa = (self.s[pInds]/starDists*u.rad).to('arcsec')

        return wa

