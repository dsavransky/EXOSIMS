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
        planInds (ndarray):
            1D numpy ndarray of indices mapping planets to target stars in
            TargetList
        nPlans (int):
            total number of planets
        sysInds (ndarray):
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
        TL = get_module(specs['modules']['TargetList'], 'TargetList')
        self.TargetList = TL(**specs)
        
        # bring inherited class objects to top level of Simulated Universe
        # optical system class object
        self.OpticalSystem = self.TargetList.OpticalSystem 
        # planet population class object
        self.PlanetPopulation = self.TargetList.PlanetPopulation 
        # planet physical model class object
        self.PlanetPhysicalModel = self.TargetList.PlanetPhysicalModel
        # zodiacal light class object
        self.ZodiacalLight = self.TargetList.ZodiacalLight 
        # background sources
        self.BackgroundSources = self.TargetList.BackgroundSources
        # completeness class object
        self.Completeness = self.TargetList.Completeness 
        # postprocessing object
        self.PostProcessing = self.TargetList.PostProcessing
    
        self.gen_planetary_systems(**specs)

    def gen_planetary_systems(self,**specs):
        """
        Generate the planetary systems for the current simulated universe.
        This routine populates arrays of the orbital elements and physical 
        characteristics of all planets, and generates indexes that map from 
        planet to parent star.
        """

        # Map planets to target stars
        self.planet_to_star()                   # generate index of target star for each planet
        self.nPlans = len(self.planInds)        # number of planets in universe

        # planet semi-major axis
        self.a = self.PlanetPopulation.gen_sma(self.nPlans)
        # planet eccentricities
        self.e = self.PlanetPopulation.gen_eccentricity(self.nPlans)
        # planet argument of periapse
        self.w = self.PlanetPopulation.gen_w(self.nPlans)   
        # planet longitude of ascending node
        self.O = self.PlanetPopulation.gen_O(self.nPlans)
        # planet inclination
        self.I = self.PlanetPopulation.gen_I(self.nPlans)
        # planet radii
        self.Rp = self.PlanetPopulation.gen_radius(self.nPlans)
        # planet masses
        self.Mp = self.PlanetPopulation.gen_mass(self.nPlans)
        # planet albedos
        self.p = self.PlanetPopulation.gen_albedo(self.nPlans)
        # planet initial positions
        self.r, self.v = self.planet_pos_vel() 
        # exo-zodi levels for systems with planets
        self.fEZ = self.ZodiacalLight.fEZ(self.TargetList,self.planInds,self.I)

    def __str__(self):
        """String representation of Simulated Universe object
        
        When the command 'print' is used on the Simulated Universe object, 
        this method will return the values contained in the object"""

        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Simulated Universe class object attributes'
        
    def planet_to_star(self):
        """Assigns index of star in target star list to each planet
        
        The prototype implementation uses the global occurrence rate as the 
        probability of each target star having one planet (thus limiting the 
        universe to single planet systems).

        Attributes updated:
            planInds (ndarray):
                1D numpy array containing indices of the target star to which 
                each planet (each element of the array) belongs
            sysInds (ndarray):
                1D numpy array of indices of the subset of the targetlist with
                planets
        
        """
        
        probs = np.random.uniform(size=self.TargetList.nStars)
        self.planInds = np.where(probs > self.eta)[0]
        self.sysInds = np.unique(self.planInds) 
        
        return 
    
        
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

        a1 = (a*(np.cos(Omega)*np.cos(omega) - np.sin(Omega)*np.cos(I)*np.sin(omega))).to(u.AU).value
        a2 = (a*(np.sin(Omega)*np.cos(omega) + np.cos(Omega)*np.cos(I)*np.sin(omega))).to(u.AU).value
        a3 = (a*np.sin(I)*np.sin(omega)).to(u.AU).value
        
        A = np.hstack((a1.reshape(len(a1),1), a2.reshape(len(a2),1), a3.reshape(len(a3),1)))*u.AU
        
        b1 = (-a*np.sqrt(1.-e**2)*(np.cos(Omega)*np.sin(omega) + np.sin(Omega)*np.cos(I)*np.cos(omega))).to(u.AU).value
        b2 = (a*np.sqrt(1.-e**2)*(-np.sin(Omega)*np.sin(omega) + np.cos(Omega)*np.cos(I)*np.cos(omega))).to(u.AU).value
        b3 = (a*np.sqrt(1.-e**2)*np.sin(I)*np.cos(omega)).to(u.AU).value
        
        B = np.hstack((b1.reshape(len(b1),1), b2.reshape(len(b2),1), b3.reshape(len(b3),1)))*u.AU
        
        Mu = const.G*(self.Mp+self.TargetList.MsTrue[self.planInds]*const.M_sun)
        
        r1 = np.cos(E) - e
        r1 = np.hstack((r1.reshape(len(r1),1), r1.reshape(len(r1),1), r1.reshape(len(r1),1)))
        
        r2 = np.sin(E)
        r2 = np.hstack((r2.reshape(len(r2),1), r2.reshape(len(r2),1), r2.reshape(len(r2),1)))
        
        r = A*r1 + B*r2
        
        v1 = (np.sqrt(Mu/a**3)/(1. - e*np.cos(E))).to(1/u.s).value
        v1 = np.hstack((v1.reshape(len(v1),1), v1.reshape(len(v1),1), v1.reshape(len(v1),1)))*(1/u.s)

        v2 = np.cos(E)
        v2 = np.hstack((v2.reshape(len(v2),1), v2.reshape(len(v2),1), v2.reshape(len(v2),1)))
        
        v = v1*(-A*r2 + B*v2)
        
        return r.to(u.km), v.to(u.km/u.s)
        
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
        if r.size == 3:
            x0 = np.hstack((x0, r.to(u.km).value, v.to(u.km/u.day).value))
        else:
            for i in xrange(r.shape[0]):
                x0 = np.hstack((x0, r[i].to(u.km).value, v[i].to(u.km/u.day).value))
                
        # calculate vector of gravitational parameter    
        mu = (const.G*(Mp + Ms*const.M_sun)).to(u.km**3/u.day**2).value
        if np.isscalar(mu):
            mu = np.array([mu])
        # use keplerSTM.py to propagate the system
        prop = planSys(x0, mu)
        try:
            prop.takeStep(dt.to(u.day).value)
        except ValueError:
            raise ValueError('planSys error')
            
        # split off position and velocity vectors
        x1 = np.array(np.hsplit(prop.x0, 2*len(r)))
        rind, vind = [], []
        for x in xrange(len(x1)):
            if x%2 == 0:
                rind.append(x)
            else:
                vind.append(x)
        rind, vind = np.array(rind), np.array(vind)
        
        # assign new position and velocity arrays
        rnew = x1[rind]*u.km
        vnew = x1[vind]*u.km/u.day
        
        return rnew, vnew
        
    def get_current_WA(self, Inds):
        """Calculate the current working angles for planets specified by the 
        given indices.

        Args:
            Inds (integer ndarray):
                Numpy ndarray containing integer indices of the planets of interest
        
        Returns:
            wa (Quantity):
                numpy ndarray of working angles (units of arcsecons)        
        """

        #calculate s values
        rs = self.r[Inds.astype(int)]
        ss = rs.to('AU').value[:,0:2]
        ss = np.sqrt(np.sum(ss**2.,1)) # projected separation in AU
        
        #calculate distance to star
        starDists = self.TargetList.dist[self.planInds[Inds]]
		
        wa = ss/starDists*u.arcsec

        return wa
