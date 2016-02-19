# -*- coding: utf-8 -*-
import numpy as np
import astropy.units as u
import astropy.constants as const
from EXOSIMS.util.keplerSTM import planSys
from EXOSIMS.util.get_module import get_module
import EXOSIMS.util.statsFun as statsFun 

class SimulatedUniverse(object):
    """Simulated Universe class template
    
    This class contains all variables and functions necessary to perform 
    Simulated Universe Module calculations in exoplanet mission simulation.
    
    It inherits the following class objects which are defined in __init__:
    TargetList, PlanetPhysicalModel
    
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
        w (ndarray):
            1D numpy ndarray containing argument of perigee in degrees
        O (ndarray):
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
        I (ndarray):
            1D numpy ndarray containing inclination in degrees for each planet            
        p (ndarray):
            1D numpy ndarray containing albedo for each planet        
        rtype (ndarray):
            1D numpy ndarray containing rock/ice fraction for each planet
        fzodicurr (ndarray):
            1D numpy ndarray containing exozodi level for each planet
    
    """

    _modtype = 'SimulatedUniverse'
    _outspec = {}
    
    def __init__(self, **specs):
        
        # get desired module names (prototype or specific)
        
        # import TargetList class
        TL = get_module(specs['modules']['TargetList'], 'TargetList')
        self.TargetList = TL(**specs)
        
        # import PlanetPhysicalModel class
        PlanPhys = get_module(specs['modules']['PlanetPhysicalModel'], 'PlanetPhysicalModel')
        self.PlanetPhysicalModel = PlanPhys(**specs)
        
        # bring inherited class objects to top level of Simulated Universe
        # optical system class object
        self.OpticalSystem = self.TargetList.OpticalSystem 
        # planet population class object
        self.PlanetPopulation = self.TargetList.PlanetPopulation 
        # zodiacal light class object
        self.ZodiacalLight = self.TargetList.ZodiacalLight 
        # background sources
        self.BackgroundSources = self.TargetList.BackgroundSources
        # completeness class object
        self.Completeness = self.TargetList.Completeness 
        # postprocessing object
        self.PostProcessing = self.TargetList.PostProcessing
        
        # planets mapped to target stars
        self.planInds = self.planet_to_star()
        # number of planets
        self.nPlans = len(self.planInds) 
        # indices of target stars with planets
        self.sysInds = np.unique(self.planInds)
        # planet semi-major axis
        self.a = self.planet_a()
        # planet eccentricities
        self.e = self.planet_e()
        # planet argument of perigee
        self.w = self.planet_w()
        # planet right ascension of the ascending node
        self.O = self.planet_O()
        # planet radii
        self.Rp = self.planet_radii() 
        # planet masses
        self.Mp = self.planet_masses()
        # planet albedos
        self.p = self.planet_albedos() 
        # inclination in degrees of each planet
        self.I = self.planet_inclinations()
        # planet initial positions
        self.r, self.v = self.planet_pos_vel() 
        # exozodi levels for systems with planets
        self.fzodicurr = self.ZodiacalLight.fzodi(self.planInds, self.I, self.TargetList)

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
        
        This method defines the data type expected, specific SimulatedUniverse 
        classes will populate these indices.
        
        This method uses the following inherited class objects:
            self.TargetList:
                TargetList class object
            self.PlanetPopulation:
                PlanetPopulation class object
                
        Returns:
            planSys (ndarray):
                1D numpy array containing indices of the target star to which 
                each planet (each element of the array) belongs
        
        """
        
        # assign between 0 and 8 planets to each star in the target list
        planSys = np.array([],dtype=int)
            
        for i in range(self.TargetList.nStars):
            nump = np.random.randint(0, high=8)
            planSys = np.hstack((planSys, np.array([i]*nump,dtype=int)))
        
        return planSys
    
    def planet_a(self):
        """Assigns each planet semi-major axis in AU
        
        This method defines the data type expected, specific SimulatedUniverse 
        classes will populate these values.
        
        This method uses the following inherited class objects:
            self.PlanetPopulation:
                PlanetPopulation class object
        
        Returns:
            a (Quantity):
                1D numpy ndarray containing semi-major axis for each planet 
                (default units of AU)
        
        """
        
        # assign planets a semi-major axis 
        a = statsFun.simpSample(self.PlanetPopulation.semi_axis, self.nPlans, self.PlanetPopulation.arange[0].value, self.PlanetPopulation.arange[1].value)*self.PlanetPopulation.arange.unit
        
        return a
        
    def planet_e(self):
        """Assigns each planet eccentricity
        
        This method defines the data type expected, specific SimulatedUniverse 
        classes will populate these values.
        
        This method uses the following inherited class object:
            self.PlanetPopulation:
                PlanetPopulation class object
        
        Returns:
            e (ndarray):
                1D numpy ndarray containing eccentricity for each planet 
        
        """
        
        # assign planets an eccentricity 
        e = statsFun.simpSample(self.PlanetPopulation.eccentricity, self.nPlans, self.PlanetPopulation.erange[0], self.PlanetPopulation.erange[1])
        
        return e
        
    def planet_w(self):
        """Assigns each planet argument of perigee
        
        This method defines the data type expected, specific SimulatedUniverse 
        classes will populate these values.
        
        This method uses the following inherited class object:
            self.PlanetPopulation:
                PlanetPopulation class object
        
        Returns:
            w (ndarray):
                1D numpy ndarray containing argument of perigee for each planet 
                in degrees
        
        """
        
        # assign planets an argument of perigee 
        w = statsFun.simpSample(self.PlanetPopulation.arg_perigee, self.nPlans, self.PlanetPopulation.wrange[0].value, self.PlanetPopulation.wrange[1].value)
                
        return w
        
    def planet_O(self):
        """Assigns each planet right ascension of the ascending node
        
        This method defines the data type expected, specific SimulatedUniverse 
        classes will populate these values.
        
        This method uses the following inherited class object:
            self.PlanetPopulation:
                PlanetPopulation class object
        
        Returns:
            O (ndarray):
                1D numpy ndarray containing right ascension of the ascending node
                for each planet in degrees
        
        """
        
        # assign planets right ascension of the ascending node 
        O = statsFun.simpSample(self.PlanetPopulation.RAAN, self.nPlans, self.PlanetPopulation.Orange[0].value, self.PlanetPopulation.Orange[1].value)
                
        return O
        
    def planet_radii(self):
        """Assigns each planet a radius in km
        
        This defines the data type expected, specific SimulatedUniverse class
        objects will populate these values.
        
        This method uses the following inherited class object:
            self.PlanetPopulation:
                PlanetPopulation class object
        
        Returns:
            R (Quantity):
                1D numpy ndarray containing radius of each planet (units of 
                distance)
        
        """
        
        # assign planets a radius 
        R = statsFun.simpSample(self.PlanetPopulation.radius, self.nPlans, self.PlanetPopulation.Rrange[0].value, self.PlanetPopulation.Rrange[1].value)*self.PlanetPopulation.Rrange.unit
        
        return R
            
    def planet_masses(self):
        """Assigns each planet mass in kg
        
        This method defines the data type expected, specific SimulatedUniverse 
        classes will populate these values.
        
        This method uses the following inherited class object:
            self.PlanetPopulation:
                PlanetPopulation class object
        
        Returns:
            M (Quantity):
                1D numpy ndarray containing mass for each planet (units of kg)
        
        """
        
        # assign planets a mass 
        M = statsFun.simpSample(self.PlanetPopulation.mass, self.nPlans, self.PlanetPopulation.Mprange[0].value, self.PlanetPopulation.Mprange[1].value)*self.PlanetPopulation.Mprange.unit
        
        return M
        
    def planet_albedos(self):
        """Assigns each planet albedo 
        
        This method defines the data type expected, specific SimulatedUniverse 
        classes will populate these values.

        This method uses the following inherited class object:
            self.PlanetPopulation:
                PlanetPopulation class object
        
        Returns:
            p (ndarray):
                1D numpy ndarray containing albedo for each planet
        
        """
        
        # assign planets an albedo uniformly distributed between min and max
        # values from PlanetPopulation class object
        p = statsFun.simpSample(self.PlanetPopulation.albedo, self.nPlans, self.PlanetPopulation.prange[0], self.PlanetPopulation.prange[1])
        
        return p
        
    def planet_inclinations(self):
        """Assigns each planet an inclination in degrees 
        
        This method uses the following inherited class object:
            self.PlanetPopulation:
                PlanetPopulation class object
                
        Returns:
            I (ndarray):
                1D numpy ndarray containing inclination of each planet in 
                degrees
        
        """
        
        I = statsFun.simpSample(self.PlanetPopulation.inclination, self.nPlans, self.PlanetPopulation.Irange[0].value, self.PlanetPopulation.Irange[1].value)
        
        return I
        
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
        
        Omega = np.radians(self.O)
        omega = np.radians(self.w)
        I = np.radians(self.I)
        a = self.a
        e = self.e
        # find eccentric anomaly in radians
        E = np.array([])
        for i in xrange(len(self.e)):
            Enew = self.eccanom(2.*np.pi*np.random.rand(), self.e[i])
            E = np.hstack((E,Enew))
            
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
        
    def eccanom(self, M, e):
        """Finds eccentric anomaly from mean anomaly and eccentricity
        
        This method uses algorithm 2 from Vallado to find the eccentric anomaly
        from mean anomaly and eccentricity.
        
        Args:
            M (float):
                mean anomaly
            e (float):
                eccentricity
                
        Returns:
            E (float):
                eccentric anomaly
        
        """

        pi = np.pi
        # initial guess        
        if (-pi < M and M < 0) or (M > pi):
            E = M - e
        else:
            E = M + e
        # Newton-Raphson setup
        i = 0
        err = 1.
        tol = np.finfo(float).eps*4.1
        maxi = 200
        # Newton-Raphson iteration
        while err > tol and i < maxi:
            Enew = E + (M - E + e*np.sin(E))/(1. - e*np.cos(E))
            err = abs(Enew - E)
            E = Enew
            i += 1
                  
        return E


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
