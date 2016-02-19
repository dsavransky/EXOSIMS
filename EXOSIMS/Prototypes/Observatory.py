# -*- coding: utf-8 -*-
import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.time import Time
import os,inspect

class Observatory(object):
    """Observatory class template
    
    This class contains all variables and methods necessary to perform
    Observatory Definition Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs: 
            user specified values
    
    Attributes:
        settlingTime (Quantity): 
            instrument settling time after repoint (default with units of day)
        thrust (Quantity): 
            occulter slew thrust (default with units of mN)
        slewIsp (Quantity): 
            occulter slew specific impulse (default with units of s)
        scMass (Quantity): 
            occulter (maneuvering sc) wet mass (default with units of kg)
        dryMass (Quantity): 
            occulter (maneuvering sc) dry mass (default with units of kg)
        coMass (Quantity): 
            telescope (or non-maneuvering sc) mass (default with units of kg)
        occulterSep (Quantity): 
            occulter-telescope distance (default with units of km)
        skIsp (Quantity): 
            station-keeping specific impulse (default with units of s)
        kogood (ndarray): 
            1D numpy ndarray of booleans where True is observable target star 
            in the target list
        r_sc (Quantity): 
            1D numpy ndarray of observatory postion vector (default with units 
            of km)
        currentSep (Quantity): 
            current occulter separation (default with units of km)
        flowRate (Quantity): 
            slew flow rate (default with units of kg/day)
    
    """

    _modtype = 'Observatory'
    _outspec = {}
     
    def __init__(self, settlingTime=1., thrust=450., slewIsp=4160.,\
                 scMass=6000., dryMass=3400., coMass=5800., skIsp=220.,\
                 defburnPortion=0.05, spkpath=None, forceStaticEphem=False,\
                 **specs):
        
        # default Observatory values
        # instrument settling time after repoint (days)
        self.settlingTime = float(settlingTime)*u.day 
        # occulter slew thrust (mN)
        self.thrust = float(thrust)*u.mN 
        # occulter slew specific impulse (s)
        self.slewIsp = float(slewIsp)*u.s 
        # occulter (maneuvering sc) initial (wet) mass (kg)
        self.scMass = float(scMass)*u.kg 
        # occulter (maneuvering sc) dry mass (kg)
        self.dryMass = float(dryMass)*u.kg 
        # telescope (or non-maneuvering sc) mass (kg)
        self.coMass = float(coMass)*u.kg 
        # station-keeping Isp (s)
        self.skIsp = float(skIsp)*u.s 
        # default burn portion
        self.defburnPortion = float(defburnPortion)

        # occulter-telescope distance (km)
        self.occulterSep = 55000.*u.km                

        #if jplephem is available, we'll use that for propagating solar system bodies
        #otherwise, use static ephemeris
        if not forceStaticEphem:
            try:
                from jplephem.spk import SPK
                self.havejplephem = True
            except ImportError:
                print "WARNING: Module jplephem not found, using static solar system ephemeris."
                self.havejplephem = False
        else:
            self.havejplephem = False
            print "Using static solar system ephemeris."

        #record all values populated so far to outspec
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key],u.Quantity):
                self._outspec[key] = self.__dict__[key].value
            else:
                self._outspec[key] = self.__dict__[key]
                   
        # initialize values updated by functions
        # observatory keepout booleans
        self.kogood = np.array([]) 
        # observatory orbit position vector (km)
        self.r_sc = np.zeros(3)*u.km 
        # current occulter separation
        self.currentSep = self.occulterSep 
        
        # set values derived from quantities above
        # slew flow rate (kg/day)
        self.flowRate = (self.thrust/const.g0/self.slewIsp).to(u.kg/u.day)
            
        # if you have jplephem, load spice file. otherwise, load static ephem.
        if self.havejplephem:
            if (spkpath is None) or not(os.path.exists(spkpath)):
                # if the path does not exist, load the default de432s.bsp
                    classpath = os.path.split(inspect.getfile(self.__class__))[0]
                    classpath = os.path.normpath(os.path.join(classpath,'..','Observatory'))
                    filename = 'de432s.bsp'
                    spkpath = os.path.join(classpath, filename)
            self.kernel = SPK.open(spkpath)
        else:
            # All ephemeride data from Vallado Appendix D.4
            # Store Mercury ephemerides data (ecliptic)
            Mercurya = 0.387098310
            Mercurye = [0.20563175, 0.000020406, -0.0000000284, -0.00000000017]
            Mercuryi = [7.004986, -0.0059516, 0.00000081, 0.000000041]
            MercuryO = [48.330893, -0.1254229, -0.00008833, -0.000000196]
            Mercuryw = [77.456119, 0.1588643, -0.00001343, 0.000000039]
            MercurylM = [252.250906, 149472.6746358, -0.00000535, 0.000000002]
            Mercury = self.SolarEph(Mercurya, Mercurye, Mercuryi, MercuryO, Mercuryw, MercurylM)
            
            # Store Venus epemerides data (ecliptic)
            Venusa = 0.723329820
            Venuse = [0.00677188, -0.000047766, 0.0000000975, 0.00000000044]
            Venusi = [3.394662, -0.0008568, -0.00003244, 0.000000010]
            VenusO = [76.679920, -0.2780080, -0.00014256, -0.000000198]
            Venusw = [131.563707, 0.0048646, -0.00138232, -0.000005332]
            VenuslM = [181.979801, 58517.8156760, 0.00000165, -0.000000002]
            Venus = self.SolarEph(Venusa, Venuse, Venusi, VenusO, Venusw, VenuslM)
            
            # Store Earth ephemerides data (ecliptic)
            Eartha = 1.000001018
            Earthe = [0.01670862, -0.000042037, -0.0000001236, 0.00000000004]
            Earthi = [0., 0.0130546, -0.00000931, -0.000000034]
            EarthO = [174.873174, -0.2410908, 0.00004067, -0.000001327]
            Earthw = [102.937348, 0.3225557, 0.00015026, 0.000000478]
            EarthlM = [100.466449, 35999.3728519, -0.00000568, 0.]
            Earth = self.SolarEph(Eartha, Earthe, Earthi, EarthO, Earthw, EarthlM)
            
            # Store Mars ephemerides data (ecliptic)
            Marsa = 1.523679342
            Marse = [0.09340062, 0.000090483, -0.0000000806, -0.00000000035]
            Marsi = [1.849726, -0.0081479, -0.00002255, -0.000000027]
            MarsO = [49.558093, -0.2949846, -0.00063993, -0.000002143]
            Marsw = [336.060234, 0.4438898, -0.00017321, 0.000000300]
            MarslM = [355.433275, 19140.2993313, 0.00000261, -0.000000003]
            Mars = self.SolarEph(Marsa, Marse, Marsi, MarsO, Marsw, MarslM)
            
            # Store Jupiter ephemerides data (ecliptic)
            Jupitera = [5.202603191, 0.0000001913]
            Jupitere = [0.04849485, 0.000163244, -0.0000004719, -0.00000000197]
            Jupiteri = [1.303270, -0.0019872, 0.00003318, 0.000000092]
            JupiterO = [100.464441, 0.1766828, 0.00090387, -0.000007032]
            Jupiterw = [14.331309, 0.2155525, 0.00072252, -0.000004590]
            JupiterlM = [34.351484, 3034.9056746, -0.00008501, 0.000000004]
            Jupiter = self.SolarEph(Jupitera, Jupitere, Jupiteri, JupiterO, Jupiterw, JupiterlM)
            
            # Store Saturn ephemerides data (ecliptic)
            Saturna = [9.554909596, -0.0000021389]
            Saturne = [0.05550862, -0.000346818, -0.0000006456, 0.00000000338]
            Saturni = [2.488878, 0.0025515, -0.00004903, 0.000000018]
            SaturnO = [113.665524, -0.2566649, -0.00018345, 0.000000357]
            Saturnw = [93.056787, 0.5665496, 0.00052809, 0.000004882]
            SaturnlM = [50.077471, 1222.1137943, 0.00021004, -0.000000019]
            Saturn = self.SolarEph(Saturna, Saturne, Saturni, SaturnO, Saturnw, SaturnlM)
            
            # Store Uranus ephemerides data (ecliptic)
            Uranusa = [19.218446062, -0.0000000372, 0.00000000098]
            Uranuse = [0.04629590, -0.000027337, 0.0000000790, 0.00000000025]
            Uranusi = [0.773196, -0.0016869, 0.00000349, 0.000000016]
            UranusO = [74.005947, 0.0741461, 0.00040540, 0.000000104]
            Uranusw = [173.005159, 0.0893206, -0.00009470, 0.000000413]
            UranuslM = [314.055005, 428.4669983, -0.00000486, 0.000000006]
            Uranus = self.SolarEph(Uranusa, Uranuse, Uranusi, UranusO, Uranusw, UranuslM)
            
            # Store Neptune ephemerides data (ecliptic)
            Neptunea = [30.110386869, -0.0000001663, 0.00000000069]
            Neptunee = [0.00898809, 0.000006408, -0.0000000008]
            Neptunei = [1.769952, 0.0002257, 0.00000023, -0.000000000]
            NeptuneO = [131.784057, -0.0061651, -0.00000219, -0.000000078]
            Neptunew = [48.123691, 0.0291587, 0.00007051, 0.]
            NeptunelM = [304.348665, 218.4862002, 0.00000059, -0.000000002]
            Neptune = self.SolarEph(Neptunea, Neptunee, Neptunei, NeptuneO, Neptunew, NeptunelM)
            
            # Store Pluto ephemerides data (ecliptic)
            Plutoa = [39.48168677, -0.00076912]
            Plutoe = [0.24880766, 0.00006465]
            Plutoi = [17.14175, 0.003075]
            PlutoO = [110.30347, -0.01036944]
            Plutow = [224.06676, -0.03673611]
            PlutolM = [238.92881, 145.2078]
            Pluto = self.SolarEph(Plutoa, Plutoe, Plutoi, PlutoO, Plutow, PlutolM)

            #store all as dictionary:
            self.planets = {'Mercury': Mercury,
                            'Venus': Venus,
                            'Earth': Earth,
                            'Mars': Mars,
                            'Jupiter': Jupiter,
                            'Saturn': Saturn,
                            'Uranus': Uranus,
                            'Neptune': Neptune,
                            'Pluto': Pluto}

    def __str__(self):
        """String representation of the Observatory object
        
        When the command 'print' is used on the Observatory object, this method
        will print the attribute values contained in the object"""
        
        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Observatory class object attributes'
        
    def orbit(self, time):
        """Finds observatory orbit position vector, returns True if successful
        
        This method finds the observatory position vector (heliocentric 
        equatorial frame) as a 1D numpy array with astropy Quantity units of km 
        and stores it in self.r_sc.
        
        This defines the data type expected, orbits are determined by specific
        instances of Observatory classes.
        
        Args:
            time (Time):
                absolute time
        
        Returns:
            success (bool):
                True if successful, False if not
        
        """
        
        self.r_sc = np.array([float(time.mjd),float(time.mjd),float(time.mjd)])*u.km
        b = np.isfinite(self.r_sc) # finds if all values are finite 
        success = all(b) # returns True if all values of self.r_sc are finite
        
        return success
    
    def keepout(self, currenttime, catalog, koangle):
        """Finds keepout Boolean values, returns True if successful
        
        This method finds the keepout Boolean values for each target star where
        True is an observable star and stores the 1D numpy array in self.kogood.
        
        Args:
            currenttime (Time):
                absolute time
            catalog (TargetList or StarCatalog):
                TargetList or StarCatalog class object
            koangle (float):
                telescope keepout angle in degrees
                
        Returns:
            success (bool):
                True if successful, False if not
        
        """
        
        # update spacecraft orbital position
        a = self.orbit(currenttime) 
        
        self.kogood = np.array([True for row in catalog.Name])
        
        # check to make sure all elements in self.kogood are Boolean
        b = [isinstance(element, np.bool_) for element in self.kogood]
        c = [a, b]        
        # return True if orbital position is successful and all elements of 
        # self.kogood are Boolean
        success = all(c) 
        
        return success

    def solarSystem_body_position(self, time, bodyname):
        """Finds position vector for solar system objects
        
        This passes all arguments to one of spk_body or keplerplanet, depending
        on the value of self.havejplephem.

        Args:
            time (Time):
                absolute time
            bodyname (str):
                solar system object name
        
        Returns:
            r_body (Quantity):
                heliocentric equatorial position vector in 1D numpy ndarray
                (units of km)
        
        """

        if self.havejplephem:
            return self.spk_body(time,bodyname)
        else:
            return self.keplerplanet(time,bodyname)


        
    def spk_body(self, time, bodyname):
        """Finds position vector for solar system objects
        
        This method uses spice kernel from NAIF to find heliocentric
        equatorial position vectors (astropy Quantity in km) for solar system
        objects.
        
        Args:
            time (Time):
                absolute time
            bodyname (str):
                solar system object name
        
        Returns:
            r_body (Quantity):
                heliocentric equatorial position vector in 1D numpy ndarray
                (units of km)
        
        """
        
        # dictionary of solar system bodies available in spice kernel
        bodies = {'Mercury':199,
                  'Venus':299,
                  'Earth':399,
                  'Mars':4,
                  'Jupiter':5,
                  'Saturn':6,
                  'Uranus':7,
                  'Neptune':8,
                  'Pluto':9,
                  'Sun':10,
                  'Moon':301}
        
        assert bodies.has_key(bodyname),\
                 "%s is not a recognized body name."%(bodyname)

        if bodies[bodyname] == 199:
            r_body = (self.kernel[0,1].compute(time.jd) + \
                    self.kernel[1,199].compute(time.jd) - \
                    self.kernel[0,10].compute(time.jd))*u.km
        elif bodies[bodyname] == 299:
            r_body = (self.kernel[0,2].compute(time.jd) + \
                    self.kernel[2,299].compute(time.jd) - \
                    self.kernel[0,10].compute(time.jd))*u.km
        elif bodies[bodyname] == 399:
            r_body = (self.kernel[0,3].compute(time.jd) + \
                    self.kernel[3,399].compute(time.jd) - \
                    self.kernel[0,10].compute(time.jd))*u.km
        elif bodies[bodyname] == 301:
            r_body = (self.kernel[0,3].compute(time.jd) + \
                    self.kernel[3,301].compute(time.jd) - \
                    self.kernel[0,10].compute(time.jd))*u.km
        else:
            r_body = (self.kernel[0,bodies[bodyname]].compute(time.jd) - \
                    self.kernel[0,10].compute(time.jd))*u.km
        
        return r_body

    def keplerplanet(self, time, bodyname):
        """Finds position vector for solar system objects
        
        This method uses algorithms 2 and 10 from Vallado 2013 to find 
        heliocentric equatorial position vectors (astropy Quantity in km) for 
        solar system objects.
        
        Args:
            time (Time):
                absolute time
            bodyname (str):
                solar system object name
                
        Returns:
            r_body (Quantity):
                heliocentric equatorial position vector in 1D numpy ndarray 
                (units of km)
        
        """
        
        if bodyname == 'Moon':
            r_Earth = self.keplerplanet(time, 'Earth')
            return r_Earth + self.moon_earth(time)

        assert self.planets.has_key(bodyname),\
                "%s is not a recognized body name."%(bodyname)

        planet = self.planets[bodyname] 
        # find Julian centuries from J2000
        TDB = self.cent(time)
        # update ephemeride data
        a = self.propeph(planet.a, TDB)
        e = self.propeph(planet.e, TDB)
        i = np.radians(self.propeph(planet.i, TDB))
        O = np.radians(self.propeph(planet.O, TDB))
        w = np.radians(self.propeph(planet.w, TDB))
        lM = np.radians(self.propeph(planet.lM, TDB))
        # Find mean anomaly and argument of perigee
        M = lM - w
        wp = w - O
        # Find eccentric anomaly
        E = self.eccanom(M,e)
        # Find true anomaly
        nu = np.arctan2(np.sin(E) * np.sqrt(1 - e**2), np.cos(E) - e)
        # Find semiparameter
        p = a*(1 - e**2)
        # position vector (km) in orbital plane
        r_planet = np.array([(p*np.cos(nu)/(1 + e*np.cos(nu))), (p*np.sin(nu))/(1 + e*np.cos(nu)), 0.])
        # position vector (km) in ecliptic plane        
        r_planet = np.dot(np.dot(self.rot(-O,3),self.rot(-i,1)),np.dot(self.rot(-wp,3),r_planet))
        # find obliquity of the ecliptic
        obe = 23.439279 - 0.0130102*TDB - 5.086e-8*(TDB**2) + 5.565e-7*(TDB**3) + 1.6e-10*(TDB**4) + 1.21e-11*(TDB**5)       
        # position vector (km) in heliocentric equatorial frame
        r_planet = np.dot(self.rot(np.radians(-obe),1),r_planet)*u.km
        
        return r_planet   
    
    def moon_earth(self, time):
        """Finds geocentric equatorial position vector (km) for Earth's moon
        
        This method uses Algorithm 31 from Vallado 2013 to find the geocentric
        equatorial position vector for Earth's moon.
        
        Args:
            time (Time):
                absolute time 
        
        Returns:
            r_moon (Quantity):
                geocentric equatorial position vector in 1D numpy array (units 
                of km) 
        
        """
        
        TDB = self.cent(time)
        la = np.radians(218.32 + 481267.8813*TDB + \
            6.29*np.sin(np.radians(134.9 + 477198.85*TDB)) - 
            1.27*np.sin(np.radians(259.2 - 413335.38*TDB)) + 
            0.66*np.sin(np.radians(235.7 + 890534.23*TDB)) + 
            0.21*np.sin(np.radians(269.9 + 954397.70*TDB)) - 
            0.19*np.sin(np.radians(357.5 + 35999.05*TDB)) - 
            0.11*np.sin(np.radians(186.6 + 966404.05*TDB)))
        
        phi = np.radians(5.13*np.sin(np.radians(93.3 + 483202.03*TDB)) + 
            0.28*np.sin(np.radians(228.2 + 960400.87*TDB)) - 
            0.28*np.sin(np.radians(318.3 + 6003.18*TDB)) - 
            0.17*np.sin(np.radians(217.6 - 407332.20*TDB)))
        
        P = np.radians(0.9508 + 0.0518*np.cos(np.radians(134.9 + 477198.85*TDB)) + 
            0.0095*np.cos(np.radians(259.2 - 413335.38*TDB)) + 
            0.0078*np.cos(np.radians(235.7 + 890534.23*TDB)) + 
            0.0028*np.cos(np.radians(269.9 + 954397.70*TDB)))
        
        e = np.radians(23.439291 - 0.0130042*TDB - 1.64e-7*TDB**2 + 5.04e-7*TDB**3)
        
        r = 1./np.sin(P)*6378.137 # km
        
        r_moon = r*np.array([np.cos(phi)*np.cos(la),
            np.cos(e)*np.cos(phi)*np.sin(la) - np.sin(e)*np.sin(phi),
            np.sin(e)*np.cos(phi)*np.sin(la) + np.cos(e)*np.sin(phi)])*u.km
        
        return r_moon
        
    def starprop(self, time, catalog, i):
        """Finds target star position vector (km) for current time (MJD)
        
        Args:
            time (Time): 
                absolute time
            catalog (TargetList or StarCatalog): 
                TargetList or StarCatalog class object
            i (int): 
                index for star catalog information

        Returns:
            r_star (Quantity): star position vector in heliocentric equatorial 
                frame in 1D numpy ndarray (units of km)
            
        """
        
        # right ascension value
        a = catalog.coords.ra[i]  
        # declination value
        d = catalog.coords.dec[i] 
        # right ascension proper motion (mas/yr)
        mua = catalog.pmra[i] 
        # declination proper motion (mas/yr)
        mud = catalog.pmdec[i] 
        # radial velocity (km/s)
        vr = catalog.rv[i] 
        # parallax (mas)
        parx = catalog.parx[i] 
        
        # helpful units
        # AU/yr in units
        AUyr = (1.*u.AU)/(1.*u.yr) 
        # km/s in units
        kms = (1.*u.km)/(1.*u.s) 
        j2000 = Time(2000., format='jyear')
        
        # proper motion magnitudes
        VE = (mua/parx)*AUyr
        VN = (mud/parx)*AUyr
        VR = vr*kms

        # conversion to heliocentric equatorial frame
        p = np.array([-np.sin(a), np.cos(a), 0.])
        q = np.array([-np.cos(a)*np.sin(d), -np.sin(a)*np.sin(d), np.cos(d)])
        r = np.array([np.cos(a)*np.cos(d), np.sin(a)*np.cos(d), np.sin(d)])
        
        # total velocity vector
        V = p*VE + q*VN + r*VR

        # initial position at J2000 epoch
        r0 = (r/parx)*1000.*u.pc
        
        # position
        r_star = r0 + V*(time.mjd - j2000.mjd)*u.d
        
        return r_star.to(u.km)
        
    def cent(self, currenttime):
        """Finds time in Julian centuries since J2000 epoch
        
        This quantity is needed for many algorithms from Vallado 2013.
        
        Args:
            currenttime (Time):
                absolute time
            
        Returns:
            TDB (float):
                time in Julian centuries since the J2000 epoch 
        
        """
        
        j2000 = Time(2000., format='jyear')
        
        TDB = (currenttime.jd - j2000.jd)/36525.
        
        return TDB
        
    def propeph(self, x, TDB):
        """Propagates ephemeride to current time and returns this value
        
        This method propagates the ephemerides from Vallado 2013 to the current
        time.
        
        Args:
            x (list):
                ephemeride list (maximum of 4 elements)
            TDB (float):
                time in Julian centuries since the J2000 epoch
        
        Returns:
            y (float):
                ephemeride value at current time
        
        """
        
        if isinstance(x, list):
            if len(x) < 4:
                q = 4 - len(x)
                i = 0
                while i < q:
                    x.append(0.)
                    i += 1
        elif (isinstance(x, float) or isinstance(x, int)):
            x = [float(x)]
            i = 0
            while i < 3:
                x.append(0.)
                i += 1
        
        y = x[0] + x[1]*TDB + x[2]*(TDB**2) + x[3]*(TDB**3)
        
        return y
        
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
        
    def rot(self, th, axis):
        """Finds the rotation matrix of angle th about the axis value
        
        Args:
            th (float):
                rotation angle in radians
            axis (int): 
                integer value denoting rotation axis (1,2, or 3)
        
        Returns:
            matrix (ndarray):
                rotation matrix defined as numpy ndarray
        
        """
        
        if axis == 1:
            return np.array([[1., 0., 0.], 
                       [0., np.cos(th), np.sin(th)], 
                       [0., -np.sin(th), np.cos(th)]])
        elif axis == 2:
            return np.array([[np.cos(th), 0., -np.sin(th)],
                       [0., 1., 0.],
                       [np.sin(th), 0., np.cos(th)]])
        elif axis == 3:
            return np.array([[np.cos(th), np.sin(th), 0.],
                       [-np.sin(th), np.cos(th), 0.],
                       [0., 0., 1.]])
        
    def distForces(self, time, targlist, s_ind):
        """Finds lateral and axial disturbance forces on an occulter 
        
        Args:
            time (TimeKeeping):
                TimeKeeping class object
            targlist (TargetList):
                TargetList class object
            s_ind (int):
                index of target star
                
        Returns:
            dF_lateral, dF_axial (Quantity, Quantity):
                lateral and axial disturbance forces (units of N)
        
        """
        
        # occulter separation distance
        occulterSep = self.occulterSep
        
        # get spacecraft position vector
        self.orbit(time.currenttimeAbs)
        r_Ts = self.r_sc
        # sun -> earth position vector
        r_Es = self.keplerplanet(time.currenttimeAbs, self.Earth)
        # sun -> target star vector
        r_ts = self.starprop(time.currenttimeAbs, targlist, s_ind)
        # Telescope -> target vector and unit vector
        r_tT = r_ts - r_Ts
        u_tT = r_tT/np.sqrt(np.sum(r_tT**2))
        # sun -> occulter vector
        r_Os = r_Ts + occulterSep*u_tT
        # Earth-Moon barycenter -> spacecraft vectors
        r_TE = r_Ts - r_Es
        r_OE = r_Os - r_Es

        # force on occulter
        F_sO = (-const.G*const.M_sun*self.scMass*r_Os/np.sqrt(np.sum(r_Os**2)**3)).to(u.N)
        mEMB = const.M_sun/328900.56
        F_EO = (-const.G*mEMB*self.scMass*r_OE/np.sqrt(np.sum(r_OE**2)**3)).to(u.N)

        F_O = F_sO + F_EO

        # force on telescope
        F_sT = (-const.G*const.M_sun*self.coMass*r_Ts/np.sqrt(np.sum(r_Ts**2))**3).to(u.N)
        F_ET = (-const.G*mEMB*self.coMass*r_TE/np.sqrt(np.sum(r_TE**2))**3).to(u.N)
        F_T = F_sT + F_ET

        # differential force
        dF = ((F_O/self.scMass - F_T/self.coMass)*self.scMass).to(u.N)
        
        dF_axial = np.dot(dF.to(u.N), u_tT)*u.N
        dF_lateral = np.sqrt(np.sum((dF - dF_axial*u_tT)**2))
        dF_axial = np.abs(dF_axial)
        
        return dF_lateral, dF_axial
        
    def mass_dec(self, dF_lateral, t_int):
        """Returns mass_used and deltaV 
        
        The values returned by this method are used to decrement spacecraft 
        mass for station-keeping.
        
        Args:
            dF_lateral (Quantity):
                lateral force on occulter (units of force)
            t_int (Quantity):
                integration time (units of time)
                
        Returns:
            intMdot, mass_used, deltaV (Quantity, Quantity, Quantity):
                mass flow rate (units like kg/day), 
                mass used in station-keeping (units of mass), 
                change in velocity required for station-keeping (velocity units 
                like km/s)
                        
        """
        
        intMdot = (1./np.cos(np.radians(45.))*np.cos(np.radians(5.))*dF_lateral/const.g0/self.skIsp).to(u.kg/u.s)
        mass_used = (intMdot*t_int).to(u.kg)
        deltaV = (dF_lateral/self.scMass*t_int).to(u.km/u.s)
        
        return intMdot, mass_used, deltaV
   
    class SolarEph:
        """Solar system ephemerides class 
        
        This class takes the constants in Appendix D.4 of Vallado as inputs
        and stores them for use in defining solar system ephemerides at a 
        given time.
        
        Args:
            a (list):
                semimajor axis list (in AU)
            e (list):
                eccentricity list
            i (list):
                inclination list
            O (list):
                right ascension of the ascending node list
            w (list):
                longitude of periapsis list
            lM (list):
                mean longitude list
                
        Each of these lists has a maximum of 4 elements. The values in 
        these lists are used to propagate the solar system planetary 
        ephemerides for a specific solar system planet.
        
        Attributes:
            a (list):
                list of semimajor axis (in AU)
            e (list):
                list of eccentricity
            i (list):
                list of inclination
            O (list):
                list of right ascension of the ascending node
            w (list):
                list of longitude of periapsis
            lM (list):
                list of mean longitude values
            
        Each of these lists has a maximum of 4 elements. The values in 
        these lists are used to propagate the solar system planetary 
        ephemerides for a specific solar system planet."""
        
        def __init__(self, a, e, i, O, w, lM):
            
            # attach units of AU
            self.a = a*u.AU 
            # change to km
            self.a = self.a.to(u.km) 
            # strip dimensions
            self.a = self.a.value 
            if not isinstance(self.a, float):
                self.a = self.a.tolist()
            # store list of dimensionless eccentricity values
            self.e = e 
            # store list of inclination values (degrees)
            self.i = i 
            # store list of right ascension of ascending node values (degrees)
            self.O = O 
            # store list of longitude of periapsis values (degrees)
            self.w = w 
            # store list of mean longitude values (degrees)
            self.lM = lM
        
        def __str__(self):
            """String representation of the SolarEph object
            
            When the command 'print' is used on the SolarEph object, this 
            method will print the attribute values contained in the object"""
            
            atts = self.__dict__.keys()
            for att in atts:
                print '%s: %r' % (att, getattr(self, att))
            
            return 'SolarEph class object attributes'
