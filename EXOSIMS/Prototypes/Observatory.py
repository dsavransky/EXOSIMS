# -*- coding: utf-8 -*-
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
import os,inspect
from EXOSIMS.util.eccanom import eccanom

class Observatory(object):
    """Observatory class template
    
    This class contains all variables and methods necessary to perform
    Observatory Definition Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs: 
            user specified values
        forceStaticEphem (bool):
            If True, forces use of stored ephemerides for solar system objects
            and uses eccanom utility to propagate.  Defaults to False.
        spkpath (str):
            Path to SPK file on disk (Defaults to de432s.bsp). 
    
    Attributes:
        settlingTime (astropy Quantity): 
            Instrument settling time after repoint in units of day
        thrust (astropy Quantity): 
            Occulter slew thrust in units of mN
        slewIsp (astropy Quantity): 
            Occulter slew specific impulse in units of s
        scMass (astropy Quantity): 
            Occulter (maneuvering sc) wet mass in units of kg
        dryMass (astropy Quantity): 
            Occulter (maneuvering sc) dry mass in units of kg
        coMass (astropy Quantity): 
            Telescope (non-maneuvering sc) mass in units of kg
        occulterSep (astropy Quantity): 
            Occulter-telescope distance in units of km
        skIsp (astropy Quantity): 
            Station-keeping specific impulse in units of s
        defburnPortion (float):
            Default burn portion
        flowRate (astropy Quantity): 
            Slew flow rate in units of kg/day
    
    Notes:
        For finding positions of solar system bodies, this routine will attempt to 
        use the jplephem module and a local SPK file on disk.  The module can be 
        installed via pip or from source.  The default SPK file can be downloaded from
        here: http://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp
        and should be placed in the Observatory subdirectory of EXOSIMS.
    
    """

    _modtype = 'Observatory'
    _outspec = {}

    def __init__(self, settlingTime=1., thrust=450., slewIsp=4160., scMass=6000.,\
                 dryMass=3400., coMass=5800., occulterSep=55000., skIsp=220.,\
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
        # occulter-telescope distance (km)
        self.occulterSep = float(occulterSep)*u.km
        # station-keeping Isp (s)
        self.skIsp = float(skIsp)*u.s 
        # default burn portion
        self.defburnPortion = float(defburnPortion)
        
        # set values derived from quantities above
        # slew flow rate (kg/day)
        self.flowRate = (self.thrust/const.g0/self.slewIsp).to('kg/day')
        
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
        
        # populate outspec
        for att in self.__dict__.keys():
            dat = self.__dict__[att]
            self._outspec[att] = dat.value if isinstance(dat,u.Quantity) else dat
        
        # define function for calculating obliquity of the ecliptic 
        # (arg Julian centuries from J2000)
        self.obe = lambda TDB: 23.439279 - 0.0130102*TDB - 5.086e-8*(TDB**2) + \
                5.565e-7*(TDB**3) + 1.6e-10*(TDB**4) + 1.21e-11*(TDB**5) 
        
        # if you have jplephem, load spice file, otherwise load static ephem
        if self.havejplephem:
            if (spkpath is None) or not(os.path.exists(spkpath)):
                # if the path does not exist, load the default de432s.bsp
                    classpath = os.path.split(inspect.getfile(self.__class__))[0]
                    classpath = os.path.normpath(os.path.join(classpath,'..','Observatory'))
                    filename = 'de432s.bsp'
                    spkpath = os.path.join(classpath, filename)
            self.kernel = SPK.open(spkpath)
        else:
            """All ephemeride data from Vallado Appendix D.4
            Values are:
            a     e             I               O                       w                   lM
            sma   eccentricity  inclination     long. ascending node    long. perihelion    mean longitude
            AU    N/A           deg             deg                     deg                 deg
            """
            
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
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Observatory class object attributes'

    def orbit(self, currentTime):
        """Finds observatory orbit position vector in heliocentric equatorial frame.
        
        This method defines the data type expected, orbits are determined by specific
        instances of Observatory classes.
        
        Args:
            currentTime (astropy Time array):
                Current absolute mission time in MJD
        
        Returns:
            r_sc (astropy Quantity nx3 array):
                Observatory (spacecraft) position vector in units of km
        
        """
        
        r_sc = np.vstack((currentTime.mjd, currentTime.mjd, currentTime.mjd)).T*u.km
        assert np.all(np.isfinite(r_sc)), 'Observatory position vector r_sc has infinite value.'
        
        return r_sc.to('km')

    def keepout(self, TL, sInds, currentTime, koangle):
        """Finds keepout Boolean values for stars of interest.
        
        This method defines the data type expected, all values are True.
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            koangle (astropy Quantity):
                Telescope keepout angle in units of degree
                
        Returns:
            kogood (boolean ndarray):
                True is a target unobstructed and observable, and False is a 
                target unobservable due to obstructions in the keepout zone.
        
        Note: If multiple times and targets, currentTime and sInds sizes must match.
        
        """
        
        # check size of arrays
        sInds = np.array(sInds,ndmin=1)
        nStars = sInds.size
        nTimes = currentTime.size
        assert nStars==1 or nTimes==1 or nTimes==nStars, 'If multiple times and targets, \
                currentTime and sInds sizes must match'
        
        # build "keepout good" array, check if all elements are Boolean
        kogood = np.ones(nStars, dtype=bool)
        trues = [isinstance(element, np.bool_) for element in kogood]
        assert all(trues), 'An element of kogood is not Boolean'
        
        return kogood

    def starprop(self, TL, sInds, currentTime):
        """Finds target star position vector (km) for current time (MJD)
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            currentTime (astropy Time):
                Current absolute mission time in MJD
        
        Returns:
            r_star (astropy Quantity nx3 array): 
                Position vectors of stars of interest in heliocentric 
                equatorial frame in units of km
        
        Note: If multiple times and targets, currentTime and sInds sizes must match.
        
        """
        
        # check size of arrays
        sInds = np.array(sInds,ndmin=1)
        nStars = sInds.size
        nTimes = currentTime.size
        assert nStars==1 or nTimes==1 or nTimes==nStars, 'If multiple times and targets, \
                currentTime and sInds sizes must match'
        
        # right ascension and declination
        ra = TL.coords.ra[sInds]
        dec = TL.coords.dec[sInds]
        # set J2000 epoch
        j2000 = Time(2000., format='jyear')
        # directions
        p0 = np.array([-np.sin(ra), np.cos(ra), np.zeros(sInds.size)])
        q0 = np.array([-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)])
        r0 = (TL.coords[sInds].cartesian.xyz/TL.coords[sInds].distance)
        # proper motion vector
        mu0 = p0*TL.pmra[sInds] + q0*TL.pmdec[sInds]
        # space velocity vector
        v = mu0/TL.parx[sInds]*u.AU + r0*TL.rv[sInds]
        # stellar position vector
        dr = (v*(currentTime.mjd - j2000.mjd)*u.day).decompose()
        r_star = (TL.coords[sInds].cartesian.xyz + dr).T
        
        return r_star.to('km')

    def solarSystem_body_position(self, currentTime, bodyname):
        """Finds position vector for solar system objects
        
        This passes all arguments to one of spk_body or keplerplanet, depending
        on the value of self.havejplephem.
        
        Args:
            currentTime (astropy Time):
                Current absolute mission time in MJD
            bodyname (string):
                Solar system object name
        
        Returns:
            r_body (astropy Quantity nx3 array):
                Heliocentric equatorial position vector in units of km
        
        """
        
        if self.havejplephem:
            r_body = self.spk_body(currentTime, bodyname)
        else:
            r_body = self.keplerplanet(currentTime, bodyname)
        
        return r_body.to('km')

    def spk_body(self, currentTime, bodyname):
        """Finds position vector for solar system objects
        
        This method uses spice kernel from NAIF to find heliocentric
        equatorial position vectors (astropy Quantity in km) for solar system
        objects.
        
        Args:
            currentTime (astropy Time):
                Current absolute mission time in MJD
            bodyname (string):
                Solar system object name
        
        Returns:
            r_body (astropy Quantity nx3 array):
                Heliocentric equatorial position vector in units of km
        
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
            r_body = (self.kernel[0,1].compute(currentTime.jd) + \
                    self.kernel[1,199].compute(currentTime.jd) - \
                    self.kernel[0,10].compute(currentTime.jd))*u.km
        elif bodies[bodyname] == 299:
            r_body = (self.kernel[0,2].compute(currentTime.jd) + \
                    self.kernel[2,299].compute(currentTime.jd) - \
                    self.kernel[0,10].compute(currentTime.jd))*u.km
        elif bodies[bodyname] == 399:
            r_body = (self.kernel[0,3].compute(currentTime.jd) + \
                    self.kernel[3,399].compute(currentTime.jd) - \
                    self.kernel[0,10].compute(currentTime.jd))*u.km
        elif bodies[bodyname] == 301:
            r_body = (self.kernel[0,3].compute(currentTime.jd) + \
                    self.kernel[3,301].compute(currentTime.jd) - \
                    self.kernel[0,10].compute(currentTime.jd))*u.km
        else:
            r_body = (self.kernel[0,bodies[bodyname]].compute(currentTime.jd) - \
                    self.kernel[0,10].compute(currentTime.jd))*u.km
        r_body = r_body.reshape(currentTime.size,3)
        
        return r_body.to('km')

    def keplerplanet(self, currentTime, bodyname):
        """Finds position vector for solar system objects
        
        This method uses algorithms 2 and 10 from Vallado 2013 to find 
        heliocentric equatorial position vectors (astropy Quantity in km) for 
        solar system objects.
        
        Args:
            currentTime (astropy Time):
                Current absolute mission time in MJD
            bodyname (string):
                Solar system object name
        
        Returns:
            r_body (astropy Quantity nx3 array):
                Heliocentric equatorial position vector in units of km
        
        """
        
        if bodyname == 'Moon':
            r_Earth = self.keplerplanet(currentTime, 'Earth')
            return r_Earth + self.moon_earth(currentTime)
        
        assert self.planets.has_key(bodyname),\
                "%s is not a recognized body name."%(bodyname)
        
        planet = self.planets[bodyname] 
        # find Julian centuries from J2000
        TDB = self.cent(currentTime)
        # update ephemeride data
        a = self.propeph(planet.a, TDB)
        e = self.propeph(planet.e, TDB)
        I = np.radians(self.propeph(planet.I, TDB))
        O = np.radians(self.propeph(planet.O, TDB))
        w = np.radians(self.propeph(planet.w, TDB))
        lM = np.radians(self.propeph(planet.lM, TDB))
        # Find mean anomaly and argument of perigee
        M = np.mod(lM - w,2*np.pi)
        wp = np.mod(w - O,2*np.pi)
        # Find eccentric anomaly
        E = eccanom(M,e)[0]
        # Find true anomaly
        nu = np.arctan2(np.sin(E) * np.sqrt(1 - e**2), np.cos(E) - e)
        # Find semiparameter
        p = a*(1 - e**2)
        # position vector (km) in orbital plane
        rx = p*np.cos(nu)/(1 + e*np.cos(nu))
        ry = p*np.sin(nu)/(1 + e*np.cos(nu))
        rz = np.zeros(currentTime.size)
        r_body = np.vstack((rx,ry,rz))
        # position vector (km) in ecliptic plane
        r_body = np.array([np.dot(np.dot(self.rot(-O[x],3),self.rot(-I[x],1)),\
                np.dot(self.rot(-wp[x],3),r_body[:,x])) for x in range(currentTime.size)]).T
        # find obliquity of the ecliptic
        obe = np.array(np.radians(self.obe(TDB)),ndmin=1)
        # position vector (km) in heliocentric equatorial frame
        r_body = np.array([np.dot(self.rot(-obe[x],1),r_body[:,x])\
                for x in range(currentTime.size)])*u.km
        
        return r_body.to('km')

    def moon_earth(self, currentTime):
        """Finds geocentric equatorial position vector (km) for Earth's moon
        
        This method uses Algorithm 31 from Vallado 2013 to find the geocentric
        equatorial position vector for Earth's moon.
        
        Args:
            currentTime (astropy Time):
                Current absolute mission time in MJD
        
        Returns:
            r_moon (astropy Quantity nx3 array):
                Geocentric equatorial position vector in units of km
        
        """
        
        TDB = self.cent(currentTime)
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
        r_moon = (r*np.vstack((np.cos(phi)*np.cos(la),
            np.cos(e)*np.cos(phi)*np.sin(la) - np.sin(e)*np.sin(phi),
            np.sin(e)*np.cos(phi)*np.sin(la) + np.cos(e)*np.sin(phi)))).T*u.km
        
        return r_moon.to('km')

    def cent(self, currentTime):
        """Finds time in Julian centuries since J2000 epoch
        
        This quantity is needed for many algorithms from Vallado 2013.
        
        Args:
            currentTime (astropy Time):
                Current absolute mission time in MJD
            
        Returns:
            TDB (float ndarray):
                time in Julian centuries since the J2000 epoch 
        
        """
        
        j2000 = Time(2000., format='jyear')
        TDB = (currentTime.jd - j2000.jd)/36525.
        
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
            y (float ndarray):
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
        
        # propagated ephem must be an array
        y = x[0] + x[1]*TDB + x[2]*(TDB**2) + x[3]*(TDB**3)
        y = np.array(y,ndmin=1)
        
        return y

    def rot(self, th, axis):
        """Finds the rotation matrix of angle th about the axis value
        
        Args:
            th (float):
                Rotation angle in radians
            axis (int): 
                Integer value denoting rotation axis (1,2, or 3)
        
        Returns:
            rot_th (float 3x3 ndarray):
                Rotation matrix
        
        """
        
        if axis == 1:
            rot_th = np.array([[1., 0., 0.], 
                    [0., np.cos(th), np.sin(th)], 
                    [0., -np.sin(th), np.cos(th)]])
        elif axis == 2:
            rot_th = np.array([[np.cos(th), 0., -np.sin(th)],
                    [0., 1., 0.],
                    [np.sin(th), 0., np.cos(th)]])
        elif axis == 3:
            rot_th = np.array([[np.cos(th), np.sin(th), 0.],
                    [-np.sin(th), np.cos(th), 0.],
                    [0., 0., 1.]])
        
        return rot_th

    def distForces(self, TL, sInd, currentTime):
        """Finds lateral and axial disturbance forces on an occulter 
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInd (integer ndarray):
                Integer indices of the star of interest
            currentTime (astropy Time):
                Current absolute mission time in MJD
                
        Returns:
            dF_lateral (astropy Quantity):
                Lateral disturbance force in units of N
            dF_axial (astropy Quantity):
                Axial disturbance force in units of N
        
        """
        
        # occulter separation distance
        occulterSep = self.occulterSep
        # get spacecraft position vector
        r_Ts = self.orbit(currentTime)[0]
        # sun -> earth position vector
        r_Es = self.solarSystem_body_position(currentTime, 'Earth')
        # sun -> target star vector
        r_ts = self.starprop(TL, sInd, currentTime)[0]
        # Telescope -> target vector and unit vector
        r_tT = r_ts - r_Ts
        u_tT = r_tT/np.sqrt(np.sum(r_tT**2))
        # sun -> occulter vector
        r_Os = r_Ts + occulterSep*u_tT
        # Earth-Moon barycenter -> spacecraft vectors
        r_TE = r_Ts - r_Es
        r_OE = r_Os - r_Es
        # force on occulter
        F_sO = (-const.G*const.M_sun*self.scMass*r_Os/np.sqrt(np.sum(r_Os**2)**3)).to('N')
        mEMB = const.M_sun/328900.56
        F_EO = (-const.G*mEMB*self.scMass*r_OE/np.sqrt(np.sum(r_OE**2)**3)).to('N')
        F_O = F_sO + F_EO
        # force on telescope
        F_sT = (-const.G*const.M_sun*self.coMass*r_Ts/np.sqrt(np.sum(r_Ts**2))**3).to('N')
        F_ET = (-const.G*mEMB*self.coMass*r_TE/np.sqrt(np.sum(r_TE**2))**3).to('N')
        F_T = F_sT + F_ET
        # differential forces
        dF = ((F_O/self.scMass - F_T/self.coMass)*self.scMass).to('N')
        dF_axial = np.dot(dF.to('N'), u_tT)
        dF_lateral = np.sqrt(np.sum((dF - dF_axial*u_tT)**2))
        dF_axial = np.abs(dF_axial)
        
        return dF_lateral, dF_axial

    def mass_dec(self, dF_lateral, t_int):
        """Returns mass_used and deltaV 
        
        The values returned by this method are used to decrement spacecraft 
        mass for station-keeping.
        
        Args:
            dF_lateral (astropy Quantity):
                Lateral disturbance force in units of N
            t_int (astropy Quantity):
                Integration time in units of day
                
        Returns:
            intMdot (astropy Quantity):
                Mass flow rate in units of kg/s
            mass_used (astropy Quantity):
                Mass used in station-keeping units of kg
            deltaV (astropy Quantity):
                Change in velocity required for station-keeping in units of km/s
                
        """
        
        intMdot = (1./np.cos(np.radians(45.))*np.cos(np.radians(5.))*dF_lateral/const.g0/self.skIsp).to('kg/s')
        mass_used = (intMdot*t_int).to('kg')
        deltaV = (dF_lateral/self.scMass*t_int).to('km/s')
        
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
            I (list):
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
            I (list):
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

        def __init__(self, a, e, I, O, w, lM):
            
            # store list of semimajor axis values (convert from AU to km)
            self.a = (a*u.AU).to('km').value
            if not isinstance(self.a, float):
                self.a = self.a.tolist()
            # store list of dimensionless eccentricity values
            self.e = e
            # store list of inclination values (degrees)
            self.I = I
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
            
            for att in self.__dict__.keys():
                print '%s: %r' % (att, getattr(self, att))
            
            return 'SolarEph class object attributes'
