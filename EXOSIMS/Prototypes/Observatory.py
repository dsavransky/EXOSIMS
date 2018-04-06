# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.eccanom import eccanom
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
from astropy.coordinates import SkyCoord
import os,inspect
import urllib

class Observatory(object):
    """Observatory class template
    
    This class contains all variables and methods necessary to perform
    Observatory Definition Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs: 
            user specified values
        spkpath (str):
            Path to SPK file on disk (Defaults to de432s.bsp). 
    
    Attributes:
        koAngleMin (astropy Quantity):
            Telescope minimum keepout angle in units of deg
        koAngleMinMoon (astropy Quantity):
            Telescope minimum keepout angle in units of deg, for the Moon only
        koAngleMinEarth (astropy Quantity):
            Telescope minimum keepout angle in units of deg, for the Earth only
        koAngleMax (astropy Quantity):
            Telescope maximum keepout angle (for occulter) in units of deg
        koAngleSmall (astropy Quantity):
            Telescope keepout angle for smaller (angular size) bodies in units of deg
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
        checkKeepoutEnd (boolean):
            Boolean signifying if the keepout method must be called at the end of 
            each observation
        forceStaticEphem (boolean):
            Boolean used to force static ephemerides
        constTOF (astropy Quantity 1x1 ndarray):
            Constant time of flight for single occulter slew in units of day
        maxdVpct (float):
            Maximum percentage of total on board fuel used for single starshade slew
    
    Notes:
        For finding positions of solar system bodies, this routine will attempt to 
        use the jplephem module and a local SPK file on disk.  The module can be 
        installed via pip or from source.  The default SPK file can be downloaded from
        here: http://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp
        and should be placed in the Observatory subdirectory of EXOSIMS.
    
    """

    _modtype = 'Observatory'

    def __init__(self, koAngleMin=45, koAngleMinMoon=None, koAngleMinEarth=None, 
            koAngleMax=90, koAngleSmall=1, settlingTime=1, thrust=450, slewIsp=4160, 
            scMass=6000, dryMass=3400, coMass=5800, occulterSep=55000, skIsp=220, 
            defburnPortion=0.05, constTOF=14, maxdVpct=0.02, spkpath=None, checkKeepoutEnd=True, 
            forceStaticEphem=False, **specs):

        #start the outspec
        self._outspec = {}
        
        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))
        
        # validate inputs
        assert isinstance(checkKeepoutEnd, bool), "checkKeepoutEnd must be a boolean."
        assert isinstance(forceStaticEphem, bool), "forceStaticEphem must be a boolean."
        
        # default Observatory values
        self.koAngleMin = koAngleMin*u.deg          # keepout minimum angle
        koAngleMinMoon = koAngleMin if koAngleMinMoon is None else koAngleMinMoon 
        self.koAngleMinMoon = koAngleMinMoon*u.deg  # keepout minimum angle: Moon-only
        koAngleMinEarth = koAngleMin if koAngleMinEarth is None else koAngleMinEarth 
        self.koAngleMinEarth = koAngleMinEarth*u.deg# keepout minimum angle: Earth-only
        self.koAngleMax = koAngleMax*u.deg          # keepout maximum angle (occulter)
        self.koAngleSmall = koAngleSmall*u.deg      # keepout angle for smaller bodies
        self.settlingTime = settlingTime*u.d        # instru. settling time after repoint
        self.thrust = thrust*u.mN                   # occulter slew thrust (mN)
        self.slewIsp = slewIsp*u.s                  # occulter slew specific impulse (s)
        self.scMass = scMass*u.kg                   # occulter initial (wet) mass (kg)
        self.dryMass = dryMass*u.kg                 # occulter dry mass (kg)
        self.coMass = coMass*u.kg                   # telescope mass (kg)
        self.occulterSep = occulterSep*u.km         # occulter-telescope distance (km)
        self.skIsp = skIsp*u.s                      # station-keeping Isp (s)
        self.defburnPortion = float(defburnPortion) # default burn portion
        self.checkKeepoutEnd = bool(checkKeepoutEnd)# true if keepout called at obs end 
        self.forceStaticEphem = bool(forceStaticEphem)# boolean used to force static ephem
        self.constTOF = np.array([constTOF])*u.d    #starshade constant slew time (day)
        self.maxdVpct = maxdVpct

        # find amount of fuel on board starshade and an upper bound for single slew dV
        self.dVtot = self.slewIsp*const.g0*np.log(self.scMass/self.dryMass)
        self.dVmax  = self.dVtot * self.maxdVpct        
        
        # set values derived from quantities above
        # slew flow rate (kg/day)
        self.flowRate = (self.thrust/const.g0/self.slewIsp).to('kg/day')
        
        # if jplephem is available, we'll use that for propagating solar system bodies
        # otherwise, use static ephemerides
        if self.forceStaticEphem is False:
            try:
                from jplephem.spk import SPK
                self.havejplephem = True
            except ImportError:
                self.vprint("WARNING: Module jplephem not found, " \
                        + "using static solar system ephemerides.")
                self.havejplephem = False
        else:
            self.havejplephem = False
            self.vprint("Using static solar system ephemerides.")
        
        # populate outspec
        for att in self.__dict__.keys():
            if att not in ['vprint','_outspec']:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat
        
        # define function for calculating obliquity of the ecliptic 
        # (arg Julian centuries from J2000)
        self.obe = lambda TDB: 23.439279 - 0.0130102*TDB - 5.086e-8*(TDB**2) + \
                5.565e-7*(TDB**3) + 1.6e-10*(TDB**4) + 1.21e-11*(TDB**5) 
        
        # if you have jplephem, load spice file, otherwise load static ephem
        if self.havejplephem:
            if (spkpath is None) or not(os.path.exists(spkpath)):
                # if the path does not exist, load the default de432s.bsp
                classpath = os.path.split(inspect.getfile(self.__class__))[0]
                classpath = os.path.normpath(os.path.join(classpath, '..', 
                        'Observatory'))
                filename = 'de432s.bsp'
                spkpath = os.path.join(classpath, filename)
                # attempt to fetch ephemeris and cache locally in EXOSIMS/Observatory/
                if not os.path.exists(spkpath) and os.access(classpath, os.W_OK|os.X_OK):
                    spk_on_web = 'https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp'
                    self.vprint("Fetching planetary ephemeris from %s to %s" % (spk_on_web, spkpath))
                    try:
                        urllib.urlretrieve(spk_on_web, spkpath)
                    except:
                        # Note: the SPK.open() below will fail in this case
                        self.vprint("Error: Remote fetch failed. Fetch manually or see install instructions.")
            self.kernel = SPK.open(spkpath)
        else:
            """All ephemeride data from Vallado Appendix D.4
            Values are: a = sma (AU), e = eccentricity, I = inclination (deg),
                        O = long. ascending node (deg), w = long. perihelion (deg),
                        lM = mean longitude (deg)
            
            """
            
            # store ephemerides data in heliocentric true ecliptic frame
            a = 0.387098310
            e = [0.20563175, 0.000020406, -0.0000000284, -0.00000000017]
            I = [7.004986, -0.0059516, 0.00000081, 0.000000041]
            O = [48.330893, -0.1254229, -0.00008833, -0.000000196]
            w = [77.456119, 0.1588643, -0.00001343, 0.000000039]
            lM = [252.250906, 149472.6746358, -0.00000535, 0.000000002]
            Mercury = self.SolarEph(a, e, I, O, w, lM)
            
            a = 0.723329820
            e = [0.00677188, -0.000047766, 0.0000000975, 0.00000000044]
            I = [3.394662, -0.0008568, -0.00003244, 0.000000010]
            O = [76.679920, -0.2780080, -0.00014256, -0.000000198]
            w = [131.563707, 0.0048646, -0.00138232, -0.000005332]
            lM = [181.979801, 58517.8156760, 0.00000165, -0.000000002]
            Venus = self.SolarEph(a, e, I, O, w, lM)
            
            a = 1.000001018
            e = [0.01670862, -0.000042037, -0.0000001236, 0.00000000004]
            I = [0., 0.0130546, -0.00000931, -0.000000034]
            O = [174.873174, -0.2410908, 0.00004067, -0.000001327]
            w = [102.937348, 0.3225557, 0.00015026, 0.000000478]
            lM = [100.466449, 35999.3728519, -0.00000568, 0.]
            Earth = self.SolarEph(a, e, I, O, w, lM)
            
            a = 1.523679342
            e = [0.09340062, 0.000090483, -0.0000000806, -0.00000000035]
            I = [1.849726, -0.0081479, -0.00002255, -0.000000027]
            O = [49.558093, -0.2949846, -0.00063993, -0.000002143]
            w = [336.060234, 0.4438898, -0.00017321, 0.000000300]
            lM = [355.433275, 19140.2993313, 0.00000261, -0.000000003]
            Mars = self.SolarEph(a, e, I, O, w, lM)
            
            a = [5.202603191, 0.0000001913]
            e = [0.04849485, 0.000163244, -0.0000004719, -0.00000000197]
            I = [1.303270, -0.0019872, 0.00003318, 0.000000092]
            O = [100.464441, 0.1766828, 0.00090387, -0.000007032]
            w = [14.331309, 0.2155525, 0.00072252, -0.000004590]
            lM = [34.351484, 3034.9056746, -0.00008501, 0.000000004]
            Jupiter = self.SolarEph(a, e, I, O, w, lM)
            
            a = [9.554909596, -0.0000021389]
            e = [0.05550862, -0.000346818, -0.0000006456, 0.00000000338]
            I = [2.488878, 0.0025515, -0.00004903, 0.000000018]
            O = [113.665524, -0.2566649, -0.00018345, 0.000000357]
            w = [93.056787, 0.5665496, 0.00052809, 0.000004882]
            lM = [50.077471, 1222.1137943, 0.00021004, -0.000000019]
            Saturn = self.SolarEph(a, e, I, O, w, lM)
            
            a = [19.218446062, -0.0000000372, 0.00000000098]
            e = [0.04629590, -0.000027337, 0.0000000790, 0.00000000025]
            I = [0.773196, -0.0016869, 0.00000349, 0.000000016]
            O = [74.005947, 0.0741461, 0.00040540, 0.000000104]
            w = [173.005159, 0.0893206, -0.00009470, 0.000000413]
            lM = [314.055005, 428.4669983, -0.00000486, 0.000000006]
            Uranus = self.SolarEph(a, e, I, O, w, lM)
            
            a = [30.110386869, -0.0000001663, 0.00000000069]
            e = [0.00898809, 0.000006408, -0.0000000008]
            I = [1.769952, 0.0002257, 0.00000023, -0.000000000]
            O = [131.784057, -0.0061651, -0.00000219, -0.000000078]
            w = [48.123691, 0.0291587, 0.00007051, 0.]
            lM = [304.348665, 218.4862002, 0.00000059, -0.000000002]
            Neptune = self.SolarEph(a, e, I, O, w, lM)
            
            a = [39.48168677, -0.00076912]
            e = [0.24880766, 0.00006465]
            I = [17.14175, 0.003075]
            O = [110.30347, -0.01036944]
            w = [224.06676, -0.03673611]
            lM = [238.92881, 145.2078]
            Pluto = self.SolarEph(a, e, I, O, w, lM)
            
            # store all as dictionary:
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
        will print the attribute values contained in the object
        
        """
        
        for att in self.__dict__.keys():
            print('%s: %r' % (att, getattr(self, att)))
        
        return 'Observatory class object attributes'

    def equat2eclip(self, r_equat, currentTime, rotsign=1):
        """Rotates heliocentric coordinates from equatorial to ecliptic frame.
        
        Args:
            r_equat (astropy Quantity nx3 array):
                Positions vector in heliocentric equatorial frame in units of AU
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            rotsign (integer):
                Optional flag, default 1, set -1 to reverse the rotation
        
        Returns:
            r_eclip (astropy Quantity nx3 array):
                Positions vector in heliocentric ecliptic frame in units of AU
        
        """
        
        # check size of arrays
        assert currentTime.size == 1 or currentTime.size == len(r_equat), \
                "If multiple times and positions, currentTime and r_equat sizes must match"
        # find Julian centuries from J2000
        TDB = self.cent(currentTime)
        # find obliquity of the ecliptic
        obe = rotsign*np.array(np.radians(self.obe(TDB)), ndmin=1)
        # positions vector in heliocentric ecliptic frame
        if currentTime.size == 1:
            r_eclip = np.array([np.dot(self.rot(obe[0], 1), 
                r_equat[x,:].to('AU').value) for x in range(len(r_equat))])*u.AU
        else:
            r_eclip = np.array([np.dot(self.rot(obe[x], 1), 
                r_equat[x,:].to('AU').value) for x in range(len(r_equat))])*u.AU
        
        return r_eclip

    def eclip2equat(self, r_eclip, currentTime):
        """Rotates heliocentric coordinates from ecliptic to equatorial frame.
        
        Args:
            r_eclip (astropy Quantity nx3 array):
                Positions vector in heliocentric ecliptic frame in units of AU
            currentTime (astropy Time array):
                Current absolute mission time in MJD
        
        Returns:
            r_equat (astropy Quantity nx3 array):
                Positions vector in heliocentric equatorial frame in units of AU
        
        """
        
        r_equat = self.equat2eclip(r_eclip, currentTime, rotsign=-1)
        
        return r_equat

    def orbit(self, currentTime, eclip=False):
        """Finds observatory orbit positions vector in heliocentric equatorial (default)
        or ecliptic frame for current time (MJD).
        
        This method returns the telescope geosynchronous circular orbit position vector.
        
        Args:
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            eclip (boolean):
                Boolean used to switch to heliocentric ecliptic frame. Defaults to 
                False, corresponding to heliocentric equatorial frame.
        
        Returns:
            r_obs (astropy Quantity nx3 array):
                Observatory orbit positions vector in heliocentric equatorial (default)
                or ecliptic frame in units of AU
        
        Note: Use eclip=True to get ecliptic coordinates.
        
        """
        
        mjdtime = np.array(currentTime.mjd, ndmin=1) # modified julian day time
        t = mjdtime % 1                     # gives percent of day
        f = 2.*np.pi                        # orbital frequency (2*pi/sideral day)
        r = (42164.*u.km).to('AU').value    # orbital height (convert from km to AU)
        I = np.radians(28.5)                # orbital inclination in degrees
        O = np.radians(228.)                # right ascension of the ascending node
        
        # observatory positions vector wrt Earth in orbital plane
        r_orb = r*np.array([np.cos(f*t), np.sin(f*t), np.zeros(t.size)])
        # observatory positions vector wrt Earth in equatorial frame
        r_obs_earth = np.dot(np.dot(self.rot(-O, 3), self.rot(-I, 1)), r_orb).T*u.AU
        # Earth positions vector in heliocentric equatorial frame
        r_earth = self.solarSystem_body_position(currentTime, 'Earth')
        # observatory positions vector in heliocentric equatorial frame
        r_obs = (r_obs_earth + r_earth).to('AU')
        
        assert np.all(np.isfinite(r_obs)), \
                "Observatory positions vector r_obs has infinite value."
        
        if eclip:
            # observatory positions vector in heliocentric ecliptic frame
            r_obs = self.equat2eclip(r_obs, currentTime)
        
        return r_obs

    def keepout(self, TL, sInds, currentTime, mode, returnExtra=False):
        """Finds keepout Boolean values for stars of interest.
        
        This method returns the keepout Boolean values for stars of interest, where
        True is an observable star.
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            mode (dict):
                Selected observing mode
            returnExtra (boolean):
                Optional flag, default False, set True to return additional rates 
                for validation
                
        Returns:
            kogood (boolean ndarray):
                True is a target unobstructed and observable, and False is a 
                target unobservable due to obstructions in the keepout zone.
        
        Note: If multiple times and targets, currentTime and sInds sizes must match.
        
        """
        
        # if multiple time values, check they are different otherwise reduce to scalar
        if currentTime.size > 1:
            if np.all(currentTime == currentTime[0]):
                currentTime = currentTime[0]
        
        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        # get all array sizes
        nStars = sInds.size
        nTimes = currentTime.size
        nBodies = 11
        assert nStars == 1 or nTimes == 1 or nTimes == nStars, \
                "If multiple times and targets, currentTime and sInds sizes must match"
        
        # observatory positions vector in heliocentric equatorial frame
        r_obs = self.orbit(currentTime)
        # traget star positions vector in heliocentric equatorial frame
        r_targ = TL.starprop(sInds, currentTime)
        # body positions vector in heliocentric equatorial frame
        r_body = np.array([
            self.solarSystem_body_position(currentTime, 'Sun').to('AU').value,
            self.solarSystem_body_position(currentTime, 'Moon').to('AU').value,
            self.solarSystem_body_position(currentTime, 'Earth').to('AU').value,
            self.solarSystem_body_position(currentTime, 'Mercury').to('AU').value,
            self.solarSystem_body_position(currentTime, 'Venus').to('AU').value,
            self.solarSystem_body_position(currentTime, 'Mars').to('AU').value,
            self.solarSystem_body_position(currentTime, 'Jupiter').to('AU').value,
            self.solarSystem_body_position(currentTime, 'Saturn').to('AU').value,
            self.solarSystem_body_position(currentTime, 'Uranus').to('AU').value,
            self.solarSystem_body_position(currentTime, 'Neptune').to('AU').value,
            self.solarSystem_body_position(currentTime, 'Pluto').to('AU').value])*u.AU
        # position vectors wrt spacecraft
        r_targ = (r_targ - r_obs).to('pc')
        r_body = (r_body - r_obs).to('AU')
        # unit vectors wrt spacecraft
        u_targ = (r_targ.value.T/np.linalg.norm(r_targ, axis=-1)).T
        u_body = (r_body.value.T/np.linalg.norm(r_body, axis=-1).T).T
        
        # create koangles for all bodies, set by telescope minimum keepout angle for 
        # brighter objects (Sun, Moon, Earth) and defaults to 1 degree for other bodies
        koangles = np.ones(nBodies)*self.koAngleMin
        # allow Moon, Earth to be set individually (default to koAngleMin)
        koangles[1] = self.koAngleMinMoon 
        koangles[2] = self.koAngleMinEarth
        # keepout angle for small bodies (other planets)
        koangles[3:] = self.koAngleSmall
        
        # find angles and make angle comparisons to build kogood array:
        # if bright objects have an angle with the target vector less than koangle 
        # (e.g. pi/4) they are in the field of view and the target star may not be
        # observed, thus ko associated with this target becomes False
        nkogood = np.maximum(nStars, nTimes)
        kogood = np.array([True]*nkogood)
        culprit = np.zeros([nkogood, nBodies])
        for i in xrange(nkogood):
            u_b = u_body[:,0,:] if nTimes == 1 else u_body[:,i,:]
            u_t = u_targ[0,:] if nStars == 1 else u_targ[i,:]
            angles = np.arccos(np.clip(np.dot(u_b, u_t), -1, 1))*u.rad
            culprit[i,:] = (angles < koangles)
            # if this mode has an occulter, check maximum keepout angle for the Sun
            if mode['syst']['occulter']:
                culprit[i,0] = (culprit[i,0] or (angles[0] > self.koAngleMax))
            if np.any(culprit[i,:]):
                kogood[i] = False
        
        # check to make sure all elements in kogood are Boolean
        trues = [isinstance(element, np.bool_) for element in kogood]
        assert all(trues), "An element of kogood is not Boolean"
        
        if returnExtra:
            return kogood, r_body, r_targ, culprit, koangles
        else:
            return kogood

    def solarSystem_body_position(self, currentTime, bodyname, eclip=False):
        """Finds solar system body positions vector in heliocentric equatorial (default)
        or ecliptic frame for current time (MJD).
        
        This passes all arguments to one of spk_body or keplerplanet, depending
        on the value of self.havejplephem.
        
        Args:
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            bodyname (string):
                Solar system object name
            eclip (boolean):
                Boolean used to switch to heliocentric ecliptic frame. Defaults to 
                False, corresponding to heliocentric equatorial frame.
        
        Returns:
            r_body (astropy Quantity nx3 array):
                Solar system body positions in heliocentric equatorial (default)
                or ecliptic frame in units of AU
        
        Note: Use eclip=True to get ecliptic coordinates.
        
        """
        
        # heliocentric
        if bodyname == 'Sun':
            return np.zeros((currentTime.size, 3))*u.AU
        
        # choose JPL or static ephemerides
        if self.havejplephem:
            r_body = self.spk_body(currentTime, bodyname, eclip=eclip).to('AU')
        else:
            r_body = self.keplerplanet(currentTime, bodyname, eclip=eclip).to('AU')
        
        return r_body

    def spk_body(self, currentTime, bodyname, eclip=False):
        """Finds solar system body positions vector in heliocentric equatorial (default)
        or ecliptic frame for current time (MJD).
        
        This method uses spice kernel from NAIF to find heliocentric 
        equatorial position vectors for solar system objects.
        
        Args:
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            bodyname (string):
                Solar system object name
            eclip (boolean):
                Boolean used to switch to heliocentric ecliptic frame. Defaults to 
                False, corresponding to heliocentric equatorial frame.
        
        Returns:
            r_body (astropy Quantity nx3 array):
                Solar system body positions in heliocentric equatorial (default)
                or ecliptic frame in units of AU
        
        Note: Use eclip=True to get ecliptic coordinates.
        
        """
        
        # dictionary of solar system bodies available in spice kernel (in km)
        bodies = {'Mercury': 199,
                  'Venus': 299,
                  'Earth': 399,
                  'Mars': 4,
                  'Jupiter': 5,
                  'Saturn': 6,
                  'Uranus': 7,
                  'Neptune': 8,
                  'Pluto': 9,
                  'Sun': 10,
                  'Moon': 301}
        assert bodies.has_key(bodyname), \
                 "%s is not a recognized body name."%(bodyname)
        
        # julian day time
        jdtime = np.array(currentTime.jd, ndmin=1)
        # body positions vector in heliocentric equatorial frame
        if bodies[bodyname] == 199:
            r_body = (self.kernel[0,1].compute(jdtime) +
                    self.kernel[1,199].compute(jdtime) -
                    self.kernel[0,10].compute(jdtime))
        elif bodies[bodyname] == 299:
            r_body = (self.kernel[0,2].compute(jdtime) +
                    self.kernel[2,299].compute(jdtime) -
                    self.kernel[0,10].compute(jdtime))
        elif bodies[bodyname] == 399:
            r_body = (self.kernel[0,3].compute(jdtime) +
                    self.kernel[3,399].compute(jdtime) -
                    self.kernel[0,10].compute(jdtime))
        elif bodies[bodyname] == 301:
            r_body = (self.kernel[0,3].compute(jdtime) +
                    self.kernel[3,301].compute(jdtime) -
                    self.kernel[0,10].compute(jdtime))
        else:
            r_body = (self.kernel[0,bodies[bodyname]].compute(jdtime) -
                    self.kernel[0,10].compute(jdtime))
        # reshape and convert units
        r_body = (r_body*u.km).T.to('AU')
        
        if eclip:
            # body positions vector in heliocentric ecliptic frame
            r_body = self.equat2eclip(r_body, currentTime)
        
        return r_body

    def keplerplanet(self, currentTime, bodyname, eclip=False):
        """Finds solar system body positions vector in heliocentric equatorial (default)
        or ecliptic frame for current time (MJD).
        
        This method uses algorithms 2 and 10 from Vallado 2013 to find 
        heliocentric equatorial position vectors for solar system objects.
        
        Args:
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            bodyname (string):
                Solar system object name
            eclip (boolean):
                Boolean used to switch to heliocentric ecliptic frame. Defaults to 
                False, corresponding to heliocentric equatorial frame.
        
        Returns:
            r_body (astropy Quantity nx3 array):
                Solar system body positions in heliocentric equatorial (default)
                or ecliptic frame in units of AU
        
        Note: Use eclip=True to get ecliptic coordinates.
        
        """
        
        # Moon positions based on Earth positions
        if bodyname == 'Moon':
            r_Earth = self.keplerplanet(currentTime, 'Earth')
            return r_Earth + self.moon_earth(currentTime)
        
        assert self.planets.has_key(bodyname),\
                "%s is not a recognized body name."%(bodyname)
        
        # find Julian centuries from J2000
        TDB = self.cent(currentTime)
        # update ephemerides data (convert sma from km to AU)
        planet = self.planets[bodyname] 
        a = (self.propeph(planet.a, TDB)*u.km).to('AU').value
        e = self.propeph(planet.e, TDB)
        I = np.radians(self.propeph(planet.I, TDB))
        O = np.radians(self.propeph(planet.O, TDB))
        w = np.radians(self.propeph(planet.w, TDB))
        lM = np.radians(self.propeph(planet.lM, TDB))
        # find mean anomaly and argument of perigee
        M = (lM - w) % (2*np.pi)
        wp = (w - O) % (2*np.pi)
        # find eccentric anomaly
        E = eccanom(M,e)[0]
        # find true anomaly
        nu = np.arctan2(np.sin(E) * np.sqrt(1 - e**2), np.cos(E) - e)
        # find semiparameter
        p = a*(1 - e**2)
        # body positions vector in orbital plane
        rx = p*np.cos(nu)/(1 + e*np.cos(nu))
        ry = p*np.sin(nu)/(1 + e*np.cos(nu))
        rz = np.zeros(currentTime.size)
        r_orb = np.array([rx,ry,rz])
        # body positions vector in heliocentric ecliptic plane
        r_body = np.array([np.dot(np.dot(self.rot(-O[x], 3), 
                self.rot(-I[x], 1)), np.dot(self.rot(-wp[x], 3), 
                r_orb[:,x])) for x in range(currentTime.size)])*u.AU
        
        if not eclip:
            # body positions vector in heliocentric equatorial frame
            r_body = self.eclip2equat(r_body, currentTime)
        
        return r_body

    def moon_earth(self, currentTime):
        """Finds geocentric equatorial positions vector for Earth's moon
        
        This method uses Algorithm 31 from Vallado 2013 to find the geocentric
        equatorial positions vector for Earth's moon.
        
        Args:
            currentTime (astropy Time array):
                Current absolute mission time in MJD
        
        Returns:
            r_moon (astropy Quantity nx3 array):
                Geocentric equatorial position vector in units of AU
        
        """
        
        TDB = np.array(self.cent(currentTime), ndmin=1)
        la = np.radians(218.32 + 481267.8813*TDB + 
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
                np.sin(e)*np.cos(phi)*np.sin(la) + np.cos(e)*np.sin(phi)])
        
        # set format and units
        r_moon = (r_moon*u.km).T.to('AU')
        
        return r_moon

    def cent(self, currentTime):
        """Finds time in Julian centuries since J2000 epoch
        
        This quantity is needed for many algorithms from Vallado 2013.
        
        Args:
            currentTime (astropy Time array):
                Current absolute mission time in MJD
            
        Returns:
            TDB (float ndarray):
                time in Julian centuries since the J2000 epoch 
        
        """
        
        j2000 = Time(2000., format='jyear')
        TDB = (currentTime.jd - j2000.jd)/36525.
        
        return TDB

    def propeph(self, x, TDB):
        """Propagates an ephemeris from Vallado 2013 to current time.
        
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
        
        # propagated ephem
        y = x[0] + x[1]*TDB + x[2]*(TDB**2) + x[3]*(TDB**3)
        # cast to array
        y = np.array(y, ndmin=1, copy=False)
        
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
            sInd (integer):
                Integer index of the star of interest
            currentTime (astropy Time):
                Current absolute mission time in MJD
                
        Returns:
            dF_lateral (astropy Quantity):
                Lateral disturbance force in units of N
            dF_axial (astropy Quantity):
                Axial disturbance force in units of N
        
        """
        
        # get spacecraft position vector
        r_obs = self.orbit(currentTime)[0]
        # sun -> earth position vector
        r_Es = self.solarSystem_body_position(currentTime, 'Earth')[0]
        # Telescope -> target vector and unit vector
        r_targ = TL.starprop(sInd, currentTime)[0] - r_obs
        u_targ = r_targ.value/np.linalg.norm(r_targ)
        # sun -> occulter vector
        r_Os = r_obs + self.occulterSep*u_targ
        # Earth-Moon barycenter -> spacecraft vectors
        r_TE = r_obs - r_Es
        r_OE = r_Os - r_Es
        # force on occulter
        Mfactor = -self.scMass*const.M_sun*const.G
        F_sO = r_Os/(np.linalg.norm(r_Os)*r_Os.unit)**3 * Mfactor
        F_EO = r_OE/(np.linalg.norm(r_OE)*r_OE.unit)**3 * Mfactor/328900.56
        F_O = F_sO + F_EO
        # force on telescope
        Mfactor = -self.coMass*const.M_sun*const.G
        F_sT = r_obs/(np.linalg.norm(r_obs)*r_obs.unit)**3 * Mfactor
        F_ET = r_TE/(np.linalg.norm(r_TE)*r_TE.unit)**3 * Mfactor/328900.56
        F_T = F_sT + F_ET
        # differential forces
        dF = F_O - F_T*self.scMass/self.coMass
        dF_axial = dF.dot(u_targ).to('N')
        dF_lateral = (dF - dF_axial*u_targ).to('N')
        dF_lateral = np.linalg.norm(dF_lateral)*dF_lateral.unit
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
        
        intMdot = (1./np.cos(np.radians(45))*np.cos(np.radians(5))*
                dF_lateral/const.g0/self.skIsp).to('kg/s')
        mass_used = (intMdot*t_int).to('kg')
        deltaV = (dF_lateral/self.scMass*t_int).to('km/s')
        
        return intMdot, mass_used, deltaV
    
    def mass_dec_sk(self, TL, sInd, currentTime, t_int):
        """Returns mass_used, deltaV and disturbance forces
        
        This method calculates all values needed to decrement spacecraft mass
        for station-keeping.
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInd (integer):
                Integer index of the star of interest
            currentTime (astropy Time):
                Current absolute mission time in MJD
            t_int (astropy Quantity):
                Integration time in units of day
                
        Returns:
            dF_lateral (astropy Quantity):
                Lateral disturbance force in units of N
            dF_axial (astropy Quantity):
                Axial disturbance force in units of N
            intMdot (astropy Quantity):
                Mass flow rate in units of kg/s
            mass_used (astropy Quantity):
                Mass used in station-keeping units of kg
            deltaV (astropy Quantity):
                Change in velocity required for station-keeping in units of km/s
        
        """
        
        dF_lateral, dF_axial = self.distForces(TL, sInd, currentTime)
        intMdot, mass_used, deltaV = self.mass_dec(dF_lateral, t_int)
        
        return dF_lateral, dF_axial, intMdot, mass_used, deltaV
    
    def calculate_dV(self,dt,TL,nA,N,tA):  
        """Finds the change in velocity needed to transfer to a new star line of sight
        
        This method sums the total delta-V needed to transfer from one star
        line of sight to another. It determines the change in velocity to move from
        one station-keeping orbit to a transfer orbit at the current time, then from
        the transfer orbit to the next station-keeping orbit at currentTime + dt.
        Station-keeping orbits are modeled as discrete boundary value problems.
        This method can handle multiple indeces for the next target stars and calculates
        the dVs of each trajectory from the same starting star.
        
        Args:
            dt (float 1x1 ndarray):
                Number of days corresponding to starshade slew time
            TL (float 1x3 ndarray):
                TargetList class object
            nA (integer):
                Integer index of the current star of interest
            N  (integer):
                Integer index of the next star(s) of interest
            tA (astropy Time array):
                Current absolute mission time in MJD
                
        Returns:
            dV (float nx6 ndarray):
                State vectors in rotating frame in normalized units
        """

        dV = np.zeros(len(N))
        
        return dV*u.m/u.s    
        
    def calculate_slewTimes(self,TL,old_sInd,sInds,currentTime):
        """Finds slew times and separation angles between target stars
        
        This method determines the slew times of an occulter spacecraft needed
        to transfer from one star's line of sight to all others in a given 
        target list.
        
        Args:
            TL (TargetList module):
                TargetList class object
            old_sInd (integer):
                Integer index of the most recently observed star
            sInds (integer):
                Integer indeces of the star of interest
            currentTime (astropy Time):
                Current absolute mission time in MJD
                
        Returns:
            sd (astropy Quantity):
                Angular separation between stars in rad
            slewTimes (astropy Quantity):
                Time to transfer to new star line of sight in units of days
        """
    
        self.ao = self.thrust/self.scMass
        slewTime_fac = (2.*self.occulterSep/np.abs(self.ao)/(self.defburnPortion/2. - 
            self.defburnPortion**2/4.)).decompose().to('d2')

        if old_sInd is None:
            sd = np.array([np.radians(0)]*TL.nStars)*u.rad
            slewTimes = np.zeros(TL.nStars)*u.d
        else:
            # position vector of previous target star
            r_old = TL.starprop(old_sInd, currentTime)[0]
            u_old = r_old.value/np.linalg.norm(r_old)
            # position vector of new target stars
            r_new = TL.starprop(sInds, currentTime)
            u_new = (r_new.value.T/np.linalg.norm(r_new, axis=1)).T
            # angle between old and new stars
            sd = np.arccos(np.clip(np.dot(u_old, u_new.T), -1, 1))*u.rad
            # calculate slew time
            slewTimes = np.sqrt(slewTime_fac*np.sin(sd/2.))
        
        return sd,slewTimes
    
    def log_occulterResults(self,DRM,slewTimes,sInd,sd,dV):
        """Updates the given DRM to include occulter values and results
        
        Args:
            DRM (dict):
                Design Reference Mission, contains the results of one complete
                observation (detection and characterization)
            slewTimes (astropy Quantity):
                Time to transfer to new star line of sight in units of days
            sInd (integer):
                Integer index of the star of interest
            sd (astropy Quantity):
                Angular separation between stars in rad
            dV (astropy Quantity):
                Delta-V used to transfer to new star line of sight in units of m/s
                
        Returns:
            DRM (dict):
                Design Reference Mission, contains the results of one complete
                observation (detection and characterization)
        
        """
        
        DRM['slew_time'] = slewTimes.to('day')
        DRM['slew_angle'] = sd.to('deg')
        
        slew_mass_used = slewTimes*self.defburnPortion*self.flowRate
        DRM['slew_dV'] = (slewTimes*self.ao*self.defburnPortion).to('m/s')
        DRM['slew_mass_used'] = slew_mass_used.to('kg')
        self.scMass = self.scMass - slew_mass_used
        DRM['scMass'] = self.scMass.to('kg')
        
        return DRM

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
        ephemerides for a specific solar system planet.
        
        """

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
            method will print the attribute values contained in the object
            
            """
            
            for att in self.__dict__.keys():
                print('%s: %r' % (att, getattr(self, att)))
            
            return 'SolarEph class object attributes'
