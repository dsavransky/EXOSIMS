# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.Observatory import Observatory
import astropy.units as u
import numpy as np

class WFIRSTObservatory(Observatory):
    """WFIRST Observatory class
    
    This class contains all variables and methods specific to the WFIRST
    observatory needed to perform Observatory Definition Module calculations
    in exoplanet mission simulation.
    
    """

    def orbit(self, currentTime, eclip=False):
        """Finds observatory orbit positions vector in heliocentric equatorial (default)
        or ecliptic frame for current time (MJD).
        
        This method returns the WFIRST geosynchronous circular orbit position vector.
        
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
