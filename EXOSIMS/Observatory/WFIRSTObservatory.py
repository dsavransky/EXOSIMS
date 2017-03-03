# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.Observatory import Observatory
import astropy.units as u
import numpy as np

class WFIRSTObservatory(Observatory):
    """WFIRST Observatory class
    
    This class contains all variables and methods specific to the WFIRST
    observatory needed to perform Observatory Definition Module calculations
    in exoplanet mission simulation."""

    def orbit(self, currentTime):
        """Finds observatory orbit position vector in heliocentric equatorial frame.
        
        This method returns the WFIRST geosynchronous circular orbit position vector
        in the heliocentric equatorial frame.
        
        Args:
            currentTime (astropy Time array):
                Current absolute mission time in MJD
        
        Returns:
            r_sc (astropy Quantity nx3 array):
                Observatory (spacecraft) position vector in units of km
        
        """
        
        i = np.radians(28.5) # orbital inclination in degrees
        O = np.radians(228.) # right ascension of the ascending node in degrees
        # orbital period is one sidereal day
        f = 2.*np.pi # orbital frequency (2*pi/period)
        r = 42164.*u.km # orbital height in km
        # Find Earth position vector (km in heliocentric equatorial frame)
        r_earth = self.solarSystem_body_position(currentTime, 'Earth')
        # Find spacecraft position vector with respect to Earth in orbital plane
        t = currentTime.mjd - np.floor(currentTime.mjd) # gives percent of day
        r_scearth = r*np.vstack((np.cos(f*t), np.sin(f*t), np.zeros(t.size)))
        # position vector wrt Earth in equatorial frame
        r_scearth = np.dot(np.dot(self.rot(-O,3), self.rot(-i,1)),r_scearth).T
        # position vector in heliocentric equatorial frame
        r_sc = r_earth + r_scearth
        
        assert np.all(np.isfinite(r_sc)), 'Observatory position vector r_sc has infinite value.'
        
        return r_sc.to('km')

    def keepout(self, TL, sInds, currentTime, koangle, returnExtra=False):
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
            koangle (astropy Quantity):
                Telescope keepout angle in units of degree
            returnExtra (boolean):
                Optional flag, default False, set True to return additional rates for validation
                
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
        nBodies = 11
        assert nStars==1 or nTimes==1 or nTimes==nStars, 'If multiple times and targets, \
                currentTime and sInds sizes must match'
        
        # Observatory position
        r_sc = self.orbit(currentTime)
        # Position vectors wrt sun, for targets and bright bodies
        r_targ = self.starprop(TL, sInds, currentTime)
        r_body = np.array([np.zeros(r_sc.shape), # sun
            self.solarSystem_body_position(currentTime, 'Moon').to('km').value,
            self.solarSystem_body_position(currentTime, 'Earth').to('km').value,
            self.solarSystem_body_position(currentTime, 'Mercury').to('km').value,
            self.solarSystem_body_position(currentTime, 'Venus').to('km').value,
            self.solarSystem_body_position(currentTime, 'Mars').to('km').value,
            self.solarSystem_body_position(currentTime, 'Jupiter').to('km').value,
            self.solarSystem_body_position(currentTime, 'Saturn').to('km').value,
            self.solarSystem_body_position(currentTime, 'Uranus').to('km').value,
            self.solarSystem_body_position(currentTime, 'Neptune').to('km').value,
            self.solarSystem_body_position(currentTime, 'Pluto').to('km').value])*u.km
        # position vectors wrt spacecraft
        r_targ -= r_sc
        r_body -= r_sc
        # unit vectors wrt spacecraft
        u_targ = (r_targ.value.T/np.linalg.norm(r_targ, axis=-1)).T
        u_body = (r_body.value.T/np.linalg.norm(r_body, axis=-1).T).T
        
        # Create koangles for all bodies, set by telescope keepout angle for brighter 
        # objects (Sun, Moon, Earth) and defaults to 1 degree for other bodies.
        koangles = np.ones(nBodies)*koangle
        koangles[3:] = 1.*u.deg
        
        # Find angles and make angle comparisons to build kogood array.
        # If bright objects have an angle with the target vector less than koangle 
        # (e.g. pi/4) they are in the field of view and the target star may not be
        # observed, thus ko associated with this target becomes False.
        nkogood = np.maximum(nStars,nTimes)
        kogood = np.array([True]*nkogood)
        culprit = np.zeros([nkogood, nBodies])
        for i in xrange(nkogood):
            u_b = u_body[:,0,:] if nTimes == 1 else u_body[:,i,:]
            u_t = u_targ[0,:] if nStars == 1 else u_targ[i,:]
            angles = np.arccos(np.dot(u_b, u_t))
            culprit[i,:] = (angles < koangles.to('rad').value)
            if np.any(culprit[i,:]):
                kogood[i] = False
        
        # check to make sure all elements in kogood are Boolean
        trues = [isinstance(element, np.bool_) for element in kogood]
        assert all(trues), 'An element of kogood is not Boolean'
        
        if returnExtra:
            return kogood, r_body, r_targ, culprit
        else:
            return kogood
