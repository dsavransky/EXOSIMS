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
            r_sc (astropy Quantity 3xn array):
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
        r_scearth = np.dot(np.dot(self.rot(-O,3), self.rot(-i,1)),r_scearth)
        # position vector in heliocentric equatorial frame
        r_sc = r_earth + r_scearth
        
        assert np.all(np.isfinite(r_sc)), 'Observatory position vector r_sc has infinite value.'
        
        return r_sc.to('km')

    def keepout(self, TL, sInds, currentTime, koangle):
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
                
        Returns:
            kogood (boolean ndarray):
                True is a target unobstructed and observable, and False is a 
                target unobservable due to obstructions in the keepout zone.
        
        Note: currentTime must be of size 1, or size of sInds.
        
        """
        
        # reshape sInds
        sInds = np.array(sInds,ndmin=1)
        # check size of currentTime
        assert (currentTime.size == 1) or currentTime.size == sInds.size, \
                    'CurrentTime must be of size 1, or size of sInds'
        
        # Observatory position
        r_sc = self.orbit(currentTime)
        
        # Position vectors wrt spacecraft for targets and bright bodies, wrt sun
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
        nBodies = np.size(r_body,0)
        r_body = r_body.reshape(3, currentTime.size, nBodies)
        
        # unit vectors wrt spacecraft
        u_targ = r_targ.value/np.linalg.norm(r_targ, axis=0)
        u_body = r_body.value/np.linalg.norm(r_body, axis=0)
        
        # Create koangles for all bodies, set by telescope keepout angle for brighter 
        # objects (Sun, Moon, Earth) and defaults to 1 degree for other bodies.
        koangles = np.ones(nBodies)*koangle
        koangles[3:] = 1.*u.deg
        
        # Find angles and make angle comparisons to build kogood array.
        # If bright objects have an angle with the target vector less than koangle 
        # (e.g. pi/4) they are in the field of view and the target star may not be
        # observed, thus ko associated with this target becomes False.
        nStars = sInds.size
        kogood = np.array([True]*nStars)
        culprit = np.zeros([nStars, nBodies])
        for i in xrange(nStars):
            u_b = u_body[:,0,:] if currentTime.size == 1 else u_body[:,i,:]
            angles = np.arccos(np.dot(u_targ[:,i],u_b))
            culprit[i,:] = (angles < koangles.to('rad').value)
            if np.any(culprit[i,:]):
                kogood[i] = False
        
        # check to make sure all elements in kogood are Boolean
        trues = [isinstance(element, np.bool_) for element in kogood]
        assert all(trues), 'An element of kogood is not Boolean'
        
#        return kogood, culprit
        return kogood

