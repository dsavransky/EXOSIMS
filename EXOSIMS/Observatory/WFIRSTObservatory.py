# -*- coding: utf-8 -*-
from __future__ import division
from builtins import range
from past.utils import old_div
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
        r_earth = self.solarSystem_body_position(currentTime, 'Earth').T
        
        # Find spacecraft position vector with respect to Earth in orbital plane
        t = currentTime.mjd - np.floor(currentTime.mjd) # gives percent of day
        r_scearth = r*np.array([np.cos(f*t), np.sin(f*t), np.zeros(t.size)]).T
        
        # Find spacecraft position vector with respect to Earth in equatorial frame
        r_scearth = np.dot(np.dot(self.rot(-O,3), self.rot(-i,1)),r_scearth.T).T
        
        # Find spacecraft position vector with respect to sun (heliocentric equatorial)
        r_sc = r_earth + r_scearth
        
        assert np.all(np.isfinite(r_sc)), 'Observatory position vector r_sc has infinite value.'
        
        return r_sc.to('km').reshape(currentTime.size,3)

    def keepout(self, TL, sInds, currentTime, r_sc, koangle):
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
            r_sc (astropy Quantity nx3 array):
                Current observatory (spacecraft) position vector in units of km
            koangle (astropy Quantity):
                Telescope keepout angle in units of degree
                
        Returns:
            kogood (boolean ndarray):
                True is a target unobstructed and observable, and False is a 
                target unobservable due to obstructions in the keepout zone.
        
        Note1: currentTime and r_sc must be of same size.
        Note2: For multiple targets, currentTime/r_sc must be of size 1 or size of sInds.
        
        """
        
        # reshape sInds
        sInds = np.array(sInds,ndmin=1)
        # reshape currentTime
        currentTime = currentTime.reshape(currentTime.size)
        # check size of currentTime and r_sc
        assert currentTime.size == r_sc.shape[0], 'CurrentTime and r_sc must be of same size.'
        # check size of sInds
        if (sInds.size > 1) & (currentTime.size > 1):
            assert currentTime.size == sInds.size, \
                    'For multiple targets, currentTime/r_sc must be of size 1 or size of sInds'
        
        # First, find unit vectors wrt spacecraft for stars of interest
        # position vectors wrt sun
        r_targ = self.starprop(TL, sInds, currentTime) 
        # position vectors wrt spacecraft
        r_targ -= r_sc
        # unit vectors wrt spacecraft
        u_targ = (old_div(r_targ.value.T,np.linalg.norm(r_targ, axis=-1))).T
        
        # Second, find unit vectors wrt spacecraft for bright objects
        # position vectors wrt sun
        r_bright = np.array([np.zeros(r_sc.shape), # sun
            self.solarSystem_body_position(currentTime, 'Moon').T.to('km').value,
            self.solarSystem_body_position(currentTime, 'Earth').T.to('km').value,
            self.solarSystem_body_position(currentTime, 'Mercury').T.to('km').value,
            self.solarSystem_body_position(currentTime, 'Venus').T.to('km').value,
            self.solarSystem_body_position(currentTime, 'Mars').T.to('km').value,
            self.solarSystem_body_position(currentTime, 'Jupiter').T.to('km').value,
            self.solarSystem_body_position(currentTime, 'Saturn').T.to('km').value,
            self.solarSystem_body_position(currentTime, 'Uranus').T.to('km').value,
            self.solarSystem_body_position(currentTime, 'Neptune').T.to('km').value,
            self.solarSystem_body_position(currentTime, 'Pluto').T.to('km').value])*u.km
        # position vectors wrt spacecraft
        r_bright -= r_sc
        # unit vectors wrt spacecraft
        u_bright = (old_div(r_bright.value.T,np.linalg.norm(r_bright, axis=-1).T)).T
        
        # Create koangles for all bodies. By default other planets have 
        # half the koangle of Sun, Moon, and Earth.
        nBodies = u_bright.shape[0]
        koangles = np.ones(nBodies)*koangle
        koangles[3:] = 1*u.deg
        
        # Find angles and make angle comparisons for kogood
        # if bright objects have an angle with the target vector less than koangle 
        # (e.g. pi/4) they are in the field of view and the target star may not be
        # observed, thus ko associated with this target becomes False
        kogood = np.array([True]*len(u_targ))
        culprit = np.zeros([len(u_targ), nBodies])
        for i in range(len(u_targ)):
            u_b = u_bright[:,0,:] if currentTime.size == 1 else u_bright[:,i,:]
            angles = np.arccos(np.dot(u_b, u_targ[i]))
            culprit[i,:] = (angles < koangles.to('rad').value)
            if np.any(culprit[i,:]):
                kogood[i] = False
        
        # check to make sure all elements in kogood are Boolean
        trues = [isinstance(element, np.bool_) for element in kogood]
        assert all(trues), 'An element of kogood is not Boolean'
        
#        return kogood, culprit
        return kogood

