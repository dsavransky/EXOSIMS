# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.Observatory import Observatory
import astropy.units as u
import numpy as np

class WFIRSTObservatory(Observatory):
    """WFIRST Observatory class
    
    This class contains all variables and methods specific to the WFIRST
    observatory needed to perform Observatory Definition Module calculations
    in exoplanet mission simulation."""
    
    def orbit(self, time):
        """Finds WFIRST geosynchronous circular orbit position vector
        
        This method finds the WFIRST geosynchronous circular orbit position 
        vector as 1D numpy array (astropy Quantity with units of km) in the
        heliocentric equatorial frame, stores this vector in self.r_sc,
        and returns True if successful.
        
        Args:
            time (Time):
                current absolute time (astropy Time)
            
        Returns:
            success (bool):
                True if successful, False if not
        
        """
        
        pi = np.pi
        i = 28.5 # orbital inclination in degrees
        i = np.radians(i)
        O = 228. # right ascension of the ascending node in degrees
        O = np.radians(O)
        # orbital period is one sidereal day
        f = 2.*pi # orbital frequency (2*pi/period)
        r = 42164.*u.km # orbital height in km
        
        # Find Earth position vector (km in heliocentric equatorial frame)
        r_earth = self.solarSystem_body_position(time, 'Earth')
        
        # Find spacecraft position vector with respect to Earth in orbital plane
        t = time.mjd - int(time.mjd) # gives percent of day
        r_scearth = r*np.array([np.cos(f*t), np.sin(f*t), 0.])
        
        # Find spacecraft position vector with respect to Earth in equatorial frame
        r_scearth = np.dot(np.dot(self.rot(-O,3), self.rot(-i,1)),r_scearth)
        
        # Find spacecraft position vector with respect to sun (heliocentric equatorial)
        self.r_sc = r_earth + r_scearth
        
        b = np.isfinite(self.r_sc) # finds if all values are finite floats
        success = all(b) # returns True if all values of self.r_sc are finite
        
        return success
        
    def keepout(self, time, catalog, koangle):
        """Finds keepout Boolean values, returns True if successful
        
        This method finds the keepout Boolean values for each target star where
        True is an observable star and stores the 1D numpy array in self.kogood.
        
        Args:
            time (Time):
                absolute time (astropy Time)
            catalog (TargetList or StarCatalog):
                TargetList or StarCatalog class object
            koangle (float):
                Telescope keepout angle in degrees
                
        Returns:
            success (bool):
                True if successful, False if not
        
        """
        
        # get updated orbital position vector
        a = self.orbit(time)
        
        # Find position and unit vectors for list of stars in catalog wrt spacecraft
        r_targ = [np.zeros((1,3))]*len(catalog.Name) # initialize list of position vectors
        u_targ = r_targ # initialize list of unit vectors
        for x in xrange(len(catalog.Name)):
            r_targ[x] = self.starprop(time, catalog, x) # position vector wrt sun
            r_targ[x] -= self.r_sc # position vector wrt spacecraft
            u_targ[x] = r_targ[x]/np.linalg.norm(r_targ[x]) # unit vector wrt spacecraft

        # list of bright object position vectors wrt sun
        r_bright = [-self.r_sc, # sun
                    self.solarSystem_body_position(time, 'Mercury'), # Mercury
                    self.solarSystem_body_position(time, 'Venus'), # Venus
                    self.solarSystem_body_position(time, 'Earth'), # Earth
                    self.solarSystem_body_position(time, 'Mars'), # Mars
                    self.solarSystem_body_position(time, 'Jupiter'), # Jupiter
                    self.solarSystem_body_position(time, 'Saturn'), # Saturn
                    self.solarSystem_body_position(time, 'Uranus'), # Uranus
                    self.solarSystem_body_position(time, 'Neptune'), # Neptune
                    self.solarSystem_body_position(time, 'Pluto'), # Pluto
                    self.solarSystem_body_position(time, 'Moon')] # moon
        u_bright = r_bright # initialize list of unit vectors

        # Find position and unit vectors for bright objects wrt spacecraft    
        for x in xrange(len(r_bright)):
            r_bright[x] -= self.r_sc # position vector wrt spacecraft
            u_bright[x] = r_bright[x]/np.linalg.norm(r_bright[x]) # unit vector
            
        # Find angles and make angle comparisons for self.kogood
        # if bright objects have an angle with the target vector less than 
        # pi/4 they are in the field of view and the target star may not be
        # observed, thus ko associated with this target becomes False
        self.kogood = np.array([True]*len(catalog.Name))
        for i in xrange(len(catalog.Name)):
            for j in xrange(len(u_bright)):
                angle = np.arccos(np.dot(u_targ[i], u_bright[j]))
                if angle < np.radians(koangle):
                    self.kogood[i] = False
                    break
        
        # check to make sure all elements in self.kogood are Boolean
        b = [isinstance(element, np.bool_) for element in self.kogood]  
        c = [a, b]
        # return True if orbital position is successful and all elements of
        # self.kogood are Boolean
        success = all(c)
        
        return success
