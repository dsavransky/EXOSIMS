# -*- coding: utf-8 -*-
from astropy.time import Time
import astropy.units as u

class TimeKeeping(object):
    """Time class template
    
    This class contains all variables and functions necessary to perform 
    Time Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs:
            user specified values
            
    Attributes:
        missionStart (Time):
            mission start time (default astropy Time in MJD)
        missionLife (Quantity):
            mission lifetime (default units of year)
        extendedLife (Quantity):
            extended mission time (default units of year)
        currenttimeNorm (Quantity):
            current mission time normalized to zero at mission start (default
            units of day)
        currenttimeAbs (Time):
            current absolute mission time (default astropy Time in MJD)
        missionFinishAbs (Time):
            mission finish absolute time (default astropy Time in MJD)
        missionFinishNorm (Quantity):
            mission finish normalized time (default units of day)
        missionPortion (float):
            portion of mission devoted to planet-finding
        duration (Quantity):
            duration of planet-finding operations (default units of day)
        nexttimeAvail (Quantity):
            next time available for planet-finding (default units of day)
        
    """

    _modtype = 'TimeKeeping'
    
    def __init__(self, **specs):
                
        # default values
        # mission start time (astropy Time object) in mjd
        self.missionStart = Time(60634., format='mjd')
        # mission lifetime astropy unit object in years
        self.missionLife = 6.*u.year
        # extended mission time astropy unit object in years
        self.extendedLife = 0.*u.year
        # portion of mission devoted to planet-finding
        self.missionPortion = 1./6.
        # duration of planet-finding operations
        self.duration = 14.*u.day
        # next time available for planet-finding
        self.nexttimeAvail = 0.*u.day
        
        # replace default values with user specified values
        atts = self.__dict__.keys()
        for att in atts:
            if att in specs:
                if att == 'missionStart':
                    self.missionStart = Time(specs[att], format='mjd')
                elif att == 'missionLife' or att == 'extendedLife':
                    self.missionLife = getattr(self, att, specs[att]*u.year)
                elif att == 'duration' or att == 'nexttimeAvail':
                    setattr(self, att, specs[att]*u.day)
                else:
                    setattr(self, att, specs[att])
                    
        # initialize values updated by functions
        # current mission time astropy unit object in days
        self.currenttimeNorm = 0.*u.day
        # current absolute mission time (astropy Time object) in mjd
        self.currenttimeAbs = self.missionStart
        
        # set values derived from quantities above
        # mission completion date (astropy Time object) in mjd
        self.missionFinishAbs = self.missionStart + self.missionLife + self.extendedLife
        # normalized mission completion date in days
        self.missionFinishNorm = self.missionLife.to(u.day) + self.extendedLife.to(u.day)
        
    
    def __str__(self):
        """String representation of the Time Keeping object
        
        When the command 'print' is used on the Time Keeping object, this 
        method will return the values contained in the object"""

        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Time Keeping class object attributes'
        
    def update_times(self, dt):
        """Updates self.currenttimeNorm and self.currenttimeAbs
        
        Args:
            dt (Quantity):
                time increment (units of time)
        
        """
        
        self.currenttimeNorm += dt
        self.currenttimeAbs += dt
        
    def duty_cycle(self, currenttime):
        """Updates available time and duration for planet-finding
        
        This method updates the available time for planet-finding activities.
        Specific classes may optionally update the duration of planet-finding
        activities as well. This method defines the input/output expected
        
        Args:
            currenttime (Quantity):
                current time in mission simulation (units of time)
                
        Returns:
            nexttime (Quantity):
                next available time for planet-finding (units of time)
        
        """
        
        if currenttime > self.nexttimeAvail + self.duration:
            # update the nexttimeAvail attribute
            self.nexttimeAvail += self.duration + (1. - self.missionPortion)/self.missionPortion*self.duration
            nexttime = self.nexttimeAvail
        else:
            nexttime = currenttime
        
        return nexttime
        
        
