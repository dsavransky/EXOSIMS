# -*- coding: utf-8 -*-
import sys
import os
import logging
import inspect
from astropy.time import Time
import astropy.units as u

# the EXOSIMS logger
Logger = logging.getLogger(__name__)

class TimeKeeping(object):
    """TimeKeeping class template.
    
    This class keeps track of the current mission elapsed time
    for exoplanet mission simulation.  It is initialized with a
    mission duration, and throughout the simulation, it allocates
    temporal intervals for observations.  Eventually, all available
    time has been allocated, and the mission is over.
    Time is allocated in contiguous windows of size "duration".  If a
    requested interval does not fit in the current window, we move to
    the next one.
    
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
    _outspec = {}
    
    def __init__(self, missionStart=60634., missionLife=6.,\
                 extendedLife=0., missionPortion = 1./6., **specs):
                
        # default values
        # mission start time (astropy Time object) in mjd
        self.missionStart = Time(float(missionStart), format='mjd')
        self._outspec['missionStart'] = float(missionStart)

        # mission lifetime astropy unit object in years
        self.missionLife = float(missionLife)*u.year
        self._outspec['missionLife'] = float(missionLife)

        # extended mission time astropy unit object in years
        self.extendedLife = float(extendedLife)*u.year
        self._outspec['extendedLife'] = float(extendedLife)

        # portion of mission devoted to planet-finding
        self.missionPortion = float(missionPortion)
        self._outspec['missionPortion'] = float(missionPortion)
        
        # duration of planet-finding operations
        self.duration = 14.*u.day
        # next time available for planet-finding
        self.nexttimeAvail = 0.*u.day
                            
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
        r"""String representation of the TimeKeeping object.
        
        When the command 'print' is used on the TimeKeeping object, this 
        method prints the values contained in the object."""

        atts = self.__dict__.keys()
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        return 'TimeKeeping instance at %.6f days' % self.currenttimeNorm.to(u.day).value
        
    def allocate_time(self, dt):
        r"""Allocate a temporal block of width dt, advancing the observation window if needed.
        
        Advance the mission time by dt units.  If this requires moving into the next observation
        window, do so.
        If dt is longer than the observation window length, making a contiguous observation is
        not possible, so return False.  If dt < 0, return False.  Otherwise, allocate time and
        return True.

        Caveats:
        [1] This does not handle allocations that move beyond the allowed mission time.  This
        would be a place for an exception that could be caught in the simulation main loop.
        For now, we check for this condition at the top of the simulation loop and not here.
        
        Args:
            dt (Quantity):
                amount of time requested (units of time)
                
        Returns:
            success (Boolean):
                True if the requested time fits in the widest window, otherwise False.
        """
        
        # get caller info
        _,filename,line_number,function_name,_,_ = inspect.stack()[1]
        location = '%s:%d(%s)' % (os.path.basename(filename), line_number, function_name)
        # if no issues, we will advance to this time
        provisional_time = self.currenttimeNorm + dt
        window_advance = False
        success = True
        if dt > self.duration:
            success = False
            description = '!too long'
        elif dt < 0:
            success = False
            description = '!negative allocation'
        elif provisional_time > self.nexttimeAvail + self.duration:
            # advance to the next observation window:
            #   add "duration" (time for our instrument's observations)
            #   also add a term for other observations based on fraction-available
            self.nexttimeAvail += (self.duration +
                                   ((1.0 - self.missionPortion)/self.missionPortion) * self.duration)
            # set current time to dt units beyond start of next window
            self.currenttimeNorm = self.nexttimeAvail + dt
            self.currenttimeAbs = self.missionStart + self.currenttimeNorm
            window_advance = True
            description = '+window'
        else:
            # simply advance by dt
            self.currenttimeNorm = provisional_time
            self.currenttimeAbs += dt
            description = 'ok'
        # Log a message for the time allocation
        message = "TK [%s]: alloc: %.2f day\t[%s]\t[%s]" % (
            self.currenttimeNorm.to(u.day).value, dt.to(u.day).value, description, location)
        Logger.info(message)
        # if False: print '***', message
        return success

    def mission_is_over(self):
        r"""Is the time allocated for the mission used up?

        This supplies an abstraction around the test:
            (currenttimeNorm > missionFinishNorm)
        so that users of the class do not have to perform arithmetic
        on class variables.

        Args:
            None

        Returns:
            is_over (Boolean):
                True if the mission time is used up, else False.
        """
        return (self.currenttimeNorm > self.missionFinishNorm)

    def update_times(self, dt):
        """Updates self.currenttimeNorm and self.currenttimeAbs
        
        Deprecated.
        
        Args:
            dt (Quantity):
                time increment (units of time)
        
        """
        if dt < 0:
            raise ValueError('update_times got negative dt: %s' % str(dt))
        self.currenttimeNorm += dt
        self.currenttimeAbs += dt
        
    def duty_cycle(self, currenttime):
        """Updates available time and duration for planet-finding.
        
        Deprecated.
        
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
        
