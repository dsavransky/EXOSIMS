# -*- coding: utf-8 -*-
import sys
import os
import logging
import inspect
from astropy.time import Time
import astropy.units as u
import numpy as np

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
        missionStart (astropy Time):
            Mission start time in MJD
        missionLife (astropy Quantity):
            Mission life time in units of year
        extendedLife (astropy Quantity):
            Extended mission time in units of year
        missionPortion (float):
            Portion of mission devoted to planet-finding
        dtAlloc (astropy Quantity):
            Default allocated temporal block in units of day
        duration (astropy Quantity):
            Maximum duration of planet-finding operations in units of day
        missionFinishNorm (astropy Quantity):
            Mission finish normalized time in units of day
        missionFinishAbs (astropy Time):
            Mission finish absolute time in MJD
        nextTimeAvail (astropy Quantity):
            Next time available for planet-finding in units of day
        currentTimeNorm (astropy Quantity):
            Current mission time normalized to zero at mission start in units of day
        currentTimeAbs (astropy Time):
            Current absolute mission time in MJD
        
    """

    _modtype = 'TimeKeeping'
    _outspec = {}

    def __init__(self, missionStart=60634., missionLife=6., extendedLife=0.,\
                  missionPortion = 1/6., dtAlloc = 1, intCutoff = 50, **specs):
        
        # illegal value checks
        assert missionLife >= 0, "Need missionLife >= 0, got %f"%missionLife
        assert extendedLife >= 0, "Need extendedLife >= 0, got %f"%extendedLife
        # arithmetic on missionPortion fails if it is outside the legal range
        assert missionPortion > 0 and missionPortion <= 1, \
          "Require missionPortion in the interval ]0,1], got %f"%missionPortion
        
        # set up state variables
        # tai scale specified because the default, utc, requires accounting for leap
        # seconds, causing warnings from astropy.time when time-deltas are added
        self.missionStart = Time(float(missionStart), format='mjd', scale='tai')
        self.missionLife = float(missionLife)*u.year
        self.extendedLife = float(extendedLife)*u.year
        self.missionPortion = float(missionPortion)
        self.dtAlloc = float(dtAlloc)*u.day
        # planet-finding operation duration is equal to Optical System intCutoff
        self.duration = float(intCutoff)*u.day
        
        # set values derived from quantities above
        self.missionFinishNorm = self.missionLife.to('day') + self.extendedLife.to('day')
        self.missionFinishAbs = self.missionStart + self.missionLife + self.extendedLife
        
        # initialize values updated by functions
        self.nextTimeAvail = 0*u.day
        self.currentTimeNorm = 0.*u.day
        self.currentTimeAbs = self.missionStart
        
        # populate outspec
        for att in self.__dict__.keys():
            dat = self.__dict__[att]
            self._outspec[att] = dat.value if isinstance(dat,(u.Quantity,Time)) else dat

    def __str__(self):
        r"""String representation of the TimeKeeping object.
        
        When the command 'print' is used on the TimeKeeping object, this 
        method prints the values contained in the object."""
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
        return 'TimeKeeping instance at %.6f days' % self.currentTimeNorm.to('day').value

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
            dt (astropy Quantity):
                Amount of time requested in units of day
                
        Returns:
            success (boolean):
                True if the requested time fits in the widest window, otherwise False.
        """
        
        # get caller info
        _,filename,line_number,function_name,_,_ = inspect.stack()[1]
        location = '%s:%d(%s)' % (os.path.basename(filename), line_number, function_name)
        # if no issues, we will advance to this time
        provisional_time = self.currentTimeNorm + dt
        window_advance = False
        success = True
        if dt > self.duration:
            success = False
            description = '!too long'
        elif dt < 0:
            success = False
            description = '!negative allocation'
        elif provisional_time > self.nextTimeAvail + self.duration:
            # advance to the next observation window:
            #   add "duration" (time for our instrument's observations)
            #   also add a term for other observations based on fraction-available
            self.nextTimeAvail += (self.duration + \
                    ((1.0 - self.missionPortion)/self.missionPortion) * self.duration)
            # set current time to dt units beyond start of next window
            self.currentTimeNorm = self.nextTimeAvail + dt
            self.currentTimeAbs = self.missionStart + self.currentTimeNorm
            window_advance = True
            description = '+window'
        else:
            # simply advance by dt
            self.currentTimeNorm = provisional_time
            self.currentTimeAbs += dt
            description = 'ok'
        # Log a message for the time allocation
        message = "TK [%s]: alloc: %.2f day\t[%s]\t[%s]" % (
            self.currentTimeNorm.to('day').value, dt.to('day').value, description, location)
        Logger.info(message)
        
        return success

    def mission_is_over(self):
        r"""Is the time allocated for the mission used up?
        
        This supplies an abstraction around the test:
            (currentTimeNorm > missionFinishNorm)
        so that users of the class do not have to perform arithmetic
        on class variables.
        
        Returns:
            is_over (Boolean):
                True if the mission time is used up, else False.
        """
        
        is_over = (self.currentTimeNorm > self.missionFinishNorm)
        
        return is_over

