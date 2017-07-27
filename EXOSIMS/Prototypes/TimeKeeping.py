import numpy as np
import astropy.units as u
from astropy.time import Time

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

    def __init__(self, missionStart=60634, missionLife=0.1, extendedLife=0, 
            missionPortion=1, OBduration=np.inf, **specs):
        
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
        
        # set values derived from quantities above
        self.missionFinishNorm = self.missionLife.to('day') + self.extendedLife.to('day')
        self.missionFinishAbs = self.missionStart + self.missionLife + self.extendedLife
        
        # initialize values updated by functions
        self.currentTimeNorm = 0.*u.day
        self.currentTimeAbs = self.missionStart
        
        # initialize observing block times arrays
        self.OBnumber = 0
        self.OBduration = float(OBduration)*u.day
        self.OBstartTimes = [0.]*u.day
        maxOBduration = self.missionFinishNorm*self.missionPortion
        self.OBendTimes = [min(self.OBduration, maxOBduration).to('d').value]*u.d
        
        # initialize single observation START and END times
        self.obsStart = 0.*u.day
        self.obsEnd = 0.*u.day
        
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
        
        is_over = (self.currentTimeNorm >= self.missionFinishNorm)
        
        return is_over

    def wait(self):
        """Waits a certain time in case no target can be observed at current time.
        
        This method is called in the run_sim() method of the SurveySimulation 
        class object. In the prototype version, it simply allocate a temporal block 
        of 1 day.
        
        """
        dt = 1.*u.day
        self.allocate_time(dt)
        #print '  Waiting %s'%dt.round(2)

    def allocate_time(self, dt):
        r"""Allocate a temporal block of width dt, advancing to the next OB if needed.
        
        Advance the mission time by dt units. If this requires moving into the next OB,
        call the next_observing_block() method of the TimeKeeping class object.
        
        Args:
            dt (astropy Quantity):
                Temporal block allocated in units of day
        
        """
        
        if dt == 0:
            return
            
        self.currentTimeNorm += dt
        self.currentTimeAbs += dt
        
        if not self.mission_is_over() and (self.currentTimeNorm 
                >= self.OBendTimes[self.OBnumber]):
            self.next_observing_block()

    def next_observing_block(self, dt=None):
        """Defines the next observing block, start and end times
        
        This method is called in the allocate_time() method of the TimeKeeping 
        class object, when the allocated time requires moving outside of the current OB.
        
        If no OB duration was specified, this method is only called in the run_sim() 
        method of the SurveySimulation class object, where a temporal block dt will
        be passed, equivalent to the time spent on one target (i.e., one observation 
        per OB).
        
        """
        
        # number of blocks to wait
        nwait = (1 - self.missionPortion)/self.missionPortion
        
        # For the default case called in SurveySimulation, OBendTime is current time
        # Note: the next OB must not happen after mission finish
        if dt is not None:
            self.OBendTimes[self.OBnumber] = self.currentTimeNorm
            nextStart = min(self.OBendTimes[self.OBnumber] + nwait*dt, 
                    self.missionFinishNorm)
            nextEnd = self.missionFinishNorm
        # else, the OB duration is a fixed value
        else:
            dt = self.OBduration
            nextStart = min(self.OBendTimes[self.OBnumber] + nwait*dt, 
                    self.missionFinishNorm)
            maxOBduration = (self.missionFinishNorm - nextStart)*self.missionPortion
            nextEnd = nextStart + min(dt, maxOBduration)
        
        # update OB arrays
        self.OBstartTimes = np.append(self.OBstartTimes.to('day').value, 
                nextStart.to('day').value)*u.day
        self.OBendTimes = np.append(self.OBendTimes.to('day').value, 
                nextEnd.to('day').value)*u.day
        self.OBnumber += 1
        
        # If mission is not over, move to the next OB, and update observation start time
        self.allocate_time(nextStart - self.currentTimeNorm)
        if self.mission_is_over():
            self.OBstartTimes = self.OBstartTimes[:-1]
            self.OBendTimes = self.OBendTimes[:-1]
            self.OBnumber -= 1
        else:
            self.obsStart = nextStart
            print 'OB%s: previous block was %s long, advancing %s.'%(self.OBnumber+1, 
                    dt.round(2), (nwait*dt).round(2))
