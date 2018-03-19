from EXOSIMS.util.vprint import vprint
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
        missionPortion (float):
            Portion of mission devoted to planet-finding
        missionFinishNorm (astropy Quantity):
            Mission finish normalized time in units of day
        missionFinishAbs (astropy Time):
            Mission finish absolute time in MJD
        currentTimeNorm (astropy Quantity):
            Current mission time normalized to zero at mission start in units of day
        currentTimeAbs (astropy Time):
            Current absolute mission time in MJD
        OBnumber (integer):
            Index/number associated with the current observing block (OB). Each 
            observing block has a duration, a start time, an end time, and can 
            host one or multiple observations.
        OBduration (astropy Quantity):
            Default allocated duration of observing blocks, in units of day. If 
            no OBduration was specified, a new observing block is created for 
            each new observation in the SurveySimulation module. 
        OBstartTimes (astropy Quantity array):
            Array containing the normalized start times of each observing block 
            throughout the mission, in units of day
        OBendTimes (astropy Quantity array):
            Array containing the normalized end times of each observing block 
            throughout the mission, in units of day
        obsStart (astropy Quantity):
            Normalized start time of the observation currently executed by the 
            Survey Simulation, in units of day
        obsEnd (astropy Quantity):
            Normalized end time of the observation currently executed by the 
            Survey Simulation, in units of day
        waitTime (astropy Quantity):
            Default allocated duration to wait in units of day, when the
            Survey Simulation does not find any observable target
        
    """

    _modtype = 'TimeKeeping'
    _outspec = {}

    def __init__(self, missionStart=60634, missionFinishNorm=0.1, 
            missionPortion=1, OBduration=np.inf, **specs):
        
        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))
        
        # illegal value checks
        assert missionFinishNorm >= 0, "Need missionFinishNorm >= 0, got %f"%missionFinishNorm
        # arithmetic on missionPortion fails if it is outside the legal range
        assert missionPortion > 0 and missionPortion <= 1, \
                "Require missionPortion in the interval ]0,1], got %f"%missionPortion
        
        # set up state variables
        # tai scale specified because the default, utc, requires accounting for leap
        # seconds, causing warnings from astropy.time when time-deltas are added
        self.missionStart = Time(float(missionStart), format='mjd', scale='tai')#the absolute date of mission start
        self.missionPortion = float(missionPortion)#the portion of missionFinishNorm the instrument can observe for
        
        # set values derived from quantities above
        self.missionFinishNorm = (float(missionFinishNorm)*u.year).to('day')#the total amount of time since mission start that can elapse
        self.missionFinishAbs = self.missionStart + self.missionFinishNorm#the absolute time the mission can possibly end
        
        # initialize values updated by functions
        self.currentTimeNorm = 0.*u.day#the current amount of time since mission start that has elapsed
        self.currentTimeAbs = self.missionStart#the absolute mission time
        
        # initialize observing block times arrays. #An Observing Block is a segment of time over which observations may take place
        self.OBnumber = -1#the number of the current Observing Block
        self.OBduration = float(OBduration)*u.day#the duration of each Observing Block
        self.OBstartTimes = list()# = [0.]*u.day#[0.]*u.day#an array of Observing Block Start Times defined as time since missionStart
        #self.OBstartTimes.append(0.*u.day)
        maxOBduration = self.missionFinishNorm*self.missionPortion#the maximum Observation Block duration limited by mission portion
        self.OBendTimes = list()
        #self.OBendTimes.append(min(self.OBduration, maxOBduration).to('day').value*u.day)
        self.advancetToStartOfNextOB()

        # initialize single detection observation time arrays. #these are the detection observations
        self.ObsNum = 0 #this is the number of detection observations that have occured
        self.ObsStartTimes = list()#*u.day
        self.ObsEndTimes = list()#*u.day
        
        # initialize time spend using instrument
        self.exoplanetObsTime = 0*u.day
        
        # populate outspec
        for att in self.__dict__.keys():
            if att not in ['vprint']:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat,(u.Quantity,Time)) else dat

    def __str__(self):
        r"""String representation of the TimeKeeping object.
        
        When the command 'print' is used on the TimeKeeping object, this 
        method prints the values contained in the object."""
        
        for att in self.__dict__.keys():
            print('%s: %r' % (att, getattr(self, att)))
        
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
        
        is_over = ((self.currentTimeNorm >= self.missionFinishNorm) or (self.exoplanetObsTime.to('day') >= self.missionFinishNorm.to('day')*self.missionPortion))
        
        return is_over

    def allocate_time(self, t, flag=1):
        r"""Allocate a temporal block of width t, advancing to the next OB if needed OR advance to absolute time t
        
        Advance the mission time by dt units. If this requires moving into the next OB,
        call the advancetToStartOfNextOB() method of the TimeKeeping class object.
        
        Args:
            t (astropy Quantity OR unitless OR Time Quantity):
                Temporal block allocated in units of day OR assumed to be day OR absolute time to advance to
            flag (integer):
                Indicates the allocated time is for the primary instrument (1) or some other instrument (0)
                By default this function assumes all allocated time is attributed to the primary instrument
        
        """
        if(type(t) == type(self.currentTimeAbs)):#t given as Time. Advance to TimeAbs t or start of next OB
            assert(t.value > self.currentTimeAbs.value, 'allocate_time tAbs must be greater than currentTimeAbs')
            #Check no OBendTimes occur between now and time to advance to
            while(not self.currentTimeAbs.value < t.value):#continue until the current mission time is greater than or equal to the set point time 
                tSkipped = 0
                if(self.OBendTimes[self.OBnumber].value + self.missionStart.value <= t.to('day').value):#Check if next OBendTime occurs between now and t
                    tSkipped = self.OBendTimes[self.OBnumber]  - self.currentTimeNorm#time between now and the end of the OB
                    self.advancetToStartOfNextOB()#calculate and advance to the start of the next Observation Block
                else:#No OBendTimes occur between now and setpoint
                    tSkipped = (t - self.currentTimeAbs).value*u.d #time advanced
                    self.currentTimeNorm = (t - self.missionStart).value*u.d#advances to t
                    self.currentTimeAbs = t#advances absolute time to t
                if(flag == 1):#adds OB time skipped over to exoplanetObsTime
                    self.exoplanetObsTime += tSkipped#increments allocated time 
        else:
            if(type(t) == type(1*u.d)):#t is a dt to be added to the current mission time in units of day
                dt = t
            else:#assume unitless t given in units of dat
                dt = t*u.d
            assert(dt > 0*u.d, 'allocated_time dt must be positive nonzero')
            assert(dt < self.OBduration, 'allocated_time dt must be less than OBduration')
            #Check adding t would exceed CURRENT OBendTime
            if (self.currentTimeNorm + dt > self.OBendTimes[self.OBnumber]):#the allocationtime would exceed the current allowed OB
                tSkipped = self.OBendTimes[self.OBnumber]-self.currentTimeNorm#We add the time at the end of the OB skipped
                self.advancetToStartOfNextOB()#calculate and advance to the start of the next Observation Block
                self.currentTimeNorm += dt#adds desired dt to start of next OB
                self.currentTimeAbs += dt#adds desired dt to start of next OB

                if(flag == 1):
                    self.exoplanetObsTime += dt + tSkipped#increments allocated time 
            else:#Operation as normal
                self.currentTimeNorm += dt
                self.currentTimeAbs += dt
                if(flag == 1):
                    self.exoplanetObsTime += dt#increments allocated time      

    def advancetToStartOfNextOB(self):
        """Advances to Start of Next Observation Block
        
        This method is called in the allocate_time() method of the TimeKeeping 
        class object, when the allocated time requires moving outside of the current OB.
        
        If no OB duration was specified, a new Observing Block is created for 
        each observation in the SurveySimulation module. 
        
        """
        self.OBnumber += 1#increase the observation block number
        
        #create times for next observation block
        self.OBstartTimes.append((self.OBnumber)*self.OBduration/self.missionPortion)#declares next OB start Time
        self.OBendTimes.append(self.OBduration + (self.OBnumber)*self.OBduration/self.missionPortion)#sets the end time of the observation block
        # Note: the next OB must not happen after mission finish
        
        self.currentTimeNorm = self.OBstartTimes[self.OBnumber]#update currentTimeNorm
        self.currentTimeAbs = self.OBstartTimes[self.OBnumber] + self.missionStart#update currentTimeAbs

        # begin Survey, and loop until mission is finished
        log_begin = 'OB%s:'%(self.OBnumber+1)#prints because this is the beginning of the nesxt observation block
        self.vprint(log_begin)

    def get_tStartNextOB(self):
        """Returns start time of next Observation Block (OB)
        Returns:
            nextObStartTime (float astropy Quantity) - the next time the observatory is available for observing ABSOLUTE
        """
        try:#If a OBstartTimes exists after the current time
            tStartNextOB = min(self.OBstartTimes[self.OBstartTimes>self.currentTimeAbs])#assumes OBstartTimes are absolute
        except:
            nwait = (1 - self.missionPortion)/self.missionPortion#(nwait+1)*self.OBduration = OB block period
            tStartNextOB = self.OBendTimes[self.OBnumber-1] + nwait*self.OBduration#d
        return tStartNextOB

    def get_tEndThisOB(self):
        """Retrieves the End Time of this OB

        Returns:
            tEndThisOB (float astropy Quantity) - the end time of the current observation block
        """
        tEndThisOB = self.OBendTimes[-1]
        return tEndThisOB

    def get_ObsDetectionMaxIntTime(self,Obs,mode):
        """Tells you the maximum Detection Observation Integration Time you can pass into observation_detection(X,intTime,X)
        Args:
            mode (dict):
                Selected observing mode for detection
        Returns:
            maxIntTimeOBendTime (astropy Quantity):
                The maximum integration time bounded by Observation Block end Time
            maxIntTimeExoplanetObsTime (astropy Quantity):
                The maximum integration time bounded by exoplanetObsTime
            maxIntTimeMissionLife (astropy Quantity):
                The maximum integration time bounded by MissionLife
        """
        maxTimeOBendTime = self.get_tEndThisOB() - self.currentTimeNorm
        maxIntTimeOBendTime = (maxTimeOBendTime - Obs.settlingTime - mode['syst']['ohTime'])/(1 + mode['timeMultiplier'] -1)

        maxTimeExoplanetObsTime = self.missionLife - self.exoplanetObsTime
        maxIntTimeExoplanetObsTime = (maxTimeExoplanetObsTime - Obs.settlingTime - mode['syst']['ohTime'])/(1 + mode['timeMultiplier'] -1)
        
        maxTimeMissionLife = self.missionLife - self.currentTimeNorm
        maxIntTimeMissionLife = (maxTimeMissionLife - Obs.settlingTime - mode['syst']['ohTime'])/(1 + mode['timeMultiplier'] -1)

        return maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife

