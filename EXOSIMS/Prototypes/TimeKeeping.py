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
        missionLife (astropy Quantity):
            Mission life time in units of year
        extendedLife (astropy Quantity):
            Extended mission time in units of year
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
        waitMultiple (float):
            Multiplier applied to the wait time in case of repeated empty lists of 
            observable targets, which makes the wait time grow exponentially. 
            As soon as an observable target is found, the wait time is reinitialized 
            to the default waitTime value.
        
    """

    _modtype = 'TimeKeeping'
    _outspec = {}

    def __init__(self, missionStart=60634, missionLife=0.1, extendedLife=0, 
            missionPortion=1, OBduration=14, waitTime=1, waitMultiple=2, **specs):
        
        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))
        
        # illegal value checks
        assert missionLife >= 0, "Need missionLife >= 0, got %f"%missionLife
        assert extendedLife >= 0, "Need extendedLife >= 0, got %f"%extendedLife
        # arithmetic on missionPortion fails if it is outside the legal range
        assert missionPortion > 0 and missionPortion <= 1, \
                "Require missionPortion in the interval ]0,1], got %f"%missionPortion
        
        # set up state variables
        # tai scale specified because the default, utc, requires accounting for leap
        # seconds, causing warnings from astropy.time when time-deltas are added
        self.missionStart = Time(float(missionStart), format='mjd', scale='tai')#the absolute date of mission start
        self.missionLife = float(missionLife)*u.year#the total amount of time since mission start that can elapse
        self.extendedLife = float(extendedLife)*u.year#the total amount of time past missionLife that can elapse
        self.missionPortion = float(missionPortion)#the portion of missionLife the instrument can observe for
        
        # set values derived from quantities above
        self.missionFinishNorm = self.missionLife.to('day') + self.extendedLife.to('day')#the total amount of time since mission start that can possibly elapse
        self.missionFinishAbs = self.missionStart + self.missionLife + self.extendedLife#the absolute time the mission can possibly end
        
        # initialize values updated by functions
        self.currentTimeNorm = 0.*u.day#the current amount of time since mission start that has elapsed
        self.currentTimeAbs = self.missionStart#the absolute mission time
        
        # initialize observing block times arrays. #An Observing Block is a segment of time over which observations may take place
        self.OBnumber = 0#the number of the current Observing Block
        self.OBduration = float(OBduration)*u.day#the duration of each Observing Block
        self.OBstartTimes = list()# = [0.]*u.day#[0.]*u.day#an array of Observing Block Start Times defined as time since missionStart
        self.OBstartTimes.append(0.*u.day)
        maxOBduration = self.missionFinishNorm*self.missionPortion#the maximum Observation Block duration limited by mission portion
        self.OBendTimes = list()
        self.OBendTimes.append(min(self.OBduration, maxOBduration).to('day').value*u.day)
        

        # initialize single observation time arrays. #these are the detection observations
        self.ObsNum = 0 #this is the number of detection observations that have occured
        self.ObsStartTimes = []*u.day
        self.ObsEndTimes = []*u.day

        # initialize single observation START and END times
        self.obsStart = 0.*u.day#an array containing the observation Start times
        self.obsEnd = 0.*u.day#an array containing the observation End times
        
        # initialize wait parameters
        self.waitTime = float(waitTime)*u.day#the default amount of time to wait in wait function
        self.waitMultiple = float(waitMultiple)#
        
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
        
        is_over = (self.currentTimeNorm >= self.missionFinishNorm)
        
        return is_over

    def wait(self):
        """DEPRICATEDWaits a certain time in case no target can be observed at current time.
        
        This method is called in the run_sim() method of the SurveySimulation 
        class object. In the prototype version, it simply allocate a temporal block 
        of 1 day.
        
        """
        self.allocate_time(self.waitTime)

    def allocate_time(self, dt):
        r"""Allocate a temporal block of width dt, advancing to the next OB if needed.
        
        Advance the mission time by dt units. If this requires moving into the next OB,
        call the advancetToStartOfNextOB() method of the TimeKeeping class object.
        
        Args:
            dt (astropy Quantity):
                Temporal block allocated in units of day
        
        """
        
        if dt == 0:
            return

        if (self.currentTimeNorm + dt > self.OBendTimes[self.OBnumber]):#the allocationtime would exceed the current allowed OB
            self.advancetToStartOfNextOB()#calculate and advance to the start of the next Observation Block
            self.currentTimeNorm += dt#adds desired dt to start of next OB
            self.currentTimeAbs += dt#adds desired dt to start of next OB
        else:
            self.currentTimeNorm += dt
            self.currentTimeAbs += dt
        
        # #Check if additional time exceeded the current observation block
        # try:#Ensure OBendTimes is not []
        #     self.OBendTimes[self.OBnumber]
        # except:#create OBendTimes
        #     self.advancetToStartOfNextOB()
        # if not (self.mission_is_over() and (self.currentTimeNorm >= self.OBendTimes[self.OBnumber])):
        #     self.advancetToStartOfNextOB()


    def advancetToStartOfNextOB(self):
        """Advances to Start of Next Observation Block
        
        This method is called in the allocate_time() method of the TimeKeeping 
        class object, when the allocated time requires moving outside of the current OB.
        
        If no OB duration was specified, a new Observing Block is created for 
        each observation in the SurveySimulation module. 
        
        """
        self.OBnumber += 1#increase the observation block number
        
        #create times for next observation block
        self.OBstartTimes.append((self.OBnumber-1)*self.OBduration/self.missionPortion)#declares next OB start Time
        #print('OBstartTimes ' + str(self.OBstartTimes[self.OBnumber]))
        #print('OBduration ' + str(self.OBduration))
        #print('missionPortion ' + str(self.missionPortion))
        self.OBendTimes.append(self.OBduration + (self.OBnumber-1)*self.OBduration/self.missionPortion)#sets the end time of the observation block
        #print('OBendTimes ' + str(self.OBendTimes[self.OBnumber]))
        # For the default case called in SurveySimulation, OBendTime is current time
        # Note: the next OB must not happen after mission finish
        
        self.currentTimeNorm = self.OBstartTimes[self.OBnumber]#update currentTimeNorm
        self.currentTimeAbs = self.OBstartTimes[self.OBnumber] + self.missionStart#update currentTimeAbs

        # begin Survey, and loop until mission is finished
        log_begin = 'OB%s:'%(self.OBnumber+1)#prints because this is the beginning of the nesxt observation block
        #self.logger.info(log_begin)
        self.vprint(log_begin)

        # if dt is not None:
        #     self.OBendTimes[self.OBnumber] = self.currentTimeNorm
        #     nextStart = min(self.OBendTimes[self.OBnumber] + nwait*dt, 
        #             self.missionFinishNorm)
        #     nextEnd = self.missionFinishNorm
        # # else, the OB duration is a fixed value
        # else:
        #     dt = self.OBduration
        #     nextStart = min(self.OBendTimes[self.OBnumber] + nwait*dt, 
        #             self.missionFinishNorm)
        #     maxOBduration = (self.missionFinishNorm - nextStart)*self.missionPortion
        #     nextEnd = nextStart + min(dt, maxOBduration)
        
        # # update OB arrays
        # self.OBstartTimes = np.append(self.OBstartTimes.to('day').value, 
        #         nextStart.to('day').value)*u.day
        # self.OBendTimes = np.append(self.OBendTimes.to('day').value, 
        #         nextEnd.to('day').value)*u.day
        # #self.OBnumber += 1#moved up
        
        # # If mission is not over, move to the next OB, and update observation start time
        # self.allocate_time(nextStart - self.currentTimeNorm)
        # if self.mission_is_over():
        #     self.OBstartTimes = self.OBstartTimes[:-1]
        #     self.OBendTimes = self.OBendTimes[:-1]
        #     self.OBnumber -= 1
        # else:
        #     self.obsStart = nextStart
        #     self.vprint('OB%s: previous block was %s long, advancing %s.'%(self.OBnumber+1, 
        #             dt.round(2), (nwait*dt).round(2)))

    def get_tStartNextOB(self):
        """Returns start time of next Observation Block (OB)
        Returns:
            nextObStartTime - the next time the observatory is available for observing
        """
        try:#If a OBstartTimes exists after the current time
            tStartNextOB = min(self.OBstartTimes[self.OBstartTimes>self.currentTimeAbs])#assumes OBstartTimes are absolute
        except:
            nwait = (1 - self.missionPortion)/self.missionPortion#(nwait+1)*self.OBduration = OB block period
            tStartNextOB = self.OBendTimes[self.OBnumber-1] + nwait*self.OBduration#d
        return tStartNextOB

    def get_tEndThisOB(self):
        """
        """
        tEndThisOB = self.OBendTimes[-1]
        return tEndThisOB