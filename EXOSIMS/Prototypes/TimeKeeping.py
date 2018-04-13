from EXOSIMS.util.vprint import vprint
import numpy as np
import astropy.units as u
from astropy.time import Time
import os, sys
import csv
from numpy import nan

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
        missionLife (astropy Quantity):
            Mission Life and mission finish normalized time in units of day
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
        
    """

    _modtype = 'TimeKeeping'
    _outspec = {}

    def __init__(self, missionStart=60634, missionLife=0.1, 
        missionPortion=1, OBduration=np.inf, missionSchedule=None, **specs):

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))
        
        # illegal value checks
        assert missionLife >= 0, "Need missionLife >= 0, got %f"%missionLife
        # arithmetic on missionPortion fails if it is outside the legal range
        assert missionPortion > 0 and missionPortion <= 1, \
                "Require missionPortion in the interval [0,1], got %f"%missionPortion
        # OBduration must be positive nonzero
        assert OBduration*u.d > 0*u.d, "Required OBduration positive nonzero, got %f"%OBduration
        
        # set up state variables
        # tai scale specified because the default, utc, requires accounting for leap
        # seconds, causing warnings from astropy.time when time-deltas are added
        self.missionStart = Time(float(missionStart), format='mjd', scale='tai')#the absolute date of mission start must have scale tai
        self.missionPortion = float(missionPortion)#the portion of missionFinishNorm the instrument can observe for
        
        # set values derived from quantities above
        self.missionLife = (float(missionLife)*u.year).to('day')#the total amount of time since mission start that can elapse
        self.missionFinishAbs = self.missionStart + self.missionLife#the absolute time the mission can possibly end
        
        # initialize values updated by functions
        self.currentTimeNorm = 0.*u.day#the current amount of time since mission start that has elapsed
        self.currentTimeAbs = self.missionStart#the absolute mission time
        
        # initialize observing block times arrays. #An Observing Block is a segment of time over which observations may take place
        self.init_OB(str(missionSchedule), OBduration*u.d)
        
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

    def init_OB(self, missionSchedule, OBduration):
        """ Initializes mission Observing Blocks from file or missionDuration, missionLife, and missionPortion
        Args:
            missionSchedule (string):
                a string containing the missionSchedule file
            OBduration (astropy Quantity):
                the duration of a single observing block
        Updates Attributes:
            OBstartTimes (astropy Quantity array):
                Updates the start times of observing blocks
            OBendTimes (astropy Quantity array):
                Updates the end times of the observing blocks
            OBnumber (integer):
                The Observing Block Number
        """
        if not missionSchedule=='None':  # If the missionSchedule is specified
            tmpOBtimes = list()
            schedulefname = str(os.path.dirname(__file__)+'/../Scripts/' + missionSchedule)
            if os.path.isfile(schedulefname):  # Check if a mission schedule is manually specified
                self.vprint("Loading Manual Schedule from %s"%missionSchedule)
                with open(schedulefname, 'rb') as f:  # load csv file
                    lines = csv.reader(f,delimiter=',')
                    self.vprint('The manual Schedule is:')
                    for line in lines:
                        tmpOBtimes.append(line)
                        self.vprint(line)
                self.OBstartTimes = np.asarray([float(item[0]) for item in tmpOBtimes])*u.d
                self.OBendTimes = np.asarray([float(item[1]) for item in tmpOBtimes])*u.d
        else:  # Automatically construct OB from OBduration, missionLife, and missionPortion
            if OBduration == np.inf*u.d:  # There is 1 OB spanning the mission
                self.OBstartTimes = np.asarray([0])*u.d
                self.OBendTimes = np.asarray([self.missionLife.to('day').value])*u.d
            else:  # OB
                startToStart = OBduration/self.missionPortion
                numBlocks = np.ceil(self.missionLife/startToStart)#This is the number of Observing Blocks
                self.OBstartTimes = np.arange(numBlocks)*startToStart
                self.OBendTimes = self.OBstartTimes + OBduration
                if self.OBendTimes[-1] > self.missionLife:  # If the end of the last observing block exceeds the end of mission
                    self.OBendTimes[-1] = self.missionLife.copy()  # Set end of last OB to end of mission
        self.OBduration = OBduration
        self.OBnumber = 0
        self.vprint('OBendTimes is: ' + str(self.OBendTimes)) # Could Be Deleted

    def mission_is_over(self, settlingTime, ohTime):
        r"""Is the time allocated for the mission used up?
        
        This supplies an abstraction around the test:
            (currentTimeNorm > missionFinishNorm)
        so that users of the class do not have to perform arithmetic
        on class variables.
        Args:
            settlingTime (astropy Quantity):
                Observatory settling time as specified in Obs.settlingTime
            ohTime (astropy Quantity):
                Instrument overhead time as specified by det_mode['syst']['ohTime']
        Returns:
            is_over (Boolean):
                True if the mission time is used up, else False.
        """
        
        is_over = ((self.currentTimeNorm + settlingTime + ohTime >= self.missionLife) \
            or (self.exoplanetObsTime.to('day') + settlingTime + ohTime >= self.missionLife.to('day')*self.missionPortion) \
            or (self.currentTimeNorm + settlingTime + ohTime >= self.OBendTimes[-1]))
        
        return is_over

    def allocate_time(self, dt, addExoplanetObsTime=True):
        r"""Allocate a temporal block of width dt
        
        Advance the mission time by dt units.
        
        Args:
            dt (astropy Quantity):
                Temporal block allocated in units of days
            addExoplanetObsTime (bool):
                Indicates the allocated time is for the primary instrument (True) or some other instrument (False)
                By default this function assumes all allocated time is attributed to the primary instrument (is True)
        Updates Attributes:
            currentTimeNorm (astropy Quantity):
                The current time since mission start
            currentTimeAbs (astropy Time Quantity):
                The current Time in MJD
        Returns:
            success (bool):
                a flag indicating the time allocation was successful or not successful
        """
        #Check dt validity
        if(dt.value <= 0 or dt.value == np.inf):
            self.vprint('The temporal block to allocate must be positive nonzero, got %f'%dt.value)
            return False #The temporal block to allocate is not positive nonzero

        #Check dt exceeds mission life
        if(self.currentTimeNorm + dt > self.missionLife):
            self.vprint('The temporal block to allocate dt=%f at curremtTimeNorm=%f would exceed missionLife %f'%(dt.value, self.currentTimeNorm.value, self.missionLife.value))
            return False #The time to allocate would exceed the missionLife

        #Check dt exceeds current OB
        if(self.currentTimeNorm + dt > self.OBendTimes[self.OBnumber]):
            self.vprint('The temporal block to allocate dt=%f at currentTimeNorm=%f would exceed the end of OBnum=%f at time OBendTimes=%f'%(dt.value, self.currentTimeNorm.value, self.OBnumber, self.OBendTimes[self.OBnumber].value))
            return False

        #Check exceeds allowed instrument Time
        if(addExoplanetObsTime):
            if(self.exoplanetObsTime + dt > self.missionLife*self.missionPortion):
                self.vprint('The temporal block to allocate dt=%f with current exoplanetObsTime=%f would exceed allowed exoplanetObsTime=%f'%(dt.value, self.exoplanetObsTime.value, self.missionLife.value*self.missionPortion))
                return False # The time to allocate would exceed the allowed exoplanet obs time
            self.currentTimeAbs += dt
            self.currentTimeNorm += dt
            self.exoplanetObsTime += dt
            return True
        else:#Time will not be counted against exoplanetObstime
            self.currentTimeAbs += dt
            self.currentTimeNorm += dt
            return True   

    def advancetToStartOfNextOB(self):
        """Advances to Start of Next Observation Block
        This method is called in the allocate_time() method of the TimeKeeping 
        class object, when the allocated time requires moving outside of the current OB.
        If no OB duration was specified, a new Observing Block is created for 
        each observation in the SurveySimulation module. 

        Updates Attributes:
            OBnumber (integer):
                The Observing Block Number
            currentTimeNorm (astropy Quantity):
                The current time since mission start
            currentTimeAbs (astropy Time Quantity):
                The current Time in MJD
        """
        self.OBnumber += 1#increase the observation block number
        self.currentTimeNorm = self.OBstartTimes[self.OBnumber]#update currentTimeNorm
        self.currentTimeAbs = self.OBstartTimes[self.OBnumber] + self.missionStart#update currentTimeAbs

        # begin Survey, and loop until mission is finished
        log_begin = 'OB%s:'%(self.OBnumber)#prints because this is the beginning of the nesxt observation block
        self.vprint(log_begin)

    def advanceToAbsTime(self,tAbs, addExoplanetObsTime=True):
        """Advances the current mission time to tAbs
        Args:
            tAbs (Astropy Quantity):
                The absolute mission time to advance currentTimeAbs to. MUST HAVE scale='tai'
            addExoplanetObsTime (bool):
                A flag indicating whether to add advanced time to exoplanetObsTime or not
        Updates Attributes:
            currentTimeNorm (astropy Quantity):
                The current time since mission start
            currentTimeAbs (astropy Time Quantity):
                The current Time in MJD
        Returns:
            success (bool):
                A bool indicating whether the operation was successful or not
        """
        #Checks on tAbs validity
        if tAbs <= self.currentTimeAbs:
            self.vprint("The time to advance to " + str(tAbs) + " is not after " + str(self.currentTimeAbs))
            return False
        if tAbs == np.inf:
            self.vprint("The time to advance to is inf")
            return False

        #Use 1 and Use 3
        if tAbs <= self.OBendTimes[self.OBnumber] + self.missionStart: # The time to advance to does not leave the current OB (and by extension the end of mission)
            t_added = (tAbs - self.currentTimeAbs).value*u.d
            self.currentTimeNorm = (tAbs - self.missionStart).to('day')
            self.currentTimeAbs = tAbs
            if addExoplanetObsTime:  # count time towards exoplanet Obs Time
                if (self.exoplanetObsTime + t_added) > (self.missionLife*self.missionPortion):
                    self.vprint("The time added to exoplanetObsTime " + str(t_added) + " would exceed the missionLife*missionPortion " + str(self.missionLife*self.missionPortion))
                    #DELETE self.vprint("Advancing to tAbs failed under Use Case 1")
                    self.exoplanetObsTime = (self.missionLife*self.missionPortion)
                    return False
                else:
                    self.exoplanetObsTime += (tAbs - self.currentTimeAbs).value*u.d
                    return True
            else:  # addExoplanetObsTime is False
                self.exoplanetObsTime += 0*u.d
            return True

        #Use 2 and Use 4
        if tAbs >= self.missionFinishAbs:  # Equal to case covered above
            t_added = (tAbs - self.currentTimeAbs).value*u.d
            self.currentTimeNorm += (tAbs - self.missionStart).value*u.d
            self.currentTimeAbs = tAbs
            if addExoplanetObsTime:  # Count time towards exoplanetObs Time
                if (self.exoplanetObsTime + t_added) > (self.missionLife*self.missionPortion):
                    self.vprint("The time added to exoplanetObsTime " + str(t_added) + " would exceed the missionLife*missionPortion " + str(self.missionLife*self.missionPortion))
                    #DELETE self.vprint("Advancing to tAbs failed under Use Case 4")
                    self.exoplanetObsTime = (self.missionLife*self.missionPortion)
                    return False
                self.exoplanetObsTime += self.missionLife - self.currentTimeNorm#Advances exoplanet time to end of mission time
            return True

        #Use 5 and 7 #extended to accomodate any current and future time between OBs
        tNorm = (tAbs - self.missionStart).value*u.d
        if np.any(  (tNorm<=self.OBstartTimes[1:-1])*(tNorm>=self.OBendTimes[0:-2])  ):  # The tAbs is between end End of an OB and start of the Next OB
            endIndex = np.where((tNorm<=self.OBstartTimes[1:-1])*(tNorm>=self.OBendTimes[0:-2])==True)[0][0]  # Return OBnumber of End Index
            t_added = self.OBendTimes[endIndex] - self.currentTimeNorm # Time to be added to exoplanetObsTime from current OB
            for ind in np.arange(endIndex+1,len(self.OBendTimes)):  # Add time for all additional OB
                t_added += self.OBendTimes[ind] - self.OBstartTimes[ind]
            self.OBnumber = endIndex + 1  # set OBnumber to correct Observing Block
            self.currentTimeNorm = self.OBstartTimes[endIndex + 1]  # Advance Time to start of next OB
            self.currentTimeAbs = self.OBstartTimes[endIndex + 1] + self.missionStart  # Advance Time to start of next OB
            if addExoplanetObsTime:  # Count time towards exoplanetObs Time
                if self.exoplanetObsTime + t_added > self.missionLife*self.missionPortion:  # We can CANNOT allocate that time to exoplanetObsTime
                    self.vprint("The time added to exoplanetObsTime " + str(t_added) + " would exceed the missionLife*missionPortion " + str(self.missionLife*self.missionPortion))
                    self.vprint("Advancing to tAbs failed under Use Case 7")
                    self.exoplanetObsTime = (self.missionLife*self.missionPortion)
                    return False
                self.exoplanetObsTime += t_added
            return True

        #Use 6 and 8 #extended to accomodate any current and future time between OBs
        if np.any((tNorm>=self.OBstartTimes[self.OBnumber:-1])*(tNorm<=self.OBendTimes[self.OBnumber:-1])):  # The tAbs is between start of a future OB and end of that OB
            endIndex = np.where((tNorm>=self.OBstartTimes[self.OBnumber:-1])*(tNorm<=self.OBendTimes[self.OBnumber:-1]))# Return index of OB that tAbs will be inside of
            t_added = 0*u.d
            if addExoplanetObsTime:  # Count time towards exoplanetObs Time
                t_added = (tAbs - self.currentTimeAbs).to('day')
                for i in np.arange(endIndex - self.OBnumber)+1:# accumulate time to subtract (time not counted against Exoplanet Obs Time)
                    index = self.OBnumber + i
                    t_added -= (self.OBstartTimes[index + 1] - self.OBendTimes[index]) #Subtract the time between these OB from the t_added to exoplanetObsTime
            self.OBnumber = endIndex  # set OBnumber to correct Observing Block
            self.currentTimeNorm = (tAbs - self.missionStart).to('day')  # Advance Time to start of next OB
            self.currentTimeAbs = tAbs  # Advance Time to start of next OB
            #Check if exoplanetObsTime would be exceeded
            if self.exoplanetObsTime + t_added > self.missionLife*self.missionPortion:
                self.vprint("The time added to exoplanetObsTime " + str((tAbs - self.currentTimeAbs).value*u.d) + " would exceed the missionLife*missionPortion " + str(self.missionLife*self.missionPortion))
                self.vprint("Advancing to tAbs failed under Use Case 8")
                self.exoplanetObsTime = (self.missionLife*self.missionPortion)
                return False
            else:
                self.exoplanetObsTime += t_added
                return True

        self.vprint('No Use Case Found in AdvanceToAbsTime')#Can delete if functioning flawlessly
        self.vprint(fail)

    def get_ObsDetectionMaxIntTime(self,settlingTime, ohTime, timeMultiplier):
        """Tells you the maximum Detection Observation Integration Time you can pass into observation_detection(X,intTime,X)
        Args:
            settlingTime (astropy Quantity):
                Observatory settling time as specified in Obs.settlingTime
            ohTime (astropy Quantity):
                Instrument overhead time as specified by det_mode['syst']['ohTime']
            timeMultiplier (float):
                Integration time multiplier as specified by det_mode['timeMultiplier']
        Returns:
            maxIntTimeOBendTime (astropy Quantity):
                The maximum integration time bounded by Observation Block end Time
            maxIntTimeExoplanetObsTime (astropy Quantity):
                The maximum integration time bounded by exoplanetObsTime
            maxIntTimeMissionLife (astropy Quantity):
                The maximum integration time bounded by MissionLife
        """
        maxTimeOBendTime = self.OBendTimes[self.OBnumber] - self.currentTimeNorm
        maxIntTimeOBendTime = (maxTimeOBendTime - settlingTime - ohTime)/(1 + timeMultiplier -1)

        maxTimeExoplanetObsTime = self.missionLife*self.missionPortion - self.exoplanetObsTime
        maxIntTimeExoplanetObsTime = (maxTimeExoplanetObsTime - settlingTime - ohTime)/(1 + timeMultiplier -1)

        maxTimeMissionLife = self.missionLife - self.currentTimeNorm
        maxIntTimeMissionLife = (maxTimeMissionLife - settlingTime - ohTime)/(1 + timeMultiplier -1)

        #Ensure all are positive or zero
        if maxIntTimeOBendTime < 0:
            maxIntTimeOBendTime = 0*u.d
        if maxIntTimeExoplanetObsTime < 0:
            maxIntTimeExoplanetObsTime = 0*u.d
        if maxIntTimeMissionLife < 0:
            maxIntTimeMissionLife = 0*u.d

        return maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife

    # def get_TAbs_mission_is_over(self):
    #     """Calculates nearest mission termination time in Absolute Time
    #     Returns:
    #         tAbs (astropy Time Quantity):
    #             The absolute time that terminaltes the mission
    #     """
    #     tAbs1 = self.OBendTimes[-1] + self.missionStart
    #     tAbs2 = (self.missionLife*self.missionPortion - self.exoplanetObsTime) + self.currentTimeAbs
    #     tAbs3 = self.missionLife + self.missionStart

    #     return min(tAbs1,tAbs2,tAbs3)
