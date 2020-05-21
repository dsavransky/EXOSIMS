from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_dirs import get_cache_dir
import numpy as np
import astropy.units as u
from astropy.time import Time
import os
import csv

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
        specs:
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
        cachedir (str):
            Path to cache directory    
    """

    _modtype = 'TimeKeeping'
    #_outspec = {}#DMITRY you lef this here. Commented out for future review

    def __init__(self, missionStart=60634, missionLife=0.1, 
        missionPortion=1, OBduration=np.inf, missionSchedule=None,
        cachedir=None, **specs):

        _outspec = {}
   
        #start the outspec
        self._outspec = {}

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec['cachedir'] = self.cachedir
        specs['cachedir'] = self.cachedir 

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
        self.missionLife = float(missionLife)*u.year#the total amount of time since mission start that can elapse MUST BE IN YEAR HERE FOR OUTSPEC
        self.missionFinishAbs = self.missionStart + self.missionLife.to('day')#the absolute time the mission can possibly end
        
        # initialize values updated by functions
        self.currentTimeNorm = 0.*u.day#the current amount of time since mission start that has elapsed
        self.currentTimeAbs = self.missionStart#the absolute mission time
        
        # initialize observing block times arrays. #An Observing Block is a segment of time over which observations may take place
        self.missionSchedule = missionSchedule
        self.init_OB(str(missionSchedule), OBduration*u.d)
        
        # initialize time spend using instrument
        self.exoplanetObsTime = 0*u.day
        
        # populate outspec
        for att in self.__dict__:
            if att not in ['vprint','_outspec']:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat,(u.Quantity,Time)) else dat

    def __str__(self):
        r"""String representation of the TimeKeeping object.
        
        When the command 'print' is used on the TimeKeeping object, this 
        method prints the values contained in the object."""
        
        for att in self.__dict__:
            print('%s: %r' % (att, getattr(self, att)))
        
        return 'TimeKeeping instance at %.6f days' % self.currentTimeNorm.to('day').value

    def init_OB(self, missionSchedule, OBduration):
        """ 
        Initializes mission Observing Blocks from file or missionDuration, missionLife,
        and missionPortion. Updates attributes OBstartTimes, OBendTimes, and OBnumber
        
        Args:
            missionSchedule (string):
                a string containing the missionSchedule file
            OBduration (astropy Quantity):
                the duration of a single observing block
        
        """
        if not missionSchedule=='None':  # If the missionSchedule is specified
            tmpOBtimes = list()
            schedulefname = str(os.path.dirname(__file__)+'/../Scripts/' + missionSchedule)#.csv file in EXOSIMS/Scripts folder
            if not os.path.isfile(schedulefname):
                #This is if we allowed the OB.csv to live in EXOSIMS/../Scripts
                #schedulefname = str(os.path.dirname(__file__)+'/../../../Scripts/' + missionSchedule)

                #Check if scriptNames in ScriptsPath
                ScriptsPath = str(os.path.dirname(__file__)+'/../../../Scripts/')
                makeSimilar_TemplateFolder = ''
                dirsFolderDown = [x[0].split('/')[-1] for x in os.walk(ScriptsPath)] #Get all directories in ScriptsPath
                for tmpFolder in dirsFolderDown:
                    if os.path.isfile(ScriptsPath + tmpFolder + '/' + missionSchedule) and not tmpFolder == '':#We found the Scripts folder containing scriptfile
                        makeSimilar_TemplateFolder = tmpFolder + '/'#We found the file!!!
                        break
                schedulefname = str(ScriptsPath + makeSimilar_TemplateFolder + missionSchedule)#.csv file in EXOSIMS/Scripts folder

            if os.path.isfile(schedulefname):  # Check if a mission schedule is manually specified
                self.vprint("Loading Manual Schedule from %s"%missionSchedule)
                with open(schedulefname, 'r') as f:  # load csv file
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
                numBlocks = np.ceil(self.missionLife.to('day')/startToStart)#This is the number of Observing Blocks
                self.OBstartTimes = np.arange(numBlocks)*startToStart
                self.OBendTimes = self.OBstartTimes + OBduration
                if self.OBendTimes[-1] > self.missionLife.to('day'):  # If the end of the last observing block exceeds the end of mission
                    self.OBendTimes[-1] = self.missionLife.to('day').copy()  # Set end of last OB to end of mission
        self.OBduration = OBduration
        self.OBnumber = 0
        self.vprint('OBendTimes is: ' + str(self.OBendTimes)) # Could Be Deleted

    def mission_is_over(self, OS, Obs, mode):
        r"""Is the time allocated for the mission used up?
        
        This supplies an abstraction around the test: ::
            
            (currentTimeNorm > missionFinishNorm)
            
        so that users of the class do not have to perform arithmetic
        on class variables.

        Args:
            OS (Optical System object):
                Optical System module for OS.haveOcculter
            Obs (Observatory Object):
                Observatory module for Obs.settlingTime
            mode (dict):
                Selected observing mode for detection (uses only overhead time)

        Returns:
            boolean:
                True if the mission time is used up, else False.
        """
        
        is_over = ((self.currentTimeNorm + Obs.settlingTime + mode['syst']['ohTime'] >= self.missionLife.to('day')) \
            or (self.exoplanetObsTime.to('day') + Obs.settlingTime + mode['syst']['ohTime'] >= self.missionLife.to('day')*self.missionPortion) \
            or (self.currentTimeNorm + Obs.settlingTime + mode['syst']['ohTime'] >= self.OBendTimes[-1]) \
            or (OS.haveOcculter and Obs.scMass < Obs.dryMass))

        if (OS.haveOcculter and Obs.scMass < Obs.dryMass):
            self.vprint('Total fuel mass (Obs.dryMass %.2f) less than (Obs.scMass %.2f) at currentTimeNorm %sd'%(Obs.dryMass.value, Obs.scMass.value, self.currentTimeNorm.to('day').round(2)))
        if (self.currentTimeNorm + Obs.settlingTime + mode['syst']['ohTime'] >= self.OBendTimes[-1]):
            self.vprint('Last Observing Block (OBnum %d, OBendTime[-1] %.2f) would be exceeded at currentTimeNorm %sd'%(self.OBnumber, self.OBendTimes[-1].value, self.currentTimeNorm.to('day').round(2)))
        if (self.exoplanetObsTime.to('day') + Obs.settlingTime + mode['syst']['ohTime'] >= self.missionLife.to('day')*self.missionPortion):
            self.vprint('exoplanetObstime (%.2f) would exceed (missionPortion*missionLife= %.2f) at currentTimeNorm %sd'%(self.exoplanetObsTime.value, self.missionPortion*self.missionLife.to('day').value, self.currentTimeNorm.to('day').round(2)))
        if (self.currentTimeNorm + Obs.settlingTime + mode['syst']['ohTime'] >= self.missionLife.to('day')):
            self.vprint('missionLife would be exceeded at %s'%self.currentTimeNorm.to('day').round(2))
        
        return is_over

    def allocate_time(self, dt, addExoplanetObsTime=True):
        r"""Allocate a temporal block of width dt
        
        Advance the mission time by dt units. Updates attributes currentTimeNorm and currentTimeAbs
        
        Args:
            dt (astropy Quantity):
                Temporal block allocated in units of days
            addExoplanetObsTime (bool):
                Indicates the allocated time is for the primary instrument (True) or some other instrument (False)
                By default this function assumes all allocated time is attributed to the primary instrument (is True)
        
        Returns:
            bool:
                a flag indicating the time allocation was successful or not successful
        """

        #Check dt validity
        if(dt.value <= 0 or dt.value == np.inf):
            self.vprint('The temporal block to allocate must be positive nonzero, got %f'%dt.value)
            return False #The temporal block to allocate is not positive nonzero

        #Check dt exceeds mission life
        if(self.currentTimeNorm + dt > self.missionLife.to('day')):
            self.vprint('The temporal block to allocate dt=%f at curremtTimeNorm=%f would exceed missionLife %f'%(dt.value, self.currentTimeNorm.value, self.missionLife.to('day').value))
            return False #The time to allocate would exceed the missionLife

        #Check dt exceeds current OB
        if(self.currentTimeNorm + dt > self.OBendTimes[self.OBnumber]):
            self.vprint('The temporal block to allocate dt=%f at currentTimeNorm=%f would exceed the end of OBnum=%f at time OBendTimes=%f'%(dt.value, self.currentTimeNorm.value, self.OBnumber, self.OBendTimes[self.OBnumber].value))
            return False

        #Check exceeds allowed instrument Time
        if(addExoplanetObsTime):
            if(self.exoplanetObsTime + dt > self.missionLife.to('day')*self.missionPortion):
                self.vprint('The temporal block to allocate dt=%f with current exoplanetObsTime=%f would exceed allowed exoplanetObsTime=%f'%(dt.value, self.exoplanetObsTime.value, self.missionLife.to('day').value*self.missionPortion))
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
        each observation in the SurveySimulation module. Updates attributes OBnumber, 
        currentTimeNorm and currentTimeAbs.

        """
        self.OBnumber += 1#increase the observation block number
        self.currentTimeNorm = self.OBstartTimes[self.OBnumber]#update currentTimeNorm
        self.currentTimeAbs = self.OBstartTimes[self.OBnumber] + self.missionStart#update currentTimeAbs

        # begin Survey, and loop until mission is finished
        log_begin = 'OB%s:'%(self.OBnumber)#prints because this is the beginning of the nesxt observation block
        self.vprint(log_begin)
        self.vprint("Advanced currentTimeNorm to beginning of next OB %.2fd"%(self.currentTimeNorm.to('day').value))

    def advanceToAbsTime(self,tAbs, addExoplanetObsTime=True):
        """Advances the current mission time to tAbs. 
        Updates attributes currentTimeNorma dn currentTimeAbs

        Args:
            tAbs (Astropy Quantity):
                The absolute mission time to advance currentTimeAbs to. MUST HAVE scale='tai'
            addExoplanetObsTime (bool):
                A flag indicating whether to add advanced time to exoplanetObsTime or not

        Returns:
            bool:
                A bool indicating whether the operation was successful or not
        """

        #Checks on tAbs validity
        if tAbs <= self.currentTimeAbs:
            self.vprint("The time to advance to " + str(tAbs) + " is not after " + str(self.currentTimeAbs))
            return False

        #Use 2 and Use 4
        if tAbs >= self.missionFinishAbs:  # 
            tmpcurrentTimeNorm = self.currentTimeNorm.copy()
            t_added = (tAbs - self.currentTimeAbs).value*u.d
            self.currentTimeNorm = (tAbs - self.missionStart).value*u.d
            self.currentTimeAbs = tAbs
            if addExoplanetObsTime:  # Count time towards exoplanetObs Time
                if (self.exoplanetObsTime + t_added) > (self.missionLife.to('day')*self.missionPortion):
                    self.vprint("exoplanetObsTime = %.2fd. The time added to exoplanetObsTime %0.2fd would exceed the missionLife*missionPortion %0.2fd" \
                        %(self.exoplanetObsTime.to('day').value, t_added.to('day').value, self.missionLife.to('day').value*self.missionPortion))
                    self.exoplanetObsTime = (self.missionLife.to('day')*self.missionPortion)
                    return False
                self.exoplanetObsTime += self.missionLife.to('day') - tmpcurrentTimeNorm#Advances exoplanet time to end of mission time
            else:
                self.exoplanetObsTime += 0*u.d
            return True

        #Use 1 and Use 3
        if tAbs <= self.OBendTimes[self.OBnumber] + self.missionStart: # The time to advance to does not leave the current OB
            t_added = (tAbs - self.currentTimeAbs).value*u.d
            self.currentTimeNorm = (tAbs - self.missionStart).to('day')
            self.currentTimeAbs = tAbs
            if addExoplanetObsTime:  # count time towards exoplanet Obs Time
                if (self.exoplanetObsTime + t_added) > (self.missionLife.to('day')*self.missionPortion):
                    self.vprint("exoplanetObsTime = %.2fd. The time added to exoplanetObsTime %0.2fd would exceed the missionLife*missionPortion %0.2fd" \
                        %(self.exoplanetObsTime.to('day').value, t_added.to('day').value, self.missionLife.to('day').value*self.missionPortion))
                    self.exoplanetObsTime = (self.missionLife.to('day')*self.missionPortion)
                    return False
                else:
                    self.exoplanetObsTime += t_added
                    return True
            else:  # addExoplanetObsTime is False
                self.exoplanetObsTime += 0*u.d
            return True

        #Use 5 and 7 #extended to accomodate any current and future time between OBs
        tNorm = (tAbs - self.missionStart).value*u.d
        if np.any((tNorm<=self.OBstartTimes[1:])*(tNorm>=self.OBendTimes[0:-1])):  # The tAbs is between end End of an OB and start of the Next OB
            endIndex = np.where((tNorm<=self.OBstartTimes[1:])*(tNorm>=self.OBendTimes[0:-1])==True)[0][0]  # Return OBnumber of End Index
            t_added = self.OBendTimes[self.OBnumber] - self.currentTimeNorm#self.OBendTimes[endIndex+1] - self.currentTimeNorm # Time to be added to exoplanetObsTime from current OB
            for ind in np.arange(self.OBnumber,endIndex):#,len(self.OBendTimes)):  # Add time for all additional OB
                t_added += self.OBendTimes[ind] - self.OBstartTimes[ind]
            while (self.OBnumber < endIndex + 1):
                self.advancetToStartOfNextOB()
            #self.OBnumber = endIndex + 1  # set OBnumber to correct Observing Block
            #self.currentTimeNorm = self.OBstartTimes[self.OBnumber]  # Advance Time to start of next OB
            #self.currentTimeAbs = self.OBstartTimes[self.OBnumber] + self.missionStart  # Advance Time to start of next OB
            if addExoplanetObsTime:  # Count time towards exoplanetObs Time
                if self.exoplanetObsTime + t_added > self.missionLife.to('day')*self.missionPortion:  # We can CANNOT allocate that time to exoplanetObsTime
                    self.vprint("exoplanetObsTime = %.2fd. The time added to exoplanetObsTime %0.2fd would exceed the missionLife*missionPortion %0.2fd" \
                        %(self.exoplanetObsTime.to('day').value, t_added.to('day').value, self.missionLife.to('day').value*self.missionPortion))
                    self.vprint("Advancing to tAbs failed under Use Case 7")  # This kind of failure is by design. It just means the mission has come to an end
                    self.exoplanetObsTime = (self.missionLife.to('day')*self.missionPortion)
                    return False
                self.exoplanetObsTime += t_added
            else:
                self.exoplanetObsTime += 0*u.d
            return True

        #Use 6 and 8 #extended to accomodate any current and future time inside OBs
        if np.any((tNorm>=self.OBstartTimes[self.OBnumber:])*(tNorm<=self.OBendTimes[self.OBnumber:])):  # The tAbs is between start of a future OB and end of that OB
            endIndex = np.where((tNorm>=self.OBstartTimes[self.OBnumber:])*(tNorm<=self.OBendTimes[self.OBnumber:])==True)[0][0]# Return index of OB that tAbs will be inside of
            endIndex += self.OBnumber            
            t_added = 0*u.d
            if addExoplanetObsTime:  # Count time towards exoplanetObs Time
                t_added += (tAbs - self.currentTimeAbs).to('day')
                for i in np.arange(self.OBnumber, endIndex):# accumulate time to subtract (time not counted against Exoplanet Obs Time)
                    index = self.OBnumber
                    t_added -= self.OBstartTimes[index + 1] - self.OBendTimes[index] #Subtract the time between these OB from the t_added to exoplanetObsTime
                    #Check if exoplanetObsTime would be exceeded
                if self.exoplanetObsTime + t_added > self.missionLife.to('day')*self.missionPortion:
                    self.vprint("exoplanetObsTime = %.2fd. The time added to exoplanetObsTime %0.2fd would exceed the missionLife*missionPortion %0.2fd" \
                        %(self.exoplanetObsTime.to('day').value, t_added.to('day').value, self.missionLife.to('day').value*self.missionPortion))
                    self.vprint("Advancing to tAbs failed under Use Case 8")  # This kind of failure is by design. It just means the mission has come to an end
                    self.exoplanetObsTime = (self.missionLife.to('day')*self.missionPortion)
                    self.OBnumber = endIndex  # set OBnumber to correct Observing Block
                    self.currentTimeNorm = (tAbs - self.missionStart).to('day')  # Advance Time to start of next OB
                    self.currentTimeAbs = tAbs  # Advance Time to start of next OB
                    return False
                else:
                    self.exoplanetObsTime += t_added
            else: # addExoplanetObsTime is False
                self.exoplanetObsTime += 0*u.d
            self.OBnumber = endIndex  # set OBnumber to correct Observing Block
            self.currentTimeNorm = (tAbs - self.missionStart).to('day')  # Advance Time to start of next OB
            self.currentTimeAbs = tAbs  # Advance Time to start of next OB
            return True

        assert False, "No Use Case Found in AdvanceToAbsTime" #Generic error if there exists some use case that I have not encountered yet.

    def get_ObsDetectionMaxIntTime(self,Obs,mode,currentTimeNorm=None,OBnumber=None):
        """Tells you the maximum Detection Observation Integration Time you can pass into observation_detection(X,intTime,X)
        Args:
            Obs (Observatory Object):
                Observatory module for Obs.settlingTime
            mode (dict):
                Selected observing mode for detection

        Returns:
            tuple:
            maxIntTimeOBendTime (astropy Quantity):
                The maximum integration time bounded by Observation Block end Time
            maxIntTimeExoplanetObsTime (astropy Quantity):
                The maximum integration time bounded by exoplanetObsTime
            maxIntTimeMissionLife (astropy Quantity):
                The maximum integration time bounded by MissionLife
        """
        if OBnumber == None:
            OBnumber = self.OBnumber
        if currentTimeNorm == None:
            currentTimeNorm = self.currentTimeNorm.copy()
            
        maxTimeOBendTime = self.OBendTimes[OBnumber] - currentTimeNorm
        maxIntTimeOBendTime = (maxTimeOBendTime - Obs.settlingTime - mode['syst']['ohTime'])/(1. + mode['timeMultiplier'] -1.)

        maxTimeExoplanetObsTime = self.missionLife.to('day')*self.missionPortion - self.exoplanetObsTime
        maxIntTimeExoplanetObsTime = (maxTimeExoplanetObsTime - Obs.settlingTime - mode['syst']['ohTime'])/(1. + mode['timeMultiplier'] -1.)

        maxTimeMissionLife = self.missionLife.to('day') - currentTimeNorm
        maxIntTimeMissionLife = (maxTimeMissionLife - Obs.settlingTime - mode['syst']['ohTime'])/(1. + mode['timeMultiplier'] -1.)

        #Ensure all are positive or zero
        if maxIntTimeOBendTime < 0.:
            maxIntTimeOBendTime = 0.*u.d
        if maxIntTimeExoplanetObsTime < 0.:
            maxIntTimeExoplanetObsTime = 0.*u.d
        if maxIntTimeMissionLife < 0.:
            maxIntTimeMissionLife = 0.*u.d

        return maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife
