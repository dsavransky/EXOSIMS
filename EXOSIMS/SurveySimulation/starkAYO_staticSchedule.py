from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import scipy
from scipy.optimize import fmin
import timeit
import csv
import os.path
import datetime

class starkAYO_staticSchedule(SurveySimulation):
    """starkAYO _static Scheduler
    
    This class implements a Scheduler that creates a list of stars to observe and integration times to observe them. It also selects the best star to observe at any moment in time
    Generates cachedfZ.csv and cachedMaxCbyTtime.csv
    If the above exist but some problem is found, moved_MacCbyTtime.csv and moved_fZAllStars.csv are created

    2nd execution time 2 min 30 sec
    """
    def __init__(self, **specs):
        SurveySimulation.__init__(self, **specs)
        # bring inherited class objects to top level of Survey Simulation
        SU = self.SimulatedUniverse
        OS = SU.OpticalSystem
        ZL = SU.ZodiacalLight
        self.Completeness = SU.Completeness
        TL = SU.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping

        
        self.starVisits = np.zeros(TL.nStars,dtype=int)
        self.starRevisit = np.array([])

        detMode = filter(lambda mode: mode['detectionMode'] == True, OS.observingModes)[0]
        spectroModes = filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes)
        self.mode = detMode
        
        #Create and start Schedule
        self.schedule = np.arange(TL.nStars)#self.schedule is meant to be editable
        self.schedule_startSaved = np.arange(TL.nStars)#preserves initial list of targets
          
        dMagLim = self.Completeness.dMagLim
        self.dmag_startSaved = np.linspace(1, dMagLim, num=1500,endpoint=True)

        sInds = self.schedule_startSaved
        dmag = self.dmag_startSaved
        WA = OS.WA0
        startTime = np.zeros(sInds.shape[0])*u.d + self.TimeKeeping.currentTimeAbs

        tovisit = np.zeros(sInds.shape[0], dtype=bool)

        #Generate fZ
        self.fZ_startSaved = self.generate_fZ(sInds)#Sept 21, 2017 execution time 0.725 sec

        #Estimate Yearly fZmin###########################################
        tmpfZ = np.asarray(self.fZ_startSaved)
        fZ_matrix = tmpfZ[sInds,:]#Apply previous filters to fZ_startSaved[sInds, 1000]
        #Find minimum fZ of each star
        fZmin = np.zeros(sInds.shape[0])
        fZminInds = np.zeros(sInds.shape[0])
        for i in xrange(len(sInds)):
            fZmin[i] = min(fZ_matrix[i,:])
            fZminInds[i] = np.argmin(fZ_matrix[i,:])
        #################################################################

        #CACHE Cb Cp Csp################################################Sept 20, 2017 execution time 10.108 sec
        fZ = fZmin/u.arcsec**2#
        fEZ = ZL.fEZ0
        mode = self.mode#resolve this mode is passed into next_target
        allModes = self.OpticalSystem.observingModes
        det_mode = filter(lambda mode: mode['detectionMode'] == True, allModes)[0]
        Cp = np.zeros([sInds.shape[0],dmag.shape[0]])
        Cb = np.zeros(sInds.shape[0])
        Csp = np.zeros(sInds.shape[0])
        for i in xrange(dmag.shape[0]):
            Cp[:,i], Cb[:], Csp[:] = OS.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dmag[i], WA, det_mode)
        self.Cb = Cb[:]/u.s#Cb[:,0]/u.s#note all Cb are the same for different dmags. They are just star dependent
        self.Csp = Csp[:]/u.s#Csp[:,0]/u.s#note all Csp are the same for different dmags. They are just star dependent
        #self.Cp = Cp[:,:] #This one is dependent upon dmag and each star
        ################################################################

        #Solve Initial Integration Times###############################################
        def CbyTfunc(t_dets, self, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp):
            CbyT = -self.Completeness.comp_per_intTime(t_dets*u.d, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)/t_dets*u.d
            return CbyT.value

        maxCbyTtime = np.zeros(sInds.shape[0])#This contains the time maxCbyT occurs at
        maxCbyT = np.zeros(sInds.shape[0])#this contains the value of maxCbyT

        dir_path = os.path.dirname(os.path.realpath(__file__))#find current directory of survey Simualtion
        fname = '/cachedMaxCbyTtime.csv'
        #Check File Length
        fileLength = 0
        numcommas = 0
        try: 
            with open(dir_path+fname, 'rb') as f:
                reader = csv.reader(f)
                your_list = list(reader)
                fileLength = len(your_list)
                numcommas = len(your_list[0])
                f.close()
        except:
            fileLength = 0
            numcommas = 0

        if(not os.path.isfile(dir_path+fname) or not (fileLength != sInds.shape[0] or numcommas != sInds.shape[0])):#If the file does not exist or is not the proper size, Recalculate
            
            for i in xrange(sInds.shape[0]):
                x0 = 0.01
                maxCbyTtime[i] = fmin(CbyTfunc, x0, xtol=1e-8, args=(self, TL, sInds[i], fZ[i], fEZ, WA, mode, self.Cb[i], self.Csp[i]), disp=False)
            t_dets = maxCbyTtime
            #Sept 27, Execution time 101 seconds for 651 stars

            #Save maxCbyT to File######################################
            try:#Here we delete the previous fZ file
                timeNow = datetime.datetime.now()
                timeString = str(timeNow.year)+'_'+str(timeNow.month)+'_'+str(timeNow.day)+'_'+str(timeNow.hour)+'_'+str(timeNow.minute)+'_'
                fnameNew = '/' + timeString +  'moved_maxCbyTtime.csv'
                os.rename(dir_path+fname,dir_path+fnameNew)
            except OSError:
                pass
            with open(dir_path+fname, "wb") as fo:
                wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
                wr.writerow(maxCbyTtime)#Write the fZ to file
                fo.close()

        #Load maxCbyTtime for Each Star From File###### Sept 28, 2017 execution time is 0.00057 sec
        maxCbyTtimeList = list()
        with open(dir_path+fname, 'rb') as f:
            reader = csv.reader(f)
            your_list = list(reader)
            f.close()
        for i in range(len(your_list)):
            tmp = np.asarray(your_list[i]).astype(np.float)
            maxCbyTtimeList.append(tmp)
        maxCbyTtime = np.asarray(maxCbyTtimeList, dtype=np.float64)
        t_dets = maxCbyTtime[0][sInds]
        #############################################################################

        #Sacrifice Stars and then Distribute Excess Mission Time################################################Sept 28, 2017 execution time 19.0 sec
        missionLength = (TK.missionFinishAbs-TK.currentTimeAbs).value*12/12#TK.missionLife.to(u.day).value#mission length in days
        overheadTime = self.Observatory.settlingTime.value + self.OpticalSystem.observingModes[0]['syst']['ohTime'].value#OH time in days
        while((sum(t_dets) + sInds.shape[0]*overheadTime) > missionLength):#the sum of star observation times is still larger than the mission length
            sInds, t_dets, sacrificedStarTime, fZ = self.sacrificeStarCbyT(sInds, t_dets, fZ, fEZ, WA, overheadTime)

        if(sum(t_dets + sInds.shape[0]*overheadTime) > missionLength):#There is some excess time
            sacrificedStarTime = missionLength - (sum(t_dets) + sInds.shape[0]*overheadTime)#The amount of time the list is under the total mission Time
            t_dets = self.distributedt(sInds, t_dets, sacrificedStarTime, fZ, fEZ, WA)
        ###############################################################################

        #STARK AYO LOOP################################################################
        savedSumComp00 = np.zeros(sInds.shape[0])
        firstIteration = 1#checks if this is the first iteration.
        numits = 0#ensure an infinite loop does not occur. Should be depricated
        lastIterationSumComp  = -10000000 #this is some ludacrisly negative number to ensure sumcomp runs. All sumcomps should be positive
        while numits < 100000 and sInds is not None:
            numits = numits+1#we increment numits each loop iteration

            #Sacrifice Lowest Performing Star################################################Sept 28, 2017 execution time 0.0744 0.032 at smallest list size
            sInds, t_dets, sacrificedStarTime, fZ = self.sacrificeStarCbyT(sInds, t_dets, fZ, fEZ, WA, overheadTime)

            #Distribute Sacrificed Time to new star observations############################# Sept 28, 2017 execution time 0.715, 0.64 at smallest (depends on # stars)
            t_dets = self.distributedt(sInds, t_dets, sacrificedStarTime, fZ, fEZ, WA)

            #AYO Termination Conditions###############################Sept 28, 2017 execution time 0.033 sec
            Comp00 = self.Completeness.comp_per_intTime(t_dets*u.d, TL, sInds, fZ, fEZ, WA, mode, self.Cb, self.Csp)

            #change this to an assert
            if 1 >= len(sInds):#if this is the last ement in the list
                break
            savedSumComp00[numits-1] = sum(Comp00)
            #If the total sum of completeness at this moment is less than the last sum, then exit
            if(sum(Comp00) < lastIterationSumComp):#If sacrificing the additional target reduced performance, then Define Output of AYO Process
                CbyT = self.Completeness.comp_per_intTime(t_dets*u.d, self.TargetList, sInds, fZ, fEZ, WA, self.mode, self.Cb, self.Csp)/t_dets#takes 5 seconds to do 1 time for all stars
                sortIndex = np.argsort(CbyT,axis=-1)[::-1]

                #This is the static optimal schedule
                self.schedule = sInds[sortIndex]
                self.t_dets = t_dets[sortIndex]
                self.CbyT = CbyT[sortIndex]
                self.fZ = fZ[sortIndex]
                self.Comp00 = Comp00[sortIndex]
                break
            else:#else set lastIterationSumComp to current sum Comp00
                lastIterationSumComp = sum(Comp00)
                self.vprint(str(numits) + ' SumComp ' + str(sum(Comp00)) + ' Sum(t_dets) ' + str(sum(t_dets)) + ' sInds ' + str(sInds.shape[0]*float(1)) + ' TimeConservation ' + str(sum(t_dets)+sInds.shape[0]*float(1)))# + ' Avg C/T ' + str(np.average(CbyT)))
        #End While Loop
        #END INIT##################################################################
        
    def choose_next_target(self,old_sInd,sInds,slewTime,intTimes):
        """Generate Next Target to Select based off of AYO at this instant in time
        Args:
            sInds - indicies of stars under consideration
            old_sInd - unused
            slewTime - unused
            intTimes - unused

        Returns:
            DRM - A blank structure
            sInd - the single index of self.schedule_startSaved to observe
            t_det - the time to observe sInd in days (u.d)
        """
        SU = self.SimulatedUniverse
        OS = SU.OpticalSystem
        ZL = SU.ZodiacalLight
        self.Completeness = SU.Completeness
        TL = SU.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        mode = self.mode
        # now, start to look for available targets
        cnt = 0
        while not TK.mission_is_over():
            TK.obsStart = TK.currentTimeNorm.to('day')

            dmag = self.dmag_startSaved
            WA = OS.WA0
            startTime = np.zeros(sInds.shape[0])*u.d + self.TimeKeeping.currentTimeAbs

            tovisit = np.zeros(self.schedule_startSaved.shape[0], dtype=bool)
            fZtovisit = np.zeros(self.schedule_startSaved.shape[0], dtype=bool)

            DRM = {}#Create DRM

            startTime = np.zeros(sInds.shape[0])*u.d + self.TimeKeeping.currentTimeAbs

            #Estimate Yearly fZmin###########################################
            tmpfZ = np.asarray(self.fZ_startSaved)
            fZ_matrix = tmpfZ[sInds,:]#Apply previous filters to fZ_startSaved[sInds, 1000]
            #Find minimum fZ of each star
            fZmin = np.zeros(sInds.shape[0])
            for i in xrange(len(sInds)):
                fZmin[i] = min(fZ_matrix[i,:])

            #Find current fZ
            indexFrac = np.interp((self.TimeKeeping.currentTimeAbs-self.TimeKeeping.missionStart).value,[0,365.25],[0,1000])#This is only good for 1 year missions right now
            fZ = np.zeros(sInds.shape[0])
            fZ[:] = (indexFrac%1)*fZ_matrix[:,int(indexFrac)] + (1-indexFrac%1)*fZ_matrix[:,int(indexFrac+1)]#this is the current fZ

            imat = [self.schedule.tolist().index(x) for x in sInds]
            CbyT = self.CbyT[imat]
            t_dets = self.t_dets[imat]
            Comp00 = self.Comp00[imat]
            imat2 = [self.schedule_startSaved.tolist().index(x) for x in sInds]
            dec = self.TargetList.coords.dec[imat2].value
        
            if len(sInds) > 0:
                # store selected star integration time
                selectInd = np.argmin(Comp00*abs(fZ-fZmin)/abs(dec))
                sInd = sInds[selectInd]#finds index of star to sacrifice
                t_det = t_dets[selectInd]*u.d

                #Create a check to determine if the mission length would be exceeded.
                timeLeft = TK.missionFinishNorm - TK.currentTimeNorm#This is how much time we have left in the mission in u.d
                if(timeLeft > (Obs.settlingTime + mode['syst']['ohTime'])):#There is enough time left for overhead time but not for the full t_det
                    if(timeLeft > (t_det+Obs.settlingTime + mode['syst']['ohTime'])):#If the nominal plan for observation time is greater than what we can do
                        t_det = t_det
                    else:
                        t_det = timeLeft - (Obs.settlingTime + mode['syst']['ohTime'])#We reassign t_det to fill the remaining time
                    break 
                else:#There is insufficient time to cover overhead time
                    TK.allocate_time(timeLeft*u.d)
                    sInd = None
                    t_det = None
                    break

            # if no observable target, call the TimeKeeping.wait() method
            else:
                TK.allocate_time(TK.waitTime*TK.waitMultiple**cnt)
                cnt += 1
        else:
            return None#DRM, None, None
        return sInd

    def calc_targ_intTime(self, sInds, startTimes, mode):
        """Finds and Returns Precomputed Observation Time
        Args:
            sInds (integer array):
                Indices of available targets
            startTimes (astropy quantity array):
                absolute start times of observations.  
                must be of the same size as sInds 
            mode (dict):
                Selected observing mode for detection
        Returns:
            intTimes (astropy Quantity array):
                Integration times for detection 
                same dimension as sInds
        """
        commonsInds = [val for val in self.schedule if val in sInds]#finds indicies in common between sInds and self.schedule
        imat = [self.schedule.tolist().index(x) for x in commonsInds]#find indicies of occurence of commonsInds in self.schedule
        intTimes = np.zeros(self.TargetList.nStars)#default observation time is 0 days
        intTimes[commonsInds] = self.t_dets[imat]#
        intTimes = intTimes*u.d#add units of day to intTimes

        return intTimes[sInds]
  
    def distributedt(self, sInds, t_dets, sacrificedStarTime, fZ, fEZ, WA):#distributing the sacrificed time
        """Distributes sacrificedStarTime amoung sInds
        Args:
            sInds[nStars] - indicies of stars in the list
            t_dets[nStars] - time to observe each star (in days)
            sacrificedStarTime - time to distribute in days
            fZ[nStars] - zodiacal light for each target
            fEZ - 0 
        Returns:
            t_dets[nStars] - time to observe each star (in days)
        """
        #Calculate dCbydT for each star at this point in time
        dCbydt = self.Completeness.dcomp_dt(t_dets*u.d, self.TargetList, sInds, fZ, fEZ, WA, self.mode, self.Cb, self.Csp)#dCbydT[nStars]#Sept 28, 2017 0.12sec

        if(len(t_dets) <= 1):
            return t_dets

        timeToDistribute = sacrificedStarTime
        dt_static = 0.1
        dt = dt_static

        #Now decide where to put dt
        numItsDist = 0
        while(timeToDistribute > 0):
            if(numItsDist > 1000000):#this is an infinite loop check
                break
            else:
                numItsDist = numItsDist + 1

            if(timeToDistribute < dt):#if the timeToDistribute is smaller than dt
                dt = timeToDistribute#dt is now the timeToDistribute
            else:#timeToDistribute >= dt under nominal conditions, this is dt to use
                dt = dt_static#this is the maximum quantity of time to distribute at a time.      

            maxdCbydtIndex = np.argmax(dCbydt)#Find most worthy target

            t_dets[maxdCbydtIndex] = t_dets[maxdCbydtIndex] + dt#Add dt to the most worthy target
            timeToDistribute = timeToDistribute - dt#subtract distributed time dt from the timeToDistribute
            dCbydt[maxdCbydtIndex] = self.Completeness.dcomp_dt(t_dets[maxdCbydtIndex]*u.d, self.TargetList, sInds[maxdCbydtIndex], fZ[maxdCbydtIndex], fEZ, WA, self.mode, self.Cb[maxdCbydtIndex], self.Csp[maxdCbydtIndex])#dCbydT[nStars]#Sept 28, 2017 0.011sec
        #End While Loop
        return t_dets

    def sacrificeStarCbyT(self, sInds, t_dets, fZ, fEZ, WA, overheadTime):
        """Sacrifice the worst performing CbyT star
        Args:
            sInds[nStars] - indicies of stars in the list
            t_dets[nStars] - time to observe each star (in days)
            fZ[nStars] - zodiacal light for each target
            fEZ - 0 
            WA - inner working angle of the instrument
            overheadTime - overheadTime added to each observation
        Return:
            sInds[nStars] - indicies of stars in the list
            t_dets[nStars] - time to observe each star (in days)
            sacrificedStarTime - time to distribute in days
            fZ[nStars] - zodiacal light for each target        
        """
        CbyT = self.Completeness.comp_per_intTime(t_dets*u.d, self.TargetList, sInds, fZ, fEZ, WA, self.mode, self.Cb, self.Csp)/t_dets#takes 5 seconds to do 1 time for all stars

        sacrificeIndex = np.argmin(CbyT)#finds index of star to sacrifice

        #Need index of sacrificed star by this point
        sacrificedStarTime = t_dets[sacrificeIndex] + overheadTime#saves time being sacrificed
        sInds = np.delete(sInds,sacrificeIndex)
        t_dets = np.delete(t_dets,sacrificeIndex)
        fZ = np.delete(fZ,sacrificeIndex)
        self.Cb = np.delete(self.Cb,sacrificeIndex)
        self.Csp = np.delete(self.Csp,sacrificeIndex)
        return sInds, t_dets, sacrificedStarTime, fZ

    def generate_fZ(self, sInds):
        """Calculates fZ values for each star over an entire orbit of the sun
        Args:
            sInds[nStars] - indicies of stars to generate yearly fZ for
        Returns:
            fZ[resolution, sInds] where fZ is the zodiacal light for each star
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))#find current directory of survey Simualtion
        fname = '/cachedfZ.csv'
        #Check File Length
        fileLength = 0
        try: 
            with open(dir_path+fname, 'rb') as f:
                reader = csv.reader(f)
                your_list = list(reader)
                fileLength = len(your_list)
                f.close()
        except:
            fileLength = 0
        #IF the Completeness vs dMag for Each Star File Does Not Exist, Calculate It
        if (not os.path.isfile(dir_path+fname) or (fileLength != sInds.shape[0])):#If this file does not exist or the length of the file is not appropriate 
            OS = self.OpticalSystem
            WA = OS.WA0
            ZL = self.ZodiacalLight
            TL = self.TargetList
            Obs = self.Observatory
            startTime = np.zeros(sInds.shape[0])*u.d + self.TimeKeeping.currentTimeAbs#Array of current times
            resolution = [j for j in range(1000)]
            fZ = np.zeros([len(resolution),sInds.shape[0]])
            dt = 365.25/len(resolution)*u.d
            for i in xrange(len(resolution)):#iterate through all times of year
                time = startTime + dt*resolution[i]
                fZ[i,:] = ZL.fZ(Obs, TL, sInds, time, self.mode)
            #This section of Code takes 68 seconds
            #Save fZ to File######################################
            try:#Here we delete the previous fZ file
                timeNow = datetime.datetime.now()
                timeString = str(timeNow.year)+'_'+str(timeNow.month)+'_'+str(timeNow.day)+'_'+str(timeNow.hour)+'_'+str(timeNow.minute)+'_'
                fnameNew = '/' + timeString +  'moved_fZAllStars.csv'
                os.rename(dir_path+fname,dir_path+fnameNew)
            except OSError:
                pass
            with open(dir_path+fname, "wb") as fo:
                wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
                for i in range(sInds.shape[0]):#iterate through all stars
                    wr.writerow(fZ[:,i])#Write the fZ to file
                fo.close()
        
        #Load fZ dMag for Each Star From File######################################
        #Sept 20, 2017 execution time 1.747 sec
        fZ = list()
        with open(dir_path+fname, 'rb') as f:
            reader = csv.reader(f)
            your_list = list(reader)
            f.close()
        for i in range(len(your_list)):
            tmp = np.asarray(your_list[i]).astype(np.float)
            fZ.append(tmp)
        return fZ


