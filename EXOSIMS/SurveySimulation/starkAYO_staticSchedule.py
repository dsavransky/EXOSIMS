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
    """starkAYO 
    
    This class implements a Scheduler that selects the current highest Completeness/Integration Time.
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
        #fZ = np.zeros(sInds.shape[0]) + 0./u.arcsec**2
        #fEZ = 0./u.arcsec**2#
        #fZ = np.zeros(sInds.shape[0]) + ZL.fZ0
        fZ = fZmin/u.arcsec**2#
        fEZ = ZL.fEZ0
        mode = self.mode#resolve this mode is passed into next_target
        allModes = self.OpticalSystem.observingModes
        det_mode = filter(lambda mode: mode['detectionMode'] == True, allModes)[0]
        #dmag = self.dmag_startSaved
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

        #Distribute Excess Mission Time################################################Sept 28, 2017 execution time 19.0 sec
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
                #print('sInds maximally sorted (This probably indicates an error)')
                break
            savedSumComp00[numits-1] = sum(Comp00)
            # # #If the total sum of completeness at this moment is less than the last sum, then exit
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
                print(str(numits) + ' SumComp ' + str(sum(Comp00)) + ' Sum(t_dets) ' + str(sum(t_dets)) + ' sInds ' + str(sInds.shape[0]*float(1)) + ' TimeConservation ' + str(sum(t_dets)+sInds.shape[0]*float(1)))# + ' Avg C/T ' + str(np.average(CbyT)))
        #End While Loop

        #Save Schedule Attributes to File######################################
        # print(self.schedule.shape)
        # print(self.t_dets.shape)
        # print(self.CbyT.shape)
        # print(self.fZ.shape)
        # print(TL.Name[self.schedule].shape)
        # print(TL.coords.ra[self.schedule].shape)
        # print(TL.coords.dec[self.schedule].shape)
        # #Calculate fZminTimes
        # print(fZminInds[self.schedule].shape)
        # fZminTimes = np.interp(fZminInds[self.schedule],[0,1000],[0,365.25])#BackCalculate Times fZmin occur at
        # print(fZminTimes.shape)
        # #completeness
        # print(Comp00[sortIndex].shape)

        # fname = '/starkAYO24mo.csv'
        # with open(dir_path+fname, "wb") as fo:
        #     wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
        #     wr.writerow(self.schedule)#1
        #     wr.writerow(self.t_dets)#2#in days
        #     wr.writerow(self.CbyT)#3#in Completeness/days
        #     wr.writerow(self.fZ.value)#fZmin#4 #in 1/arcsec^2
        #     wr.writerow(TL.Name[self.schedule])#4
        #     wr.writerow(TL.coords.ra[self.schedule].value)#4#in  deg
        #     wr.writerow(TL.coords.dec[self.schedule].value)#5#in deg
        #     wr.writerow(fZminTimes)#6
        #     wr.writerow(Comp00[sortIndex])#7 Completeness
        #     #wr.writerow(fZminInds[self.schedule])
        #     fo.close()
        #
        #END INIT##################################################################
        
    def choose_next_target(self,old_sInd,sInds,slewTime):
        """Generate Next Target to Select based off of AYO at this instant in time
        Args:
            sInds - unnecessary
            old_sInd - unused

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
            #SU = self.SimulatedUniverse
            #OS = SU.OpticalSystem
            #ZL = SU.ZodiacalLight
            #self.Completeness = SU.Completeness
            #TL = SU.TargetList
            #Obs = self.Observatory
            #TK = self.TimeKeeping
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
            #print('indexFrac is ' + str(indexFrac))
            fZ = np.zeros(sInds.shape[0])
            fZ[:] = (indexFrac%1)*fZ_matrix[:,int(indexFrac)] + (1-indexFrac%1)*fZ_matrix[:,int(indexFrac+1)]#this is the current fZ

            #Find sInds in common between self.schedule and sInds
            CbyT = np.zeros(sInds.shape[0])
            t_dets = np.zeros(sInds.shape[0])
            dec = np.zeros(sInds.shape[0])
            Comp00 = np.zeros(sInds.shape[0])
            cond = np.zeros((self.schedule.shape[0],), dtype=bool)#for schedule inds
            cond2 = np.zeros((self.schedule_startSaved.shape[0],), dtype=bool)#for start_saved lists
            index = 0#a temporary value
            for i in range(sInds.shape[0]):
                cond = (self.schedule-sInds[i]) == 0
                CbyT[i] = self.CbyT[cond]
                t_dets[i] = self.t_dets[cond]
                Comp00[i] = self.Comp00[cond]

                cond2 = (self.schedule_startSaved-sInds[i]) == 0
                dec[i] = self.TargetList.coords.dec[cond2].value
        

            #sInd = sInds[np.argmax(CbyT)]#finds index of star to sacrifice
            #t_det = self.t_dets[np.argmax(CbyT)]*u.d
            if len(sInds) > 0:
                # store selected star integration time

                sInd = sInds[np.argmin(Comp00*abs(fZ-fZmin)*abs(dec))]#finds index of star to sacrifice
                t_det = t_dets[np.argmin(Comp00*abs(fZ-fZmin)*abs(dec))]*u.d
                Comp00_single = Comp00[np.argmin(Comp00*abs(fZ-fZmin)*abs(dec))]
                print(Comp00_single)
                #print(saltyburrito)
                #Comp00 = self.Completeness.comp_per_intTime(t_det*u.d, TL, sInd, fZ[sInd], fEZ, WA, mode, self.Cb[sInd], self.Csp[sInd])

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


            # sInd = sInds[np.argmin(abs(fZ-fZmin)*abs(dec))]#finds index of star to sacrifice
            # t_det = self.t_dets[np.argmin(abs(fZ-fZmin)*abs(dec))]*u.d

            # self.starVisits[sInd] += 1#Update List of Visited stars########
            # TK.allocate_time(Obs.settlingTime + mode['syst']['ohTime'])# Aluserslocate settling time + overhead time

            # return DRM, sInd, t_det

        else:
            return None#DRM, None, None
        return sInd

    def calc_targ_intTime(self, sInds, startTimes, mode):
        """Helper method for next_target to aid in overloading for alternative implementations.

        Given a subset of targets, calculate their integration times given the
        start of observation time.

        Prototype just calculates integration times for fixed contrast depth.  

        Note: next_target filter will discard targets with zero integration times.
        
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
        
        #Trash startTimes, modemysInds
        #sInds_schedule = self.schedule
        #I need the indicies of the sInds in common between self.schedule and sInds
        #indiciesInCommon = [x for x in self.schedule if x in sInds]
        
        commonInds = [self.schedule[i] for i in np.arange(len(self.schedule)) if self.schedule[i] in sInds]
        intTimes = np.zeros(self.TargetList.nStars)
        intTimes[commonInds] = [self.t_dets[i] for i in np.arange(len(self.schedule)) if self.schedule[i] in sInds]
        intTimes = intTimes*u.d
        
        #notInCommonInds = [sInds[i] for i in np.arange(len(sInds)) if sInds[i] not in self.schedule]#unnecessary Just default all others to 0
        
        #mysInds = [x for x in self.schedule if x in sInds]
        #indicies = [np.where(self.schedule == x)[0][0] for x in mysInds]

        #intTimes = np.zeros(TL.nStars)*u.d
        #intTimes[commonInds] = self.t_dets[commonInds]

        #intTimes has length TL.nStars
        #intTimes[sInds] has length sInds
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
            print('t_dets has one element or less')
            return t_dets

        timeToDistribute = sacrificedStarTime

        #Decide Size of dt
        #if(sacrificedStarTime/50.0 < 0.1):
        #    dt_static = sacrificedStarTime/50.0#1
        #else:
        dt_static = 0.1
        #dt_static = 0.01
        dt = dt_static

        #Now decide where to put dt
        numItsDist = 0
        while(timeToDistribute > 0):
            if(numItsDist > 1000000):#this is an infinite loop check
                print('numItsDist>1000000')
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
        """
        This function calculates fZ values for each star over an entire orbit of the sun
        This function is called in init
        returns: fZ[resolution, sInds] where fZ is the zodiacal light for each star
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
        #print('fileLength is ' + str(fileLength))
        #IF the Completeness vs dMag for Each Star File Does Not Exist, Calculate It
        if (not os.path.isfile(dir_path+fname) or (fileLength != sInds.shape[0])):#If this file does not exist or the length of the file is not appropriate 
            print('Calculating fZ for Each Star over 1 Yr')
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
                print('Trying to save fZ vs Time for Each Star to File')
                timeNow = datetime.datetime.now()
                timeString = str(timeNow.year)+'_'+str(timeNow.month)+'_'+str(timeNow.day)+'_'+str(timeNow.hour)+'_'+str(timeNow.minute)+'_'
                fnameNew = '/' + timeString +  'moved_fZAllStars.csv'
                os.rename(dir_path+fname,dir_path+fnameNew)
            except OSError:
                print('There was an error writing the moved_fZAllStars file')
                pass
            with open(dir_path+fname, "wb") as fo:
                wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
                for i in range(sInds.shape[0]):#iterate through all stars
                    wr.writerow(fZ[:,i])#Write the fZ to file
                fo.close()
                print('Finished Saving fZ vs ToY for Each Star to File')
        
        #Load fZ dMag for Each Star From File######################################
        #Sept 20, 2017 execution time 1.747 sec
        print('Load fZ for Each Star from File')
        fZ = list()
        with open(dir_path+fname, 'rb') as f:
            reader = csv.reader(f)
            your_list = list(reader)
            f.close()
        for i in range(len(your_list)):
            tmp = np.asarray(your_list[i]).astype(np.float)
            fZ.append(tmp)
        return fZ


