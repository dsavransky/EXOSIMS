from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import scipy
from scipy.optimize import fmin
#import timeit
import csv
import os.path

class starkAYO(SurveySimulation):
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
        #END INIT##################################################################

    def next_target(self, sInds, mode):
        """Generate Next Target to Select based off of AYO at this instant in time
        Args:
            sInds - unnecessary
            mode - 

        Returns:
            DRM - A blank structure
            sInd - the single index of self.schedule_startSaved to observe
            t_det - the time to observe sInd in days (u.d)
        """
        sInds = self.schedule_startSaved

        SU = self.SimulatedUniverse
        OS = SU.OpticalSystem
        ZL = SU.ZodiacalLight
        self.Completeness = SU.Completeness
        TL = SU.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        TK.obsStart = TK.currentTimeNorm.to('day')

        dmag = self.dmag_startSaved
        WA = OS.WA0
        startTime = np.zeros(sInds.shape[0])*u.d + self.TimeKeeping.currentTimeAbs

        tovisit = np.zeros(sInds.shape[0], dtype=bool)

        DRM = {}#Create DRM

        #Filter KOGOODSTART########################################################
        kogoodStart = Obs.keepout(TL, sInds, startTime, mode)#outputs array where 0 values are good keepout stars
        sInds = sInds[np.where(kogoodStart)[0]]
        ###########################################################################
        
        #Filter out previously visited stars#######################################Aug 28, 2017 execution time 0.235
        if np.any(sInds):
            tovisit[sInds] = (self.starVisits[sInds] == self.starVisits[sInds].min())
            if self.starRevisit.size != 0:
                dt_max = 1.*u.week
                dt_rev = np.abs(self.starRevisit[:,1]*u.day - TK.currentTimeNorm)
                ind_rev = [int(x) for x in self.starRevisit[dt_rev < dt_max,0] if x in sInds]
                tovisit[ind_rev] = True
            sInds = np.where(tovisit)[0]
        #########################################################################
        startTime = np.zeros(sInds.shape[0])*u.d + self.TimeKeeping.currentTimeAbs
        
        #CACHE Cb Cp Csp################################################Sept 20, 2017 execution time 10.108 sec
        fZ = ZL.fZ(Obs, TL, sInds, startTime, self.mode)#0./u.arcsec**2#np.zeros(sInds.shape[0]) + 0./u.arcsec**2
        fEZ = ZL.fEZ0#0./u.arcsec**2#
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
        missionLength = (TK.missionFinishAbs-TK.currentTimeAbs).value#TK.missionLife.to(u.day).value#mission length in days
        overheadTime = self.Observatory.settlingTime.value + self.OpticalSystem.observingModes[0]['syst']['ohTime'].value#OH time in days
        while((sum(t_dets) + sInds.shape[0]*overheadTime) > missionLength):#the sum of star observation times is still larger than the mission length
            sInds, t_dets, sacrificedStarTime, fZ = self.sacrificeStarCbyT(sInds, t_dets, fZ, fEZ, WA, overheadTime)

        if(sum(t_dets + sInds.shape[0]*overheadTime) > missionLength):#There is some excess time
            sacrificedStarTime = missionLength - (sum(t_dets) + sInds.shape[0]*overheadTime)#The amount of time the list is under the total mission Time
            t_dets = self.distributedt(sInds, t_dets, sacrificedStarTime, fZ, fEZ, WA)
        ###############################################################################

        #STARK AYO LOOP################################################################
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
                #print(saltyburrito)
                break

            # # #If the total sum of completeness at this moment is less than the last sum, then exit
            if(sum(Comp00) < lastIterationSumComp):#If sacrificing the additional target reduced performance, then Define Output of AYO Process
                CbyT = self.Completeness.comp_per_intTime(t_dets*u.d, self.TargetList, sInds, fZ, fEZ, WA, self.mode, self.Cb, self.Csp)/t_dets#takes 5 seconds to do 1 time for all stars
                selectIndex = np.argmax(CbyT)#finds index of star to sacrifice
                sInd = sInds[selectIndex]

                t_det = t_dets[selectIndex]*u.d
                
                self.starVisits[sInd] += 1#Update List of Visited stars########
                TK.allocate_time(Obs.settlingTime + mode['syst']['ohTime'])# Aluserslocate settling time + overhead time
                return DRM, sInd, t_det
            else:#else set lastIterationSumComp to current sum Comp00
                lastIterationSumComp = sum(Comp00)
                print(str(numits) + ' SumComp ' + str(sum(Comp00)) + ' Sum(t_dets) ' + str(sum(t_dets)) + ' sInds ' + str(sInds.shape[0]*float(1)) + ' TimeConservation ' + str(sum(t_dets)+sInds.shape[0]*float(1)))# + ' Avg C/T ' + str(np.average(CbyT)))
        #End While Loop

    def choose_next_target(self,old_sInd,sInds,slewTime):
        """Select Highest C/T target
        Args:
            old_sInd - unnecessary
            sInds - star indicies being considered
            slewTime - time to slew to all stars
        Returns:
            sInd - indicie of the star to observe
        """
        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping
        
        comps = TL.comp0[sInds]#completeness of each star in TL
        updated = (self.starVisits[sInds] > 0)#what does starVisits contain?
        comps[updated] = Comp.completeness_update(TL, sInds[updated],TK.currentTimeNorm)
        tint = TL.tint0[sInds]
        
        selMetric=comps/tint#selMetric is the selection metric being used. Here it is Completeness/integration time

        tmp = sInds[selMetric == max(selMetric)]#this selects maximum completeness/integration time
        sInd = tmp[0]#casts numpy array to single integer
        
        return sInd
  
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
        dCbydt = self.Completeness.dcomp_dt(t_dets*u.d, self.TargetList, sInds, fZ, fEZ, WA, self.mode, self.Cb, self.Csp)#dCbydT[nStars]

        if(len(t_dets) <= 1):
            print('t_dets has one element or less')
            return t_dets

        timeToDistribute = sacrificedStarTime

        #Decide Size of dt
        if(sacrificedStarTime/50.0 < 0.1):
            dt_static = sacrificedStarTime/50.0#1
        else:
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
            dCbydt[maxdCbydtIndex] = self.Completeness.dcomp_dt(t_dets[maxdCbydtIndex]*u.d, self.TargetList, sInds[maxdCbydtIndex], fZ[maxdCbydtIndex], fEZ, WA, self.mode, self.Cb[maxdCbydtIndex], self.Csp[maxdCbydtIndex])#dCbydT[nStars]
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
