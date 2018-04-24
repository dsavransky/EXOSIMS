from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
from numpy import nan
import scipy
#DELETEimport csv
import os.path
from astropy.coordinates import SkyCoord
try:
    import cPickle as pickle
except:
    import pickle
from pylab import *

class starkAYO_staticSchedule(SurveySimulation):
    """starkAYO _static Scheduler
    
    This class implements a Scheduler that creates a list of stars to observe and integration times to observe them. It also selects the best star to observe at any moment in time

    2nd execution time 2 min 30 sec
    """
    def __init__(self, cacheOptTimes=False, **specs):
        SurveySimulation.__init__(self, **specs)

        assert isinstance(cacheOptTimes, bool), 'cacheOptTimes must be boolean.'
        self._outspec['cacheOptTimes'] = cacheOptTimes

        #Load cached Observation Times
        self.starkt0 = None
        if cacheOptTimes:#Checks if flag to load cached optimal times exists
            cachefname = self.cachefname + 'starkt0'  # Generate cache Name
            if os.path.isfile(cachefname):#check if file exists
                self.vprint("Loading cached t0 from %s"%cachefname)
                with open(cachefname, 'rb') as f:#load from cache
                    self.starkt0 = pickle.load(f)
                sInds = np.arange(self.TargetList.nStars)

        # bring inherited class objects to top level of Survey Simulation
        SU = self.SimulatedUniverse
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        #self.Completeness = SU.Completeness
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        #Create and start Schedule
        self.schedule = np.arange(TL.nStars)#self.schedule is meant to be editable
        self.schedule_startSaved = np.arange(TL.nStars)#preserves initial list of targets
          
        dMagLim = self.Completeness.dMagLim
        self.dmag_startSaved = np.linspace(1, dMagLim, num=1500,endpoint=True)

        sInds = self.schedule_startSaved.copy()
        dmag = self.dmag_startSaved
        WA = OS.WA0
        startTime = np.zeros(sInds.shape[0])*u.d + TK.currentTimeAbs

        #DELETE tovisit = np.zeros(sInds.shape[0], dtype=bool)
        #Generate fZ #no longer necessary because called by calcfZmin
        #DELETE ZL.fZ_startSaved = ZL.generate_fZ(sInds)#
        #Estimate Yearly fZmin###########################################
        #DELETEself.fZmin, self.fZminInds = self.calcfZmin(sInds,ZL.fZ_startSaved)
        self.fZmin, self.abdTimefZmin = ZL.calcfZmin(sInds, Obs, TL, TK, self.mode, self.cachefname)
        #Estimate Yearly fZmax###########################################
        #DELETEself.fZmax, self.fZmaxInds = self.calcfZmax(Obs,TL,TK,sInds,self.mode,ZL.fZ_startSaved)
        self.fZmax, self.abdTimefZmax = ZL.calcfZmax(sInds, Obs, TL, TK, self.mode, self.cachefname)
        #################################################################

        #CACHE Cb Cp Csp################################################Sept 20, 2017 execution time 10.108 sec
        #DELETEfZ = self.fZmin
        #DELETEfEZ = ZL.fEZ0#DELETE*np.ones(TL.nStars)
        #DELETE mode = self.mode#resolve this mode is passed into next_target
        #DELETE allModes = self.OpticalSystem.observingModes
        #DELETE det_mode = filter(lambda mode: mode['detectionMode'] == True, allModes)[0]
        #WE DO NOT NEED TO CALCULATE OVER EVERY DMAG
        Cp = np.zeros([sInds.shape[0],dmag.shape[0]])
        Cb = np.zeros(sInds.shape[0])
        Csp = np.zeros(sInds.shape[0])
        for i in xrange(dmag.shape[0]):
            Cp[:,i], Cb[:], Csp[:] = OS.Cp_Cb_Csp(TL, sInds, self.fZmin, ZL.fEZ0, dmag[i], WA, self.mode)
        self.Cb = Cb[:]/u.s#Cb[:,0]/u.s#note all Cb are the same for different dmags. They are just star dependent
        self.Csp = Csp[:]/u.s#Csp[:,0]/u.s#note all Csp are the same for different dmags. They are just star dependent
        #self.Cp = Cp[:,:] #This one is dependent upon dmag and each star
        ################################################################

        #Calculate Initial Integration Times###########################################
        maxCbyTtime = self.calcTinit(sInds, TL, self.fZmin, ZL.fEZ0, WA, self.mode)
        t_dets = maxCbyTtime#[sInds] #t_dets has length TL.nStars

        #LETS CHANGE T_DETS SO THAT IT REPRESENTS ALL STARS FOR NOW

        #Sacrifice Stars and then Distribute Excess Mission Time################################################Sept 28, 2017 execution time 19.0 sec
        #DELETE missionLength = (TK.missionLife.to(u.d)*TK.missionPortion).value#mission length in days
        overheadTime = Obs.settlingTime.value + OS.observingModes[0]['syst']['ohTime'].value#OH time in days
        while((sum(t_dets) + sInds.shape[0]*overheadTime) > (TK.missionLife*TK.missionPortion).to('day').value):#the sum of star observation times is still larger than the mission length
            sInds, t_dets, sacrificedStarTime= self.sacrificeStarCbyT(sInds, t_dets, self.fZmin[sInds], ZL.fEZ0, WA, overheadTime)
        self.vprint('Started with ' + str(TL.nStars) + ' sacrificed down to ' + str(sInds.shape[0]))

        if(sum(t_dets + sInds.shape[0]*overheadTime) > (TK.missionLife*TK.missionPortion).to('day').value):#There is some excess time
            sacrificedStarTime = (TK.missionLife*TK.missionPortion).to('day').value - (sum(t_dets) + sInds.shape[0]*overheadTime)#The amount of time the list is under the total mission Time
            t_dets = self.distributedt(sInds, t_dets, sacrificedStarTime, self.fZmin[sInds], ZL.fEZ0, WA)
        ###############################################################################

        #STARK AYO LOOP################################################################
        #DELETEsavedSumComp00 = np.zeros(sInds.shape[0])
        firstIteration = 1#checks if this is the first iteration.
        numits = 0#ensure an infinite loop does not occur. Should be depricated
        lastIterationSumComp  = -10000000 #this is some ludacrisly negative number to ensure sumcomp runs. All sumcomps should be positive
        while numits < 100000 and sInds is not None:
            numits = numits+1#we increment numits each loop iteration

            #Sacrifice Lowest Performing Star################################################Sept 28, 2017 execution time 0.0744 0.032 at smallest list size
            sInds, t_dets, sacrificedStarTime = self.sacrificeStarCbyT(sInds, t_dets, self.fZmin[sInds], ZL.fEZ0, WA, overheadTime)

            #Distribute Sacrificed Time to new star observations############################# Sept 28, 2017 execution time 0.715, 0.64 at smallest (depends on # stars)
            t_dets = self.distributedt(sInds, t_dets, sacrificedStarTime, self.fZmin[sInds], ZL.fEZ0, WA)

            #AYO Termination Conditions###############################Sept 28, 2017 execution time 0.033 sec
            Comp00 = self.Completeness.comp_per_intTime(t_dets*u.d, TL, sInds, self.fZmin[sInds], ZL.fEZ0, WA, self.mode, self.Cb[sInds], self.Csp[sInds])

            #change this to an assert
            if 1 >= len(sInds):#if this is the last ement in the list
                break
            #DELETEsavedSumComp00[numits-1] = sum(Comp00)
            #If the total sum of completeness at this moment is less than the last sum, then exit
            if(sum(Comp00) < lastIterationSumComp):#If sacrificing the additional target reduced performance, then Define Output of AYO Process
                CbyT = self.Completeness.comp_per_intTime(t_dets*u.d, self.TargetList, sInds, self.fZmin[sInds], ZL.fEZ0, WA, self.mode, self.Cb[sInds], self.Csp[sInds])/t_dets#takes 5 seconds to do 1 time for all stars
                sortIndex = np.argsort(CbyT,axis=-1)[::-1]

                #This is the static optimal schedule
                self.schedule = sInds[sortIndex]
                self.t_dets = t_dets[sortIndex]
                self.CbyT = CbyT[sortIndex]
                #self.fZ = fZ[sortIndex]
                self.Comp00 = Comp00[sortIndex]
                break
            else:#else set lastIterationSumComp to current sum Comp00
                lastIterationSumComp = sum(Comp00)
                self.vprint(str(numits) + ' SumComp ' + str(round(sum(Comp00),2)) + ' Sum(t_dets) ' + str(round(sum(t_dets),2)) + ' sInds ' + str(sInds.shape[0]) + ' TimeConservation ' + str(round(sum(t_dets)+sInds.shape[0]*overheadTime,2)))# + ' Avg C/T ' + str(np.average(CbyT)))
        
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
            sInd - the single index of self.schedule_startSaved to observe
            waitTime - a strategic amount of time to wait (this module always returns None)
        """
        SU = self.SimulatedUniverse
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping

        #indexFrac = np.interp((TK.currentTimeNorm).value%365.25,[0,365.25],[0,1000])#float from 0 to 1000 of where at in 1 year
        #tmp = np.asarray(ZL.fZ_startSaved)[self.schedule,:]
        #fZ_matrixSched = (indexFrac%1)*tmp[:,int(indexFrac)] + (1-indexFrac%1)*tmp[:,int(indexFrac%1+1)]#A simple interpolant

        #fZ_matrixSched = ZL.fZ(Obs, TL, sInds, TK.currentTimeAbs, self.mode)
        fZ_matrixSched = np.zeros(TL.nStars)
        fZ_matrixSched[sInds] = ZL.fZ(Obs, TL, sInds, TK.currentTimeAbs, self.mode)
        #fZ_matrixSched = np.asarray(ZL.fZ_startSaved)[self.schedule,indexFrac]#has shape [self.schedule.shape[0], 1000]
        #The above line might not work because it must be filtered down one index at a time...
        fZminSched = np.zeros(TL.nStars)
        fZminSched[sInds] = self.fZmin[sInds].value #has shape [self.schedule.shape[0]]

        commonsInds = np.intersect1d(self.schedule,sInds)

        #commonsInds = [x for x in self.schedule if x in sInds]#finds indicies in common between sInds and self.schedule (we need the inherited filtering from sInds)
        #indmap = [self.schedule.tolist().index(x) for x in commonsInds]#get index of schedule for the Inds in self.schedule and sInds
        #indmap = [x for x in np.arange(self.schedule.shape[0]) if self.schedule[x] in sInds]
        indmap1 = [self.schedule.tolist().index(x) for x in self.schedule if x in sInds]#maps self.schedule defined to Inds ##find indicies of occurence of commonsInds in self.schedule
        indmap2 = [sInds.tolist().index(x) for x in sInds if x in self.schedule]# maps sInds defined to Inds
        #indmap = [x for x in self.schedule if x in sInds]

        #CbyT = np.zeros(TL.nStars)
        #CbyT[self.schedule] = self.CbyT
        CbyT = self.CbyT[indmap1]
        #t_dets = np.zeros(TL.nStars)
        #t_dets[self.schedule] = self.t_dets
        t_dets = self.t_dets[indmap1]
        #Comp00 = np.zeros(TL.nStars)
        #Comp00[self.schedule] = self.Comp00
        Comp00 = self.Comp00[indmap1]

        
        fZ = fZ_matrixSched[indmap2]#DONE
        fZmin = fZminSched[indmap2]#DONE

        # ###############################################################################################
        # # now, start to look for available targets
        # #DELETE dmag = self.dmag_startSaved
        # #DELETE WA = OS.WA0
        # #DELETE startTime = np.zeros(sInds.shape[0])*u.d + TK.currentTimeAbs

        # #Estimate Yearly fZmin###########################################
        # #DELETE tmpfZ = np.asarray(ZL.fZ_startSaved)
        # #GOOD
        # fZ_matrixSched = np.asarray(ZL.fZ_startSaved)[self.schedule,:]#has shape [self.schedule.shape[0], 1000]
        # #Find minimum fZ of each star
        # # fZmintmp = np.zeros(self.schedule.shape[0])
        # # for i in xrange(self.schedule.shape[0]):
        # #     fZmintmp[i] = min(fZ_matrix[i,:])
        # #GOOD
        # fZminSched = self.fZmin[self.schedule].value #has len self.schedule.shape[0]
        # #Find current fZ
        # # indexFrac = np.interp((TK.currentTimeAbs-TK.missionStart).value%365.25,[0,365.25],[0,1000])#This is only good for 1 year missions right now
        # # fZinterp = np.zeros(self.schedule.shape[0])
        # # fZinterp[:] = (indexFrac%1)*fZ_matrix[:,int(indexFrac)] + (1-indexFrac%1)*fZ_matrix[:,int(indexFrac%1+1)]#this is the current fZ
        # #GOOD
        # indexFrac = int(np.interp((TK.currentTimeNorm).value%365.25,[0,365.25],[0,1000]))#This is only good for 1 year missions right now
        # fZinterp = np.zeros(self.schedule.shape[0])
        # fZinterp[:] = (indexFrac%1)*fZ_matrixSched[:,int(indexFrac)] + (1-indexFrac%1)*fZ_matrixSched[:,int(indexFrac%1+1)]#this is the current fZ dimensions [self.schedule.shape[0], 1000]

        # commonsInds = [x for x in self.schedule if x in sInds]#finds indicies in common between sInds and self.schedule (we need the inherited filtering from sInds)
        # imat = [self.schedule.tolist().index(x) for x in commonsInds]#get index of schedule for the Inds in self.schedule and sInds
        # CbyT = self.CbyT[imat]
        # t_dets = self.t_dets[imat]
        # Comp00 = self.Comp00[imat]
        # fZ = fZinterp[imat]
        # fZmin = fZminSched[imat]

        #commonsInds2 = [x for x in self.schedule_startSaved if((x in sInds) and (x in self.schedule))]#finds indicies in common between sInds and self.schedule
        #indmap2 = [self.schedule_startSaved.tolist().index(x) for x in commonsInds]
        tmp = TL.coords.dec[self.schedule].value
        dec = tmp[indmap1]

        # #currentTime = TK.currentTimeAbs
        # r_targ = TL.starprop(np.asarray(indmap2).astype(int),TK.currentTimeAbs,False)
        # #dec = np.zeros(len(imat2))
        # #for i in np.arange(len(imat2)):
        # c = SkyCoord(r_targ[:,0],r_targ[:,1],r_targ[:,2],representation='cartesian')
        # c.representation = 'spherical'
        # dec = c.dec

        #print saltyburrito
        if len(indmap1) > 0:
            # store selected star integration time
            selectInd = np.argmin(Comp00*abs(fZ-fZmin)/abs(dec))
            sInd = self.schedule[indmap1[selectInd]]
            #sInd = sInds[selectInd]#finds index of star to sacrifice
            #t_det = t_dets[selectInd]*u.d
            
            #assert intTimes[indmap1[selectInd]] != 0*u.d
            return sInd, None
        else: # return a strategic amount of time to wair
            return None, 1*u.d
        
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
        #commonsInds = [val for val in self.schedule if val in sInds]#finds indicies in common between sInds and self.schedule

        imat = [self.schedule.tolist().index(x) for x in self.schedule if x in sInds]#find indicies of occurence of commonsInds in self.schedule
        intTimes = np.zeros(self.TargetList.nStars)#default observation time is 0 days
        intTimes[self.schedule[imat]] = self.t_dets[imat]#
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
            t_dets[len(sInds)] - time to observe each star (in days)
        """
        #Calculate dCbydT for each star at this point in time
        dCbydt = self.Completeness.dcomp_dt(t_dets*u.d, self.TargetList, sInds, fZ, fEZ, WA, self.mode, self.Cb[sInds], self.Csp[sInds])#dCbydT[len(sInds)]#Sept 28, 2017 0.12sec

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
        """
        CbyT = self.Completeness.comp_per_intTime(t_dets*u.d, self.TargetList, sInds, self.fZmin[sInds], fEZ, WA, self.mode, self.Cb[sInds], self.Csp[sInds])/t_dets#takes 5 seconds to do 1 time for all stars

        sacrificeIndex = np.argmin(CbyT)#finds index of star to sacrifice

        #Need index of sacrificed star by this point
        sacrificedStarTime = t_dets[sacrificeIndex] + overheadTime#saves time being sacrificed
        sInds = np.delete(sInds,sacrificeIndex)
        t_dets = np.delete(t_dets,sacrificeIndex)
        #DELETEfZ = np.delete(fZ,sacrificeIndex)
        #DELETEself.Cb = np.delete(self.Cb,sacrificeIndex)
        #DELETEself.Csp = np.delete(self.Csp,sacrificeIndex)
        return sInds, t_dets, sacrificedStarTime

    def calcTinit(self, sInds, TL, fZ, fEZ, WA, mode):
        """ Calculate Initial Values for starkAYO (use max C/T) <- this is a poor IC selection
        """
        cachefname = self.cachefname + 'maxCbyTt0'  # Generate cache Name

        #Check if file exists#######################################################################
        if os.path.isfile(cachefname):#check if file exists
            self.vprint("Loading cached maxCbyTt0 from %s"%cachefname)
            with open(cachefname, 'rb') as f:#load from cache
                maxCbyTtime = pickle.load(f)
            return maxCbyTtime
        ###########################################################################################

        self.vprint("Calculating maxCbyTt0")
        maxCbyTtime = np.zeros(sInds.shape[0])#This contains the time maxCbyT occurs at
        maxCbyT = np.zeros(sInds.shape[0])#this contains the value of maxCbyT
        #Solve Initial Integration Times###############################################
        def CbyTfunc(t_dets, self, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp):
            CbyT = -self.Completeness.comp_per_intTime(t_dets*u.d, TL, sInds, fZ, fEZ, WA, mode, Cb, Csp)/t_dets*u.d
            return CbyT.value

        
        # t_dets = np.logspace(-5,1,num=100,base=10)#(np.arange(0,100)+1)/10.
        # fig = figure(1)
        # i=1
        # CbyTfuncvals = np.zeros(len(t_dets))
        # compvals = np.zeros(len(t_dets))
        # for j in np.arange(len(t_dets)):
        #     compvals[j] = self.Completeness.comp_per_intTime(t_dets[j]*u.d, TL, sInds[i], fZ[i], fEZ, WA, mode, self.Cb[i], self.Csp[i])
        #     #CbyTfuncvals[j] = CbyTfunc(t_dets[j], self, TL, sInds[i], fZ[i], fEZ, WA, mode, self.Cb[i], self.Csp[i])
        # plot(t_dets,compvals)
        # xscale('log')
        # show(block=False)


        #Calculate Maximum C/T
        for i in xrange(sInds.shape[0]):
            x0 = 0.5
            retVals = scipy.optimize.fmin(CbyTfunc, x0, args=(self, TL, sInds[i], fZ[i], fEZ, WA, mode, self.Cb[i], self.Csp[i]), xtol=1e-8, ftol=1e-8, disp=True)
            maxCbyTtime[i] = retVals[0]
            #DELETE if i in [1,2,3] and maxCbyTtime[i] == 0.5:
            #DELETE     print(saltyburrito)
            self.vprint("Max C/T calc completion: " + str(float(i)/sInds.shape[0]) + ' ' + str(maxCbyTtime[i]))
        #Sept 27, Execution time 101 seconds for 651 stars
        #DELETE if maxCbyTtime[0] == 0.5 or maxCbyTtime[1] == 0.5 or maxCbyTtime[2] == 0.5:#print statement to tell me if the values being returned are silly
        #DELETE     print saltyburrito

        with open(cachefname, "wb") as fo:
            #DELETEwr = csv.writer(fo, quoting=csv.QUOTE_ALL)
            pickle.dump(maxCbyTtime,fo)
            self.vprint("Saved cached 1st year Tinit to %s"%cachefname)
        return maxCbyTtime
