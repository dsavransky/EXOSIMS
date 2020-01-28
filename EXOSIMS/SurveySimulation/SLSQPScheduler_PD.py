from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import astropy.constants as const
import numpy as np
from ortools.linear_solver import pywraplp
from scipy.optimize import minimize,minimize_scalar
import os
import time
try:
   import cPickle as pickle
except:
   import pickle
from astropy.time import Time

class SLSQPScheduler_PD(SurveySimulation):
    """SLSQPScheduler
    
    This class implements a continuous optimization of integration times
    using the scipy minimize function with method SLSQP.  ortools with the CBC 
    linear solver is used to find an initial solution consistent with the constraints.
    For details see Keithly et al. 2019. Alternatively: Savransky et al. 2017 (SPIE).

    Args:         
        \*\*specs:
            user specified values

    Notes:
        Due to the time costs of the current comp_per_inttime calculation in GarrettCompleteness
        this should be used with BrownCompleteness.

        Requires ortools
    
    """

    def __init__(self, cacheOptTimes=False, staticOptTimes=False, selectionMetric='maxC', Izod='current',
        maxiter=60, ftol=1e-3, **specs): #fZminObs=False,
        
        #initialize the prototype survey
        SurveySimulation.__init__(self, **specs)

        #Calculate fZmax
        self.valfZmax, self.absTimefZmax = self.ZodiacalLight.calcfZmax(np.arange(self.TargetList.nStars), self.Observatory, self.TargetList,
            self.TimeKeeping, list(filter(lambda mode: mode['detectionMode'] == True, self.OpticalSystem.observingModes))[0], self.cachefname)

        assert isinstance(staticOptTimes, bool), 'staticOptTimes must be boolean.'
        self.staticOptTimes = staticOptTimes
        self._outspec['staticOptTimes'] = self.staticOptTimes

        assert isinstance(cacheOptTimes, bool), 'cacheOptTimes must be boolean.'
        self._outspec['cacheOptTimes'] = cacheOptTimes

        assert selectionMetric in ['maxC','Izod-Izodmin','Izod-Izodmax',
            '(Izod-Izodmin)/(Izodmax-Izodmin)',
            '(Izod-Izodmin)/(Izodmax-Izodmin)/CIzod', #(Izod-Izodmin)/(Izodmax-Izodmin)/CIzodmin is simply this but with Izod='fZmin'
            'TauIzod/CIzod', #TauIzodmin/CIzodmin is simply this but with Izod='fZmin'
            'random',
            'priorityObs'], 'selectionMetric not valid input' # Informs what selection metric to use
        self.selectionMetric = selectionMetric
        self._outspec['selectionMetric'] = self.selectionMetric

        assert Izod in ['fZmin','fZ0','fZmax','current'], 'Izod not valid input' # Informs what Izod to optimize integration times for [fZmin, fZmin+45d, fZ0, fZmax, current]
        self.Izod = Izod
        self._outspec['Izod'] = self.Izod

        assert isinstance(maxiter, int), 'maxiter is not an int' # maximum number of iterations to optimize integration times for
        assert maxiter >= 1, 'maxiter must be positive real'
        self.maxiter = maxiter
        self._outspec['maxiter'] = self.maxiter

        assert isinstance(ftol, float), 'ftol must be boolean' # tolerance to place on optimization
        assert ftol > 0, 'ftol must be positive real'
        self.ftol = ftol
        self._outspec['ftol'] = self.ftol


        #some global defs
        self.detmode = list(filter(lambda mode: mode['detectionMode'] == True, self.OpticalSystem.observingModes))[0]
        self.ohTimeTot = self.Observatory.settlingTime + self.detmode['syst']['ohTime'] # total overhead time per observation
        self.maxTime = self.TimeKeeping.missionLife*self.TimeKeeping.missionPortion # total mission time

        self.constraints = {'type':'ineq',
                            'fun': lambda x: self.maxTime.to(u.d).value - np.sum(x[x*u.d > 0.1*u.s]) - #maxTime less sum of intTimes
                                             np.sum(x*u.d > 0.1*u.s).astype(float)*self.ohTimeTot.to(u.d).value, # sum of True -> goes to 1 x OHTime
                            'jac':lambda x: np.ones(len(x))*-1.}

        self.t0 = None
        if cacheOptTimes:
            #Generate cache Name########################################################################
            cachefname = self.cachefname + 't0'
            
            if os.path.isfile(cachefname):
                self.vprint("Loading cached t0 from %s"%cachefname)
                with open(cachefname, 'rb') as f:
                    try:
                        self.t0 = pickle.load(f)
                    except UnicodeDecodeError:
                        self.t0 = pickle.load(f,encoding='latin1')
                sInds = np.arange(self.TargetList.nStars)
                fZ = np.array([self.ZodiacalLight.fZ0.value]*len(sInds))*self.ZodiacalLight.fZ0.unit
                self.scomp0 = -self.objfun(self.t0.to('day').value,sInds,fZ)


        if self.t0 is None:
            #1. find nominal background counts for all targets in list
            dMagint = 25.0 # this works fine for WFIRST
            _, Cbs, Csps = self.OpticalSystem.Cp_Cb_Csp(self.TargetList, np.arange(self.TargetList.nStars),  
                    self.ZodiacalLight.fZ0, self.ZodiacalLight.fEZ0, dMagint, self.WAint, self.detmode)

            #find baseline solution with dMagLim-based integration times
            #3.
            t0 = self.OpticalSystem.calc_intTime(self.TargetList, np.arange(self.TargetList.nStars),  
                    self.ZodiacalLight.fZ0, self.ZodiacalLight.fEZ0, self.dMagint, self.WAint, self.detmode)
            #4.
            comp0 = self.Completeness.comp_per_intTime(t0, self.TargetList, self.TimeKeeping, np.arange(self.TargetList.nStars), 
                    self.ZodiacalLight.fZ0, self.ZodiacalLight.fEZ0, self.WAint, self.detmode, C_b=Cbs, C_sp=Csps)
            
            #### 5. Formulating MIP to filter out stars we can't or don't want to reasonably observe
            solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING) # create solver instance
            xs = [ solver.IntVar(0.0,1.0, 'x'+str(j)) for j in np.arange(len(comp0)) ] # define x_i variables for each star either 0 or 1
            self.vprint('Finding baseline fixed-time optimal target set.')

            #constraint is x_i*t_i < maxtime
            constraint = solver.Constraint(-solver.infinity(),self.maxTime.to(u.day).value) #hmmm I wonder if we could set this to 0,maxTime
            for j,x in enumerate(xs):
                constraint.SetCoefficient(x, t0[j].to('day').value + self.ohTimeTot.to(u.day).value) # this forms x_i*(t_0i+OH) for all i

            #objective is max x_i*comp_i
            objective = solver.Objective()
            for j,x in enumerate(xs):
                objective.SetCoefficient(x, comp0[j])
            objective.SetMaximization()

            #solver.EnableOutput()# this line enables output of the CBC MIXED INTEGER PROGRAM (Was hard to find don't delete)
            solver.SetTimeLimit(5*60*1000)#time limit for solver in milliseconds
            cpres = solver.Solve() # actually solve MIP
            x0 = np.array([x.solution_value() for x in xs]) # convert output solutions

            self.scomp0 = np.sum(comp0*x0) # calculate sum Comp from MIP
            self.t0 = t0 # assign calculated t0

            #Observation num x0=0 @ dMagint=25 is 1501
            #Observation num x0=0 @ dMagint=30 is 1501...

            #now find the optimal eps baseline and use whichever gives you the highest starting completeness
            self.vprint('Finding baseline fixed-eps optimal target set.')
            def totCompfeps(eps):
                compstars,tstars,x = self.inttimesfeps(eps, Cbs.to('1/d').value, Csps.to('1/d').value)
                return -np.sum(compstars*x)
            #Note: There is no way to seed an initial solution to minimize scalar 
            #0 and 1 are supposed to be the bounds on epsres. I could define upper bound to be 0.01, However defining the bounds to be 5 lets the solver converge
            epsres = minimize_scalar(totCompfeps,method='bounded',bounds=[0,7], options={'disp': 3, 'xatol':self.ftol, 'maxiter': self.maxiter})  #adding ftol for initial seed. could be different ftol
                #https://docs.scipy.org/doc/scipy/reference/optimize.minimize_scalar-bounded.html#optimize-minimize-scalar-bounded
            comp_epsmax,t_epsmax,x_epsmax = self.inttimesfeps(epsres['x'],Cbs.to('1/d').value, Csps.to('1/d').value)
            if np.sum(comp_epsmax*x_epsmax) > self.scomp0:
                x0 = x_epsmax
                self.scomp0 = np.sum(comp_epsmax*x_epsmax) 
                self.t0 = t_epsmax*x_epsmax*u.day

            ##### Optimize the baseline solution
            self.vprint('Optimizing baseline integration times.')
            sInds = np.arange(self.TargetList.nStars)
            if self.Izod == 'fZ0': # Use fZ0 to calculate integration times
                fZ = np.array([self.ZodiacalLight.fZ0.value]*len(sInds))*self.ZodiacalLight.fZ0.unit
            elif self.Izod == 'fZmin': # Use fZmin to calculate integration times
                fZ = self.valfZmin[sInds]
            elif self.Izod == 'fZmax': # Use fZmax to calculate integration times
                fZ = self.valfZmax[sInds]
            elif self.Izod == 'current': # Use current fZ to calculate integration times
                fZ = self.ZodiacalLight.fZ(self.Observatory, self.TargetList, sInds, self.TimeKeeping.currentTimeAbs.copy()+np.zeros(self.TargetList.nStars)*u.d, self.detmode)

            maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife = self.TimeKeeping.get_ObsDetectionMaxIntTime(self.Observatory, self.detmode, self.TimeKeeping.currentTimeNorm.copy())
            maxIntTime   = min(maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife) # Maximum intTime allowed
            bounds = [(0,maxIntTime.to(u.d).value) for i in np.arange(len(sInds))]
            initguess = x0*self.t0.to(u.d).value
            self.save_initguess = initguess


            #While we use all sInds as input, theoretically, this can be solved faster if we use the following lines:
            #sInds = np.asarray([sInd for sInd in sInds if np.bool(x0[sInd])])
            #bounds = [(0,maxIntTime.to(u.d).value) for i in np.arange(len(sInds))]
            #and use initguess[sInds], fZ[sInds], and self.t0[sInds].
            #There was no noticable performance improvement
            ires = minimize(self.objfun, initguess, jac=self.objfun_deriv, args=(sInds,fZ), 
                    constraints=self.constraints, method='SLSQP', bounds=bounds, options={'maxiter':self.maxiter, 'ftol':self.ftol, 'disp': True}) #original method

            assert ires['success'], "Initial time optimization failed."

            self.t0 = ires['x']*u.d
            self.scomp0 = -ires['fun']

            if cacheOptTimes:
                with open(cachefname,'wb') as f:
                    pickle.dump(self.t0, f)
                self.vprint("Saved cached optimized t0 to %s"%cachefname)

        #Redefine filter inds
        self.intTimeFilterInds = np.where((self.t0.value > 0.)*(self.t0.value <= self.OpticalSystem.intCutoff.value) > 0.)[0] # These indices are acceptable for use simulating    
        
        # This is used to keep track of the star_index and the time for planned observations
        self.mission_schedule = [] 
        self.initial_schedule = []
        self.second_schedule = []
        
        # This will be used to keep track of when the target optimization is made
        self.opt_times = []
        
    def run_sim(self):
        """Performs the survey simulation 
        
        """
        
        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # TODO: start using this self.currentSep
        # set occulter separation if haveOcculter
        if OS.haveOcculter == True:
            self.currentSep = Obs.occulterSep
        
        # Dictionary that keeps track of simulated planets not eliminated from a star
        # self.vprint('Resting the dynamic completeness DICTIONARY ALSKJD:ALKSJJJJJAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        self.Completeness.dc_dict = {}
        
        # choose observing modes selected for detection (default marked with a flag)
        allModes = OS.observingModes
        det_mode = list(filter(lambda mode: mode['detectionMode'] == True, allModes))[0]
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = list(filter(lambda mode: 'spec' in mode['inst']['name'], allModes))
        if np.any(spectroModes):
            char_mode = spectroModes[0]
        # if no spectro mode, default char mode is first observing mode
        else:
            char_mode = allModes[0]
        
        # begin Survey, and loop until mission is finished
        log_begin = 'OB%s: survey beginning.'%(TK.OBnumber)
        self.logger.info(log_begin)
        self.vprint(log_begin)
        t0 = time.time()
        sInd = None
        ObsNum = 0
        
        # Create first target list 
        first_block_time = 0.5*TK.missionPortion*TK.missionLife
        time_spent = 0*u.d
        
        ##################################################
        
        # Next target advances time so change that in the accounting for stuff
        # below until it matches with the simulation timing things. Need exoplanetObsTime
        # calculation to match what we have. Or maybe we can just use that and then
        # reset everything after the optimization.
        ##############################################
        
        
        # while time_spent < first_block_time:
        #     DRM, sInd, det_intTime, waitTime = self.next_target(sInd, det_mode)
        #     self.first_targets_list.append((DRM, sInd, det_intTime, waitTime))
        #     print(det_intTime)
        #     if waitTime is None:
        #         waitTime = 0*u.d
        #     print(waitTime)
            
        #     time_spent += det_intTime + waitTime
        #     print(time_spent)
       
        planning_observations = True
        # Save the initial times so that they can be used to reset the times later
        initial_exoplanet_obs_time = TK.exoplanetObsTime.copy()
        initial_time_norm = TK.currentTimeNorm.copy()
        initial_time_abs = TK.currentTimeAbs.copy()
        print('Creating initial schedule')
        while planning_observations:
            # Make intiial mission plan by choosing all the targets but not simulating
            # detections. Then resetting the times aftwerwards
            DRM, sInd, det_intTime, waitTime = self.next_target(sInd, det_mode)
            self.mission_schedule.append((DRM, sInd, det_intTime, waitTime))
            self.initial_schedule.append((DRM, sInd, det_intTime, waitTime))
            
            extraTime = det_intTime*(det_mode['timeMultiplier'] - 1.)
            TK.allocate_time(det_intTime + extraTime + Obs.settlingTime + det_mode['syst']['ohTime'], True)#allocates time
            
            if TK.exoplanetObsTime > 0.5*TK.missionPortion*TK.missionLife:
                # When we've completed the first half of the planning stop
                print('Completed first schedule')
                TK.exoplanetObsTime = initial_exoplanet_obs_time
                TK.currentTimeNorm = initial_time_norm
                TK.currentTimeAbs = initial_time_abs
                planning_observations = False
        
        first_half = True
        while not TK.mission_is_over(OS, Obs, det_mode):
            # acquire the NEXT TARGET star index and create DRM
            old_sInd = sInd #used to save sInd if returned sInd is None
            
            # Check for when to schedule the second half of the mission
            if ObsNum == len(self.initial_schedule):
                print('Creating second schedule')
                first_half = False
                planning_observations = True
                
                # Save the initial times so that they can be used to reset the times later
                initial_exoplanet_obs_time = TK.exoplanetObsTime.copy()
                initial_time_norm = TK.currentTimeNorm.copy()
                initial_time_abs = TK.currentTimeAbs.copy()
                while planning_observations:
                    # Make intiial mission plan by choosing all the targets but not simulating
                    # detections. Then 
                    DRM, sInd, det_intTime, waitTime = self.next_target(sInd, det_mode)
                    self.mission_schedule.append((DRM, sInd, det_intTime, waitTime))
                    self.second_schedule.append((DRM, sInd, det_intTime, waitTime))

                    
                    if sInd is not None:
                        extraTime = det_intTime*(det_mode['timeMultiplier'] - 1.)
                        TK.allocate_time(det_intTime + extraTime + Obs.settlingTime + det_mode['syst']['ohTime'], True)#allocates time
                    else:
                        # When there are no observable targets left finish the schedule
                        print('Completed second schedule')
                        TK.exoplanetObsTime = initial_exoplanet_obs_time
                        TK.currentTimeNorm = initial_time_norm
                        TK.currentTimeAbs = initial_time_abs
                        planning_observations = False
                        
                    if TK.exoplanetObsTime > TK.missionPortion*TK.missionLife:
                        # If we're out of integration time finish the schedule
                        print('Completed second schedule')
                        TK.exoplanetObsTime = initial_exoplanet_obs_time
                        TK.currentTimeNorm = initial_time_norm
                        TK.currentTimeAbs = initial_time_abs
                        planning_observations = False
                    
                    
            DRM, sInd, det_intTime, waitTime = self.mission_schedule[ObsNum]
            
            if sInd is not None:
                ObsNum += 1 #we're making an observation so increment observation number
                
                if OS.haveOcculter == True:
                    # advance to start of observation (add slew time for selected target)
                    
                    success = TK.advanceToAbsTime(TK.currentTimeAbs.copy() + waitTime)
                    
                # beginning of observation, start to populate DRM
                if ObsNum == len(self.initial_schedule):
                    # This only optimizes before the first observation or after the last planned one from
                    DRM['schedule_opt'] = True
                else:
                    DRM['schedule_opt'] = False
                DRM['star_ind'] = sInd
                if sInd in self.Completeness.dc_dict:
                    DRM['revisit'] = True
                else:
                    DRM['revisit'] = False
                DRM['star_name'] = TL.Name[sInd]
                DRM['arrival_time'] = TK.currentTimeNorm.to('day').copy()
                DRM['OB_nb'] = TK.OBnumber
                DRM['ObsNum'] = ObsNum
                pInds = np.where(SU.plan2star == sInd)[0]
                DRM['plan_inds'] = pInds.astype(int)
                log_obs = ('  Observation #%s, star ind %s (of %s) with %s planet(s), ' \
                        + 'mission time at Obs start: %s, exoplanetObsTime: %s')%(ObsNum, sInd, TL.nStars, len(pInds), 
                        TK.currentTimeNorm.to('day').copy().round(2), TK.exoplanetObsTime.to('day').copy().round(2))
                self.logger.info(log_obs)
                self.vprint(log_obs)
                
                # PERFORM DETECTION and populate revisit list attribute
                detected, det_fZ, det_systemParams, det_SNR, FA = \
                        self.observation_detection(sInd, det_intTime.copy(), det_mode)
                # update the occulter wet mass
                if OS.haveOcculter == True:
                    DRM = self.update_occulter_mass(DRM, sInd, det_intTime.copy(), 'det')
                # populate the DRM with detection results
                DRM['det_time'] = det_intTime.to('day')
                DRM['det_status'] = detected
                DRM['det_SNR'] = det_SNR
                DRM['det_fZ'] = det_fZ.to('1/arcsec2')
                DRM['det_params'] = det_systemParams
                
                # Eliminate potential planets from the dynamic completeness 
                # calculations
                if first_half:
                    self.Completeness.dc_dict_update(TL, TK, sInd, detected)
                
                    
                # PERFORM CHARACTERIZATION and populate spectra list attribute
                if char_mode['SNR'] not in [0, np.inf]:
                    characterized, char_fZ, char_systemParams, char_SNR, char_intTime = \
                            self.observation_characterization(sInd, char_mode)
                else:
                    char_intTime = None
                    lenChar = len(pInds) + 1 if FA else len(pInds)
                    characterized = np.zeros(lenChar, dtype=float)
                    char_SNR = np.zeros(lenChar, dtype=float)
                    char_fZ = 0./u.arcsec**2
                    char_systemParams = SU.dump_system_params(sInd)
                assert char_intTime != 0, "Integration time can't be 0."
                # update the occulter wet mass
                if OS.haveOcculter == True and char_intTime is not None:
                    DRM = self.update_occulter_mass(DRM, sInd, char_intTime, 'char')
                # populate the DRM with characterization results
                DRM['char_time'] = char_intTime.to('day') if char_intTime else 0.*u.day
                DRM['char_status'] = characterized[:-1] if FA else characterized
                DRM['char_SNR'] = char_SNR[:-1] if FA else char_SNR
                DRM['char_fZ'] = char_fZ.to('1/arcsec2')
                DRM['char_params'] = char_systemParams
                # populate the DRM with FA results
                DRM['FA_det_status'] = int(FA)
                DRM['FA_char_status'] = characterized[-1] if FA else 0
                DRM['FA_char_SNR'] = char_SNR[-1] if FA else 0.
                DRM['FA_char_fEZ'] = self.lastDetected[sInd,1][-1]/u.arcsec**2 \
                        if FA else 0./u.arcsec**2
                DRM['FA_char_dMag'] = self.lastDetected[sInd,2][-1] if FA else 0.
                DRM['FA_char_WA'] = self.lastDetected[sInd,3][-1]*u.arcsec \
                        if FA else 0.*u.arcsec
                
                # populate the DRM with observation modes
                DRM['det_mode'] = dict(det_mode)
                del DRM['det_mode']['inst'], DRM['det_mode']['syst']
                DRM['char_mode'] = dict(char_mode)
                del DRM['char_mode']['inst'], DRM['char_mode']['syst']

                DRM['exoplanetObsTime'] = TK.exoplanetObsTime.copy()
                
                # append result values to self.DRM
                self.DRM.append(DRM)

                # handle case of inf OBs and missionPortion < 1
                if np.isinf(TK.OBduration) and (TK.missionPortion < 1.):
                    self.arbitrary_time_advancement(TK.currentTimeNorm.to('day').copy() - DRM['arrival_time'])
                
            else:#sInd == None
                sInd = old_sInd#Retain the last observed star
                if(TK.currentTimeNorm.copy() >= TK.OBendTimes[TK.OBnumber]): # currentTime is at end of OB
                    #Conditional Advance To Start of Next OB
                    if not TK.mission_is_over(OS, Obs,det_mode):#as long as the mission is not over
                        TK.advancetToStartOfNextOB()#Advance To Start of Next OB
                elif(waitTime is not None):
                    #CASE 1: Advance specific wait time
                    success = TK.advanceToAbsTime(TK.currentTimeAbs.copy() + waitTime, self.defaultAddExoplanetObsTime)
                    self.vprint('waitTime is not None')
                else:
                    startTimes = TK.currentTimeAbs.copy() + np.zeros(TL.nStars)*u.d # Start Times of Observations
                    observableTimes = Obs.calculate_observableTimes(TL,np.arange(TL.nStars),startTimes,self.koMaps,self.koTimes,self.mode)[0]
                    #CASE 2 If There are no observable targets for the rest of the mission
                    if((observableTimes[(TK.missionFinishAbs.copy().value*u.d > observableTimes.value*u.d)*(observableTimes.value*u.d >= TK.currentTimeAbs.copy().value*u.d)].shape[0]) == 0):#Are there any stars coming out of keepout before end of mission
                        self.vprint('No Observable Targets for Remainder of mission at currentTimeNorm= ' + str(TK.currentTimeNorm.copy()))
                        #Manually advancing time to mission end
                        TK.currentTimeNorm = TK.missionLife
                        TK.currentTimeAbs = TK.missionFinishAbs
                    else:#CASE 3    nominal wait time if at least 1 target is still in list and observable
                        #TODO: ADD ADVANCE TO WHEN FZMIN OCURS
                        inds1 = np.arange(TL.nStars)[observableTimes.value*u.d > TK.currentTimeAbs.copy().value*u.d]
                        inds2 = np.intersect1d(self.intTimeFilterInds, inds1) #apply intTime filter
                        inds3 = self.revisitFilter(inds2, TK.currentTimeNorm.copy() + self.dt_max.to(u.d)) #apply revisit Filter #NOTE this means stars you added to the revisit list 
                        self.vprint("Filtering %d stars from advanceToAbsTime"%(TL.nStars - len(inds3)))
                        oTnowToEnd = observableTimes[inds3]
                        if not oTnowToEnd.value.shape[0] == 0: #there is at least one observableTime between now and the end of the mission
                            tAbs = np.min(oTnowToEnd)#advance to that observable time
                        else:
                            tAbs = TK.missionStart + TK.missionLife#advance to end of mission
                        tmpcurrentTimeNorm = TK.currentTimeNorm.copy()
                        success = TK.advanceToAbsTime(tAbs, self.defaultAddExoplanetObsTime)#Advance Time to this time OR start of next OB following this time
                        self.vprint('No Observable Targets a currentTimeNorm= %.2f Advanced To currentTimeNorm= %.2f'%(tmpcurrentTimeNorm.to('day').value, TK.currentTimeNorm.to('day').value))
        else:#TK.mission_is_over()
            dtsim = (time.time() - t0)*u.s
            log_end = "Mission complete: no more time available.\n" \
                    + "Simulation duration: %s.\n"%dtsim.astype('int') \
                    + "Results stored in SurveySimulation.DRM (Design Reference Mission)."
            self.logger.info(log_end)
            self.vprint(log_end)
    
    def inttimesfeps(self,eps,Cb,Csp):
        """
        Compute the optimal subset of targets for a given epsilon value
        where epsilon is the maximum completeness gradient.

        Everything is in units of days
        """

        tstars = (-Cb*eps*np.sqrt(np.log(10.)) + np.sqrt((Cb*eps)**2.*np.log(10.) + 
                   5.*Cb*Csp**2.*eps))/(2.0*Csp**2.*eps*np.log(10.)) # calculating Tau to achieve dC/dT #double check

        compstars = self.Completeness.comp_per_intTime(tstars*u.day, self.TargetList, self.TimeKeeping,
                np.arange(self.TargetList.nStars), self.ZodiacalLight.fZ0, 
                self.ZodiacalLight.fEZ0, self.WAint, self.detmode, C_b=Cb/u.d, C_sp=Csp/u.d)

        
        solver = pywraplp.Solver('SolveIntegerProblem',pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        xs = [ solver.IntVar(0.0,1.0, 'x'+str(j)) for j in np.arange(len(compstars)) ]
        constraint = solver.Constraint(-solver.infinity(), self.maxTime.to(u.d).value)

        for j,x in enumerate(xs):
            constraint.SetCoefficient(x, tstars[j] + self.ohTimeTot.to(u.day).value)

        objective = solver.Objective()
        for j,x in enumerate(xs):
            objective.SetCoefficient(x, compstars[j])
        objective.SetMaximization()
        #solver.EnableOutput() # this line enables output of the CBC MIXED INTEGER PROGRAM (Was hard to find don't delete)
        solver.SetTimeLimit(5*60*1000)#time limit for solver in milliseconds


        cpres = solver.Solve()
        #self.vprint(solver.result_status())


        x = np.array([x.solution_value() for x in xs])
        #self.vprint('Solver is FEASIBLE: ' + str(solver.FEASIBLE))
        #self.vprint('Solver is OPTIMAL: ' + str(solver.OPTIMAL))
        #self.vprint('Solver is BASIC: ' + str(solver.BASIC))

        return compstars,tstars,x

    
    def objfun(self,t,sInds,fZ):
        """
        Objective Function for SLSQP minimization. Purpose is to maximize summed completeness

        Args:
            t (ndarray):
                Integration times in days. NB: NOT an astropy quantity.
            sInds (ndarray):
                Target star indices (of same size as t)
            fZ (astropy Quantity):
                Surface brightness of local zodiacal light in units of 1/arcsec2
                Same size as t

        """
        good = t*u.d >= 0.1*u.s # inds that were not downselected by initial MIP

        comp = self.Completeness.comp_per_intTime(t[good]*u.d, self.TargetList, self.TimeKeeping, sInds[good], fZ[good], 
                self.ZodiacalLight.fEZ0, self.WAint[sInds][good], self.detmode)
        #self.vprint(-comp.sum()) # for verifying SLSQP output
        return -comp.sum()


    def objfun_deriv(self,t,sInds,fZ):
        """
        Jacobian of objective Function for SLSQP minimization. 

        Args:
            t (astropy Quantity):
                Integration times in days. NB: NOT an astropy quantity.
            sInds (ndarray):
                Target star indices (of same size as t)
            fZ (astropy Quantity):
                Surface brightness of local zodiacal light in units of 1/arcsec2
                Same size as t

        """
        good = t*u.d >= 0.1*u.s # inds that were not downselected by initial MIP

        tmp = self.Completeness.dcomp_dt(t[good]*u.d, self.TargetList, sInds[good], fZ[good], 
                self.ZodiacalLight.fEZ0, self.WAint[sInds][good], self.detmode).to("1/d").value

        jac = np.zeros(len(t))
        jac[good] = tmp
        return -jac



    def calc_targ_intTime(self, sInds, startTimes, mode):
        """
        Given a subset of targets, calculate their integration times given the
        start of observation time.

        This implementation updates the optimized times based on current conditions and 
        mission time left.

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
            astropy Quantity array:
                Integration times for detection. Same dimension as sInds
        """
 
        if self.staticOptTimes:
            intTimes = self.t0[sInds]
        else:
            # assumed values for detection
            if self.Izod == 'fZ0': # Use fZ0 to calculate integration times
                fZ = np.array([self.ZodiacalLight.fZ0.value]*len(sInds))*self.ZodiacalLight.fZ0.unit
            elif self.Izod == 'fZmin': # Use fZmin to calculate integration times
                fZ = self.valfZmin[sInds]
            elif self.Izod == 'fZmax': # Use fZmax to calculate integration times
                fZ = self.valfZmax[sInds]
            elif self.Izod == 'current': # Use current fZ to calculate integration times
                fZ = self.ZodiacalLight.fZ(self.Observatory, self.TargetList, sInds, startTimes, mode)

            #### instead of actual time left, try bounding by maxTime - detection time used
            #need to update time used in choose_next_target
            
            timeLeft = (self.TimeKeeping.missionLife - self.TimeKeeping.currentTimeNorm.copy())*self.TimeKeeping.missionPortion
            bounds = [(0,timeLeft.to(u.d).value) for i in np.arange(len(sInds))]

            initguess = self.t0[sInds].to(u.d).value
            ires = minimize(self.objfun, initguess, jac=self.objfun_deriv, args=(sInds,fZ), constraints=self.constraints,
                    method='SLSQP', bounds=bounds, options={'disp':True,'maxiter':self.maxiter,'ftol':self.ftol})
            
            #update default times for these targets
            self.t0[sInds] = ires['x']*u.d

            intTimes = ires['x']*u.d
            
        intTimes[intTimes < 0.1*u.s] = 0.0*u.d
            
        return intTimes

    def choose_next_target(self, old_sInd, sInds, slewTimes, intTimes):
        """
        
        Given a subset of targets (pre-filtered by method next_target or some 
        other means), select the best next one. 

        Args:
            old_sInd (integer):
                Index of the previous target star
            sInds (integer array):
                Indices of available targets
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            intTimes (astropy Quantity array):
                Integration times for detection in units of day
        
        Returns:
            tuple:
            sInd (integer):
                Index of next target star
            waitTime (astropy Quantity):
                the amount of time to wait (this method returns None)
        
        """
        #Do Checking to Ensure There are Targetswith Positive Nonzero Integration Time
        tmpsInds = sInds
        sInds = sInds[np.where(intTimes.value > 1e-10)[0]]#filter out any intTimes that are essentially 0
        intTimes = intTimes[intTimes.value > 1e-10]
        
        
        
        # calcualte completeness values for current intTimes
        if self.Izod == 'fZ0': # Use fZ0 to calculate integration times
            fZ = np.array([self.ZodiacalLight.fZ0.value]*len(sInds))*self.ZodiacalLight.fZ0.unit
        elif self.Izod == 'fZmin': # Use fZmin to calculate integration times
            fZ = self.valfZmin[sInds]
        elif self.Izod == 'fZmax': # Use fZmax to calculate integration times
            fZ = self.valfZmax[sInds]
        elif self.Izod == 'current': # Use current fZ to calculate integration times
            fZ = self.ZodiacalLight.fZ(self.Observatory, self.TargetList, sInds,  
                self.TimeKeeping.currentTimeAbs.copy() + slewTimes[sInds], self.detmode)
        
        # Dynamic completeness
        # comps = self.Completeness.comp_per_intTime(intTimes, self.TargetList, sInds, fZ, 
        #         self.ZodiacalLight.fEZ0, self.WAint[sInds], self.detmode)
        comps = self.Completeness.comp_per_intTime(intTimes, self.TargetList, self.TimeKeeping, sInds, fZ, 
                self.ZodiacalLight.fEZ0, self.WAint[sInds], self.detmode)
        
        
        #### Selection Metric Type
        valfZmax = self.valfZmax[sInds]
        valfZmin = self.valfZmin[sInds]
        if self.selectionMetric == 'maxC': #A choose target with maximum completeness
            sInd = np.random.choice(sInds[comps == max(comps)])
        elif self.selectionMetric == 'Izod-Izodmin': #B choose target closest to its fZmin
            selectInd = np.argmin(fZ - valfZmin)
            sInd = sInds[selectInd]
        elif self.selectionMetric == 'Izod-Izodmax': #C choose target furthest from fZmax
            selectInd = np.argmin(fZ - valfZmax)#this is most negative when fZ is smallest 
            sInd = sInds[selectInd]
        elif self.selectionMetric == '(Izod-Izodmin)/(Izodmax-Izodmin)': #D choose target closest to fZmin with largest fZmin-fZmax variation
            selectInd = np.argmin((fZ - valfZmin)/(valfZmin - valfZmax))#this is most negative when fZ is smallest 
            sInd = sInds[selectInd]
        elif self.selectionMetric == '(Izod-Izodmin)/(Izodmax-Izodmin)/CIzod': #E = D + current completeness at intTime optimized at 
            selectInd = np.argmin((fZ - valfZmin)/(valfZmin - valfZmax)*(1./comps))
            sInd = sInds[selectInd]
        #F is simply E but where comp is calculated sing fZmin
        # elif self.selectionMetric == '(Izod-Izodmin)/(Izodmax-Izodmin)/CIzodmin': #F = D + current completeness at Izodmin and intTime
        #     selectInd = np.argmin((fZ - valfZmin)/(valfZmin - valfZmax)*(1./comps))
        #     sInd = sInds[selectInd]
        elif self.selectionMetric == 'TauIzod/CIzod': #G maximum C/T
            selectInd = np.argmin(intTimes/comps)
            sInd = sInds[selectInd]
        elif self.selectionMetric == 'random': #I random selection of available
            sInd = np.random.choice(sInds)
        elif self.selectionMetric == 'priorityObs': # Advances time to 
            # Apply same filters as in next_target (the issue here is that we might want to make a target observation that
            #   is currently in keepout so we need to "add back in those targets")
            sInds = np.arange(self.TargetList.nStars)
            sInds = sInds[np.where(self.t0.value > 1e-10)[0]]
            sInds = np.intersect1d(self.intTimeFilterInds, sInds)
            sInds = self.revisitFilter(sInds, self.TimeKeeping.currentTimeNorm.copy())

            TK = self.TimeKeeping

            #### Pick which absTime
            #We will readjust self.absTimefZmin later
            tmpabsTimefZmin = list() # we have to do this because "self.absTimefZmin does not support item assignment" BS
            for i in np.arange(len(self.fZQuads)):
                fZarr = np.asarray([self.fZQuads[i][j][1].value for j in np.arange(len(self.fZQuads[i]))]) # create array of fZ for the Target Star
                fZarrInds = np.where( np.abs(fZarr - self.valfZmin[i].value) < 0.000001*np.min(fZarr))[0]

                dt = self.t0[i] # amount to subtract from points just about to enter keepout
                #Extract fZ Type
                assert not len(fZarrInds) == 0, 'This should always be greater than 0'
                if len(fZarrInds) == 2:
                    fZminType0 = self.fZQuads[i][fZarrInds[0]][0]
                    fZminType1 = self.fZQuads[i][fZarrInds[1]][0]
                    if fZminType0 == 2 and fZminType1 == 2: #Ah! we have a local minimum fZ!
                        #which one occurs next?
                        tmpabsTimefZmin.append(self.whichTimeComesNext([self.fZQuads[i][fZarrInds[0]][3],self.fZQuads[i][fZarrInds[1]][3]]))
                    elif (fZminType0 == 0 and fZminType1 == 1) or (fZminType0 == 1 and fZminType1 == 0): # we have an entering and exiting or exiting and entering
                        if fZminType0 == 0: # and fZminType1 == 1
                            tmpabsTimefZmin.append(self.whichTimeComesNext([self.fZQuads[i][fZarrInds[0]][3]-dt,self.fZQuads[i][fZarrInds[1]][3]]))
                        else: # fZminType0 == 1 and fZminType1 == 0
                            tmpabsTimefZmin.append(self.whichTimeComesNext([self.fZQuads[i][fZarrInds[0]][3],self.fZQuads[i][fZarrInds[1]][3]-dt]))
                    elif fZminType1 == 2 or fZminType0 == 2: # At least one is local minimum
                        if fZminType0 == 2:
                            tmpabsTimefZmin.append(self.whichTimeComesNext([self.fZQuads[i][fZarrInds[0]][3]-dt,self.fZQuads[i][fZarrInds[1]][3]]))
                        else: # fZminType1 == 2
                            tmpabsTimefZmin.append(self.whichTimeComesNext([self.fZQuads[i][fZarrInds[0]][3],self.fZQuads[i][fZarrInds[1]][3]-dt]))
                    else: # Throw error
                        raise Exception('A fZminType was not assigned or handled correctly 1')
                elif len(fZarrInds) == 1:
                    fZminType0 = self.fZQuads[i][fZarrInds[0]][0]
                    if fZminType0 == 2: # only 1 local fZmin
                        tmpabsTimefZmin.append(self.fZQuads[i][fZarrInds[0]][3])
                    elif fZminType0 == 0: # entering
                        tmpabsTimefZmin.append(self.fZQuads[i][fZarrInds[0]][3] - dt)
                    elif fZminType0 == 1: # exiting
                        tmpabsTimefZmin.append(self.fZQuads[i][fZarrInds[0]][3])
                    else: # Throw error
                        raise Exception('A fZminType was not assigned or handled correctly 2')
                elif len(fZarrInds) == 3:
                    #Not entirely sure why 3 is occuring. Looks like entering, exiting, and local minima exist.... strange
                    tmpdt = list()
                    for k in np.arange(3):
                        if self.fZQuads[i][fZarrInds[k]][0] == 0:
                            tmpdt.append(dt)
                        else:
                            tmpdt.append(0.*u.d)
                    tmpabsTimefZmin.append(self.whichTimeComesNext([self.fZQuads[i][fZarrInds[0]][3]-tmpdt[0],self.fZQuads[i][fZarrInds[1]][3]-tmpdt[1],self.fZQuads[i][fZarrInds[2]][3]-tmpdt[2]]))
                elif len(fZarrInds) >= 4:
                    raise Exception('Unexpected Error: Number of fZarrInds was 4')
                    #might check to see if local minimum and koentering/exiting happened
                elif len(fZarrInds) == 0:
                    raise Exception('Unexpected Error: Number of fZarrInds was 0')

            #### reassign
            tmpabsTimefZmin = Time(np.asarray([tttt.value for tttt in tmpabsTimefZmin]),format='mjd',scale='tai')
            self.absTimefZmin = tmpabsTimefZmin

            #### Time relative to now where fZmin occurs
            timeWherefZminOccursRelativeToNow = self.absTimefZmin.value - TK.currentTimeAbs.copy().value #of all targets
            indsLessThan0 = np.where((timeWherefZminOccursRelativeToNow < 0))[0] # find all inds that are less than 0
            cnt = 0.
            while len(indsLessThan0) > 0: #iterate until we know the next time in the future where fZmin occurs for all targets
                cnt += 1.
                timeWherefZminOccursRelativeToNow[indsLessThan0] = self.absTimefZmin.copy().value[indsLessThan0]\
                    - TK.currentTimeAbs.copy().value + cnt*365.25 #take original and add 365.25 until we get the right number of years to add
                indsLessThan0 = np.where((timeWherefZminOccursRelativeToNow < 0))[0]
            timeToStartfZmins = timeWherefZminOccursRelativeToNow#contains all "next occuring" fZmins in absolute time

            timefZminAfterNow = [timeToStartfZmins[i] for i in sInds]#filter by times in future and times not filtered
            timeToAdvance = np.min(np.asarray(timefZminAfterNow))#find the minimum time

            tsInds = np.where((timeToStartfZmins == timeToAdvance))[0]#find the index of the minimum time and return that sInd
            tsInds = [i for i in tsInds if i in sInds]
            if len(tsInds) > 1:
                sInd = tsInds[0]
            else:
                sInd = tsInds[0]
            del timefZminAfterNow

            #The folllowing is useless I think
            # if len(self.revisitFilter(np.where(self.t0.value >1e-10)[0], self.TimeKeeping.currentTimeNorm.copy())) == 0:
            #     print(saltyburrito)
            #     return None, None

            #Advance To fZmin of Target
            success = self.TimeKeeping.advanceToAbsTime(Time(timeToAdvance+TK.currentTimeAbs.copy().value, format='mjd', scale='tai'), False)

            #Check if exoplanetObsTime would be exceeded
            OS = self.OpticalSystem
            Comp = self.Completeness
            TL = self.TargetList
            Obs = self.Observatory
            TK = self.TimeKeeping
            allModes = OS.observingModes
            mode = list(filter(lambda mode: mode['detectionMode'] == True, allModes))[0]
            maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife = TK.get_ObsDetectionMaxIntTime(Obs, mode)
            maxIntTime = min(maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife)#Maximum intTime allowed
            intTimes2 = self.calc_targ_intTime(sInd, TK.currentTimeAbs.copy(), mode)
            if intTimes2 > maxIntTime: # check if max allowed integration time would be exceeded
                self.vprint('max allowed integration time would be exceeded')
                sInd = None
                waitTime = 1.*u.d
        #H is simply G but where comp and intTime are calculated using fZmin
        #elif self.selectionMetric == 'TauIzodmin/CIzodmin': #H maximum C at fZmin / T at fZmin


        if not sInd == None:
            if self.t0[sInd] < 1.0*u.s: # We assume any observation with integration time of less than 1 second is not a valid integration time
                self.vprint('sInd to None is: ' + str(sInd))
                sInd = None
        

        return sInd, None

    def arbitrary_time_advancement(self,dt):
        """ Handles fully dynamically scheduled case where OBduration is infinite and
        missionPortion is less than 1.
        Input dt is the total amount of time, including all overheads and extras
        used for the previous observation.
        """
        if self.selectionMetric == 'priorityObs':
            pass
        else:
            self.TimeKeeping.allocate_time( dt*(1. - self.TimeKeeping.missionPortion)/self.TimeKeeping.missionPortion,\
                addExoplanetObsTime=False )


    def whichTimeComesNext(self, absTs):
        """ Determine which absolute time comes next from current time
        Specifically designed to determine when the next local zodiacal light event occurs form fZQuads 
        Args:
            absTs (list) - the absolute times of different events (list of absolute times)
        Return:
            absT (astropy time quantity) - the absolute time which occurs next
        """
        TK = self.TimeKeeping
        #Convert Abs Times to norm Time
        tabsTs = list()
        for i in np.arange(len(absTs)):
            tabsTs.append((absTs[i] - TK.missionStart).value) # all should be in first year
        tSinceStartOfThisYear = TK.currentTimeNorm.copy().value%365.25
        if len(tabsTs) == len(np.where(tSinceStartOfThisYear < np.asarray(tabsTs))[0]): # time strictly less than all absTs
            absT = absTs[np.argmin(tabsTs)]
        elif len(tabsTs) == len(np.where(tSinceStartOfThisYear > np.asarray(tabsTs))[0]):
            absT = absTs[np.argmin(tabsTs)]
        else: #Some are above and some are below
            tmptabsTsInds = np.where(tSinceStartOfThisYear < np.asarray(tabsTs))[0]
            absT = absTs[np.argmin(np.asarray(tabsTs)[tmptabsTsInds])] # of times greater than current time, returns smallest

        return absT
    
    
    def scheduleRevisit(self,sInd,smin,det,pInds):
        """A Helper Method for scheduling revisits after observation detection
        Args:
            sInd - sInd of the star just detected
            smin - minimum separation of the planet to star of planet just detected
            det - list of which planets around the target star were detected
            pInds - Indices of planets around target star
        
        Note:
            Updates self.starRevisit attribute only
        """
        TK = self.TimeKeeping
        TL = self.TargetList
        SU = self.SimulatedUniverse
        
        
        # if sInd not in self.Completeness.dc_dict:
        #     # If dynamic completeness for the star then generate a list of 
        #     # true values as the potential planets, where each True represents
        #     # the fact that the planet hasn't been eliminated from search
        #     # The potential planets are those defined in instantiation of this
        #     # completeness module as a_vals, e_vals, etc
        #     potential_planets = np.ones(self.Completeness.Nplanets)
        # else:
        #     # If it's been checked already then get the list containing the 
        #     # array with which planets have been eliminated already
        #     potential_planets = self.Completeness.dc_dict[sInd]
            
        # # Get the values for the propagated planets
        # a_p = self.a_vals[potential_planets]
        # e_p = self.e_vals[potential_planets]
        # M0_p = self.M0_vals[potential_planets]
        # I_p = self.I_vals[potential_planets]
        # w_p = self.w_vals[potential_planets]
        # Rp_p = self.Rp_vals[potential_planets]
        # p_p = self.p_vals[potential_planets]
        
        
        # in both cases (detection or false alarm), schedule a revisit 
        # based on minimum separation
        Ms = TL.MsTrue[sInd]
        if smin is not None:#smin is None if no planet was detected
            sp = smin
            if np.any(det):
                pInd_smin = pInds[det][np.argmin(SU.s[pInds[det]])]
                Mp = SU.Mp[pInd_smin]
            else:
                Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3./mu)
            t_rev = TK.currentTimeNorm.copy() + T/2.
        # otherwise, revisit based on average of population semi-major axis and mass
        else:
            sp = SU.s.mean()
            Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3./mu)
            t_rev = TK.currentTimeNorm.copy() + 0.75*T

        # finally, populate the revisit list (NOTE: sInd becomes a float)
        revisit = np.array([sInd, t_rev.to('day').value])
        if self.starRevisit.size == 0:#If starRevisit has nothing in it
            self.starRevisit = np.array([revisit])#initialize sterRevisit
        else:
            revInd = np.where(self.starRevisit[:,0] == sInd)[0]#indices of the first column of the starRevisit list containing sInd 
            if revInd.size == 0:
                self.starRevisit = np.vstack((self.starRevisit, revisit))
            else:
                self.starRevisit[revInd,1] = revisit[1]