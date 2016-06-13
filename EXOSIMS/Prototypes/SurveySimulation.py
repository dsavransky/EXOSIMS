# -*- coding: utf-8 -*-
import numpy as np
import sys
import astropy.units as u
import astropy.constants as const
import logging
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.deltaMag import deltaMag

# the EXOSIMS logger
Logger = logging.getLogger(__name__)

class SurveySimulation(object):
    """Survey Simulation class template
    
    This class contains all variables and methods necessary to perform
    Survey Simulation Module calculations in exoplanet mission simulation.
    
    It inherits the following class objects which are defined in __init__:
    Simulated Universe, Observatory, TimeKeeping, PostProcessing
    
    Args:
        \*\*specs:
            user specified values
            
    Attributes:
        SimulatedUniverse (SimulatedUniverse):
            SimulatedUniverse class object
        Observatory (Observatory):
            Observatory class object
        TimeKeeping (TimeKeeping):
            TimeKeeping class object
        PostProcessing (PostProcessing):
            PostProcessing class object
        TargetList (TargetList):
            TargetList class object
        PlanetPhysicalModel (PlanetPhysicalModel):
            PlanetPhysicalModel class object
        OpticalSystem (OpticalSystem):
            OpticalSystem class object
        PlanetPopulation (PlanetPopulation):
            PlanetPopulation class object
        ZodiacalLight (ZodiacalLight):
            ZodiacalLight class object
        Completeness (Completeness):
            Completeness class object
        DRM (list):
            list containing results of survey simulation
        
    """

    _modtype = 'SurveySimulation'
    _outspec = {}

    def __init__(self,scriptfile=None,logLevel='ERROR',**specs):
        """Initializes Survey Simulation with default values
        
        Input: 
            scriptfile:
                JSON script file.  If not set, assumes that 
                dictionary has been passed through specs 
            specs: 
                Dictionary containing user specification values and 
                a dictionary of modules.  The module dictionary can contain
                string values, in which case the objects will be instantiated,
                or object references.
        """
        
        # toggle the logging level: INFO, DEBUG, WARNING, ERROR, CRITICAL
        if logLevel.upper() == 'INFO':
            logging.basicConfig(level=logging.INFO)
        elif logLevel.upper() == 'DEBUG':
            logging.basicConfig(level=logging.DEBUG)
        elif logLevel.upper() == 'WARNING':
            logging.basicConfig(level=logging.WARNING)
        elif logLevel.upper() == 'ERROR':
            logging.basicConfig(level=logging.ERROR)
        elif logLevel.upper() == 'CRITICAL':
            logging.basicConfig(level=logging.CRITICAL)
        
        # if a script file is provided read it in
        if scriptfile is not None:
            import json
            import os.path
            assert os.path.isfile(scriptfile), "%s is not a file."%scriptfile
            
            try:
                script = open(scriptfile).read()
                specs = json.loads(script)
            except ValueError:
                sys.stderr.write("Error.  Script file `%s' is not valid JSON." % scriptfile)
                # must re-raise, or the error will be masked 
                raise
            except:
                sys.stderr.write("Unexpected error while reading specs file: " + sys.exc_info()[0])
                raise
            
            # modules array must be present
            if 'modules' not in specs.keys():
                raise ValueError("No modules field found in script.")
        
        #if any of the modules is a string, assume that they are all strings and we need to initalize
        if isinstance(specs['modules'].itervalues().next(),basestring):
            
            # import desired module names (prototype or specific)
            self.SimulatedUniverse = get_module(specs['modules'] \
                    ['SimulatedUniverse'],'SimulatedUniverse')(**specs)
            self.Observatory = get_module(specs['modules'] \
                    ['Observatory'],'Observatory')(**specs)
            self.TimeKeeping = get_module(specs['modules'] \
                    ['TimeKeeping'],'TimeKeeping')(**specs)
            
            # bring inherited class objects to top level of Survey Simulation
            SU = self.SimulatedUniverse
            self.OpticalSystem = SU.OpticalSystem
            self.PlanetPopulation = SU.PlanetPopulation
            self.ZodiacalLight = SU.ZodiacalLight
            self.BackgroundSources = SU.BackgroundSources
            self.Completeness = SU.Completeness
            self.PlanetPhysicalModel = SU.PlanetPhysicalModel
            self.PostProcessing = SU.PostProcessing
            self.TargetList = SU.TargetList
        else:
            #these are the modules that must be present if passing instantiated objects
            neededObjMods = ['SimulatedUniverse',
                          'Observatory',
                          'TimeKeeping',
                          'PostProcessing',
                          'OpticalSystem',
                          'PlanetPopulation',
                          'ZodiacalLight',
                          'Completeness',
                          'PlanetPhysicalModel',
                          'TargetList']
            
            #ensure that you have the minimal set
            for modName in neededObjMods:
                if modName not in specs['modules'].keys():
                    raise ValueError("%s module is required but was not provided." % modName)
            
            for modName in specs['modules'].keys():
                assert (specs['modules'][modName]._modtype == modName), \
                "Provided instance of %s has incorrect modtype."%modName
                
                setattr(self, modName, specs['modules'][modName])
        
        # list of simulation results, each item is a dictionary
        self.DRM = []

    def __str__(self):
        """String representation of the Survey Simulation object
        
        When the command 'print' is used on the Survey Simulation object, this 
        method will return the values contained in the object"""
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Survey Simulation class object attributes'

    def run_sim(self):
        """Performs the survey simulation 
        
        This method has access to the following:
            Obs:
                Observatory class object
            TK:
                TimeKeeping class object
            SU:
                SimulatedUniverse class object
            TL:
                TargetList class object
            PPro:
                PostProcessing class object
            OS:
                OpticalSystem class object
            PPop:
                PlanetPopulation class object
        
        Returns:
            string (str):
                String 'Simulation results in .DRM'
        
        """
        
        Obs = self.Observatory
        TK = self.TimeKeeping
        SU = self.SimulatedUniverse
        TL = self.TargetList
        PPro = self.PostProcessing
        OS = self.OpticalSystem
        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        
        Logger.info('run_sim beginning')
        # initialize values updated later
        # number of visits to target star
        visited = np.zeros(TL.nStars,dtype=int)
        # target revisit list
        revisit_list = np.array([])
        extended_list = np.array([])
        # current time (normalized to zero at mission start) of planet positions
        planPosTime = np.array([TK.currentTimeNorm.to('day').value]*SU.nPlans)*u.day
        # number of planet observations
        observed = np.zeros((SU.nPlans,), dtype=int)
        # set occulter separation if haveOcculter
        if OS.haveOcculter:
            Obs.currentSep = Obs.occulterSep
        
        # initialize run options
        # keep track of spectral characterizations, 0 is no characterization
        spectra = np.zeros((SU.nPlans,1), dtype=int)
        # get index of first target star
        sInd,_ = self.next_target()
        
        # loop until mission is finished
        while not TK.mission_is_over():
            Logger.info('current time is %r' % TK.currentTimeNorm)
            print 'Current mission time: ', TK.currentTimeNorm
            obsbegin = TK.currentTimeNorm.to('day')

            # dictionary containing results
            DRM = {}
            DRM['target_ind'] = sInd                                 # target star index
            DRM['arrival_time'] = TK.currentTimeNorm.to('day').value # arrival time
            if OS.haveOcculter:
                DRM['scMass'] = Obs.scMass.to('kg').value            # spacecraft mass

            # get target list star index of detections for extended_list 
            if TK.currentTimeNorm > TK.missionLife and extended_list.shape[0] == 0:
                for i in xrange(len(self.DRM)):
                    if self.DRM[i]['det'] == 1:
                        extended_list = np.hstack((extended_list, self.DRM[i]['target_ind']))
                        extended_list = np.unique(extended_list)
            
            # filter planet indices
            pInds = np.where(SU.plan2star == sInd)[0]           # belonging to target star
            nPlans = len(pInds)
            WA = SU.get_current_WA(pInds)
            pInds = pInds[(WA>OS.IWA) * (WA<OS.OWA)]            # inside [IWA-OWA]
            Phi = PPMod.calc_Phi(np.arcsin(SU.s[pInds]/SU.d[pInds]))
            dMag = deltaMag(SU.p[pInds],SU.Rp[pInds],SU.d[pInds],Phi)
            pInds = pInds[dMag < OS.dMagLim]                    # bright enough
            Logger.info('Observing %r/%r planets around star #%r/%r.'%(len(pInds),\
                    nPlans,sInd+1,TL.nStars))
            
            # update visited list for current star
            visited[sInd] += 1
            # find out if observations are possible and get relevant data
            observationPossible, t_int, DRM = self.observation_detection(pInds, sInd, DRM, planPosTime)
            t_int += Obs.settlingTime
            # store detection integration time
            DRM['det_int_time'] = t_int.to('day').value
            if not TK.allocate_time(t_int):
                # time too large: skip it and allocate default value dtAlloc
                observationPossible = False
                TK.allocate_time(TK.dtAlloc)
            if pInds.shape[0] != 0:
                Logger.info('Imaging possible: %s', observationPossible)
            
            # determine detection, missed detection, false alarm booleans
            FA, DET, MD, NULL = PPro.det_occur(observationPossible)
            
            # encode detection status
            s, DRM, observed = self.det_data(DRM, FA, DET, MD, sInd, pInds, \
                    observationPossible, observed)
            
            # perform characterization if SNchar defined
            if PPro.SNchar > 0:
                DRM, FA, spectra = self.observation_characterization(observationPossible, \
                        pInds, sInd, spectra, DRM, FA, t_int)
            if pInds.shape[0] != 0:
                Logger.info('Characterization possible: %s', observationPossible)
            
            # schedule a revisit
            if pInds.shape[0] != 0 and (DET or FA):
                # if there are planets, revisit based on planet with minimum separation
                sp = np.min(s)
                Mp = SU.Mp[pInds[np.argmin(s)]]
                mu = const.G*(Mp + TL.MsTrue[sInd]*const.M_sun)
                T = 2.*np.pi*np.sqrt(sp**3/mu)
                t_rev = TK.currentTimeNorm + T/2.
            else:
                # revisit based on average of population semi-major axis and mass
                sp = SU.s.mean()
                Mp = SU.Mp.mean()
                mu = const.G*(Mp + TL.MsTrue[sInd]*const.M_sun)
                T = 2.*np.pi*np.sqrt(sp**3/mu)
                t_rev = TK.currentTimeNorm + 0.75*T
            
            # populate revisit list (sInd is converted to float)
            revisit = np.array([sInd, t_rev.to('day').value])
            if revisit_list.size == 0:
                revisit_list = np.array([revisit])
            else:
                revisit_list = np.vstack((revisit_list, revisit))
            
            # update completeness values
            obsend = TK.currentTimeNorm.to('day')
            nexttime = TK.currentTimeNorm
            TL.comp0 = self.Completeness.completeness_update(sInd, TL, obsbegin, \
                    obsend, nexttime)
            
            # acquire a new target star index
            sInd, DRM = self.next_target(sInd, DRM)
            
            # append result values to self.DRM
            self.DRM.append(DRM)
            
            # with occulter if spacecraft fuel is depleted, exit loop
            if OS.haveOcculter and Obs.scMass < Obs.dryMass:
                print 'Total fuel mass exceeded at %r' % TK.currentTimeNorm
                break
        
        Logger.info('run_sim finishing OK')
        print 'Survey simulation: finishing OK'
        return 'Simulation results in .DRM'

    def next_target(self, sInd=None, DRM={}):
        """Finds index of next target star
        
        This method chooses the next target star index at random based on which
        stars are available, except for the first target star which is the one
        with maximum completeness. Returns None if no target could be found. Also 
        updates DRM if there is an occulter.
        
        Args:
            sInd (int):
                index of current target star
            DRM (dict):
                dictionary of simulation results
                
        Returns:
            new_sInd, DRM (int, dict):
                index of next target star, dictionary of simulation results
                new_sInd is None if no target could be found
        
        """
        
        Obs = self.Observatory
        TK = self.TimeKeeping
        TL = self.TargetList
        OS = self.OpticalSystem
        
        while not TK.mission_is_over():
            # find keepout Boolean values (kogood)
            Obs.keepout(TK.currentTimeAbs, TL, OS.telescopeKeepout)
            # if observable targets, pick one, else allocate time and try again
            if np.any(Obs.kogood):
                if sInd == None:
                    new_sInd = np.argmax(TL.comp0*Obs.kogood)
                else:
                    new_sInd = np.random.choice(np.where(Obs.kogood)[0])
                break
            else:
                TK.allocate_time(TK.dtAlloc)
        else:
            Logger.info('No more targets available')
            return None, DRM
        
        if OS.haveOcculter and sInd != None:
            # add transit time and reduce starshade mass
            ao = Obs.thrust/Obs.scMass
            targetSep = Obs.occulterSep
            # find position vector of previous target star
            r_old = Obs.starprop(TK.currentTimeAbs, TL, sInd)
            # find position vector of new target star
            r_new = Obs.starprop(TK.currentTimeAbs, TL, new_sInd)
            # find unit vectors
            u_old = r_old/np.sqrt(np.sum(r_old**2))
            u_new = r_new/np.sqrt(np.sum(r_new**2))
            # find angle between old and new stars
            sd = np.arccos(np.dot(u_old, u_new))
            # find slew distance
            slew_dist = 2.*targetSep*np.sin(sd/2.)
            slew_time = np.sqrt(slew_dist/np.abs(ao)/(Obs.defburnPortion/2. - Obs.defburnPortion**2/4.))
            mass_used = slew_time*Obs.defburnPortion*Obs.flowRate
            
            TK.allocate_time(slew_time)
            
            DRM['slew_time'] = slew_time.to('day').value
            DRM['slew_dV'] = (ao*slew_time*Obs.defburnPortion).to('m/s').value
            DRM['slew_mass_used'] = mass_used.to('kg').value
            DRM['slew_angle'] = sd
        
        return new_sInd, DRM

    def observation_detection(self, pInds, sInd, DRM, planPosTime):
        """Finds if planet observations are possible and relevant information
        
        This method makes use of the following inherited class objects:
        
        Args:
            pInds (ndarray):
                1D numpy ndarray of planet indices
            sInd (int):
                target star index
            DRM (dict):
                dictionary containing simulation results
            planPosTime (Quantity):
                1D numpy ndarray containing times of planet positions (units of
                time)
        
        Returns:
            observationPossible, t_int, DRM (ndarray, Quantity, dict, Quantity, ndarray, Quantity):
                1D numpy ndarray of booleans indicating if an observation of 
                each planet is possible, integration time (units of time), 
                dictionary containing survey simulation results, apparent 
                separation (units of distance), 1D numpy ndarray of delta 
                magnitude, difference in magnitude between planet and star,
                irradiance (units of :math:`1/(m^2*nm*s)`)
        
        """
        
        Obs = self.Observatory
        TK = self.TimeKeeping
        SU = self.SimulatedUniverse
        TL = self.TargetList
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        
        # initialize with True if planets are present at the target star 
        observationPossible = np.ones(len(pInds),bool) if len(pInds) else False
        # propagate the system if a planet position time do not match up with current time
        for i,pInd in enumerate(pInds):
            if planPosTime[pInd] != TK.currentTimeNorm:
                # propagate planet positions and velocities
                try:
                    dt = TK.currentTimeNorm - planPosTime[pInd]
                    SU.r[pInd], SU.v[pInd],SU.s[pInd],SU.d[pInd] = \
                            SU.prop_system(SU.r[pInd],SU.v[pInd],SU.Mp[pInd],\
                            TL.MsTrue[sInd],dt)
                    # update planet position times
                    planPosTime[pInd] += dt
                except ValueError:
                    observationPossible[i] = False
        
        # set integration time to max integration time as a default
        t_int = TL.maxintTime[sInd]
        
        # determine true integration time and update observationPossible
        if np.any(observationPossible):
            sInds = np.array([sInd]*len(pInds))
            Phi = PPMod.calc_Phi(np.arcsin(SU.s[pInds]/SU.d[pInds]))
            dMag = deltaMag(SU.p[pInds],SU.Rp[pInds],SU.d[pInds],Phi)
            WA = SU.get_current_WA(pInds)
            fEZ = SU.fEZ[pInds]
            fZ = ZL.fZ(TL,sInds,OS.Imager['lam'],Obs.r_sc)
            t_trueint = OS.calc_intTime(TL,sInds,dMag,WA,fEZ,fZ)
            observationPossible &= (t_trueint <= OS.intCutoff)
        
        # determine if planets are observable at the end of observation
        # and update integration time
        if np.any(observationPossible):
            try:
                t_int, observationPossible = self.check_visible_end(observationPossible,\
                        t_int, t_trueint, sInd, pInds, False)
            except ValueError:
                observationPossible = False
        
        if OS.haveOcculter:
            # find disturbance forces on occulter
            dF_lateral, dF_axial = Obs.distForces(TK, TL, sInd)
            # store these values
            DRM['dF_lateral'] = dF_lateral.to('N').value
            DRM['dF_axial'] = dF_axial.to('N').value
            # decrement mass for station-keeping
            intMdot, mass_used, deltaV = Obs.mass_dec(dF_lateral, t_int)
            # store these values
            DRM['det_dV'] = deltaV.to('m/s').value
            DRM['det_mass_used'] = mass_used.to('kg').value
            Obs.scMass -= mass_used
            
            # patch negative t_int
            if np.any(t_int < 0):
                Logger.warning('correcting negative t_int to arbitrary value')
                t_int = (1.0+np.random.rand())*u.day
        
        return observationPossible, t_int, DRM

    def observation_characterization(self, observationPossible, pInds, sInd, spectra, DRM, FA, t_int):
        """Finds if characterizations are possible and relevant information
        
        Args:
            observationPossible (ndarray):
                1D numpy ndarray of booleans indicating if an observation of 
                each planet is possible
            pInds (ndarray):
                1D numpy ndarray of planet indices
            sInd (int):
                target star index
            spectra (ndarray):
                numpy ndarray of values indicating if planet spectra has been
                captured
            DRM (dict):
                dictionary containing survey simulation results
            FA (bool):
                False Alarm boolean
            t_int (Quantity):
                integration time (units of time)
        
        Returns:
            DRM, FA, spectra (dict, bool, ndarray):
                dictionary containing survey simulation results, False Alarm 
                boolean, numpy ndarray of values indicating if planet spectra 
                has been captured
                
        """
        
        Obs = self.Observatory
        SU = self.SimulatedUniverse
        TL = self.TargetList
        TK = self.TimeKeeping
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        
        # check if characterization has been done
        if pInds.shape[0] != 0:
            if np.any(observationPossible):
                if np.any(spectra[pInds[observationPossible],0] == 0):
                    # perform first characterization
                    # find characterization time
                    sInds = np.array([sInd]*len(pInds))
                    Phi = PPMod.calc_Phi(np.arcsin(SU.s[pInds]/SU.d[pInds]))
                    dMag = deltaMag(SU.p[pInds],SU.Rp[pInds],SU.d[pInds],Phi)
                    WA = SU.get_current_WA(pInds)
                    fEZ = SU.fEZ[pInds]
                    fZ = ZL.fZ(TL,sInds,OS.Spectro['lam'],Obs.r_sc)
                    t_char = OS.calc_charTime(TL,sInds,dMag,WA,fZ,fEZ)
                    # account for 5 bands and one coronagraph
                    t_char *= 4
                    # patch negative t_char
                    if np.any(t_char < 0):
                        Logger.warning('correcting negative t_char to arb. value')
                        t_char_value = (4+2*np.random.rand())*u.day
                        t_char[t_char < 0] = t_char_value
                    
                    # determine which planets will be observable at the end of observation
                    charPossible = observationPossible & (t_char <= OS.intCutoff)
                    
                    try:
                        t_char, charPossible, chargo = self.check_visible_end(charPossible, \
                                t_char, t_char, sInd, pInds, True)
                    except ValueError:
                        chargo = False
                    
                    if chargo:
                        # encode relevant first characterization data
                        if OS.haveOcculter:
                            # decrement sc mass
                            # find disturbance forces on occulter
                            dF_lateral, dF_axial = Obs.distForces(TK, TL, sInd)
                            # decrement mass for station-keeping
                            intMdot, mass_used, deltaV = Obs.mass_dec(dF_lateral, t_int)
                            mass_used_char = t_char*intMdot
                            deltaV_char = dF_lateral/Obs.scMass*t_char
                            Obs.scMass -= mass_used_char
                            # encode information in DRM
                            DRM['char_1_time'] = t_char.to('day').value
                            DRM['char_1_dV'] = deltaV_char.to('m/s').value
                            DRM['char_1_mass_used'] = mass_used_char.to('kg').value
                        else:
                            DRM['char_1_time'] = t_char.to('day').value
                        
                        # if integration time goes beyond observation duration, set quantities
                        if not TK.allocate_time(t_char.max()):
                            charPossible = False
                        
                        # if this was a false alarm, it has been noted, update FA
                        if FA:
                            FA = False
                        
                        # if planet is visible at end of characterization,
                        # spectrum is captured
                        if np.any(charPossible):
                            if OS.haveOcculter:
                                spectra[pInds[charPossible],0] = 1
                                # encode success
                                DRM['char_1_success'] = 1
                            else:
                                lamEff = np.arctan(SU.s[pInds]/(TL.dist[sInd])) / OS.IWA.to('rad')
                                lamEff *= OS.Spectro['lam']/OS.pupilDiam*np.sqrt(OS.pupilArea/OS.shapeFac)
                                charPossible = np.logical_and(charPossible, lamEff >= 800.*u.nm)
                                # encode results
                                if np.any(charPossible):
                                    spectra[pInds[charPossible],0] = 1
                                    DRM['char_1_success'] = 1
                                else:
                                    DRM['char_1_success'] = lamEff.max().to('nm').value
        
        return DRM, FA, spectra

    def check_visible_end(self, observationPossible, t_int, t_trueint, sInd, pInds, t_char_calc):
        """Determines if planets are visible at the end of the observation time
        
        Args:
            observationPossible (ndarray):
                1D numpy ndarray of booleans indicating if an observation of 
                each planet is possible
            t_int (Quantity):
                integration time (units of time)
            t_trueint (Quantity):
                1D numpy ndarray of calculated integration times for planets
                (units of time)
            sInd (int):
                target star index
            pInds (ndarray):
                1D numpy ndarray of planet indices
            t_char_calc (bool):
                boolean where True is for characterization calculations
                
        Returns:
            t_int, observationPossible, chargo (Quantity, ndarray, bool):
                true integration time for planetary system (units of day) 
                (t_char_calc = False) or maximum characterization time for 
                planetary system (t_char_calc = True), updated 1D numpy ndarray 
                of booleans indicating if an observation or characterization of 
                each planet is possible, boolean where True is to encode 
                characterization data (t_char_calc = True only)
        
        """
        
        TL = self.TargetList
        Obs = self.Observatory
        SU = self.SimulatedUniverse
        OS = self.OpticalSystem
        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        
        # set chargo to False initially
        chargo = False
        for i in xrange(len(pInds)):
            if observationPossible[i]:
                # is planet visible at the end of the observation time?
                if Obs.kogood[sInd]:
                    # propagate planet to observational period end, and find dMagend
                    dt = t_int[i] if t_char_calc else t_trueint[i] + Obs.settlingTime
                    j = pInds[[i]] # must be an array of size 1
                    rend, vend, send, dend = SU.prop_system(SU.r[j],SU.v[j],\
                            SU.Mp[j],TL.MsTrue[sInd], dt)
                    Phi = PPMod.calc_Phi(np.arcsin(send/dend))
                    dMagend = deltaMag(SU.p[j],SU.Rp[j],dend,Phi)[0]
                    WAend = np.arctan(send/TL.dist[sInd])[0]
                    if (dMagend <= OS.dMagLim) * (WAend >= OS.IWA):
                        obsRes = 1
                        # planet visible at the end of observation
                        if not t_char_calc:
                            # update integration time
                            t_int = max(t_trueint[i],t_int) if t_int != TL.maxintTime[sInd] else t_trueint[i]
                    else:
                        obsRes = 0
                else:
                    obsRes = -1
                    if t_char_calc:
                        chargo = True
        
        if t_char_calc:
            if np.any(observationPossible):
                t_int = np.max(t_int[observationPossible])
            return t_int, observationPossible, chargo
        else:
            return t_int, observationPossible

    def det_data(self, DRM, FA, DET, MD, sInd, pInds, observationPossible, observed):
        """Determines detection status
        
        This method encodes detection status (FA, DET, MD) values in the DRM 
        dictionary.
        
        This method accesses the following inherited class objects:
            OS:
                OpticalSystem class object
            TL:
                TargetList class object
            self.PlanetPopulation:
                PlanetPopulation class object
        
        Args:
            DRM (dict):
                dictionary containing simulation results
            FA (bool):
                Boolean signifying False Alarm
            DET (bool):
                Boolean signifying DETection
            MD (bool):
                Boolean signifying Missed Detection
            sInd (int):
                index of star in target list
            pInds (ndarray):
                idices of planets belonging to target star
            observationPossible (ndarray):
                1D numpy ndarray of booleans indicating if an observation of 
                each planet is possible
            observed (ndarray):
                1D numpy ndarray indicating number of observations for each
                planet
        
        Returns:
            s, DRM, observed (Quantity, dict, ndarray):
                apparent separation (units of distance), delta magnitude, 
                irradiance (units of flux per time), dictionary containing 
                simulation results, 1D numpy ndarray indicating number of 
                observations for each planet
        
        """
        
        SU = self.SimulatedUniverse
        TL = self.TargetList
        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        
        # default DRM detection status to null detection
        DRM['det_status'] = 0
        # apparent separation placeholders
        s = SU.s[pInds] if pInds.size else np.array([1.])*u.AU
        
        if FA: # false alarm
            DRM['det_status'] = -2
            ds = np.random.rand()*(PPop.arange.max() - PPop.arange.min())
            s += ds*np.sqrt(TL.L[sInd]) if PPop.scaleOrbits else ds
        elif MD: # missed detection
            DRM['det_status'] = -1
        elif DET: # detection
            observed[pInds[observationPossible]] += 1
            DRM['det_status'] = observationPossible.astype(int).tolist()
            DRM['det_WA'] = np.arctan(s/TL.dist[sInd]).min().to('mas').value
            Phi = PPMod.calc_Phi(np.arcsin(SU.s[pInds]/SU.d[pInds]))
            DRM['det_dMag'] = deltaMag(SU.p[pInds], SU.Rp[pInds], SU.d[pInds],Phi).max()
        
        return s, DRM, observed
