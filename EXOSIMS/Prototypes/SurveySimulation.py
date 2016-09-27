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
        PlanetPopulation (PlanetPopulation module):
            PlanetPopulation class object
        PlanetPhysicalModel (PlanetPhysicalModel module):
            PlanetPhysicalModel class object
        OpticalSystem (OpticalSystem module):
            OpticalSystem class object
        ZodiacalLight (ZodiacalLight module):
            ZodiacalLight class object
        BackgroundSources (BackgroundSources module):
            BackgroundSources class object
        PostProcessing (PostProcessing module):
            PostProcessing class object
        Completeness (Completeness module):
            Completeness class object
        TargetList (TargetList module):
            TargetList class object
        SimulatedUniverse (SimulatedUniverse module):
            SimulatedUniverse class object
        Observatory (Observatory module):
            Observatory class object
        TimeKeeping (TimeKeeping module):
            TimeKeeping class object
        nt_flux (integer):
            observation time sampling
        targetVisits (integer ndarray):
            Contains the number of times a target was visited
        DRM (list):
            list containing results of survey simulation
        
    """

    _modtype = 'SurveySimulation'
    _outspec = {}

    def __init__(self,scriptfile=None,logLevel='ERROR',nt_flux=1,**specs):
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
            self.PlanetPopulation = SU.PlanetPopulation
            self.PlanetPhysicalModel = SU.PlanetPhysicalModel
            self.OpticalSystem = SU.OpticalSystem
            self.ZodiacalLight = SU.ZodiacalLight
            self.BackgroundSources = SU.BackgroundSources
            self.PostProcessing = SU.PostProcessing
            self.Completeness = SU.Completeness
            self.TargetList = SU.TargetList
        else:
            #these are the modules that must be present if passing instantiated objects
            neededObjMods = ['PlanetPopulation',
                          'PlanetPhysicalModel',
                          'OpticalSystem',
                          'ZodiacalLight',
                          'BackgroundSources',
                          'PostProcessing',
                          'Completeness',
                          'TargetList',
                          'SimulatedUniverse',
                          'Observatory',
                          'TimeKeeping']
            
            #ensure that you have the minimal set
            for modName in neededObjMods:
                if modName not in specs['modules'].keys():
                    raise ValueError("%s module is required but was not provided." % modName)
            
            for modName in specs['modules'].keys():
                assert (specs['modules'][modName]._modtype == modName), \
                "Provided instance of %s has incorrect modtype."%modName
                
                setattr(self, modName, specs['modules'][modName])
        
        # observation time sampling (must be an integer)
        self.nt_flux = int(nt_flux)
        # number of visits to target star
        self.targetVisits = np.zeros(self.TargetList.nStars,dtype=int)
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
        
        Returns:
            string (str):
                String 'Simulation results in .DRM'
        
        """
        
        OS = self.OpticalSystem
        PP = self.PostProcessing
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        Logger.info('run_sim beginning')
        # initialize lists updated later
        revisit_list = np.array([])
        extended_list = np.array([])
        # number of planet observations
        observed = np.zeros((SU.nPlans,), dtype=int)
        # set occulter separation if haveOcculter
        if OS.haveOcculter:
            Obs.currentSep = Obs.occulterSep
        
        # initialize run options
        # keep track of spectral characterizations, 0 is no characterization
        spectra = np.zeros(SU.nPlans, dtype=int)
        # target index
        sInd = None
        
        # loop until mission is finished
        while not TK.mission_is_over():
            # Acquire a new target star index:
            # - update the currentTime (including settlingTime, ohTime, slewTime)
            # - calculate local zodi and integration time for the selected target
            # - update DRM
            DRM = {}
            DRM, sInd, t_int = self.next_target(revisit_list, DRM, sInd)
            
            if sInd:
                # get target list star index of detections for extended_list 
                if TK.currentTimeNorm > TK.missionLife and extended_list.shape[0] == 0:
                    for i in xrange(len(self.DRM)):
                        if self.DRM[i]['det_status'] == 1:
                            extended_list = np.hstack((extended_list, self.DRM[i]['target_ind']))
                            extended_list = np.unique(extended_list)
                
                # beginning of observation
                obsbegin = TK.currentTimeNorm.to('day')
                Logger.info('current time is %r' % obsbegin)
                print 'Current mission time: ', obsbegin
                
                # find out if observations are possible and get relevant data
                SNR, DRM = self.observation_detection(sInd, DRM, t_int)
                
                # determine detection, missed detection, false alarm booleans
                FA, MD = PP.det_occur(SNR)
                
                # encode detection status, and populate revisit list
                revisit_list, DRM = self.det_data(revisit_list, sInd, DRM, FA, MD)
                
                # perform characterization, default char mode is first spectro/IFS mode
                spectroModes = filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes)
                if spectroModes:
                    charMode = spectroModes[0]
                # if no spectro mode, default char mode is first observing mode
                else:
                    charMode = OS.observingModes[0]
                DRM['charMode'] = charMode
                spectra, DRM = self.observation_characterization(sInd, spectra, \
                        DRM, FA, MD, charMode)
                
                # update completeness values
                obsend = TK.currentTimeNorm.to('day')
                nexttime = TK.currentTimeNorm
                TL.comp0 = self.Completeness.completeness_update(TL, sInd, obsbegin, \
                        obsend, nexttime)
                
                # append result values to self.DRM
                self.DRM.append(DRM)
                
                # with occulter, if spacecraft fuel is depleted, exit loop
                if OS.haveOcculter and Obs.scMass < Obs.dryMass:
                    print 'Total fuel mass exceeded at %r' % TK.currentTimeNorm
                    break
            
        Logger.info('run_sim finishing OK')
        print 'Survey simulation: finishing OK'
        return 'Simulation results in .DRM'

    def next_target(self, revisit_list, DRM={}, old_sInd=None):
        """Finds index of next target star
        
        This method chooses the next target star index based on which
        stars are available, their integration time, and maximum completeness.
        Also updates DRM. Returns None if no target could be found.
        
        Args:
            revisit_list (nx2 float ndarray):
                contains indices of targets to revisit and revisit times 
                of these targets in units of day
            DRM (dict):
                dictionary containing survey simulation results
            old_sInd (integer):
                index of the previous target star
                
        Returns:
            DRM (dict):
                dictionary containing survey simulation results
            sInd (integer):
                index of next target star. Default to None.
            t_int (astropy Quantity):
                selected star integration time in units of day. Default to None.
        
        """
        
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # select detection mode
        mode = OS.detectionMode
        # allocate settling time + overhead time
        TK.allocate_time(Obs.settlingTime + mode['syst']['ohTime'])
        # In case of an occulter, initialize slew time factor
        # (add transit time and reduce starshade mass)
        if OS.haveOcculter:
            ao = Obs.thrust/Obs.scMass
            slewTime_fac = (2.*Obs.occulterSep/np.abs(ao) \
                    /(Obs.defburnPortion/2. - Obs.defburnPortion**2/4.)).decompose().to('d2')
        
        # Now, start to look for available targets
        while not TK.mission_is_over():
            # 0/ initialize arrays
            slewTime = np.zeros(TL.nStars)*u.d
            fZs = np.zeros(TL.nStars)/u.arcsec**2
            t_ints = np.zeros(TL.nStars)*u.d
            tovisit = np.zeros(TL.nStars, dtype=bool)
            sInds = np.array(range(TL.nStars))
            
            # 1/ find spacecraft orbital START positions and filter out unavailable 
            # targets. If occulter, each target has its own START position.
            if OS.haveOcculter:
                # find angle between old and new stars, default to pi/2 for first target
                if old_sInd == None:
                    sd = np.array([np.pi/2.]*TL.nStars)*u.rad
                else:
                    # position vector of previous target star
                    r_old = Obs.starprop(TL, old_sInd, TK.currentTimeAbs)
                    u_old = r_old/np.sqrt(np.sum(r_old**2))
                    # position vector of new target stars
                    r_new = Obs.starprop(TL, sInds, TK.currentTimeAbs)
                    u_new = r_new/np.sqrt(np.sum(r_new**2))
                    # angle between old and new stars
                    sd = np.arccos(np.dot(u_old, u_new.T))[0]
                    sd[np.where(np.isnan(sd))] = 0.
                # calculate slew time
                slewTime = np.sqrt(slewTime_fac*np.sin(sd/2.))
            
            startTime = TK.currentTimeAbs + slewTime
            r_sc = Obs.orbit(startTime)
            kogoodStart = Obs.keepout(TL, sInds, startTime, r_sc, OS.telescopeKeepout)
            sInds = sInds[np.where(kogoodStart)[0]]
            
            # 2/ calculate integration times for the preselected targets, 
            # and filter out t_ints > intCutoff
            if np.any(sInds):
                fZ = ZL.fZ(TL, sInds, mode['lam'], r_sc[sInds])
                fEZ = ZL.fEZ0
                t_ints[sInds] = OS.calc_maxintTime(TL, sInds, fZ, fEZ, mode)
                # include integration time multiplier
                t_tot = t_ints*mode['timeMultiplier']
                # total time must be positive and shorter than treshold
                sInds = np.where((0 < t_tot) & (t_tot < OS.intCutoff))[0]
            
            # 3/ find spacecraft orbital END positions (for each candidate target), 
            # and filter out unavailable targets
            if np.any(sInds):
                endTime = TK.currentTimeAbs + t_ints[sInds]
                r_sc = Obs.orbit(endTime)
                kogoodEnd = Obs.keepout(TL, sInds, endTime, r_sc, OS.telescopeKeepout)
                sInds = sInds[np.where(kogoodEnd)[0]]
            
            # 4/ filter out all previously (more) visited targets, unless in 
            # revisit list, with time within some dt (+- 1 week) of start
            if np.any(sInds):
                tovisit[sInds] = (self.targetVisits[sInds] == self.targetVisits[sInds].min())
                if revisit_list.size != 0:
                    dt_max = 1.*u.week
                    dt_rev = np.abs(revisit_list[:,1]*u.day - TK.currentTimeNorm)
                    ind_rev = [int(x) for x in revisit_list[dt_rev < dt_max,0] if x in sInds]
                    tovisit[ind_rev] = True
                sInds = np.where(tovisit)[0]
            
            # 5/ choose best target from remaining
            if np.any(sInds):
                # prototype version choose sInd among targets with highest completeness
                mask = np.where(TL.comp0[sInds] == max(TL.comp0[sInds]))[0]
                sInd = np.random.choice(sInds[mask])
                # update visited list for current star
                self.targetVisits[sInd] += 1
                # store relevant values
                t_int = t_ints[sInd]
                break
            
            # if no observable target, allocate time and try again
            # TODO: improve how to deal with no available target
            else:
                TK.allocate_time(TK.dtAlloc)
            
        else:
            Logger.info('Mission complete: no more time available')
            return DRM, None, None
        
        # Once a target is chosen, populate DRM
        DRM['target_ind'] = sInd
        DRM['det_int_time'] = t_int.to('day').value
        
        if OS.haveOcculter:
            # update current time by adding slew time 
            TK.allocate_time(slewTime[sInd])
            if TK.mission_is_over():
                Logger.info('Mission complete: no more time available')
                return DRM, None, None
            # store values related to slew time
            DRM['slew_time'] = slewTime[sInd].to('day').value
            DRM['slew_angle'] = sd[sInd].to('rad').value
            slew_mass_used = slewTime[sInd]*Obs.defburnPortion*Obs.flowRate
            DRM['slew_dV'] = (slewTime[sInd]*ao*Obs.defburnPortion).to('m/s').value
            DRM['slew_mass_used'] = slew_mass_used.to('kg').value
            # find disturbance forces on occulter
            dF_lateral, dF_axial = Obs.distForces(TL, sInd, TK.currentTimeAbs)
            DRM['dF_lateral'] = dF_lateral.to('N').value
            DRM['dF_axial'] = dF_axial.to('N').value
            # store values related to detection time (station-keeping)
            intMdot, det_mass_used, deltaV = Obs.mass_dec(dF_lateral, t_int)
            DRM['det_dV'] = deltaV.to('m/s').value
            DRM['det_mass_used'] = det_mass_used.to('kg').value
            # store initial spacecraft mass, then update it
            DRM['scMass'] = Obs.scMass.to('kg').value
            Obs.scMass -= (slew_mass_used + det_mass_used)
        
        # store arrival time (after eventual slew time)
        DRM['arrival_time'] = TK.currentTimeNorm.to('day').value
        
        return DRM, sInd, t_int

    def calc_signal_noise(self, sInd, dt, mode):
        """Calculate the signal and noise fluxes for a given time interval

        Args:
            sInd (integer):
                Target star index
            dt (astropy Quantity):
                Integration time interval
            mode (dict):
                Selected observing mode (from OpticalSystem)
        
        Returns:
            Signal (float)
                Counts of signal
            Noise (float)
                Counts of background noise variance
        """
        
        PPMod = self.PlanetPhysicalModel
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # allocate first half of dt
        TK.allocate_time(dt/2.)
        # find spacecraft position and ZodiacalLight
        r_sc = Obs.orbit(TK.currentTimeAbs)
        fZ = ZL.fZ(TL, sInd, mode['lam'], r_sc)
        # propagate the system to match up with current time
        SU.prop_system(sInd, TK.currentTimeNorm)
        # calculate fEZ, dMag and WA, for the planets pInds
        pInds = np.where(SU.plan2star == sInd)[0] #planet indices
        fEZ = SU.fEZ[pInds]
        Phi = PPMod.calc_Phi(np.arcsin(SU.s[pInds]/SU.d[pInds]))
        dMag = deltaMag(SU.p[pInds], SU.Rp[pInds], SU.d[pInds], Phi)
        WA = np.arctan(SU.s[pInds]/TL.dist[sInd])
        # find electron counts for planet, background, and speckle residual 
        C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, fEZ, dMag, WA, mode)
        # calculate signal and noise levels (based on Nemati14 formula)
        Signal = (C_p*dt).decompose().value
        Noise = np.sqrt((C_b*dt + (C_sp*dt)**2).decompose().value)
        # allocate second half of dt
        TK.allocate_time(dt/2.)
        
        return Signal, Noise

    def observation_detection(self, sInd, DRM, t_int):
        """Finds if planet observations are possible and relevant information
        
        Args:
            sInd (integer):
                target star index
            DRM (dict):
                dictionary containing survey simulation results
            t_int (astropy Quantity):
                integration time for the selected target in units of day
        
        Returns:
            SNR (float ndarray):
                signal-to-noise ratio of the planets around the selected target
            DRM (dict):
                dictionary containing survey simulation results
        
        """
        
        PPMod = self.PlanetPhysicalModel
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # select detection mode
        mode = OS.detectionMode
        # planet indices around target
        pInds = np.where(SU.plan2star == sInd)[0]
        # are there planets around this target? if yes, integrate the planet flux 
        if len(pInds) > 0:
            dt = t_int/self.nt_flux
            # initialise Signal and Noise arrays
            Signal = np.zeros((self.nt_flux, len(pInds)))
            Noise = np.zeros((self.nt_flux, len(pInds)))
            for i in range(self.nt_flux):
                s,n = self.calc_signal_noise(sInd, dt, mode)
                Signal[i,:] = s
                Noise[i,:] = n
            # calculate SNR
            SNR = Signal.sum(0) / Noise.sum(0)
            # allocate extra time for timeMultiplier
            t_extra = t_int*(mode['timeMultiplier'] - 1)
            TK.allocate_time(t_extra)
        
        # if no planet, SNR array is empty
        # just observe for the duration t_tot (including time multiplier)
        else:
            SNR = np.array([])
            t_tot = t_int*(mode['timeMultiplier'])
            TK.allocate_time(t_tot)
        
        # populate DRM
        DRM['plan_inds'] = pInds.astype(int).tolist()
        DRM['SNR'] = SNR.tolist()
        
        return SNR, DRM

    def det_data(self, revisit_list, sInd, DRM, FA, MD):
        """Determines detection status
        
        This method encodes detection status values in the DRM 
        dictionary.
        
        Args:
            revisit_list (nx2 float ndarray):
                contains indices of targets to revisit and revisit times 
                of these targets in units of day
            sInd (int):
                index of star in target list
            DRM (dict):
                dictionary containing survey simulation results
            FA (boolean):
                False alarm (false positive) boolean.
            MD (boolean ndarray):
                Missed detection (false negative) boolean with the size of 
                number of planets around the target.
        
        Returns:
            revisit_list (nx2 float ndarray):
                contains indices of targets to revisit and revisit times 
                of these targets in units of day
            DRM (dict):
                dictionary containing survey simulation results
        
        """
        
        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        TL = self.TargetList
        SU = self.SimulatedUniverse
        TK = self.TimeKeeping
        
        # default DRM detection status to null detection
        DRM['det_status'] = 0
        smin = None
        
        # detection status = 1, missed detection status = -1
        pInds = np.where(SU.plan2star == sInd)[0]
        if len(pInds) > 0:
            DRM['det_pInds'] = pInds
            DRM['det_status'] = (1 - MD.astype(int)*2).tolist()
            DRM['det_WA'] = np.arctan(SU.s[pInds]/TL.dist[sInd]).to('mas').value
            Phi = PPMod.calc_Phi(np.arcsin(SU.s[pInds]/SU.d[pInds]))
            DRM['det_dMag'] = deltaMag(SU.p[pInds], SU.Rp[pInds], SU.d[pInds], Phi)
            # calculate minimum separation of detected planets
            if np.any(pInds[~MD]):
                smin = np.min(SU.s[pInds][~MD])
                Logger.info('Detected planet(s) %r of target %r' % (pInds, sInd))
                print 'Detected planet(s)', pInds, 'of target', sInd
        
        # is there a false positive?
        if FA:
            DRM['det_FA'] = 1
            # generate apparent separation sFA
            sFA = 1.*u.AU
            ds = np.random.rand()*(PPop.arange.max() - PPop.arange.min())
            sFA += ds*np.sqrt(TL.L[sInd]) if PPop.scaleOrbits else ds
            # check if sFA is smaller than the separation of detected planets
            smin = np.minimum(smin,sFA) if smin else sFA 
            Logger.info('False Alarm at target %r' % sInd)
            print 'False Alarm at target', sInd
        
        # if there are planets detected, or a false alarm, schedule a revisit 
        # based on planet with minimum separation
        if smin:
            sp = smin
            Mp = SU.Mp[pInds[np.argmin(SU.s[pInds])]]
            mu = const.G*(Mp + TL.MsTrue[sInd]*const.M_sun)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm + T/2.
        # else, revisit based on average of population semi-major axis and mass
        else:
            sp = SU.s.mean()
            Mp = SU.Mp.mean()
            mu = const.G*(Mp + TL.MsTrue[sInd]*const.M_sun)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm + 0.75*T
        
        # populate revisit list (sInd becomes a float)
        revisit = np.array([sInd, t_rev.to('day').value])
        if revisit_list.size == 0:
            revisit_list = np.array([revisit])
        else:
            revisit_list = np.vstack((revisit_list, revisit))
        
        return revisit_list, DRM

    def observation_characterization(self, sInd, spectra, DRM, FA, MD, mode):
        """Finds if characterizations are possible and relevant information
        
        Args:
            sInd (integer):
                target star index
            spectra (float ndarray):
                contains values indicating if planet spectra have been captured
            DRM (dict):
                dictionary containing survey simulation results
            FA (boolean):
                False alarm (false positive) boolean.
            MD (boolean ndarray):
                Missed detection (false negative) boolean with the size of 
                number of planets around the target.
            mode (dict):
                Selected characterization mode.
        
        Returns:
            spectra (float ndarray):
                contains values indicating if planet spectra have been captured
            DRM (dict):
                dictionary containing survey simulation results
                
        """
        
        PPMod = self.PlanetPhysicalModel
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # initialize
        t_char = 0.*u.d
        tochar = False
        # planet indices around target
        pInds = np.where(SU.plan2star == sInd)[0]
        # are there planets detected around this target, and not characterized yet?
        if np.any(pInds):
            tochar = (spectra[pInds[~MD]] == 0)
        # if yes, perform first characterization
        if np.any(tochar):
            # find spacecraft position and ZodiacalLight
            r_sc = Obs.orbit(TK.currentTimeAbs)
            fZ = ZL.fZ(TL, sInd, mode['lam'], r_sc)
            # propagate the whole system to match up with current time
            SU.prop_system(sInd, TK.currentTimeNorm)
            # calculate fEZ, dMag and WA, for the planets pInds to characterize
            pInds = pInds[~MD][tochar]
            fEZ = SU.fEZ[pInds]
            Phi = PPMod.calc_Phi(np.arcsin(SU.s[pInds]/SU.d[pInds]))
            dMag = deltaMag(SU.p[pInds], SU.Rp[pInds], SU.d[pInds], Phi)
            WA = np.arctan(SU.s[pInds]/TL.dist[sInd]).to('arcsec')
            # calculate characterization times, and filter out planets with t_char > intCutoff
            sInds = np.array([sInd]*len(pInds))
            t_chars = OS.calc_intTime(TL, sInds, fZ, fEZ, dMag, WA, mode)
            mask = np.where(t_chars < OS.intCutoff)[0]
            pInds = pInds[mask]
            if np.any(pInds):
                t_char = np.max(t_chars[mask])
                Logger.info('Characterized planet(s) %r of target %r' % (pInds, sInd))
                print 'Characterized planet(s)', pInds, 'of target', sInd
            
            # if there was a false alarm, find the characterization time with 
            # standard values: fEZ0, dMagLim, IWA
            if FA:
                t_char_FA = OS.calc_intTime(TL, sInd, fZ, ZL.fEZ0, OS.dMagLim, OS.IWA, mode)
                if t_char_FA < OS.intCutoff:
                    t_char = np.maximum(t_char, t_char_FA)
            
#           # TODO: understand this block:
#           # account for 5 bands and one coronagraph
#           t_char *= 4
            
            # check if target still observable at the end of characterization time
            if t_char > 0:
                endTime = TK.currentTimeAbs + t_char
                r_sc = Obs.orbit(endTime)
                kogoodEnd = Obs.keepout(TL, sInd, endTime, r_sc, OS.telescopeKeepout)
                if not kogoodEnd:
                    t_char = 0*u.d
                else:
                    # if planet is visible at end of characterization, spectrum is captured
                    spectra[pInds] += 1
                    # encode relevant first characterization data
                    DRM['char_1_time'] = t_char.to('day').value
                    if OS.haveOcculter:
                        # find disturbance forces on occulter
                        dF_lateral, dF_axial = Obs.distForces(TL, sInd, TK.currentTimeAbs)
                        # decrement mass for station-keeping
                        intMdot, mass_used, deltaV = Obs.mass_dec(dF_lateral, t_int)
                        mass_used_char = t_char*intMdot
                        deltaV_char = dF_lateral/Obs.scMass*t_char
                        Obs.scMass -= mass_used_char
                        # encode information in DRM
                        DRM['char_1_dV'] = deltaV_char.to('m/s').value
                        DRM['char_1_mass_used'] = mass_used_char.to('kg').value
                        DRM['char_1_success'] = 1
                        
#                     # TODO: understand this block:
#                     else:
#                         lamEff = np.arctan(SU.s[pInds]/TL.dist[sInd]) / OS.IWA.to('rad')
#                         lamEff *= OS.Spectro['lam']/OS.pupilDiam*np.sqrt(OS.pupilArea/OS.shapeFac)
#                         charPossible =  (lamEff >= 800.*u.nm)
#                         # encode results
#                         if np.any(charPossible):
#                             spectra[pInds[charPossible]] = 1
#                             DRM['char_1_success'] = 1
#                         else:
#                             DRM['char_1_success'] = lamEff.max().to('nm').value
        
        return spectra, DRM

