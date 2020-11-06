from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import EXOSIMS, os
import astropy.units as u
import astropy.constants as const
import numpy as np
import itertools
from scipy import interpolate
try:
    import cPickle as pickle
except:
    import pickle
import time
from EXOSIMS.util.deltaMag import deltaMag


class tieredScheduler(SurveySimulation):
    """tieredScheduler 
    
    This class implements a tiered scheduler that independantly schedules the observatory
    while the starshade slews to its next target.
    
    Args:
        coeffs (iterable 7x1):
            Cost function coefficients: slew distance, completeness, intTime,
            deep-dive least visited ramp, deep-dive unvisited ramp, unvisited ramp, 
            and least-visited ramp
        occHIPs (iterable nx1):
            List of star HIP numbers to initialize occulter target list.
        topstars (integer):
            Number of HIP numbers to recieve preferential treatment.
        revisit_wait (float):
            Wait time threshold for star revisits. The value given is the fraction of a 
            characterized planet's period that must be waited before scheduling a revisit.
        revisit_weight (float):
            Weight used to increase preference for coronograph revisits.
        GAPortion (float):
            Portion of mission time used for general astrophysics.
        int_inflection (boolean):
            Calculate integration time using the pre-calculated integration time curves.
            Default is False.
        GA_simult_det_fraction (float):
            Fraction of detection time to be considered as GA time.
        promote_hz_stars (boolean):
            Flag that allows promotion of targets with planets in the habitable zone 
            to the occulter target list.
        phase1_end (int):
            Number of days to wait before the end of phase 1, when phase 1 ends,
            target promotion begins.
        n_det_remove (int):
            Minimum number of visits with no detections required to filter off star
        n_det_min (int):
            Minimum number of detections required for promotion
        occ_max_visits (int):
            Number of maximum visits to a star allowed by the occulter.
        max_successful_chars (int):
            Maximum number of successful characterizations on a given star before 
            it is removed from the target list.
        max_successful_dets (int):
            Maximum number of successful detections on a given star before 
            it is removed from the target list.
        nmax_promo_det (int):
            Number of detection on a star required to be promoted regardless of
            detection occurance times.
        lum_exp (int):
            Exponent used in the luminosity weighting function.
        tot_det_int_cutoff (float):
            Number of total days the scheduler is allowed to spend on detections.
        \*\*specs:
            user specified values
    """

    def __init__(self, coeffs=[2,1,1,8,4,1,1], occHIPs=[], topstars=0, revisit_wait=0.5, 
                 revisit_weight=1.0, GAPortion=.25, int_inflection=False,
                 GA_simult_det_fraction=.07, promote_hz_stars=False, phase1_end=365, 
                 n_det_remove=3, n_det_min=3, occ_max_visits=3, max_successful_chars=1,
                 max_successful_dets=4, nmax_promo_det=4, lum_exp=1, tot_det_int_cutoff=None,
                 **specs):
        
        SurveySimulation.__init__(self, **specs)
        
        #verify that coefficients input is iterable 4x1
        if not(isinstance(coeffs,(list,tuple,np.ndarray))) or (len(coeffs) != 7):
            raise TypeError("coeffs must be a 7 element iterable")

        TK = self.TimeKeeping
        TL = self.TargetList
        OS = self.OpticalSystem
        SU = self.SimulatedUniverse

        #Add to outspec
        self._outspec['coeffs'] = coeffs
        self._outspec['occHIPs'] = occHIPs
        self._outspec['topstars'] = topstars
        self._outspec['revisit_wait'] = revisit_wait
        self._outspec['revisit_weight'] = revisit_weight
        self._outspec['GAPortion'] = GAPortion
        self._outspec['int_inflection'] = int_inflection
        self._outspec['GA_simult_det_fraction'] = GA_simult_det_fraction
        self._outspec['promote_hz_stars'] = promote_hz_stars
        self._outspec['phase1_end'] = phase1_end
        self._outspec['n_det_remove'] = n_det_remove
        self._outspec['n_det_min'] = n_det_min
        self._outspec['occ_max_visits'] = occ_max_visits
        self._outspec['max_successful_chars'] = max_successful_chars
        self._outspec['lum_exp'] = lum_exp

        #normalize coefficients
        coeffs = np.array(coeffs)
        coeffs = coeffs/np.linalg.norm(coeffs, ord=1)
        
        self.coeffs = coeffs
        if occHIPs != []:
            occHIPs_path = os.path.join(EXOSIMS.__path__[0],'Scripts',occHIPs)
            assert os.path.isfile(occHIPs_path), "%s is not a file."%occHIPs_path
            with open(occHIPs_path, 'r') as ofile:
                HIPsfile = ofile.read()
            self.occHIPs = HIPsfile.split(',')
            if len(self.occHIPs) <= 1:
                self.occHIPs = HIPsfile.split('\n')
        else:
            # assert occHIPs != [], "occHIPs target list is empty, occHIPs file must be specified in script file"
            self.occHIPs = occHIPs

        self.occHIPs = [hip.strip() for hip in self.occHIPs]

        self.occ_arrives = TK.currentTimeAbs.copy()                # The timestamp at which the occulter finishes slewing
        self.occ_starRevisit = np.array([])                        # Array of star revisit times
        self.occ_starVisits = np.zeros(TL.nStars, dtype=int)       # The number of times each star was visited by the occulter
        self.is_phase1 = True                                      # Flag that determines whether or not we are in phase 1
        self.phase1_end = TK.missionStart.copy() + phase1_end*u.d  # The designated end time for the first observing phase
        self.FA_status = np.zeros(TL.nStars, dtype=bool)           # False Alarm status array 
        self.GA_percentage = GAPortion                        # Percentage of mission devoted to general astrophysics
        self.GAtime = 0.*u.d                                  # Current amount of time devoted to GA
        self.GA_simult_det_fraction = GA_simult_det_fraction  # Fraction of detection time allocated to GA
        self.goal_GAtime = None                               # The desired amount of GA time based off of mission time
        self.curves = None
        self.ao = None
        self.int_inflection = int_inflection                  # Use int_inflection to calculate int times
        self.promote_hz_stars = promote_hz_stars              # Flag to promote hz stars
        self.last_chard = None                                # Keeps track of last characterized star to avoid repeats
        self.lum_exp = lum_exp                                # The exponent to use for luminosity weighting on coronograph targets 

        self.ready_to_update = False
        self.occ_slewTime = 0.*u.d
        self.occ_sd = 0.*u.rad

        self.sInd_charcounts = {}                                   # Number of characterizations by star index
        self.sInd_detcounts = np.zeros(TL.nStars, dtype=int)        # Number of detections by star index
        self.sInd_dettimes = {}
        self.n_det_remove = n_det_remove                        # Minimum number of visits with no detections required to filter off star
        self.n_det_min = n_det_min                              # Minimum number of detections required for promotion
        self.occ_max_visits = occ_max_visits                    # Maximum number of allowed occulter visits
        self.max_successful_chars = max_successful_chars        # Maximum allowed number of successful chars of deep dive targets before removal from target list
        self.max_successful_dets = max_successful_dets
        self.nmax_promo_det = nmax_promo_det
        if tot_det_int_cutoff is None:
            self.tot_det_int_cutoff = np.inf
        else:
            self.tot_det_int_cutoff = tot_det_int_cutoff*u.d
        self.tot_dettime = 0.0*u.d

        self.topstars = topstars   # Allow preferential treatment of top n stars in occ_sInds target list
        self.coeff_data_a3 = []
        self.coeff_data_a4 = []
        self.coeff_time = []

        # self.revisit_wait = revisit_wait * u.d
        EEID = 1*u.AU*np.sqrt(TL.L)
        mu = const.G*(TL.MsTrue)
        T = (2.*np.pi*np.sqrt(EEID**3/mu)).to('d')
        self.revisit_wait = revisit_wait * T

        self.revisit_weight = revisit_weight
        self.no_dets = np.ones(self.TargetList.nStars, dtype=bool)

        self.promoted_stars = []     # list of stars promoted from the coronograph list to the starshade list
        self.ignore_stars = []       # list of stars that have been removed from the occ_sInd list
        self.t_char_earths = np.array([]) # corresponding integration times for earths

        # Precalculating intTimeFilter
        allModes = OS.observingModes
        char_mode = list(filter(lambda mode: 'spec' in mode['inst']['name'], allModes))[0]
        sInds = np.arange(TL.nStars) #Initialize some sInds array
        #ORIGINALself.occ_valfZmin, self.occ_absTimefZmin = self.ZodiacalLight.calcfZmin(sInds, self.Observatory, TL, self.TimeKeeping, char_mode, self.cachefname) # find fZmin to use in intTimeFilter
        koMap = self.koMaps[char_mode['syst']['name']]
        self.fZQuads = self.ZodiacalLight.calcfZmin(sInds, self.Observatory, TL, self.TimeKeeping, char_mode, self.cachefname, koMap, self.koTimes) # find fZmin to use in intTimeFilter
        self.occ_valfZmin, self.occ_absTimefZmin = self.ZodiacalLight.extractfZmin_fZQuads(self.fZQuads)
        fEZ = self.ZodiacalLight.fEZ0 # grabbing fEZ0
        dMag = self.dMagint[sInds] # grabbing dMag
        WA = self.WAint[sInds] # grabbing WA
        self.occ_intTimesIntTimeFilter = self.OpticalSystem.calc_intTime(TL, sInds, self.occ_valfZmin, fEZ, dMag, WA, self.mode)*char_mode['timeMultiplier'] # intTimes to filter by
        self.occ_intTimeFilterInds = np.where((self.occ_intTimesIntTimeFilter > 0)*(self.occ_intTimesIntTimeFilter <= self.OpticalSystem.intCutoff) > 0)[0] # These indices are acceptable for use simulating

        # Promote all stars assuming they have known earths
        occ_sInds_with_earths = []
        if TL.earths_only:

            Obs = self.Observatory
            ZL = self.ZodiacalLight
            char_mode = list(filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes))[0]

            # check for earths around the available stars
            for sInd in np.arange(TL.nStars):
                pInds = np.where(SU.plan2star == sInd)[0]
                pinds_earthlike = self.is_earthlike(pInds, sInd)
                if np.any(pinds_earthlike):
                    self.known_earths = np.union1d(self.known_earths, pInds[pinds_earthlike]).astype(int)
                    occ_sInds_with_earths.append(sInd)
            self.promoted_stars = np.union1d(self.promoted_stars, occ_sInds_with_earths).astype(int)

            # calculate example integration times
            sInds = SU.plan2star[self.known_earths]
            fZ = ZL.fZ(Obs, TL, sInds, TK.currentTimeAbs.copy(), char_mode)
            fEZ = SU.fEZ[self.known_earths].to('1/arcsec2')
            WAp = SU.WA[self.known_earths]
            dMag = SU.dMag[self.known_earths]
            self.t_char_earths = OS.calc_intTime(TL, sInds, fZ, fEZ, dMag, WAp, char_mode)


    def run_sim(self):
        """Performs the survey simulation 
        
        Returns:
            mission_end (string):
                Message printed at the end of a survey simulation.
        
        """
        
        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        Comp = self.Completeness
        
        # TODO: start using this self.currentSep
        # set occulter separation if haveOcculter
        self.currentSep = Obs.occulterSep
        
        # Choose observing modes selected for detection (default marked with a flag),
        det_mode = list(filter(lambda mode: mode['detectionMode'] == True, OS.observingModes))[0]
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = list(filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes))
        if np.any(spectroModes):
            char_mode = spectroModes[0]
        # if no spectro mode, default char mode is first observing mode
        else:
            char_mode = OS.observingModes[0]
        
        # Begin Survey, and loop until mission is finished
        self.logger.info('OB{}: survey beginning.'.format(TK.OBnumber+1))
        self.vprint('OB{}: survey beginning.'.format(TK.OBnumber+1))
        t0 = time.time()
        sInd = None
        occ_sInd = None
        cnt = 0

        while not TK.mission_is_over(OS, Obs, det_mode):
             
            # Acquire the NEXT TARGET star index and create DRM
            prev_occ_sInd = occ_sInd
            old_sInd = sInd #used to save sInd if returned sInd is None
            waitTime = None
            DRM, sInd, occ_sInd, t_det, sd, occ_sInds = self.next_target(sInd, occ_sInd, det_mode, char_mode)

            true_t_det = t_det*det_mode['timeMultiplier'] + Obs.settlingTime + det_mode['syst']['ohTime']
            if sInd != occ_sInd and sInd is not None:
                assert t_det != 0, "Integration time can't be 0."

            if sInd is not None and (TK.currentTimeAbs.copy() + true_t_det) >= self.occ_arrives and occ_sInd != self.last_chard:
                sInd = occ_sInd
            if sInd == occ_sInd:
                self.ready_to_update = True

            time2arrive = self.occ_arrives - TK.currentTimeAbs.copy()

            if sInd is not None:
                cnt += 1

                # clean up revisit list when one occurs to prevent repeats
                if np.any(self.starRevisit) and np.any(np.where(self.starRevisit[:,0] == float(sInd))):
                    s_revs = np.where(self.starRevisit[:,0] == float(sInd))[0]
                    t_revs = np.where(self.starRevisit[:,1]*u.day - TK.currentTimeNorm.copy() < 0*u.d)[0]
                    self.starRevisit = np.delete(self.starRevisit, np.intersect1d(s_revs,t_revs),0)

                # get the index of the selected target for the extended list
                if TK.currentTimeNorm.copy() > TK.missionLife and self.starExtended.shape[0] == 0:
                    for i in range(len(self.DRM)):
                        if np.any([x == 1 for x in self.DRM[i]['plan_detected']]):
                            self.starExtended = np.hstack((self.starExtended, self.DRM[i]['star_ind']))
                            self.starExtended = np.unique(self.starExtended)
               
                # Beginning of observation, start to populate DRM
                DRM['OB_nb'] = TK.OBnumber+1
                DRM['ObsNum'] = cnt
                DRM['star_ind'] = sInd
                pInds = np.where(SU.plan2star == sInd)[0]
                DRM['plan_inds'] = pInds.astype(int).tolist()

                if sInd == occ_sInd:
                    # wait until expected arrival time is observed
                    if time2arrive > 0*u.d:
                        TK.advanceToAbsTime(self.occ_arrives)
                        if time2arrive > 1*u.d:
                            self.GAtime = self.GAtime + time2arrive.to('day')

                TK.obsStart = TK.currentTimeNorm.copy().to('day')

                self.logger.info('  Observation #%s, target #%s/%s with %s planet(s), mission time: %s'\
                        %(cnt, sInd+1, TL.nStars, len(pInds), TK.obsStart.round(2)))
                self.vprint('  Observation #%s, target #%s/%s with %s planet(s), mission time: %s'\
                        %(cnt, sInd+1, TL.nStars, len(pInds), TK.obsStart.round(2)))

                DRM['arrival_time'] = TK.currentTimeNorm.copy().to('day')

                if sInd != occ_sInd:
                    self.starVisits[sInd] += 1
                    # PERFORM DETECTION and populate revisit list attribute.
                    # First store fEZ, dMag, WA
                    if np.any(pInds):
                        DRM['det_fEZ'] = SU.fEZ[pInds].to('1/arcsec2').value.tolist()
                        DRM['det_dMag'] = SU.dMag[pInds].tolist()
                        DRM['det_WA'] = SU.WA[pInds].to('mas').value.tolist()
                    detected, det_fZ, det_systemParams, det_SNR, FA = self.observation_detection(sInd, t_det, det_mode)

                    if np.any(detected):
                        self.sInd_detcounts[sInd] += 1
                        self.sInd_dettimes[sInd] = (self.sInd_dettimes.get(sInd) or []) + [TK.currentTimeNorm.copy().to('day')]
                        self.vprint('  Det. results are: %s'%(detected))

                    # update GAtime
                    self.GAtime = self.GAtime + t_det.to('day')*self.GA_simult_det_fraction
                    self.tot_dettime += t_det.to('day')

                    # populate the DRM with detection results
                    DRM['det_time'] = t_det.to('day')
                    DRM['det_status'] = detected
                    DRM['det_SNR'] = det_SNR
                    DRM['det_fZ'] = det_fZ.to('1/arcsec2')
                    DRM['det_params'] = det_systemParams
                    DRM['FA_det_status'] = int(FA)

                    det_comp = Comp.comp_per_intTime(t_det, TL, sInd, det_fZ,
                                                     self.ZodiacalLight.fEZ0, self.WAint[sInd], det_mode)[0]
                    DRM['det_comp'] = det_comp
                    DRM['det_mode'] = dict(det_mode)
                    del DRM['det_mode']['inst'], DRM['det_mode']['syst']
                
                elif sInd == occ_sInd:
                    self.occ_starVisits[occ_sInd] += 1
                    self.last_chard = occ_sInd
                    # PERFORM CHARACTERIZATION and populate spectra list attribute.
                    occ_pInds = np.where(SU.plan2star == occ_sInd)[0]
                    sInd = occ_sInd

                    DRM['slew_time'] = self.occ_slewTime.to('day').value
                    DRM['slew_angle'] = self.occ_sd.to('deg').value
                    slew_mass_used = self.occ_slewTime*Obs.defburnPortion*Obs.flowRate
                    DRM['slew_dV'] = (self.occ_slewTime*self.ao*Obs.defburnPortion).to('m/s').value
                    DRM['slew_mass_used'] = slew_mass_used.to('kg')
                    Obs.scMass = Obs.scMass - slew_mass_used
                    DRM['scMass'] = Obs.scMass.to('kg')

                    self.logger.info('  Starshade and telescope aligned at target star')
                    self.vprint('  Starshade and telescope aligned at target star')

                     # PERFORM CHARACTERIZATION and populate spectra list attribute
                    characterized, char_fZ, char_systemParams, char_SNR, char_intTime = \
                            self.observation_characterization(sInd, char_mode)
                    if np.any(characterized):
                        self.vprint('  Char. results are: %s'%(characterized.T))
                    else:
                        # make sure we don't accidentally double characterize
                        TK.advanceToAbsTime(TK.currentTimeAbs.copy() + .01*u.d)
                    assert char_intTime != 0, "Integration time can't be 0."
                    if np.any(occ_pInds):
                        DRM['char_fEZ'] = SU.fEZ[occ_pInds].to('1/arcsec2').value.tolist()
                        DRM['char_dMag'] = SU.dMag[occ_pInds].tolist()
                        DRM['char_WA'] = SU.WA[occ_pInds].to('mas').value.tolist()
                    DRM['char_mode'] = dict(char_mode)
                    del DRM['char_mode']['inst'], DRM['char_mode']['syst']

                    # update the occulter wet mass
                    if OS.haveOcculter and char_intTime is not None:
                        DRM = self.update_occulter_mass(DRM, sInd, char_intTime, 'char')
                        char_comp = Comp.comp_per_intTime(char_intTime, TL, occ_sInd, char_fZ,
                                                          self.ZodiacalLight.fEZ0, self.WAint[occ_sInd], char_mode)[0]
                        DRM['char_comp'] = char_comp
                    FA = False
                    # populate the DRM with characterization results
                    DRM['char_time'] = char_intTime.to('day') if char_intTime else 0.*u.day
                    #DRM['char_counts'] = self.sInd_charcounts[sInd]
                    DRM['char_status'] = characterized[:-1] if FA else characterized
                    DRM['char_SNR'] = char_SNR[:-1] if FA else char_SNR
                    DRM['char_fZ'] = char_fZ.to('1/arcsec2')
                    DRM['char_params'] = char_systemParams
                    # populate the DRM with FA results
                    DRM['FA_det_status'] = int(FA)
                    DRM['FA_char_status'] = characterized[-1] if FA else 0
                    DRM['FA_char_SNR'] = char_SNR[-1] if FA else 0.
                    DRM['FA_char_fEZ'] = self.lastDetected[sInd,1][-1]/u.arcsec**2 if FA else 0./u.arcsec**2
                    DRM['FA_char_dMag'] = self.lastDetected[sInd,2][-1] if FA else 0.
                    DRM['FA_char_WA'] = self.lastDetected[sInd,3][-1]*u.arcsec if FA else 0.*u.arcsec

                    # add star back into the revisit list
                    if np.any(characterized):
                        char = np.where(characterized)[0]
                        pInds = np.where(SU.plan2star == sInd)[0]
                        smin = np.min(SU.s[pInds[char]])
                        pInd_smin = pInds[np.argmin(SU.s[pInds[char]])]

                        Ms = TL.MsTrue[sInd]
                        sp = smin
                        Mp = SU.Mp[pInd_smin]
                        mu = const.G*(Mp + Ms)
                        T = 2.*np.pi*np.sqrt(sp**3/mu)
                        t_rev = TK.currentTimeNorm.copy() + T/2.

                self.goal_GAtime = self.GA_percentage * TK.currentTimeNorm.copy().to('day')
                goal_GAdiff = self.goal_GAtime - self.GAtime

                # allocate extra time to GA if we are falling behind
                if goal_GAdiff > 1*u.d and TK.currentTimeAbs.copy() < self.occ_arrives:
                    GA_diff = min(self.occ_arrives - TK.currentTimeAbs.copy(), goal_GAdiff)
                    self.vprint('Allocating time %s to general astrophysics'%(GA_diff))
                    self.GAtime = self.GAtime + GA_diff
                    TK.advanceToAbsTime(TK.currentTimeAbs.copy() + GA_diff)
                # allocate time if there is no target for the starshade
                elif goal_GAdiff > 1*u.d and (self.occ_arrives - TK.currentTimeAbs.copy()) < -5*u.d and not np.any(occ_sInds):
                    self.vprint('No Available Occulter Targets: Allocating time %s to general astrophysics'%(goal_GAdiff))
                    self.GAtime = self.GAtime + goal_GAdiff
                    TK.advanceToAbsTime(TK.currentTimeAbs.copy() + goal_GAdiff)

                DRM['exoplanetObsTime'] = TK.exoplanetObsTime.copy()
                # Append result values to self.DRM
                self.DRM.append(DRM)

                # Calculate observation end time
                TK.obsEnd = TK.currentTimeNorm.copy().to('day')

                # With prototype TimeKeeping, if no OB duration was specified, advance
                # to the next OB with timestep equivalent to time spent on one target
                if np.isinf(TK.OBduration) and (TK.missionPortion < 1):
                    self.arbitrary_time_advancement(TK.currentTimeNorm.to('day').copy() - DRM['arrival_time'])
                
                # With occulter, if spacecraft fuel is depleted, exit loop
                if Obs.scMass < Obs.dryMass:
                    self.vprint('Total fuel mass exceeded at %s' %TK.obsEnd.round(2))
                    break

            else:#sInd == None
                sInd = old_sInd#Retain the last observed star
                if(TK.currentTimeNorm.copy() >= TK.OBendTimes[TK.OBnumber]): # currentTime is at end of OB
                    #Conditional Advance To Start of Next OB
                    if not TK.mission_is_over(OS, Obs,det_mode):#as long as the mission is not over
                        TK.advancetToStartOfNextOB()#Advance To Start of Next OB
                elif(waitTime is not None):
                    #CASE 1: Advance specific wait time
                    success = TK.advanceToAbsTime(TK.currentTimeAbs.copy() + waitTime)
                    self.vprint('waitTime is not None')
                else:
                    startTimes = TK.currentTimeAbs.copy() + np.zeros(TL.nStars)*u.d # Start Times of Observations
                    observableTimes = Obs.calculate_observableTimes(TL,np.arange(TL.nStars),startTimes,self.koMaps,self.koTimes,self.mode)[0]
                    #CASE 2 If There are no observable targets for the rest of the mission
                    if((observableTimes[(TK.missionFinishAbs.copy().value*u.d > observableTimes.value*u.d)*(observableTimes.value*u.d >= TK.currentTimeAbs.copy().value*u.d)].shape[0]) == 0):#Are there any stars coming out of keepout before end of mission
                        self.vprint('No Observable Targets for Remainder of mission at currentTimeNorm= ' + str(TK.currentTimeNorm.copy()))
                        #Manually advancing time to mission end
                        TK.currentTimeNorm = TK.missionLife.copy()
                        TK.currentTimeAbs = TK.missionFinishAbs.copy()
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
                        success = TK.advanceToAbsTime(tAbs)#Advance Time to this time OR start of next OB following this time
                        self.vprint('No Observable Targets a currentTimeNorm= %.2f Advanced To currentTimeNorm= %.2f'%(tmpcurrentTimeNorm.to('day').value, TK.currentTimeNorm.to('day').value))
        
        else:
            dtsim = (time.time()-t0)*u.s
            mission_end = "Mission complete: no more time available.\n"\
                    + "Simulation duration: %s.\n" %dtsim.astype('int')\
                    + "Results stored in SurveySimulation.DRM (Design Reference Mission)."

            self.logger.info(mission_end)
            self.vprint(mission_end)

            return mission_end


    def promote_coro_targets(self, occ_sInds, sInds):
        """
        Determines which coronograph targets to promote to occulter targets

        Args:
            occ_sInds (numpy array):
                occulter targets
            sInds (numpy array):
                coronograph targets

        Returns:
            occ_sInds (numpy array):
                updated occulter targets
        """

        TK = self.TimeKeeping
        SU = self.SimulatedUniverse
        TL = self.TargetList
        promoted_occ_sInds = np.array([], dtype=int)

        # if phase 1 has ended
        if TK.currentTimeAbs > self.phase1_end:
            if self.is_phase1 is True:
                self.vprint('Entering detection phase 2: target list for occulter expanded')
                self.is_phase1 = False
            # If we only want to promote stars that have planets in the habitable zone
            if self.promote_hz_stars:
                # stars must have had >= n_det_min detections
                promote_stars = sInds[np.where(self.sInd_detcounts[sInds] >= self.n_det_min)[0]]
                if np.any(promote_stars):
                    for sInd in promote_stars:
                        pInds = np.where(SU.plan2star == sInd)[0]
                        sp = SU.s[pInds]
                        Ms = TL.MsTrue[sInd]
                        Mp = SU.Mp[pInds]
                        mu = const.G*(Mp + Ms)
                        T = (2.*np.pi*np.sqrt(sp**3/mu)).to('d')
                        # star must have detections that span longer than half a period and be in the habitable zone
                        # and have a smaller radius that a sub-neptune
                        pinds_earthlike = self.is_earthlike(pInds, sInd)
                        if (np.any((T/2.0 < (self.sInd_dettimes[sInd][-1] - self.sInd_dettimes[sInd][0]))) and np.any(pinds_earthlike)) \
                          or ((self.sInd_detcounts[sInd] >= self.nmax_promo_det) and np.any(pinds_earthlike)):
                            earthlikes = pInds[pinds_earthlike]
                            self.known_earths = np.union1d(self.known_earths, pInds[pinds_earthlike]).astype(int)
                            promoted_occ_sInds = np.append(promoted_occ_sInds, sInd)
                            if sInd not in self.promoted_stars:
                                self.promoted_stars.append(sInd)
                occ_sInds = np.union1d(occ_sInds, promoted_occ_sInds)
            else:
                occ_sInds = np.union1d(occ_sInds, sInds[np.where((self.starVisits[sInds] == self.nVisitsMax) & 
                                                                 (self.occ_starVisits[sInds] == 0))[0]])

        occ_sInds = np.union1d(occ_sInds, np.intersect1d(sInds, self.known_rocky))
        self.promoted_stars = list(np.union1d(self.promoted_stars, np.intersect1d(sInds, self.known_rocky)).astype(int))
        return occ_sInds.astype(int)


    def next_target(self, old_sInd, old_occ_sInd, det_mode, char_mode):
        """Finds index of next target star and calculates its integration time.
        
        This method chooses the next target star index based on which
        stars are available, their integration time, and maximum completeness.
        Returns None if no target could be found.
        
        Args:
            old_sInd (integer):
                Index of the previous target star for the telescope
            old_occ_sInd (integer):
                Index of the previous target star for the occulter
            det_mode (dict):
                Selected observing mode for detection
            char_mode (dict):
                Selected observing mode for characterization
                
        Returns:
            DRM (dicts):
                Contains the results of survey simulation
            sInd (integer):
                Index of next target star. Defaults to None.
            occ_sInd (integer):
                Index of next occulter target star. Defaults to None.
            t_det (astropy Quantity):
                Selected star integration time for detection in units of day. 
                Defaults to None.
        
        """

        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        SU = self.SimulatedUniverse
        
        # selecting appropriate koMap
        occ_koMap = self.koMaps[char_mode['syst']['name']]
        koMap = self.koMaps[det_mode['syst']['name']]
        
        # Create DRM
        DRM = {}
        
        # In case of an occulter, initialize slew time factor
        # (add transit time and reduce starshade mass)
        assert OS.haveOcculter == True
        self.ao = Obs.thrust/Obs.scMass

        # Star indices that correspond with the given HIPs numbers for the occulter
        # XXX ToDo: print out HIPs that don't show up in TL
        HIP_sInds = np.where(np.in1d(TL.Name, self.occHIPs))[0]
        if TL.earths_only:
            HIP_sInds = np.union1d(HIP_sInds, self.promoted_stars).astype(int)
        sInd = None
    
        # Now, start to look for available targets
        while not TK.mission_is_over(OS, Obs, det_mode):
            # allocate settling time + overhead time
            tmpCurrentTimeAbs = TK.currentTimeAbs.copy()
            tmpCurrentTimeNorm = TK.currentTimeNorm.copy()
            occ_tmpCurrentTimeAbs = TK.currentTimeAbs.copy()
            occ_tmpCurrentTimeNorm = TK.currentTimeNorm.copy()

            # 0 initialize arrays
            slewTimes = np.zeros(TL.nStars)*u.d
            fZs = np.zeros(TL.nStars)/u.arcsec**2
            dV = np.zeros(TL.nStars)*u.m/u.s
            intTimes = np.zeros(TL.nStars)*u.d
            occ_intTimes = np.zeros(TL.nStars)*u.d
            occ_tovisit = np.zeros(TL.nStars, dtype=bool)
            sInds = np.arange(TL.nStars)

            # 1 Find spacecraft orbital START positions and filter out unavailable 
            # targets. If occulter, each target has its own START position.
            sd = Obs.star_angularSep(TL, old_occ_sInd, sInds, tmpCurrentTimeAbs)
            obsTimes = Obs.calculate_observableTimes(TL, sInds, tmpCurrentTimeAbs, self.koMaps, self.koTimes, char_mode)
            slewTimes = Obs.calculate_slewTimes(TL, old_occ_sInd, sInds, sd, obsTimes, tmpCurrentTimeAbs)

            # 2.1 filter out totTimes > integration cutoff
            if len(sInds) > 0:
                occ_sInds = np.intersect1d(self.occ_intTimeFilterInds, sInds)
            if len(sInds) > 0:
                sInds = np.intersect1d(self.intTimeFilterInds, sInds)
            
            # Starttimes based off of slewtime
            occ_startTimes = occ_tmpCurrentTimeAbs.copy() + slewTimes
            occ_startTimesNorm = occ_tmpCurrentTimeNorm.copy() + slewTimes

            startTimes = tmpCurrentTimeAbs.copy() + np.zeros(TL.nStars)*u.d
            startTimesNorm = tmpCurrentTimeNorm.copy()

            # 2.5 Filter stars not observable at startTimes
            try:
                tmpIndsbool = list()
                for i in np.arange(len(occ_sInds)):
                    koTimeInd = np.where(np.round(occ_startTimes[occ_sInds[i]].value) - self.koTimes.value==0)[0][0] # find indice where koTime is endTime[0]
                    tmpIndsbool.append(occ_koMap[occ_sInds[i]][koTimeInd].astype(bool)) #Is star observable at time ind
                sInds_occ_ko = occ_sInds[tmpIndsbool]
                occ_sInds = sInds_occ_ko[np.where(np.in1d(sInds_occ_ko, HIP_sInds))[0]]
                del tmpIndsbool
            except:#If there are no target stars to observe 
                sInds_occ_ko = np.asarray([],dtype=int)
                occ_sInds = np.asarray([],dtype=int)

            try:
                tmpIndsbool = list()
                for i in np.arange(len(sInds)):
                    koTimeInd = np.where(np.round(startTimes[sInds[i]].value) - self.koTimes.value==0)[0][0] # find indice where koTime is endTime[0]
                    tmpIndsbool.append(koMap[sInds[i]][koTimeInd].astype(bool)) #Is star observable at time ind
                sInds = sInds[tmpIndsbool]
                del tmpIndsbool
            except:#If there are no target stars to observe 
                sInds = np.asarray([],dtype=int)

            # 2.9 Occulter target promotion step
            occ_sInds = self.promote_coro_targets(occ_sInds, sInds_occ_ko)

            # 3 Filter out all previously (more-)visited targets, unless in 
            # revisit list
            if len(sInds.tolist()) > 0:
                sInds = self.revisitFilter(sInds, TK.currentTimeNorm.copy())

            # revisit list, with time after start
            if np.any(occ_sInds):
                occ_tovisit[occ_sInds] = (self.occ_starVisits[occ_sInds] == self.occ_starVisits[occ_sInds].min())
                if self.occ_starRevisit.size != 0:
                    dt_rev = TK.currentTimeNorm.copy() - self.occ_starRevisit[:,1]*u.day
                    ind_rev = [int(x) for x in self.occ_starRevisit[dt_rev > 0, 0] if x in occ_sInds]
                    occ_tovisit[ind_rev] = True
                occ_sInds = np.where(occ_tovisit)[0]

            # 4 calculate integration times for ALL preselected targets, 
            # and filter out totTimes > integration cutoff
            maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife = TK.get_ObsDetectionMaxIntTime(Obs, det_mode)
            maxIntTime = min(maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife, OS.intCutoff)#Maximum intTime allowed

            maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife = TK.get_ObsDetectionMaxIntTime(Obs, char_mode)
            occ_maxIntTime = min(maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife, OS.intCutoff)#Maximum intTime allowed

            if len(occ_sInds) > 0:
                if self.int_inflection:
                    fEZ = ZL.fEZ0
                    WA = self.WAint
                    occ_intTimes[occ_sInds] = self.calc_int_inflection(occ_sInds, fEZ, occ_startTimes, WA[occ_sInds], char_mode, ischar=True)
                    totTimes = occ_intTimes*char_mode['timeMultiplier']
                    occ_endTimes = occ_startTimes + totTimes
                else:
                    # characterization_start = occ_startTimes
                    occ_intTimes[occ_sInds] = self.calc_targ_intTime(occ_sInds, occ_startTimes[occ_sInds], char_mode) * (1 + self.charMargin)

                    # Adjust integration time for stars with known earths around them
                    for occ_star in occ_sInds:
                        if occ_star in self.promoted_stars:
                            occ_earths = np.intersect1d(np.where(SU.plan2star == occ_star)[0], self.known_earths).astype(int)
                            if np.any(occ_earths):
                                fZ = ZL.fZ(Obs, TL, occ_star, occ_startTimes[occ_star], char_mode)
                                fEZ = SU.fEZ[occ_earths].to('1/arcsec2').value/u.arcsec**2
                                if SU.lucky_planets:
                                    phi = (1/np.pi)*np.ones(len(SU.d))
                                    dMag = deltaMag(SU.p, SU.Rp, SU.d, phi)[occ_earths]                   # delta magnitude
                                    WA = np.arctan(SU.a/TL.dist[SU.plan2star]).to('arcsec')[occ_earths]   # working angle
                                else:
                                    dMag = SU.dMag[occ_earths]
                                    WA = SU.WA[occ_earths]

                                if np.all((WA < char_mode['IWA']) | (WA > char_mode['OWA'])):
                                    occ_intTimes[occ_star] = 0.*u.d
                                else:
                                    earthlike_inttimes = OS.calc_intTime(TL, occ_star, fZ, fEZ, dMag, WA, char_mode) * (1 + self.charMargin)
                                    earthlike_inttime = earthlike_inttimes[(earthlike_inttimes < occ_maxIntTime)]
                                    if len(earthlike_inttime) > 0:
                                        occ_intTimes[occ_star] = np.max(earthlike_inttime)
                                    else:
                                        occ_intTimes[occ_star] = np.max(earthlike_inttimes)
                    occ_endTimes = occ_startTimes + (occ_intTimes * char_mode['timeMultiplier']) + Obs.settlingTime + char_mode['syst']['ohTime']

                    occ_sInds = occ_sInds[(occ_intTimes[occ_sInds] <= occ_maxIntTime)]  # Filters targets exceeding maximum intTime
                    occ_sInds = occ_sInds[(occ_intTimes[occ_sInds] > 0.0*u.d)]  # Filters with an inttime of 0
                
                if occ_maxIntTime.value <= 0:
                    occ_sInds = np.asarray([],dtype=int)

            if len(sInds.tolist()) > 0:
                intTimes[sInds] = self.calc_targ_intTime(sInds, startTimes[sInds], det_mode)
                sInds = sInds[(intTimes[sInds] <= maxIntTime)]  # Filters targets exceeding end of OB
                endTimes = startTimes + intTimes
                
                if maxIntTime.value <= 0:
                    sInds = np.asarray([],dtype=int)

            # 5.2 find spacecraft orbital END positions (for each candidate target), 
            # and filter out unavailable targets
            if len(occ_sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
                try: # endTimes may exist past koTimes so we have an exception to hand this case
                    tmpIndsbool = list()
                    for i in np.arange(len(occ_sInds)):
                        koTimeInd = np.where(np.round(occ_endTimes[occ_sInds[i]].value)-self.koTimes.value==0)[0][0] # find indice where koTime is endTime[0]
                        tmpIndsbool.append(occ_koMap[occ_sInds[i]][koTimeInd].astype(bool)) #Is star observable at time ind
                    occ_sInds = occ_sInds[tmpIndsbool]
                    del tmpIndsbool
                except:
                    occ_sInds = np.asarray([],dtype=int)

            if len(sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
                try: # endTimes may exist past koTimes so we have an exception to hand this case
                    tmpIndsbool = list()
                    for i in np.arange(len(sInds)):
                        koTimeInd = np.where(np.round(endTimes[sInds[i]].value)-self.koTimes.value==0)[0][0] # find indice where koTime is endTime[0]
                        tmpIndsbool.append(koMap[sInds[i]][koTimeInd].astype(bool)) #Is star observable at time ind
                    sInds = sInds[tmpIndsbool]
                    del tmpIndsbool
                except:
                    sInds = np.asarray([],dtype=int)

            # 5.3 Filter off current occulter target star from detection list
            if old_occ_sInd is not None:
                sInds = sInds[np.where(sInds != old_occ_sInd)]
                occ_sInds = occ_sInds[(occ_sInds != old_occ_sInd)]

            # 6.1 Filter off any stars visited by the occulter more than the max number of times
            if np.any(occ_sInds):
                occ_sInds = occ_sInds[(self.occ_starVisits[occ_sInds] < self.occ_max_visits)]

            # 6.2 Filter off coronograph stars with too many visits and no detections
            no_dets = np.logical_and((self.starVisits[sInds] > self.n_det_remove), (self.sInd_detcounts[sInds] == 0))
            sInds = sInds[np.where(np.invert(no_dets))[0]]

            max_dets = np.where(self.sInd_detcounts[sInds] < self.max_successful_dets)[0]
            sInds = sInds[max_dets]

            # 7 Filter off cornograph stars with too-long inttimes
            if self.occ_arrives > TK.currentTimeAbs:
                available_time = self.occ_arrives - TK.currentTimeAbs.copy()
                if np.any(sInds[intTimes[sInds] < available_time]):
                    sInds = sInds[intTimes[sInds] < available_time]

            # 8 remove occ targets on ignore_stars list
            occ_sInds = np.setdiff1d(occ_sInds, np.intersect1d(occ_sInds, self.ignore_stars))

            t_det = 0*u.d
            occ_sInd = old_occ_sInd
            if np.any(sInds):
                # choose sInd of next target
                sInd = self.choose_next_telescope_target(old_sInd, sInds, intTimes[sInds])
                # store relevant values
                t_det = intTimes[sInd]

            # 8 Choose best target from remaining
            # if the starshade has arrived at its destination, or it is the first observation
            if np.any(occ_sInds):
                if old_occ_sInd is None or ((TK.currentTimeAbs.copy() + t_det) >= self.occ_arrives and self.ready_to_update):
                    occ_sInd = self.choose_next_occulter_target(old_occ_sInd, occ_sInds, occ_intTimes)
                    if old_occ_sInd is None:
                        self.occ_arrives = TK.currentTimeAbs.copy()
                    else:
                        self.occ_arrives = occ_startTimes[occ_sInd]
                        self.occ_slewTime = slewTimes[occ_sInd]
                        self.occ_sd = sd[occ_sInd]
                    self.ready_to_update = False
                elif not np.any(sInds):
                    TK.advanceToAbsTime(TK.currentTimeAbs.copy() + 1*u.d)
                    continue

            if occ_sInd is not None:
                sInds = sInds[(sInds != occ_sInd)]

            if self.tot_det_int_cutoff < self.tot_dettime:
                sInds = np.array([])

            if np.any(sInds):
                # choose sInd of next target
                sInd = self.choose_next_telescope_target(old_sInd, sInds, intTimes[sInds])
                # store relevant values
                t_det = intTimes[sInd]
            else:
                sInd = None

            # if no observable target, call the TimeKeeping.wait() method
            if not np.any(sInds) and not np.any(occ_sInds):
                self.vprint('No Observable Targets at currentTimeNorm= ' + str(TK.currentTimeNorm.copy()))
                return DRM, None, None, None, None, None
            break

        else:
            self.logger.info('Mission complete: no more time available')
            self.vprint( 'Mission complete: no more time available')
            return DRM, None, None, None, None, None

        if TK.mission_is_over(OS, Obs, det_mode):
            self.logger.info('Mission complete: no more time available')
            self.vprint( 'Mission complete: no more time available')
            return DRM, None, None, None, None, None

        return DRM, sInd, occ_sInd, t_det, sd, occ_sInds

    def choose_next_occulter_target(self, old_occ_sInd, occ_sInds, intTimes):
        """Choose next target for the occulter based on truncated 
        depth first search of linear cost function.
        
        Args:
            old_occ_sInd (integer):
                Index of the previous target star
            occ_sInds (integer array):
                Indices of available targets
            intTimes (astropy Quantity array):
                Integration times for detection in units of day
                
        Returns:
            sInd (integer):
                Index of next target star
        
        """

        # Choose next Occulter target

        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping
        OS = self.OpticalSystem

        # reshape sInds, store available top9 sInds
        occ_sInds = np.array(occ_sInds, ndmin=1)
        top_HIPs = self.occHIPs[:self.topstars]
        top_sInds = np.intersect1d(np.where(np.in1d(TL.Name, top_HIPs))[0], occ_sInds)

        # current stars have to be in the adjmat
        if (old_occ_sInd is not None) and (old_occ_sInd not in occ_sInds):
            occ_sInds = np.append(occ_sInds, old_occ_sInd)

        # get completeness values
        comps = Comp.completeness_update(TL, occ_sInds, self.occ_starVisits[occ_sInds], TK.currentTimeNorm.copy())
        
        # if first target, or if only 1 available target, choose highest available completeness
        nStars = len(occ_sInds)
        if (old_occ_sInd is None) or (nStars == 1):
            occ_sInd = np.random.choice(occ_sInds[comps == max(comps)])
            return occ_sInd
        
        # define adjacency matrix
        A = np.zeros((nStars, nStars))

        # consider slew distance when there's an occulter
        r_ts = TL.starprop(occ_sInds, TK.currentTimeAbs.copy())
        u_ts = (r_ts.to('AU').value.T/np.linalg.norm(r_ts.to('AU').value, axis=1)).T
        angdists = np.arccos(np.clip(np.dot(u_ts, u_ts.T), -1, 1))
        A[np.ones((nStars),dtype=bool)] = angdists
        A = self.coeffs[0]*(A)/np.pi

        # add factor due to completeness
        A = A + self.coeffs[1]*(1 - comps)

        # add factor due to intTime
        intTimes[old_occ_sInd] = np.inf
        A = A + self.coeffs[2]*(intTimes[occ_sInds]/OS.intCutoff)

        # add factor for unvisited ramp for deep dive stars
        if np.any(top_sInds):
             # add factor for least visited deep dive stars
            f_uv = np.zeros(nStars)
            u1 = np.in1d(occ_sInds, top_sInds)
            u2 = self.occ_starVisits[occ_sInds]==min(self.occ_starVisits[top_sInds])
            unvisited = np.logical_and(u1, u2)
            f_uv[unvisited] = float(TK.currentTimeNorm.copy()/TK.missionLife.copy())**2
            A = A - self.coeffs[3]*f_uv

            self.coeff_data_a3.append([occ_sInds,f_uv])

            # add factor for unvisited deep dive stars
            no_visits = np.zeros(nStars)
            #no_visits[u1] = np.ones(len(top_sInds))
            u2 = self.occ_starVisits[occ_sInds]==0
            unvisited = np.logical_and(u1, u2)
            no_visits[unvisited] = 1.
            A = A - self.coeffs[4]*no_visits

            self.coeff_data_a4.append([occ_sInds, no_visits])
            self.coeff_time.append(TK.currentTimeNorm.copy().value)

        # add factor due to unvisited ramp
        f_uv = np.zeros(nStars)
        unvisited = self.occ_starVisits[occ_sInds]==0
        f_uv[unvisited] = float(TK.currentTimeNorm.copy()/TK.missionLife.copy())**2
        A = A - self.coeffs[5]*f_uv

        # add factor due to revisited ramp
        if self.occ_starRevisit.size != 0:
            f2_uv = 1 - (np.in1d(occ_sInds, self.occ_starRevisit[:,0]))
            A = A + self.coeffs[6]*f2_uv

        # kill diagonal
        A = A + np.diag(np.ones(nStars)*np.Inf)

        # take two traversal steps
        step1 = np.tile(A[occ_sInds==old_occ_sInd,:],(nStars,1)).flatten('F')
        step2 = A[np.array(np.ones((nStars,nStars)),dtype=bool)]
        tmp = np.nanargmin(step1+step2)
        occ_sInd = occ_sInds[int(np.floor(tmp/float(nStars)))]

        return occ_sInd

    def choose_next_telescope_target(self, old_sInd, sInds, t_dets):
        """Choose next telescope target based on star completeness and integration time.
        
        Args:
            old_sInd (integer):
                Index of the previous target star
            sInds (integer array):
                Indices of available targets
            t_dets (astropy Quantity array):
                Integration times for detection in units of day
                
        Returns:
            sInd (integer):
                Index of next target star
        
        """
        
        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping
        OS = self.OpticalSystem
        Obs = self.Observatory
        allModes = OS.observingModes

        nStars = len(sInds)

        # reshape sInds
        sInds = np.array(sInds,ndmin=1)

        # 1/ Choose next telescope target
        comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], TK.currentTimeNorm.copy())

        # add weight for star revisits
        ind_rev = []
        if self.starRevisit.size != 0:
            dt_rev = self.starRevisit[:,1]*u.day - TK.currentTimeNorm.copy()
            ind_rev = [int(x) for x in self.starRevisit[dt_rev < 0*u.d, 0] if x in sInds]

        f2_uv = np.where((self.starVisits[sInds] > 0) & (self.starVisits[sInds] < self.nVisitsMax), 
                          self.starVisits[sInds], 0) * (1 - (np.in1d(sInds, ind_rev, invert=True)))

        L = TL.L[sInds]
        l_extreme = max([np.abs(np.log10(np.min(TL.L[sInds]))), np.abs(np.log10(np.max(TL.L[sInds])))])
        if l_extreme == 0.0:
            l_weight = 1
        else:
            l_weight = 1 - np.abs(np.log10(TL.L[sInds])/l_extreme)**self.lum_exp

        t_weight = t_dets/np.max(t_dets)
        weights = ((comps + self.revisit_weight*f2_uv/float(self.nVisitsMax))/t_weight)*l_weight
        # weights = (comps + self.revisit_weight*f2_uv/float(self.nVisitsMax))*l_weight

        sInd = np.random.choice(sInds[weights == max(weights)])

        #Check if exoplanetObsTime would be exceeded
        mode = list(filter(lambda mode: mode['detectionMode'] == True, allModes))[0]
        maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife = TK.get_ObsDetectionMaxIntTime(Obs, mode)
        maxIntTime = min(maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife)#Maximum intTime allowed
        intTimes2 = self.calc_targ_intTime(np.array([sInd]), TK.currentTimeAbs.copy(), mode)
        if intTimes2 > maxIntTime: # check if max allowed integration time would be exceeded
            self.vprint('max allowed integration time would be exceeded')
            sInd = None
            waitTime = 1.*u.d
        return sInd


    def calc_int_inflection(self, t_sInds, fEZ, startTime, WA, mode, ischar=False):
        """Calculate integration time based on inflection point of Completeness as a function of int_time
        
        Args:
            t_sInds (integer array):
                Indices of the target stars
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            startTime (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode

        Returns:
            int_times (astropy quantity array):
                The suggested integration time
        
        """

        Comp = self.Completeness
        TL = self.TargetList
        ZL = self.ZodiacalLight
        Obs = self.Observatory

        num_points = 500
        intTimes = np.logspace(-5, 2, num_points)*u.d
        sInds = np.arange(TL.nStars)
        WA = self.WAint   # don't use WA input because we don't know planet positions before characterization
        curve = np.zeros([1, sInds.size, intTimes.size])

        Cpath = os.path.join(Comp.classpath, Comp.filename+'.fcomp')

        # if no preexisting curves exist, either load from file or calculate
        if self.curves is None:
            if os.path.exists(Cpath):
                self.vprint( 'Loading cached completeness file from "{}".'.format(Cpath))
                with open(Cpath, 'rb') as cfile:
                    curves = pickle.load(cfile)
                self.vprint( 'Completeness curves loaded from cache.')
            else:
                # calculate completeness curves for all sInds
                self.vprint( 'Cached completeness file not found at "{}".'.format(Cpath))
                self.vprint( 'Beginning completeness curve calculations.')
                curves = {}
                for t_i, t in enumerate(intTimes):
                    fZ = ZL.fZ(Obs, TL, sInds, startTime, mode)
                    # curves[0,:,t_i] = OS.calc_dMag_per_intTime(t, TL, sInds, fZ, fEZ, WA, mode)
                    curve[0,:,t_i] = Comp.comp_per_intTime(t, TL, sInds, fZ, fEZ, WA, mode)
                curves[mode['systName']] = curve
                with open(Cpath, 'wb') as cfile:
                    pickle.dump(curves, cfile)
                self.vprint( 'completeness curves stored in {}'.format(Cpath))

            self.curves = curves

        # if no curves for current mode
        if mode['systName'] not in self.curves.keys() or TL.nStars != self.curves[mode['systName']].shape[1]:
            for t_i, t in enumerate(intTimes):
                fZ = ZL.fZ(Obs, TL, sInds, startTime, mode)
                curve[0,:,t_i] = Comp.comp_per_intTime(t, TL, sInds, fZ, fEZ, WA, mode)

            self.curves[mode['systName']] = curve
            with open(Cpath, 'wb') as cfile:
                pickle.dump(self.curves, cfile)
            self.vprint( 'recalculated completeness curves stored in {}'.format(Cpath))

        int_times = np.zeros(len(t_sInds))*u.d
        for i, sInd in enumerate(t_sInds):
            c_v_t = self.curves[mode['systName']][0,sInd,:]
            dcdt = np.diff(c_v_t)/np.diff(intTimes)

            # find the inflection point of the completeness graph
            if ischar is False:
                target_point = max(dcdt).value + 10*np.var(dcdt).value
                idc = np.abs(dcdt - target_point/(1*u.d)).argmin()
                int_time = intTimes[idc]
                int_time = int_time*self.starVisits[sInd]

                # update star completeness
                idx = (np.abs(intTimes - int_time)).argmin()
                comp = c_v_t[idx]
                TL.comp[sInd] = comp
            else:
                idt = np.abs(intTimes - max(intTimes)).argmin()
                idx = np.abs(c_v_t - c_v_t[idt]*.9).argmin()

                # idx = np.abs(comps - max(comps)*.9).argmin()
                int_time = intTimes[idx]
                comp = c_v_t[idx]

            int_times[i] = int_time

        int_times[int_times<2.000e-5*u.d] = 0.0 *u.d
        return int_times


    def observation_characterization(self, sInd, mode):
        """Finds if characterizations are possible and relevant information
        
        Args:
            sInd (integer):
                Integer index of the star of interest
            mode (dict):
                Selected observing mode for characterization
        
        Returns:
            characterized (integer list):
                Characterization status for each planet orbiting the observed 
                target star including False Alarm if any, where 1 is full spectrum, 
                -1 partial spectrum, and 0 not characterized
            fZ (astropy Quantity):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            systemParams (dict):
                Dictionary of time-dependant planet properties averaged over the 
                duration of the integration
            SNR (float ndarray):
                Characterization signal-to-noise ratio of the observable planets. 
                Defaults to None.
            intTime (astropy Quantity):
                Selected star characterization time in units of day. Defaults to None.
        """

        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        # selecting appropriate koMap
        koMap = self.koMaps[mode['syst']['name']]

        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        pinds_earthlike = np.array([])
        det = np.ones(pInds.size, dtype=bool)
        fEZs = SU.fEZ[pInds].to('1/arcsec2').value
        dMags = SU.dMag[pInds]
        WAs = SU.WA[pInds].to('arcsec').value

        FA = (det.size == pInds.size + 1)
        if FA == True:
            pIndsDet = np.append(pInds, -1)[det]
        else:
            pIndsDet = pInds[det]

        # initialize outputs, and check if any planet to characterize
        characterized = np.zeros(det.size, dtype=int)
        fZ = 0./u.arcsec**2
        systemParams = SU.dump_system_params(sInd) # write current system params by default
        SNR = np.zeros(len(det))
        intTime = None
        if len(det) == 0: # nothing to characterize
            HIP_sInds = np.where(np.in1d(TL.Name, self.occHIPs))[0]
            if sInd in HIP_sInds:
                startTime = TK.currentTimeAbs.copy()
                startTimeNorm = TK.currentTimeNorm.copy()
                intTime = self.calc_targ_intTime(np.array([sInd]), startTime, mode)[0]
                extraTime = intTime*(mode['timeMultiplier'] - 1.)#calculates extraTime
                # add a predetermined margin to the integration times
                intTime = intTime*(1 + self.charMargin)
                # apply time multiplier
                totTime = intTime*(mode['timeMultiplier'])
                # end times
                endTimes = startTime + totTime
                endTimesNorm = startTimeNorm + totTime
                # planets to characterize
                tochar = ((totTime > 0) & (totTime <= OS.intCutoff) & 
                        (endTimesNorm <= TK.OBendTimes[TK.OBnumber]))
                success = TK.allocate_time(intTime + extraTime + mode['syst']['ohTime'] + Obs.settlingTime, True)#allocates time
                if success == False or not tochar:
                    intTime = None
            if sInd not in self.sInd_charcounts.keys():
                self.sInd_charcounts[sInd] = characterized
            return characterized, fZ, systemParams, SNR, intTime

        # look for last detected planets that have not been fully characterized
        if (FA == False): # only true planets, no FA
            tochar = (self.fullSpectra[pIndsDet] != -2)
        else: # mix of planets and a FA
            truePlans = pIndsDet[:-1]
            tochar = np.append((self.fullSpectra[truePlans] == 0), True)

        # 1/ find spacecraft orbital START position and check keepout angle
        if np.any(tochar):
            # start times
            startTime = TK.currentTimeAbs.copy()
            startTimeNorm = TK.currentTimeNorm.copy()
            # planets to characterize
            koTimeInd = np.where(np.round(startTime.value)-self.koTimes.value==0)[0][0]  # find indice where koTime is startTime[0]
            # wherever koMap is 1, the target is observable
            tochar[tochar] = koMap[sInd][koTimeInd]

        # 2/ if any planet to characterize, find the characterization times
        if np.any(tochar):
            # propagate the whole system to match up with current time
            # calculate characterization times at the detected fEZ, dMag, and WA
            pinds_earthlike = np.logical_and(np.array([(p in self.known_earths) for p in pIndsDet]), tochar)

            fZ = ZL.fZ(Obs, TL, sInd, startTime, mode)
            fEZ = fEZs[tochar]/u.arcsec**2
            WAp = self.WAint[sInd]*np.ones(len(tochar))
            dMag = self.dMagint[sInd]*np.ones(len(tochar))

            # if lucky_planets, use lucky planet params for dMag and WA
            if SU.lucky_planets:
                phi = (1/np.pi)*np.ones(len(SU.d))
                e_dMag = deltaMag(SU.p, SU.Rp, SU.d, phi)     # delta magnitude
                e_WA = np.arctan(SU.a/TL.dist[SU.plan2star]).to('arcsec')# working angle
            else:
                e_dMag = SU.dMag
                e_WA = SU.WA
            WAp[pinds_earthlike[tochar]] = e_WA[pIndsDet[pinds_earthlike]]
            dMag[pinds_earthlike[tochar]] = e_dMag[pIndsDet[pinds_earthlike]]

            intTimes = np.zeros(len(tochar))*u.day
            if self.int_inflection:
                for i,j in enumerate(WAp):
                    if tochar[i]:
                        intTimes[i] = self.calc_int_inflection([sInd], fEZ[i], startTime, j, mode, ischar=True)[0]
            else:
                intTimes[tochar] = OS.calc_intTime(TL, sInd, fZ, fEZ, dMag, WAp, mode)

            # add a predetermined margin to the integration times
            intTimes = intTimes*(1 + self.charMargin)
            # apply time multiplier
            totTimes = intTimes*(mode['timeMultiplier'])
            # end times
            endTimes = startTime + totTimes
            endTimesNorm = startTimeNorm + totTimes
            # planets to characterize
            tochar = ((totTimes > 0) & (totTimes <= OS.intCutoff) & 
                    (endTimesNorm <= TK.OBendTimes[TK.OBnumber]))

        # 3/ is target still observable at the end of any char time?
        if np.any(tochar) and Obs.checkKeepoutEnd:
            koTimeInds = np.zeros(len(endTimes.value[tochar]),dtype=int)

            # find index in koMap where each endTime is closest to koTimes
            for t,endTime in enumerate(endTimes.value[tochar]):
                if endTime > self.koTimes.value[-1]:
                    # case where endTime exceeds largest koTimes element
                    endTimeInBounds = np.where(np.floor(endTime)-self.koTimes.value==0)[0]
                    koTimeInds[t] = endTimeInBounds[0] if endTimeInBounds.size is not 0 else -1
                else:
                    koTimeInds[t] = np.where(np.round(endTime)-self.koTimes.value==0)[0][0]  # find indice where koTime is endTimes[0]
            tochar[tochar] = [koMap[sInd][koT] if koT >= 0 else 0 for koT in koTimeInds]

        # 4/ if yes, perform the characterization for the maximum char time
        if np.any(tochar):
            #Save Current Time before attempting time allocation
            currentTimeNorm = TK.currentTimeNorm.copy()
            currentTimeAbs = TK.currentTimeAbs.copy()

            if np.any(np.logical_and(pinds_earthlike, tochar)):
                intTime = np.max(intTimes[np.logical_and(pinds_earthlike, tochar)])
            else:
                intTime = np.max(intTimes[tochar])
            extraTime = intTime*(mode['timeMultiplier'] - 1.)#calculates extraTime
            success = TK.allocate_time(intTime + extraTime + mode['syst']['ohTime'] + Obs.settlingTime, True)#allocates time
            if success == False: #Time was not successfully allocated
                #Identical to when "if char_mode['SNR'] not in [0, np.inf]:" in run_sim()
                char_intTime = None
                lenChar = len(pInds) + 1 if FA else len(pInds)
                characterized = np.zeros(lenChar, dtype=float)
                char_SNR = np.zeros(lenChar, dtype=float)
                char_fZ = 0./u.arcsec**2
                char_systemParams = SU.dump_system_params(sInd)
                return characterized, char_fZ, char_systemParams, char_SNR, char_intTime

            pIndsChar = pIndsDet[tochar]
            log_char = '   - Charact. planet(s) %s (%s/%s detected)'%(pIndsChar, 
                    len(pIndsChar), len(pIndsDet))
            self.logger.info(log_char)
            self.vprint(log_char)

            # SNR CALCULATION:
            # first, calculate SNR for observable planets (without false alarm)
            planinds = pIndsChar[:-1] if pIndsChar[-1] == -1 else pIndsChar
            SNRplans = np.zeros(len(planinds))
            if len(planinds) > 0:
                # initialize arrays for SNR integration
                fZs = np.zeros(self.ntFlux)/u.arcsec**2
                systemParamss = np.empty(self.ntFlux, dtype='object')
                Ss = np.zeros((self.ntFlux, len(planinds)))
                Ns = np.zeros((self.ntFlux, len(planinds)))
                # integrate the signal (planet flux) and noise
                dt = intTime/float(self.ntFlux)
                timePlus = Obs.settlingTime.copy() + mode['syst']['ohTime'].copy()#accounts for the time since the current time
                for i in range(self.ntFlux):
                    # calculate signal and noise (electron count rates)
                    if SU.lucky_planets:
                        fZs[i] = ZL.fZ(Obs, TL, sInd, currentTimeAbs, mode)[0]
                        Ss[i,:], Ns[i,:] = self.calc_signal_noise(sInd, planinds, dt, mode, 
                                            fZ=fZs[i])
                    # allocate first half of dt
                    timePlus += dt/2.
                    # calculate current zodiacal light brightness
                    fZs[i] = ZL.fZ(Obs, TL, sInd, currentTimeAbs + timePlus, mode)[0]
                    # propagate the system to match up with current time
                    SU.propag_system(sInd, currentTimeNorm + timePlus - self.propagTimes[sInd])
                    self.propagTimes[sInd] = currentTimeNorm + timePlus
                    # save planet parameters
                    systemParamss[i] = SU.dump_system_params(sInd)
                    # calculate signal and noise (electron count rates)
                    if not SU.lucky_planets:
                        Ss[i,:], Ns[i,:] = self.calc_signal_noise(sInd, planinds, dt, mode, 
                                            fZ=fZs[i])
                    # allocate second half of dt
                    timePlus += dt/2.

                # average output parameters
                fZ = np.mean(fZs)
                systemParams = {key: sum([systemParamss[x][key]
                        for x in range(self.ntFlux)])/float(self.ntFlux)
                        for key in sorted(systemParamss[0])}
                # calculate planets SNR
                S = Ss.sum(0)
                N = Ns.sum(0)
                SNRplans[N > 0] = S[N > 0]/N[N > 0]
                # allocate extra time for timeMultiplier

            # if only a FA, just save zodiacal brightness in the middle of the integration
            else:
                totTime = intTime*(mode['timeMultiplier'])
                fZ = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs.copy() + totTime/2., mode)[0]

            # calculate the false alarm SNR (if any)
            SNRfa = []
            if pIndsChar[-1] == -1:
                fEZ = fEZs[-1]/u.arcsec**2
                dMag = dMags[-1]
                WA = WAs[-1]*u.arcsec
                C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, fEZ, dMag, WA, mode)
                S = (C_p*intTime).decompose().value
                N = np.sqrt((C_b*intTime + (C_sp*intTime)**2).decompose().value)
                SNRfa = S/N if N > 0 else 0.

            # save all SNRs (planets and FA) to one array
            SNRinds = np.where(det)[0][tochar]
            SNR[SNRinds] = np.append(SNRplans, SNRfa)

            # now, store characterization status: 1 for full spectrum, 
            # -1 for partial spectrum, 0 for not characterized
            char = (SNR >= mode['SNR'])
            # initialize with full spectra
            characterized = char.astype(int)
            WAchar = WAs[char]*u.arcsec
            # find the current WAs of characterized planets
            WA = WAs*u.arcsec
            if FA:
                WAs = np.append(WAs, WAs[-1]*u.arcsec)

            all_full = np.copy(characterized)
            all_full[char] = 1
            if sInd not in self.sInd_charcounts.keys():
                self.sInd_charcounts[sInd] = all_full
            else:
                self.sInd_charcounts[sInd] = self.sInd_charcounts[sInd] + all_full
            # encode results in spectra lists (only for planets, not FA)
            charplans = characterized[:-1] if FA else characterized
            self.fullSpectra[pInds[charplans == 1]] += 1
            self.partialSpectra[pInds[charplans == -1]] += 1

        # in both cases (detection or false alarm), schedule a revisit 
        smin = np.min(SU.s[pInds[det]])
        Ms = TL.MsTrue[sInd]

        # if target in promoted_stars list, schedule revisit based off of semi-major axis
        if sInd in self.promoted_stars:
            sp = np.min(SU.a[pInds[det]]).to('AU')
            if np.any(det):
                pInd_smin = pInds[det][np.argmin(SU.a[pInds[det]])]
                Mp = SU.Mp[pInd_smin]
            else:
                Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm.copy() + T/3.
        # otherwise schedule revisit based off of seperation
        elif smin is not None:
            sp = smin
            if np.any(det):
                pInd_smin = pInds[det][np.argmin(SU.s[pInds[det]])]
                Mp = SU.Mp[pInd_smin]
            else:
                Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm.copy() + T/2.
        # otherwise, revisit based on average of population semi-major axis and mass
        else:
            sp = SU.s.mean()
            Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm.copy() + 0.75*T

        # finally, populate the revisit list (NOTE: sInd becomes a float)
        revisit = np.array([sInd, t_rev.to('day').value])
        if self.occ_starRevisit.size == 0:
            self.occ_starRevisit = np.array([revisit])
        else:
            revInd = np.where(self.occ_starRevisit[:,0] == sInd)[0]
            if revInd.size == 0:
                self.occ_starRevisit = np.vstack((self.occ_starRevisit, revisit))
            else:
                self.occ_starRevisit[revInd, 1] = revisit[1]

        # add stars to filter list
        if np.any(characterized.astype(int) == 1):
            top_HIPs = self.occHIPs[:self.topstars]

            # if a top star has had max_successful_chars remove from list
            if np.any(self.sInd_charcounts[sInd] >= self.max_successful_chars):
                self.ignore_stars.append(sInd)

            # if a promoted star has an earthlike char, then ignore
            # if sInd in self.promoted_stars:
            #     c_plans = pInds[charplans == 1]
            #     if np.any(np.logical_and((SU.a[c_plans] > .95*u.AU),(SU.a[c_plans] < 1.67*u.AU))):
            #         if np.any((.8*(SU.a[c_plans]**-.5).value < SU.Rp[c_plans].value) & (SU.Rp[c_plans].value < 1.4)):
            #             self.ignore_stars.append(sInd)

        return characterized.astype(int), fZ, systemParams, SNR, intTime


    def revisitFilter(self, sInds, tmpCurrentTimeNorm):
        """Helper method for Overloading Revisit Filtering

        Args:
            sInds - indices of stars still in observation list
            tmpCurrentTimeNorm (MJD) - the simulation time after overhead was added in MJD form
        Returns:
            sInds - indices of stars still in observation list
        """
        tovisit = np.zeros(self.TargetList.nStars, dtype=bool)#tovisit is a boolean array containing the 
        if len(sInds) > 0:#so long as there is at least 1 star left in sInds
            tovisit[sInds] = ((self.starVisits[sInds] == min(self.starVisits[sInds])) \
                    & (self.starVisits[sInds] < self.nVisitsMax))# Checks that no star has exceeded the number of revisits
            if self.starRevisit.size != 0:#There is at least one revisit planned in starRevisit
                dt_rev = self.starRevisit[:,1]*u.day - tmpCurrentTimeNorm#absolute temporal spacing between revisit and now.

                #return indices of all revisits within a threshold dt_max of revisit day and indices of all revisits with no detections past the revisit time
                ind_rev2 = [int(x) for x in self.starRevisit[dt_rev < 0*u.d, 0] if (x in sInds)]
                tovisit[ind_rev2] = (self.starVisits[ind_rev2] < self.nVisitsMax)
            sInds = np.where(tovisit)[0]

        return sInds


    def scheduleRevisit(self, sInd, smin, det, pInds):
        """A Helper Method for scheduling revisits after observation detection

        Args:
            sInd - sInd of the star just detected
            smin - minimum separation of the planet to star of planet just detected
            det - 
            pInds - Indices of planets around target star
        Return:
            updates self.starRevisit attribute
        """
        TK = self.TimeKeeping
        TL = self.TargetList
        SU = self.SimulatedUniverse
        # in both cases (detection or false alarm), schedule a revisit 
        # based on minimum separation
        Ms = TL.MsTrue[sInd]
        if smin is not None and np.nan not in smin: #smin is None if no planet was detected
            sp = smin
            if np.any(det):
                pInd_smin = pInds[det][np.argmin(SU.s[pInds[det]])]
                Mp = SU.Mp[pInd_smin]
            else:
                Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm.copy() + T/2.
        # otherwise, revisit based on average of population semi-major axis and mass
        else:
            sp = SU.s.mean()
            Mp = SU.Mp.mean()
            mu = const.G*(Mp + Ms)
            T = 2.*np.pi*np.sqrt(sp**3/mu)
            t_rev = TK.currentTimeNorm.copy() + 0.75*T
        # if no detections then schedule revisit based off of revisit_wait
        t_rev = TK.currentTimeNorm.copy() + self.revisit_wait[sInd]
        # finally, populate the revisit list (NOTE: sInd becomes a float)
        revisit = np.array([sInd, t_rev.to('day').value])
        if self.starRevisit.size == 0:#If starRevisit has nothing in it
            self.starRevisit = np.array([revisit])#initialize sterRevisit
        else:
            revInd = np.where(self.starRevisit[:,0] == sInd)[0]#indices of the first column of the starRevisit list containing sInd 
            if revInd.size == 0:
                self.starRevisit = np.vstack((self.starRevisit, revisit))
            else:
                self.starRevisit[revInd,1] = revisit[1]#over

