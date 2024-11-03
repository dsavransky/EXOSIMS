# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import EXOSIMS
import os
import numpy as np
import logging
import astropy.units as u
import time

Logger = logging.getLogger(__name__)


class SS_char_only2(SurveySimulation):
    def __init__(self, occHIPs=[], **specs):

        SurveySimulation.__init__(self, **specs)

        self._outspec["occHIPs"] = occHIPs

        if occHIPs is not []:
            occHIPs_path = os.path.join(EXOSIMS.__path__[0], "Scripts", occHIPs)
            assert os.path.isfile(occHIPs_path), "%s is not a file." % occHIPs_path
            with open(occHIPs_path, "r") as ofile:
                HIPsfile = ofile.read()
            self.occHIPs = HIPsfile.split(",")
            if len(self.occHIPs) <= 1:
                self.occHIPs = HIPsfile.split("\n")
        else:
            self.occHIPs = occHIPs

    def run_sim(self):
        """Performs the survey simulation"""

        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping

        # TODO: start using this self.currentSep
        # set occulter separation if haveOcculter
        if OS.haveOcculter:
            self.currentSep = Obs.occulterSep

        # choose observing modes selected for detection (default marked with a flag)
        allModes = OS.observingModes
        det_mode = list(filter(lambda mode: mode["detectionMode"], allModes))[0]
        # and for characterization (default is first spectro/IFS mode)
        spectroModes = list(
            filter(lambda mode: "spec" in mode["inst"]["name"], allModes)
        )
        if np.any(spectroModes):
            char_mode = spectroModes[0]
        # if no spectro mode, default char mode is first observing mode
        else:
            char_mode = allModes[0]

        # begin Survey, and loop until mission is finished
        log_begin = "OB%s: survey beginning." % (TK.OBnumber + 1)
        self.logger.info(log_begin)
        self.vprint(log_begin)
        t0 = time.time()
        sInd = None
        cnt = 0
        while not TK.mission_is_over():

            # save the start time of this observation (BEFORE any OH/settling/slew time)
            TK.obsStart = TK.currentTimeNorm.to("day")

            # acquire the NEXT TARGET star index and create DRM
            DRM, sInd, det_intTime = self.next_target(sInd, det_mode)
            assert det_intTime != 0, "Integration time can't be 0."

            if sInd is not None:
                cnt += 1
                # get the index of the selected target for the extended list
                if TK.currentTimeNorm > TK.missionLife and len(self.starExtended) == 0:
                    for i in range(len(self.DRM)):
                        if np.any([x == 1 for x in self.DRM[i]["plan_detected"]]):
                            self.starExtended = np.unique(
                                np.append(self.starExtended, self.DRM[i]["star_ind"])
                            )

                # beginning of observation, start to populate DRM
                DRM["star_ind"] = sInd
                DRM["star_name"] = TL.Name[sInd]
                DRM["arrival_time"] = TK.currentTimeNorm.to("day")
                DRM["OB_nb"] = TK.OBnumber + 1
                pInds = np.where(SU.plan2star == sInd)[0]
                DRM["plan_inds"] = pInds.astype(int)
                log_obs = (
                    "  Observation #%s, target #%s/%s with %s planet(s), "
                    + "mission time: %s"
                ) % (cnt, sInd + 1, TL.nStars, len(pInds), TK.obsStart.round(2))
                self.logger.info(log_obs)

                self.vprint(log_obs)

                FA = False

                # PERFORM CHARACTERIZATION and populate spectra list attribute
                if char_mode["SNR"] not in [0, np.inf]:
                    (
                        characterized,
                        char_fZ,
                        char_systemParams,
                        char_SNR,
                        char_intTime,
                    ) = self.observation_characterization(sInd, char_mode)
                else:
                    char_intTime = None
                    lenChar = len(pInds) + 1 if FA else len(pInds)
                    characterized = np.zeros(lenChar, dtype=float)
                    char_SNR = np.zeros(lenChar, dtype=float)
                    char_fZ = 0.0 / u.arcsec**2
                    char_systemParams = SU.dump_system_params(sInd)
                assert char_intTime != 0, "Integration time can't be 0."
                # update the occulter wet mass
                if OS.haveOcculter and char_intTime is not None:
                    DRM = self.update_occulter_mass(DRM, sInd, char_intTime, "char")
                # populate the DRM with characterization results
                DRM["char_time"] = (
                    char_intTime.to("day") if char_intTime is not None else 0.0 * u.day
                )
                DRM["char_status"] = characterized[:-1] if FA else characterized
                DRM["char_SNR"] = char_SNR[:-1] if FA else char_SNR
                DRM["char_fZ"] = char_fZ.to("1/arcsec2")
                DRM["char_params"] = char_systemParams
                # populate the DRM with FA results
                DRM["FA_det_status"] = int(FA)
                DRM["FA_char_status"] = characterized[-1] if FA else 0
                DRM["FA_char_SNR"] = char_SNR[-1] if FA else 0.0
                DRM["FA_char_fEZ"] = (
                    self.lastDetected[sInd, 1][-1] / u.arcsec**2
                    if FA
                    else 0.0 / u.arcsec**2
                )
                DRM["FA_char_dMag"] = self.lastDetected[sInd, 2][-1] if FA else 0.0
                DRM["FA_char_WA"] = (
                    self.lastDetected[sInd, 3][-1] * u.arcsec if FA else 0.0 * u.arcsec
                )

                DRM["char_mode"] = dict(char_mode)
                del DRM["char_mode"]["inst"], DRM["char_mode"]["syst"]

                # append result values to self.DRM
                self.DRM.append(DRM)

                # calculate observation end time
                TK.obsEnd = TK.currentTimeNorm.to("day")

                # with prototype TimeKeeping, if no OB duration was specified, advance
                # to the next OB with timestep equivalent to time spent on one target
                if np.isinf(TK.OBduration):
                    obsLength = (TK.obsEnd - TK.obsStart).to("day")
                    TK.next_observing_block(dt=obsLength)

        else:
            dtsim = (time.time() - t0) * u.s
            log_end = (
                "Mission complete: no more time available.\n"
                + "Simulation duration: %s.\n" % dtsim.astype("int")
                + "Results stored in SurveySimulation.DRM (Design Reference Mission)."
            )
            self.logger.info(log_end)

            self.vprint(log_end)

    def choose_next_target(self, old_sInd, sInds, slewTimes, intTimes):
        """Choose next target at random

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
            sInd (integer):
                Index of next target star

        """

        TL = self.TargetList

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        occ_sInds = np.where(np.in1d(TL.Name, self.occHIPs))[0]
        n_sInds = np.intersect1d(sInds, occ_sInds)

        # pick one
        if len(n_sInds) == 0:
            sInd = np.random.choice(sInds)
        else:
            sInd = np.random.choice(n_sInds)

        return sInd

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
        koMap = self.koMaps[mode["syst"]["name"]]

        # find indices of planets around the target
        pInds = np.where(SU.plan2star == sInd)[0]
        # get the last detected planets, and check if there was a FA
        # det = self.lastDetected[sInd,0]
        det = np.ones(pInds.size, dtype=bool)
        fEZs = SU.fEZ[pInds].to("1/arcsec2").value
        dMags = SU.dMag[pInds]
        WAs = SU.WA[pInds].to("arcsec").value

        FA = det.size == pInds.size + 1
        if FA:
            pIndsDet = np.append(pInds, -1)[det]
        else:
            pIndsDet = pInds[det]

        # initialize outputs, and check if any planet to characterize
        characterized = np.zeros(det.size, dtype=int)
        fZ = 0.0 / u.arcsec**2
        systemParams = SU.dump_system_params(
            sInd
        )  # write current system params by default
        SNR = np.zeros(len(det))
        intTime = None
        if len(det) == 0:  # nothing to characterize
            return characterized, fZ, systemParams, SNR, intTime

        # look for last detected planets that have not been fully characterized
        if not (FA):  # only true planets, no FA
            tochar = self.fullSpectra[pIndsDet] == 0
        else:  # mix of planets and a FA
            truePlans = pIndsDet[:-1]
            tochar = np.append((self.fullSpectra[truePlans] == 0), True)

        # look for last detected planets that have not been fully characterized
        tochar = np.zeros(len(det), dtype=bool)
        if not (FA):
            tochar[det] = self.fullSpectra[pInds[det]] != 1
        elif pInds[det].size > 1:
            tochar[det] = np.append((self.fullSpectra[pInds[det][:-1]] != 1), True)
        else:
            tochar[det] = np.array([True])

        # 1/ find spacecraft orbital START position and check keepout angle
        if np.any(tochar):
            # start times
            startTime = TK.currentTimeAbs
            startTimeNorm = TK.currentTimeNorm
            # planets to characterize
            koTimeInd = np.where(np.round(startTime.value) - self.koTimes.value == 0)[
                0
            ][
                0
            ]  # find indice where koTime is startTime[0]
            # wherever koMap is 1, the target is observable
            tochar[tochar] = koMap[sInd][koTimeInd]

        # 2/ if any planet to characterize, find the characterization times
        if np.any(tochar):
            # propagate the whole system to match up with current time
            # calculate characterization times at the detected fEZ, dMag, and WA
            fZ = ZL.fZ(Obs, TL, sInd, startTime, mode)
            # fEZ = self.lastDetected[sInd,1][tochar]/u.arcsec**2
            # dMag = self.lastDetected[sInd,2][tochar]
            # WA = self.lastDetected[sInd,3][tochar]*u.mas
            fEZ = fEZs[tochar] / u.arcsec**2
            dMag = dMags[tochar]
            WAp = WAs[tochar] * u.arcsec

            intTimes = np.zeros(len(pInds)) * u.d
            intTimes[tochar] = OS.calc_intTime(TL, sInd, fZ, fEZ, dMag, WAp, mode)
            intTimes[~np.isfinite(intTimes)] = 0 * u.d
            # add a predetermined margin to the integration times
            intTimes = intTimes * (1 + self.charMargin)
            # apply time multiplier
            totTimes = intTimes * (mode["timeMultiplier"])
            # end times
            endTimes = startTime + totTimes
            endTimesNorm = startTimeNorm + totTimes
            # planets to characterize
            tochar = (
                (totTimes > 0)
                & (totTimes <= OS.intCutoff)
                & (endTimesNorm <= TK.OBendTimes[TK.OBnumber])
            )

        # 3/ is target still observable at the end of any char time?
        if np.any(tochar) and Obs.checkKeepoutEnd:
            koTimeInds = np.zeros(len(endTimes.value[tochar]), dtype=int)
            # find index in koMap where each endTime is closest to koTimes
            for t, endTime in enumerate(endTimes.value[tochar]):
                if endTime > self.koTimes.value[-1]:
                    # case where endTime exceeds largest koTimes element
                    endTimeInBounds = np.where(
                        np.floor(endTime) - self.koTimes.value == 0
                    )[0]
                    koTimeInds[t] = (
                        endTimeInBounds[0] if endTimeInBounds.size != 0 else -1
                    )
                else:
                    koTimeInds[t] = np.where(
                        np.round(endTime) - self.koTimes.value == 0
                    )[0][
                        0
                    ]  # find indice where koTime is endTimes[0]
            tochar[tochar] = [koMap[sInd][koT] if koT >= 0 else 0 for koT in koTimeInds]

        # 4/ if yes, perform the characterization for the maximum char time
        if np.any(tochar):
            intTime = np.max(intTimes[tochar])
            pIndsChar = pIndsDet[tochar]
            log_char = "   - Charact. planet(s) %s (%s/%s detected)" % (
                pIndsChar,
                len(pIndsChar),
                len(pIndsDet),
            )
            self.logger.info(log_char)

            self.vprint(log_char)

            # SNR CALCULATION:
            # first, calculate SNR for observable planets (without false alarm)
            planinds = pIndsChar[:-1] if pIndsChar[-1] == -1 else pIndsChar
            SNRplans = np.zeros(len(planinds))
            if len(planinds) > 0:
                # initialize arrays for SNR integration
                fZs = np.zeros(self.ntFlux) / u.arcsec**2
                systemParamss = np.empty(self.ntFlux, dtype="object")
                Ss = np.zeros((self.ntFlux, len(planinds)))
                Ns = np.zeros((self.ntFlux, len(planinds)))
                # integrate the signal (planet flux) and noise
                dt = intTime / self.ntFlux
                for i in range(self.ntFlux):
                    # allocate first half of dt
                    TK.allocate_time(dt / 2.0)
                    # calculate current zodiacal light brightness
                    fZs[i] = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
                    # propagate the system to match up with current time
                    SU.propag_system(sInd, TK.currentTimeNorm - self.propagTimes[sInd])
                    self.propagTimes[sInd] = TK.currentTimeNorm
                    # save planet parameters
                    systemParamss[i] = SU.dump_system_params(sInd)
                    # calculate signal and noise (electron count rates)
                    Ss[i, :], Ns[i, :] = self.calc_signal_noise(
                        sInd, planinds, dt, mode, fZ=fZs[i]
                    )
                    # allocate second half of dt
                    TK.allocate_time(dt / 2.0)

                # average output parameters
                fZ = np.mean(fZs)
                systemParams = {
                    key: sum([systemParamss[x][key] for x in range(self.ntFlux)])
                    / float(self.ntFlux)
                    for key in sorted(systemParamss[0])
                }
                # calculate planets SNR
                S = Ss.sum(0)
                N = Ns.sum(0)
                SNRplans[N > 0] = S[N > 0] / N[N > 0]
                # allocate extra time for timeMultiplier
                extraTime = intTime * (mode["timeMultiplier"] - 1)
                TK.allocate_time(extraTime)

            # if only a FA, just save zodiacal brightness in the middle of the
            # integration
            else:
                totTime = intTime * (mode["timeMultiplier"])
                TK.allocate_time(totTime / 2.0)
                fZ = ZL.fZ(Obs, TL, sInd, TK.currentTimeAbs, mode)[0]
                TK.allocate_time(totTime / 2.0)

            # calculate the false alarm SNR (if any)
            SNRfa = []
            if pIndsChar[-1] == -1:
                fEZ = fEZs[-1] / u.arcsec**2
                dMag = dMags[-1]
                WA = WAs[-1] * u.arcsec
                C_p, C_b, C_sp = OS.Cp_Cb_Csp(TL, sInd, fZ, fEZ, dMag, WA, mode)
                S = (C_p * intTime).decompose().value
                N = np.sqrt((C_b * intTime + (C_sp * intTime) ** 2).decompose().value)
                SNRfa = S / N if N > 0 else 0.0

            # save all SNRs (planets and FA) to one array
            SNRinds = np.where(det)[0][tochar]
            SNR[SNRinds] = np.append(SNRplans, SNRfa)

            # now, store characterization status: 1 for full spectrum,
            # -1 for partial spectrum, 0 for not characterized
            char = SNR >= mode["SNR"]
            # initialize with full spectra
            characterized = char.astype(int)
            WAchar = WAs[char] * u.arcsec
            # find the current WAs of characterized planets
            WA = WAs * u.arcsec
            if FA:
                WAs = np.append(WAs, WAs[-1] * u.arcsec)
            # check for partial spectra
            IWA_max = mode["IWA"] * (1 + mode["BW"] / 2.0)
            OWA_min = mode["OWA"] * (1 - mode["BW"] / 2.0)
            char[char] = (WAchar < IWA_max) | (WAchar > OWA_min)
            characterized[char] = -1
            # encode results in spectra lists (only for planets, not FA)
            charplans = characterized[:-1] if FA else characterized
            self.fullSpectra[pInds[charplans == 1]] += 1
            self.partialSpectra[pInds[charplans == -1]] += 1

        return characterized.astype(int), fZ, systemParams, SNR, intTime
