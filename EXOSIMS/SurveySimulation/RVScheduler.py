import copy
import math
from collections import Counter
from pathlib import Path

import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import RVtoImaging.utils as utils

from EXOSIMS.SurveySimulation.coroOnlyScheduler import coroOnlyScheduler
from EXOSIMS.util.deltaMag import deltaMag


class RVScheduler(coroOnlyScheduler):
    """RVScheduler

    This class optimizes an observing schedule based on precursor radial
    velocity orbit fits.

    Args:
        **specs:
            user specified values

    Notes:

    """

    def __init__(self, **specs):

        # initialize the prototype survey
        coroOnlyScheduler.__init__(self, **specs)
        self.forced_observations = None
        self.forced_observations_remain = False

    def sim_fixed_schedule(self, schedule):
        """
        Method that takes a schedule and simulates observations

        Args:
            schedule (~numpy.ndarray(float))
        """
        OS = self.OpticalSystem
        SU = self.SimulatedUniverse
        TK = self.TimeKeeping
        TL = self.TargetList
        # Comp = self.Completeness
        base_det_mode = list(
            filter(lambda mode: mode["detectionMode"], OS.observingModes)
        )[0]
        df = schedule.sort_values("time")
        schedule_detections = 0
        schedule_failures = 0
        unexpected_detections = 0

        res = {}
        suc_coeffs = []
        fail_coeffs = []

        suc_errors = []
        fail_errors = []

        suc_periods = []
        fail_periods = []

        all_detected_pinds = []

        # for obs_ind in range(schedule.shape[0]):
        for _, row in df.iterrows():
            sInd = row.sInd
            obs_time = row.time
            int_time = row.int_time
            fitted_as = row.all_planet_as * u.AU
            system_params = SU.dump_system_params(sInd)
            pInds = np.where(self.SimulatedUniverse.plan2star == sInd)[0]
            true_as = SU.a[pInds]
            pInds = np.where(SU.plan2star == sInd)[0]
            # pInd = pInds[np.argmin(np.abs(np.median(pop.a) - SU.a[pInds]))]

            # Get the closest planet index
            fitted_pinds = np.zeros(len(row.all_planet_pops), dtype=int)
            fitted_error = np.zeros(len(row.all_planet_pops))
            for i, fita in enumerate(fitted_as):
                diffs = np.abs(fita - true_as)
                closest_ind = np.argmin(diffs)
                percent_error = 100 * np.abs(
                    ((fita - true_as[closest_ind]) / fita).value
                )
                fitted_pinds[i] = closest_ind
                fitted_error[i] = percent_error
            unfitted_pinds = np.delete(np.arange(0, len(pInds)), fitted_pinds)

            expected_detection_inds = fitted_pinds[np.array(row.all_planet_thresholds)]
            expected_detection_coeff = np.array(row.all_planet_coeffs)[
                row.all_planet_thresholds
            ]
            expected_detection_error = fitted_error[np.array(row.all_planet_thresholds)]
            expected_detection_period = fitted_as[np.array(row.all_planet_thresholds)]
            expected_detection_est_WA = np.array(row.WAs)[
                np.array(row.all_planet_thresholds)
            ]
            expected_detection_est_dMag = np.array(row.dMags)[
                np.array(row.all_planet_thresholds)
            ]
            expected_detection_true_WA = system_params["WA"][expected_detection_inds]
            expected_detection_true_dMag = system_params["dMag"][
                expected_detection_inds
            ]
            TK.advanceToAbsTime(obs_time)
            detected, fZ, systemParams, SNR, FA = self.observation_detection(
                sInd, int_time, base_det_mode
            )
            fit_detections = np.array(detected)[expected_detection_inds]
            fit_snrs = SNR[expected_detection_inds]
            for (
                fit_detection,
                pop_ind,
                coeff,
                error,
                period,
                ind,
                snr,
                est_WA,
                est_dMag,
                true_WA,
                true_dMag,
            ) in zip(
                fit_detections,
                np.where(np.array(row.all_planet_thresholds) == True)[0],
                expected_detection_coeff,
                expected_detection_error,
                expected_detection_period,
                expected_detection_inds,
                fit_snrs,
                expected_detection_est_WA,
                expected_detection_est_dMag,
                expected_detection_true_WA,
                expected_detection_true_dMag,
            ):
                true_pind = pInds[ind]
                if true_pind not in res.keys():
                    res[true_pind] = {
                        "success": 0,
                        "fail": 0,
                        "sInd": sInd,
                        "pop": row.all_planet_pops[pop_ind],
                        "period": period,
                        "percent_error": error,
                        "det_status": [],
                        "coeff": [],
                        "obs_time": [],
                        "int_time": [],
                        "SNR": [],
                        "est_WA": [],
                        "true_WA": [],
                        "est_dMag": [],
                        "true_dMag": [],
                        "fZ": [],
                        "fEZ": [],
                    }
                if fit_detection == 1:
                    schedule_detections += 1
                    suc_coeffs.append(coeff)
                    suc_errors.append(error)
                    suc_periods.append(period)
                    res[true_pind]["success"] += 1
                else:
                    schedule_failures += 1
                    fail_coeffs.append(coeff)
                    fail_errors.append(error)
                    fail_periods.append(period)
                    res[true_pind]["fail"] += 1
                res[true_pind]["det_status"].append(fit_detection)
                res[true_pind]["coeff"].append(coeff)
                res[true_pind]["obs_time"].append(obs_time)
                res[true_pind]["int_time"].append(int_time)
                res[true_pind]["SNR"].append(snr)
                res[true_pind]["est_WA"].append(est_WA)
                res[true_pind]["est_dMag"].append(est_dMag)
                res[true_pind]["true_WA"].append(true_WA)
                res[true_pind]["true_dMag"].append(true_dMag)
                res[true_pind]["fZ"].append(fZ)
                res[true_pind]["fEZ"].append(systemParams["fEZ"][closest_ind])
            other_detections = np.array(detected)[unfitted_pinds]
            unexpected_detections += len(np.where(other_detections == 1)[0])
            if len(np.where(detected == 1)[0]) > 0:
                all_detected_pinds.extend(pInds[np.where(detected == 1)[0]])
        resdf = pd.DataFrame.from_dict(res)
        flat_info = {
            "det_status": [],
            "SNR": [],
            "obs_time": [],
            "int_time": [],
            "pind": [],
            "sInd": [],
            "dist": [],
            "Vmag": [],
            # "fit_prob": [],
            "planets_fitted": [],
            "n_rv_obs": [],
            "rv_obs_baseline": [],
            "best_rv_precision": [],
            "truea": [],
            "esta": [],
            "esttc": [],
            "trueecc": [],
            "estecc": [],
            "truep": [],
            "estp": [],
            "estK": [],
            "trueI": [],
            "period_error": [],
            "coeff": [],
            "est_WA": [],
            "true_WA": [],
            "est_dMag": [],
            "true_dMag": [],
            "fZ": [],
            "fEZ": [],
        }
        for pind in resdf.keys():
            nobs = len(resdf[pind].obs_time)
            for i in range(nobs):
                flat_info["pind"].append(pind)
                flat_info["sInd"].append(resdf[pind]["sInd"])
                flat_info["dist"].append(TL.dist[resdf[pind]["sInd"]].to(u.pc).value)
                flat_info["Vmag"].append(TL.Vmag[resdf[pind]["sInd"]])
                pop = resdf[pind]["pop"]
                # flat_info["fit_prob"].append(pop.chains_spec["best_prob"])
                flat_info["planets_fitted"].append(pop.chains_spec["planets_fitted"])
                flat_info["n_rv_obs"].append(pop.chains_spec["observations"])
                flat_info["rv_obs_baseline"].append(
                    pop.chains_spec["observational_baseline"]
                )
                flat_info["best_rv_precision"].append(pop.chains_spec["best_precision"])
                # flat_info["pop"].append(resdf[pind]["pop"])
                flat_info["esta"].append(np.median(pop.a.to(u.AU).value))
                flat_info["esttc"].append(np.median(pop.T_c.jd))
                flat_info["estecc"].append(np.median(pop.e))
                flat_info["estK"].append(np.median(pop.K.value))
                flat_info["estp"].append(np.median(pop.p))

                # flat_info["trueperiod"].append(SU.[pind]["pop"].T.to(u.yr).value)
                flat_info["truea"].append(SU.a.to(u.AU).value[pind])
                flat_info["trueecc"].append(SU.e[pind])
                flat_info["truep"].append(SU.p[pind])
                flat_info["trueI"].append(SU.I[pind].to(u.deg).value)

                flat_info["period_error"].append(resdf[pind]["percent_error"])
                flat_info["det_status"].append(resdf[pind]["det_status"][i])
                flat_info["coeff"].append(resdf[pind]["coeff"][i])
                flat_info["obs_time"].append(resdf[pind]["obs_time"][i].jd)
                flat_info["int_time"].append(resdf[pind]["int_time"][i].to(u.d).value)
                flat_info["SNR"].append(resdf[pind]["SNR"][i])
                flat_info["est_WA"].append(resdf[pind]["est_WA"][i])
                flat_info["est_dMag"].append(resdf[pind]["est_dMag"][i])
                flat_info["true_WA"].append(resdf[pind]["true_WA"][i].value)
                flat_info["true_dMag"].append(resdf[pind]["true_dMag"][i])
                flat_info["fZ"].append(resdf[pind]["fZ"][i].value)
                flat_info["fEZ"].append(resdf[pind]["fEZ"][i].value)
        flatdf = pd.DataFrame.from_dict(flat_info)
        flatdf["err_dMag"] = flatdf["est_dMag"] - flatdf["true_dMag"]
        flatdf["err_WA"] = flatdf["est_WA"] - flatdf["true_WA"]
        summary_stats = {
            "planets_in_universe": len(SU.plan2star),
            "unique_planets_detected": len(np.unique(all_detected_pinds)),
            "obs_per_detected_planet": Counter(all_detected_pinds),
            "n_observations": df.shape[0],
            "scheduled_success": schedule_detections,
            "scheduled_failure": schedule_failures,
            "unexpected_detections": unexpected_detections,
            "one_detection": sum(resdf.loc["success"] == 1),
            "two_detections": sum(resdf.loc["success"] == 2),
            "three_plus_detections": sum(resdf.loc["success"] > 2),
            "int_time": sum(resdf.loc["int_time"].sum()),
        }
        resdf = resdf.drop("pop")
        return resdf, flatdf, summary_stats

    def instantiate_forced_observations(self, forced_observations, systems):
        """
        Create the data necessary to force observations based on precursor
        radial velocity data.
        """
        self.forced_observations = forced_observations
        self.next_forced_ind = 0
        self.next_forced_obs = self.forced_observations[0]
        self.forced_obs_inds = [
            np.argwhere(self.TargetList.Name == obs.star_name.replace("P", "P "))[0][0]
            for obs in self.forced_observations
        ]
        # self.forced_obs_inds = [obs.star_ind for obs in self.forced_observations]
        self.forced_observations_remain = True
        self.systems = systems

    def random_walk_sim(self, nsims, missionLength):
        OS = self.OpticalSystem
        SU = self.SimulatedUniverse
        TK = self.TimeKeeping
        TL = self.TargetList
        Comp = self.Completeness
        ZL = self.ZodiacalLight
        mode = list(filter(lambda mode: mode["detectionMode"], OS.observingModes))[0]
        sInds = np.arange(0, len(TL.Name), 1)
        int_times = OS.calc_intTime(
            TL, sInds, ZL.fZ0, ZL.fEZ0, TL.int_dMag, TL.int_WA, mode
        )
        # Use intCutoff
        sInds = sInds[int_times <= OS.intCutoff]
        kot = self.koTimes
        kom = self.koMaps[mode["syst"]["name"]]
        randomdf = pd.DataFrame(
            np.zeros((nsims, 7)),
            columns=[
                "detections",
                "unique_detections",
                "observations",
                "int_time",
                "one_detection",
                "two_detections",
                "three_plus_detections",
            ],
        )
        for sim in range(nsims):
            self.reset_sim(rewindPlanets=True, genNewPlanets=False)
            start_time = TK.currentTimeAbs.jd

            pdf = pd.DataFrame(
                np.zeros((len(SU.plan2star), 2)), columns=["detections", "observations"]
            )
            star_observations = 0
            planet_observations = 0
            total_int_time = 0 * u.d
            while TK.currentTimeAbs.jd < (start_time + missionLength.to(u.d).value):
                # Keepout inds
                start_dec, start_int = math.modf(TK.currentTimeAbs.jd)
                if start_dec >= 0.5:
                    # index of the closest kotime (floor) from the observation
                    # start time
                    ko_start_ind = np.where(kot.jd == start_int + 0.5)[0][0]
                else:
                    ko_start_ind = np.where(kot.jd == start_int - 0.5)[0][0]

                # Get the keepout end time for each star
                not_in_keepout_sInds = []
                for sInd in sInds:
                    end_dec, end_int = math.modf(
                        (TK.currentTimeAbs + int_times[sInd]).jd
                    )
                    if end_dec >= 0.5:
                        ko_end_ind = np.where(kot.jd == end_int + 0.5)[0][0]
                    else:
                        ko_end_ind = np.where(kot.jd == end_int - 0.5)[0][0]

                    if ko_start_ind == ko_end_ind:
                        # If there's only only ko block to consider
                        if kom[sInd, ko_start_ind]:
                            not_in_keepout_sInds.append(sInd)
                    else:
                        # If we span multiple blocks
                        if np.all(kom[sInd, ko_start_ind:ko_end_ind]):
                            not_in_keepout_sInds.append(sInd)
                next_sInd = np.random.choice(not_in_keepout_sInds)

                # TK.advanceToAbsTime(obs_time)
                detected, fZ, systemParams, SNR, FA = self.observation_detection(
                    next_sInd, int_times[next_sInd], mode
                )
                star_observations += 1
                planet_observations += len(detected)
                total_int_time += int_times[next_sInd]
                for i, pind in enumerate(np.where(SU.plan2star == next_sInd)[0]):
                    pind_detected = detected[i]
                    pdf.at[pind, "detections"] += pind_detected
                    pdf.at[pind, "observations"] += 1
                    # detected_pinds = np.where(SU.plan2star == next_sInd)[0][
                    #     detected.astype(bool)
                    # ]
                    # for pind in detected_pinds:
                    #     detected_inds[pind] += 1
            randomdf.at[sim, "detections"] = pdf.detections.sum()
            randomdf.at[sim, "unique_planets_detected"] = sum(pdf.detections > 0)
            randomdf.at[sim, "n_observations"] = star_observations
            randomdf.at[sim, "int_time"] = total_int_time.to(u.d).value
            randomdf.at[sim, "one_detection"] = sum(pdf.detections == 1)
            randomdf.at[sim, "two_detections"] = sum(pdf.detections == 2)
            randomdf.at[sim, "three_plus_detections"] = sum(pdf.detections > 2)

        return randomdf

    def next_target(self, old_sInd, det_modes, char_modes):
        """Finds index of next target star and calculates its integration time.

        This method chooses the next target star index based on which
        stars are available, their integration time, and maximum completeness.
        Returns None if no target could be found.

        Args:
            old_sInd (integer):
                Index of the previous target star
            mode (dict):
                Selected observing mode for detection

        Returns:
            tuple:
                DRM (dict):
                    Design Reference Mission, contains the results of one complete
                    observation (detection and characterization)
                sInd (integer):
                    Index of next target star. Defaults to None.
                intTime (astropy Quantity):
                    Selected star integration time for detection in units of day.
                    Defaults to None.
                waitTime (astropy Quantity):
                    a strategically advantageous amount of time to wait in the case
                    of an occulter for slew times

        """
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        SU = self.SimulatedUniverse

        # create DRM
        DRM = {}

        # allocate settling time + overhead time
        tmpCurrentTimeAbs = (
            TK.currentTimeAbs.copy() + Obs.settlingTime + det_modes[0]["syst"]["ohTime"]
        )
        tmpCurrentTimeNorm = (
            TK.currentTimeNorm.copy()
            + Obs.settlingTime
            + det_modes[0]["syst"]["ohTime"]
        )

        # create appropriate koMap
        koMap = self.koMaps[det_modes[0]["syst"]["name"]]
        char_koMap = self.koMaps[char_modes[0]["syst"]["name"]]

        # look for available targets
        # 1. initialize arrays
        slewTimes = np.zeros(TL.nStars) * u.d
        # fZs = np.zeros(TL.nStars) / u.arcsec**2.0
        # dV = np.zeros(TL.nStars) * u.m / u.s
        intTimes = np.zeros(TL.nStars) * u.d
        char_intTimes = np.zeros(TL.nStars) * u.d
        char_intTimes_no_oh = np.zeros(TL.nStars) * u.d
        # obsTimes = np.zeros([2, TL.nStars]) * u.d
        char_tovisit = np.zeros(TL.nStars, dtype=bool)
        sInds = np.arange(TL.nStars)

        # 2. find spacecraft orbital START positions (if occulter, positions
        # differ for each star) and filter out unavailable targets
        # sd = None

        # 2.1 filter out totTimes > integration cutoff
        if len(sInds.tolist()) > 0:
            char_sInds = np.intersect1d(sInds, self.promoted_stars)
            sInds = np.intersect1d(self.intTimeFilterInds, sInds)

        # start times, including slew times
        startTimes = tmpCurrentTimeAbs.copy() + slewTimes
        startTimesNorm = tmpCurrentTimeNorm.copy() + slewTimes

        # 2.5 Filter stars not observable at startTimes
        try:
            tmpIndsbool = list()
            for i in np.arange(len(sInds)):
                koTimeInd = np.where(
                    np.round(startTimes[sInds[i]].value) - self.koTimes.value == 0
                )[0][
                    0
                ]  # find indice where koTime is startTime[0]
                tmpIndsbool.append(
                    koMap[sInds[i]][koTimeInd].astype(bool)
                )  # Is star observable at time ind
            sInds = sInds[tmpIndsbool]
            del tmpIndsbool
        except:  # noqa: E722  # If there are no target stars to observe
            sInds = np.asarray([], dtype=int)

        try:
            tmpIndsbool = list()
            for i in np.arange(len(char_sInds)):
                koTimeInd = np.where(
                    np.round(startTimes[char_sInds[i]].value) - self.koTimes.value == 0
                )[0][
                    0
                ]  # find indice where koTime is startTime[0]
                tmpIndsbool.append(
                    char_koMap[char_sInds[i]][koTimeInd].astype(bool)
                )  # Is star observable at time ind
            char_sInds = char_sInds[tmpIndsbool]
            del tmpIndsbool
        except:  # noqa: E722 If there are no target stars to observe
            char_sInds = np.asarray([], dtype=int)

        # 3. filter out all previously (more-)visited targets, unless in
        if len(sInds.tolist()) > 0:
            sInds = self.revisitFilter(sInds, tmpCurrentTimeNorm)

        # revisit list, with time after start
        if np.any(char_sInds):

            char_tovisit[char_sInds] = (self.char_starVisits[char_sInds] == 0) & (
                self.char_starVisits[char_sInds] < self.nVisitsMax
            )
            if self.char_starRevisit.size != 0:
                dt_rev = TK.currentTimeNorm.copy() - self.char_starRevisit[:, 1] * u.day
                ind_rev = [
                    int(x)
                    for x in self.char_starRevisit[dt_rev > 0 * u.d, 0]
                    if x in char_sInds
                ]
                char_tovisit[ind_rev] = self.char_starVisits[ind_rev] < self.nVisitsMax
            char_sInds = np.where(char_tovisit)[0]

        # 4.1 calculate integration times for ALL preselected targets
        (
            maxIntTimeOBendTime,
            maxIntTimeExoplanetObsTime,
            maxIntTimeMissionLife,
        ) = TK.get_ObsDetectionMaxIntTime(Obs, det_modes[0])
        maxIntTime = min(
            maxIntTimeOBendTime,
            maxIntTimeExoplanetObsTime,
            maxIntTimeMissionLife,
            OS.intCutoff,
        )  # Maximum intTime allowed

        if len(sInds.tolist()) > 0:
            intTimes[sInds] = self.calc_targ_intTime(
                sInds, startTimes[sInds], det_modes[0]
            ) * (1 + self.detMargin)
            sInds = sInds[
                (intTimes[sInds] <= maxIntTime)
            ]  # Filters targets exceeding end of OB
            intTimes[np.where(np.isnan(intTimes))[0]] = np.finfo(np.float64).max * u.d
            endTimes = startTimes + intTimes

            if maxIntTime.value <= 0:
                sInds = np.asarray([], dtype=int)

        if len(char_sInds) > 0:
            for char_mode in char_modes:
                (
                    maxIntTimeOBendTime,
                    maxIntTimeExoplanetObsTime,
                    maxIntTimeMissionLife,
                ) = TK.get_ObsDetectionMaxIntTime(Obs, char_mode)
                char_maxIntTime = min(
                    maxIntTimeOBendTime,
                    maxIntTimeExoplanetObsTime,
                    maxIntTimeMissionLife,
                    OS.intCutoff,
                )  # Maximum intTime allowed

                char_mode_intTimes = np.zeros(TL.nStars) * u.d
                char_mode_intTimes[char_sInds] = self.calc_targ_intTime(
                    char_sInds, startTimes[char_sInds], char_mode
                ) * (1 + self.charMargin)

                # Adjust integration time for stars with known earths around them
                for char_star in char_sInds:
                    char_earths = np.intersect1d(
                        np.where(SU.plan2star == char_star)[0], self.known_earths
                    ).astype(int)
                    if np.any(char_earths):
                        fZ = ZL.fZ(Obs, TL, char_star, startTimes[char_star], char_mode)
                        fEZ = SU.fEZ[char_earths].to("1/arcsec2").value / u.arcsec**2
                        if SU.lucky_planets:
                            phi = (1 / np.pi) * np.ones(len(SU.d))
                            dMag = deltaMag(SU.p, SU.Rp, SU.d, phi)[
                                char_earths
                            ]  # delta magnitude
                            WA = np.arctan(SU.a / TL.dist[SU.plan2star]).to("arcsec")[
                                char_earths
                            ]  # working angle
                        else:
                            dMag = SU.dMag[char_earths]
                            WA = SU.WA[char_earths]

                        if np.all((WA < char_mode["IWA"]) | (WA > char_mode["OWA"])):
                            char_mode_intTimes[char_star] = 0.0 * u.d
                        else:
                            earthlike_inttimes = OS.calc_intTime(
                                TL, char_star, fZ, fEZ, dMag, WA, char_mode
                            ) * (1 + self.charMargin)
                            earthlike_inttime = earthlike_inttimes[
                                (earthlike_inttimes < char_maxIntTime)
                            ]
                            if len(earthlike_inttime) > 0:
                                char_mode_intTimes[char_star] = np.max(
                                    earthlike_inttime
                                )
                char_intTimes_no_oh += char_mode_intTimes
                char_intTimes += char_mode_intTimes + char_mode["syst"]["ohTime"]
            # Filter char_intTimes to make nan integration times correspond to the
            # maximum float value because Time cannot handle nan values
            char_intTimes[np.where(np.isnan(char_intTimes))[0]] = (
                np.finfo(np.float64).max * u.d
            )
            char_endTimes = (
                startTimes
                + (char_intTimes * char_mode["timeMultiplier"])
                + Obs.settlingTime
            )

            char_sInds = char_sInds[
                (char_intTimes_no_oh[char_sInds] > 0.0 * u.d)
            ]  # Filters with an inttime of 0

            if char_maxIntTime.value <= 0:
                char_sInds = np.asarray([], dtype=int)

        # 5 remove char targets on ignore_stars list
        sInds = np.setdiff1d(
            sInds, np.intersect1d(sInds, self.promoted_stars).astype(int)
        )
        char_sInds = np.setdiff1d(
            char_sInds, np.intersect1d(char_sInds, self.ignore_stars)
        )

        # 6.2 Filter off coronograph stars with too many visits and no detections
        no_dets = np.logical_and(
            (self.starVisits[sInds] > self.n_det_remove),
            (self.sInd_detcounts[sInds] == 0),
        )
        sInds = sInds[np.where(np.invert(no_dets))[0]]

        max_dets = np.where(self.sInd_detcounts[sInds] < self.max_successful_dets)[0]
        sInds = sInds[max_dets]

        # 5.1 TODO Add filter to filter out stars entering and exiting keepout
        # between startTimes and endTimes

        # 5.2 find spacecraft orbital END positions (for each candidate target),
        # and filter out unavailable targets
        if len(sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
            # endTimes may exist past koTimes so we have an exception to hand this case
            try:
                tmpIndsbool = list()
                for i in np.arange(len(sInds)):
                    koTimeInd = np.where(
                        np.round(endTimes[sInds[i]].value) - self.koTimes.value == 0
                    )[0][
                        0
                    ]  # find indice where koTime is endTime[0]
                    tmpIndsbool.append(
                        koMap[sInds[i]][koTimeInd].astype(bool)
                    )  # Is star observable at time ind
                sInds = sInds[tmpIndsbool]
                del tmpIndsbool
            except:  # noqa: E722
                sInds = np.asarray([], dtype=int)

        if len(char_sInds.tolist()) > 0 and Obs.checkKeepoutEnd:
            # try: # endTimes may exist past koTimes so we have an exception to
            # hand this case
            tmpIndsbool = list()
            for i in np.arange(len(char_sInds)):
                try:
                    koTimeInd = np.where(
                        np.round(char_endTimes[char_sInds[i]].value)
                        - self.koTimes.value
                        == 0
                    )[0][
                        0
                    ]  # find indice where koTime is endTime[0]
                    tmpIndsbool.append(
                        char_koMap[char_sInds[i]][koTimeInd].astype(bool)
                    )  # Is star observable at time ind
                except:  # noqa: E722
                    tmpIndsbool.append(False)
            if np.any(tmpIndsbool):
                char_sInds = char_sInds[tmpIndsbool]
            else:
                char_sInds = np.asarray([], dtype=int)
            del tmpIndsbool

        # t_det = 0 * u.d
        det_mode = copy.deepcopy(det_modes[0])

        if self.forced_observations is not None:
            # Remove sInds of stars that we have observations already for
            inds_to_keep = [
                i for i, ind in enumerate(sInds) if ind not in self.forced_obs_inds
            ]
            if len(inds_to_keep) > 0:
                sInds = sInds[inds_to_keep]
            else:
                sInds = np.array([])
            char_inds_to_keep = [
                i for i, ind in enumerate(char_sInds) if ind not in self.forced_obs_inds
            ]
            if len(char_inds_to_keep) > 0:
                char_sInds = char_sInds[char_inds_to_keep]
            else:
                char_sInds = np.array([])
            # intTimes = intTimes[inds_to_keep]

        # 6. choose best target from remaining
        if len(sInds.tolist()) > 0:
            # choose sInd of next target
            if np.any(char_sInds):
                try:
                    sInd, waitTime = self.choose_next_target(
                        old_sInd, char_sInds, slewTimes, char_intTimes[char_sInds]
                    )
                except:
                    breakpoint()
                # store selected star integration time
                intTime = char_intTimes[sInd]
            else:
                sInd, waitTime = self.choose_next_target(
                    old_sInd, sInds, slewTimes, intTimes[sInds]
                )
                # store selected star integration time
                intTime = intTimes[sInd]

            # Should Choose Next Target decide there are no stars it wishes to
            # observe at this time.
            if (sInd is None) and (waitTime is not None):
                self.vprint(
                    (
                        "There are no stars Choose Next Target would like to Observe. "
                        "Waiting {}"
                    ).format(waitTime)
                )
                return DRM, None, None, waitTime, det_mode
            elif (sInd is None) and (waitTime is None):
                self.vprint(
                    (
                        "There are no stars Choose Next Target would like to Observe "
                        "and waitTime is None"
                    )
                )
                return DRM, None, None, waitTime, det_mode

            # Perform dual band detections if necessary
            if (
                TL.int_WA[sInd] > det_modes[1]["IWA"]
                and TL.int_WA[sInd] < det_modes[1]["OWA"]
            ):
                det_mode["BW"] = det_mode["BW"] + det_modes[1]["BW"]
                det_mode["inst"]["sread"] = (
                    det_mode["inst"]["sread"] + det_modes[1]["inst"]["sread"]
                )
                det_mode["inst"]["idark"] = (
                    det_mode["inst"]["idark"] + det_modes[1]["inst"]["idark"]
                )
                det_mode["inst"]["CIC"] = (
                    det_mode["inst"]["CIC"] + det_modes[1]["inst"]["CIC"]
                )
                det_mode["syst"]["optics"] = np.mean(
                    (det_mode["syst"]["optics"], det_modes[1]["syst"]["optics"])
                )
                det_mode["instName"] = "combined"

            intTime = self.calc_targ_intTime(
                np.array([sInd]), startTimes[sInd], det_mode
            )[0] * (1 + self.detMargin)

            if intTime > maxIntTime and maxIntTime > 0 * u.d:
                intTime = maxIntTime

        # if no observable target, advanceTime to next Observable Target
        else:
            self.vprint(
                "No Observable Targets at currentTimeNorm= "
                + str(TK.currentTimeNorm.copy())
            )
            return DRM, None, None, None, det_mode

        # store normalized start time for future completeness update
        self.lastObsTimes[sInd] = startTimesNorm[sInd]

        if self.forced_observations_remain and (
            intTime + waitTime + TK.currentTimeAbs.copy() > self.next_forced_obs.time
        ):
            # sInd = self.next_forced_obs.star_ind
            sInd = np.argwhere(
                self.TargetList.Name
                == self.next_forced_obs.star_name.replace("P", "P ")
            )[0][0]
            intTime = self.next_forced_obs.int_time
            waitTime = self.next_forced_obs.time - TK.currentTimeAbs.copy()
            print(
                (
                    f"Forced observation of {self.next_forced_obs.star_name} at "
                    f"{self.next_forced_obs.time.decimalyear}"
                )
            )
            self.next_forced_ind += 1
            if self.next_forced_ind < len(self.forced_observations):
                self.next_forced_obs = self.forced_observations[self.next_forced_ind]
            else:
                self.forced_observations_remain = False
                print("\nDone with forced observations\n\n")
        return DRM, sInd, intTime, waitTime, det_mode

    # def observation_detection(self, sInd, intTime, mode):
    #     """Determines SNR and detection status for a given integration time
    #     for detection. Also updates the lastDetected and starRevisit lists.

    #     Args:
    #         sInd (int):
    #             Integer index of the star of interest
    #         intTime (~astropy.units.Quantity(~numpy.ndarray(float))):
    #             Selected star integration time for detection in units of day.
    #             Defaults to None.
    #         mode (dict):
    #             Selected observing mode for detection

    #     Returns:
    #         tuple:
    #             detected (numpy.ndarray(int)):
    #                 Detection status for each planet orbiting the observed target star:
    #                 1 is detection, 0 missed detection, -1 below IWA, and -2 beyond OWA
    #             fZ (astropy.units.Quantity(numpy.ndarray(float))):
    #                 Surface brightness of local zodiacal light in units of 1/arcsec2
    #             systemParams (dict):
    #                 Dictionary of time-dependant planet properties averaged over the
    #                 duration of the integration
    #             SNR (numpy.darray(float)):
    #                 Detection signal-to-noise ratio of the observable planets
    #             FA (bool):
    #                 False alarm (false positive) boolean

    #     """

    #     PPop = self.PlanetPopulation
    #     ZL = self.ZodiacalLight
    #     PPro = self.PostProcessing
    #     TL = self.TargetList
    #     SU = self.SimulatedUniverse
    #     Obs = self.Observatory
    #     TK = self.TimeKeeping

    #     # Save Current Time before attempting time allocation
    #     currentTimeNorm = TK.currentTimeNorm.copy()
    #     currentTimeAbs = TK.currentTimeAbs.copy()

    #     # Allocate Time
    #     extraTime = intTime * (mode["timeMultiplier"] - 1.0)  # calculates extraTime
    #     success = TK.allocate_time(
    #         intTime + extraTime + Obs.settlingTime + mode["syst"]["ohTime"], True
    #     )  # allocates time
    #     assert success, "Could not allocate observation detection time ({}).".format(
    #         intTime + extraTime + Obs.settlingTime + mode["syst"]["ohTime"]
    #     )
    #     dt = intTime / float(
    #         self.ntFlux
    #     )  # calculates partial time to be added for every ntFlux

    #     # find indices of planets around the target
    #     pInds = np.where(SU.plan2star == sInd)[0]

    #     # initialize outputs
    #     detected = np.array([], dtype=int)
    #     fZ = 0.0 / u.arcsec**2
    #     systemParams = SU.dump_system_params(
    #         sInd
    #     )  # write current system params by default
    #     SNR = np.zeros(len(pInds))

    #     # if any planet, calculate SNR
    #     if len(pInds) > 0:
    #         # initialize arrays for SNR integration
    #         fZs = np.zeros(self.ntFlux) / u.arcsec**2
    #         systemParamss = np.empty(self.ntFlux, dtype="object")
    #         Ss = np.zeros((self.ntFlux, len(pInds)))
    #         Ns = np.zeros((self.ntFlux, len(pInds)))
    #         # integrate the signal (planet flux) and noise
    #         timePlus = (
    #             Obs.settlingTime.copy() + mode["syst"]["ohTime"].copy()
    #         )  # accounts for the time since the current time
    #         for i in range(self.ntFlux):
    #             # allocate first half of dt
    #             timePlus += dt / 2.0
    #             # calculate current zodiacal light brightness
    #             fZs[i] = ZL.fZ(Obs, TL, sInd, currentTimeAbs + timePlus, mode)[0]
    #             # propagate the system to match up with current time
    #             SU.propag_system(
    #                 sInd, currentTimeNorm + timePlus - self.propagTimes[sInd]
    #             )
    #             self.propagTimes[sInd] = currentTimeNorm + timePlus
    #             # save planet parameters
    #             systemParamss[i] = SU.dump_system_params(sInd)
    #             # calculate signal and noise (electron count rates)
    #             Ss[i, :], Ns[i, :] = self.calc_signal_noise(
    #                 sInd, pInds, dt, mode, fZ=fZs[i]
    #             )
    #             # allocate second half of dt
    #             timePlus += dt / 2.0

    #         # average output parameters
    #         fZ = np.mean(fZs)
    #         systemParams = {
    #             key: sum([systemParamss[x][key] for x in range(self.ntFlux)])
    #             / float(self.ntFlux)
    #             for key in sorted(systemParamss[0])
    #         }
    #         # calculate SNR
    #         S = Ss.sum(0)
    #         N = Ns.sum(0)
    #         SNR[N > 0] = S[N > 0] / N[N > 0]

    #     # if no planet, just save zodiacal brightness in the middle of the integration
    #     else:
    #         totTime = intTime * (mode["timeMultiplier"])
    #         fZ = ZL.fZ(Obs, TL, sInd, currentTimeAbs + totTime / 2.0, mode)[0]

    #     # find out if a false positive (false alarm) or any false negative
    #     # (missed detections) have occurred
    #     FA, MD = PPro.det_occur(SNR, mode, TL, sInd, intTime)

    #     # populate detection status array
    #     # 1:detected, 0:missed, -1:below IWA, -2:beyond OWA
    #     if len(pInds) > 0:
    #         detected = (~MD).astype(int)
    #         WA = (
    #             np.array(
    #                 [
    #                     systemParamss[x]["WA"].to("arcsec").value
    #                     for x in range(len(systemParamss))
    #                 ]
    #             )
    #             * u.arcsec
    #         )
    #         detected[np.all(WA < mode["IWA"], 0)] = -1
    #         detected[np.all(WA > mode["OWA"], 0)] = -2

    #     # if planets are detected, calculate the minimum apparent separation
    #     smin = None
    #     det = detected == 1  # If any of the planets around the star have been detected
    #     if np.any(det):
    #         smin = np.min(SU.s[pInds[det]])
    #         log_det = "   - Detected planet inds %s (%s/%s)" % (
    #             pInds[det],
    #             len(pInds[det]),
    #             len(pInds),
    #         )
    #         self.logger.info(log_det)
    #         self.vprint(log_det)

    #     # populate the lastDetected array by storing det, fEZ, dMag, and WA
    #     self.lastDetected[sInd, :] = [
    #         det,
    #         systemParams["fEZ"].to("1/arcsec2").value,
    #         systemParams["dMag"],
    #         systemParams["WA"].to("arcsec").value,
    #     ]

    #     # in case of a FA, generate a random delta mag (between PPro.FAdMag0 and
    #     # TL.saturation_dMag) and working angle (between IWA and min(OWA, a_max))
    #     if FA:
    #         WA = (
    #             np.random.uniform(
    #                 mode["IWA"].to("arcsec").value,
    #                 np.minimum(mode["OWA"], np.arctan(max(PPop.arange) / TL.dist[sInd]))
    #                 .to("arcsec")
    #                 .value,
    #             )
    #             * u.arcsec
    #         )
    #         dMag = np.random.uniform(PPro.FAdMag0(WA), TL.saturation_dMag)
    #         self.lastDetected[sInd, 0] = np.append(self.lastDetected[sInd, 0], True)
    #         self.lastDetected[sInd, 1] = np.append(
    #             self.lastDetected[sInd, 1], ZL.fEZ0.to("1/arcsec2").value
    #         )
    #         self.lastDetected[sInd, 2] = np.append(self.lastDetected[sInd, 2], dMag)
    #         self.lastDetected[sInd, 3] = np.append(
    #             self.lastDetected[sInd, 3], WA.to("arcsec").value
    #         )
    #         sminFA = np.tan(WA) * TL.dist[sInd].to("AU")
    #         smin = np.minimum(smin, sminFA) if smin is not None else sminFA
    #         log_FA = "   - False Alarm (WA=%s, dMag=%s)" % (
    #             np.round(WA, 3),
    #             round(dMag, 1),
    #         )
    #         self.logger.info(log_FA)
    #         self.vprint(log_FA)

    #     # Schedule Target Revisit
    #     self.scheduleRevisit(sInd, smin, det, pInds)

    #     self.plot_obs(systemParams, mode, sInd, pInds, SNR, detected)

    #     return detected.astype(int), fZ, systemParams, SNR, FA

    def plot_obs(self, systemParams, mode, sInd, pInds, SNR, detected):
        """
        Plots the planet location in separation-dMag space over the completeness PDF.
        Useful for checking scaling and visualizing why a target observation failed. The
        plots are saved in the according to the simulation's seed in the plot_obs folder.
        Args:
            SS (SurveySimulation module):
                SurveySimulation class object
            systemParams (dict):
                Dictionary of time-dependant planet properties averaged over the
                duration of the integration
            mode (dict):
                Selected observing mode for detection
            sInd (integer):
                Integer index of the star of interest
            pInds (list of integers):
                Index values of the planets of interest
            SNR (float ndarray):
                Detection signal-to-noise ratio of the observable planets
            detected (integer ndarray):
                Detection status for each planet orbiting the observed target star:
                1 is detection, 0 missed detection, -1 below IWA, and -2 beyond OWA
        """
        SS = self
        try:
            SS.counter += 1
        except:
            SS.counter = 0
        TL = SS.TargetList
        Comp = SS.Completeness
        SU = SS.SimulatedUniverse
        dMag = systemParams["dMag"]
        WA = systemParams["WA"]
        dMag_range = np.linspace(16, 35, 5)
        # Making custom colormap for unnormalized values
        dMag_norm = mpl.colors.Normalize(vmin=dMag_range[0], vmax=dMag_range[-1])
        IWA = mode["IWA"]
        OWA = mode["OWA"]
        FR = 10 ** (dMag / (-2.5))

        x_Hrange = Comp.xnew
        y_Hrange = Comp.ynew
        H = Comp.Cpdf
        distance = TL.dist[sInd]
        # comp0 = TL.comp0[sInd]
        intCutoff_comp = TL.intCutoff_comp[sInd]
        saturation_comp = TL.saturation_comp[sInd]
        L = TL.L[sInd]

        smin = np.tan(IWA.to(u.rad)) * distance.to(u.AU) / np.sqrt(L)
        smax = np.tan(OWA.to(u.rad)) * distance.to(u.AU) / np.sqrt(L)
        WA = WA / np.sqrt(L)
        dMag -= 2.5 * np.log10(L)

        dMagint = TL.int_dMag[sInd]
        scaled_dMagint = dMagint - 2.5 * np.log10(L)
        int_WA = TL.int_WA[sInd]
        scaled_WAint = int_WA / np.sqrt(L)
        s_int = np.tan(scaled_WAint.to(u.rad)) * distance.to(u.AU)
        # x_Hrange = x_Hrange * np.sqrt(L)
        # y_Hrange = y_Hrange + np.log10(L)

        my_cmap = plt.get_cmap("viridis")
        edge_cmap = plt.get_cmap("plasma")
        fig, ax = plt.subplots(figsize=[9, 9])
        extent = [x_Hrange[0], x_Hrange[-1], y_Hrange[0], y_Hrange[-1]]
        levels = np.logspace(-6, -1, num=30)
        H_scaled = H / 10000
        ax.contourf(
            H_scaled,
            levels=levels,
            cmap=my_cmap,
            origin="lower",
            extent=extent,
            norm=mpl.colors.LogNorm(),
        )
        # ax.plot(wa_range, dmaglims, c='r') # Contrast curve line
        FR_norm = mpl.colors.LogNorm()
        sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=FR_norm)
        sm._A = []
        sm.set_array(np.logspace(-6, -1))
        fig.subplots_adjust(left=0.15, right=0.85)
        cbar_ax = fig.add_axes([0.865, 0.125, 0.02, 0.75])
        cbar = fig.colorbar(sm, cax=cbar_ax, label=r"Normalized Density")
        # ax.plot(wa_range, dmaglims, c='r')
        # populate detection status array
        # 1:detected, 0:missed, -1:below IWA, -2:beyond OWA
        det_dict = {1: "detected", 0: "Missed", -1: "below_IWA", -2: "beyond_OWA"}
        WA = WA.flatten()
        det_str = ""
        ax.scatter(
            s_int.to(u.AU).value,
            scaled_dMagint,
            color=edge_cmap(0),
            s=50,
            label="Value used to calculate integration time",
        )
        for i, pInd in enumerate(pInds):
            # color = my_cmap(dMag_norm(dMag[i]))
            s_i = np.tan(WA[i].to(u.rad)) * distance.to(u.AU)
            s_nom = np.tan(SU.WA[pInd].to(u.rad)) * distance.to(u.AU)
            detection_status = det_dict[detected[i]]
            det_str += str(i) + "_" + detection_status
            color = edge_cmap((i + 1) / (len(pInds) + 1))
            ax.scatter(
                s_i.to(u.AU).value,
                dMag[i],
                s=100,
                label=f"Planet: {pInd},\
                       SNR: {SNR[i]:.2f}",
                color=color,
            )
            # ax.axvline(s_nom.to(u.AU).value, color='k', label=f'Nominal s for planet {pInd}')
        # ax.set_title(
        #     f"comp0: {comp0:.2f}, intCutoff_comp: {intCutoff_comp:.2f},\
        #              saturation_comp: {saturation_comp:.2f}"
        # )
        ax.set_xlim(0, 3)
        ax.set_ylim(dMag_range[0], dMag_range[-1])
        ax.set_xlabel("s (AU)")
        ax.set_ylabel("dMag")
        ax.axvline(x=smin.to(u.AU).value, color="k", label="Min s (IWA)")
        ax.axvline(x=smax.to(u.AU).value, color="k", label="Max s (OWA)")
        ax.axhline(
            y=TL.saturation_dMag[sInd] - 2.5 * np.log10(L),
            color=my_cmap(0),
            label="saturation_dMag",
        )
        ax.axhline(
            y=TL.intCutoff_dMag[sInd] - 2.5 * np.log10(L),
            color=my_cmap(0.5),
            label="intCutoff_dMag",
        )
        ax.axhline(
            y=TL.int_dMag[sInd] - 2.5 * np.log10(L), color=my_cmap(1), label="dMagint"
        )
        # pos = ax.get_position()
        # ax.set_position([pos.x0, pos.y0, pos.width*0.2, pos.height])
        # ax.legend(loc='center right', bbox_to_anchor=(-.25, 0.5))
        ax.legend()
        folder = Path(f"plot_obs/{SS.seed:.0f}/")
        folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(folder, f"sInd_{sInd}_obs_{SS.counter}_status_{det_str}.png"))
        # breakpoint()
        plt.close()
