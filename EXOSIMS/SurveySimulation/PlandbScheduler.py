from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
from EXOSIMS.util.get_dirs import get_cache_dir
from EXOSIMS.util.utils import genHexStr
from pathlib import Path
import astropy.units as u
import numpy as np
import itertools
import pickle


class PlandbScheduler(SurveySimulation):
    def __init__(self, cachedir=None, **specs):
        SurveySimulation.__init__(self, **specs)

        # start the outspec
        self._outspec = {}

        SU = self.SimulatedUniverse
        TL = self.TargetList
        OS = self.OpticalSystem
        PPop = self.PlanetPopulation

        detmode = list(filter(lambda mode: mode["detectionMode"], OS.observingModes))[0]
        mode = detmode
        self.default_mode = mode

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec["cachedir"] = self.cachedir
        specs["cachedir"] = self.cachedir

        SU_vals_str = f"{str(SU.interval)} {str(SU.missionStart)} {str(SU.missionEnd)}"

        TL_vals_str = f"{str(TL.populate_target_list)}"

        PPop_vals_str = f"{str(PPop.filename)}"

        vals_hash = genHexStr(SU_vals_str + PPop_vals_str + TL_vals_str)

        filename = (
            f"{PPop.__class__.__name__}_"
            f"{TL.__class__.__name__}_"
            f"{OS.__class__.__name__}_"
            f"vals_{vals_hash}_mode_{mode['hex']}"
        )
        probabilityfilepath = Path(self.cachedir, filename)

        if probabilityfilepath.exists():
            with open(probabilityfilepath, "rb") as f:
                self.Pdet = pickle.load(f)
                self.Pdet = self.Pdet.values
                self.vprint(f"Loaded probability values from {probabilityfilepath}")
        else:
            self.Pdet = SU.TimeProbability()
            with open(probabilityfilepath, "wb") as f:
                pickle.dump(self.Pdet, f)
                self.vprint(f"Probability values stored in {probabilityfilepath}")

        # analytically finding the cutoff value for Pdet
        self.PdetCutoff = 0.00

        # initialize the length of selected targets for using in choose_next_target
        self.sIndList = 0

    def next_target(self, old_sInd, mode):
        TL = self.TargetList
        TK = self.TimeKeeping
        Obs = self.Observatory
        OS = self.OpticalSystem

        # create DRM
        DRM = {}

        # allocate settling time + overhead time
        tmpCurrentTimeAbs = (
            TK.currentTimeAbs.copy() + Obs.settlingTime + mode["syst"]["ohTime"]
        )
        tmpCurrentTimeNorm = (
            TK.currentTimeNorm.copy() + Obs.settlingTime + mode["syst"]["ohTime"]
        )

        # creating koMap
        koMap = self.koMaps[mode["syst"]["name"]]

        # initialize arrays
        slewTimes = np.zeros(TL.nStars) * u.d
        dV = np.zeros(TL.nStars) * u.m / u.s
        intTimes = np.zeros(TL.nStars) * u.d
        obsTimes = np.zeros([2, TL.nStars]) * u.d
        sInds = np.arange(TL.nStars)

        # 2. find spacecraft orbital START positions (if occulter, positions
        # differ for each star) and filter out unavailable targets
        sd = None
        if OS.haveOcculter:
            sd = Obs.star_angularSep(TL, old_sInd, sInds, tmpCurrentTimeAbs)
            obsTimes = Obs.calculate_observableTimes(
                TL, sInds, tmpCurrentTimeAbs, self.koMaps, self.koTimes, mode
            )
            slewTimes = Obs.calculate_slewTimes(
                TL, old_sInd, sInds, sd, obsTimes, tmpCurrentTimeAbs
            )

        # 2.1 filter out totTimes > integration cutoff
        if len(sInds.tolist()) > 0:
            sInds = np.intersect1d(self.intTimeFilterInds, sInds)

        # start times
        startTimes = tmpCurrentTimeAbs.copy() + slewTimes
        startTimesNorm = tmpCurrentTimeNorm.copy() + slewTimes
        startTime = np.round(startTimesNorm.value).astype(int)

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
        except:  # noqa: E722 If there are no target stars to observe
            sInds = np.asarray([], dtype=int)

        # 3. filter out all previously (more-)visited targets, unless in
        if len(sInds.tolist()) > 0:
            sInds = self.revisitFilter(sInds, tmpCurrentTimeNorm)

        # 4.1 calculate integration times for ALL preselected targets
        (
            maxIntTimeOBendTime,
            maxIntTimeExoplanetObsTime,
            maxIntTimeMissionLife,
        ) = TK.get_ObsDetectionMaxIntTime(Obs, mode)
        maxIntTime = min(
            maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife
        )  # Maximum intTime allowed

        if len(sInds.tolist()) > 0:
            if OS.haveOcculter and (old_sInd is not None):
                (
                    sInds,
                    slewTimes[sInds],
                    intTimes[sInds],
                    dV[sInds],
                ) = self.refineOcculterSlews(
                    old_sInd, sInds, slewTimes, obsTimes, sd, mode
                )
                endTimes = tmpCurrentTimeAbs.copy() + intTimes + slewTimes
            else:
                intTimes[sInds] = self.calc_targ_intTime(sInds, startTimes[sInds], mode)
                sInds = sInds[
                    np.where(intTimes[sInds] <= maxIntTime)
                ]  # Filters targets exceeding end of OB
                endTimes = tmpCurrentTimeAbs.copy() + intTimes
                endTimesNorm = tmpCurrentTimeNorm.copy() + intTimes
                endTime = np.round(endTimesNorm.value).astype(int)

                if maxIntTime.value <= 0:
                    sInds = np.asarray([], dtype=int)

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

        # 6. choose best target from remaining
        if len(sInds.tolist()) > 0:
            sInd, waitTime = self.choose_next_target(
                old_sInd, sInds, slewTimes, intTimes[sInds]
            )
            # populate DRM with pdet values of sind from starttimes to endtimes
            DRM["p_values"] = self.Pdet[sInd, startTime[sInd] : endTime[sInd] + 1]

            # Should Choose Next Target decide there are no stars it wishes to
            # observe at this time.
            if (sInd is None) and (waitTime is not None):
                self.vprint(
                    (
                        "There are no stars Choose Next Target would like to Observe. "
                        "Waiting {}"
                    ).format(waitTime)
                )
                return DRM, None, None, waitTime
            elif (sInd is None) and (waitTime is not None):
                self.vprint(
                    (
                        "There are no stars Choose Next Target would like to Observe "
                        "and waitTime is None"
                    )
                )
                return DRM, None, None, waitTime

            # store selected star integration time
            intTime = intTimes[sInd]

        # if no observable target, advanceTime to next Observable Target
        else:
            self.vprint(
                "No Observable Targets at currentTimeNorm= "
                + str(TK.currentTimeNorm.copy())
            )
            return DRM, None, None, None
            # update visited list for selected star
        self.starVisits[sInd] += 1
        # store normalized start time for future completeness update
        self.lastObsTimes[sInd] = startTimesNorm[sInd]

        # populate DRM with occulter related values
        if OS.haveOcculter:
            DRM = Obs.log_occulterResults(
                DRM, slewTimes[sInd], sInd, sd[sInd], dV[sInd]
            )
            return DRM, sInd, intTime, slewTimes[sInd]

        return DRM, sInd, intTime, waitTime

    def choose_next_target(self, old_sInd, sInds, slewTimes, intTimes):
        SU = self.SimulatedUniverse
        TL = self.TargetList
        TK = self.TimeKeeping
        OS = self.OpticalSystem
        Obs = self.Observatory
        allModes = OS.observingModes

        Pdet = self.Pdet[sInds]

        if self.sIndList == 0:
            # cast sInds to array
            sInds = np.array(sInds, ndmin=1, copy=False)

            mode = self.default_mode

            intTimes = intTimes + Obs.settlingTime + mode["syst"]["ohTime"]

            # find integration time of sInd with maximum Pdet at the current time
            startTime = (
                TK.currentTimeNorm.copy() + Obs.settlingTime + mode["syst"]["ohTime"]
            )
            startTime = np.round(startTime.value).astype(int)

            Pdetmax = np.argmax(Pdet[:, startTime])

            intTime = intTimes[Pdetmax]

            # find the end time of the observation of the star with maximum Pdet
            endTime = (
                TK.currentTimeNorm.copy()
                + intTime
                + Obs.settlingTime
                + mode["syst"]["ohTime"]
            )
            endTime = np.round(endTime.value).astype(int)

            # find the indices of the stars that exhibits above cutoff Pdet
            # during the observation of the star with maximum Pdet
            tmpsInds = np.where(
                (Pdet[:, startTime : endTime + 1] >= self.PdetCutoff).all(axis=1)
            )[0]

            if len(tmpsInds) > 6:
                meanPdet = np.mean(Pdet[tmpsInds, startTime : endTime + 1], axis=1)
                sIndsgroup = tmpsInds[np.argsort(meanPdet)[::-1][:6]]
            else:
                sIndsgroup = tmpsInds

            # combination of indices in sIndsgroup
            combinations = [
                np.array(c)
                for r in range(1, len(sIndsgroup) + 1)
                for c in itertools.combinations(sIndsgroup, r)
            ]

            # sum of integration times of the combinations
            combintTimes = [np.sum(intTimes[c]) for c in combinations]
            combintTimes = np.array([time.value for time in combintTimes])

            # checking if any values in combintTimes are less than intTime
            combval = np.where(combintTimes <= intTime.value)[0]

            if len(combval) != 0:
                # storing the combinations with integration times less than intTime
                possiblecombs = [combinations[c] for c in combval]

                # selecting the combinations with maximum sInds
                sIndsizes = [len(c) for c in possiblecombs]

                maxsInds = max(sIndsizes)

                combmaxsInds = np.where(np.array(sIndsizes) == maxsInds)[0]

                # calc sum of Pdet for each combination selected
                sumPdet = [meanPdet[possiblecombs[ind]].sum() for ind in combmaxsInds]

                # selecting the combination with maximum sum of Pdet
                maxsumPdet = np.argmax(sumPdet)

                # selecting the sInds of the combination with maximum sum of Pdet
                chosensInds = possiblecombs[combmaxsInds[maxsumPdet]]

                # sorting the sInds based on Pdet value
                self.chosensInds = chosensInds[np.argsort(meanPdet[chosensInds])[::-1]]

                # number of targets selected
                self.sIndList = len(self.chosensInds)

                sInd = self.chosensInds[0]
                self.chosensInds = np.delete(self.chosensInds, 0)
                self.sIndList -= 1
                return sInd, None

            else:
                sInd = Pdetmax
                self.sIndList = 1
                return sInd, None
                # if the target doesn't follow cutoff, select next target

        else:
            sInd = self.chosensInds[0]
            self.chosensInds = np.delete(self.chosensInds, 0)
            self.sIndList -= 1

        return sInd, None
