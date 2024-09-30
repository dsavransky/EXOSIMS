from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_dirs import get_cache_dir
import numpy as np
import astropy.units as u
from astropy.time import Time
import os
import csv


class TimeKeeping(object):
    """:ref:`TimeKeeping` Prototype

    This class keeps track of the current mission elapsed time
    for exoplanet mission simulation.  It is initialized with a
    mission duration, and throughout the simulation, it allocates
    temporal intervals for observations.  Eventually, all available
    time has been allocated, and the mission is over.
    Time is allocated in contiguous windows of size "duration".  If a
    requested interval does not fit in the current window, we move to
    the next one.

    Args:
        missionStart (float):
            Mission start date in MJD. Defaults to 60634 (11-20-2024)
        missionLife (float):
            Mission duration (in years). Defaults to 0.1
        missionPortion (float):
            Fraction of mission devoted to exoplanet imaging science.
            Must be between 0 and 1. Defaults to 1
        OBduration (float):
            Observing block length (in days). If infinite, do not
            define observing blocks. Defaults to np.inf
        missionSchedule (str, optional):
            Full path to mission schedule file stored on disk.
            Defaults None.
        cachedir (str, optional):
            Full path to cachedir.
            If None (default) use default (see :ref:`EXOSIMSCACHE`)
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        _outspec (dict):
            :ref:`sec:outspec`
        cachedir (str):
            Path to the EXOSIMS cache directory (see :ref:`EXOSIMSCACHE`)
        currentTimeAbs (astropy.time.core.Time):
            Current absolute mission time in MJD
        currentTimeNorm (astropy.units.quantity.Quantity):
            Current mission time minus mission start time.
        exoplanetObsTime (astropy.units.quantity.Quantity):
            How much time has been used so far on exoplanet science.
        missionFinishAbs (astropy.time.core.Time):
            Mission end time in MJD
        missionLife (astropy.units.quantity.Quantity):
            Total mission duration
        missionPortion (float):
            Fraction of mission devoted to exoplanet science
        missionStart (astropy.time.core.Time):
            Start time of mission in MJD
        OBduration (astropy.units.quantity.Quantity):
            Observing block length
        OBendTimes (astropy.units.quantity.Quantity):
            Array containing the normalized end times of each observing block
            throughout the mission
        OBnumber (int):
            Index of the current observing block (OB). Each
            observing block has a duration, a start time, an end time, and can
            host one or multiple observations
        OBstartTimes (astropy.units.quantity.Quantity):
            Array containing the normalized start times of each observing block
            throughout the mission
    """

    _modtype = "TimeKeeping"

    def __init__(
        self,
        missionStart=60634,
        missionLife=0.1,
        missionPortion=1,
        OBduration=np.inf,
        missionSchedule=None,
        cachedir=None,
        **specs
    ):

        # start the outspec
        self._outspec = {}

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec["cachedir"] = self.cachedir
        specs["cachedir"] = self.cachedir

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get("verbose", True))

        # illegal value checks
        assert missionLife >= 0, "Need missionLife >= 0, got %f" % missionLife
        # arithmetic on missionPortion fails if it is outside the legal range
        assert missionPortion > 0 and missionPortion <= 1, (
            "Require missionPortion in the interval [0,1], got %f" % missionPortion
        )
        # OBduration must be positive nonzero
        assert OBduration * u.d > 0 * u.d, (
            "Required OBduration positive nonzero, got %f" % OBduration
        )

        # set up state variables
        # tai scale specified because the default, utc, requires accounting for leap
        # seconds, causing warnings from astropy.time when time-deltas are added
        # Absolute mission start time:
        self.missionStart = Time(float(missionStart), format="mjd", scale="tai")
        # Fraction of missionLife allowed for exoplanet science
        self.missionPortion = float(missionPortion)
        # Total mission duration
        self.missionLife = float(missionLife) * u.year

        # populate outspec
        for att in self.__dict__:
            if att not in ["vprint", "_outspec"]:
                dat = self.__dict__[att]
                self._outspec[att] = (
                    dat.value if isinstance(dat, (u.Quantity, Time)) else dat
                )

        # Absolute mission end time
        self.missionFinishAbs = self.missionStart + self.missionLife.to("day")

        # initialize values updated by various class methods
        # the current mission elapsed time (0 at mission start)
        self.currentTimeNorm = 0.0 * u.day
        # current absolute mission time (equals missionStart at mission start)
        self.currentTimeAbs = self.missionStart

        # initialize observing block times arrays.
        # An Observing Block is a segment of time over which observations may take place
        self.init_OB(str(missionSchedule), OBduration * u.d)
        self._outspec["missionSchedule"] = missionSchedule
        self._outspec["OBduration"] = OBduration

        # initialize time spend using instrument
        self.exoplanetObsTime = 0 * u.day

    def __str__(self):
        r"""String representation of the TimeKeeping object.

        When the command 'print' is used on the TimeKeeping object, this
        method prints the values contained in the object."""

        for att in self.__dict__:
            print("%s: %r" % (att, getattr(self, att)))

        return (
            "TimeKeeping instance at %.6f days" % self.currentTimeNorm.to("day").value
        )

    def init_OB(self, missionSchedule, OBduration):
        """
        Initializes mission Observing Blocks from file or missionDuration, missionLife,
        and missionPortion. Updates attributes OBstartTimes, OBendTimes, and OBnumber

        Args:
            missionSchedule (str):
                A string containing the missionSchedule file (or "None").

            OBduration (~astropy.units.Quantity):
                Observing block length

        Returns:
            None

        """
        if not missionSchedule == "None":  # If the missionSchedule is specified
            tmpOBtimes = list()
            schedulefname = str(
                os.path.dirname(__file__) + "/../Scripts/" + missionSchedule
            )  # .csv file in EXOSIMS/Scripts folder
            if not os.path.isfile(schedulefname):
                # Check if scriptNames in ScriptsPath
                ScriptsPath = str(os.path.dirname(__file__) + "/../../../Scripts/")
                makeSimilar_TemplateFolder = ""
                dirsFolderDown = [
                    x[0].split("/")[-1] for x in os.walk(ScriptsPath)
                ]  # Get all directories in ScriptsPath
                for tmpFolder in dirsFolderDown:
                    if (
                        os.path.isfile(ScriptsPath + tmpFolder + "/" + missionSchedule)
                        and not tmpFolder == ""
                    ):  # We found the Scripts folder containing scriptfile
                        makeSimilar_TemplateFolder = (
                            tmpFolder + "/"
                        )  # We found the file!!!
                        break
                schedulefname = str(
                    ScriptsPath + makeSimilar_TemplateFolder + missionSchedule
                )  # .csv file in EXOSIMS/Scripts folder

            if os.path.isfile(
                schedulefname
            ):  # Check if a mission schedule is manually specified
                self.vprint("Loading Manual Schedule from %s" % missionSchedule)
                with open(schedulefname, "r") as f:  # load csv file
                    lines = csv.reader(f, delimiter=",")
                    self.vprint("The manual Schedule is:")
                    for line in lines:
                        tmpOBtimes.append(line)
                        self.vprint(line)
                self.OBstartTimes = (
                    np.asarray([float(item[0]) for item in tmpOBtimes]) * u.d
                )
                self.OBendTimes = (
                    np.asarray([float(item[1]) for item in tmpOBtimes]) * u.d
                )
        # Automatically construct OB from OBduration, missionLife, and missionPortion
        else:
            if OBduration == np.inf * u.d:  # There is 1 OB spanning the mission
                self.OBstartTimes = np.asarray([0]) * u.d
                self.OBendTimes = np.asarray([self.missionLife.to("day").value]) * u.d
            else:  # OB
                startToStart = OBduration / self.missionPortion
                numBlocks = np.ceil(
                    self.missionLife.to("day") / startToStart
                )  # This is the number of Observing Blocks
                self.OBstartTimes = np.arange(numBlocks) * startToStart
                self.OBendTimes = self.OBstartTimes + OBduration
                if self.OBendTimes[-1] > self.missionLife.to(
                    "day"
                ):  # If the end of the last observing block exceeds the end of mission
                    self.OBendTimes[-1] = self.missionLife.to(
                        "day"
                    ).copy()  # Set end of last OB to end of mission
        self.OBduration = OBduration
        self.OBnumber = 0
        self.vprint("OBendTimes is: " + str(self.OBendTimes))

    def mission_is_over(self, OS, Obs, mode):
        r"""Are the mission time, or any other mission resources, exhausted?

        Args:
            OS (:ref:`OpticalSystem`):
                Optical System object
            Obs (:ref:`Observatory`):
                Observatory object
            mode (dict):
                Selected observing mode for detection (uses only overhead time)

        Returns:
            bool:
                True if the mission time or fuel are used up, else False.
        """

        # let's be optimistic and assume we still have time
        is_over = False

        # if we've exceeded total mission time (or overhead on the next observation
        # will make us exceed total time, we're done)
        if self.currentTimeNorm + Obs.settlingTime + mode["syst"][
            "ohTime"
        ] >= self.missionLife.to("day"):
            self.vprint(
                "missionLife would be exceeded at %s"
                % self.currentTimeNorm.to("day").round(2)
            )
            is_over = True

        # if we've used up our time allocation (or overhead on the next observation
        # will make us exceed it, we're done)
        if (
            self.exoplanetObsTime.to("day") + Obs.settlingTime + mode["syst"]["ohTime"]
            >= self.missionLife.to("day") * self.missionPortion
        ):
            self.vprint(
                (
                    "exoplanetObstime ({:.2f}) would exceed "
                    "(missionPortion*missionLife) = {:.2f}) at currentTimeNorm = {}"
                ).format(
                    self.exoplanetObsTime,
                    self.missionPortion * self.missionLife.to("day"),
                    self.currentTimeNorm.to("day").round(2),
                )
            )
            is_over = True

        # if overheads will put us past the end of the final observing block, we're done
        if (
            self.currentTimeNorm + Obs.settlingTime + mode["syst"]["ohTime"]
            >= self.OBendTimes[-1]
        ):
            self.vprint(
                (
                    "Last Observing Block (# {}, end time: {:.2f}) would be "
                    "exceeded at currentTimeNorm {}"
                ).format(
                    self.OBnumber,
                    self.OBendTimes[-1],
                    self.currentTimeNorm.to("day").round(2),
                )
            )
            is_over = True

        # and now, all the fuel stuff
        if OS.haveOcculter:
            # handle case of separate fuel tanks for slew and sk:
            if Obs.twotanks:
                if Obs.skMass <= 0 * u.kg:
                    self.vprint(
                        "Stationkeeping fuel exhausted at currentTimeNorm %s"
                        % (self.currentTimeNorm.to("day").round(2))
                    )
                    # see if we can refuel
                    if not (Obs.refuel_tank(self, tank="sk")):
                        is_over = True

                if Obs.slewMass <= 0 * u.kg:
                    self.vprint(
                        "Slew fuel exhausted at currentTimeNorm %s"
                        % (self.currentTimeNorm.to("day").round(2))
                    )
                    # see if we can refuel
                    if not (Obs.refuel_tank(self, tank="slew")):
                        is_over = True

            # now consider case of only one tank
            else:
                if Obs.scMass <= Obs.dryMass:
                    self.vprint(
                        "Fuel exhausted at currentTimeNorm %s"
                        % (self.currentTimeNorm.to("day").round(2))
                    )
                    # see if we can refuel
                    if not (Obs.refuel_tank(self)):
                        is_over = True

        return is_over

    def allocate_time(self, dt, addExoplanetObsTime=True):
        r"""Allocate a temporal block of width dt

        Advance the mission time by dt units. Updates attributes currentTimeNorm and
        currentTimeAbs

        Args:
            dt (~astropy.units.Quantity):
                Temporal block allocated in units of days
            addExoplanetObsTime (bool):
                Indicates the allocated time is for the primary instrument (True)
                or some other instrument (False)
                By default this function assumes all allocated time is attributed to
                the primary instrument (is True)

        Returns:
            bool:
                a flag indicating the time allocation was successful or not successful
        """

        # Check dt validity
        if dt.value <= 0 or dt.value == np.inf:
            self.vprint("dt must be positive and nonzero (got {})".format(dt))
            return False  # The temporal block to allocate is not positive nonzero

        # Check dt exceeds mission life
        if self.currentTimeNorm + dt > self.missionLife.to("day"):
            self.vprint(
                (
                    "Allocating dt = {} at curremtTimeNorm = {} would exceed "
                    "missionLife = {}"
                ).format(
                    dt,
                    self.currentTimeNorm,
                    self.missionLife.to("day"),
                )
            )
            return False  # The time to allocate would exceed the missionLife

        # Check dt exceeds current OB
        if self.currentTimeNorm + dt > self.OBendTimes[self.OBnumber]:
            self.vprint(
                (
                    "Allocating dt = {} at currentTimeNorm = {} would exceed the end "
                    "of observing block #{} with end time {}"
                ).format(
                    dt,
                    self.currentTimeNorm,
                    self.OBnumber,
                    self.OBendTimes[self.OBnumber],
                )
            )
            return False

        # Check exceeds allowed instrument Time
        if addExoplanetObsTime:
            if (
                self.exoplanetObsTime + dt
                > self.missionLife.to("day") * self.missionPortion
            ):
                self.vprint(
                    (
                        "Allocating dt = {} with current exoplanetObsTime = {} would "
                        "exceed (missionPortion*missionLife) = {:.2f}"
                    ).format(
                        dt,
                        self.exoplanetObsTime,
                        self.missionLife.to("day") * self.missionPortion,
                    )
                )
                return False

            self.currentTimeAbs += dt
            self.currentTimeNorm += dt
            self.exoplanetObsTime += dt
            return True
        else:  # Time will not be counted against exoplanetObstime
            self.currentTimeAbs += dt
            self.currentTimeNorm += dt
            return True

    def advancetToStartOfNextOB(self):
        """Advances to Start of Next Observation Block
        This method is called in the allocate_time() method of the TimeKeeping
        class object, when the allocated time requires moving outside of the current OB.
        If no OB duration was specified, a new Observing Block is created for
        each observation in the SurveySimulation module. Updates attributes OBnumber,
        currentTimeNorm and currentTimeAbs.

        """
        self.OBnumber += 1  # increase the observation block number
        self.currentTimeNorm = self.OBstartTimes[
            self.OBnumber
        ]  # update currentTimeNorm
        self.currentTimeAbs = (
            self.OBstartTimes[self.OBnumber] + self.missionStart
        )  # update currentTimeAbs

        # begin Survey, and loop until mission is finished
        log_begin = "OB%s:" % (
            self.OBnumber
        )  # prints because this is the beginning of the nesxt observation block
        self.vprint(log_begin)
        self.vprint(
            "Advanced currentTimeNorm to beginning of next OB %.2fd"
            % (self.currentTimeNorm.to("day").value)
        )

    def advanceToAbsTime(self, tAbs, addExoplanetObsTime=True):
        """Advances the current mission time to tAbs.
        Updates attributes currentTimeNorma dn currentTimeAbs

        Args:
            tAbs (~astropy.time.Time):
                The absolute mission time to advance currentTimeAbs to.
                MUST HAVE scale='tai'
            addExoplanetObsTime (bool):
                A flag indicating whether to add advanced time to exoplanetObsTime or
                not

        Returns:
            bool:
                A bool indicating whether the operation was successful or not
        """

        # Checks on tAbs validity
        if tAbs <= self.currentTimeAbs:
            self.vprint(
                "The time to advance to "
                + str(tAbs)
                + " is not after "
                + str(self.currentTimeAbs)
            )
            return False

        # Use 2 and Use 4
        if tAbs >= self.missionFinishAbs:  #
            tmpcurrentTimeNorm = self.currentTimeNorm.copy()
            t_added = (tAbs - self.currentTimeAbs).value * u.d
            self.currentTimeNorm = (tAbs - self.missionStart).value * u.d
            self.currentTimeAbs = tAbs
            if addExoplanetObsTime:  # Count time towards exoplanetObs Time
                if (self.exoplanetObsTime + t_added) > (
                    self.missionLife.to("day") * self.missionPortion
                ):
                    self.vprint(
                        (
                            "Adding {} to current exoplanetObsTime ({:0.2f}) would "
                            "exceed (missionLife*missionPortion) = {:0.2f}"
                        ).format(
                            t_added.to("day"),
                            self.exoplanetObsTime.to("day"),
                            self.missionLife.to("day") * self.missionPortion,
                        )
                    )
                    self.exoplanetObsTime = (
                        self.missionLife.to("day") * self.missionPortion
                    )
                    return False
                self.exoplanetObsTime += (
                    self.missionLife.to("day") - tmpcurrentTimeNorm
                )  # Advances exoplanet time to end of mission time
            else:
                self.exoplanetObsTime += 0 * u.d
            return True

        # Use 1 and Use 3
        if (
            tAbs <= self.OBendTimes[self.OBnumber] + self.missionStart
        ):  # The time to advance to does not leave the current OB
            t_added = (tAbs - self.currentTimeAbs).value * u.d
            self.currentTimeNorm = (tAbs - self.missionStart).to("day")
            self.currentTimeAbs = tAbs
            if addExoplanetObsTime:  # count time towards exoplanet Obs Time
                if (self.exoplanetObsTime + t_added) > (
                    self.missionLife.to("day") * self.missionPortion
                ):
                    self.vprint(
                        (
                            "Adding {} to current exoplanetObsTime ({:0.2f}) would "
                            "exceed (missionLife*missionPortion) = {:0.2f}"
                        ).format(
                            t_added.to("day"),
                            self.exoplanetObsTime.to("day"),
                            self.missionLife.to("day") * self.missionPortion,
                        )
                    )
                    self.exoplanetObsTime = (
                        self.missionLife.to("day") * self.missionPortion
                    )
                    return False
                else:
                    self.exoplanetObsTime += t_added
                    return True
            else:  # addExoplanetObsTime is False
                self.exoplanetObsTime += 0 * u.d
            return True

        # Use 5 and 7 #extended to accomodate any current and future time between OBs
        tNorm = (tAbs - self.missionStart).value * u.d
        if np.any(
            (tNorm <= self.OBstartTimes[1:]) & (tNorm >= self.OBendTimes[0:-1])
        ):  # The tAbs is between end End of an OB and start of the Next OB
            # Return OBnumber of End Index
            endIndex = np.where(
                (tNorm <= self.OBstartTimes[1:]) & (tNorm >= self.OBendTimes[0:-1])
            )[0][0]
            # self.OBendTimes[endIndex+1] - self.currentTimeNorm
            # Time to be added to exoplanetObsTime from current OB
            t_added = self.OBendTimes[self.OBnumber] - self.currentTimeNorm
            for ind in np.arange(
                self.OBnumber, endIndex
            ):  # ,len(self.OBendTimes)):  # Add time for all additional OB
                t_added += self.OBendTimes[ind] - self.OBstartTimes[ind]
            while self.OBnumber < endIndex + 1:
                self.advancetToStartOfNextOB()
            # self.OBnumber = endIndex + 1  # set OBnumber to correct Observing Block
            # self.currentTimeNorm = self.OBstartTimes[self.OBnumber]
            # Advance Time to start of next OB
            # self.currentTimeAbs = self.OBstartTimes[self.OBnumber] + self.missionStart
            # Advance Time to start of next OB
            if addExoplanetObsTime:  # Count time towards exoplanetObs Time
                if (
                    self.exoplanetObsTime + t_added
                    > self.missionLife.to("day") * self.missionPortion
                ):  # We can CANNOT allocate that time to exoplanetObsTime
                    self.vprint(
                        (
                            "Adding {} to current exoplanetObsTime ({:0.2f}) would "
                            "exceed (missionLife*missionPortion) = {:0.2f}"
                        ).format(
                            t_added.to("day"),
                            self.exoplanetObsTime.to("day"),
                            self.missionLife.to("day") * self.missionPortion,
                        )
                    )
                    # This kind of failure is by design.
                    # It just means the mission has come to an end
                    self.vprint("Advancing to tAbs failed under Use Case 7")
                    self.exoplanetObsTime = (
                        self.missionLife.to("day") * self.missionPortion
                    )
                    return False
                self.exoplanetObsTime += t_added
            else:
                self.exoplanetObsTime += 0 * u.d
            return True

        # Use 6 and 8 #extended to accomodate any current and future time inside OBs
        # The tAbs is between start of a future OB and end of that OB
        if np.any(
            # fmt: off
            (tNorm >= self.OBstartTimes[self.OBnumber:])
            & (tNorm <= self.OBendTimes[self.OBnumber:])
            # fmt: on
        ):
            # fmt: off
            endIndex = np.where(
                (tNorm >= self.OBstartTimes[self.OBnumber:])
                & (tNorm <= self.OBendTimes[self.OBnumber:])
            )[0][
                0
            ]  # Return index of OB that tAbs will be inside of
            # fmt: on
            endIndex += self.OBnumber
            t_added = 0 * u.d
            if addExoplanetObsTime:  # Count time towards exoplanetObs Time
                t_added += (tAbs - self.currentTimeAbs).to("day")
                for i in np.arange(self.OBnumber, endIndex):
                    # accumulate time to subtract (time not counted against Exoplanet
                    # Obs Time)
                    # Subtract the time between these OB from the t_added to
                    # exoplanetObsTime
                    index = self.OBnumber
                    t_added -= self.OBstartTimes[index + 1] - self.OBendTimes[index]
                # Check if exoplanetObsTime would be exceeded
                if (
                    self.exoplanetObsTime + t_added
                    > self.missionLife.to("day") * self.missionPortion
                ):
                    self.vprint(
                        (
                            "Adding {} to current exoplanetObsTime ({:0.2f}) would "
                            "exceed (missionLife*missionPortion) = {:0.2f}"
                        ).format(
                            t_added.to("day"),
                            self.exoplanetObsTime.to("day"),
                            self.missionLife.to("day") * self.missionPortion,
                        )
                    )

                    # This kind of failure is by design.
                    # It just means the mission has come to an end
                    self.vprint("Advancing to tAbs failed under Use Case 8")
                    self.exoplanetObsTime = (
                        self.missionLife.to("day") * self.missionPortion
                    )
                    self.OBnumber = endIndex  # set OBnumber to correct Observing Block
                    self.currentTimeNorm = (tAbs - self.missionStart).to(
                        "day"
                    )  # Advance Time to start of next OB
                    self.currentTimeAbs = tAbs  # Advance Time to start of next OB
                    return False
                else:
                    self.exoplanetObsTime += t_added
            else:  # addExoplanetObsTime is False
                self.exoplanetObsTime += 0 * u.d
            self.OBnumber = endIndex  # set OBnumber to correct Observing Block
            self.currentTimeNorm = (tAbs - self.missionStart).to(
                "day"
            )  # Advance Time to start of next OB
            self.currentTimeAbs = tAbs  # Advance Time to start of next OB
            return True

        # Generic error if there exists some use case that I have not encountered yet.
        assert False, "No Use Case Found in AdvanceToAbsTime"

    def get_ObsDetectionMaxIntTime(
        self, Obs, mode, currentTimeNorm=None, OBnumber=None
    ):
        """Tells you the maximum Detection Observation Integration Time you can pass
        into observation_detection(X,intTime,X)

        Args:
            Obs (:ref:`Observatory`):
                Observatory object
            mode (dict):
                Selected observing mode for detection
            currentTimeNorm (astropy.unit.Quantity, optional):
                Time since mission start
            OBnumber (float, optional):
                Current observing block

        Returns:
            tuple:
                maxIntTimeOBendTime (astropy.units.Quantity):
                    The maximum integration time bounded by Observation Block end Time
                maxIntTimeExoplanetObsTime (astropy.units.Quantity):
                    The maximum integration time bounded by exoplanetObsTime
                maxIntTimeMissionLife (astropy.units.Quantity):
                    The maximum integration time bounded by MissionLife
        """
        if OBnumber is None:
            OBnumber = self.OBnumber
        if currentTimeNorm is None:
            currentTimeNorm = self.currentTimeNorm.copy()

        maxTimeOBendTime = self.OBendTimes[OBnumber] - currentTimeNorm
        maxIntTimeOBendTime = (
            maxTimeOBendTime - Obs.settlingTime - mode["syst"]["ohTime"]
        ) / (1.0 + mode["timeMultiplier"] - 1.0)

        maxTimeExoplanetObsTime = (
            self.missionLife.to("day") * self.missionPortion - self.exoplanetObsTime
        )
        maxIntTimeExoplanetObsTime = (
            maxTimeExoplanetObsTime - Obs.settlingTime - mode["syst"]["ohTime"]
        ) / (1.0 + mode["timeMultiplier"] - 1.0)

        maxTimeMissionLife = self.missionLife.to("day") - currentTimeNorm
        maxIntTimeMissionLife = (
            maxTimeMissionLife - Obs.settlingTime - mode["syst"]["ohTime"]
        ) / (1.0 + mode["timeMultiplier"] - 1.0)

        # Ensure all are positive or zero
        if maxIntTimeOBendTime < 0.0:
            maxIntTimeOBendTime = 0.0 * u.d
        if maxIntTimeExoplanetObsTime < 0.0:
            maxIntTimeExoplanetObsTime = 0.0 * u.d
        if maxIntTimeMissionLife < 0.0:
            maxIntTimeMissionLife = 0.0 * u.d

        return maxIntTimeOBendTime, maxIntTimeExoplanetObsTime, maxIntTimeMissionLife
