from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import numpy as np
from EXOSIMS.util._numpy_compat import copy_if_needed


class cbytScheduler(SurveySimulation):
    """cbytScheduler - Completeness-by-time Scheduler

    This class implements a Scheduler that selects the current highest
    Completeness/Integration Time.

    Args:
        **specs:
            user specified values

    """

    def __init__(self, **specs):

        SurveySimulation.__init__(self, **specs)

    def choose_next_target(self, old_sInd, sInds, slewTimes, intTimes):
        """Choose next target based on truncated depth first search
        of linear cost function.

        Args:
            old_sInd (int):
                Index of the previous target star
            sInds (int numpy.ndarray):
                Indices of available targets
            slewTimes (astropy.units.Quantity numpy.ndarray):
                slew times to all stars (must be indexed by sInds)
            intTimes (astropy.units.Quantity numpy.ndarray):
                Integration times for detection in units of day

        Returns:
            tuple:
                sInd (int):
                    Index of next target star
                waitTime (astropy.units.Quantity or None):
                    the amount of time to wait (this method returns None)

        """

        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)
        # calculate dt since previous observation
        dt = TK.currentTimeNorm + slewTimes[sInds] - self.lastObsTimes[sInds]
        # get dynamic completeness values
        comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], dt)

        # selection metric being used: completeness/integration time
        selMetric = comps / intTimes

        # selecting the target star to observe
        sInd = sInds[selMetric == max(selMetric)][0]

        return sInd, None
