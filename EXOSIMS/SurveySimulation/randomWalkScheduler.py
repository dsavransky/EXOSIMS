from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import numpy as np
from EXOSIMS.util._numpy_compat import copy_if_needed


class randomWalkScheduler(SurveySimulation):
    """randomWalkScheduler

    This class implements a random walk scheduler that selects the
    next target at random from the pool of currently available targets.

    This is useful for mapping out the space of possible mission outcomes
    for a fixed population of planets in order to validate other schedulers.

    """

    def choose_next_target(self, old_sInd, sInds, slewTimes, intTimes):
        """Choose next target at random

        Args:
            old_sInd (int):
                Index of the previous target star
            sInds (int array):
                Indices of available targets
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            intTimes (astropy Quantity array):
                Integration times for detection in units of day

        Returns:
            tuple:
                sInd (int):
                    Index of next target star
                waitTime (astropy Quantity):
                    the amount of time to wait (this method returns None)
        """

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)

        # pick one
        sInd = np.random.choice(sInds)

        if slewTimes[sInd] > 0:
            return sInd, slewTimes[sInd]
        else:
            return sInd, None
