from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import numpy as np
import EXOSIMS
import os
from EXOSIMS.util._numpy_compat import copy_if_needed


class randomWalkScheduler2(SurveySimulation):
    """randomWalkScheduler2

    This class implements a random walk scheduler that selects the
    next target at random from the pool of currently available targets.

    This is useful for mapping out the space of possible mission outcomes
    for a fixed population of planets in order to validate other schedulers.

    The random walk will attempt to first choose from occulter targets before
    defaulting back to the general target list.

    Args:
        occHIPs (iterable nx1):
            List of star HIP numbers to initialize occulter target list.
        **specs:
            user specified values

    """

    def __init__(self, occHIPs=[], **specs):
        SurveySimulation.__init__(self, **specs)
        self._outspec["occHIPs"] = occHIPs

        if occHIPs != []:
            occHIPs_path = os.path.join(EXOSIMS.__path__[0], "Scripts", occHIPs)
            assert os.path.isfile(occHIPs_path), "%s is not a file." % occHIPs_path
            HIPsfile = open(occHIPs_path, "r").read()
            self.occHIPs = HIPsfile.split(",")
            if len(self.occHIPs) <= 1:
                self.occHIPs = HIPsfile.split("\n")
        else:
            self.occHIPs = occHIPs

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
        sInds = np.array(sInds, ndmin=1, copy=copy_if_needed)
        occ_sInds = np.where(np.in1d(TL.Name, self.occHIPs))[0]
        n_sInds = np.intersect1d(sInds, occ_sInds)

        # pick one
        if len(n_sInds) == 0:
            sInd = np.random.choice(sInds)
        else:
            sInd = np.random.choice(n_sInds)

        return sInd, slewTimes[sInd]
