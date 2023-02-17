# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.PostProcessing import PostProcessing
import numpy as np


class PostProcessing2(PostProcessing):
    """PostProcessing2 class

    Updated PostProcessing det_occur function that utilized BackgroundSource
    module GalaxiesFaintStars to calculate FA probability.

    """

    def __init__(self, **specs):
        """
        Constructor for class PostProcessing2
        """
        PostProcessing.__init__(self, **specs)

    def det_occur(self, SNR, mode, TL, sInd, intTime):
        """Determines if a detection has occurred and returns booleans

        This method returns two booleans where True gives the case.

        Args:
            SNR (float ndarray):
                signal-to-noise ratio of the planets around the selected target
            mode (dict):
                Selected observing mode
            TL (TargetList module):
                TargetList class object
            sInd (integer):
                Index of the star being observed
            intTime (astropy Quantity):
                Selected star integration time for detection

        Returns:
            tuple:
            FA (boolean):
                False alarm (false positive) boolean.
            MD (boolean ndarray):
                Missed detection (false negative) boolean with the size of
                number of planets around the target.

        Note:
            This implementation assumes the dark hole is set by intCutoff_dMag.
            Alternatively, the true integration depth could be calculated from the
            integration time.
        """

        # get background source false alarm rate
        BS = self.BackgroundSources
        intDepth = np.array([TL.intCutoff_dMag + TL.Vmag[sInd]])
        bs_density = BS.dNbackground(TL.coords[[sInd]], intDepth)
        OWA_solidangle = mode["OWA"] ** 2
        FABP = (
            (bs_density * OWA_solidangle).decompose().value
        )  # false positive rate due to background sources

        # initialize
        FA = False
        MD = np.array([False] * len(SNR))

        # 1/ For the whole system: is there a False Alarm (false positive)?
        p1 = np.random.rand()
        p2 = np.random.rand()
        if (p1 <= self.FAP) or (p2 <= FABP[0]):
            FA = True

        # 2/ For each planet: is there a Missed Detection (false negative)?
        SNRmin = mode["SNR"]
        MD[SNR < SNRmin] = True

        return FA, MD
