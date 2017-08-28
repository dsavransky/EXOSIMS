# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.PostProcessing import PostProcessing
import numpy as np
import astropy.units as u
import scipy.stats as st
import scipy.interpolate
import numbers
from EXOSIMS.util.get_module import get_module

class PostProcessing2(object):
    """PostProcessing2 class

    Updated PostProcessing det_occur function that utilized BackgroundSource
    module GalaxiesFaintStars to calculate FA probability.
    
    """
    def __init__(self, **specs):
        """
        Constructor for class PostProcessing2
        """
        PostProcessing.__init__(self, **specs)

    def det_occur(self, SNR, SNRmin, sInd, coords, intDepths, OWA):
        """Determines if a detection has occurred and returns booleans 
        
        This method returns two booleans where True gives the case.
        
        Args:
            SNR (float ndarray):
                signal-to-noise ratio of the planets around the selected target
            SNRmin (float):
                signal-to-noise ratio threshold for detection
            sInd (int):
                targed star index
            coords (astropy SkyCoord array):
                SkyCoord object containing right ascension, declination, and 
                distance to star of the planets of interest in units of deg, deg and pc
            intDepths (float ndarray):
                Integration depths equal to the limiting planet magnitude 
                (Vmag+dMagLim), i.e. the V magnitude of the dark hole to be 
                produced for each target. Must be of same length as coords.
            OWA (float):
                Outer Working Angle of the observation mode in arcsec
        
        Returns:
            FA (boolean):
                False alarm (false positive) boolean.
            MD (boolean ndarray):
                Missed detection (false negative) boolean with the size of 
                number of planets around the target.
       
        Notes:
            TODO: Add backgroundsources hook
        
        """

        BS = self.BackgroundSources
        bs_density = BS.dNbackground(coords, intDepths).to(1/u.arcsec**2)

        OWA_solidangle = OWA**2

        FABP = bs_density[sInd] * OWA_solidangle # false positive rate due to background sources
        
        # initialize
        FA = False
        MD = np.array([False]*len(SNR))
        
        # 1/ For the whole system: is there a False Alarm (false positive)?
        p1 = np.random.rand()
        p2 = np.random.rand()
        if p1 <= self.FAP  or p2 <= FABP:
            FA = True
        
        # 2/ For each planet: is there a Missed Detection (false negative)?
        MD[SNR < SNRmin] = True
        
        return FA, MD
