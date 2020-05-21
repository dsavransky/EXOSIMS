# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.get_dirs import get_cache_dir
import numpy as np
import astropy.units as u
import os
import scipy.interpolate
import numbers
from astropy.io import fits
import sys

# Python 3 compatibility:
if sys.version_info[0] > 2:
    basestring = str

class PostProcessing(object):
    """Post Processing class template
    
    This class contains all variables and functions necessary to perform 
    Post Processing Module calculations in exoplanet mission simulation.
    
    Args:
        specs:
            user specified values
            
    Attributes:
        BackgroundSources (BackgroundSources module):
            BackgroundSources class object
        FAP (float):
            False Alarm Probability
        MDP (float):
            Missed Detection Probability
        ppFact (float, callable):
            Post-processing contrast factor, between 0 and 1: either a scalar 
            for constant gain, or a two-column array for separation-dependent 
            gain, where the first column contains the angular separation in 
            units of arcsec. May be data or FITS filename.
        FAdMag0 (float, callable):
            Minimum delta magnitude that can be obtained by a false alarm: either a scalar 
            for constant dMag, or a two-column array for separation-dependent 
            dMag, where the first column contains the angular separation in 
            units of arcsec. May be data or FITS filename.
        cachedir (str):
            Path to cache directory
    
    """
    
    _modtype = 'PostProcessing'
    
    def __init__(self, FAP=3e-7, MDP=1e-3, ppFact=1.0, ppFact_char=1.0, FAdMag0=15, cachedir=None, **specs):

        #start the outspec
        self._outspec = {}

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec['cachedir'] = self.cachedir
        specs['cachedir'] = self.cachedir 

        
        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))
        
        self.FAP = float(FAP)       # false alarm probability
        self.MDP = float(MDP)       # missed detection probability
        
        # check for post-processing factor, function of the working angle
        if isinstance(ppFact, basestring):
            pth = os.path.normpath(os.path.expandvars(ppFact))
            assert os.path.isfile(pth), "%s is not a valid file."%pth
            with fits.open(pth) as ff:
                dat = ff[0].data
            assert len(dat.shape) == 2 and 2 in dat.shape, \
                    "Wrong post-processing gain data shape."
            WA, G = (dat[0], dat[1]) if dat.shape[0] == 2 else (dat[:,0], dat[:,1])
            assert np.all(G > 0) and np.all(G <= 1), \
                    "Post-processing gain must be positive and smaller than 1."
            # gain outside of WA values defaults to 1
            Ginterp = scipy.interpolate.interp1d(WA, G, kind='cubic',
                    fill_value=1., bounds_error=False)
            self.ppFact = lambda s: np.array(Ginterp(s.to('arcsec').value), ndmin=1)
        elif isinstance(ppFact, numbers.Number):
            assert ppFact > 0 and ppFact <= 1, \
                    "Post-processing gain must be positive and smaller than 1."
            self.ppFact = lambda s, G=float(ppFact): G

        # check for post-processing factor, function of the working angle
        if isinstance(ppFact_char, basestring):
            pth = os.path.normpath(os.path.expandvars(ppFact_char))
            assert os.path.isfile(pth), "%s is not a valid file."%pth
            with fits.open(pth) as ff:
                dat = ff[0].data
            assert len(dat.shape) == 2 and 2 in dat.shape, \
                    "Wrong post-processing-char gain data shape."
            WA, G = (dat[0], dat[1]) if dat.shape[0] == 2 else (dat[:,0], dat[:,1])
            assert np.all(G > 0) and np.all(G <= 1), \
                    "Post-processing-char gain must be positive and smaller than 1."
            # gain outside of WA values defaults to 1
            Ginterp = scipy.interpolate.interp1d(WA, G, kind='cubic',
                    fill_value=1., bounds_error=False)
            self.ppFact_char = lambda s: np.array(Ginterp(s.to('arcsec').value), ndmin=1)
        elif isinstance(ppFact_char, numbers.Number):
            assert ppFact_char > 0 and ppFact_char <= 1, \
                    "Post-processing-char gain must be positive and smaller than 1."
            self.ppFact_char = lambda s, G=float(ppFact_char): G
            
        # check for minimum FA delta magnitude, function of the working angle
        if isinstance(FAdMag0, basestring):
            pth = os.path.normpath(os.path.expandvars(FAdMag0))
            assert os.path.isfile(pth), "%s is not a valid file."%pth
            with fits.open(pth) as ff:
                dat = ff[0].data
            assert len(dat.shape) == 2 and 2 in dat.shape, \
                    "Wrong FAdMag0 data shape."
            WA, G = (dat[0], dat[1]) if dat.shape[0] == 2 else (dat[:,0], dat[:,1])
            # gain outside of WA values defaults to 25
            Ginterp = scipy.interpolate.interp1d(WA, G, kind='cubic',
                    fill_value=25., bounds_error=False)
            self.FAdMag0 = lambda s: np.array(Ginterp(s.to('arcsec').value),
                    ndmin=1)
        elif isinstance(FAdMag0, numbers.Number):
            self.FAdMag0 = lambda s, G=float(FAdMag0): G
        
        # populate outspec
        for att in self.__dict__:
            if att not in ['vprint', 'ppFact', 'ppFact_char', 'FAdMag0','_outspec']:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat

        # populate with values which may be interpolants
        self._outspec['ppFact'] = ppFact
        self._outspec['ppFact_char'] = ppFact_char
        self._outspec['FAdMag0'] = FAdMag0
        
        # instantiate background sources object
        self.BackgroundSources = get_module(specs['modules']['BackgroundSources'],
                'BackgroundSources')(**specs)

    def __str__(self):
        """String representation of Post Processing object
        
        When the command 'print' is used on the Post Processing object, 
        this method will return the values contained in the object
        
        """
        
        for att in self.__dict__:
            print('%s: %r' % (att, getattr(self, att)))
        
        return 'Post Processing class object attributes'

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
       
        Notes:
            The prototype implemenation does not consider background sources
            in calculating false positives, however, the unused TargetList, 
            integration time and star index inputs are part of the interface
            to allow an implementation to do this.
        """
        
        # initialize
        FA = False
        MD = np.array([False]*len(SNR))
        
        # 1/ For the whole system: is there a False Alarm (false positive)?
        p = np.random.rand()
        if p <= self.FAP:
            FA = True
        
        # 2/ For each planet: is there a Missed Detection (false negative)?
        SNRmin = mode['SNR']
        MD[SNR < SNRmin] = True
        
        return FA, MD

