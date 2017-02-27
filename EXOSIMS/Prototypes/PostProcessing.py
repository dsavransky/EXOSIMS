# -*- coding: utf-8 -*-
import numpy as np
import astropy.units as u
import scipy.stats as st
import scipy.interpolate
import numbers
from EXOSIMS.util.get_module import get_module

class PostProcessing(object):
    """Post Processing class template
    
    This class contains all variables and functions necessary to perform 
    Post Processing Module calculations in exoplanet mission simulation.
    
    Args:
        \*\*specs:
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
        maxFAfluxratio (float, callable):
            Maximum flux ratio that can be obtained by a false alarm: either a scalar 
            for constant flux ratio, or a two-column array for separation-dependent 
            flux ratio, where the first column contains the angular separation in 
            units of arcsec. May be data or FITS filename.
            
    """
    
    _modtype = 'PostProcessing'
    _outspec = {}

    def __init__(self,FAP=3e-7,MDP=1e-3,ppFact=1.0,maxFAfluxratio=1e-6,**specs):
        
        self.FAP = float(FAP)       # false alarm probability
        self.MDP = float(MDP)       # missed detection probability
        
        # check for post-processing factor, function of the working angle
        if isinstance(ppFact,basestring):
            pth = os.path.normpath(os.path.expandvars(ppFact))
            assert os.path.isfile(pth), "%s is not a valid file."%pth
            dat = fits.open(pth)[0].data
            assert len(dat.shape) == 2 and 2 in dat.shape, "Wrong post-processing gain data shape."
            WA,G = (dat[0],dat[1]) if dat.shape[0] == 2 else (dat[:,0],dat[:,1])
            assert np.all(G>0) and np.all(G<=1), \
                    "Post-processing gain must be positive and smaller than 1."
            # gain outside of WA values defaults to 1
            Ginterp = scipy.interpolate.interp1d(WA, G, kind='cubic',\
                    fill_value=1., bounds_error=False)
            self.ppFact = lambda s: np.array(Ginterp(s.to('arcsec').value),ndmin=1)
        elif isinstance(ppFact,numbers.Number):
            assert ppFact>0 and ppFact<=1, \
                    "Post-processing gain must be positive and smaller than 1."
            self.ppFact = lambda s, G=float(ppFact): G
            
        # check for max FA flux ratio, function of the working angle
        if isinstance(maxFAfluxratio,basestring):
            pth = os.path.normpath(os.path.expandvars(maxFAfluxratio))
            assert os.path.isfile(pth), "%s is not a valid file."%pth
            dat = fits.open(pth)[0].data
            assert len(dat.shape) == 2 and 2 in dat.shape, "Wrong max FA flux ratio data shape."
            WA,G = (dat[0],dat[1]) if dat.shape[0] == 2 else (dat[:,0],dat[:,1])
            assert np.all(G>0) and np.all(G<=1), \
                    "Max FA flux ratio must be positive and smaller than 1."
            # gain outside of WA values defaults to 1
            Ginterp = scipy.interpolate.interp1d(WA, G, kind='cubic',\
                    fill_value=1., bounds_error=False)
            self.maxFAfluxratio = lambda s: np.array(Ginterp(s.to('arcsec').value),ndmin=1)
        elif isinstance(maxFAfluxratio,numbers.Number):
            assert maxFAfluxratio>0 and maxFAfluxratio<=1, \
                    "Max FA flux ratio must be positive and smaller than 1."
            self.maxFAfluxratio = lambda s, G=float(maxFAfluxratio): G
        
        # populate outspec
        # populate with value which may be interpolants
        self._outspec['ppFact'] = ppFact
        self._outspec['maxFAfluxratio'] = maxFAfluxratio
        for att in self.__dict__.keys():
            if att not in ['ppFact','maxFAfluxratio']:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat,u.Quantity) else dat
        
        #instantiate background sources object
        self.BackgroundSources = get_module(specs['modules']['BackgroundSources'], \
                'BackgroundSources')(**specs)

    def __str__(self):
        """String representation of Post Processing object
        
        When the command 'print' is used on the Post Processing object, 
        this method will return the values contained in the object"""
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Post Processing class object attributes'

    def det_occur(self, SNR, SNRmin):
        """Determines if a detection has occurred and returns booleans 
        
        This method returns two booleans where True gives the case.
        
        Args:
            SNR (float ndarray):
                signal-to-noise ratio of the planets around the selected target
            SNRmin (float):
                signal-to-noise ratio threshold for detection
        
        Returns:
            FA (boolean):
                False alarm (false positive) boolean.
            MD (boolean ndarray):
                Missed detection (false negative) boolean with the size of 
                number of planets around the target.
       
        Notes:
            TODO: Add backgroundsources hook
        """
        
        # initialize
        FA = False
        MD = np.array([False]*len(SNR))
        
        # 1/ For the whole system: is there a False Alarm (false positive)?
        p = np.random.rand()
        if p <= self.FAP:
            FA = True
        
        # 2/ For each planet: is there a Missed Detection (false negative)?
        MD[SNR < SNRmin] = True
        
        return FA, MD

