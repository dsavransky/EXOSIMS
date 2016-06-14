from EXOSIMS.PlanetPhysicalModel.FortneyMarleyCahoyMix1 import FortneyMarleyCahoyMix1
from EXOSIMS.PlanetPhysicalModel.func import piece_linear, ProbRGivenM, classification
import astropy.units as u
import astropy.constants as const
import numpy as np
import os, inspect, h5py 


class Forecaster(FortneyMarleyCahoyMix1):
    """
    Planet M-R relation model based on the FORECASTER software, Chen & Kippling 2016.
    This module requires to download the following files:
    - fitting_parameters.h5
    - func.py
    from the FORECASTER GitHub repository at https://github.com/chenjj2/forecaster
    
    Args: 
        \*\*specs: 
            user specified values
    
    """

    def __init__(self, n_pop=4, **specs):
        
        FortneyMarleyCahoyMix1.__init__(self, **specs)
        
        # number of category
        self.n_pop = int(n_pop)
        
        # read parameter file: fitting_parameters.h5
        classpath = os.path.split(inspect.getfile(self.__class__))[0]
        filename = 'fitting_parameters.h5'
        parampath = os.path.join(classpath, filename)
        h5 = h5py.File(parampath, 'r')
        self.all_hyper = h5['hyper_posterior'][:]
        h5.close()

    def calc_radius_from_mass(self, M):
        """
        Forecast the Radius distribution given the mass distribution.
        
        Args:
            M (astropy Quantity array)
               Planet mass values in units of kg
        
        Returns:
            R (astropy Quantity array)
                Planet radius values in units of km
        
        """
        
        mass = (M/const.M_earth).decompose().value
        assert np.min(mass) > 3e-4 and np.max(mass) < 3e5, \
                "Mass range out of model expectation. Returning None."
        
        sample_size = len(mass)
        logm = np.log10(mass)
        prob = np.random.random(sample_size)
        logr = np.ones_like(logm)
        hyper_ind = np.random.randint(low=0, high=np.shape(self.all_hyper)[0], size=sample_size)
        hyper = self.all_hyper[hyper_ind,:]
        
        for i in range(sample_size):
            logr[i] = piece_linear(hyper[i], logm[i], prob[i])
        
        radius_sample = 10.** logr
        
        return radius_sample*const.R_earth.to('km')

