# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import EXOSIMS.util.statsFun as statsFun
import numpy as np
from matplotlib import pyplot as plt
import astropy.units as u
import astropy.constants as const


class PlanetFunc(PlanetPopulation):
    """Planet Functions inherits all methods and attributes of the
    PlanetPopulation class
    
    This class contains functions to describe to probablity distribution
    of each parameter and the number of planets in the model"""
    def __init__(self, **specs):
        PlanetPopulation.__init__(self, **specs)
        # overwrite default values
        # semi-major axis range
        self.arange = np.array([.1, 100])*u.AU
        # eccentricity range
        self.erange = np.array([10.*np.finfo(np.float).eps, 0.8]) 
        #planetary radius range
        self.Rrange = np.array([10.*np.finfo(np.float).eps, const.R_earth.value*20.])*const.R_earth.unit
        # planetary mass range
        self.Mprange = np.array([const.M_jup.value*6.3e-5, const.M_jup.value*28.5])*const.M_jup.unit

    
    def semi_axis(self, x, beta=-0.61, s0=35):
        """
        Provides PDF of semi-major axis
        
        Function is logarithmic
        Cumming et. al (2008)
        """
        return (x**(beta))*(np.exp(-((x/s0)**2)))/8.388972895945018
    
   
    def eccentricity(self, x, sigma=.25):
        """
        Provides PDF of semi-major axis
        
        Function is a Rayleigh distribution
        Kipping et. al (2013)
        """
        return (x/sigma**2)*np.exp(-x**2/(2.*sigma**2))/0.994023977104994
  
    
    def mass(self, x):
        """
        Provides PDF of exoplanet masses
        
        Function is an exponential
        Marcey et. al (2005)
        """
        return  (x**-1.05)/(self.Mprange.min().value**(-0.05) - self.Mprange.max().value**(-0.05))/20.


    def radius(self, x):
        """
        Provides PDF of planet radius
        
        Function is a peacewise normal distribution
        Fressin et. al (2013)
        """
        # scale factor
        r = const.R_earth.value
        x = x/r
        
        a1 = 0.05346
        b1 = 1.301
        c1 = 0.3434
        
        a2 = 0.07602
        b2 = 2.185
        c2 = 1.116
        
        a3 = 0.007558
        b3 = 12.51
        c3 = 4.905
        
        res = a1*np.exp(-((x-b1)/c1)**2)
        res += a2*np.exp(-((x-b2)/c2)**2)
        res += a3*np.exp(-((x-b3)/c3)**2)
                
        return res/1576509.7512160488
       
    def inds_select(self, x, low, high):
        """Finds indices meeting specific conditions
        
        Args:
            x (ndarray):
                1D numpy ndarray of values
            low (float):
                Lower bound
            high (fload):
                Upper bound
                
        Returns:
            inds (ndarray):
                1D numpy ndarray of indices meeting the conditions
            
        """
        
        inds_low = np.where(x >= low)[0]
        inds_high = np.where(x < high)[0]
        inds = np.intersect1d(inds_low, inds_high)
        
        return inds
    
    
    def plot(self, f):
        """
        Used to plot PDFs as histograms in order to view their accuracy
        """
        planets = 1000
        minimum = self.Rrange.min().value
        maximum = self.Rrange.max().value
        distr = statsFun.simpSample(f, planets, minimum, maximum)
        plt.hist(distr)
        plt.show()    
    
   
