# -*- coding: utf-8 -*-
from astropy import units as u
import numpy as np

class OpticalSystem(object):
    """Optical System class template
    
    This class contains all variables and methods necessary to perform
    Optical System Definition Module calculations in exoplanet mission 
    simulation.
    
    Args:
        \*\*specs:
            user specified values
            
    Attributes:
        lam (Quantity):
            detection wavelength (default units of nm)
        shapeFac (float):
            shape factor (also known as fill factor) where 
            :math:`shapeFac * diameter^2 = Area`
        pupilArea (Quantity):
            entrance pupil area (default units of m\ :sup:`2`)
        SNchar (float):
            Signal to Noise Ratio for characterization
        haveOcculter (bool):
            boolean signifying if the system has an occulter
        pixelArea (Quantity):
            pixel area (default units of m\ :sup:`2`)
        focalLength (Quantity):
            focal length (default units of m)
        IWA (Quantity):
            Inner Working Angle (default units of arcsecond)
        OWA (Quantity):
            Outer Working Angle (default units of arcsecond)
        dMagLim (float):
            limiting delta magnitude. **Note: wavelength dependent?**
        throughput (callable(lam, alpha)):
            optical system throughput (must be callable - can be lambda, function,
            scipy.interpolate.interp2d object, etc.) with inputs wavelength (astropy
            Quantity) and angular separation (astropy Quantity).
        contrast (callable(lam, alpha)):
            optical system contrast curve (must be callable - can be lambda, function,
            scipy.interpolate.interp2d object, etc.) with inputs wavelength (astropy
            Quantity) and angular separation (astropy Quantity).
        dr (Quantity):
            detector dark-current rate per pixel (default units of 1/s)
        sigma_r (float):
            detector read noise
        t_exp (Quantity):
            exposure time per read (default units of s)
        PSF (callable(lam, alpha)):
            point spread function - 2D ndarray of values, normalized to 1 at the core
            (must be callable - can be lambda, function,
            scipy.interpolate.interp2d object, etc.) with inputs wavelength (astropy
            Quantity) and angular separation (astropy Quantity). Note normalization 
            means that all throughput effects must be contained in the throughput 
            attribute.
        PSFSampling (Quantity):
            Sampling of PSF in arcsec/pixel (default units of arcsec)
        QE (callable(lam)):
            detector quantum efficiency (must be callable - can be lambda, function,
            scipy.interpolate.interp2d object, etc.) with inputs wavelength (astropy
            Quantity).  This represents the primary (detection) detector. 
        eta2 (float):
            post coronagraph attenuation
        deltaLambda (Quantity):
            bandpass (default units of nm)
        telescopeKeepout (float):
            telescope keepout angle in degrees
        specLambda (Quantity):
            spectral wavelength of interest (default units of nm)
            Note:  probably belongs in postprocessing or rules.
        Rspec (float):
            spectral resolving power :math:`\lambda/\Delta\lambda`
        Npix (float):
            number of noise pixels
        Ndark (float):
            number of dark frames used
        optical_oh (Quantity):
            optical system overhead (default units of day)
        darkHole (Quantity):
            dark hole size (default units of rad)
        intCutoff (Quantity):
            integration cutoff (default units of day)
            
    """

    _modtype = 'OpticalSystem'
    
    def __init__(self, **specs):
                
        # default Optical System values
        # detection wavelength (nm)        
        self.lam = 500.*u.nm 
        # shapeFac*diameter**2 = Area
        self.shapeFac = np.pi/4. 
        # entrance pupil area
        self.pupilArea = 4.*np.pi*(u.m)**2 
        # Signal to Noise Ratio for characterization
        self.SNchar = 11. 
        # boolean signifying if system has occulter
        self.haveOcculter = False 
        # pixel area (m**2)
        self.pixelArea = 1e-10*(u.m)**2 
        # focal length (m)
        self.focalLength = 240.*u.m 
        # Inner Working Angle (arcseconds)
        self.IWA = 0.075*u.arcsec 
        # Outer Working Angle (arcseconds)
        self.OWA = np.inf*u.arcsec 
        # limiting delta magnitude
        self.dMagLim = 26. 
        # optical system throughput
        self.throughput = lambda lam, alpha: 0.5 
        # optical system designed suppression level
        self.contrast = lambda lam, alpha: 1.e-10 
        # detector dark-current rate per pixel
        self.dr = 0.001*(1/u.s) 
        # detector read noise (e/s)
        self.sigma_r = 3. 
        # exposure time per read (s)
        self.t_exp = 1000.*u.s 
        # detection quantum efficiency
        self.QE = lambda lam: 0.5 
        # post coronagraph attenuation
        self.eta2 = 0.57 
        # bandpass (nm)
        self.deltaLambda = 100.*u.nm 
        # point spread function
        self.PSF = lambda lam, alpha: np.ones((3,3))
        # sampling of PSF in arcsec/pixel
        self.PSFSampling = 10.*u.arcsec
        # keepout angle in degrees
        self.telescopeKeepout = 45.
        # spectral wavelength of interest
        self.specLambda = 760.*u.nm
        # spectral resolving power
        self.Rspec = 70.
        # number of noise pixels
        self.Npix = 14.3
        # number of dark frames used
        self.Ndark = 10.
        # optical system overhead time
        self.optical_oh = 1.*u.day
        # dark hole size
        self.darkHole = np.pi*u.rad
        # integration cutoff
        self.intCutoff = 50.*u.day 
        
        # replace default values with any user specified values
        atts = self.__dict__.keys()
        for att in atts:
            if att in specs:
                if att == 'lam' or att == 'deltaLambda' or att == 'specLambda':
                    # set lam with proper units
                    setattr(self, att, specs[att]*u.nm)
                elif att == 'pupilArea' or att == 'pixelArea':
                    # set pupilArea or pixelArea with proper units
                    setattr(self, att, specs[att]*(u.m)**2)
                elif att == 'focalLength':
                    # set focalLength with proper units
                    setattr(self, att, specs[att]*u.m)
                elif att == 'IWA' or att == 'OWA':
                    # set IWA or OWA with proper units
                    setattr(self, att, specs[att]*u.arcsec)
                elif att == 'dr':
                    # set dr with proper units
                    setattr(self, att, specs[att]*(1/u.s))
                elif att == 't_exp':
                    # set t_exp with proper units
                    setattr(self, att, specs[att]*u.s)
                elif att == 'intCutoff':
                    setattr(self, att, specs[att]*u.day)
                else:
                    setattr(self, att, specs[att])
                    
        # set values derived from quantities above
        if np.isinf(self.OWA.value):
            self.OWA = np.pi/2
                
    def __str__(self):
        """String representation of the Optical System object
        
        When the command 'print' is used on the Optical System object, this 
        method will print the attribute values contained in the object"""
        
        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Optical System class object attributes'
        
    def calc_maxintTime(self, targlist):
        """Finds maximum integration time for target systems 
        
        This method is called in the __init__ method of the TargetList class
        object.
        
        This method defines the data type expected, maximum integration time is
        determined by specific OpticalSystem classes.
        
        Args:
            targlist (TargetList):
                TargetList class object which, in addition to TargetList class
                object attributes, has available:
                    targlist.OpticalSystem:
                        OpticalSystem class object
                    targlist.PlanetPopulation:
                        PlanetPopulation class object
                    targlist.ZodiacalLight:
                        ZodiacalLight class object
                    targlist.Completeness:
                        Completeness object
        
        Returns:
            maxintTime (Quantity):
                1D numpy ndarray of maximum integration times (default units of
                day)
        
        """
        
        # maximum integration time is given as 1 day for each system in target list
        maxintTime = np.array([1.]*len(targlist.Name))*u.day
        
        return maxintTime
        
    def calc_intTime(self, targlist, universe, s_ind, planInds):
        """Finds integration time for a specific target system 
        
        This method is called by a method in the SurveySimulation object class.
        
        This method defines the data type expected, integration time is 
        determined by specific OpticalSystem classes.
        
        Args:
            targlist (TargetList):
                TargetList class object
            universe (SimulatedUniverse):
                SimulatedUniverse class object
            s_ind (int):
                target star index
            planInds (ndarray):
                planet index linking back to target star
        
        Returns:
            intTime (Quantity):
                1D numpy ndarray of integration times (default units of day)
      
      """
        
        # integration time given as 1 day
        intTime = np.array([1.]*len(planInds))*u.day
        
        return intTime
