# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem
from astropy import units as u
import numpy as np

class WFIRSTOpticalSystem(OpticalSystem):
    """WFIRST Optical System class
    
    This class contains all variables and methods specific to the WFIRST
    optical system needed to perform Optical System Module calculations
    in exoplanet mission simulation.
    
    Args:
        \*\*specs:
            user specified values
    
    Attributes:
        dAlpha:
            pixel size in square arcseconds (astropy Quantity in arcsec**2)
        Psi:
            instrument specific value 'sharpness' derived from PSF
        Xi:
            instrument specific value derived from PSF
        P1:
            instrument specific value derived from PSF
        alphaBar:
            instrument specific value
        Ta:
            Airy throughput
    
    """
    
    def __init__(self, **specs):
                
        OpticalSystem.__init__(self, **specs)
                    
        # set values derived from quantities above
        # pixel size in square arcseconds (as**2)
        self.dAlpha = (self.lam.to(u.m)**2)*self.shapeFac/(4.*self.pupilArea)*(180.*3600./np.pi)**2*(u.arcsec)**2 
        self.Psi = 1. # instrument specific value "sharpness" derived from PSF
        self.Xi = 1. # instrument specific value derived from PSF
        self.P1 = 1. # instrument specific value derived from PSF
        # instrument specific value
        self.alphaBar = (self.pixelArea*(self.pupilArea/self.shapeFac)/self.focalLength**2/self.lam**2).decompose() 
        self.Ta = self.throughput*self.shapeFac*self.P1*self.alphaBar # Airy throughput 
        
    def calc_maxintTime(self, targlist):
        """Finds maximum integration time for target systems 
        
        This method is called in the __init__ for the TargetList class object.
        
        Args:
            targlist:
                TargetList class object which also has access to:
                    targlist.ZodiacalLight:
                        ZodiacalLight class object
        
        Returns:
            maxintTime:
                1D numpy array containing maximum integration time for target
                list stars (astropy Quantity with units of day)
        
        """
        
                
        dmags = targlist.OpticalSystem.dMagLim # limiting delta magnitude
        throughput = targlist.OpticalSystem.throughput # throughput at IWA
        contrast = targlist.OpticalSystem.contrast # contrast at IWA
        Vmag = targlist.Vmag # visual magnitude
        opt = targlist.OpticalSystem # OpticalSystem module
        zodi = targlist.ZodiacalLight # ZodiacalLight module

        # exozodi level
        fzodi = zodi.fbeta(zodi.eclip_lats(targlist.coords).value) + 2.*zodi.exozodi*2.44*(2.5**(4.78-targlist.MV))
        # band-specific flux for zero-magnitude star        
        F = 9.57e7*(1/u.m)**2*(1/u.nm)*(1/u.s)
        
        Qmin = (10.**(dmags/2.5)*contrast +
        10.**((Vmag + dmags - 23.54)/2.5)*fZodiacalLight*opt.dAlpha.value +
        10.**((Vmag + dmags)/2.5)*
        (opt.darkRate + (opt.readNoise**2/opt.texp))/
        (F*opt.QE*opt.attenuation*opt.deltaLambda*opt.pupilArea*throughput))**(-1)

        Qbar = Qmin*opt.P1
        # average irradiance in detection band (photons/m**2/nm/s)
        Ip = F*10.**(-(Vmag + dmags)/2.5)
        beta = opt.QE*opt.attenuation*opt.deltaLambda*opt.pupilArea*Ip # photons/s
        # maximum time in seconds
        t_max = ((rules.K - rules.gamma*np.sqrt(1.+Qbar*opt.Xi/opt.Psi))**2/(Qbar*opt.Ta*opt.Psi))/beta
        # apply observational duty cycle        
        t_max = t_max*rules.dutyCycle
        
        return t_max
        
    def calc_intTime(self, targlist, opt, rules, universe, s_ind, planInds):
        """Finds integration time for a specific target system 
        
        This method is called from a method in the SurveySimulation class
        object.
        
        Args:
            targlist:
                TargetList class object
            opt:
                OpticalSystem class object
            rules:
                Rules class object
            universe:
                SimulatedUniverse class object
            s_ind:
                target star index
            planInds:
                planet indices linking back to target star
        
        Returns:
            intTime:
                1D numpy array of integration times (astropy Quantity with 
                units of day)
        
        """
        
        # integration time given as 1 day
        intTime = np.array([1.]*len(planInds))*u.day
        
        return intTime
