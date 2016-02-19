# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem
from astropy import units as u
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

class OptSys_Kasdin(OpticalSystem):
    """WFIRST Optical System class
    
    This class contains all variables and methods specific to the WFIRST
    optical system needed to perform Optical System Module calculations
    in exoplanet mission simulation.
    
    Args:
        \*\*specs:
            user specified values

    """
    
    def __init__(self, **specs):
                
        OpticalSystem.__init__(self, **specs)

    def calc_maxintTime(self, targlist):
        """Finds maximum integration time for target systems 
        
        This method is called in the __init__ for the TargetList class object.
        
        Args:
            targlist:
                TargetList class object
        
        Returns:
            maxintTime:
                1D numpy array containing maximum integration time for target
                list stars (astropy Quantity with units of day)
        
        """
        
        # Inclination for max zodi level
        Imax = np.array([0.]*targlist.nStars);
        Imin = np.array([0.0403/(2*0.000269)]*targlist.nStars); # from Lindler
        
        # Calculate IWA and OWA, defined as angular separations
        # corresponding to 50% of maximum throughput
        xmin = self.IWA.value;
        xmax = self.OWA.value;
        xopt = opt.fmin(lambda x:-self.throughput(self.lam,x),xmax,disp=0);
        Tmax = self.throughput(self.lam,xopt);
        IWA = opt.fsolve(lambda x:self.throughput(self.lam,x)-Tmax/2.,xmin);
        OWA = xmax-opt.fsolve(lambda x:self.throughput(self.lam,xmax-x)-Tmax/2.,0.);
        
        # calculate max integration time of the stars of interest
        maxintTime = self.calc_intTime(targlist,range(targlist.nStars),self.dMagLim,IWA,Imax);
        
        return maxintTime
        
    def calc_intTime(self, targlist, starInd, dMagPlan, WA, I):
        """Finds integration time for a specific target system 
        
        This method is called from a method in the SurveySimulation class
        object.
        
        Args:
            targlist:
                TargetList class object
            starInd (integer ndarray):
                Numpy ndarray containing integer indices of the stars of interest, 
                with the length of the number of planets of interest.
            dMagPlan:
                Numpy ndarray containing differences in magnitude between planets 
                and their host star
            WA:
                Numpy ndarray containing working angles of the planets of interest
            I:
                Numpy ndarray containing inclinations of the planets of interest
        
        Returns:
            intTime:
                1D numpy array of integration times (astropy Quantity with 
                units of day)
        
        """

        PP = targlist.PostProcessing                # post-processing module
        ppFact = PP.ppFact;                         # post-processing contrast factor
        Vmag = targlist.Vmag[starInd];              # star visual magnitude
        zodi = targlist.ZodiacalLight               # zodiacalLight module
        EZlevel = zodi.fzodi(starInd,I,targlist);   # exozodi level
        EZmag = zodi.exozodiMag;                    # 1 zodi brightness in mag per asec2
        Omega = (0.7*self.lam.to(u.m))**2*self.shapeFac/self.pupilArea\
                *(u.rad.to(u.arcsec))**2;           # pixel size in square arcseconds

        # values taken from the imaging camera
        for syst in self.scienceInstruments:
            if 'imag' in syst['type']:
                QEfunc = syst['QE'];
                darkCurrent = syst['darkCurrent'];
                CIC = syst['CIC'];
                readNoise = syst['readNoise'];
                texp = syst['texp'];
                pixelPitch = syst['pixelPitch'];
                focalLength = syst['focalLength'];
                ENF = syst['ENF'];
                G_EM = syst['G_EM'];

        # values derived from the normalized PSF
        PSF = self.PSF(self.lam, WA);
        Pbar = PSF/np.max(PSF);
        P1 = np.sum(Pbar);
        Psi = np.sum(Pbar**2)/(np.sum(Pbar))**2;
        Xi = np.sum(Pbar**3)/(np.sum(Pbar))**3;
        # nb of pixels for photometry aperture = 1/sharpness
        Npix = 1./Psi;

        # throughput, contrast
        T = self.attenuation**2*self.throughput(self.lam, WA);
        Q = ppFact*self.contrast(self.lam, WA);

        # average irradiance in detection band [photons /s /m2 /nm]
        F0 = 3631*1.51e7/self.lam *u.photon/u.s/u.m**2; #zero-magnitude star = 3631 Jy
        I_psf = F0*10.**(-Vmag/2.5);
        I_pl = F0*10.**(-(Vmag + dMagPlan)/2.5); 
        I_CG = I_psf * Q;
        I_zodi = F0*10.**(-EZmag/2.5)*EZlevel*Omega;

        # electron rates [ s^-1 ]
        QE = QEfunc(self.lam) /u.photon;
        r_pl = I_pl*QE*T*self.pupilArea*self.deltaLam;
        r_CG = I_CG*QE*T*self.pupilArea*self.deltaLam;
        r_zodi = I_zodi*QE*T*self.pupilArea*self.deltaLam;
        r_dark = darkCurrent*Npix;
        r_cic = CIC*Npix/texp;
        r_read = (readNoise/G_EM)**2*Npix/texp;

        # Kasdin06+ method
        r_noise = ENF**2*(r_CG + r_zodi + r_dark + r_cic) + r_read;
        Qbar = r_pl/r_noise*P1;
        beta = r_pl/T;
        K = st.norm.ppf(1-PP.FAP);                  # false alarm threshold
        gamma = st.norm.ppf(1-PP.MDP);              # missed detection threshold
        deltaAlphaBar = (pixelPitch**2/focalLength**2*self.pupilDiam**2/self.lam**2)\
                .decompose();                       # dimensionless pixel size
        Ta = T*self.shapeFac*deltaAlphaBar*P1;      # Airy throughput 

        intTime = 1./beta*(K - gamma*np.sqrt(1.+Qbar*Xi/Psi))**2/(Qbar*Ta*Psi)

        return intTime.to(u.day)

