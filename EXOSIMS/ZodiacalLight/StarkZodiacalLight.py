# -*- coding: utf-8 -*-
from EXOSIMS.Prototypes.ZodiacalLight import ZodiacalLight
import numpy as np
from astropy import units as u
from scipy.interpolate import interp1d, interp2d

class StarkZodiacalLight(ZodiacalLight):
    """Stark Zodiacal Light class
    
    This class contains all variables and methods necessary to perform
    Zodiacal Light Module calculations in exoplanet mission simulation using
    the model from Stark et al. 2014."""

    def fzodi(self, sInd, I, targlist):
        """Returns total zodi flux levels (local and exo)  
        
        This method is called in __init__ of SimulatedUniverse.
        
        Args:
            sInd (integer ndarray):
                Numpy ndarray containing integer indices of the stars of interest, 
                with the length of the number of planets of interest.
            I:
                Numpy ndarray containing inclinations of the planets of interest
            targlist:
                TargetList class object
        
        Returns:
            fzodicurr (ndarray):
                1D numpy ndarray of zodiacal light levels

        """
         
        # maximum V magnitude
        MV = targlist.MV 
        # ecliptic latitudes
        beta = self.eclip_lats(targlist.coords).value 
        #solar elongation
        elon = 59                       # need to set elongation as an input
        # wavelength
        lam = targlist.OpticalSystem.Imager['lam'].value
        
        i = np.where(I.value > 90.)
        if type(I) == np.ndarray:
            I[i] = 180. - I[i]
        
        if self.exoZvar == 0:
            R = self.exoZnumber
        else:
            # assume log-normal distribution of variance
            mu = np.log(self.exoZnumber) - 0.5*np.log(1. + self.exoZvar/self.exoZnumber**2)
            v = np.sqrt(np.log(self.exoZvar/self.exoZnumber**2 + 1.))
            R = np.random.lognormal(mean=mu, sigma=v, size=(len(sInd),))

        fzodi = self.fbeta(beta[sInd],elon,lam) + 2.*R*self.fbeta(I,elon,lam)*2.5**(4.78-MV[sInd])

        return fzodi
        
    def fbeta(self, beta, elon, lam):
        """Variation of zodiacal light with viewing angle, based on interpolation 
        of Leinert et al. (1998) table.
        
        Args:
            beta (ndarray):
                ecliptic latitude angle in degrees
            elon (ndarray):
                solar elongation angle in degrees
            lam:
                wavelength

        Returns:
            f (ndarray):
                zodiacal light in zodi
        
        """

        # Table 17 in Leinert et al. (1998)
        # Zodiacal light brightness function of solar longitude (rows) and beta values (columns)
        # Values given in W m−2 sr−1 μm−1 for a wavelength of 500 nm
        # Note: for a starshade, we assume a mean solar elongation of 59 degrees
        try:
            path = os.path.split(inspect.getfile(self.__class__))[0]
            Izod = np.load(os.path.join(path, 'Leinert98_table17.npy'))
            beta_vector = np.array([0.,5,10,15,20,25,30,45,60,75])
            sollong_vector = np.array([0.,5,10,15,20,25,30,35,40,45,60,75,90,105,120,135,150,165,180])
            beta_array,sollong_array = np.meshgrid(beta_vector,sollong_vector)
            sinbeta = np.sin(np.radians(beta))
            cosbeta = np.cos(np.radians(beta))
            sinelon = np.sin(np.radians(elon))
            coselon = np.cos(np.radians(elon))
            # inclination to point at beta of all targets
            inclination = np.degrees(np.arcsin(sinbeta/sinelon))
            # solar longitude of targets at elongation of ss_elongation
            cossollong = coselon/cosbeta
            sinsollong = sinelon/cosbeta*np.cos(np.radians(inclination))
            sollong = np.degrees(np.arctan2(sinsollong,cossollong))
            # Now that we have the solar longitude and beta of every target at a
            # specific solar elongation, we can calculate their zodi values
            sollong_indices = sollong/5.
            beta_indices = beta/5.
            j = np.where(sollong >= 45)[0]
            if j:
                sollong_indices[j] = 9.+(sollong[j]-45.)/15.
            j = np.where(beta >= 30)[0]
            if j:
                beta_indices[j] = 6.+(beta[j]-30.)/15.
            k = np.where(abs(sollong_vector-90.) == np.min(abs(sollong_vector-90.)))[0]
            z = Izod/Izod[k,0]
            y = np.array(range(z.shape[0]))
            x = np.array(range(z.shape[1]))
            fbeta = np.diag(interp2d(x,y,z)(beta_indices,sollong_indices))

        except:
            print "WARNING: failed interpolation, using Stark's fit to table\
             near sollong ~ 135 degrees."
            sinbeta = abs(np.sin(np.radians(beta)))
            I135_o_I90 = 0.69
            f135 = 1.02331 - 0.565652*sinbeta - 0.883996*sinbeta**2 + 0.852900*sinbeta**3
            fbeta = I135_o_I90 * f135

        # wavelength dependence, from Table 19 in Leinert et al 1998:
        zodi_lam = np.array([0.2,0.3,0.4,0.5,0.7,0.9,1.0,1.2,2.2,3.5,4.8,12,25,60,100,140])  # microns
        zodi_Blam = np.array([2.5e-8,5.3e-7,2.2e-6,2.6e-6,2.0e-6,1.3e-6,1.2e-6,8.1e-7,1.7e-7,5.2e-8,1.2e-7,7.5e-7,3.2e-7,1.8e-8,3.2e-9,6.9e-10]) # W m^-2 sr^-1 micron^-1
        #It's better to interpolate w/ a quadratic in log-log space
        x = np.log10(zodi_lam)
        y = np.log10(zodi_Blam)
        f_corr = 10.**(interp1d(x,y,kind='quadratic')(np.log10(lam*1e-3)))
        f_corr *= 1e7                   # convert W to erg s^-1
        f_corr /= 4.25e10               # convert sr^-1 to arcsec^-2
        f_corr /= 1000.                 # convert micron^-1 to nm^-1
        h = 6.6260755e-27               # Planck constant in erg s
        c = 2.99792458e8                # speed of light in vacuum in m s-1
        ephoton = h*c/(lam*1e-9)        # energy of a photon in erg
        f_corr /= ephoton               # photons s^-1 m^-2 arcsec^-2 nm^-1
        F0 = 3631*1.51e7/lam            # zero-magnitude star in photon s-1 m-2 nm-1
        f_corr /= F0                    # color correction factor
        
        fbeta *= f_corr
        
        return fbeta


