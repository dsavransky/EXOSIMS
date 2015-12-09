# -*- coding: utf-8 -*-
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from EXOSIMS.util.get_module import get_module

class TargetList(object):
    """Target List class template
    
    This class contains all variables and functions necessary to perform 
    Target List Module calculations in exoplanet mission simulation.
    
    It inherits the following class objects which are defined in __init__:
    StarCatalog, OpticalSystem, PlanetPopulation, ZodiacalLight, Completeness
    
    Args:
        \*\*specs:
            user specified values
            
    Attributes:
        OpticalSystem (OpticalSystem):
            OpticalSystem class object
        PlanetPopulation (PlanetPopulation):
            PlanetPopulation class object
        ZodiacalLight (ZodiacalLight):
            ZodiacalLight class object
        Completeness (Completeness):
            Completeness class object
        Name (ndarray):
            1D numpy ndarray of star names
        Type (ndarray):
            1D numpy ndarray of star types
        Spec (ndarray):
            1D numpy ndarray of spectral types
        parx (ndarray):
            1D numpy ndarray of parallax in milliarcseconds
        Umag (ndarray):
            1D numpy ndarray of U magnitude
        Bmag (ndarray):
            1D numpy ndarray of B magnitude
        Vmag (ndarray):
            1D numpy ndarray of V magnitude
        Rmag (ndarray):
            1D numpy ndarray of R magnitude
        Imag (ndarray):
            1D numpy ndarray of I magnitude
        Jmag (ndarray):
            1D numpy ndarray of J magnitude
        Hmag (ndarray):
            1D numpy ndarray of H magnitude
        Kmag (ndarray):
            1D numpy ndarray of K magnitude
        dist (ndarray):
            1D numpy ndarray of distance in parsecs to star
        BV (ndarray):
            1D numpy ndarray of B-V Johnson magnitude
        MV (ndarray):
            1D numpy ndarray of absolute V magnitude
        BC (ndarray):
            1D numpy ndarray of bolometric correction
        L (ndarray):
            1D numpy ndarray of stellar luminosity in Solar luminosities
        coords (SkyCoord):
            numpy ndarray of astropy SkyCoord objects containing right ascension
            and declination in degrees
        pmra (ndarray):
            1D numpy ndarray of proper motion in right ascension in
            milliarcseconds/year
        pmdec (ndarray):
            1D numpy ndarray of proper motion in declination in 
            milliarcseconds/year
        rv (ndarray):
            1D numpy ndarray of radial velocity in km/s
        Binary_Cut (ndarray):
            1D numpy ndarray of booleans where True is a star with a companion 
            closer than 10 arcsec
        maxintTime (Quantity):
            1D numpy ndarray containing maximum integration time (units of time)
        comp0 (ndarray):
            1D numpy ndarray containing completeness value for each target star
        MsEst (ndarray):
            1D numpy ndarray containing 'approximate' stellar mass in M_sun
        MsTrue (ndarray):
            1D numpy ndarray containing 'true' stellar mass in M_sun
    
    """

    _modtype = 'TargetList'
    
    def __init__(self, keepStarCatalog=False, **specs):
        """Initializes target list
                
        nan data from star catalog quantities are removed
        binary stars are removed
        maximum integration time is calculated
        Filters applied to star catalog data:
            *nan data from star catalog quantities are removed
            *systems with planets inside the IWA removed
            *systems where maximum delta mag is not in allowable orbital range 
            removed
            *systems where integration time is longer than maximum time removed
            *systems not meeting the completeness threshold removed
            
        Additional filters are given in specific TargetList classes."""
        
        # get desired module names (specific or prototype)
                    
        # import StarCatalog class
        Cat = get_module(specs['modules']['StarCatalog'], 'StarCatalog')
        # import OpticalSystem class
        Opt = get_module(specs['modules']['OpticalSystem'], 'OpticalSystem')
        # import ZodiacalLight class
        Zodi = get_module(specs['modules']['ZodiacalLight'], 'ZodiacalLight')
        # import Completeness class
        Comp = get_module(specs['modules']['Completeness'], 'Completeness')
        # import BackgroundSources class
        Back = get_module(specs['modules']['BackgroundSources'], 'BackgroundSources')


        self.StarCatalog = Cat(**specs) # star catalog data
        self.OpticalSystem = Opt(**specs) # optical system object 
        self.ZodiacalLight = Zodi(**specs) # zodiacal light model object 
        self.BackgroundSources = Back(**specs) #background sources model object
        self.Completeness = Comp(**specs) # completeness model object 
        self.PlanetPopulation = self.Completeness.PlanetPopulation # planet population object 
        
        # list of possible Star Catalog attributes
        atts = ['Name', 'Type', 'Spec', 'parx', 'Umag', 'Bmag', 'Vmag', 'Rmag', 
                'Imag', 'Jmag', 'Hmag', 'Kmag', 'dist', 'BV', 'MV', 'BC', 'L', 
                'coords', 'pmra', 'pmdec', 'rv', 'Binary_Cut']
        # bring Star Catalog values to top level of Target List
        for att in atts:
            if type(getattr(self.StarCatalog, att)) == np.ma.core.MaskedArray:
                setattr(self, att, getattr(self.StarCatalog, att).filled(fill_value=float('nan')))
            else:
                setattr(self, att, getattr(self.StarCatalog, att))
            
        # filter out nan attribute values from Star Catalog
        self.nan_filter(atts)
        # populate completion values
        self.comp0 = self.Completeness.target_completeness(self)
        # include completeness now that it is set
        atts.append('comp0')
        # populate maximum integration time
        self.maxintTime = self.OpticalSystem.calc_maxintTime(self)
        # include integration time now that it is set
        atts.append('maxintTime')
        # calculate 'true' and 'approximate' stellar masses
        self.stellar_mass()
        # if additional filters are desired, need atts fully populated
        atts.append('MsEst')
        atts.append('MsTrue')
        
        # filter out binary stars
        self.binary_filter(atts)
        # filter out systems with planets within the IWA
        self.outside_IWA_filter(atts)
        # filter out systems where maximum delta mag is not in allowable orbital range
        self.max_dmag_filter(atts)
        # filter out systems where integration time is longer than maximum time
        self.int_cutoff_filter(atts)
        # filter out systems which do not reach the completeness threshold
        self.completeness_filter(atts)
        
        # have target list, no need for catalog now
        if not keepStarCatalog:
            del self.StarCatalog
        
    def __str__(self):
        """String representation of the Target List object
        
        When the command 'print' is used on the Target List object, this method
        will return the values contained in the object"""

        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Target List class object attributes'
                
    def nan_filter(self, atts):
        """Populates Target List and filters out values which are nan
        
        Args:
            atts (list):
                list of StarCatalog class object attribute names
        
        """
        
        # filter out nan values in numerical attributes
        for att in atts:
            if getattr(self, att).shape[0] == 0:
                pass
            elif type(getattr(self, att)[0]) == str:
                i = np.where(getattr(self, att) != float('nan'))
                self.revise_lists(atts, i)
            elif type(getattr(self, att)[0]) != np.unicode_ and type(getattr(self, att)[0]) != np.bool_:
                if att == 'coords':
                    i1 = np.where(~np.isnan(self.coords.ra.value))
                    i2 = np.where(~np.isnan(self.coords.dec.value))
                    if (i1[0]==i2[0]).all():
                        i = i1
                    elif i1[0] < i2[0]:
                        i = i1
                    else:
                        i = i2
                else:
                    i = np.where(~np.isnan(getattr(self, att)))

                self.revise_lists(atts, i)
                
    def binary_filter(self, atts):
        """Removes stars which have attribute Binary_Cut == True
        
        Args:
            atts (list):
                list of StarCatalog class object attribute names
        
        """
        
        # indices from Target List to keep
        i = np.where(self.Binary_Cut == False)
        
        self.revise_lists(atts, i)
        
    def life_expectancy_filter(self, atts):
        """Removes stars from Target List which have BV < 0.3
        
        Args:
            atts (list):
                list of StarCatalog class object attribute names
        
        """
        
        # indices from Target List to keep
        i = np.where(self.BV > 0.3)
        
        self.revise_lists(atts, i)
        
    def main_sequence_filter(self, atts):
        """Removes stars from Target List which are not main sequence
        
        Args:
            atts (list):
                list of StarCatalog class object attribute names
        
        """
        
        # indices from Target List to keep
        i1 = np.where(np.logical_and(self.BV < 0.74, self.MV < (6*self.BV + 1.8)))
        i2 = np.where(np.logical_and(self.BV >= 0.74, np.logical_and(self.BV < 1.37, self.MV < (4.3*self.BV+3.05))))
        i3 = np.where(np.logical_and(self.BV >= 1.37, self.MV < (18*self.BV - 15.7)))
        i4 = np.where(np.logical_and(self.BV < 0.87, self.MV > (-8*(self.BV-1.35)**2+7.01)))
        i5 = np.where(np.logical_and(self.BV >= 0.87, np.logical_and(self.BV < 1.45, self.MV < (5*self.BV+0.81))))
        i6 = np.where(np.logical_and(self.BV >= 1.45, self.MV > (18*self.BV-18.04)))
        
        ia = np.append(i1, i2)
        ia = np.append(ia, i3)
        ia = np.unique(ia)
        
        ib = np.append(i4, i5)
        ib = np.append(ib, i6)
        ib = np.unique(ib)
        
        i = np.intersect1d(ia,ib)
       
        self.revise_lists(atts, i)
        
    def fgk_filter(self, atts):
        """Includes only F, G, K spectral type stars in Target List
        
        Args:
            atts (list):
                list of StarCatalog class object attribute names
        
        """
        
        # indices from target list to keep
        iF = np.where(np.core.defchararray.startswith(self.Spec, 'F'))
        iG = np.where(np.core.defchararray.startswith(self.Spec, 'G'))
        iK = np.where(np.core.defchararray.startswith(self.Spec, 'K'))
        
        i = np.append(iF, iG)
        i = np.append(i, iK)
        i = np.unique(i)
        
        self.revise_lists(atts, i)
    
    def vis_mag_filter(self, atts, Vmagcrit):
        """Includes stars which are below the maximum apparent visual magnitude
        
        Args:
            atts (list):
                list of StarCatalog class object attribute names
            Vmagcrit (float):
                maximum apparent visual magnitude
        
        """
        
        # indices from Target List to keep
        i = np.where(self.Vmag < Vmagcrit)
        
        self.revise_lists(atts, i)

    def outside_IWA_filter(self, atts):
        """Includes stars with planets with orbits outside of the IWA 
        
        This method uses the following inherited class objects:
            self.OpticalSystem:
                OpticalSystem class object
            self.PlanetPopulation:
                PlanetPopulation class object
                
        Args:
            atts (list):
                list of StarCatalog class object attribute names
        
        """
        
        if self.PlanetPopulation.scaleOrbits:
            i = np.where(np.max(self.PlanetPopulation.rrange) > (np.tan(self.OpticalSystem.IWA)*self.dist/np.sqrt(self.L))*u.pc)
        else:
            i = np.where(np.max(self.PlanetPopulation.rrange) > (np.tan(self.OpticalSystem.IWA)*self.dist*u.pc))
   
        self.revise_lists(atts, i[0])
        
    def int_cutoff_filter(self, atts):
        """Includes stars if calculated integration time is less than cutoff
        
        This method uses the following inherited class object:
            self.rules:
                Rules class object
        
        Args:
            atts (list):
                list of StarCatalog class object attribute names
        
        """
        
        i = np.where(self.maxintTime <= self.OpticalSystem.intCutoff)

        self.revise_lists(atts, i[0])
        
    def max_dmag_filter(self, atts):
        """Includes stars if maximum delta mag is in the allowed orbital range
        
        This method uses the following inherited class objects:
            self.PlanetPopulation:
                PlanetPopulation class object
            self.OpticalSystem:
                OpticalSystem class object
                
        Args:
            atts (list):
                list of StarCatalog class object attribute names
                
        """
        
        betastar = 1.10472881476178 # radians
        
        rhats = np.tan(self.OpticalSystem.IWA)*self.dist*u.pc/np.sin(betastar)
        
        if self.PlanetPopulation.scaleOrbits:
            rhats = rhats/np.sqrt(self.L)
        
        # out of range rhats
        below = np.where(rhats < np.min(self.PlanetPopulation.rrange))
        above = np.where(rhats > np.max(self.PlanetPopulation.rrange))

        # s and beta arrays
        ss = np.tan(self.OpticalSystem.IWA)*self.dist*u.pc
        if self.PlanetPopulation.scaleOrbits:
            ss = ss/np.sqrt(self.L)
        
        betas = np.zeros((len(ss))) + betastar
        
        # fix out of range values
        ss[below] = np.min(self.PlanetPopulation.rrange)*np.sin(betastar)
        if self.PlanetPopulation.scaleOrbits:
            betas[above] = np.arcsin((np.tan(self.OpticalSystem.IWA)*self.dist[above]*u.pc/np.sqrt(self.L[above])/np.max(self.PlanetPopulation.rrange)).decompose())
        else:
            betas[above] = np.arcsin((np.tan(self.OpticalSystem.IWA)*self.dist[above]*u.pc/np.max(self.PlanetPopulation.rrange)).decompose())                
        
        # calculate delta mag
        Phis = (np.sin(betas)+(np.pi - betas)*np.cos(betas))/np.pi
        t1 = np.max(self.PlanetPopulation.Rrange).to(u.AU).value**2*np.max(self.PlanetPopulation.prange)      
        cdMag = np.where(-2.5*np.log10(t1*Phis*np.sin(betas)**2/ss.to(u.AU).value**2) < self.OpticalSystem.dMagLim)
        
        self.revise_lists(atts, cdMag)
        
    def completeness_filter(self, atts):
        """Includes stars if completeness is larger than the minimum value
        
        This method uses the following inherited class object:
            self.Completeness:
                Completeness class object
                
        Args:
            atts (list):
                list of StarCatalog class object attribute names
        
        """
        
        i = np.where(self.comp0 > self.Completeness.minComp)

        self.revise_lists(atts, i)        
        
    def revise_lists(self, atts, ind):
        """Replaces Target List catalog attributes with filtered values
        
        Args:
            atts (list):
                list of StarCatalog class object attribute names
            ind (ndarray):
                1D numpy ndarray of indices to keep
        
        """
        
        for att in atts:
            if att == 'coords':
                self.coords = SkyCoord(ra=self.coords.ra[ind].value, dec=self.coords.dec[ind].value, unit='deg')
            else:
                if getattr(self, att).size != 0:
                    setattr(self, att, getattr(self, att)[ind])
    
    def stellar_mass(self):
        """Populates target list with 'true' and 'approximate' stellar masses
        
        This method calculates stellar mass via the formula relating absolute V
        magnitude and stellar mass.  The values are in terms of M_sun.
        
        """
        
        # 'approximate' stellar mass
        self.MsEst = (10.**(0.002456*self.MV**2 - 0.09711*self.MV + 0.4365))
        # normally distributed 'error'
        err = (np.random.random(len(self.MV))*2. - 1.)*0.07
        self.MsTrue = (1. + err)*self.MsEst
        
