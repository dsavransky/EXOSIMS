# -*- coding: utf-8 -*-
from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.deltaMag import deltaMag
import numpy as np
import numbers
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.io
import re
import scipy.interpolate
import os.path
import inspect


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
        (StarCatalog values)
            Mission specific filtered star catalog values from StarCatalog class object:
            Name, Spec, Umag, Bmag, Vmag, Rmag, Imag, Jmag, Hmag, Kmag, BV, MV,
            BC, L, Binary_Cut, dist, parx, coords, pmra, pmdec, rv
        StarCatalog (StarCatalog module):
            StarCatalog class object (only retained if keepStarCatalog is True)
        PlanetPopulation (PlanetPopulation module):
            PlanetPopulation class object
        PlanetPhysicalModel (PlanetPhysicalModel module):
            PlanetPhysicalModel class object
        OpticalSystem (OpticalSystem module):
            OpticalSystem class object
        ZodiacalLight (ZodiacalLight module):
            ZodiacalLight class object
        BackgroundSources (BackgroundSources module):
            BackgroundSources class object
        PostProcessing (PostProcessing module):
            PostProcessing class object
        Completeness (Completeness module):
            Completeness class object
        tint0 (astropy Quantity array):
            Minimum integration time values for each target star in units of day
        comp0 (ndarray):
            Initial completeness value for each target star
        MsEst (float ndarray):
            'approximate' stellar mass in units of solar mass
        MsTrue (float ndarray):
            'true' stellar mass in units of solar mass
        nStars (integer):
            Number of target stars
        staticStars (boolean):
            Boolean used to force static target positions set at mission start time
        keepStarCatalog (boolean):
            Boolean used to avoid deleting StarCatalog after TargetList was built
        fillPhotometry (boolean):
            Defaults False.  If True, attempts to fill in missing target photometric 
            values using interpolants of tabulated values for the stellar type.
    
    """

    _modtype = 'TargetList'
    
    def __init__(self, missionStart=60634, staticStars=True, 
            keepStarCatalog=False, fillPhotometry=False, explainFiltering=False, **specs):
       
        #start the outspec
        self._outspec = {}

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))
        
        # validate TargetList inputs
        assert isinstance(staticStars, bool), "staticStars must be a boolean."
        assert isinstance(keepStarCatalog, bool), "keepStarCatalog must be a boolean."
        assert isinstance(fillPhotometry, bool), "fillPhotometry must be a boolean."
        assert isinstance(explainFiltering, bool), "explainFiltering must be a boolean."
        self.staticStars = bool(staticStars)
        self.keepStarCatalog = bool(keepStarCatalog)
        self.fillPhotometry = bool(fillPhotometry)
        self.explainFiltering = bool(explainFiltering)
        
        # check if KnownRVPlanetsTargetList is using KnownRVPlanets
        if specs['modules']['TargetList'] == 'KnownRVPlanetsTargetList':
            assert specs['modules']['PlanetPopulation'] == 'KnownRVPlanets', \
            'KnownRVPlanetsTargetList must use KnownRVPlanets'
        else:
            assert specs['modules']['PlanetPopulation'] != 'KnownRVPlanets', \
            'This TargetList cannot use KnownRVPlanets'
        
        # populate outspec
        for att in self.__dict__.keys():
            if att not in ['vprint','_outspec']:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat, u.Quantity) else dat
        
        # get desired module names (specific or prototype) and instantiate objects
        self.StarCatalog = get_module(specs['modules']['StarCatalog'],
                'StarCatalog')(**specs)
        self.OpticalSystem = get_module(specs['modules']['OpticalSystem'],
                'OpticalSystem')(**specs)
        self.ZodiacalLight = get_module(specs['modules']['ZodiacalLight'],
                'ZodiacalLight')(**specs)
        self.PostProcessing = get_module(specs['modules']['PostProcessing'],
                'PostProcessing')(**specs)
        self.Completeness = get_module(specs['modules']['Completeness'],
                'Completeness')(**specs)
        
        # bring inherited class objects to top level of Simulated Universe
        self.BackgroundSources = self.PostProcessing.BackgroundSources

        #if specs contains a completeness_spec then we are going to generate separate instances
        #of planet population and planet physical model for completeness and for the rest of the sim
        if specs.has_key('completeness_specs'):
            self.PlanetPopulation = get_module(specs['modules']['PlanetPopulation'],'PlanetPopulation')(**specs)
            self.PlanetPhysicalModel = self.PlanetPopulation.PlanetPhysicalModel
        else:
            self.PlanetPopulation = self.Completeness.PlanetPopulation
            self.PlanetPhysicalModel = self.Completeness.PlanetPhysicalModel
        
        # list of possible Star Catalog attributes
        self.catalog_atts = ['Name', 'Spec', 'parx', 'Umag', 'Bmag', 'Vmag', 'Rmag', 
                'Imag', 'Jmag', 'Hmag', 'Kmag', 'dist', 'BV', 'MV', 'BC', 'L', 
                'coords', 'pmra', 'pmdec', 'rv', 'Binary_Cut']
        
        # now populate and filter the list
        self.populate_target_list(**specs)
        # generate any completeness update data needed
        self.Completeness.gen_update(self)
        self.filter_target_list(**specs)
        # have target list, no need for catalog now
        if not keepStarCatalog:
            self.StarCatalog = specs['modules']['StarCatalog']
        # add nStars to outspec
        self._outspec['nStars'] = self.nStars
        
        # if staticStars is True, the star coordinates are taken at mission start, 
        # and are not propagated during the mission
        self.starprop_static = None
        if self.staticStars is True:
            allInds = np.arange(self.nStars,dtype=int)
            missionStart = Time(float(missionStart), format='mjd', scale='tai')
            self.starprop_static = lambda sInds, currentTime, eclip=False, \
                    c1=self.starprop(allInds, missionStart, eclip=False), \
                    c2=self.starprop(allInds, missionStart, eclip=True): \
                    c1[sInds] if eclip==False else c2[sInds]

    def __str__(self):
        """String representation of the Target List object
        
        When the command 'print' is used on the Target List object, this method
        will return the values contained in the object
        
        """
        
        for att in self.__dict__.keys():
            print('%s: %r' % (att, getattr(self, att)))
        
        return 'Target List class object attributes'

    def populate_target_list(self,**specs):
        """ This function is actually responsible for populating values from the star 
        catalog (or any other source) into the target list attributes.

        The prototype implementation does the following:
        
        Copy directly from star catalog and remove stars with any NaN attributes
        Calculate completeness and max integration time, and generates stellar masses.
        
        """
        
        SC = self.StarCatalog
        OS = self.OpticalSystem
        ZL = self.ZodiacalLight
        Comp = self.Completeness
        
        # bring Star Catalog values to top level of Target List
        for att in self.catalog_atts:
            if type(getattr(SC, att)) == np.ma.core.MaskedArray:
                setattr(self, att, getattr(SC, att).filled(fill_value=float('nan')))
            else:
                setattr(self, att, getattr(SC, att))
        
        # number of target stars
        self.nStars = len(self.Name)
        if self.explainFiltering:
            print("%d targets imported from star catalog."%self.nStars)
    
        if self.fillPhotometry:
            self.fillPhotometryVals()

        # filter out nan attribute values from Star Catalog
        self.nan_filter()
        if self.explainFiltering:
            print("%d targets remain after nan filtering."%self.nStars)

        # populate completeness values
        self.comp0 = Comp.target_completeness(self)
        # populate minimum integration time values
        self.tint0 = OS.calc_minintTime(self)
        # calculate 'true' and 'approximate' stellar masses
        self.stellar_mass()
        
        # include new attributes to the target list catalog attributes
        self.catalog_atts.append('comp0')
        self.catalog_atts.append('tint0')
    
    def fillPhotometryVals(self):
        """
        This routine attempts to fill in missing photometric values, including
        the luminosity, absolute magnitude, V band bolometric correction, and the 
        apparent VBHJK magnitudes by interpolating values from a table of standard
        stars by spectral type.  

        The data is from:
        "A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence"
        http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
        Eric Mamajek (JPL/Caltech, University of Rochester) 
        Version 2017.09.06

        """
        
        #Looking for file EEM_dwarf_UBVIJHK_colors_Teff.txt in the TargetList folder
        filename = 'EEM_dwarf_UBVIJHK_colors_Teff.txt'
        classpath = os.path.split(inspect.getfile(self.__class__))[0]
        classpath = os.path.normpath(os.path.join(classpath, '..', 
                'TargetList'))
        datapath = os.path.join(classpath, filename)
        assert os.path.isfile(datapath),'Could not locate %s in TargetList directory.'%filename

        data = astropy.io.ascii.read(datapath,fill_values=[('...',np.nan),('....',np.nan),('.....',np.nan)])

        specregex = re.compile('([OBAFGKMLTY])(\d*\.\d+|\d+)V')
        specregex2 = re.compile('([OBAFGKMLTY])(\d*\.\d+|\d+).*')

        MK = []
        MKn = []
        for s in data['SpT'].data:
            m = specregex.match(s)
            MK.append(m.groups()[0])
            MKn.append(m.groups()[1])
        MK = np.array(MK)
        MKn = np.array(MKn)

        #create dicts of interpolants
        Mvi = {}
        BmVi = {}
        logLi = {}
        BCi = {}
        VmKi = {}
        HmKi = {}
        JmHi = {}
        for l in 'OBAFGKM':
            Mvi[l] = scipy.interpolate.interp1d(MKn[MK==l].astype(float),data['Mv'][MK==l].data.astype(float),bounds_error=False,fill_value='extrapolate')
            BmVi[l] = scipy.interpolate.interp1d(MKn[MK==l].astype(float),data['B-V'][MK==l].data.astype(float),bounds_error=False,fill_value='extrapolate')
            logLi[l] = scipy.interpolate.interp1d(MKn[MK==l].astype(float),data['logL'][MK==l].data.astype(float),bounds_error=False,fill_value='extrapolate')
            VmKi[l] = scipy.interpolate.interp1d(MKn[MK==l].astype(float),data['V-Ks'][MK==l].data.astype(float),bounds_error=False,fill_value='extrapolate')
            HmKi[l] = scipy.interpolate.interp1d(MKn[MK==l].astype(float),data['H-K'][MK==l].data.astype(float),bounds_error=False,fill_value='extrapolate')
            JmHi[l] = scipy.interpolate.interp1d(MKn[MK==l].astype(float),data['J-H'][MK==l].data.astype(float),bounds_error=False,fill_value='extrapolate')
            BCi[l] = scipy.interpolate.interp1d(MKn[MK==l].astype(float),data['BCv'][MK==l].data.astype(float),bounds_error=False,fill_value='extrapolate')


        #first try to fill in missing Vmags
        if np.any(np.isnan(self.Vmag)):
            inds = np.where(np.isnan(self.Vmag))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    self.Vmag[i] = Mvi[m.groups()[0]](m.groups()[1])
                    self.MV[i] = self.Vmag[i] - 5*(np.log10(self.dist[i].to('pc').value) - 1)

        #next, try to fill in any missing B mags
        if np.any(np.isnan(self.Bmag)):
            inds = np.where(np.isnan(self.Bmag))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    self.BV[i] = BmVi[m.groups()[0]](m.groups()[1])
                    self.Bmag[i] = self.BV[i] + self.Vmag[i]

        #next fix any missing luminosities
        if np.any(np.isnan(self.L)):
            inds = np.where(np.isnan(self.L))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    self.L[i] = 10.0**logLi[m.groups()[0]](m.groups()[1])

        #and bolometric corrections
        if np.any(np.isnan(self.BC)):
            inds = np.where(np.isnan(self.BC))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    self.BC[i] = BCi[m.groups()[0]](m.groups()[1])


        #next fill in K mags
        if np.any(np.isnan(self.Kmag)):
            inds = np.where(np.isnan(self.Kmag))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    VmK = VmKi[m.groups()[0]](m.groups()[1])
                    self.Kmag[i] = self.Vmag[i] - VmK

        #next fill in H mags
        if np.any(np.isnan(self.Hmag)):
            inds = np.where(np.isnan(self.Hmag))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    HmK = HmKi[m.groups()[0]](m.groups()[1])
                    self.Hmag[i] = self.Kmag[i] + HmK

        #next fill in J mags
        if np.any(np.isnan(self.Jmag)):
            inds = np.where(np.isnan(self.Jmag))[0]
            for i in inds:
                m = specregex2.match(self.Spec[i])
                if m:
                    JmH = JmHi[m.groups()[0]](m.groups()[1])
                    self.Jmag[i] = self.Hmag[i] + JmH


    
    def filter_target_list(self, **specs):
        """This function is responsible for filtering by any required metrics.
        
        The prototype implementation removes the following stars:
            * Stars with NAN values in their parameters
            * Binary stars
            * Systems with planets inside the OpticalSystem fundamental IWA
            * Systems where minimum integration time is longer than OpticalSystem cutoff
            * Systems not meeting the Completeness threshold
        
        Additional filters can be provided in specific TargetList implementations.
        
        """
        
        # filter out binary stars
        self.binary_filter()
        if self.explainFiltering:
            print("%d targets remain after binary filter."%self.nStars)
        
        # filter out systems with planets within the IWA
        self.outside_IWA_filter()
        if self.explainFiltering:
            print("%d targets remain after IWA filter."%self.nStars)

        # filter out systems where minimum integration time is longer than cutoff
        self.int_cutoff_filter()
        if self.explainFiltering:
            print("%d targets remain after integration time cutoff filter."%self.nStars)
        
        # filter out systems which do not reach the completeness threshold
        self.completeness_filter()
        if self.explainFiltering:
            print("%d targets remain after completeness filter."%self.nStars)


    def nan_filter(self):
        """Populates Target List and filters out values which are nan
        
        """
        
        # filter out nan values in numerical attributes
        for att in self.catalog_atts:
            if getattr(self, att).shape[0] == 0:
                pass
            elif type(getattr(self, att)[0]) == str:
                # FIXME: intent here unclear: 
                #   note float('nan') is an IEEE NaN, getattr(.) is a str, and != on NaNs is special
                i = np.where(getattr(self, att) != float('nan'))[0]
                self.revise_lists(i)
            # exclude non-numerical types
            elif type(getattr(self, att)[0]) not in (np.unicode_, np.string_, np.bool_):
                if att == 'coords':
                    i1 = np.where(~np.isnan(self.coords.ra.to('deg').value))[0]
                    i2 = np.where(~np.isnan(self.coords.dec.to('deg').value))[0]
                    i = np.intersect1d(i1,i2)
                else:
                    i = np.where(~np.isnan(getattr(self, att)))[0]
                self.revise_lists(i)

    def binary_filter(self):
        """Removes stars which have attribute Binary_Cut == True
        
        """
        
        i = np.where(self.Binary_Cut == False)[0]
        self.revise_lists(i)

    def life_expectancy_filter(self):
        """Removes stars from Target List which have BV < 0.3
        
        """
        
        i = np.where(self.BV > 0.3)[0]
        self.revise_lists(i)

    def main_sequence_filter(self):
        """Removes stars from Target List which are not main sequence
        
        """
        
        # indices from Target List to keep
        i1 = np.where((self.BV < 0.74) & (self.MV < 6*self.BV + 1.8))[0]
        i2 = np.where((self.BV >= 0.74) & (self.BV < 1.37) & \
                (self.MV < 4.3*self.BV + 3.05))[0]
        i3 = np.where((self.BV >= 1.37) & (self.MV < 18*self.BV - 15.7))[0]
        i4 = np.where((self.BV < 0.87) & (self.MV > -8*(self.BV - 1.35)**2 + 7.01))[0]
        i5 = np.where((self.BV >= 0.87) & (self.BV < 1.45) & \
                (self.MV < 5*self.BV + 0.81))[0]
        i6 = np.where((self.BV >= 1.45) & (self.MV > 18*self.BV - 18.04))[0]
        ia = np.append(np.append(i1, i2), i3)
        ib = np.append(np.append(i4, i5), i6)
        i = np.intersect1d(np.unique(ia), np.unique(ib))
        self.revise_lists(i)

    def fgk_filter(self):
        """Includes only F, G, K spectral type stars in Target List
        
        """
        
        spec = np.array(map(str, self.Spec))
        iF = np.where(np.core.defchararray.startswith(spec, 'F'))[0]
        iG = np.where(np.core.defchararray.startswith(spec, 'G'))[0]
        iK = np.where(np.core.defchararray.startswith(spec, 'K'))[0]
        i = np.append(np.append(iF, iG), iK)
        i = np.unique(i)
        self.revise_lists(i)

    def vis_mag_filter(self, Vmagcrit):
        """Includes stars which are below the maximum apparent visual magnitude
        
        Args:
            Vmagcrit (float):
                maximum apparent visual magnitude
        
        """
        
        i = np.where(self.Vmag < Vmagcrit)[0]
        self.revise_lists(i)

    def outside_IWA_filter(self):
        """Includes stars with planets with orbits outside of the IWA 
        
        """
        
        PPop = self.PlanetPopulation
        OS = self.OpticalSystem
        
        s = np.tan(OS.IWA)*self.dist
        L = np.sqrt(self.L) if PPop.scaleOrbits else 1.
        i = np.where(s < L*np.max(PPop.rrange))[0]
        self.revise_lists(i)

    def max_dmag_filter(self):
        """Includes stars if maximum delta mag is in the allowed orbital range
        
        Removed from prototype filters. Prototype is already calling the 
        int_cutoff_filter with OS.dMag0 and the completeness_filter with Comp.dMagLim
        
        """
        
        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        Comp = self.Completeness
        
        # s and beta arrays
        s = np.tan(self.OpticalSystem.WA0)*self.dist
        if PPop.scaleOrbits:
            s /= np.sqrt(self.L)
        beta = np.array([1.10472881476178]*len(s))*u.rad
        
        # fix out of range values
        below = np.where(s < np.min(PPop.rrange)*np.sin(beta))[0]
        above = np.where(s > np.max(PPop.rrange)*np.sin(beta))[0]
        s[below] = np.sin(beta[below])*np.min(PPop.rrange)
        beta[above] = np.arcsin(s[above]/np.max(PPop.rrange))
        
        # calculate delta mag
        p = np.max(PPop.prange)
        Rp = np.max(PPop.Rprange)
        d = s/np.sin(beta)
        Phi = PPMod.calc_Phi(beta)
        i = np.where(deltaMag(p, Rp, d, Phi) < Comp.dMagLim)[0]
        self.revise_lists(i)

    def int_cutoff_filter(self):
        """Includes stars if calculated minimum integration time is less than cutoff
        
        """
        
        i = np.where(self.tint0 < self.OpticalSystem.intCutoff)[0]
        self.revise_lists(i)

    def completeness_filter(self):
        """Includes stars if completeness is larger than the minimum value
        
        """
        
        i = np.where(self.comp0 >= self.Completeness.minComp)[0]
        self.revise_lists(i)

    def revise_lists(self, sInds):
        """Replaces Target List catalog attributes with filtered values, 
        and updates the number of target stars.
        
        Args:
            sInds (integer ndarray):
                Integer indices of the stars of interest
        
        """
       
        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        
        if len(sInds) == 0:
            raise IndexError("Target list filtered to empty.")
        
        for att in self.catalog_atts:
            if att == 'coords':
                ra = self.coords.ra[sInds].to('deg')
                dec = self.coords.dec[sInds].to('deg')
                self.coords = SkyCoord(ra, dec, self.dist.to('pc'))
            else:
                if getattr(self, att).size != 0:
                    setattr(self, att, getattr(self, att)[sInds])
        try:
            self.Completeness.revise_updates(sInds)
        except AttributeError:
            pass
        self.nStars = len(sInds)
        assert self.nStars, "Target list is empty: nStars = %r"%self.nStars

    def stellar_mass(self):
        """Populates target list with 'true' and 'approximate' stellar masses
        
        This method calculates stellar mass via the formula relating absolute V
        magnitude and stellar mass.  The values are in units of solar mass.
        
        """
        
        # 'approximate' stellar mass
        self.MsEst = (10.**(0.002456*self.MV**2 - 0.09711*self.MV + 0.4365))*u.solMass
        # normally distributed 'error'
        err = (np.random.random(len(self.MV))*2. - 1.)*0.07
        self.MsTrue = (1. + err)*self.MsEst
        
        # if additional filters are desired, need self.catalog_atts fully populated
        self.catalog_atts.append('MsEst')
        self.catalog_atts.append('MsTrue')

    def starprop(self, sInds, currentTime, eclip=False):
        """Finds target star positions vector in heliocentric equatorial (default)
        or ecliptic frame for current time (MJD).
        
        This method uses ICRS coordinates which is approximately the same as 
        equatorial coordinates. 
        
        Args:
            sInds (integer ndarray):
                Integer indices of the stars of interest
            currentTime (astropy Time):
                Current absolute mission time in MJD
            eclip (boolean):
                Boolean used to switch to heliocentric ecliptic frame. Defaults to 
                False, corresponding to heliocentric equatorial frame.
        
        Returns:
            r_targ (astropy Quantity nx3 array): 
                Target star positions vector in heliocentric equatorial (default)
                or ecliptic frame in units of pc
        
        Note: Use eclip=True to get ecliptic coordinates.
        
        """
        
        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        
        # if the starprop_static method was created (staticStars is True), then use it
        if self.starprop_static is not None:
            return self.starprop_static(sInds, currentTime, eclip)
        
        # get all array sizes
        nStars = sInds.size
        nTimes = currentTime.size
        assert nStars==1 or nTimes==1 or nTimes==nStars, \
                "If multiple times and targets, currentTime and sInds sizes must match"
        
        # target star ICRS coordinates
        coord_old = self.coords[sInds]
        # right ascension and declination
        ra = coord_old.ra
        dec = coord_old.dec
        # directions
        p0 = np.array([-np.sin(ra), np.cos(ra), np.zeros(sInds.size)])
        q0 = np.array([-np.sin(dec)*np.cos(ra), -np.sin(dec)*np.sin(ra), np.cos(dec)])
        r0 = coord_old.cartesian.xyz/coord_old.distance
        # proper motion vector
        mu0 = p0*self.pmra[sInds] + q0*self.pmdec[sInds]
        # space velocity vector
        v = mu0/self.parx[sInds]*u.AU + r0*self.rv[sInds]
        # set J2000 epoch
        j2000 = Time(2000., format='jyear')
        # target star positions vector in heliocentric equatorial frame
        dr = v*(currentTime.mjd - j2000.mjd)*u.day
        r_targ = (coord_old.cartesian.xyz + dr).T.to('pc')
        
        if eclip:
            # transform to heliocentric true ecliptic frame
            coord_new = SkyCoord(r_targ[:,0], r_targ[:,1], r_targ[:,2], 
                    representation='cartesian')
            r_targ = coord_new.heliocentrictrueecliptic.cartesian.xyz.T.to('pc')
        
        return r_targ

    def starMag(self, sInds, lam):
        """Calculates star visual magnitudes with B-V color using empirical fit 
        to data from Pecaut and Mamajek (2013, Appendix C).
        The expression for flux is accurate to about 7%, in the range of validity 
        400 nm < λ < 1000 nm (Traub et al. 2016).
        
        Args:
            sInds (integer ndarray):
                Indices of the stars of interest
            lam (astropy Quantity):
                Wavelength in units of nm
        
        Returns:
            mV (float ndarray):
                Star visual magnitudes with B-V color
        
        """
        
        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        
        Vmag = self.Vmag[sInds]
        BV = self.BV[sInds]
        
        lam_um = lam.to('um').value
        if lam_um < .550:
            b = 2.20
        else:
            b = 1.54
        mV = Vmag + b*BV*(1/lam_um - 1.818)
        
        return mV

    def stellarTeff(self, sInds):
        """Calculate the effective stellar temperature based on B-V color.
        
        This method uses the empirical fit from Ballesteros (2012) doi:10.1209/0295-5075/97/34008
        
        Args:
            sInds (integer ndarray):
                Indices of the stars of interest
        
        Returns:
            Teff (Quantity array):
                Stellar effective temperatures in degrees K
        
        """
        
        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        
        Teff = 4600.0*u.K * (1.0/(0.92*self.BV[sInds] + 1.7) + 1.0/(0.92*self.BV[sInds] + 0.62))
        
        return Teff
