# -*- coding: utf-8 -*-
from astropy import units as u
import numpy as np
import os.path
import astropy.io.fits as fits
import numbers
from operator import itemgetter
import scipy.interpolate
import scipy.optimize

class OpticalSystem(object):
    """Optical System class template
    
    This class contains all variables and methods necessary to perform
    Optical System Definition Module calculations in exoplanet mission 
    simulation.
    
    Args:
        \*\*specs:
            User specified values.
            
    Attributes:
        lam (Quantity):
            detection wavelength (default units of nm)
        deltaLam (Quantity):
            bandpass (default units of nm)
        obscurFac (float):
            Obscuration factor due to secondary mirror and spiders        
        shapeFac (float):
            shape factor (also known as fill factor) where 
            :math:`shapeFac * pupilDiam^2 * (1-obscurFac) = pupilArea`
        pupilDiam (Quantity):
            entrance pupil diameter (default units of m)
        pupilArea (Quantity):
            entrance pupil area (default units of m\ :sup:`2`)
        SNchar (float):
            Signal to Noise Ratio for characterization
        IWA (Quantity):
            Fundamental Inner Working Angle (default units of arcsecond)
        OWA (Quantity):
            Fundamental Outer Working Angle (default units of arcsecond)
        dMagLim (float):
            Fundamental limiting delta magnitude. 
        telescopeKeepout (float):
            telescope keepout angle in degrees
        attenuation (float):
            non-coronagraph attenuation (throughput)
        specLam (Quantity):
            spectral wavelength of interest (default units of nm)
            Note:  probably belongs in postprocessing or rules.
        intCutoff (Quantity):
            integration cutoff (default units of day)
        Npix (float):
            number of noise pixels
        Ndark (float):
            number of dark frames used
        starlightSuppressionSystems (list of dicts):
            All starlight suppression system attributes (variable)
        scienceInstruments (list of dicts):
            All science instrument attributes (variable)

    Common Starlight Suppression System Attributes:

        haveOcculter (bool):
            boolean signifying if the system has an occulter
        PSF (callable(lam, alpha)):
            point spread function - 2D ndarray of values, normalized to 1 at
            the core (must be callable - can be lambda, function,
            scipy.interpolate.interp2d object, etc.) with inputs wavelength
            (astropy Quantity) and angular separation (astropy Quantity). Note
            normalization means that all throughput effects must be contained
            in the throughput attribute.
        PSFSampling (Quantity):
            Sampling of PSF in arcsec/pixel (default units of arcsec)
        throughput (callable(lam, alpha)):
            optical system throughput (must be callable - can be lambda,
            function, scipy.interpolate.interp2d object, etc.) with inputs
            wavelength (astropy Quantity) and angular separation (astropy 
            Quantity).
        contrast (callable(lam, alpha)):
            optical system contrast curve (must be callable - can be lambda,
            function, scipy.interpolate.interp2d object, etc.) with inputs
            wavelength (astropy Quantity) and angular separation (astropy 
            Quantity).

    Common Science Instrument Attributes:
        QE (callable(lam)):
            detector quantum efficiency (must be callable - can be lambda,
            function, scipy.interpolate.interp2d object, etc.) with inputs 
            wavelength (astropy Quantity).  
     
    """

    _modtype = 'OpticalSystem'
    _outspec = {}

    def __init__(self, lam=500., deltaLam=100., obscurFac=0.2, shapeFac=np.pi/4.,\
            pupilDiam=4., SNchar=11., telescopeKeepout=45., attenuation=0.57,\
            specLam=760., specBW=0.18, intCutoff=50., Npix=14.3, Ndark=10.,\
            IWA=None, OWA=None, dMagLim=None, starlightSuppressionSystems=None,\
            scienceInstruments=None, **specs):

        #must have a starlight suppression system and science instrument defined
        if not starlightSuppressionSystems:
            raise ValueError("No starlight suppression systems defined.")
        if not scienceInstruments:
            raise ValueError("No science isntrument defined.")

        #load all values with defaults
        self.lam = float(lam)*u.nm              # detection wavelength (nm) 
        self.deltaLam = float(deltaLam)*u.nm    # detection bandwidth (nm) 
        self.obscurFac = float(obscurFac)       # obscuration factor
        self.shapeFac = float(shapeFac)         # shape factor
        self.pupilDiam = float(pupilDiam)*u.m   # entrance pupil diameter
        self.pupilArea = self.shapeFac*self.pupilDiam**2.*(1-self.obscurFac)\
                                                # entrance pupil area
        self.SNchar = float(SNchar)             # SNR for characterization
        self.telescopeKeepout = float(telescopeKeepout)*u.deg\
                                                # keepout angle in degrees
        self.attenuation = float(attenuation)   # non-coronagraph attenuation factor
        self.specLam = float(specLam)*u.nm      # spectral wavelength of interest
        self.specBW = float(specBW)             # spectral bandwidth fraction 
        self.specDeltaLam = self.specLam*specBW # spectral bandwidth
        self.intCutoff = float(intCutoff)*u.day # integration time cutoff 
        self.Npix = float(Npix)                 # number of noise pixels
        self.Ndark = float(Ndark)               # number of dark frames used
        
        #populate all scalar atributes to outspec
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key],u.Quantity):
                self._outspec[key] = self.__dict__[key].value
            else:
                self._outspec[key] = self.__dict__[key]

        #now loop through all starlight suppression systems
        systIWAs = []
        systOWAs = []
        systdMagLims = []
        self._outspec['starlightSuppressionSystems'] = []
        for syst in starlightSuppressionSystems:
            # The only things that are really required is the system number, 
            # the type (external/internal) and that the whole thing is a dict.
            assert isinstance(syst,dict),\
                    "Starlight suppression systems must be defined as dicts."
            assert syst.has_key('starlightSuppressionSystemNumber'),\
                    "All starlight suppression systems must have "\
                    "key starlightSuppressionSystemNumber."
            assert syst.has_key('type') and isinstance(syst['type'],basestring),\
                    "All starlight suppression systems must have key type."
            self._outspec['starlightSuppressionSystems'].append(syst.copy())

            #set an occulter, for an external or hybrid system
            self.haveOcculter = syst['type'].lower() in ('external', 'hybrid')

            #handle inf OWA
            if syst.has_key('OWA') and (syst['OWA'] == 0):
                syst['OWA'] = np.Inf

            #check for throughput
            if syst.has_key('throughput'):
                if isinstance(syst['throughput'],basestring):
                    pth = os.path.normpath(os.path.expandvars(syst['throughput']))
                    assert os.path.isfile(pth),\
                            "%s is not a valid file."%pth

                    with fits.open(pth) as tmp:
                        if (len(tmp[0].data.shape) == 2) and (2 in tmp[0].data.shape):
                            if tmp[0].data.shape[0] != 2:
                                dat = tmp[0].data.T
                            else:
                                dat = tmp[0].data
                    WA = dat[0]
                    T = dat[1]
                    Tinterp = scipy.interpolate.interp1d(WA, T, kind='cubic',\
                            fill_value=np.nan, bounds_error=False)
                    syst['throughput'] = lambda lam,alpha: Tinterp(alpha)

                    # Calculate max throughput
                    Tmax = scipy.optimize.minimize(lambda x:-syst['throughput'](lam,x),\
                            WA[np.argmax(T)],bounds=((np.min(WA),np.max(WA)),) )
                    if Tmax.success:
                        Tmax = -Tmax.fun[0]
                    else:
                        print "Warning: failed to find maximum of throughput "\
                                "interpolant for starlight suppression system "\
                                "#%d"%syst['starlightSuppressionSystemNumber']
                        Tmax = np.Tmax(T)

                    # Calculate IWA and OWA, defined as angular separations
                    # corresponding to 50% of maximum throughput
                    WA_min = scipy.optimize.fsolve(lambda x:syst['throughput']\
                            (lam,x)-Tmax/2.,np.min(WA));
                    WA_max = np.max(WA)-scipy.optimize.fsolve(lambda x:syst['throughput']\
                            (lam,np.max(WA)-x)-Tmax/2.,0.);                    
                    if not syst.has_key('IWA') or syst['IWA'] < np.min(WA):
                        syst['IWA'] = WA_min
                    if not syst.has_key('OWA') or syst['OWA'] > np.max(WA):
                        syst['OWA'] = WA_max

                    #basic validation here for size and IWA/OWA
                    #syst['throughput'] = lambda or interp
                elif isinstance(syst['throughput'],numbers.Number):
                    syst['throughput'] = lambda lam, alpha, T=float(syst['throughput']): T
            
            #check for contrast
            if syst.has_key('contrast'):
                if isinstance(syst['contrast'],basestring):
                    pth = os.path.normpath(os.path.expandvars(syst['contrast']))
                    assert os.path.isfile(pth),\
                            "%s is not a valid file."%pth

                    with fits.open(pth) as tmp:
                        if (len(tmp[0].data.shape) == 2) and (2 in tmp[0].data.shape):
                            if tmp[0].data.shape[0] != 2:
                                dat = tmp[0].data.T
                            else:
                                dat = tmp[0].data
                    WA = dat[0]
                    C = dat[1]
                    Cinterp = scipy.interpolate.interp1d(WA, C, kind='cubic',\
                            fill_value=np.nan, bounds_error=False)
                    syst['contrast'] = lambda lam,alpha: Cinterp(alpha)

                    if not syst.has_key('IWA') or syst['IWA'] < np.min(WA):
                        syst['IWA'] = np.min(WA)
                    if not syst.has_key('OWA') or syst['OWA'] > np.max(WA):
                        syst['OWA'] = np.max(WA)
                    if not syst.has_key('dMagLim'):
                        Cmin = scipy.optimize.minimize(Cinterp, WA[np.argmin(C)],\
                                bounds=((np.min(WA),np.max(WA)),) )
                        if Cmin.success:
                            Cmin = Cmin.fun[0]
                        else:
                            print "Warning: failed to find minimum of contrast "\
                                    "interpolant for starlight suppression system "\
                                    "#%d"%syst['starlightSuppressionSystemNumber']
                            Cmin = np.min(C)

                        syst['dMagLim'] = -2.5*np.log10(Cmin)

                elif isinstance(syst['contrast'],numbers.Number):
                    if not syst.has_key('dMagLim'):
                        syst['dMagLim'] = -2.5*np.log10(float(syst['contrast']))

                    syst['contrast'] = lambda lam, alpha, C=float(syst['contrast']): C

            #check for PSF
            if syst.has_key('PSF'):
                if isinstance(syst['PSF'],basestring):
                    pth = os.path.normpath(os.path.expandvars(syst['PSF']))
                    assert os.path.isfile(pth),\
                            "%s is not a valid file."%pth
                            
                    with fits.open(pth) as tmp:
                        if len(tmp[0].data.shape) == 2:
                            syst['PSF'] = lambda lam, alpha, P=tmp[0].data: P

                        if 'SAMPLING' in tmp[0].header.keys():
                            syst['PSFSampling'] = tmp[0].header['SAMPLING']

                    #basic validation here for size and IWA/OWA
                    #syst['PSF'] = lambda or interp
                else:
                    #placeholder PSF
                    syst['PSF'] = lambda lam, alpha: np.ones((3,3))

            # sampling of PSF in arcsec/pixel (from specs or PSF file header)
            if syst.has_key('PSFSampling'):
                syst['PSFSampling'] = float(syst['PSFSampling'])*u.arcsec

            # optical system overhead time
            if syst.has_key('opticaloh'):
                syst['opticaloh'] = float(syst['opticaloh'])*u.day
            else:
                syst['opticaloh'] = 1.*u.day

            #time multipliers
            if not syst.has_key('detectionTimeMultipler'):
                syst['detectionTimeMultiplier'] = 1.
            if not syst.has_key('characterizationTimeMultiplier'):
                syst['characterizationTimeMultiplier'] = 1.
    
            #check for system's IWA, OWA and dMagLim
            if syst.has_key('IWA'):
                systIWAs.append(syst['IWA'])
            if syst.has_key('OWA'):
                systOWAs.append(syst['OWA'])
            if syst.has_key('dMagLim'):
                systdMagLims.append(syst['dMagLim'])
                
        #sort suppression systems by number
        self.starlightSuppressionSystems = sorted(starlightSuppressionSystems,\
                key=itemgetter('starlightSuppressionSystemNumber')) 

        #populate IWA, OWA, and dMagLim as required
        if IWA is not None:
            self.IWA = float(IWA)*u.arcsec
        else:
            if len(systIWAs) > 0:
                self.IWA = float(min(systIWAs))*u.arcsec
            else:
                raise ValueError("Could not determine fundamental IWA.")

        if OWA is not None:
            if OWA == 0:
                self.OWA = np.inf*u.arcsec
            else:
                self.OWA = float(OWA)*u.arcsec
        else:
            if len(systOWAs) > 0:
                self.OWA = float(max(systOWAs))*u.arcsec
            else:
                raise ValueError("Could not determine fundamental OWA.")

        if dMagLim is not None:
            self.dMagLim = float(dMagLim)
        else:
            if len(systdMagLims) > 0:
                self.dMagLim = float(max(systdMagLims))
            else:
                raise ValueError("Could not determine fundamental dMagLim.")

        assert self.IWA < self.OWA, "Fundamental IWA must be smaller that the OWA."

        #finish populating outspec
        self._outspec['IWA'] = self.IWA.value
        self._outspec['OWA'] = self.OWA.value
        self._outspec['dMagLim'] = self.dMagLim

        #now go through all science Instruments
        self._outspec['scienceInstruments'] = []
        for syst in scienceInstruments:
            assert isinstance(syst,dict), "Science instruments must be defined as dicts."
            assert syst.has_key('scienceInstrumentNumber'),\
                    "All science instruments must have key scienceInstrumentNumber."
            assert syst.has_key('type') and isinstance(syst['type'],basestring),\
                    "All science instruments must have key type."

            self._outspec['scienceInstruments'].append(syst.copy())

            if syst.has_key('pixelPitch'):          #pixel pitch in m
                syst['pixelPitch'] = float(syst['pixelPitch'])*(u.m)
            if syst.has_key('focalLength'):         #focal length in m
                syst['focalLength'] = float(syst['focalLength'])*u.m 
            if syst.has_key('darkCurrent'):         # detector dark-current rate per pixel
                syst['darkCurrent'] = float(syst['darkCurrent'])*(1/u.s) 
            if syst.has_key('texp'):                # exposure time per read (s)
                syst['texp'] = float(syst['texp'])*u.s
            if syst.has_key('readNoise'):           # detector read noise (e/s)
                syst['readNoise'] = float(syst['readNoise'])
            if syst.has_key('Rspec'):               # spectral resolving power
                syst['Rspec'] = float(syst['Rspec'])
            if syst.has_key('ENF'):                 # excess noise factor
                syst['ENF'] = float(syst['ENF'])
            if syst.has_key('CIC'):                 # clock-induced-charge
                syst['CIC'] = float(syst['CIC'])
            if syst.has_key('G_EM'):                # electron multiplication gain
                syst['G_EM'] = float(syst['G_EM'])

            if syst.has_key('QE'):
                if isinstance(syst['QE'],basestring):
                    assert os.path.isfile(syst['QE']),\
                            "%s is not a valid file."%syst['QE']
                    tmp = fits.open(syst['QE'])
                    #basic validation here for size and wavelength
                    #syst['QE'] = lambda or interp
                elif isinstance(syst['QE'],numbers.Number):
                    syst['QE'] = lambda lam, QE=float(syst['QE']): QE
        
        self.scienceInstruments = sorted(scienceInstruments,\
                key=itemgetter('scienceInstrumentNumber')) 

        # Finally: if we only have one science instrument/suppression system, 
        # bring their attributes to top
        if len(self.starlightSuppressionSystems) == 1:
            for key in self.starlightSuppressionSystems[0].keys():
                if key not in self.__dict__.keys():
                    setattr(self, key, self.starlightSuppressionSystems[0][key])
        if len(self.scienceInstruments) == 1:
            for key in self.scienceInstruments[0].keys():
                if key not in self.__dict__.keys():
                    setattr(self, key, self.scienceInstruments[0][key])
       
        # set values derived from quantities above
        #if np.isinf(self.OWA.value):
        #    self.OWA = np.pi/2

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
        object. The working angle is set to the optical system IWA value, and
        the planet inclination is set to 0.
        
        Args:
            targlist:
                TargetList class object
        
        Returns:
            maxintTime:
                1D numpy array containing maximum integration time for target
                list stars (astropy Quantity with units of day)
        
        """
        
        maxintTime = self.calc_intTime(targlist,range(targlist.nStars),self.dMagLim,self.IWA,0.);
        
        return maxintTime

    def calc_intTime(self, targlist, starInd, dMagPlan, WA, I):
        """Finds integration time for a specific target system 
        
        This method is called by a method in the SurveySimulation class object.
        This method defines the data type expected, integration time is 
        determined by specific OpticalSystem classes.
        
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
            intTime (Quantity):
                1D numpy ndarray of integration times (default units of day)
        
        """
        
        intTime = np.array([1.]*targlist.nStars)*u.day
        
        return intTime

    def calc_charTime(self, targlist, starInd, dMagPlan, WA, I):
        """Finds characterization time for a specific target system 
        
        This method is called by a method in the SurveySimulation class object.
        This method defines the data type expected, characterization time is 
        determined by specific OpticalSystem classes.
        
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
            charTime (Quantity):
                1D numpy ndarray of characterization times (default units of day)
        
        """
        
        charTime = np.array([1.]*targlist.nStars)*u.day
        
        return charTime