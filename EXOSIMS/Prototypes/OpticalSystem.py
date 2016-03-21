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
        obscurFac (float):
            Obscuration factor due to secondary mirror and spiders
        shapeFac (float):
            shape factor (also known as fill factor) where 
            :math:`shapeFac * pupilDiam^2 * (1-obscurFac) = pupilArea`
        pupilDiam (Quantity):
            entrance pupil diameter (default units of m)
        pupilArea (Quantity):
            entrance pupil area (default units of m\ :sup:`2`)
        telescopeKeepout (float):
            telescope keepout angle in degrees
        attenuation (float):
            non-coronagraph attenuation (throughput)
        intCutoff (Quantity):
            integration cutoff (default units of day)
        Npix (float):
            number of noise pixels
        Ndark (float):
            number of dark frames used
        F0 (callable(lam)):
            Spectral flux density
        IWA (Quantity):
            Fundamental Inner Working Angle (default units of arcsecond)
        OWA (Quantity):
            Fundamental Outer Working Angle (default units of arcsecond)
        dMagLim (float):
            Fundamental limiting delta magnitude. 
        scienceInstruments (list of dicts):
            All science instrument attributes (variable)
        starlightSuppressionSystems (list of dicts):
            All starlight suppression system attributes (variable)

    Common Science Instrument Attributes:
        ScInstNumber:
            Science instrument number
        type:
            Instrument type (imager, spectrograph, etc.)
        lam (Quantity):
            Central wavelength (default units of nm)
        deltaLam (Quantity):
            Bandwidth (default units of nm)
        BW (float):
            Bandwidth fraction
        pitch (float):
            Pixel pitch (default units of m)
        focal (float):
            Focal length (default units of m)
        idark (float):
            Dark current rate (default units of s^-1)
        texp (float):
            Exposure time per frame (default units of s)
        sread (float):
            Detector readout noise
        Rs (float):
            Spectral resolving power
        ENF (float):
            Excess noise factor
        CIC (float):
            Clock-induced-charge
        Gem (float):
            Electron multiplication gain
        Ns (float):
            Number of spectral elements in each band
        QE (callable(lam)):
            Quantum efficiency (must be callable - can be lambda function, 
            scipy.interpolate.interp2d object, etc.) with input 
            wavelength (astropy Quantity).  
     
    Common Starlight Suppression System Attributes:
        SSSystNumber:
            Starlight Suppression System number
        type:
            System type (internal, external, hybrid, etc.)
        haveOcculter (bool):
            boolean signifying if the system has an occulter
        throughput (callable(lam, WA)):
            optical system throughput (must be callable - can be lambda,
            function, scipy.interpolate.interp2d object, etc.) with inputs
            wavelength (astropy Quantity) and angular separation/working angle (astropy 
            Quantity).
        contrast (callable(lam, WA)):
            optical system contrast curve (must be callable - can be lambda,
            function, scipy.interpolate.interp2d object, etc.) with inputs
            wavelength (astropy Quantity) and angular separation/working angle (astropy 
            Quantity).
        IWA:
            Inner working angle
        OWA:
            Outer working angle
        dMagLim:
            System delta magnitude limit
        PSF (callable(lam, WA)):
            point spread function - 2D ndarray of values, normalized to 1 at
            the core (must be callable - can be lambda, function,
            scipy.interpolate.interp2d object, etc.) with inputs wavelength
            (astropy Quantity) and angular separation/working angle (astropy Quantity). Note
            normalization means that all throughput effects must be contained
            in the throughput attribute.
        PSFSampling (Quantity):
            Sampling of PSF in arcsec/pixel (default units of arcsec)
        ohTime (astropy Quantity):
            Overhead time (default units of days)
        imagTimeMult (float):
            Imaging time multiplier
        charTimeMult (float):
            Characterization time multiplier
                    
    """

    _modtype = 'OpticalSystem'
    _outspec = {}

    def __init__(self,obscurFac=0.2,shapeFac=np.pi/4,pupilDiam=4,telescopeKeepout=45,\
            attenuation=0.57,intCutoff=50.,Npix=14.3,Ndark=10,scienceInstruments=None,\
            lam=500,BW=0.2,QE=0.9,idark=9e-5,CIC=0.0013,sread=3,texp=1000,ENF=1,Gem=1,\
            Rs=70,pitch=13e-6,focal=240,starlightSuppressionSystems=None,ohTime=1,\
            imagTimeMult=1,charTimeMult=1,IWA=None,OWA=None,dMagLim=None,**specs):

        #must have a starlight suppression system and science instrument defined
        if not starlightSuppressionSystems:
            raise ValueError("No starlight suppression systems defined.")
        if not scienceInstruments:
            raise ValueError("No science isntrument defined.")

        #load all values with defaults
        self.obscurFac = float(obscurFac)       # obscuration factor
        self.shapeFac = float(shapeFac)         # shape factor
        self.pupilDiam = float(pupilDiam)*u.m   # entrance pupil diameter
        self.pupilArea = (1-self.obscurFac)*self.shapeFac*self.pupilDiam**2\
                                                # entrance pupil area
        self.telescopeKeepout = float(telescopeKeepout)*u.deg\
                                                # keepout angle in degrees
        self.attenuation = float(attenuation)   # non-coronagraph attenuation factor
        self.intCutoff = float(intCutoff)*u.d   # integration time cutoff
        self.Npix = float(Npix)                 # number of noise pixels
        self.Ndark = float(Ndark)               # number of dark frames used

        # Spectral flux density ~9.5e7 [ph/s/m2/nm] @ 500nm
        # F0(lambda) function of wavelength, based on Traub et al. 2016 (JATIS):
        self.F0 = lambda lam: 1e4*10**(4.01-(1e-3*lam.value-0.55)/0.77)*u.ph/u.s/u.m**2/u.nm 

        #populate all scalar atributes to outspec
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key],u.Quantity):
                self._outspec[key] = self.__dict__[key].value
            else:
                self._outspec[key] = self.__dict__[key]

        #now go through all science Instruments
        self._outspec['scienceInstruments'] = []
        for inst in scienceInstruments:
            assert isinstance(inst,dict), "Science instruments must be defined as dicts."
            assert inst.has_key('ScInstNumber'),"All science instruments must "\
                    "have key ScInstNumber."
            assert inst.has_key('type') and isinstance(inst['type'],basestring),\
                    "All science instruments must have key type."

            # When provided, always use bandwidth (nm) instead of bandwidth fraction.
            inst['lam'] = float(inst.get('lam',lam))*u.nm       # central wavelength (nm)
            inst['deltaLam'] = float(inst.get('deltaLam',inst['lam'].value\
                    *inst.get('BW',BW)))*u.nm                   # bandwidth (nm)
            inst['BW'] = inst['deltaLam']/inst['lam']           # bandwidth fraction

            # Default lam and BW updated with values from imager (ScInstNumber=0)
            if inst['ScInstNumber'] == 0:
                lam,BW = inst.get('lam').value,inst.get('BW').value

            # Loading detector specifications
            inst['pitch'] = float(inst.get('pitch',pitch))*u.m  # pixel pitch
            inst['focal'] = float(inst.get('focal',focal))*u.m  # focal length
            inst['idark'] = float(inst.get('idark',idark))/u.s  # dark-current rate
            inst['texp'] = float(inst.get('texp',texp))*u.s     # exposure time per frame
            inst['sread'] = float(inst.get('sread',sread))      # detector readout noise
            inst['Rs'] = float(inst.get('Rs',Rs))               # spectral resolving power
            inst['ENF'] = float(inst.get('ENF',ENF))            # excess noise factor
            inst['CIC'] = float(inst.get('CIC',CIC))            # clock-induced-charge
            inst['Gem'] = float(inst.get('Gem',Gem))            # e- multiplication gain

            # number of spectral elements in each band
            inst['Ns'] = inst['BW']*inst['Rs'] if 'spec' in inst['type'].lower() else 1.
            
            # quantum efficiency
            if inst.has_key('QE'):
                if isinstance(inst['QE'],basestring):
                    assert os.path.isfile(inst['QE']),\
                            "%s is not a valid file."%inst['QE']
                    tmp = fits.open(inst['QE'])
                    #basic validation here for size and wavelength
                    #inst['QE'] = lambda or interp
                elif isinstance(inst['QE'],numbers.Number):
                    inst['QE'] = lambda lam, QE=float(inst['QE']): QE/u.photon
            
            self._outspec['scienceInstruments'].append(inst.copy())

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
            assert syst.has_key('SSSystNumber'),"All starlight suppression "\
                    "systems must have key SSSystNumber."
            assert syst.has_key('type') and isinstance(syst['type'],basestring),\
                    "All starlight suppression systems must have key type."

            #set an occulter, for an external or hybrid system
            syst['haveOcculter'] = syst['type'].lower() in ('external', 'hybrid')

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
                    syst['throughput'] = lambda lam, WA: Tinterp(WA)

                    # Calculate max throughput
                    Tmax = scipy.optimize.minimize(lambda x:-syst['throughput'](lam,x),\
                            WA[np.argmax(T)],bounds=((np.min(WA),np.max(WA)),) )
                    if Tmax.success:
                        Tmax = -Tmax.fun[0]
                    else:
                        print "Warning: failed to find maximum of throughput "\
                                "interpolant for starlight suppression system "\
                                "#%d"%syst['SSSystNumber']
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
                    syst['throughput'] = lambda lam, WA, T=float(syst['throughput']): T
            
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
                    syst['contrast'] = lambda lam, WA: Cinterp(WA)

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
                                    "#%d"%syst['SSSystNumber']
                            Cmin = np.min(C)

                        syst['dMagLim'] = -2.5*np.log10(Cmin)

                elif isinstance(syst['contrast'],numbers.Number):
                    if not syst.has_key('dMagLim'):
                        syst['dMagLim'] = -2.5*np.log10(float(syst['contrast']))

                    syst['contrast'] = lambda lam, WA, C=float(syst['contrast']): C

            #check for PSF
            if syst.has_key('PSF'):
                if isinstance(syst['PSF'],basestring):
                    pth = os.path.normpath(os.path.expandvars(syst['PSF']))
                    assert os.path.isfile(pth),\
                            "%s is not a valid file."%pth
                            
                    with fits.open(pth) as tmp:
                        if len(tmp[0].data.shape) == 2:
                            syst['PSF'] = lambda lam, WA, P=tmp[0].data: P

                        if 'SAMPLING' in tmp[0].header.keys():
                            syst['PSFSampling'] = tmp[0].header['SAMPLING']

                    #basic validation here for size and IWA/OWA
                    #syst['PSF'] = lambda or interp
                else:
                    #placeholder PSF
                    syst['PSF'] = lambda lam, WA: np.ones((3,3))

            # sampling of PSF in arcsec/pixel (from specs or PSF file header)
            if syst.has_key('PSFSampling'):
                syst['PSFSampling'] = float(syst['PSFSampling'])*u.arcsec

            # Overhead time
            syst['ohTime'] = float(syst['ohTime'] if syst.has_key('ohTime')\
                    else ohTime)*u.day
            # Time multipliers
            syst['imagTimeMult'] = float(syst['imagTimeMult']\
                    if syst.has_key('imagTimeMult') else imagTimeMult)
            syst['charTimeMult'] = float(syst['charTimeMult']\
                    if syst.has_key('charTimeMult') else charTimeMult)

            #check for system's IWA, OWA and dMagLim
            if syst.has_key('IWA'):
                systIWAs.append(syst['IWA'])
            if syst.has_key('OWA'):
                systOWAs.append(syst['OWA'])
            if syst.has_key('dMagLim'):
                systdMagLims.append(syst['dMagLim'])

            self._outspec['starlightSuppressionSystems'].append(syst.copy())

        #now, sort science instruments and suppression systems by number
        self.scienceInstruments = sorted(scienceInstruments,key=itemgetter('ScInstNumber')) 
        self.starlightSuppressionSystems = sorted(starlightSuppressionSystems,\
                key=itemgetter('SSSystNumber')) 

        #populate IWA, OWA, and dMagLim as required
        if IWA is not None:
            self.IWA = float(IWA)*u.arcsec
        elif len(systIWAs) > 0:
            self.IWA = float(min(systIWAs))*u.arcsec
        else:
            raise ValueError("Could not determine fundamental IWA.")

        if OWA is not None:
            self.OWA = float(OWA)*u.arcsec if OWA != 0 else np.inf*u.arcsec
        elif len(systOWAs) > 0:
            self.OWA = float(max(systOWAs))*u.arcsec
        else:
            raise ValueError("Could not determine fundamental OWA.")

        if dMagLim is not None:
            self.dMagLim = float(dMagLim)
        elif len(systdMagLims) > 0:
            self.dMagLim = float(max(systdMagLims))
        else:
            raise ValueError("Could not determine fundamental dMagLim.")

        assert self.IWA < self.OWA, "Fundamental IWA must be smaller that the OWA."

        #finish populating outspec
        self._outspec['IWA'] = self.IWA.value
        self._outspec['OWA'] = self.OWA.value
        self._outspec['dMagLim'] = self.dMagLim

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

    def __str__(self):
        """String representation of the Optical System object
        
        When the command 'print' is used on the Optical System object, this 
        method will print the attribute values contained in the object"""
        
        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Optical System class object attributes'

    def starMag(self,targlist,sInd,lam):
        """Calculates star visual magnitudes with B-V color, based on Traub et al. 2016.
        using empirical fit to data from Pecaut and Mamajek (2013, Appendix C).
        The expression for flux is accurate to about 7%, in the range of validity 
        400 nm < λ < 1000 nm (Traub et al. 2016).

        Args:
            targlist:
                TargetList class object
            sInd:
                (integer ndarray) indices of the stars of interest, 
                with the length of the number of planets of interest.
            lam:
                Wavelength in nm
        
        Returns:
            mag:
                Star visual magnitudes with B-V color

        """

        Vmag = targlist.Vmag[sInd]
        BV = targlist.BV[sInd]
        if lam.value < 550.:
            b = 2.20
        else:
            b = 1.54
        mag = Vmag + b*BV*(1000./lam.value - 1.818)

        return mag

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

        sInd = range(targlist.nStars)
        I = np.array([0.]*targlist.nStars)*u.deg
        maxintTime = self.calc_intTime(targlist,sInd,I,self.dMagLim,self.IWA);
        
        return maxintTime

    def calc_intTime(self, targlist, sInd, I, dMag, WA):
        """Finds integration time for a specific target system 
        
        This method is called by a method in the SurveySimulation class object.
        This method defines the data type expected, integration time is 
        determined by specific OpticalSystem classes.
        
        Args:
            targlist:
                TargetList class object
            sInd (integer ndarray):
                Numpy ndarray containing integer indices of the stars of interest, 
                with the length of the number of planets of interest.
            dMag:
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

    def calc_charTime(self, targlist, sInd, I, dMag, WA):
        """Finds characterization time for a specific target system 
        
        This method is called by a method in the SurveySimulation class object.
        This method defines the data type expected, characterization time is 
        determined by specific OpticalSystem classes.
        
        Args:
            targlist:
                TargetList class object
            sInd (integer ndarray):
                Numpy ndarray containing integer indices of the stars of interest, 
                with the length of the number of planets of interest.
            I:
                Numpy ndarray containing inclinations of the planets of interest
            dMag:
                Numpy ndarray containing differences in magnitude between planets 
                and their host star
            WA:
                Numpy ndarray containing working angles of the planets of interest
        
        Returns:
            charTime (Quantity):
                1D numpy ndarray of characterization times (default units of day)
        
        """
        
        charTime = np.array([1.]*targlist.nStars)*u.day
        
        return charTime