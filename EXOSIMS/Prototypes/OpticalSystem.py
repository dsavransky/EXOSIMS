# -*- coding: utf-8 -*-
import astropy.units as u
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
        haveOcculter (bool):
            boolean signifying if the system has an occulter
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
        type:
            Instrument type (e.g. imaging, spectro)
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
        CIC (float):
            Clock-induced-charge
        ENF (float):
            Excess noise factor
        Gem (float):
            Electron multiplication gain
        Rs (float):
            Spectral resolving power
        Ns (float):
            Number of spectral elements in each band
        QE (callable(lam)):
            Quantum efficiency (must be callable - can be lambda function, 
            scipy.interpolate.interp2d object, etc.) with input 
            wavelength (astropy Quantity).  
     
    Common Starlight Suppression System Attributes:
        type:
            System type (e.g. internal, external, hybrid), should also contain the
            type of science instrument it can be used with (e.g. imaging, spectro)
        throughput (callable(lam, WA)):
            optical system throughput (must be callable - can be lambda,
            function, scipy.interpolate.interp2d object, etc.) with inputs
            wavelength (astropy Quantity) and angular separation/working angle 
            (astropy Quantity).
        contrast (callable(lam, WA)):
            optical system contrast curve (must be callable - can be lambda,
            function, scipy.interpolate.interp2d object, etc.) with inputs
            wavelength (astropy Quantity) and angular separation/working angle 
            (astropy Quantity).
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
            (astropy Quantity) and angular separation/working angle (astropy 
            Quantity). Note: normalization means that all throughput effects 
            must be contained in the throughput attribute.
        samp (Quantity):
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
            attenuation=0.57,intCutoff=50,Npix=14.3,Ndark=10,scienceInstruments=None,\
            lam=500,BW=0.2,pitch=13e-6,focal=240,idark=9e-5,texp=1e3,sread=3,CIC=0.0013,\
            ENF=1,Gem=1,Rs=70,QE=0.9,starlightSuppressionSystems=None,throughput=1e-2,\
            contrast=1e-9,PSF=np.ones((3,3)),samp=10,ohTime=1,imagTimeMult=1,\
            charTimeMult=1,IWA=None,OWA=None,dMagLim=None,**specs):

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

        # loop through all science Instruments (must have one defined)
        assert scienceInstruments, "No science isntrument defined."
        self.scienceInstruments = scienceInstruments
        self._outspec['scienceInstruments'] = []
        for ninst,inst in enumerate(self.scienceInstruments):
            assert isinstance(inst,dict), "Science instruments must be defined as dicts."
            assert inst.has_key('type') and isinstance(inst['type'],basestring),\
                    "All science instruments must have key type."
            #populate with values that may be filenames (interpolants)
            inst['QE'] = inst.get('QE',QE)
            self._outspec['scienceInstruments'].append(inst.copy())

            # When provided, always use bandwidth (nm) instead of bandwidth fraction.
            inst['lam'] = float(inst.get('lam',lam))*u.nm       # central wavelength (nm)
            inst['deltaLam'] = float(inst.get('deltaLam',inst['lam'].value\
                    *inst.get('BW',BW)))*u.nm                   # bandwidth (nm)
            inst['BW'] = float(inst['deltaLam']/inst['lam'])    # bandwidth fraction
            # Default lam and BW updated with values from first instrument
            if ninst == 0:
                lam, BW = inst.get('lam').value, inst.get('BW')

            # Loading detector specifications
            inst['pitch'] = float(inst.get('pitch',pitch))*u.m  # pixel pitch
            inst['focal'] = float(inst.get('focal',focal))*u.m  # focal length
            inst['idark'] = float(inst.get('idark',idark))/u.s  # dark-current rate
            inst['texp'] = float(inst.get('texp',texp))*u.s     # exposure time per frame
            inst['sread'] = float(inst.get('sread',sread))      # detector readout noise
            inst['CIC'] = float(inst.get('CIC',CIC))            # clock-induced-charge
            inst['ENF'] = float(inst.get('ENF',ENF))            # excess noise factor
            inst['Gem'] = float(inst.get('Gem',Gem))            # e- multiplication gain
            inst['Rs'] = float(inst.get('Rs',Rs))               # spectral resolving power
            inst['Ns'] = float(inst['Rs']*inst['BW']) if 'spec' in inst['type'] \
                    .lower() else 1.            # number of spectral elements in each band

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

            #populate detector specifications to outspec
            for key in inst.keys():
                if key not in ['QE']:
                    att = inst[key]
                    self._outspec['scienceInstruments'][ninst][key] = att.value \
                            if isinstance(att,u.Quantity) else att

        # loop through all starlight suppression systems (must have one defined)
        assert starlightSuppressionSystems, "No starlight suppression systems defined."
        self.starlightSuppressionSystems = starlightSuppressionSystems
        self.haveOcculter = False
        self._outspec['starlightSuppressionSystems'] = []
        for nsyst,syst in enumerate(self.starlightSuppressionSystems):
            assert isinstance(syst,dict),\
                    "Starlight suppression systems must be defined as dicts."
            assert syst.has_key('type') and isinstance(syst['type'],basestring),\
                    "All starlight suppression systems must have key type."
            #populate with values that may be filenames (interpolants)
            syst['throughput'] = syst.get('throughput',throughput)
            syst['contrast'] = syst.get('contrast',contrast)
            syst['PSF'] = syst.get('PSF',PSF)
            self._outspec['starlightSuppressionSystems'].append(syst.copy())

            #set an occulter, for an external or hybrid system
            if syst['type'].lower() in ('external', 'hybrid'):
                self.haveOcculter = True

            #handle inf OWA
            if syst.get('OWA') == 0:
                syst['OWA'] = np.Inf

            #check for throughput
            if isinstance(syst['throughput'],basestring):
                pth = os.path.normpath(os.path.expandvars(syst['throughput']))
                assert os.path.isfile(pth),\
                        "%s is not a valid file."%pth
                dat = fits.open(pth)[0].data
                assert len(dat.shape) == 2 and 2 in dat.shape, "Wrong "\
                        "throughput data shape."
                WA = dat[0] if dat.shape[0] == 2 else dat[:,0]
                T = dat[1] if dat.shape[0] == 2 else dat[:,1]
                assert np.all(T>=0), "Throughput must be positive."
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
                            "#%d"%(nsyst+1)
                    Tmax = np.Tmax(T)

                # Calculate IWA and OWA, defined as angular separations
                # corresponding to 50% of maximum throughput
                WA_min = scipy.optimize.fsolve(lambda x:syst['throughput']\
                        (lam,x)-Tmax/2.,np.min(WA))[0];
                WA_max = np.max(WA)-scipy.optimize.fsolve(lambda x:syst['throughput']\
                        (lam,np.max(WA)-x)-Tmax/2.,0.)[0];
                syst['IWA'] = max(np.min(WA),syst.get('IWA',WA_min))
                syst['OWA'] = min(np.max(WA),syst.get('OWA',WA_max))

            elif isinstance(syst['throughput'],numbers.Number):
                assert syst['throughput']>0, "Throughput must be positive."
                syst['throughput'] = lambda lam, WA, T=float(syst['throughput']): T

            #check for contrast
            if isinstance(syst['contrast'],basestring):
                pth = os.path.normpath(os.path.expandvars(syst['contrast']))
                assert os.path.isfile(pth),\
                        "%s is not a valid file."%pth
                dat = fits.open(pth)[0].data
                assert len(dat.shape) == 2 and 2 in dat.shape, "Wrong "\
                        "contrast data shape."
                WA = dat[0] if dat.shape[0] == 2 else dat[:,0]
                C = dat[1] if dat.shape[0] == 2 else dat[:,1]
                assert np.all(C>=0), "Contrast must be positive."
                Cinterp = scipy.interpolate.interp1d(WA, C, kind='cubic',\
                        fill_value=np.nan, bounds_error=False)
                syst['contrast'] = lambda lam, WA: Cinterp(WA)

                # Constraining IWA and OWA
                syst['IWA'] = max(np.min(WA),syst.get('IWA',np.min(WA)))
                syst['OWA'] = min(np.max(WA),syst.get('OWA',np.max(WA)))
                if not syst.has_key('dMagLim'):
                    Cmin = scipy.optimize.minimize(Cinterp, WA[np.argmin(C)],\
                            bounds=((np.min(WA),np.max(WA)),) )
                    if Cmin.success:
                        Cmin = Cmin.fun[0]
                    else:
                        print "Warning: failed to find minimum of contrast "\
                                "interpolant for starlight suppression system "\
                                "#%d"%(nsyst+1)
                        Cmin = np.min(C)
                    syst['dMagLim'] = -2.5*np.log10(Cmin)

            elif isinstance(syst['contrast'],numbers.Number):
                assert syst['contrast']>0, "Contrast must be positive."
                if not syst.has_key('dMagLim'):
                    syst['dMagLim'] = -2.5*np.log10(float(syst['contrast']))
                syst['contrast'] = lambda lam, WA, C=float(syst['contrast']): C

            #check for PSF
            if isinstance(syst['PSF'],basestring):
                pth = os.path.normpath(os.path.expandvars(syst['PSF']))
                assert os.path.isfile(pth),\
                        "%s is not a valid file."%pth
                hdr = fits.open(pth)[0].header
                dat = fits.open(pth)[0].data
                assert len(dat.shape) == 2, "Wrong PSF data shape."
                assert np.any(dat), "PSF must be != 0"
                syst['PSF'] = lambda lam, WA, P=dat: P
                if hdr.get('SAMPLING') is not None:
                    syst['samp'] = hdr.get('SAMPLING')
            else:
                assert np.any(syst['PSF']), "PSF must be != 0"
                syst['PSF'] = lambda lam, WA, P=np.array(syst['PSF']).astype(float): P

            #default IWA/OWA if not specified or calculated
            if not(syst.get('IWA')):
                syst['IWA'] = IWA if IWA else 0.
            if not(syst.get('OWA')):
                syst['OWA'] = OWA if OWA else np.Inf

            # Loading system specifications
            syst['IWA'] = float(syst.get('IWA'))*u.arcsec           # inner WA
            syst['OWA'] = float(syst.get('OWA'))*u.arcsec           # outer WA
            syst['dMagLim'] = float(syst.get('dMagLim'))            # outer WA
            syst['samp'] = float(syst.get('samp',samp))*u.arcsec    # PSF sampling
            syst['ohTime'] = float(syst.get('ohTime',ohTime))*u.d   # overhead time
            # imaging and characterization time multipliers
            syst['imagTimeMult'] = float(syst.get('imagTimeMult',imagTimeMult))
            syst['charTimeMult'] = float(syst.get('charTimeMult',charTimeMult))

            #populate system specifications to outspec
            for key in syst.keys():
                if key not in ['throughput','contrast','PSF']:
                    att = syst[key]
                    self._outspec['starlightSuppressionSystems'][nsyst][key] \
                            = att.value if isinstance(att,u.Quantity) else att

        # populate fundamental IWA, OWA, and dMagLim as required
        IWAs = [x.get('IWA') for x in self.starlightSuppressionSystems \
                if x.get('IWA') is not None]
        if IWA is not None:
            self.IWA = float(IWA)*u.arcsec
        elif IWAs:
            self.IWA = min(IWAs)
        else:
            raise ValueError("Could not determine fundamental IWA.")

        OWAs = [x.get('OWA') for x in self.starlightSuppressionSystems \
                if x.get('OWA') is not None]
        if OWA is not None:
            self.OWA = float(OWA)*u.arcsec if OWA != 0 else np.inf*u.arcsec
        elif OWAs:
            self.OWA = max(OWAs)
        else:
            raise ValueError("Could not determine fundamental OWA.")

        assert self.IWA < self.OWA, "Fundamental IWA must be smaller that the OWA."

        dMagLims = [x.get('dMagLim') for x in self.starlightSuppressionSystems \
                if x.get('dMagLim') is not None]
        if dMagLim is not None:
            self.dMagLim = float(dMagLim)
        elif dMagLims:
            self.dMagLim = max(dMagLims)
        else:
            raise ValueError("Could not determine fundamental dMagLim.")

        # populate outspec with all OpticalSystem scalar attributes
        for key in self.__dict__.keys():
            if key not in ['F0','scienceInstruments','starlightSuppressionSystems']:
                att = self.__dict__[key]
                self._outspec[key] = att.value if isinstance(att,u.Quantity) else att

        # default detectors and imagers
        self.Imager = self.scienceInstruments[0]
        self.ImagerSyst = self.starlightSuppressionSystems[0]
        self.Spectro = self.scienceInstruments[-1]
        self.SpectroSyst = self.starlightSuppressionSystems[-1]

    def __str__(self):
        """String representation of the Optical System object
        
        When the command 'print' is used on the Optical System object, this 
        method will print the attribute values contained in the object"""
        
        atts = self.__dict__.keys()
        
        for att in atts:
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Optical System class object attributes'

    def starMag(self, targlist, sInds, lam):
        """Calculates star visual magnitudes with B-V color, based on Traub et al. 2016.
        using empirical fit to data from Pecaut and Mamajek (2013, Appendix C).
        The expression for flux is accurate to about 7%, in the range of validity 
        400 nm < λ < 1000 nm (Traub et al. 2016).

        Args:
            targlist:
                TargetList class object
            sInds:
                (integer ndarray) indices of the stars of interest, 
                with the length of the number of planets of interest.
            lam:
                Wavelength in nm
        
        Returns:
            mag:
                Star visual magnitudes with B-V color

        """

        Vmag = targlist.Vmag[sInds]
        BV = targlist.BV[sInds]
        if lam.value < 550.:
            b = 2.20
        else:
            b = 1.54
        mag = Vmag + b*BV*(1000./lam.value - 1.818)

        return mag

    def Cp_Cb(self, targlist, sInds, I, dMag, WA, inst, syst, Npix):
        """ Calculates electron count rates for planet signal and background noise.

        Args:
            targlist:
                TargetList class object
            sInds (integer ndarray):
                Numpy ndarray containing integer indices of the stars of interest, 
                with the length of the number of planets of interest.
            I:
                Numpy ndarray containing inclinations of the planets of interest
            dMag:
                Numpy ndarray containing differences in magnitude between planets 
                and their host star
            WA:
                Numpy ndarray containing working angles of the planets of interest
            inst:
                Selected scienceInstrument
            syst:
                Selected starlightSuppressionSystem
            Npix:
                Number of noise pixels

        Returns:
            C_p:
                1D numpy array of planet signal electron count rate (units of s^-1)
            C_b:
                1D numpy array of background noise electron count rate (units of s^-1)
        
        """
        
        lam = inst['lam']                           # central wavelength
        deltaLam = inst['deltaLam']                 # bandwidth
        QE = inst['QE'](lam)                        # quantum efficiency
        Q = syst['contrast'](lam, WA)               # contrast
        T = syst['throughput'](lam, WA) / inst['Ns'] \
                * self.attenuation**2               # throughput
        mV = self.starMag(targlist,sInds,lam)       # star visual magnitude
        zodi = targlist.ZodiacalLight               # zodiacalLight module
        fZ = zodi.fZ(targlist,sInds,lam)            # surface brightness of local zodi
        fEZ = zodi.fEZ(targlist,sInds,I)            # surface brightness of exo-zodi
        X = np.sqrt(2)/2                            # aperture photometry radius (in lam/D)
        Theta = (X*lam/self.pupilDiam*u.rad).to('arcsec') # angular radius (in arcseconds)
        Omega = np.pi*Theta**2                      # solid angle subtended by the aperture

        # electron count rates [ s^-1 ]
        C_F0 = self.F0(lam)*QE*T*self.pupilArea*deltaLam
        C_p = C_F0*10.**(-0.4*(mV + dMag))          # planet signal
        C_s = C_F0*10.**(-0.4*mV)*Q                 # residual suppressed starlight (coro)
        C_z = C_F0*(fZ+fEZ)*Omega                   # zodiacal light = local + exo
        C_id = Npix*inst['idark']                   # dark current
        C_cc = Npix*inst['CIC']/inst['texp']        # clock-induced-charge
        C_sr = Npix*(inst['sread']/inst['Gem'])**2/inst['texp'] # readout noise
        C_b = inst['ENF']**2*(C_s + C_z + C_id + C_cc) + C_sr   # total noise budget
        
        return C_p, C_b

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

        sInds = range(targlist.nStars)
        I = np.array([0.]*targlist.nStars)*u.deg
        maxintTime = self.calc_intTime(targlist,sInds,I,self.dMagLim,self.IWA);
        
        return maxintTime

    def calc_intTime(self, targlist, sInds, I, dMag, WA):
        """Finds integration time for a specific target system 
        
        This method is called by a method in the SurveySimulation class object.
        This method defines the data type expected, integration time is 
        determined by specific OpticalSystem classes.
        
        Args:
            targlist:
                TargetList class object
            sInds (integer ndarray):
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
            intTime (Quantity):
                1D numpy ndarray of integration times (default units of day)
        
        """
        
        intTime = np.array([1.]*len(I))*u.day
        
        return intTime

    def calc_charTime(self, targlist, sInds, I, dMag, WA):
        """Finds characterization time for a specific target system 
        
        This method is called by a method in the SurveySimulation class object.
        This method defines the data type expected, characterization time is 
        determined by specific OpticalSystem classes.
        
        Args:
            targlist:
                TargetList class object
            sInds (integer ndarray):
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
        
        charTime = np.array([1.]*len(I))*u.day
        
        return charTime