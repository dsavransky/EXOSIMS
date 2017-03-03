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
            Shape factor of the unobscured pupil area, so that
            shapeFac * pupilDiam^2 * (1-obscurFac) = pupilArea
        pupilDiam (astropy Quantity):
            Entrance pupil diameter in units of m
        pupilArea (astropy Quantity):
            Entrance pupil area in units of m2
        telescopeKeepout (astropy Quantity):
            Telescope keepout angle in units of deg
        attenuation (float):
            Non-coronagraph attenuation, equal to the throughput of the optical 
            system without the coronagraph elements
        intCutoff (astropy Quantity):
            Maximum allowed integration time in units of day
        Ndark (float):
            Number of dark frames used
        haveOcculter (boolean):
            Boolean signifying if the system has an occulter
        F0 (callable(lam)):
            Spectral flux density
        IWA (astropy Quantity):
            Fundamental Inner Working Angle in units of arcsec
        OWA (astropy Quantity):
            Fundamental Outer Working Angle in units of arcsec
        dMagLim (float):
            Fundamental limiting delta magnitude
        scienceInstruments (list of dicts):
            All science instrument attributes (variable)
        starlightSuppressionSystems (list of dicts):
            All starlight suppression system attributes (variable)
        observingModes (list of dicts):
            Mission observing modes attributes
        
    Common science instrument attributes:
        name (string):
            Instrument name (e.g. imager-EMCCD, spectro-CCD), should contain the type of
            instrument (imager or spectro). Every instrument should have a unique name.
        pitch (astropy Quantity):
            Pixel pitch in units of m
        focal (astropy Quantity):
            Focal length in units of m
        idark (astropy Quantity):
            Detector dark-current per pixel in units of 1/s
        CIC (float):
            Clock-induced-charge per frame per pixel
        sread (float):
            Detector effective read noise per frame per pixel
        texp (astropy Quantity):
            Exposure time per frame in units of s
        ENF (float):
            Excess noise factor
        Rs (float):
            Spectral resolving power
        QE (float, callable):
            Detector quantum efficiency: either a scalar for constant QE, or a 
            two-column array for wavelength-dependent QE, where the first column 
            contains the wavelengths in units of nm. May be data or FITS filename.
        
    Common starlight suppression system attributes:
        name (string):
            System name (e.g. HLC-465, HLC-565, SPC-660), should also contain the
            central wavelength the system is optimized for. Every system must have 
            a unique name. 
        lam (astropy Quantity):
            Central wavelength in units of nm
        deltaLam (astropy Quantity):
            Bandwidth in units of nm
        BW (float):
            Bandwidth fraction
        IWA (astropy Quantity):
            Inner working angle in units of arcsec
        OWA (astropy Quantity):
            Outer working angle in units of arcsec
        occ_trans (float, callable):
            Intensity transmission of extended background sources such as zodiacal light.
            Includes pupil mask, occulter, Lyot stop and polarizer.
        core_thruput (float, callable):
            System throughput in the FWHM region of the planet PSF core.
        core_contrast (float, callable):
            System contrast = mean_intensity / PSF_peak
        core_mean_intensity (float, callable):
            Mean starlight residual normalized intensity per pixel, required to calculate 
            the total core intensity as core_mean_intensity * Npix. If not specified, 
            then the total core intensity is equal to core_contrast * core_thruput.
        core_area (astropy Quantity, callable):
            Area of the FWHM region of the planet PSF, in units of arcsec^2
        platescale (float):
            Platescale used for this set of coronagraph parameters.
        PSF (float, callable):
            Point spread function - 2D ndarray of values, normalized to 1 at
            the core. Note: normalization means that all throughput effects 
            must be contained in the throughput attribute.
        samp (astropy Quantity):
            Sampling of PSF in units of arcsec (per pixel)
        ohTime (astropy Quantity):
            Overhead time in units of days
        occulter (boolean):
            True if the system has an occulter (external or hybrid system), 
            otherwise False (internal system)
        occulterDiameter (astropy Quantity):
            Occulter diameter in units of m. Measured petal tip-to-tip.
        NocculterDistances (integer):
            Number of telescope separations the occulter operates over (number of 
            occulter bands). If greater than 1, then the occulter description is 
            an array of dicts.
        occulterDistance (astropy Quantity):
            Telescope-occulter separation in units of km.
        occulterBlueEdge (astropy Quantity):
            Occulter blue end of wavelength band in units of nm.
        occulterRedEdge (astropy Quantity):
            Occulter red end of wavelength band in units of nm.
    
    Common observing mode attributes:
        instName (string):
            Instrument name. Must match with the name of a defined 
            Science Instrument.
        systName (string):
            System name. Must match with the name of a defined 
            Starlight Suppression System.
        inst (dict):
            Selected instrument of the observing mode.
        syst (dict):
            Selected system of the observing mode.
        detectionMode (boolean):
            True if this observing mode is the detection mode, otherwise False. 
            Only one detection mode can be specified.
        SNR (float):
            Signal-to-noise ratio threshold
        timeMultiplier (float):
            Integration time multiplier
        lam (astropy Quantity):
            Central wavelength in units of nm
        deltaLam (astropy Quantity):
            Bandwidth in units of nm
        BW (float):
            Bandwidth fraction
            
    """

    _modtype = 'OpticalSystem'
    _outspec = {}

    def __init__(self,obscurFac=0.1,shapeFac=np.pi/4,pupilDiam=4,telescopeKeepout=45,\
            attenuation=0.5,intCutoff=50,Ndark=10,dMagLim=22.5,scienceInstruments=None,\
            pitch=1e-5,focal=100,idark=5e-4,CIC=5e-3,sread=0.2,texp=1000,ENF=1,Rs=70,\
            QE=0.9,starlightSuppressionSystems=None,lam=500,BW=0.2,occ_trans=0.2,\
            core_thruput=1e-2,core_contrast=1e-9,platescale=None,PSF=np.ones((3,3)),\
            samp=10,ohTime=1,observingModes=None,SNR=5,timeMultiplier=1,IWA=None,\
            OWA=None,**specs):
        
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
        self.Ndark = float(Ndark)               # number of dark frames used
        self.dMagLim = float(dMagLim)           # fundamental delta magnitude limit
        
        # Spectral flux density ~9.5e7 [ph/s/m2/nm] @ 500nm
        # F0(lambda) function of wavelength, based on Traub et al. 2016 (JATIS):
        self.F0 = lambda l: 1e4*10**(4.01-(l.to('nm').value-550)/770)*u.ph/u.s/u.m**2/u.nm 
        
        # loop through all science Instruments (must have one defined)
        assert scienceInstruments, "No science instrument defined."
        self.scienceInstruments = scienceInstruments
        self._outspec['scienceInstruments'] = []
        for ninst,inst in enumerate(self.scienceInstruments):
            assert isinstance(inst,dict), "Science instruments must be defined as dicts."
            assert inst.has_key('name') and isinstance(inst['name'],basestring),\
                    "All science instruments must have key name."
            # populate with values that may be filenames (interpolants)
            inst['QE'] = inst.get('QE',QE)
            self._outspec['scienceInstruments'].append(inst.copy())
            
            # Loading detector specifications
            inst['pitch'] = float(inst.get('pitch',pitch))*u.m  # pixel pitch
            inst['focal'] = float(inst.get('focal',focal))*u.m  # focal length
            inst['idark'] = float(inst.get('idark',idark))/u.s  # dark-current rate
            inst['CIC'] = float(inst.get('CIC',CIC))            # clock-induced-charge
            inst['sread'] = float(inst.get('sread',sread))      # effective readout noise
            inst['texp'] = float(inst.get('texp',texp))*u.s     # exposure time per frame
            inst['ENF'] = float(inst.get('ENF',ENF))            # excess noise factor
            inst['Rs'] = float(inst.get('Rs',Rs)) if 'spec' in inst['name'] \
                    .lower() else 1.                            # spectral resolving power
            
            # quantum efficiency
            if inst.has_key('QE'):
                if isinstance(inst['QE'],basestring):
                    assert os.path.isfile(inst['QE']),\
                            "%s is not a valid file."%inst['QE']
                    tmp = fits.open(inst['QE'])
                    #basic validation here for size and wavelength
                    #inst['QE'] = lambda or interp
                elif isinstance(inst['QE'],numbers.Number):
                    inst['QE'] = lambda l, QE=float(inst['QE']): QE/u.photon
            
            # populate detector specifications to outspec
            for att in inst.keys():
                if att not in ['QE']:
                    dat = inst[att]
                    self._outspec['scienceInstruments'][ninst][att] = dat.value \
                            if isinstance(dat,u.Quantity) else dat
        
        # loop through all starlight suppression systems (must have one defined)
        assert starlightSuppressionSystems, "No starlight suppression systems defined."
        self.starlightSuppressionSystems = starlightSuppressionSystems
        self.haveOcculter = False
        self._outspec['starlightSuppressionSystems'] = []
        for nsyst,syst in enumerate(self.starlightSuppressionSystems):
            assert isinstance(syst,dict),\
                    "Starlight suppression systems must be defined as dicts."
            assert syst.has_key('name') and isinstance(syst['name'],basestring),\
                    "All starlight suppression systems must have key name."
            # populate with values that may be filenames (interpolants)
            syst['occ_trans'] = syst.get('occ_trans',occ_trans)
            syst['core_thruput'] = syst.get('core_thruput',core_thruput)
            syst['core_contrast'] = syst.get('core_contrast',core_contrast)
            syst['core_mean_intensity'] = syst.get('core_mean_intensity') # no default
            syst['core_area'] = syst.get('core_area') # no default
            syst['PSF'] = syst.get('PSF',PSF)
            self._outspec['starlightSuppressionSystems'].append(syst.copy())
            
            # set an occulter, for an external or hybrid system
            syst['occulter'] = syst.get('occulter',False)
            if syst['occulter'] == True:
                self.haveOcculter = True
            
            # handle inf OWA
            if syst.get('OWA') == 0:
                syst['OWA'] = np.Inf
            
            # When provided, always use deltaLam instead of BW (bandwidth fraction)
            syst['lam'] = float(syst.get('lam',lam))*u.nm       # central wavelength (nm)
            syst['deltaLam'] = float(syst.get('deltaLam',syst['lam'].value\
                    *syst.get('BW',BW)))*u.nm                   # bandwidth (nm)
            syst['BW'] = float(syst['deltaLam']/syst['lam'])    # bandwidth fraction
            # Default lam and BW updated with values from first instrument
            if nsyst == 0:
                lam, BW = syst.get('lam').value, syst.get('BW')
            
            # Get coronagraph input parameters
            syst = self.get_coro_param(syst, 'occ_trans')
            syst = self.get_coro_param(syst, 'core_thruput')
            syst = self.get_coro_param(syst, 'core_contrast', fill=1.)
            syst = self.get_coro_param(syst, 'core_mean_intensity')
            syst = self.get_coro_param(syst, 'core_area')
            syst['platescale'] = syst.get('platescale',platescale)
            
            # Get PSF
            if isinstance(syst['PSF'],basestring):
                pth = os.path.normpath(os.path.expandvars(syst['PSF']))
                assert os.path.isfile(pth),\
                        "%s is not a valid file."%pth
                hdr = fits.open(pth)[0].header
                dat = fits.open(pth)[0].data
                assert len(dat.shape) == 2, "Wrong PSF data shape."
                assert np.any(dat), "PSF must be != 0"
                syst['PSF'] = lambda l, s, P=dat: P
                if hdr.get('SAMPLING') is not None:
                    syst['samp'] = hdr.get('SAMPLING')
            else:
                assert np.any(syst['PSF']), "PSF must be != 0"
                syst['PSF'] = lambda l, s, P=np.array(syst['PSF']).astype(float): P
            
            # default IWA/OWA if not specified or calculated
            if not(syst.get('IWA')):
                syst['IWA'] = IWA if IWA else 0.
            if not(syst.get('OWA')):
                syst['OWA'] = OWA if OWA else np.Inf
            
            # Loading system specifications
            syst['IWA'] = float(syst.get('IWA'))*u.arcsec           # inner WA
            syst['OWA'] = float(syst.get('OWA'))*u.arcsec           # outer WA
            syst['samp'] = float(syst.get('samp',samp))*u.arcsec    # PSF sampling
            syst['ohTime'] = float(syst.get('ohTime',ohTime))*u.d   # overhead time
            
            #populate system specifications to outspec
            for att in syst.keys():
                if att not in ['occ_trans','core_thruput','core_contrast',\
                        'core_mean_intensity','core_area','PSF']:
                    dat = syst[att]
                    self._outspec['starlightSuppressionSystems'][nsyst][att] \
                            = dat.value if isinstance(dat,u.Quantity) else dat
        
        # loop through all observing modes
        # if no observing mode defined, create a default mode:
        if observingModes == None:
            inst = self.scienceInstruments[0]
            syst = self.starlightSuppressionSystems[0]
            observingModes = [{'detectionMode': True,
                               'instName': inst['name'],
                               'systName': syst['name']}]
        self.observingModes = observingModes
        self._outspec['observingModes'] = []
        for nmode,mode in enumerate(self.observingModes):
            assert isinstance(mode,dict),\
                    "Observing modes must be defined as dicts."
            assert mode.has_key('instName') and mode.has_key('systName'),\
                    "All observing modes must have key instName and systName."
            assert np.any([mode['instName'] == inst['name'] for inst in \
                    self.scienceInstruments]), "The mode's instrument name " + \
                    mode['instName'] + " does not exist."
            assert np.any([mode['systName'] == syst['name'] for syst in \
                    self.starlightSuppressionSystems]), "The mode's system name " + \
                    mode['systName'] + " does not exist."
            self._outspec['observingModes'].append(mode.copy())
            
            # Loading mode specifications
            mode['SNR'] = float(mode.get('SNR',SNR))
            mode['timeMultiplier'] = float(mode.get('timeMultiplier',timeMultiplier))
            mode['detectionMode'] = mode.get('detectionMode',False)
            mode['inst'] = [inst for inst in self.scienceInstruments \
                    if inst['name'] == mode['instName']][0]
            mode['syst'] = [syst for syst in self.starlightSuppressionSystems \
                    if syst['name'] == mode['systName']][0]
            # get mode wavelength and bandwidth. If not specified, take system values.
            # When provided, always use deltaLam instead of BW (bandwidth fraction)
            syst_lam = mode['syst']['lam'].to('nm').value
            syst_BW = mode['syst']['BW']
            mode['lam'] = float(mode.get('lam',syst_lam))*u.nm
            mode['deltaLam'] = float(mode.get('deltaLam',mode['lam'].value \
                    *mode.get('BW',syst_BW)))*u.nm
            mode['BW'] = float(mode['deltaLam']/mode['lam'])
            # get mode IWA and OWA: rescale if the mode wavelength is different than 
            # the wavelength at which the system is defined
            mode['IWA'] = mode['syst']['IWA']
            mode['OWA'] = mode['syst']['OWA']
            if mode['lam'] != mode['syst']['lam']:
                mode['IWA'] = mode['IWA']*mode['lam']/mode['syst']['lam']
                mode['OWA'] = mode['OWA']*mode['lam']/mode['syst']['lam']
        
        # check for only one detection mode
        detectionModes = filter(lambda mode: mode['detectionMode'] == True, self.observingModes)
        assert len(detectionModes) <= 1, "More than one detection mode specified."
        # if not specified, default detection mode is first imager mode
        if len(detectionModes) == 0:
            imagerModes = filter(lambda mode: 'imag' in mode['inst']['name'], self.observingModes)
            if imagerModes:
                imagerModes[0]['detectionMode'] = True
            # if no imager mode, default detection mode is first observing mode
            else:
                self.observingModes[0]['detectionMode'] = True
        
        # populate fundamental IWA and OWA as required
        IWAs = [x.get('IWA') for x in self.observingModes if x.get('IWA') is not None]
        if IWA is not None:
            self.IWA = float(IWA)*u.arcsec
        elif IWAs:
            self.IWA = min(IWAs)
        else:
            raise ValueError("Could not determine fundamental IWA.")
        
        OWAs = [x.get('OWA') for x in self.observingModes if x.get('OWA') is not None]
        if OWA is not None:
            self.OWA = float(OWA)*u.arcsec if OWA != 0 else np.inf*u.arcsec
        elif OWAs:
            self.OWA = max(OWAs)
        else:
            raise ValueError("Could not determine fundamental OWA.")
        
        assert self.IWA < self.OWA, "Fundamental IWA must be smaller that the OWA."
        
        # populate outspec with all OpticalSystem scalar attributes
        for att in self.__dict__.keys():
            if att not in ['F0','scienceInstruments','starlightSuppressionSystems',\
                    'observingModes']:
                dat = self.__dict__[att]
                self._outspec[att] = dat.value if isinstance(dat,u.Quantity) else dat

    def __str__(self):
        """String representation of the Optical System object
        
        When the command 'print' is used on the Optical System object, this 
        method will print the attribute values contained in the object"""
        
        for att in self.__dict__.keys():
            print '%s: %r' % (att, getattr(self, att))
        
        return 'Optical System class object attributes'

    def get_coro_param(self, syst, param_name, fill=0.):
        """ For a given starlightSuppressionSystem, loads an input parameter that
        depends on the angular separation and updates the system. Also updates the 
        IWA and OWA of the system.
        
        Args:
            syst (dict):
                Dictionnary containing the parameters of one starlight suppression system
            param_name (string):
                Name of the parameter that must be loaded
            fill (float):
                Fill value for working angles outside of the input array definition
        
        Returns:
            syst (dict):
                Updated dictionnary of parameters
        
        """
        
        assert isinstance(param_name, basestring), "param_name must be a string."
        if isinstance(syst[param_name], basestring):
            pth = os.path.normpath(os.path.expandvars(syst[param_name]))
            assert os.path.isfile(pth),\
                    "%s is not a valid file."%pth
            dat = fits.open(pth)[0].data
            assert len(dat.shape) == 2 and 2 in dat.shape, "Wrong "+param_name+" data shape."
            WA,D = (dat[0],dat[1]) if dat.shape[0] == 2 else (dat[:,0],dat[:,1])
            assert np.all(D>=0) and np.all(D<=1), \
                    param_name+" must be positive and smaller than 1."
            # parameter values outside of WA
            Dinterp = scipy.interpolate.interp1d(WA, D, kind='cubic',\
                    fill_value=fill, bounds_error=False)
            syst[param_name] = lambda l, s: np.array(Dinterp(s.to('arcsec').value),ndmin=1)
            # update IWA and OWA
            syst['IWA'] = max(np.min(WA), syst.get('IWA',np.min(WA)))
            syst['OWA'] = min(np.max(WA), syst.get('OWA',np.max(WA)))
            
        elif isinstance(syst[param_name],numbers.Number):
            assert syst[param_name]>=0 and syst[param_name]<=1, \
                    param_name+" must be positive and smaller than 1."
            syst[param_name] = lambda l, s, D = float(syst[param_name]): \
                    ((s >= syst['IWA']) & (s <= syst['OWA']))*(D-fill)+fill
            
        else:
            syst[param_name] = None
        
        return syst

    def Cp_Cb_Csp(self, TL, sInds, fZ, fEZ, dMag, WA, mode, returnExtra=False):
        """ Calculates electron count rates for planet signal, background noise, 
        and speckle residuals.
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            dMag (float ndarray):
                Differences in magnitude between planets and their host star
            WA (astropy Quantity array):
                Working angles of the planets of interest in units of mas
            mode (dict):
                Selected observing mode
            returnExtra (boolean):
                Optional flag, default False, set True to return additional rates for validation
        
        Returns:
            C_p (astropy Quantity array):
                Planet signal electron count rate in units of 1/s
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s
        
        """
        
        # get scienceInstrument and starlightSuppressionSystem
        inst = mode['inst']
        syst = mode['syst']
        
        # get mode wavelength
        lam = mode['lam']
        # get mode bandwidth (including any IFS spectral resolving power)
        deltaLam = lam/inst['Rs'] if 'spec' in inst['name'].lower() else mode['deltaLam']
        
        # if the mode wavelength is different than the wavelength at which the system 
        # is defined, we need to rescale the working angles
        if lam != syst['lam']:
            WA = WA*lam/syst['lam']
        
        # get star magnitude
        sInds = np.array(sInds,ndmin=1)
        mV = TL.starMag(sInds,lam)
        
        # solid angle of photometric aperture, specified by core_area(optional), 
        # otherwise obtained from (lambda/D)^2
        Omega = syst['core_area'](lam,WA)*u.arcsec**2 if syst['core_area'] else \
                np.pi*(np.sqrt(2)/2*lam/self.pupilDiam*u.rad)**2
        # number of pixels in the photometric aperture = Omega / theta^2 
        Npix = (Omega / (inst['pitch']/inst['focal']*u.rad)**2).decompose().value
        
        # get coronagraph input parameters
        occ_trans = syst['occ_trans'](lam,WA)
        core_thruput = syst['core_thruput'](lam,WA)
        core_contrast = syst['core_contrast'](lam,WA)
        
        # get stellar residual intensity in the planet PSF core
        # OPTION 1: if core_mean_intensity is missing, use the core_contrast
        if syst['core_mean_intensity'] == None:
            core_intensity = core_contrast * core_thruput
        # OPTION 2: otherwise use core_mean_intensity
        else:
            core_mean_intensity = syst['core_mean_intensity'](lam,WA)
            # if a platesale was specified with the coro parameters, apply correction
            if syst['platescale'] != None:
                platescale = inst['pitch']/inst['focal']/(lam/self.pupilDiam)
                core_mean_intensity *= platescale/syst['platescale']
            core_intensity = core_mean_intensity * Npix
        
        # ELECTRON COUNT RATES [ s^-1 ]
        # spectral flux density = F0 * A * Dlam * QE * T (non-coro attenuation)
        C_F0 = self.F0(lam)*self.pupilArea*deltaLam*inst['QE'](lam)*self.attenuation
        # planet signal
        C_p = C_F0*10.**(-0.4*(mV + dMag))*core_thruput
        # starlight residual
        C_sr = C_F0*10.**(-0.4*mV)*core_intensity
        # zodiacal light
        C_z = C_F0*fZ*Omega*occ_trans
        # exozodiacal light
        C_ez = C_F0*fEZ*Omega*core_thruput
        # dark current
        C_dc = Npix*inst['idark']
        # clock-induced-charge
        C_cc = Npix*inst['CIC']/inst['texp']
        # readout noise
        C_rn = Npix*inst['sread']**2/inst['texp']
        # background
        C_b = inst['ENF']**2*(C_sr+C_z+C_ez+C_dc+C_cc)+C_rn 
        # spatial structure to the speckle including post-processing contrast factor
        C_sp = C_sr*TL.PostProcessing.ppFact(WA)
        
        # organize components into an optional fourth result
        C_extra = dict(
            C_sr = C_sr.to('1/s'),
            C_z  = C_z.to('1/s'),
            C_ez = C_ez.to('1/s'),
            C_dc = C_dc.to('1/s'),
            C_cc = C_cc.to('1/s'),
            C_rn = C_rn.to('1/s'))
        
        if returnExtra:
            return C_p.to('1/s'), C_b.to('1/s'), C_sp.to('1/s'), C_extra
        else:
            return C_p.to('1/s'), C_b.to('1/s'), C_sp.to('1/s')

    def calc_intTime(self, TL, sInds, fZ, fEZ, dMag, WA, mode):
        """Finds integration time for a specific target system 
        
        This method is called by a method in the SurveySimulation class object.
        This method defines the data type expected, integration time is 
        determined by specific OpticalSystem classes.
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            dMag (float ndarray):
                Differences in magnitude between planets and their host star
            WA (astropy Quantity array):
                Working angles of the planets of interest in units of arcsec
            mode (dict):
                Selected observing mode
        
        Returns:
            intTime (astropy Quantity array):
                Integration times in units of day
        
        """
        
        # reshape sInds
        sInds = np.array(sInds,ndmin=1)
        intTime = np.ones(len(sInds))*u.day
        
        return intTime

    def calc_maxintTime(self, TL, sInds, fZ, fEZ, mode):
        """Finds maximum integration time for target systems 
        
        This method is called in the run_sim() method of the SurveySimulation
        class object. It calculates the default maximum integration time for a
        fixed dMagLim, and at the optical system IWA.
        
        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            mode (dict):
                Selected observing mode
        
        Returns:
            maxintTime (astropy Quantity array):
                Maximum integration times for target list stars in units of day
        
        """
        
        # calc integration time, for dMag = dMagLim, and WA = IWA
        dMag = self.dMagLim
        WA = self.IWA
        mode = self.observingModes[0]
        maxintTime = self.calc_intTime(TL, sInds, fZ, fEZ, dMag, WA, mode)
        
        return maxintTime
