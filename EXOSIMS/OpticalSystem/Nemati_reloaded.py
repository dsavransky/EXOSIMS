#from EXOSIMS.OpticalSystem.Nemati import Nemati
#from EXOSIMS.Prototypes import TimeKeeping as TK
#from EXOSIMS.Prototypes import SimulatedUnivers as SU
from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem
import astropy.units as u
from astropy.io import fits
import astropy.constants as const
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

access_path = '../../../../EXOSIMS/EXOSIMS-master/EXOSIMS/OpticalSystem/'

class Nemati_reloaded(OpticalSystem):
    """Nemati Optical System class
    
    This class contains all variables and methods necessary to perform
    Optical System Module calculations in exoplanet mission simulation using
    the model from Nemati 2014.
    
    Args:
        \*\*specs:
            user specified values
    
    """
    
    def __init__(self, **specs):
        
        OpticalSystem.__init__(self, **specs)
        
    def Cp_Cb_Csp(self, TL, sInds, fZ, fEZ, dMag, WA, mode, TK=None, returnExtra=False):
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
                Working angles of the planets of interest in units of arcsec
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
        
        # coronagraph parameters
        occ_trans = syst['occ_trans'](lam, WA)
        core_thruput = syst['core_thruput'](lam, WA)
        core_contrast = syst['core_contrast'](lam, WA)
        core_area = syst['core_area'](lam, WA)
        
        print occ_trans, core_thruput
        
        # solid angle of photometric aperture, specified by core_area (optional)
        Omega = core_area*u.arcsec**2
        # if zero, get omega from (lambda/D)^2
        Omega[Omega == 0] = np.pi*(np.sqrt(2)/2*lam/self.pupilDiam*u.rad)**2
        # number of pixels per lenslet
        pixPerLens = inst['lenslSamp']**2
        # number of pixels in the photometric aperture = Omega / theta^2 
        Npix = pixPerLens*(Omega/inst['pixelScale']**2).decompose().value
        
        # get stellar residual intensity in the planet PSF core
        # OPTION 1: if core_mean_intensity is missing, use the core_contrast
        if syst['core_mean_intensity'] == None:
            core_intensity = core_contrast*core_thruput
        # OPTION 2: otherwise use core_mean_intensity
        else:
            core_mean_intensity = syst['core_mean_intensity'](lam, WA)
            # if a platesale was specified with the coro parameters, apply correction
            if syst['core_platescale'] != None:
                core_mean_intensity *= (inst['pixelScale']/syst['core_platescale'] \
                        /(lam/self.pupilDiam)).decompose().value
            core_intensity = core_mean_intensity*Npix
        
        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        # get star magnitude
        mV = TL.starMag(sInds, lam)
        
        # ELECTRON COUNT RATES [ s^-1 ]
        # spectral flux density = F0 * A * Dlam * QE * T (attenuation due to optics)
        attenuation = inst['optics']*syst['optics']
        C_F0 = self.F0(lam)*self.pupilArea*deltaLam*inst['QE'](lam)*attenuation
        # planet conversion rate (planet shot)
        C_p0 = C_F0*10.**(-0.4*(mV + dMag))*core_thruput
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
        C_rn = Npix*inst['sread']/inst['texp']
        
        # C_p = PLANET SIGNAL RATE
        # photon counting efficiency
        PCeff = inst['PCeff']
        # radiation dosage
        radDos = mode['radDos']
        # photon-converted 1 frame (minimum 1 photon)
        phConv = np.clip(((C_p0 + C_sr + C_z + C_ez)/Npix \
                *inst['texp']).decompose().value, 1, None)
        # net charge transfer efficiency 
        C_p = C_p0*PCeff*NCTE
        
        # C_b = NOISE VARIANCE RATE
        # corrections for Ref star Differential Imaging e.g. dMag=3 and 20% time on ref
        # k_SZ for speckle and zodi light, and k_det for detector
        k_SZ = 1 + 1./(10**(0.4*self.ref_dMag)*self.ref_Time) if self.ref_Time > 0 else 1.
        k_det = 1 + self.ref_Time
        # calculate Cb
        ENF2 = inst['ENF']**2
        C_b = k_SZ*ENF2*(C_sr + C_z + C_ez) + k_det*(ENF2*(C_dc + C_cc) + C_rn)
        # for characterization, Cb must include the planet
        if mode['detectionMode'] == False:
            C_b = C_b + ENF2*C_p0
        
        # C_sp = spatial structure to the speckle including post-processing contrast factor
        C_sp = C_sr*TL.PostProcessing.ppFact(WA)
        print 'TEST - Cp_Cb_Csp'
        if returnExtra:
            # organize components into an optional fourth result
            C_extra = dict(C_sr = C_sr.to('1/s'),
                       C_z = C_z.to('1/s'),
                       C_ez = C_ez.to('1/s'),
                       C_dc = C_dc.to('1/s'),
                       C_cc = C_cc.to('1/s'),
                       C_rn = C_rn.to('1/s'))
            return C_p.to('1/s'), C_b.to('1/s'), C_sp.to('1/s'), C_extra
        else:
            return C_p.to('1/s'), C_b.to('1/s'), C_sp.to('1/s')

    def Cp_Cb_Csp_new(self, TL, sInds, fZ, fEZ, dMag, WA, mode, TK=None, returnExtra=False):
        
        lam = mode['lam'] # Wavelenght: nm
        syst = mode['syst'] # Starlight suppression system
        inst = mode['inst'] # Instrument
        inst_name = mode['instName'] # Instrument name
        OS = TL.OpticalSystem # Optical System
        
        A_PSF = syst['core_area'](lam, WA) # PSF Area
        C_CG = syst['core_contrast'](lam, WA) # Coronnagraph contrast
        I_pk = syst['core_mean_intensity'](lam, WA) # Peak intensity
        tau_core = syst['core_thruput'](lam, WA)*inst['MUF_thruput'] # Core thruput
        tau_occ = syst['occ_trans'](lam, WA) # Occular transmission
                
        k_pp = 5*TL.PostProcessing.ppFact(WA)*5 # Post processing factor
        eta_QE = inst['QE'](lam)
        
        # Bandwidth & Resolution
        BW = mode['BW']
        R = inst['Rs']
        
        refl_derate = inst['refl_derate']
        
        tau_HRC = inst['HRC'](lam)*refl_derate*u.ph
        tau_FSS = inst['FSS'](lam)*refl_derate*u.ph
        tau_Al = inst['Al'](lam)*refl_derate*u.ph
        
        Nlensl = inst['Nlensl']
        lenslSamp = inst['lenslSamp']
        
        # Primary mirror diameter
        D_PM = OS.pupilDiam # m
        
        lam_c = inst['lam_c']
        lam_d = inst['lam_d']

        # tau_BBAR = 0.99; tau_color-filter = 0.9; tau_imager = 0.9; tau_spect. = 0.8
        if 'spec' in inst_name.lower():
            tau_refl = tau_HRC**7 * tau_FSS**16 * tau_Al**3 * 0.99**10 * 0.9 * 0.8
            f_SR = 1/(BW*R)
            m_pix = Nlensl*(lam/lam_c)**2*lenslSamp**2
        else:
            tau_refl = tau_HRC**7 * tau_FSS**13 * tau_Al**2 * 0.99 * 0.9 * 0.9
            f_SR = 1.0
            m_pix = A_PSF*(2*lam*D_PM/(lam_d*lam_c))**2*(np.pi/180/3600)**2

        # Point source thruput
        tau_PS = tau_core*tau_refl
        
        # Obscuration due to secondary mirror and spiders
        f_o = OS.obscurFac
        # Aperture shape factor
        f_s = OS.shapeFac # pi/4
        
        A_col = f_s*D_PM**2*(1 - f_o)
        
        m_s = TL.Vmag                
        D_s = TL.dist
        F_0 = self.F0*u.nm
        
        ###
        m_s = 5.0
        D_s = 10.0*u.AU
        ###
        
        F_s = F_0*10**(-0.4*m_s)
        F_P_s = 10**(-0.4*dMag)
        F_p = F_P_s*F_s
        
        k_s = syst['k_samp']
        
        m_pixCG = A_PSF*(D_PM/(lam_d*k_s))**2*(np.pi/180/3600)**2
        
        # !!! Need to figure out F0, then do we do this calculation here or in zodiacal light? would need to have nzodi in zodiacal light calculation.
        F_ezo = F_0*fEZ*u.arcsec**2
        F_lzo = F_0*fZ*u.arcsec**2
        
        tau_unif = tau_occ*tau_refl
        
        r_pl = f_SR*F_p*A_col*tau_PS*eta_QE
        r_sp = f_SR*F_s*C_CG*I_pk*m_pixCG*tau_refl*A_col*eta_QE
        r_ezo = f_SR*F_ezo*A_PSF*A_col*tau_unif*eta_QE
        r_lzo = r_ezo*F_lzo/F_ezo
        
        r_ph = (r_pl + r_sp + r_ezo + r_lzo)/m_pix
        
        t_MF = TK.missionPortion
        t_f = inst['texp']
        
        k_RN = inst['kRN']
        
        k_EM = round(-5*k_RN/np.log(0.9), 2)
        L_CR = 0.0323*k_EM + 133.5
        
        CTE_derate = inst['CTE_derate']
        k_e = t_f*r_ph
        eta_PC = 0.9
        eta_HP = 1 - t_MF/20
        eta_CR = 1 - 8.5/u.s*t_f*L_CR/1024**2
        eta_NCT = CTE_derate*max(0, min(1 + t_MF*(0.858 - 1), 1 + t_MF*(0.858 - 1 + 3.24*(k_e - 0.089))))
        
        deta_QE = eta_QE*eta_PC*eta_HP*eta_CR*eta_NCT
        
        # Excess noise factor
        ENF = inst['ENF']
        k_d = inst['dark_derate']
        
        f_ref = self.ref_Time
        dmag_s = self.ref_dMag
        f_b = 10**(0.4*dmag_s)
        
        k_sp = 1 + 1/(f_ref*f_b)
        k_det = 1 + 1/(f_ref*f_b**2)
        k_CIC = k_d*(k_EM*4.337e-6 + 7.6e-3)
        
        i_d = k_d*(1.5 + t_MF/2)/u.s/3600
        
        pixel_size = inst['pixelSize']
        n_pix = inst['pixelNumber']**2
        
        r_dir = 625*m_pix*(pixel_size/(0.2*u.m))**2*u.ph/u.s
        r_indir = 1.25*np.pi*m_pix/n_pix*u.ph/u.s
        
        eta_ph = r_dir + r_indir
        eta_e = eta_ph*deta_QE
        
        r_DN = ENF**2*i_d*m_pix
        r_CIC = ENF**2*k_CIC*m_pix/t_f
        r_lum = ENF**2*eta_e
        r_RN = (k_RN/k_EM)**2*m_pix/t_f # !!! Update in equations
        
        dC_CG = C_CG/(5*k_pp)
        
        C_p = f_SR*F_p*A_col*tau_PS*deta_QE
        
        C_b = ENF**2*(r_pl + k_sp*(r_sp + r_ezo) + k_det*(r_lzo + r_DN + r_CIC + r_lum + r_RN))
        
        C_sp = f_SR*F_s*dC_CG*I_pk*m_pixCG*tau_refl*A_col*deta_QE
        
        return C_p, C_b, C_sp

    def calc_intTime(self, TL, sInds, fZ, fEZ, dMag, WA, mode, TK=None):
        """Finds integration times of target systems for a specific observing 
        mode (imaging or characterization), based on Nemati 2014 (SPIE).
        
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
        
        # electron counts
        C_p, C_b, C_sp = self.Cp_Cb_Csp_new(TL, sInds, fZ, fEZ, dMag, WA, mode, TK)
        
        # get SNR threshold
        SNR = mode['SNR']
        # calculate integration time based on Nemati 2014
        with np.errstate(divide='ignore', invalid='ignore'):
            intTime = np.true_divide(SNR**2*C_b, (C_p**2 - (SNR*C_sp)**2))
#           infinite and NAN are set to zero
        intTime[np.isinf(intTime) | np.isnan(intTime)] = 0.*u.d
        # negative values are set to zero
        intTime[intTime < 0] = 0.*u.d
        
        return intTime.to('day')
        return intTime

    def calc_dMag_per_intTime(self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None):
        """Finds achievable dMag for one integration time per star in the input 
        list at one working angle.
        
        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light for each star in sInds
                in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light for each star in sInds
                in units of 1/arcsec2
            WA (astropy Quantity array):
                Working angle for each star in sInds in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)
            
        Returns:
            dMag (ndarray):
                Achievable dMag for given integration time and working angle
                
        """
        
        # cast sInds, WA, fZ, fEZ, and intTimes to arrays
        sInds = np.array(sInds, ndmin=1, copy=False)
        WA = np.array(WA.value, ndmin=1)*WA.unit
        fZ = np.array(fZ.value, ndmin=1)*fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1)*fEZ.unit
        intTimes = np.array(intTimes.value, ndmin=1)*intTimes.unit
        assert len(intTimes) == len(sInds), "intTimes and sInds must be same length"
        assert len(fEZ) == len(sInds), "fEZ must be an array of length len(sInds)"
        assert len(fZ) == len(sInds), "fZ must be an array of length len(sInds)"
        assert len(WA) == len(sInds), "WA must be an array of length len(sInds)"

        # get scienceInstrument and starlightSuppressionSystem
        inst = mode['inst']
        syst = mode['syst']
        
        # get mode wavelength
        lam = mode['lam']
        # get mode bandwidth (including any IFS spectral resolving power)
        deltaLam = lam/inst['Rs'] if 'spec' in inst['name'].lower() else mode['deltaLam']
        
        # get star magnitude
        mV = TL.starMag(sInds, lam)
        
        # get signal to noise ratio
        SNR = mode['SNR']
        
        # spectral flux density = F0 * A * Dlam * QE * T (attenuation due to optics)
        attenuation = inst['optics']*syst['optics']
        C_F0 = self.F0(lam)*self.pupilArea*deltaLam*inst['QE'](lam)*attenuation
        
        # get core_thruput
        core_thruput = syst['core_thruput'](lam, WA)
        
        # calculate planet delta magnitude
        dMagLim = np.zeros(len(sInds)) + 25
        if (C_b is None) or (C_sp is None):
            _, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dMagLim, WA, mode)
        dMag = -2.5*np.log10((SNR*np.sqrt(C_b/intTimes + C_sp**2)/(C_F0*10.0**(-0.4*mV)*core_thruput*inst['PCeff'])).decompose().value)
        
        return dMag

    def ddMag_dt(self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None):
        """Finds derivative of achievable dMag with respect to integration time
        
        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light for each star in sInds
                in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light for each star in sInds
                in units of 1/arcsec2
            WA (astropy Quantity array):
                Working angle for each star in sInds in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)
            
        Returns:
            ddMagdt (ndarray):
                Derivative of achievable dMag with respect to integration time
        
        """
        
        # cast sInds, WA, fZ, fEZ, and intTimes to arrays
        sInds = np.array(sInds, ndmin=1, copy=False)
        WA = np.array(WA.value, ndmin=1)*WA.unit
        fZ = np.array(fZ.value, ndmin=1)*fZ.unit
        fEZ = np.array(fEZ.value, ndmin=1)*fEZ.unit
        intTimes = np.array(intTimes.value, ndmin=1)*intTimes.unit
        assert len(intTimes) == len(sInds), "intTimes and sInds must be same length"
        assert len(fEZ) == len(sInds), "fEZ must be an array of length len(sInds)"
        assert len(fZ) == len(sInds), "fZ must be an array of length len(sInds)"
        assert len(WA) == len(sInds), "WA must be an array of length len(sInds)"
        
        dMagLim = np.zeros(len(sInds)) + 25
        if (C_b is None) or (C_sp is None):
            _, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dMagLim, WA, mode)
        ddMagdt = 2.5/(2.0*np.log(10.0))*(C_b/(C_b*intTimes + (C_sp*intTimes)**2)).to('1/s').value
        
        return ddMagdt/u.s