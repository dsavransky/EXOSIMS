from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem
import astropy.units as u
from astropy.io import fits
import astropy.constants as const
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

class Nemati_2019(OpticalSystem):
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
            TK (TimeKeeping module):
                TimeKeeping class object
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
        
        f_ref = self.ref_Time # fraction of time spent on ref star for RDI
        dmag_s = self.ref_dMag # reference star dMag for RDI
        k_pp = 5*TL.PostProcessing.ppFact(WA) # post processing factor
        OS = TL.OpticalSystem # optical system module
        m_s = TL.Vmag # V magnitude        
        
        D_PM = OS.pupilDiam # primary mirror diameter in units of m
        f_o = OS.obscurFac # obscuration due to secondary mirror and spiders
        f_s = OS.shapeFac # aperture shape factor
        
        lam = mode['lam'] # wavelenght in units of nm
        inst_name = mode['instName'] # instrument name
        BW = mode['BW'] # bandwidth
        syst = mode['syst'] # starlight suppression system
        inst = mode['inst'] # instrument dictionary
            
        
        F0_dict = {}
        F_0 = np.ndarray((TL.nStars))
        for i in range(TL.nStars):
            spec = TL.Spec[i]
            name = TL.Name[i]
            if spec in F0_dict.keys():
                F_0[i] = F0_dict[spec]
            else:
                F_0[i] = TL.F0(BW, lam, spec, name)
                F0_dict[spec] = F_0[i]
        
        A_PSF = syst['core_area'](lam, WA) # PSF area
        C_CG = syst['core_contrast'](lam, WA) # coronnagraph contrast
        I_pk = syst['core_mean_intensity'](lam, WA) # peak intensity
        tau_core = syst['core_thruput'](lam, WA)*inst['MUF_thruput'] # core thruput
        tau_occ = syst['occ_trans'](lam, WA) # Occular transmission

        R = inst['Rs'] # resolution
        eta_QE = inst['QE'](lam) # quantum efficiency        
        refl_derate = inst['refl_derate']        
        tau_HRC = inst['HRC'](lam)*refl_derate*u.ph
        tau_FSS = inst['FSS'](lam)*refl_derate*u.ph
        tau_Al = inst['Al'](lam)*refl_derate*u.ph        
        Nlensl = inst['Nlensl']
        lenslSamp = inst['lenslSamp']
        lam_c = inst['lam_c']
        lam_d = inst['lam_d']
        k_s = inst['k_samp']
        t_f = inst['texp']        
        k_RN = inst['kRN']
        CTE_derate = inst['CTE_derate']
        ENF = inst['ENF'] # excess noise factor
        k_d = inst['dark_derate']
        pixel_size = inst['pixelSize']
        n_pix = inst['pixelNumber']**2
        
        t_now = (TK.currentTimeNorm.to(u.d)).value*30.4375 # current time in units of months
        t_EOL = 63. # mission total lifetime in months
        t_MF = t_now/t_EOL

        tau_BBAR = 0.99
        tau_color_filt = 0.9
        tau_imager = 0.9
        tau_spect = 0.8
        if 'spec' in inst_name.lower():
            tau_refl = tau_HRC**7 * tau_FSS**16 * tau_Al**3 * tau_BBAR**10 * tau_color_filt * tau_spect
            f_SR = 1/(BW*R)
            m_pix = Nlensl*(lam/lam_c)**2*lenslSamp**2
        else:
            tau_refl = tau_HRC**7 * tau_FSS**13 * tau_Al**2 * tau_BBAR * tau_color_filt * tau_imager
            f_SR = 1.0
            m_pix = A_PSF*(2*lam*D_PM/(lam_d*lam_c))**2*(np.pi/180/3600)**2

        # Point source thruput
        tau_PS = tau_core*tau_refl
                
        A_col = f_s*D_PM**2*(1 - f_o)
        
        F_s = F_0*10**(-0.4*m_s)*u.ph/u.s/u.m**2
        F_P_s = 10**(-0.4*dMag)
        F_p = F_P_s*F_s
                
        m_pixCG = A_PSF*(D_PM/(lam_d*k_s))**2*(np.pi/180/3600)**2
        
        F_ezo = F_0*fEZ*u.arcsec**2*u.ph/u.s/u.m**2
        F_lzo = F_0*fZ*u.arcsec**2*u.ph/u.s/u.m**2
        
        tau_unif = tau_occ*tau_refl
        
        r_pl = f_SR*F_p*A_col*tau_PS*eta_QE
        r_sp = f_SR*F_s*C_CG*I_pk*m_pixCG*tau_refl*A_col*eta_QE
        r_ezo = f_SR*F_ezo*A_PSF*A_col*tau_unif*eta_QE
        r_lzo = r_ezo*F_lzo/F_ezo
        
        r_ph = (r_pl + r_sp + r_ezo + r_lzo)/m_pix
                
        k_EM = round(-5*k_RN/np.log(0.9), 2)
        L_CR = 0.0323*k_EM + 133.5
        
        k_e = t_f*r_ph
        eta_PC = 0.9
        eta_HP = 1 - t_MF/20
        eta_CR = 1 - 8.5/u.s*t_f*L_CR/1024**2
        eta_NCT = [CTE_derate*max(0, min(1 + t_MF*(0.858 - 1), 1 + t_MF*(0.858 - 1 + 3.24*(i - 0.089)))) for i in k_e]
        
        deta_QE = eta_QE*eta_PC*eta_HP*eta_CR*eta_NCT
        
        f_b = 10**(0.4*dmag_s)
        
        k_sp = 1 + 1/(f_ref*f_b)
        k_det = 1 + 1/(f_ref*f_b**2)
        k_CIC = k_d*(k_EM*4.337e-6 + 7.6e-3)
        
        i_d = k_d*(1.5 + t_MF/2)/u.s/3600
                
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
            TK (TimeKeeping module):
                TimeKeeping class object
        
        Returns:
            intTime (astropy Quantity array):
                Integration times in units of day
        
        """
        
        # electron counts
        C_p, C_b, C_sp = self.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dMag, WA, mode, TK=TK)
        
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
    
    def calc_minintTime(self, TL):
        """Finds minimum integration times for the target list filtering.
        
        This method is called in the TargetList class object. It calculates the 
        minimum (optimistic) integration times for all the stars from the target list, 
        in the ideal case of no zodiacal noise. It uses a very favorable planet flux
        ratio (dMag0, 15 by default) and working angle (WA0, by default equal to 
        the detection IWA-OWA midpoint).
        
        Args:
            TL (TargetList module):
                TargetList class object
        
        Returns:
            minintTime (astropy Quantity array):
                Minimum integration times for target list stars in units of day
        
        """
        
        # select detection mode
        mode = filter(lambda mode: mode['detectionMode'] == True, self.observingModes)[0]
        
        # define attributes for integration time calculation
        sInds = np.arange(TL.nStars)
        fZ = 0./u.arcsec**2
        fEZ = 0./u.arcsec**2
        dMag = self.dMag0
        WA = self.WA0
        
        # calculate minimum integration time
        minintTime = self.calc_intTime(TL, sInds, fZ, fEZ, dMag, WA, mode, TK={'currentTimeNorm':0*u.d})
        
        return minintTime

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