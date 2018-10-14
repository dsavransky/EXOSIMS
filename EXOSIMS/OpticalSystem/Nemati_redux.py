from EXOSIMS.OpticalSystem.Nemati import Nemati
import astropy.units as u
from astropy.io import fits
import astropy.constants as const
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

access_path = '../../../../EXOSIMS/EXOSIMS-master/EXOSIMS/OpticalSystem/'

class Nemati_redux(Nemati):
    """Nemati Optical System class
    
    This class contains all variables and methods necessary to perform
    Optical System Module calculations in exoplanet mission simulation using
    the model from Nemati 2014.
    
    Args:
        \*\*specs:
            user specified values
    
    """
    
    def __init__(self, **specs):
        
        Nemati.__init__(self, **specs)
        
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
    
    """ADDED FUNCTIONS - jturner"""

    def scenario_info(self, ScenarioSelected):
        """Returns information about the selected scenario.
        Colimns:
            Scenario (string)
            Center wavelength [nm] (float)
            BW (float)
            FP Type (string)
            Coronagraph (string)
            Months at L2 [mos] (float)
            R Required (float)
            t_max [hrs] (float)
            Required SNR (float)
            Fiducial Planet (string)
            Ann Zone (float)
            t_frame (float)
            Required t_core (float)
            Disturbance (string);           Requirement Level Performance
            Sensitivity (string);           "
            Disturbance MUF (string);       "
            pp Factor (float);              "
            t_core MUF (float);             "
            Duty Factor (float);            "
            Throughtput Derate (string);    "
            Det Model (string);             "
            Disturbance (string);           CBE Level Performance
            Sensitivity (string);           "
            Disturbance MUF (string);       "
            pp Factor (float);              "
            t_core MUF (float);             "
            Duty Factor (float);            "
            Throughtput Derate (string);    "
            Det Model (string);             "
            
        Args:
            ScenarioSelected (string):
                Scenario to match in table of all scenario data
        
        Returns:
            scenario (list):
                List containing information about selected scenario
            
        """
        # Get full scenarioTable data
        scenariolines = open(access_path + 'scenarioTable.txt').read().splitlines()[1:]
        scenarioTable = [i.split('\t') for i in scenariolines]
        scenarioTable = self.str2float(scenarioTable)
        # Get row matching ScenarioSelected
        scenarios = [i[0] for i in scenarioTable]
        row = scenarios.index(ScenarioSelected)
        scenario = scenarioTable[row]
        return scenario
    
    def Throughput(self, lambda_nm, derate_factor):
        """Creates dictionary of throughputs for various coronagraph elements 
        based on coatings/material used. Calculates transmission/reflection
        throughput.
        
        Args:
            lambda_nm (astropy nm)
            derate_factor (float)
            
        Returns:
            tau_DIpathRT (float): Imaging throughput
            tau_IFSpathRT (float): IFS throughput
        
        """
        # Data for HRC coronagraph mask                                         (Throughput!S5:AG405)
        hdu1 = fits.open(access_path + 'HRC.fits')
        HRC_data = hdu1[1].data
        # HRC typical reflection (closest wavelength)                           (Throughput!D8)
        HRC_wave = HRC_data.field('Wavelength')
        row = min(range(len(HRC_wave)), key = lambda x: abs(HRC_wave[x]*1e3 - lambda_nm/u.nm))
        HRC = HRC_data.field(1)[row]*derate_factor
        # Data for FS99 coronagraph mask                                        (Throughput!AI5:AW405)
        hdu2 = fits.open(access_path + 'FS99.fits')
        FS99_data = hdu2[1].data
        # FSS99_600 typical reflection (closest wavelength)                     (Throughput!E8)
        FS99_wave = FS99_data.field('Wavelength')
        row = min(range(len(FS99_wave)), key = lambda x: abs(FS99_wave[x]*1e3 - lambda_nm/u.nm))
        FSS99_600 = FS99_data.field(1)[row]*derate_factor
        # Data for Aluminum coronagraph mask                                    (Throughput!BP5:CD405)
        hdu3 = fits.open(access_path + 'AL.fits')
        AL_data = hdu3[1].data       
        # Al typical reflection (closest wavelength)                            (Throughput!F8)
        AL_wave = AL_data.field('Wavelength')
        row = min(range(len(AL_wave)), key = lambda x: abs(AL_wave[x]*1e3 - lambda_nm/u.nm))
        Al = AL_data.field(1)[row]*derate_factor
        # BBARx2 typical transmission                                           (Throughput!H9)
        BBARx2 = 0.99
        # Throughputs for coronagraph elements
        # {Element: [Hybrid Lyot Coronagraph Imaging Path, Shaped Pupil Coronagraph IFS Path]}
        HLC_Im_thruput = {'AFTA Pupil':         [1,         1],
                          'T1':                 [HRC,       HRC],
                          'T2':                 [HRC,       HRC],
                          'COR F1':             [FSS99_600, FSS99_600],
                          'COR F2':             [FSS99_600, FSS99_600],
                          'M3':                 [FSS99_600, FSS99_600],
                          'COL F1':             [FSS99_600, FSS99_600],
                          'M4':                 [FSS99_600, FSS99_600],
                          'COL F2':             [FSS99_600, FSS99_600],
                          'Pupil@FSM':          [FSS99_600, FSS99_600],
                          'FSM':                [FSS99_600, FSS99_600],
                          'R1 OAP1':            [FSS99_600, FSS99_600],
                          'Focusing Mirror':    [FSS99_600, FSS99_600],
                          'R1 OAP2':            [FSS99_600, FSS99_600],
                          'DM1':                [Al,        Al],
                          'DM2':                [Al,        Al],
                          'R2 OAP1':            [FSS99_600, FSS99_600],
                          'FM':                 [FSS99_600, FSS99_600],
                          'R2 OAP2':            [FSS99_600, FSS99_600],
                          'HLC-FM':             [FSS99_600, 1],
                          'SPC-SP Mask':        [1,         Al],
                          'R3 OAP1':            [FSS99_600, FSS99_600],
                          'FPM-HLC':            [BBARx2,    1],
                          'FPM-SPC':            [1,         1],
                          'R3 OAP2':            [FSS99_600, FSS99_600],
                          'Lyot-HLC':           [1,         1],
                          'Lyot-SPC':           [1,         1],
                          'FS OAP1':            [FSS99_600, FSS99_600],
                          'Field Stop':         [1,         1],
                          'FS OAP2':            [FSS99_600, FSS99_600],
                          'Imgr/pupil/IFS':     [FSS99_600, FSS99_600],
                          'Polarizer':          [1,         1],
                          'Rad shield M':       [FSS99_600, 1],
                          'Img Cam':            [1,         1],
                          'IFS FM1':            [1,         FSS99_600],
                          'IFS RM1':            [1,         FSS99_600],
                          'IFS RM2':            [1,         FSS99_600],
                          'IFS FM2':            [1,         FSS99_600],
                          'IFS lenslets':       [1,         BBARx2],
                          'IFS Pinhole Array':  [1,         0.8],
                          'IFS Lens1':          [1,         BBARx2],
                          'IFS Prism1':         [1,         BBARx2],
                          'IFS Prism2':         [1,         BBARx2],
                          'IFS Lens2':          [1,         BBARx2],
                          'IFS FM3':            [1,         FSS99_600],
                          'IFS Cam':            [1,         1],
                          'LOWFS L1':           [1,         1],
                          'LOWFS L2':           [1,         1],
                          'LOWFS L3':           [1,         1],
                          'LOWFS Cam':          [1,         1]}
        # tau_DIpathRT, tau_IFSpathRT are products of throughputs
        tau_DIpathRT = 1
        tau_IFSpathRT = 1
        for key, value in HLC_Im_thruput.iteritems():
            tau_DIpathRT = tau_DIpathRT*value[0]
            tau_IFSpathRT = tau_IFSpathRT*value[1]
        return tau_DIpathRT, tau_IFSpathRT
    
    def CG_info(self, CGcase, tau_DIpathRT = 0, tau_IFSpathRT = 0):
        """Returns list containing information about the coronagraphs
        Columns:
            CG (string)
            Name (string)
            BW (float)
            Cor for Pol(int)
            Polarize Loss (int)
            Sampling (float)
            t_refl (float)
        
        Args:
            CGcase (string): Coronagraph type
            tau_DIpathRT (float): Imaging throughput
            tau_IFSpathRT (float): IFS throughput
            
        Returns:
            CGrow (list):
                List containing information about the coronagraphs
                
        """
        # Get coronagraph data table
        CGlines = open(access_path + 'CGtable.txt').read().splitlines()
        CGtable = [i.split('\t') for i in CGlines]
        CGtable = self.str2float(CGtable)
        # Modify table
        for i in range(len(CGtable)):
            if i in range(1, 4):
                CGtable[i][-1] = tau_IFSpathRT
            else:
                CGtable[i][-1] = tau_DIpathRT
        CGcases = [i[0] for i in CGtable]
        CGrow = CGtable[CGcases.index(CGcase)]
        return CGrow
    
    def fiducial_info(self, AU, R_jupiter, Fiducial_planet_band):
        """Returns specified band information for fiducial planet.
        
        Args:
            AU (astropy AU): Astronomical unit
            R_jupiter (astropy R_jup): Jupiter radius
        
        Returns:
            fid_planet (list):
                List containing information about the fiducial planet.
        
        """
        # Get fiducial planet information                                       (Scenario!C44:I51)
        fiducial_lines = open(access_path + 'fiducial_table.txt').read().splitlines()
        fiducial_table = [i.split('\t') for i in fiducial_lines]
        fiducial_table = self.str2float(fiducial_table)
        # Modify table     
        for i in range(len(fiducial_table[-1])):
            fiducial_table[-1][i] = fiducial_table[5][i]*AU*np.sqrt(fiducial_table[1][i]*0.000000001/fiducial_table[7][i])/R_jupiter
        fid_col_ind = fiducial_table[0].index(Fiducial_planet_band)
        fid_planet = [i[fid_col_ind] for i in fiducial_table]
        return fid_planet
        
    def planet_info(self, Distance = 0, Host_V = 0, Radius = 0, SMA = 0, Ph_Ang = 0, Albedo = 0, select_Vmag1 = 0, select_Vmag2 = 0, change_SMA1 = 0, change_SMA2 = 0, tau_ceti_e_phase_angle = 0, n_zodi = 0):
        """Returns information on possible simulation target planets. Must be
        called twice since some elements are dynamically dependent on other
        independent elements in table. First called with no argurments, returns
        table which yields arguments for second call.
        
        Args: 
            Distance (float): Distance from observation to star
            Host_V (float): Magnitude of host star
            Radius (float): Radius of planet in Jupiter radii
            SMA (float): Semimajor axis of planet
            Ph_Ang (float): Phase angle of planet
            Albedo (float): Albedo of planet
            select_Vmag1 (float):
            select_Vmag2 (float):
            change_SMA1 (float):
            change_SMA2 (float):
            tau_ceti_e_phase_angle (float):
            n_zodi (float):
        
        Returns:
            planet_table (list):
                List containing information about target planets.
        
        """
        # Information on target planet                                          (SNR!A53:N157)
        planet_lines = open(access_path + 'planet_table.txt').read().splitlines()
        planet_table = [i.split('\t') for i in planet_lines]
        planet_table = self.str2float(planet_table)
        # Fist call
        if Distance == 0:
            return planet_table
        # Second call - modify table
        else:
            planet_table[0][2] = Host_V
            planet_table[0][3] = Distance
            planet_table[0][6] = SMA
            planet_table[0][9] = Ph_Ang
            planet_table[0][10] = Albedo
            planet_table[0][11] = Radius
            planet_table[1][2] = select_Vmag1
            planet_table[1][6] = change_SMA1
            planet_table[2][2] = select_Vmag2
            planet_table[2][6] = change_SMA2
            planet_table[6][9] = tau_ceti_e_phase_angle
            planet_table[6][10] = planet_table[6][4]*(np.sin(np.deg2rad(tau_ceti_e_phase_angle)) + (np.pi - np.deg2rad(tau_ceti_e_phase_angle))*np.cos(np.deg2rad(tau_ceti_e_phase_angle)))/np.pi
            for i in range(42, 105):
                planet_table[i][5] = n_zodi
            return planet_table
        
    def flux_info(self, wl_bandlo, wl_bandhi, Ephoton, Spectral_Type):
        """Calculates 0-magnitude flux for specified stellar type
        
        Args:
            wl_bandlo (astropy nm): Band minimum wavelength
            wl_bandhi (astropy nm): Band maximum wavelength
            Ephoton (astropy kg m2 s-2): Photon energy at central wavelength
            Spectral_Type (str): Stellar type name
        
        Returns:
            Integrated_Flux (astropy ph m-2 s-1): 
                0-magnitude flux for specified stellar type
        
        """
        # Spectral flux at 0-magnitude for specified stellar types              (Spectra!B3:L804)
        hdu = fits.open(access_path + 'spectra.fits')
        data = hdu[1].data
        wavelengths = data.field(0)
        # Wavelength step size
        dlambda = (wavelengths[1] - wavelengths[0])*u.m # m
        # Flux corresponding to specified stellar type
        intensities = data.field(Spectral_Type)
        # Reimann integration of flux (step size is very small)
        intensity = 0
        for j in range(len(wavelengths)):
            w = (wavelengths[j]*u.m).to(u.nm)
            if w >= wl_bandlo and w <= wl_bandhi:
                intensity += intensities[j]
        intensity = intensity*(u.ph*u.kg/(u.s*u.s*u.s*u.m)) # kg m-1 s-3
        Integrated_Flux = intensity*dlambda/Ephoton # ph m-2 s-1
        return Integrated_Flux
    
    def QE_info(self, val, lambda_nm):
        """Returns quantum effiiency of detector at specified wavelength
        
        Args:
            val (str): Quantum efficiency curve name
            lambda_nm (astropy nm): Central wavelength
            
        Returns:
            QE (float): Quantum efficiency
        
        """
        # Quantum efficiency table                                              (Detector!P9:X90)
        hdu = fits.open(access_path + 'QEtable.fits')
        data = hdu[1].data
        wavelengths = data.field(0)
        col = data.field(val)
        row = min(range(len(wavelengths)), key = lambda x: abs(wavelengths[x]*u.nm - lambda_nm))
        QE = col[row]
        return QE
    
    def error_info(self, planetNo, scenario, planetWA, DisturbanceCase, SensCaseSel, MUFcaseSel, CS_NmodeCoef, CS_Nstat, CS_Nmech, CS_Nmuf):
        """Calcuates error information given observation specifications.
        
        Args:
            planetNo (int): Target planet number
            scenario (str): Scenario
            planetWA (float): Planet observing WA
            DisturbanceCase (str): Disturbance case
            SensCaseSel (str): Sensitivity case
            MUFcaseSel (str): MUF case
            CS_NmodeCoef (int): Contrast stability mode coefficients
            CS_Nstat (int): Contrast stability statistics
            CS_Nmech (int): Contrast stability mechanics
            CS_Nmuf (int): Contrast stability MUF casses
            
        Returns:
            error_table (list): Error information for selected observation                
            AnnZone (float): Annular zone
            MUFindex (int): MUF index
                
        """
        # MUF table                                                             (SensitivityMUF!C3:G23)
        MUFlines = open(access_path + 'MUF_table.txt').read().splitlines()
        MUFtable = [i.split('\t') for i in MUFlines]
        MUFtable = self.str2float(MUFtable)
        MUFcases = MUFtable[0]
        MUFtable = MUFtable[2:]
        # MUF index                                                             (Scenario!Q48)
        MUFindex = MUFcases.index(MUFcaseSel)
        # MUFs                                                                  (C Stability!E48:E68)
        MUF = [] # 1x21
        for i in range(21):
            MUF.append(MUFtable[i][MUFindex])   
        # Disturbance live                                                      (DisturbanceLive!AI4:AI1179)
        DisturbanceLivelines = open(access_path + 'DisturbanceLive_table.txt').read().splitlines()
        DisturbanceLivetable = [i.split('\t') for i in DisturbanceLivelines]
        DisturbanceLivetable = self.str2float(DisturbanceLivetable, 0)
        # Modify disturbance live table
        DisturbanceLive = [] # 1x1176
        rows = 21
        cols = 28
        for i in range(1176):
            Liverow = i%rows + (rows+4)*(i/(rows*cols)) + 3
            Livecol = (i/rows)%cols + 1
            DisturbanceLive.append(DisturbanceLivetable[Liverow][Livecol])
        # Disturbance table                                                     (Disturbance!H6:U1181)
        Disturbancelines = open(access_path + 'Disturbance_table.txt').read().splitlines()[1:]
        Disturbancetable = [i.split('\t') for i in Disturbancelines]
        Disturbancetable = self.str2float(Disturbancetable)
        for i in range(len(Disturbancetable)):
            Disturbancetable[i][3] = DisturbanceLive[i]
        # Disturbances                                                          (Disturbance!H4:U4)
        DisturbanceCaseList = ['rqt10hr', 'rqt10hr_1mas', 'rqt171012', 'live', 'rqt10hr171212', 'rqt40hr171212', 'rqt10hr171221', 'rqt40hr171221', 'cbe10hr171221', 'rqt10hr180109', 'rqt40hr180109', 'cbe10hr180109', 'cbe10hr180130', 'cbe10hr180306']
        DisturbanceCaseIndex = DisturbanceCaseList.index(DisturbanceCase)
        # Contrast stability disturbance table                                  (C Stability!R40:AS60)
        Disturbances = [] # 21x28
        for i in range(21):
            Disturbance_row = []
            for j in range(28):
                row = i + CS_NmodeCoef*(j + CS_Nstat*CS_Nmech*MUFindex)
                col = DisturbanceCaseIndex
                Disturbance_row.append(Disturbancetable[row][col])
            Disturbances.append(Disturbance_row)   
        # Contrast sensitivity vectors                                          (Sensitivities!D5:J109)
        Contrastlines = open(access_path + 'Contrast_table.txt').read().splitlines()
        Contrasttable = [i.split('\t') for i in Contrastlines]
        Contrasttable = self.str2float(Contrasttable)
        # Annular Zone table                                                    (AnnZoneList!C3:P11)
        AnnZonelines = open(access_path + 'AnnZone_table.txt').read().splitlines()
        AnnZonetable = [i.split('\t') for i in AnnZonelines]
        AnnZonetable = self.str2float(AnnZonetable)
        # Contrast sensitivity vector names                                     (Sensitivities!D4:R4)
        SensCases = ['HLC150818_Dwight', 'SPCgen3', 'SPC170714_design', 'HLC150818_BBit61', 'SPC170714_10per', 'HLC150818_61_10per', 'SPC170831_design', 'future 4', 'future 5', 'future 6', 'future 7', 'future 8', 'future 9', 'future 10', 'future 11']
        Senscol = SensCases.index(SensCaseSel)
        AnnZoneLookupTable = [i[Senscol] for i in AnnZonetable]
        # Annular zone                                                          (SNR!AJ41)
        if planetNo == 1:
            AnnZone = scenario[10]
        else:
            AnnZone_val = max(AnnZoneLookupTable[0], min(AnnZoneLookupTable[4], np.floor(planetWA)))
            AnnZone = AnnZoneLookupTable.index(AnnZone_val)
        # Sensitivity cases                                                     (C Stability!H48:H68)
        SensCaseSels = [] # 1x21
        for i in range(21):
            Sensrow = i + CS_NmodeCoef*AnnZone
            SensCaseSels.append(Contrasttable[int(Sensrow)][int(Senscol)])
        # Stabilities                                                           (C Stability!F48:F68)
        Sensitivities = [] # 1x21
        for i in range(21):
            Sensitivities.append(MUF[i]*SensCaseSels[i])
        # Unit multipliers                                                      (C Stability!AU40:AU60)
        unit_mult = [float(i) for i in open(access_path + 'unit_mult.txt').readlines()] # 1x21
        # Error mechanism table                                                 (C Stability!R6:AS29)
        Error_Mechanism  = [] # 21x28
        for i in range(21):
            Error_row = []
            for j in range(28):
                Error_row.append(Sensitivities[i]*(Disturbances[i][j]*unit_mult[i])**2)
            Error_Mechanism.append(Error_row)
        # Total errors                                                          (C Stability!R31:AS31)
        totals = [] # 1x28
        for i in range(len(Error_Mechanism[0])):
            Error_col = [j[i] for j in Error_Mechanism]
            totals.append(sum(Error_col))
        # Error statistics                                                      (C Stability!D26:G32)
        error_table = [] # 7x4
        c = 0
        for i in range(7):
            row = []
            for i in range(4):
                row.append(totals[c]*0.000000001)
                c += 1
            error_table.append(row)
        return error_table, AnnZone, MUFindex 
    
    def DetectorModel_info(self, DetectorEOLmonths, MissionEpoch, Dark_derate, missionFraction, CTE_derate, dqeFluxSlope, dqeKnee, dqeKneeFlux, SignalPerPixPerFrame):
        # Corrected for 0-indexing
        # Detector!B12:M18  DectectorModelParTable
        DectectorModelParTablelines = open(access_path + 'DetectorModelParTable.txt').read().splitlines()
        table = [i.split('\t') for i in DectectorModelParTablelines]
        table = self.str2float(table)
        table[0][9] = Dark_derate*(1.5 + (MissionEpoch/DetectorEOLmonths)*(2 - 1.5))
        for i in range(1, 5):
            table[i][9] = 1.5 + (MissionEpoch/DetectorEOLmonths)*(2 - 1.5)
        for i in range(0, 5):
            table[i][11] = 0.05*missionFraction
        max_cond = max(0, min(1 + missionFraction*(dqeKnee - 1), 1+ missionFraction*(dqeKnee - 1) + missionFraction*dqeFluxSlope*(SignalPerPixPerFrame - dqeKneeFlux)))
        table[0][10] = CTE_derate*max_cond
        table[1][10] = max_cond
        table[2][10] = 0.5*(1 + max_cond)
        table[3][10] = 0.93*max_cond
        table[4][10] = 0.5*(1 + max_cond)
        DetectorModelParTable = table
        return DetectorModelParTable    
    
    def str2float(self, table, zero = 1):
        """Convert all numerical values into floats
        
        Args:
            table (list): List to be converted
            zero (int): 
                Optional: Default is 1, in which case non-numerical values are
                left unchanged. If set to 0, non-numerical vlaues are set to 0.
        
        Returns:
            table (list): Converted table
            
        """
        # Iterate through table. Try converting all values to floats, if exception
        # is raised, leave value alone / set to 0 based on optional parameter.
        for i in range(len(table)):
            for j in range(len(table[i])):
                try:
                    table[i][j] = float(table[i][j])
                except:
                    if zero == 1:
                        table[i][j] = table[i][j]
                    if zero == 0:
                        table[i][j] = 0.0
        return table
    
    def Cp_Cb_Csp_MOD(self, TL, sInds, fZ, fEZ, dMag, WA, mode, returnExtra=False):
        # Astronomical constants
        AU = const.au
        R_jupiter = const.R_jup
        pc = const.pc
        arcsec = 1*u.arcsec.to(u.rad)
        h_planck = const.h
        c_light = const.c
        # Scenario Name                                                         (Scenario!D36)
#        ScenarioSelected = 'SPEC IFS - Band 3'        
        ScenarioSelected = 'IMG NFOV - Band 1'       
        # Fiducial planet number                                                (SNR!H4)
        planetNo = 1 
        # Case type                                                             (Error Budget!BI3)
        Case_Type = 'CBE'
        # Quantum efficiency curve                                              (SNR!AJ33)
        QE_curve = 'e2v_Spec'
        # Call scenario_info to get scenario information                        (Scenario!C5:AE21)
        scenario = self.scenario_info(ScenarioSelected)
        # Center wavelength                                                     (Scenario!D24)
        lambda_nm = scenario[1]*u.nm # nm
        # Table necessary for selecting perf level                              (Scenario!V42:W46)
        Offset_table = {'REQ':       0,
                        'CBE':       8,
                        'Goal':      8,
                        'Unity MUF': 8,
                        'Best Case': 8}
        # Offset                                                                (Scenario!W39)
        Offset = Offset_table[Case_Type]
        # Reflectivity derate                                                   (Throughput!04:P5)
        Reflectivity_derate = {'REQ': 0.992,
                               'CBE': 1.0}
        # Throughput derate                                                     (Scenario!K27)
        derate_schedule = scenario[Offset + 19]
        # Derate factor                                                         (Throughput!E5)
        derate_factor = Reflectivity_derate[derate_schedule]
        # tau_DIpathRT, tau_IFSpathRT                                           (Throughput!D66, M66)
        tau_DIpathRT, tau_IFSpathRT = self.Throughput(lambda_nm, derate_factor)
        # Coronagraph                                                           (Scenario!G24)
        Coronagraph_name = scenario[4]
        # Call CG_info to get coronagraph information                           (SNR!C39:I46)
        coronagraph = self.CG_info(Coronagraph_name, tau_DIpathRT, tau_IFSpathRT)
        ## START calculations for dynamic data in planet information table
        # Fiducial planet information                                           (Scenario!L24)
        Fiducial_planet_band = scenario[9]
        fid_planet = self.fiducial_info(AU, R_jupiter, Fiducial_planet_band)
        # Star Distance                                                         (Scenario!M44)      
        Distance = fid_planet[4]
        # Star magnitude                                                        (Scenario!M45)
        Host_V = fid_planet[3]
        # Planet redius                                                         (Scenario!M46)
        Radius = fid_planet[8]
        # Planet semimajor axis                                                 (Scenario!M47)
        SMA = fid_planet[5]
        # Planet phase angle                                                    (Scenario!M48)
        Ph_Ang = fid_planet[6]
        # Planet albedo                                                         (Scenario!M49)
        Albedo = fid_planet[7]
        # Star Vmag                                                             (SNR!T67)
        select_Vmag1 = 5.0
        # Dust disk sensitivity Vmag                                            (SNR!T135)
        select_Vmag2 = 5.0
        # Star WA                                                               (SNR!T68)
        select_WA1 = 3.2
        # Dust Disk sensitivity WA                                              (SNR!T136)
        select_WA2 = 7.0
        # Primary mirror diameter                                               (SNR!AJ130)
        D_PM = 2.37*u.m # m
        # Tau ceti e phase angle                                                (Scenario!AQ7)
        tau_ceti_e_phase_angle = 90.0
        # n zodi                                                                (SNR!AR65)
        n_zodi = 20.0
        # Information on target planets                                         (SNR!A53:N157)
        planet_table_pre = self.planet_info()
        planet_pre = planet_table_pre[planetNo - 1]
        # Star Distance                                                         (SNR!AJ10)
        Star_distance = planet_pre[3]*u.pc # pc
        # Planet phase angle                                                    (SNR!AJ14)
        orbitPhaseAngle = planet_pre[9]*u.deg # deg
        # Change in semimajor axis                                              (SNR!T69)
        change_SMA1 = select_WA1*(lambda_nm/D_PM).decompose()*Star_distance*(pc/AU)/np.sin(np.deg2rad(orbitPhaseAngle))
        # Dust Disk sensitivity changin in semimajor axis                       (SNR!T137)
        change_SMA2 = select_WA2*(lambda_nm/D_PM).decompose()*Star_distance*(pc/AU)/np.sin(np.deg2rad(orbitPhaseAngle))
        ## END calculations for dynamic data in planet information table
        # Information on target planet
        planet_table = self.planet_info(Distance, Host_V, Radius, SMA, Ph_Ang, Albedo, select_Vmag1, select_Vmag2, change_SMA1, change_SMA2, tau_ceti_e_phase_angle, n_zodi)
        planet = planet_table[planetNo - 1]
        # Planet semimajor axis                                                 (SNR!AJ15)
        Planet_SMA = planet[6]*u.AU # AU
        # Planet albedo                                                         (SNR!AJ16)
        Planet_Albedo = planet[10]
        # Planet Radius                                                         (SNR!AJ17)
        Planet_Radius = planet[11]
        # Planet flux ratio                                                     (SNR!AJ65)
        planetFluxRatio = Planet_Albedo*(Planet_Radius*R_jupiter/(Planet_SMA*AU/u.AU))**2
        # Spectral type                                                         (SNR!AJ12)
        Spectral_Type = planet[8]
        # Planet separation phase angle                                         (SNR!AJ66)
        Pl_separation = ((Planet_SMA*np.sin(orbitPhaseAngle))/(Star_distance)).decompose()/u.mas.to(u.rad)*u.mas # mas
        # Diffraction limit                                                     (SNR!AJ71)
        lam_over_D = (lambda_nm/D_PM).decompose()/u.mas.to(u.rad)*u.mas # mas
        # Planet positional WA at specified orbital phase angle                 (SNR!AJ67)
        Planet_Positional_WA = 1e-10 + Pl_separation/lam_over_D
        # Coronagraph type
        CG_Type = coronagraph[1]
        # Coronaph information
        CG_lines = open(access_path + CG_Type + '.txt').read().splitlines()
        CG = [i.split('\t') for i in CG_lines]
        CG = self.str2float(CG)
        # Maximum working angle                                                 (SNR!AR17)
        OWA = np.max([i[0] for i in CG[1:]])
        # Minimum working angle                                                 (SNR!AR16)
        IWA = np.min([i[0] for i in CG[1:]])
        # Planet observing WA                                                   (SNR!AJ68)
        if Planet_Positional_WA < OWA:
            planetWA = Planet_Positional_WA 
        else:
            planetWA = IWA + 0.8*(OWA - IWA)
        # Coronagraph information for planet observing WA
        CG_lam = [i[0] for i in CG[1:]]
        SPCrow = min(range(len(CG_lam)), key = lambda i: abs(CG_lam[i] - planetWA) if CG_lam[i] < planetWA else max(CG_lam))
        CGrow = CG[1:][SPCrow]
        # PSF area on sky                                                       (SNR!AR14)
        PSF_Area = CGrow[6]*u.arcsec*u.arcsec # as2
        # Design Wavelength                                                     (SNR!AR9)
        Design_Wavelength = (D_PM*CGrow[1]*(arcsec)/CGrow[0]).to(u.nm) # nm
        # Focal plane type                                                      (SNR!AJ37)
        Focal_Plane_Type = scenario[3]
        # Resolution                                                            (Scenario!I24)
        Resolution = scenario[6]
        # Bandwidth                                                             (Scenario!E24)
        bandwidth = scenario[2]
        # SNR!AJ91          f_SR
        if Focal_Plane_Type == 'IFS':
            # f_sr_IFS: -1 if resolution undefined                              (SNR!AR46)
            try:
                f_SR = 1.0/(Resolution*bandwidth)
            except:
                f_SR = -1.0
            # Nyquist sampled at this wavelength (from IFS1 and IMG1)           (SNR!AR48)
            Critical_lambda = 705*u.nm # nm
            # Total lenslets covered by core                                    (SNR!AJ94)
            IFS_Lensl_per_PSF = 5.0
            # Pixels per spectral element                                       (SNR!AJ96)
            IFS_Spectral_Samp = 2.0
            # Pixels in spatial direction                                       (SNR!AJ95)
            IFS_Spatial_Samp = 2.0
            # Number of pixels contributing to ROI for SNR computation          (SNR!AR47)
            mpix = IFS_Lensl_per_PSF*(lambda_nm/Critical_lambda)**2*IFS_Spectral_Samp*IFS_Spatial_Samp
        else:
            # f_sr_Imaging                                                      (SNR!AS46)
            f_SR = 1
            # Critical imaging wavelength, Nyquist sampled (from IFS1 and IMG1) (SNR!AS48)
            Critical_lambda = 508*u.nm # nm
            # Number of pixels contributing to ROI for SNR computation          (SNR!AS47)
            mpix = PSF_Area*(arcsec/u.arcsec)**2*(lambda_nm/Design_Wavelength)**2*(2*(D_PM/Critical_lambda).decompose())**2
        # Band minimum wavelength                                               (SNR!AJ99)
        wl_bandlo = lambda_nm*(1 - bandwidth/2)
        # Band maximum wavelength                                               (SNR!AJ100)
        wl_bandhi = lambda_nm*(1 + bandwidth/2)
        # Photon energy                                                         (SNR!AJ102)
        Ephoton = (h_planck*c_light/lambda_nm).decompose() # kg m2 s-2
        # 0-magnitude flux                                                      (SNR!AJ72)
        zero_Mag_Flux = self.flux_info(wl_bandlo, wl_bandhi, Ephoton, Spectral_Type) # ph s-1 m-2
        # Star apparent magnitude                                               (SNR!AJ11)          
        Star_apparent_mag = planet[2]*u.mag # mag
        # Star flux                                                             (SNR!AJ73)
        Star_Flux = zero_Mag_Flux*10**(-0.4*Star_apparent_mag/u.mag) # ph m-2 s-1
        # Planet flux                                                           (SNR!AJ74)
        planet_Flux = planetFluxRatio*Star_Flux # ph m-2 s-1
        # Sec. obscuration fraction                                             (SNR!AJ131)
        Sec_Obs_frac = 0.32
        # Struts obscuration                                                    (SNR!AJ132)
        Struts_obscuration = 0.07
        # Clear aperture fraction                                               (SNR!AJ133)
        Clear_aperture_fraction = (1 - Struts_obscuration)*(1 - Sec_Obs_frac**2)
        # Collection area                                                       (SNR!AJ135)
        Acol = ((np.pi/4)*D_PM**2)*Clear_aperture_fraction # m2
        # Correction for inclination polarization                               (SNR!AR19)
        Correction_Incl_pol = 1 + coronagraph[3]
        # Core Throughput                                                       (SNR!AR12)
        Core_Throughput = CGrow[4]*Correction_Incl_pol
        # MUF Case Selected                                                     (Scenario!I27)
        MUF_Thp = scenario[Offset + 17]
        # Required tau core                                                     (Scenario!O24)
        REQ_tau_core = scenario[12]
        # Core throughput                                                       (SNR!AJ108)
        if derate_schedule == 'CBE':
            t_core = Core_Throughput*MUF_Thp
        else:
            t_core = REQ_tau_core
        # Occular throughput                                                    (SNR!AR15)
        t_occ = CGrow[7]*Correction_Incl_pol
        # PSF throughput                                                        (SNR!AJ109)
        try:
            t_psf = t_core/t_occ
        except:
            t_psf = 0
        # Reflection throughput                                                 (SNR!AR18)
        t_refl = coronagraph[6]
        # Filter throughput                                                     (SNR!AJ115)
        t_filt = 0.9
        # Polarizer throughput                                                  (SNR!AR20)
        t_pol = coronagraph[4]
        # Uniform throughput                                                    (SNR!AJ119)
        t_unif = t_occ*t_refl*t_filt*t_pol
        # Planet throughput                                                     (SNR!AJ118)
        t_pnt = t_unif*t_psf
        # Quantum efficiency                                                    (SNR!AR37)
        QE = self.QE_info(QE_curve, lambda_nm)
        # Detector EOL months                                                   (Detector!C5)
        DetectorEOLmonths = 5.25*12
        # Months at L2                                                          (Scenario!H24)
        Months_at_L2 = scenario[5]  
        # Radiation exposure                                                    (Scenario!H24)
        Radiation_Exposure = Months_at_L2
        # Radiation dosage                                                      (SNR!AJ35)
        missionFraction = Radiation_Exposure/DetectorEOLmonths
        # Mission epoch                                                         (SNR!AJ34)
        MissionEpoch = Radiation_Exposure
        # Dark noise derate                                                     (Detector!L6)
        Dark_derate = 1.1
        # CTE derate                                                            (Detector!L5)
        CTE_derate = 0.83
        # Radiation damage dependent CTE flux slope                             (Detector!AC37)
        dqeFluxSlope = 3.24*u.s/u.ph # s ph-1
        # Radiation damage dependent CTE knee                                   (Detector!AC38)
        dqeKnee = 0.858
        # Radiation damage dependent CTE knee flux                              (Detector!AC39)
        dqeKneeFlux = 0.089*u.ph/u.s # ph s-1
        # PSF peak intensity                                                    (SNR!AR13)
        Coronagraph_I_pk = CGrow[5]*Correction_Incl_pol
        # Coronagraph intrinsic sampling                                        (SNR!AR21)
        CG_intrinsic_sampling = coronagraph[5]
        # Pixels within PSF core assuming instrinsic sampling                   (SNR!AR22)
        mpixIntrinsic = (PSF_Area*(arcsec/u.arcsec)**2/(CG_intrinsic_sampling*Design_Wavelength/D_PM)**2).decompose()
        # Contrast_per_design, Intensity/PSF peak                               (SNR!AR11)
        Contrast_per_design = CGrow[3]
        # Post processing performance                                           (Scenario!H27)
        k_pp = scenario[16]
        # Selected MUF case                                                     (Scenario!G27)
        MUFcaseSel = scenario[Offset + 15]
        # Contrast stability mode coefficients                                  (C Stability!E38)
        CS_NmodeCoef = 21
        # Contrast stability statistics                                         (C Stability!E39)
        CS_Nstat = 7
        # Contrast stability mechanics                                          (C Stability!E40)
        CS_Nmech = 4
        # Contrast stability MUF casses                                         (C Stability!E40)
        CS_Nmuf = 2
        # Disturbance Case                                                      (Scenario!E27)
        DisturbanceCase = scenario[Offset + 13]
        # Sensitivity case                                                      (Scenario!F27)
        SensCaseSel = scenario[Offset + 14]
        # Error information and byproducts of calculations
        error_info = self.error_info(planetNo, scenario, planetWA, DisturbanceCase, SensCaseSel, MUFcaseSel, CS_NmodeCoef, CS_Nstat, CS_Nmech, CS_Nmuf)
        error_table = error_info[0]
        AnnZone = error_info[1]
        MUFindex = error_info[2]
        # InitialRawContrast!E2:K21
        InitialRawContrastlines = open(access_path + 'InitialRawContrast_table.txt').read().splitlines()
        InitialRawContrasttable = [i.split('\t') for i in InitialRawContrastlines]
        InitialRawContrasttable = self.str2float(InitialRawContrasttable)
        # InitialRawContrast!E1:K1
        InitialRawContrasts = ['HLC150818_Dwight', 'SPC170714_design', 'HLC150818_BBit61', 'SPCgen3	SPC170714_10per', 'HLC150818_61_10per', 'SPC170831_design']
        InitRawrow_M = 2*AnnZone + 10*MUFindex
        InitRawrow_V = 1 + 2*AnnZone + 10*MUFindex
        InitRawcol = InitialRawContrasts.index(SensCaseSel)
        # C Stability!D34   CGI_Initial_NI_M
        CGI_Initial_NI_M = InitialRawContrasttable[int(InitRawrow_M)][int(InitRawcol)]
        # C Stability!E34   CGI_Initial_NI_V
        CGI_Initial_NI_V = InitialRawContrasttable[int(InitRawrow_V)][int(InitRawcol)]     
        # C Stability!D21   raw_NI_M
        raw_NI_M = sum([i[0] for i in error_table]) + CGI_Initial_NI_M
        # C Stability!E21   raw_NI_v
        raw_NI_V = sum([i[1] for i in error_table]) + CGI_Initial_NI_V
        # C Stability!H21   raw_NI_sum
        raw_NI_sum = raw_NI_M + raw_NI_V
        # NItoContrast!B2:H6
        NI_contrastlines = open(access_path + 'NItoContrast_table.txt').read().splitlines()
        NI_contrasttable = [i.split('\t') for i in NI_contrastlines]
        NI_contrasttable = self.str2float(NI_contrasttable)
        NIrow = AnnZone
        NIcol = InitRawcol
        # C Stability!J38   NI_contrast
        NI_contrast = NI_contrasttable[int(NIrow)][int(NIcol)]
        # C Stability!E4    CS_rawContrast
        # = C Stability!E8
        CS_rawContrast = raw_NI_sum/NI_contrast
        # C Stability!F22   Diff_NI_DM
        Diff_NI_DM = np.sqrt(2*raw_NI_M*sum([i[2] for i in error_table]))
        # C Stability!G22   Diff_NI_DV
        Diff_NI_DV = np.sqrt(sum([i[3]**2 for i in error_table]))
        # C Stability!H22   Diff_NI_sum
        Diff_NI_sum = np.sqrt(Diff_NI_DM**2 + Diff_NI_DV**2)
        # C Stability!E9    Diff_contrast
        Diff_contrast = Diff_NI_sum/NI_contrast
        # C Stability!E5    CS_deltaC
        CS_deltaC = Diff_contrast/k_pp
        # SNR!C6:E7         ContrastSrcTable
        ContrastSrcTable = [['CG Design Perf', Contrast_per_design, Contrast_per_design*k_pp*(1/5)],
                            ['Disturb x Sens', CS_rawContrast, CS_deltaC]]
        ContrastSource = [i[0] for i in ContrastSrcTable]
        # Scenario!C40      Contrast_Scenario
        Contrast_Scenario = 'Disturb x Sens'
        # Scenario!D40      SelContrast
        Contrastrow = ContrastSource.index(Contrast_Scenario)
        SelContrast = ContrastSrcTable[Contrastrow][1]
        # SNR!AJ61          Coronagraph_Contrast
        Coronagraph_Contrast = SelContrast
        # SNR!AJ120         t_speckle
        t_speckle = t_refl*t_filt*t_pol
        # SNR!AJ142         k_comp
        k_comp = 1.0
        # SNR!AJ47          sp_bkgRate
        sp_bkgRate = k_comp*f_SR*Star_Flux*Coronagraph_Contrast*Coronagraph_I_pk*mpixIntrinsic*t_speckle*Acol*QE # ph s-1
        # SNR!AJ20          Nzodi
        Nzodi = planet[5]
        # SNR!AJ75          Star_abs_mag
        Star_abs_mag = Star_apparent_mag - 5*np.log10(Star_distance/(10*u.pc))*u.mag # mag
        # SNR!BB34          Msun
        Msun = 4.83*u.mag # mag
        # SNR!AJ19          ExoZodi_1AU
        ExoZodi_1AU = 22.0*u.mag # mag
        # SNR!AJ80          ExoZodi_flux
        ExoZodi_flux = Nzodi*zero_Mag_Flux*10**(-0.4*(Star_abs_mag - Msun + ExoZodi_1AU)/u.mag)/(Planet_SMA**2)*(u.AU*u.AU/(u.arcsec*u.arcsec)) # ph as-2 m-2 s-1
        # SNR!AJ83          ezo_bkgRate
        ezo_bkgRate = f_SR*ExoZodi_flux*PSF_Area*Acol*t_unif*QE # ph s-1
        # SNR!AJ18          Local_Zodi
        Local_Zodi = 23.0
        # SNR!AJ81          Local_Zodi_flux
        Local_Zodi_flux = zero_Mag_Flux*10**(-0.4*Local_Zodi)*(1/(u.arcsec*u.arcsec)) # ph as-2 m-2 s-1
        # SNR!AJ84          lzo_bkgRate
        lzo_bkgRate = f_SR*Local_Zodi_flux*PSF_Area*Acol*t_unif*QE # ph s-1
        # SNR!AJ49          zo_bkgRate
        zo_bkgRate = ezo_bkgRate + lzo_bkgRate
        # SNR!AJ45          pl_convertRate
        pl_convertRate = f_SR*planet_Flux*Acol*t_pnt*QE # ph s-1
        # SNR!AJ50          photo_converted_rate
        photo_converted_rate = (pl_convertRate + zo_bkgRate + sp_bkgRate)/mpix # ph s-1
        # SNR!AJ42          frameTime
        # = Scenario!N24
        frameTime = scenario[11] 
        # SNR!AJ52          signalPerPixPerFrame
        SignalPerPixPerFrame = frameTime*photo_converted_rate 
        # Detector!B12:M18  DectectorModelParTable
        DetectorModelParTable = self.DetectorModel_info(DetectorEOLmonths, MissionEpoch, Dark_derate, missionFraction, CTE_derate, dqeFluxSlope, dqeKnee, dqeKneeFlux, SignalPerPixPerFrame)
        # SNR!AJ32          Sensor_Model
        # = Scenario!L27
        Sensor_Model = scenario[Offset + 20]
        # Detector!B12:B18  SensorModels
        SensorModels = [i[0] for i in DetectorModelParTable]
        row = SensorModels.index(Sensor_Model)
        # SNR!AR38          Photon_counting_Eff
        Photon_counting_Eff = 1 - DetectorModelParTable[row][5]
        # SNR!AR39          Hot_pixels
        Hot_pixels = 1 - DetectorModelParTable[row][11]
        # SNR!AR33          Cosmic_Ray_Tail_Length
        Cosmic_Ray_Tail_Length = DetectorModelParTable[row][6]
        # SNR!AR34          Cosmic_Ray_Hits
        Cosmic_Ray_Hits = 5*1.7*frameTime
        # SNR!AR40          Cosmic_Rays
        Cosmic_Rays = 1 - Cosmic_Ray_Hits*Cosmic_Ray_Tail_Length/(1024**2)
        # SNR!AR41          Net_charge_transfer_eff
        Net_charge_transfer_eff = DetectorModelParTable[row][10]
        # SNR!M21           QE_adjust
        QE_adjust = 1
        # SNR!AR42:AS42     dQE
        dQE = QE*Photon_counting_Eff*Hot_pixels*Cosmic_Rays*Net_charge_transfer_eff*QE_adjust
        # SNR!AJ46          pl_signalRate
        pl_signalRate = f_SR*planet_Flux*Acol*t_pnt*dQE
        C_p = pl_signalRate
    
        # SNR!AJ103         ENF
        ENF = 1.0
        # SNR!L39           Ref_Star_dmag
        # = SNR!AJ13
        # = Scenario!M36
        Ref_Star_dmag = 3.0
        # SNR!L40           Bright_Ratio
        Bright_Ratio = 10**(0.4*Ref_Star_dmag)
        # SNR!L41           tfRDI
        tfRDI = 0.2
        # SNR!L42           beta_RDI
        betaRDI = 1/(Bright_Ratio*tfRDI)
        # SNR!L43           k_sp
        k_sp = 1 + betaRDI
        # SNR!L44           k_det
        k_det = 1 + (betaRDI**2)*tfRDI
        # SNR!L45            k_lzo
        k_lzo = k_det
        # SNR!L46           k_ezo
        k_ezo = k_sp
        # SNR!M19           darkCurrent_Adjust
        darkCurrent_Adjust = 1.0
        # SNR!G23           Dark_Current_Epoch
        # = SNR!AR31
        Dark_Current_Epoch = darkCurrent_Adjust*DetectorModelParTable[row][9]/3600*u.ph/u.s # ph s-1
        # SNR!L7            darkNoiseRate
        darkNoiseRate = ENF**2*Dark_Current_Epoch*mpix # ph s-1
        # SNR!M20           CIC_Adjust
        CIC_Adjust = 1.0
        # SNR!G24           CIC_Epoch
        # = SNR!AR32
        CIC_Epoch = CIC_Adjust*DetectorModelParTable[row][7]*u.ph/u.s # ph s-1
        # SNR!L8            CICnoiseRate
        CICnoiseRate = ENF**2*CIC_Epoch*(mpix/frameTime) # ph s-1
        # SNR!AJ160         Expected_Phot_Rate
        Expected_Phot_Rate = 0.00016*u.ph/u.s # ph s-1
        # SNR!AJ163         elec_rate_per_SNR_region
        elec_rate_per_SNR_region = Expected_Phot_Rate*mpix*dQE # ph s-1
        # SNR!L9            luminesRate
        luminesRate = ENF**2*elec_rate_per_SNR_region # ph s-1
        # SNR!G22           Effective_Read_Noise
        # = ANR!AR30
        Effective_Read_Noise = DetectorModelParTable[row][8]
        # SNR!L10           readNoiseRate
        readNoiseRate = (mpix/frameTime)*Effective_Read_Noise**2*u.ph/u.s # ph s-1
        # SNR!L11           totNoiseVarRate
        totNoiseVarRate = ENF**2*(pl_convertRate + (k_sp*sp_bkgRate) + \
                          (k_lzo*lzo_bkgRate) + (k_ezo*ezo_bkgRate) + \
                          k_det*(darkNoiseRate + CICnoiseRate + luminesRate)) \
                          + k_det*readNoiseRate
        C_b = totNoiseVarRate
        
        # SNR!AJ142         k_comp
        k_comp = 1
        # Scenario!E40      selDeltaC
        selDeltaC = ContrastSrcTable[Contrastrow][2]
        # SNR!AJ48          residSpecRate
        residSpecRate = k_comp*f_SR*Star_Flux*selDeltaC*Coronagraph_I_pk*mpixIntrinsic*t_speckle*Acol*dQE
        C_sp = residSpecRate
        
        # SNR!D12           SNRtarget
        # = SNR!AJ26
        # = Scenario!K24
        SNRtarget = scenario[8]
        
        # SNR!H13           tSNRrraw
        tSNRraw = (np.true_divide(SNRtarget**2*C_b, (C_p**2 - SNRtarget**2*C_sp**2))*u.ph).to(u.h) # hr
        print 'C_p', C_p*u.s/u.ph-0.115210598349869
        print 'C_b', C_b*u.s/u.ph-0.257961002348134
        print 'C_sp', C_sp*u.s/u.ph-0.002267156853525
        print 'tSNRraw', tSNRraw, tSNRraw/u.h-0.561588897210313
        return C_p/u.ph, C_b/u.ph, C_sp/u.ph

    def calc_intTime(self, TL, sInds, fZ, fEZ, dMag, WA, mode):
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
        C_p, C_b, C_sp = self.Cp_Cb_Csp_MOD(TL, sInds, fZ, fEZ, dMag, WA, mode)
        
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