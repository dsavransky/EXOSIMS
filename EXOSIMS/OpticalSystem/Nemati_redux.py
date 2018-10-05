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
    
    """ADDED FUNCTIONS"""

    def scenario_info(self, CGtable, ScenarioSelected):
        # Corrected for 0-indexing
        # Scenario!C5:AE21  scenarioTable
        scenariolines = open(access_path + 'scenarioTable.txt').read().splitlines()[1:]
        scenarioTable = [i.split('\t') for i in scenariolines]
        scenarioTable = self.str2float(scenarioTable)
        row = [i[0] for i in CGtable].index(scenarioTable[3][4])
        scenarioTable[3][2] = CGtable[row][2]
        row = [i[0] for i in CGtable].index(scenarioTable[7][4])
        scenarioTable[7][2] = CGtable[row][2]
        scenarios = [i[0] for i in scenarioTable]
        row = scenarios.index(ScenarioSelected)
        scenarioline = scenarioTable[row]
        return scenarioline
    
    def CG_info(self, tau_DIpathRT = 0, tau_IFSpathRT = 0):
        """Returns table containing information about the coronographs.
        Columns:
            CG (string)
            Name (string)
            BW (float)
            Cor for Pol(int)
            Polarize Loss (int)
            Sampling (float)
            t_refl (float)
        Note:   Some elements of the table are used in calculations in
                Cb_Cp_Csp_MOD, the results of which are inputs to other
                elements in the table. Therefore CG_info is called twice within
                Cb_Cp_Csp_MOD. It is first called without arguments to return
                independent values, and is later called with arguments to
                return all values, dependent and independent.
        
        Args:
            tau_DIpathRT ():
            tau_IFSpathRT ():
            
        Returns:
            CGtable (list):
                Table containing information about the coronographs.
                
        """
        CGlines = open(access_path + 'CGtable.txt').read().splitlines()
        CGtable = [i.split('\t') for i in CGlines]
        CGtable = self.str2float(CGtable)
        if tau_DIpathRT == 0 and tau_IFSpathRT == 0:    
            return CGtable
        else:
            for i in range(len(CGtable)):
                if i in range(1, 4):
                    CGtable[i][-1] = tau_IFSpathRT
                else:
                    CGtable[i][-1] = tau_DIpathRT
            return CGtable
        
    def FP_info(self, Focal_Plane_Type, FPattributes, n):
        # Corrected for 0-indexing
        row = n
        col = FPattributes[0].index(Focal_Plane_Type)
        FP_info = FPattributes[row][col]
        return FP_info
    
    def QE_info(self, val, lambda_nm):
        # Corrected for 0-indexing
        # Detector!P8:X90   QEtable
        QElines = open(access_path + 'QEtable.txt').read().splitlines()
        QEtable = [i.split('\t') for i in QElines]
        QEtable = self.str2float(QEtable)
        # Detector!Q8:X8    QEnum
        QEnum = QEtable[0]
        # Detector!P9:X9    QElist
        QElist = QEtable[1]
        # Detector!P10:X90  QEcurves
        QEcurves = QEtable[2:]
        row = 0
        col = QElist.index(val)
        num = QEnum[col]
        col0 = [i[0] for i in QEcurves]
        row = min(range(len(col0)), key = lambda x: abs(col0[x] - lambda_nm))
        col = int(col)
        return QEcurves[row][col]
    
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
        max_cond = max(0, min(1 + missionFraction*(dqeKnee - 1), 1+ missionFraction*(dqeKnee - 1) + missionFraction*dqeFluxSlope*(SignalPerPixPerFrame*u.J*u.m - dqeKneeFlux)))
        table[0][10] = CTE_derate*max_cond
        table[1][10] = max_cond
        table[2][10] = 0.5*(1 + max_cond)
        table[3][10] = 0.93*max_cond
        table[4][10] = 0.5*(1 + max_cond)
        DetectorModelParTable = table
        return DetectorModelParTable
        
    def flux_info(self, wl_bandlo, wl_bandhi, dlambda, Ephoton):
        # SNR!C171:L172
        flux_0lines = open(access_path + 'flux_0.txt').read().splitlines()
        flux_0table = [i.split('\t') for i in flux_0lines]
        flux_0table = self.str2float(flux_0table)
        hdu = fits.open(access_path + 'spectra.fits')
        data = hdu[1].data
        headers = hdu[1].header
        wavelengths = data.field(0)
        for i in range(len(flux_0table[1][1:])):
            intensities = data.field(i+2)
            intensity = 0
            for j in range(len(wavelengths)):
                w = wavelengths[j]
                if w >= wl_bandlo and w <= wl_bandhi:
                    intensity += intensities[j]
            Integrated_Flux = intensity*dlambda/Ephoton
            flux_0table[1][i+1] = Integrated_Flux
        return flux_0table
    
    def Throughput(self, HRC, FSS99_600, Al, BBARx2, Color_filter, Polarizer):
        HLC = open(access_path + 'HLC.txt').read().splitlines()
        HLC = [i.split('\t') for i in HLC]
        SPC = open(access_path + 'SPC.txt').read().splitlines()
        SPC = [i.split('\t') for i in SPC]
        els = open(access_path + 'throughput_elements.txt').read().splitlines()
        els = [i.split('\t') for i in els]
        Throughput = {}
        HLC_dict = {}
        SPC_dict = {}
        data = [HLC, SPC]
        dcts = [HLC_dict, SPC_dict]
        for i in range(2):
            Im_path_dict = {}
            IFS_path_dict = {}
            LOWFS_path_dict = {}
            dat = data[i]
            dct = dcts[i]
            sub_dcts = [Im_path_dict, IFS_path_dict, LOWFS_path_dict]
            for j in range(len(sub_dcts)):
                sub_dct = sub_dcts[j]
                for k in range(len(els)):
                    el = els[k][0]
                    path = []
                    for l in range(2):
                        try:
                            val = float(dat[k][2*j:(2*j+2)][l])
                            if str(val)[:5] == '0.986':
                                val = HRC
                            elif str(val)[:5] == '0.984':
                                val = FSS99_600
                            elif str(val)[:5] == '0.874':
                                val = Al
                            elif str(val)[:4] == '0.99':
                                val = BBARx2
                            path.append(val)
                        except:
                            path.append('')
                    sub_dct[el] = path
            dct['Imaging'] = Im_path_dict
            dct['IFS'] = IFS_path_dict
            dct['LOWFS'] = LOWFS_path_dict
        Throughput['HLC'] = HLC_dict
        Throughput['SPC'] = SPC_dict
        Throughput['HLC']['Imaging']['Color FW'][0] = Color_filter
        Throughput['SPC']['Imaging']['Color FW'][0] = Color_filter
        Throughput['SPC']['IFS']['Color FW'][0] = Color_filter
        Throughput['HLC']['Imaging']['Polarizer'][0] = Polarizer*BBARx2
        return Throughput
        
    def planet_info(self, Distance = 0, Host_V = 0, Radius = 0, SMA = 0, Ph_Ang = 0, Albedo = 0, select_Vmag1 = 0, select_Vmag2 = 0, change_SMA1 = 0, change_SMA2 = 0, tau_ceti_e_phase_angle = 0, n_zodi = 0):
        # Corrected for 0-indexing
        planet_lines = open(access_path + 'planet_table.txt').read().splitlines()
        planet_table = [i.split('\t') for i in planet_lines]
        planet_table = self.str2float(planet_table)
        if Distance == 0:
            return planet_table
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
    
    def fiducial_info(self, AU, R_jupiter, Fiducial_planet):
        # Corrected for 0-indexing
        fiducial_lines = open(access_path + 'fiducial_table.txt').read().splitlines()
        fiducial_table = [i.split('\t') for i in fiducial_lines]
        fiducial_table = self.str2float(fiducial_table)
        for i in range(len(fiducial_table[-1])):
            fiducial_table[-1][i] = fiducial_table[5][i]*AU*np.sqrt(fiducial_table[1][i]*0.000000001/fiducial_table[7][i])/R_jupiter
        fid_col_ind = fiducial_table[0].index(Fiducial_planet)
        fid_col = [i[fid_col_ind] for i in fiducial_table]
        return fid_col
    
    def error_info(self, planetNo, scenarioline, planetWA, DisturbanceCase, SensCaseSel, MUFcaseSel, CS_NmodeCoef, CS_Nstat, CS_Nmech, CS_Nmuf):
        # SensitivityMUF!C3:G23
        MUFlines = open(access_path + 'MUF_table.txt').read().splitlines()
        MUFtable = [i.split('\t') for i in MUFlines]
        MUFtable = self.str2float(MUFtable)
        # SensitivityMUF!C1LG1
        MUFcases = MUFtable[0]
        MUFtable = MUFtable[2:]
        MUFcol = MUFcases.index(MUFcaseSel)
        # Scenario!Q48      MUFindex
        MUFindex = MUFcol
        # C Stability!E48:E68
        MUF = [] # 1x21
        for i in range(21):
            MUF.append(MUFtable[i][MUFcol])   
        # DisturbanceLive!A1:AC1179
        DisturbanceLivelines = open(access_path + 'DisturbanceLive_table.txt').read().splitlines()
        DisturbanceLivetable = [i.split('\t') for i in DisturbanceLivelines]
        DisturbanceLivetable = self.str2float(DisturbanceLivetable, 0)
        # DisturbanceLive!AI4:AI1179
        DisturbanceLive = [] # 1x1176
        rows = 21
        cols = 28
        for i in range(1176):
            Liverow = i%rows + (rows+4)*(i/(rows*cols)) + 3
            Livecol = (i/rows)%cols + 1
            DisturbanceLive.append(DisturbanceLivetable[Liverow][Livecol])
        # Disturbance!H6:U1181
        Disturbancelines = open(access_path + 'Disturbance_table.txt').read().splitlines()
        Disturbancetable = [i.split('\t') for i in Disturbancelines]
        Disturbancetable = self.str2float(Disturbancetable)
        for i in range(len(Disturbancetable)):
            Disturbancetable[i][3] = DisturbanceLive[i]
        # Disturbance!H4:U4
        DisturbanceCaseList = ['rqt10hr', 'rqt10hr_1mas', 'rqt171012', 'live', 'rqt10hr171212', 'rqt40hr171212', 'rqt10hr171221', 'rqt40hr171221', 'cbe10hr171221', 'rqt10hr180109', 'rqt40hr180109', 'cbe10hr180109', 'cbe10hr180130', 'cbe10hr180306']
        DisturbanceCaseIndex = DisturbanceCaseList.index(DisturbanceCase)
        # C Stability!R40:AS60
        Disturbances = [] # 21x28
        for i in range(21):
            Disturbance_row = []
            for j in range(28):
                row = i + CS_NmodeCoef*(j + CS_Nstat*CS_Nmech*MUFindex)
                col = DisturbanceCaseIndex
                Disturbance_row.append(Disturbancetable[row][col])
            Disturbances.append(Disturbance_row)   
        # Sensitivities!D5:J109
        Contrastlines = open(access_path + 'Contrast_table.txt').read().splitlines()
        Contrasttable = [i.split('\t') for i in Contrastlines]
        Contrasttable = self.str2float(Contrasttable)
        # AnnZoneList!C3:P11
        AnnZonelines = open(access_path + 'AnnZone_table.txt').read().splitlines()
        AnnZonetable = [i.split('\t') for i in AnnZonelines]
        AnnZonetable = self.str2float(AnnZonetable)
        # Sensitivities!D4:R4
        SensCases = ['HLC150818_Dwight', 'SPCgen3', 'SPC170714_design', 'HLC150818_BBit61', 'SPC170714_10per', 'HLC150818_61_10per', 'SPC170831_design', 'future 4', 'future 5', 'future 6', 'future 7', 'future 8', 'future 9', 'future 10', 'future 11']
        Senscol = SensCases.index(SensCaseSel)
        # SNR!AR128:AR137   AnnZoneLookupTable
        AnnZoneLookupTable = []
        for i in range(len(AnnZonetable)):
            AnnZoneLookupTable.append(AnnZonetable[i][Senscol])
        # SNR!AJ41          AnnZone
        if planetNo == 1:
            # = Scenario!M24
            # Corrected for 0-indexing
            AnnZone = scenarioline[10]
        else:
            AnnZone_val = max(AnnZoneLookupTable[0], min(AnnZoneLookupTable[4], np.floor(planetWA)))
            AnnZone = AnnZoneLookupTable.index(AnnZone_val)
#            AnnZone = min(range(len(AnnZoneLookupTable)), key = lambda i: abs(AnnZoneLookupTable[i] - AnnZone_val) if AnnZoneLookupTable[i] < AnnZone_val else max(AnnZoneLookupTable))
        # C Stability!H48:H68
        SensCaseSels = [] # 1x21
        for i in range(21):
            Sensrow = i + CS_NmodeCoef*AnnZone
            SensCaseSels.append(Contrasttable[int(Sensrow)][int(Senscol)])
        # C Stability!F48:F68
        Sensitivities = [] # 1x21
        for i in range(21):
            Sensitivities.append(MUF[i]*SensCaseSels[i])
        # C Stability!AU40:AU60
        unit_mult = [float(i) for i in open(access_path + 'unit_mult.txt').readlines()] # 1x21
        # C Stability!R6:AS29
        Error_Mechanism  = [] # 21x28
        for i in range(21):
            Error_row = []
            for j in range(28):
                Error_row.append(Sensitivities[i]*(Disturbances[i][j]*unit_mult[i])**2)
            Error_Mechanism.append(Error_row)
        # C Stability!R31:AS31
        totals = [] # 1x28
        for i in range(len(Error_Mechanism[0])):
            Error_col = [j[i] for j in Error_Mechanism]
            totals.append(sum(Error_col))
        # C Stability!D26:G32
        error_table = [] # 7x4
        c = 0
        for i in range(7):
            row = []
            for i in range(4):
                row.append(totals[c]*0.000000001)
                c += 1
            error_table.append(row)
        return error_table, AnnZone, MUFindex     
    
    def str2float(self, table, zero = 1):
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
        # Scenario!D36      ScenarioSelected
#        ScenarioSelected = 'SPEC IFS - Band 3'        
        ScenarioSelected = 'IMG NFOV - Band 1'       
        # SNR!H4            planetNo
        planetNo = 1 
        # SNR!C39:I46       CGtable
        CGtable = self.CG_info(0, 0)
        # SNR!AJ37          Focal_Plane_Type
        scenarioline = self.scenario_info(CGtable, ScenarioSelected)
        # SNR!BB45          nm
        nm = 1e-9
        # SNR!AJ22          lambda_
        # Scenario!D24
        lambda_nm = scenarioline[1]
        lambda_ = nm*lambda_nm
        # Throughput!04:P5  Reflectivity_de_rate
        Reflectivity_de_rate = [['REQ', 0.992],
                                ['CBE', 1.0]]
        # Scenario!V42:W46  Offset_table
        Offset_table = [['REQ',         0],
                        ['CBE',         8],
                        ['Goal',        8],
                        ['Unity MUF',   8],
                        ['Best Case',   8]]
        Offsets = [i[0] for i in Offset_table]
        # Scenario!D27      Case_Type
        # = Error Budget!BI3
        Case_Type = 'CBE'
        # Scenario!W39      Offset
        Offset = Offset_table[Offsets.index(Case_Type)][1]
        # SNR!AJ40          de_rate_schedule
        # =  Scenario!K27
        de_rate_schedule = scenarioline[Offset + 19]
        # Throughput!D5     derate_factor
        derate_factor = de_rate_schedule
        # Throughput!E5     derate_factor_um
        Reflectivitylines = [i[0] for i in Reflectivity_de_rate]
        row = Reflectivitylines.index(derate_factor)
        col = 1
        derate_factor_um = Reflectivity_de_rate[row][col]
        # Throughput!D4     Wavelength
        Wavelength = lambda_/(0.000001)
        # Throughput!S5:AG405
        hdu1 = fits.open(access_path + 'HRC.fits')
        HRC_data = hdu1[1].data
        # Throughput!D8     HRC
        HRC_wave = HRC_data.field('Wavelength')
        row = min(range(len(HRC_wave)), key = lambda x: abs(HRC_wave[x] - Wavelength))
        col = 1
        HRC = HRC_data.field(col)[row]*derate_factor_um
        # Throughput!AI5:AW405
        hdu2 = fits.open(access_path + 'FS99.fits')
        FS99_data = hdu2[1].data
        # Throughput!E8     FSS99_600
        FS99_wave = FS99_data.field('Wavelength')
        row = min(range(len(FS99_wave)), key = lambda x: abs(FS99_wave[x] - Wavelength))
        col = 1
        FSS99_600 = FS99_data.field(col)[row]*derate_factor_um
        # Throughput!BP5:CD405
        hdu3 = fits.open(access_path + 'AL.fits')
        AL_data = hdu3[1].data       
        # Throughput!F8     Al
        AL_wave = AL_data.field('Wavelength')
        row = min(range(len(AL_wave)), key = lambda x: abs(AL_wave[x] - Wavelength))
        col = 1
        Al = AL_data.field(col)[row]*derate_factor_um
        # Throughput!AY5:BM405
        hdu4 = fits.open(access_path + 'AR.fits')
        AR_data = hdu4[1].data
        # Throughput!G8     AR
        AR_wave = AR_data.field('Wavelength')
        row = min(range(len(AR_wave)), key = lambda x: abs(AR_wave[x] - Wavelength))
        col = 1
        AR = AR_data.field(col)[row]
        # Throughput!H9     BBARx2
        BBARx2 = 0.99
        # Throughput!I9     Color_filter
        Color_filter = 0.9
        # Throughput!J9     Polarizer
        Polarizer = 0.485
        # Throughput!C11:P64
        Throughput = self.Throughput(HRC, FSS99_600, Al, BBARx2, Color_filter, Polarizer)
        # Throughput!D66    tau_DIpathRT
        product = 1
        for key, value in Throughput['HLC']['Imaging'].iteritems():
            if key != 'Color FW' and key != 'Polarizer' and value[0] != '':
                product = product*value[0]
        tau_DIpathRT = product
        # Throughput!M66    tau_IFSpathRT
        product = 1
        for key, value in Throughput['SPC']['IFS'].iteritems():
            if key != 'Color FW' and value[0] != '':
                product = product*value[0]
        tau_IFSpathRT = product   
        # SNR!C39:I46       CGtable
        CGtable = self.CG_info(tau_DIpathRT, tau_IFSpathRT)
        CGcases = [i[0] for i in CGtable]
        # SNR!AJ37
        Focal_Plane_Type = scenarioline[3]
        # SNR!AJ21          Resolution
        # Scenario!I24      
        Resolution = scenarioline[6]
        #SNR!AJ23           bandwidth
        # Scenario!E24      
        bandwidth = scenarioline[2]
        # SNR!AR46          f_sr_IFS
        try:
            f_sr_IFS = 1.0/(Resolution*bandwidth)
        except:
            f_sr_IFS = -1.0
        # SNR!AJ94          IFS_Lensl_per_PSF
        IFS_Lensl_per_PSF = 5.0
        # SNR!AR48          Critical_lambda
        Critical_lambda = 7.05e-7
        # SNR!AJ96          IFS_Spectral_Samp
        IFS_Spectral_Samp = 2.0
        # SNR!AJ95          IFS_Spatial_Samp
        IFS_Spatial_Samp = 2.0
        # SNR!AR47          mpix_IFS
        mpix_IFS = IFS_Lensl_per_PSF*(lambda_/Critical_lambda)**2*IFS_Spectral_Samp*IFS_Spatial_Samp
        planet_table_pre = self.planet_info()
        planet_row = planetNo - 1
        # Scenario!L43      Fiducial_planet
        # = SNR!AJ29        FidPlanetCase
        # = Scenario!L24    Fid_Planet
        Fiducial_planet = scenarioline[9]
        # SNR!BB37          AU
        AU = const.au
        # SNR!BB36          R_jupiter
        R_jupiter = const.R_jup
        fid_col = self.fiducial_info(AU, R_jupiter, Fiducial_planet)
        # Scenario!M44      Distance
        Distance = fid_col[4]
        # Scenario!M45      Host_V
        Host_V = fid_col[3]
        # Scenario!M46      Radius
        Radius = fid_col[8]
        # Scenario!M47      SMA
        SMA = fid_col[5]
        # Scenario!M48      Ph_Ang
        Ph_Ang = fid_col[6]
        # Scenario!M49      Albedo
        Albedo = fid_col[7]
        # SNR!T67           select_Vmag1
        select_Vmag1 = 5.0
        # SNR!T135          select_Vmag2
        select_Vmag2 = 5.0
        # SNR!T68           select_WA1
        select_WA1 = 3.2
        # SNR!T136          select_WA2
        select_WA2 = 7.0
        # SNR!AJ130         D_PM
        D_PM = 2.37
        # SNR!AJ10          Star_distance
        Star_distance = planet_table_pre[planet_row][3]
        # SNR!BB38          pc
        pc = const.pc
        # SNR!AJ14          orbitPhaseAngle
        orbitPhaseAngle = planet_table_pre[planet_row][9]
        # deg ???? References in SNR!AJ66, cannot be found. Manual calculation shows deg = 1.
        # SNR!T69           change_SMA1
        change_SMA1 = select_WA1*(lambda_/D_PM)*Star_distance*(pc/AU)/np.sin(np.deg2rad(orbitPhaseAngle))
        # SNR!T137          change_SMA2
        change_SMA2 = select_WA2*(lambda_/D_PM)*Star_distance*(pc/AU)/np.sin(np.deg2rad(orbitPhaseAngle))
        # Scenario!AQ7      tau_ceti_e_phase_angle
        tau_ceti_e_phase_angle = 90.0
        # SNR!AR65          n_zodi
        n_zodi = 20.0
        # SNR!AJ15          Planet_SMA
        planet_table = self.planet_info(Distance, Host_V, Radius, SMA, Ph_Ang, Albedo, select_Vmag1, select_Vmag2, change_SMA1, change_SMA2, tau_ceti_e_phase_angle, n_zodi)
        planet_row = planetNo - 1
        Planet_SMA = planet_table[planet_row][6]
        # SNR!BB41          arcsec
        arcsec = np.pi/180/3600
        # SNR!BB42
        mas = arcsec/1000
        # SNR!AJ66          Pl_separation
        Pl_separation = ((Planet_SMA*np.sin(np.deg2rad(orbitPhaseAngle))*AU)/(Star_distance*pc))/mas
        # SNR!AJ71          lam_over_D
        lam_over_D = lambda_/D_PM/mas
        # SNR!AJ67          Planet_Positional_WA
        Planet_Positional_WA = 0.0000000001 + Pl_separation/lam_over_D
        # SNR!AJ28          Coronograph_type
        Coronograph_type = scenarioline[4]
        # From SNR!AJ28
        CGcase = Coronograph_type
        # SNR!AR7           CG_Type
        CGrow = CGcases.index(CGcase)
        CG_Type = CGtable[CGrow][1]
        # CG Tables!N6:U40
        SPC_Tech_Demo_lines = open(access_path + CG_Type + '.txt').read().splitlines()
        SPC_Tech_Demo = [i.split('\t') for i in SPC_Tech_Demo_lines]
        SPC_Tech_Demo = self.str2float(SPC_Tech_Demo)
        # SNR!AR17          Max_Working_Angle
        Max_Working_Angle = np.max([i[0] for i in SPC_Tech_Demo]) # Max value in fist column of table
        # SNR!AJ70
        Dark_Hole_OWA = Max_Working_Angle
        # SNR!AR16          Min_Working_Angle
        Min_Working_Angle = np.min([i[0] for i in SPC_Tech_Demo]) # Min value in fist column of table
        # SNR!AJ69
        Dark_Hole_IWA = Min_Working_Angle
        # SNR!AJ68          planetWA
        if Planet_Positional_WA < Dark_Hole_OWA:
            planetWA = Planet_Positional_WA 
        else:
            planetWA = Dark_Hole_IWA + 0.8*(Dark_Hole_OWA - Dark_Hole_IWA)
        # SNR!AR14          PSF_Area_on_Sky
        rlamD = [i[0] for i in SPC_Tech_Demo]
        SPCrow = min(range(len(rlamD)), key = lambda i: abs(rlamD[i] - planetWA) if rlamD[i] < planetWA else max(rlamD))
        PSF_Area_on_Sky = SPC_Tech_Demo[SPCrow][6]
        # SNR!AJ85          omegaPSF
        omegaPSF = PSF_Area_on_Sky
        # SNR!AR9           Design_Wavelength
        Design_Wavelength = D_PM*SPC_Tech_Demo[SPCrow][1]*arcsec/SPC_Tech_Demo[SPCrow][0]
        # SNR!AS48          Critical_lambda_Imaging
        Critical_lambda_Imaging = 5.08e-7
        # SNR!AS47          mpix_Imaging
        mpix_Imaging = omegaPSF*arcsec**2*(lambda_/Design_Wavelength)**2*(2*D_PM/Critical_lambda_Imaging)**2
        # SNR!AS49          Focal_Plane_plate_scale
        Focal_Plane_plate_scale = Critical_lambda_Imaging/D_PM/2/mas
        # SNR!AR45:AS49     FPattributes
        FPattributes = [['IFS',         'Imaging'], 
                        [f_sr_IFS,      1.0],
                        [mpix_IFS,      mpix_Imaging],
                        [7.05e-7,       5.08e-7],
                        ['--',          Focal_Plane_plate_scale]]
        # SNR!AJ91          f_SR
        f_SR = self.FP_info(Focal_Plane_Type, FPattributes, 1)
        # SNR!AJ16          Planet_Albedo
        Planet_Albedo = planet_table[planet_row][10]
        # SNR!AJ17          Planet_Radius
        Planet_Radius = planet_table[planet_row][11]
        # SNR!AJ65          planetFluxRatio
        planetFluxRatio = Planet_Albedo*(Planet_Radius*R_jupiter/(Planet_SMA*AU))**2
        # SNR!AJ12          Spectral_Type
        Spectral_Type = planet_table[planet_row][8]
        # SNR!AJ99          wl_bandlo
        wl_bandlo = lambda_*(1 - bandwidth/2)
        # SNR!AJ100         wl_bandhi
        wl_bandhi = lambda_*(1 + bandwidth/2)
        # Spectra!B4        lambda_1
        lambda_1 = 0.00000025
        # Spectra!B5        lambda_2
        lambda_2 = 0.000000251
        # SNR!AJ101         dlambda
        dlambda = lambda_2 - lambda_1
        # SNR!BB27          h_planck
        h_planck = const.h
        # SNR!BB28          c_light
        c_light = const.c
        # SNR!AJ102         Ephoton
        Ephoton = h_planck*c_light/lambda_
        # SNR!AJ72          zero_Mag_Flux
        flux_0table = self.flux_info(wl_bandlo, wl_bandhi, dlambda, Ephoton)
        col = flux_0table[0].index(Spectral_Type)
        zero_Mag_Flux = flux_0table[1][col]
        # SNR!AJ11          Star_apparent_mag
        Star_apparent_mag = planet_table[planet_row][2]
        # SNR!AJ73          Star_Flux
        Star_Flux = zero_Mag_Flux*10**(-0.4*Star_apparent_mag)
        # SNR!AJ74          planet_Flux
        planet_Flux = planetFluxRatio*Star_Flux
        # SNR!AJ131         Sec_Obs_frac
        Sec_Obs_frac = 0.32
        # SNR!AJ132         Struts_obscuration
        Struts_obscuration = 0.07
        # SNR!AJ133         Clear_aperture_fraction
        Clear_aperture_fraction = (1 - Struts_obscuration)*(1 - Sec_Obs_frac**2)
        # SNR!AJ135         Acol
        Acol = ((np.pi/4)*D_PM**2)*Clear_aperture_fraction
        # SNR!AR19          Correction_Incl_pol
        Correction_Incl_pol = 1 + CGtable[CGrow][3]
        # SNR!AR12          Core_Throughput
        Core_Throughput = SPC_Tech_Demo[SPCrow][4]*Correction_Incl_pol
        # SNR!AJ39          MUF_Thp
        # = Scenario!I27
        MUF_Thp = scenarioline[Offset + 17]
        # Scenario!O24      REQ_tau_core
        REQ_tau_core = scenarioline[12]
        # SNR!AJ108         t_core
        if de_rate_schedule == 'CBE':
            t_core = Core_Throughput*MUF_Thp
        else:
            t_core = REQ_tau_core
        # SNR!AJ112         t_occ
        # = SNR!AR15
        t_occ = SPC_Tech_Demo[SPCrow][7]*Correction_Incl_pol
        # SNR!AJ109         t_psf
        try:
            t_psf = t_core/t_occ
        except:
            t_psf = 0
        # SNR!AJ114         t_refl
        # = SNR!AR18
        t_refl = CGtable[CGrow][6]
        # SNR!AJ115         t_filt
        t_filt = 0.9
        # SNR!AJ116         t_pol
        # = SNR!AR20
        t_pol = CGtable[CGrow][4]
        # SNR!AJ119         t_unif
        t_unif = t_occ*t_refl*t_filt*t_pol
        # SNR!AJ118         t_pnt
        t_pnt = t_unif*t_psf
        # SNR!AJ33          QE_curve
        QE_curve = 'e2v_Spec'
        # SNR!AR37          QE
        QE = self.QE_info(QE_curve, lambda_nm)
        # Detector!C5       DetectorEOLmonths
        DetectorEOLmonths = 5.25*12
        # Scenario!H24      Months_at_L2
        Months_at_L2 = scenarioline[5]
        # SNR!AJ34          Radiation_Exposure
        # = Scenario!H24    
        Radiation_Exposure = Months_at_L2
        # Detector!C6       MissionEpoch
        # = SNR!AJ34
        MissionEpoch = Radiation_Exposure
        # Detector!L6       Dark_derate
        Dark_derate = 1.1
        # SNR!AJ35          missionFraction
        missionFraction = Radiation_Exposure/DetectorEOLmonths
        # Detector!L5       CTE_derate
        CTE_derate = 0.83
        # Detector!AC37     dqeFluxSlope
        dqeFluxSlope = 3.24
        # Detector!AC38     dqeKnee
        dqeKnee = 0.858
        # Detector!AC39     dqeKneeFlux
        dqeKneeFlux = 0.089
        # SNR!AJ93          mpix
        mpix = self.FP_info(Focal_Plane_Type, FPattributes, 2)
        # SNR!AJ62          Coronograph_I_pk
        # = SNR!AR13
        Coronograph_I_pk = SPC_Tech_Demo[SPCrow][5]*Correction_Incl_pol
        # SNR!AR21          CG_intrinsic_sampling
        CG_intrinsic_sampling = CGtable[CGrow][5]
        # SNR!AR22          mpixIntrinsic
        mpixIntrinsic = omegaPSF*arcsec**2/(CG_intrinsic_sampling*Design_Wavelength/D_PM)**2
        # SNR!AR11          Contrast_per_design
        Contrast_per_design = SPC_Tech_Demo[SPCrow][3]
        # SNR!AJ36          k_pp
        # = Scenario!H27
        k_pp = scenarioline[16]
        # SNR!AJ38
        # Scenario!G27
        MUFcaseSel = scenarioline[Offset + 15]
        # C Stability!E38   CS_NmodeCoef
        CS_NmodeCoef = 21
        # C Stability!E39   CS_Nstat
        CS_Nstat = 4
        # C Stability!E40   CS_Nmech
        CS_Nmech = 7
        # C Stability!E40   CS_Nmuf
        CS_Nmuf = 2
        # SNR!AJ30          DisturbanceCase
        # = Scenario!E27
        DisturbanceCase = scenarioline[Offset + 13]
        # SNR!AJ31          SensCaseSel
        # = Scenario!F27
        SensCaseSel = scenarioline[Offset + 14]
        error_info = self.error_info(planetNo, scenarioline, planetWA, DisturbanceCase, SensCaseSel, MUFcaseSel, CS_NmodeCoef, CS_Nstat, CS_Nmech, CS_Nmuf)
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
        # SNR!AJ61          Coronograph_Contrast
        Coronograph_Contrast = SelContrast
        # SNR!AJ120         t_speckle
        t_speckle = t_refl*t_filt*t_pol
        # SNR!AJ142         k_comp
        k_comp = 1.0
        # SNR!AJ47          sp_bkgRate
        sp_bkgRate = k_comp*f_SR*Star_Flux*Coronograph_Contrast*Coronograph_I_pk*mpixIntrinsic*t_speckle*Acol*QE
        # SNR!AJ20          Nzodi
        Nzodi = planet_table[planet_row][5]
        # SNR!AJ72          inBWflux0
        inBWflux0 = zero_Mag_Flux
        # SNR!AJ75          Star_abs_mag
        Star_abs_mag = Star_apparent_mag - 5*np.log10(Star_distance/10)
        # SNR!BB34          Msun
        Msun = 4.83
        # SNR!AJ19          ExoZodi_1AU
        ExoZodi_1AU = 22.0
        # SNR!AJ80          ExoZodi_flux
        ExoZodi_flux = Nzodi*inBWflux0*10**(-0.4*(Star_abs_mag - Msun + ExoZodi_1AU))/(Planet_SMA**2)
        # SNR!AJ83          ezo_bkgRate
        ezo_bkgRate = f_SR*ExoZodi_flux*omegaPSF*Acol*t_unif*QE
        # SNR!AJ18          Local_Zodi
        Local_Zodi = 23.0
        # SNR!AJ81          Local_Zodi_flux
        Local_Zodi_flux = inBWflux0*10**(-0.4*Local_Zodi)
        # SNR!AJ84          lzo_bkgRate
        lzo_bkgRate = f_SR*Local_Zodi_flux*omegaPSF*Acol*t_unif*QE
        # SNR!AJ49          zo_bkgRate
        zo_bkgRate = ezo_bkgRate + lzo_bkgRate
        # SNR!AJ45          pl_convertRate
        pl_convertRate = f_SR*planet_Flux*Acol*t_pnt*QE
        # SNR!AJ50          photo_converted_rate
        photo_converted_rate = (pl_convertRate + zo_bkgRate + sp_bkgRate)/mpix
        # SNR!AJ42          frameTime
        # = Scenario!N24
        frameTime = scenarioline[11] 
        # SNR!AJ52          signalPerPixPerFrame
        SignalPerPixPerFrame = frameTime*photo_converted_rate
        # Detector!B12:M18  DectectorModelParTable
        DetectorModelParTable = self.DetectorModel_info(DetectorEOLmonths, MissionEpoch, Dark_derate, missionFraction, CTE_derate, dqeFluxSlope, dqeKnee, dqeKneeFlux, SignalPerPixPerFrame)
        # SNR!AJ32          Sensor_Model
        # = Scenario!L27
        Sensor_Model = scenarioline[Offset + 20]
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
        Dark_Current_Epoch = darkCurrent_Adjust*DetectorModelParTable[row][9]/3600
        # SNR!L7            darkNoiseRate
        darkNoiseRate = ENF**2*Dark_Current_Epoch*mpix
        # SNR!M20           CIC_Adjust
        CIC_Adjust = 1.0
        # SNR!G24           CIC_Epoch
        # = SNR!AR32
        CIC_Epoch = CIC_Adjust*DetectorModelParTable[row][7]
        # SNR!L8            CICnoiseRate
        CICnoiseRate = ENF**2*CIC_Epoch*(mpix/frameTime)
        # SNR!AJ160         Expected_Phot_Rate
        Expected_Phot_Rate = 0.00016
        # SNR!AJ163         elec_rate_per_SNR_region
        elec_rate_per_SNR_region = Expected_Phot_Rate*mpix*dQE
        # SNR!L9            luminesRate
        luminesRate = ENF**2*elec_rate_per_SNR_region
        # SNR!G22           Effective_Read_Noise
        # = ANR!AR30
        Effective_Read_Noise = DetectorModelParTable[row][8]
        # SNR!L10           readNoiseRate
        readNoiseRate = (mpix/frameTime)*Effective_Read_Noise**2
        # SNR!L11           totNoiseVarRate
        totNoiseVarRate = ENF**2*(pl_convertRate*u.J*u.m + (k_sp*sp_bkgRate*u.J*u.m) + \
                          (k_lzo*lzo_bkgRate*u.J*u.m) + (k_ezo*ezo_bkgRate*u.J*u.m) + \
                          k_det*(darkNoiseRate + CICnoiseRate + luminesRate)) \
                          + k_det*readNoiseRate
        C_b = totNoiseVarRate
        
        # SNR!AJ142         k_comp
        k_comp = 1
        # Scenario!E40      selDeltaC
        selDeltaC = ContrastSrcTable[Contrastrow][2]
        # SNR!AJ48          residSpecRate
        residSpecRate = k_comp*f_SR*Star_Flux*selDeltaC*Coronograph_I_pk*mpixIntrinsic*t_speckle*Acol*dQE
        C_sp = residSpecRate
        
        # SNR!D12           SNRtarget
        # = SNR!AJ26
        # = Scenario!K24
        SNRtarget = scenarioline[8]
        
        # SNR!H13           tSNRrraw
        tSNRraw = np.true_divide(SNRtarget**2*C_b, (C_p**2 - SNRtarget**2*C_sp**2))/3600.0
        print 'C_p', C_p*u.J*u.m-0.115210598349869
        print 'C_b', C_b-0.257961002348134
        print 'C_sp', C_sp*u.J*u.m-0.002267156853525
        print 'tSNRraw', tSNRraw, tSNRraw/(1.*u.J*u.J*u.m*u.m)-0.561588897210313
        print 'hi'
        return C_p, C_b, C_sp

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