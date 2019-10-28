from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem
from EXOSIMS.OpticalSystem.Nemati import Nemati
import astropy.units as u
from astropy.io import fits
import astropy.constants as const
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

class Nemati_2019(Nemati):
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

        #If amici-spec, load Disturb x Sens Tables
        amici_mode = [self.observingModes[i] for i in np.arange(len(self.observingModes)) if self.observingModes[i]['instName'] == 'amici-spec']
        if len(amici_mode) > 0:
            #find index of amici_mode
            amici_mode_index = [i for i in np.arange(len(self.observingModes)) if self.observingModes[i]['instName'] == 'amici-spec'][0] #take first amici-spec instName found
            
            #Specifically for the Disturb X Sens observing mode (STILL NOT SURE HOW TO DECIDED IT IS DISTURB X SENS MODE)
            
            #### LOAD IN Disturbance Table from Disturbance Tab in Bijan2019 spreadsheet
            #I copied H6-U1181 to a text file and converted the range into CSV
            fname = self.observingModes[amici_mode_index]['DisturbXSens_DisturbanceTable']
            import csv
            #C:/Users/Dean/Documents/BijanNemati2019/DisturbXSens_DisturbanceTable.csv

            def extractedCSVTable(fname):
                """
                Args:
                    fname (string) - full filepath to the the csv file
                Returns:
                    tList (numpy array) - 2D array of table values [row,col]
                """
                tList = list()
                with open(fname, newline='') as f:
                    csvreader = csv.reader(f,delimiter=',')
                    for row in csvreader:
                        trow = list()
                        for i in np.arange(len(row)):
                            if row[i] == '':
                                continue
                            else:
                                trow.append(float(row[i]))
                        tList.append(trow)
                return np.asarray(tList)

            self.observingModes[amici_mode_index]['DisturbXSens_DisturbanceTable'] = extractedCSVTable(fname) #Disturbance table on the Disturbance Sheet in Bijan2019 model
            self.observingModes[amici_mode_index]['DisturbanceCases'] = ['rqt10hr', 'rqt10hr_1mas', 'rqt171012', 'live', 'rqt10hr171212', 'rqt40hr171212', 'rqt10hr171221', 'rqt40hr171221',\
                    'cbe10hr171221', 'rqt10hr180109', 'rqt40hr180109', 'cbe10hr180109', 'cbe10hr180130', 'cbe10hr180306']
            #I have no idea what the above clumn labels mean but self.observingModes[amici_mode_index]['DisturbXSens_DisturbanceTableColumnLabels'][0] refers to
            #   self.observingModes[amici_mode_index]['DisturbXSens_DisturbanceTable'][:,0] #KEEP
            ####
            fname2 = self.observingModes[amici_mode_index]['DisturbXSens_DisturbanceLiveSTD_MUF_Table']#.csv #From DisturbanceLive!B4-AC24
            self.observingModes[amici_mode_index]['DisturbXSens_DisturbanceLiveSTD_MUF_Table'] = extractedCSVTable(fname2)
            fname3 = self.observingModes[amici_mode_index]['DisturbXSens_DisturbanceLiveUNITY_MUF_Table']#.csv #From DisturbanceLive!B29-AC49
            self.observingModes[amici_mode_index]['DisturbXSens_DisturbanceLiveUNITY_MUF_Table'] = extractedCSVTable(fname3)
            #DisturbanceLive!$AL is the column-wise interpretation of DisturbXSens_DisturbanceLiveSTD_MUF_Table
            self.observingModes[amici_mode_index]['DisturbXSens_DisturbanceTable'][:,3] = np.concatenate((self.observingModes[amici_mode_index]['DisturbXSens_DisturbanceLiveSTD_MUF_Table'].flatten(),\
                        self.observingModes[amici_mode_index]['DisturbXSens_DisturbanceLiveUNITY_MUF_Table'].flatten()))
            #The Disturbance Table starting at CStability!U40 references the DisturbXSens_DisturbanceTable

            #### Load Sensitivity MUF Table
            fname4 = self.observingModes[amici_mode_index]['DisturbXSens_SensitivityMUF'] #.csv #From SensitivityMUF!C3-G23
            self.observingModes[amici_mode_index]['DisturbXSens_SensitivityMUF'] = extractedCSVTable(fname4)
            #Index Labels of Sensitivity MUF Table Columns
            #KEEPself.observingModes[amici_mode_index]['SensitivityCases'] = ['Standard', 'Unity', 'MUF_o1', 'MUF_o2', 'MUF_o3'] #MUFcases, SensitivityMUF!C1-G1
            #Case used for this mode
            #self.observingModes[amici_mode_index]['DisturbanceCase']

            #### Load Contrast Sensitivity Vectors Table from Sensitivities Tab
            fname5 = self.observingModes[amici_mode_index]['DisturbXSens_ContrastSensitivityVectorsTable'] #.csv #From Sensitivities!D5-P109
            self.observingModes[amici_mode_index]['DisturbXSens_ContrastSensitivityVectorsTable'] = extractedCSVTable(fname5) #DisturbXSens_ContrastSensitivityVectorsTable.csv
            #Column labels of the ContrastVectorsTable
            self.observingModes[amici_mode_index]['DisturbXSens_ContrastSensitivityVectorsColLabels'] 

            #### Load Annular Zone Master Table
            fname6 = self.observingModes[amici_mode_index]['DisturbXSens_AnnZoneMasterTable'] #.csv #From AnnZoneList!C2-O11
            self.observingModes[amici_mode_index]['DisturbXSens_AnnZoneMasterTable'] = extractedCSVTable(fname6) #DisturbXSens_AnnZoneMasterTable.csv

            #### Load Initial Raw Contrast Table
            fname7 = self.observingModes[amici_mode_index]['DisturbXSens_InitialRawContrastTable'] #.csv #From InitialRawContrast!E2-Q21
            self.observingModes[amici_mode_index]['DisturbXSens_InitialRawContrastTable'] = extractedCSVTable(fname7) #DisturbXSens_InitialRawContrast.csv
            #self.observingModes[amici_mode_index]['DisturbXSens_InitialRawContrastCols']

            #### Load NItoContrast Table
            fname8 = self.observingModes[amici_mode_index]['DisturbXSens_NItoContrastTable'] #.csv #From NItoContrast!B2-N6 #DisturbXSens_NItoContrastTable.csv
            self.observingModes[amici_mode_index]['DisturbXSens_NItoContrastTable'] = extractedCSVTable(fname8) #DisturbXSens_NItoContrastTable.csv

            #self.observingModes[amici_mode_index]['DisturbXSens_DisturbanceTable']

        #print(saltyburrito)
        
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
            TK (TimeKeeping object):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.
            returnExtra (boolean):
                Optional flag, default False, set True to return additional rates for validation
        
        Returns:
            C_p (astropy Quantity array):
                Planet signal electron count rate in units of 1/s
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s
            C_sp (astropy Quantity array):
                1/s
        
        """
        
        if TK == None:
            t_now = 0.
        else:
            t_now = (TK.currentTimeNorm.to(u.d)).value/30.4375 # current time in units of months
        
        f_ref = self.ref_Time # fraction of time spent on ref star for RDI
        print("f_ref: " + str(f_ref))
        dmag_s = self.ref_dMag # reference star dMag for RDI
        k_pp = TL.PostProcessing.ppFact(WA) # post processing factor
        m_s = TL.Vmag # V magnitude
        print("m_s[sInds]: " + str(m_s[sInds]))
        
        D_PM = self.pupilDiam # primary mirror diameter in units of m
        f_o = self.obscurFac # obscuration due to secondary mirror and spiders
        f_s = self.shapeFac # aperture shape factor
        
        lam = mode['lam'] # wavelenght in units of nm
        inst_name = mode['instName'] # instrument name
        BW = mode['BW'] # bandwidth
        syst = mode['syst'] # starlight suppression system
        inst = mode['inst'] # instrument dictionary
        
        
        F_0 = TL.starF0(sInds,mode)*BW*lam
        print("BW: " + str(BW))
        print("Lam: " + str(lam))
        print("sInds: " + str(sInds))
        #print("mode: " + str(mode))

        
        A_PSF = syst['core_area'](lam, WA) # PSF area
        #ORIGINAL C_CG = syst['core_contrast'](lam, WA) # coronnagraph contrast
        C_CG = 3.3*10.**-9. #This is the input in the excel spreadsheet SNR!T45
        I_pk = syst['core_mean_intensity'](lam, WA) # peak intensity
        tau_core = syst['core_thruput'](lam, WA)*inst['MUF_thruput'] # core thruput
        tau_occ = syst['occ_trans'](lam, WA) # Occular transmission
        print("I_pk: " + str(I_pk))
        print("tau_occ: " + str(tau_occ))
        print("A_PSF: " + str(A_PSF))

        R = inst['Rs'] # resolution
        print("R: " + str(R))
        eta_QE = inst['QE'](lam) # quantum efficiency        
        print("eta_QE: " + str(eta_QE))
        refl_derate = inst['refl_derate']
        print("refl_derate: " + str(refl_derate))        
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
        print('ENF: ' + str(ENF))
        k_d = inst['dark_derate']
        pixel_size = inst['pixelSize']
        n_pix = inst['pixelNumber']**2.
        
        t_EOL = 63. # mission total lifetime in months
        t_MF = t_now/t_EOL #Mission fraction = (Radiation Exposure)/EOL
        print('t_now: ' + str(t_now))
        print('t_MF: ' + str(t_MF))

        #Convert to inputs
        #tau_BBAR = 0.99
        #tau_color_filt = 0.9 #tau_CF in latex
        #tau_imager = 0.9 #tau_Im in latex
        #tau_spect = 0.8 #tau_SPC in latex
        #tau_clr = 1.
        if 'amici' in inst_name.lower():
            tau_refl = tau_HRC**7. * tau_FSS**16. * tau_Al**3. * mode['tau_BBAR']**10. * mode['tau_color_filt'] * mode['tau_clr']**3. * mode['tau_spect']
            f_SR = 1./(BW*R)
            nld = (inst['Fnum']*lam/pixel_size).decompose().value
            ncore_x = 2.*0.942*nld
            ncore_y = 2.*0.45*nld
            Rcore = 0.000854963720695323*(lam.to(u.nm).value)**2. - 1.51313623178303*(lam.to(u.nm).value) + 707.897720948325
            dndl = Rcore*ncore_x/lam
            mse_y = ncore_y
            mse_x = (dndl*lam/R).value
            m_pix = mse_x*mse_y 
        elif 'spec' in inst_name.lower():
            tau_refl = tau_HRC**7. * tau_FSS**16. * tau_Al**3. * mode['tau_BBAR']**10. * mode['tau_color_filt'] * mode['tau_spect']
            f_SR = 1./(BW*R)
            m_pix = Nlensl*(lam/lam_c)**2*lenslSamp**2.
        else:
            tau_refl = tau_HRC**7. * tau_FSS**13. * tau_Al**2. * mode['tau_BBAR'] * mode['tau_color_filt'] * mode['tau_imager']
            f_SR = 1.0
            m_pix = A_PSF*(2.*lam*D_PM/(lam_d*lam_c))**2.*(np.pi/180./3600.)**2.
        print("m_pix: " + str(m_pix))
        print("f_SR: " + str(f_SR))

        # Point source thruput
        print("tau_core: " + str(tau_core))
        print("tau_refl: " + str(tau_refl))
        tau_PS = tau_core*tau_refl
        
        print("f_s: " + str(f_s))
        print("D_PM: " + str(D_PM))
        print("f_o: " + str(f_o))
        A_col = f_s*D_PM**2.*(1. - f_o)
        print("A_col: " + str(A_col))
        
        for i in sInds:
            F_s = F_0*10.**(-0.4*m_s[i])
        F_P_s = 10.**(-0.4*dMag)
        F_p = F_P_s*F_s
        print("F_s: " + str(F_s))
        print("F_p: " + str(F_p))
        print("F_P_s: " + str(F_P_s))

                
        #ORIGINALm_pixCG = A_PSF*(D_PM/(lam_d*k_s))**2.*(np.pi/180./3600.)**2.
        m_pixCG = A_PSF*(np.pi/180./3600.)**2./((lam_d*k_s)/D_PM)**2.
        print("k_s: " + str(k_s))
        print("lam_d: " + str(lam_d))
        print("m_pixCG: " + str(m_pixCG.decompose()))
        
        #ORIGINALF_ezo = F_0*fEZ*u.arcsec**2.
        n_ezo = 1.
        M_ezo = -2.5*np.log10(fEZ.value)
        M_sun = 4.83
        a_p = 3.31 #in AU This is what is applied in the Bijan spreadsheet.....
        ##### TODO 5. should be m_s, but I had to spoof this to get the right mag for the star... hmmmm.... fix
        F_ezo = F_0*n_ezo*(10.**(-0.4*(m_s[sInds]-M_sun+M_ezo)))/a_p**2.
        F_lzo = F_0*fZ*u.arcsec**2.
        print("F_ezo: " + str(F_ezo))
        print("M_ezo: " + str(M_ezo))
        print("F_lzo: " + str(F_lzo))
        print("F_0: " + str(F_0))

        tau_unif = tau_occ*tau_refl*mode['tau_pol']
        print("tau_unif: " + str(tau_unif))
        
        tau_sp = tau_refl*mode['tau_pol'] # tau_pol is the polarizer thruput. tau_sp is teh speckle throughput

        r_pl = f_SR*F_p*A_col*tau_PS*eta_QE
        #ORIGINALr_sp = f_SR*F_s*C_CG*I_pk*m_pixCG*tau_refl*A_col*eta_QE
        r_sp = f_SR*F_s*C_CG*I_pk*m_pixCG*tau_sp*A_col*eta_QE #Dean replaces with tau_sp as in Bijan latex doc and  excel sheet
        r_ezo = f_SR*F_ezo*A_PSF*A_col*tau_unif*eta_QE
        r_lzo = r_ezo*F_lzo/F_ezo
        print("r_pl: " + str(r_pl.decompose()))
        print("r_sp: " + str(r_sp.decompose()))
        print("r_ezo: " + str(r_ezo))
        print("r_lzo: " + str(r_lzo))

        
        r_ph = (r_pl + r_sp + r_ezo + r_lzo)/m_pix
        #DELETEr_ph = (np.zeros(len(r_pl))+1.9*10.**-3.)*u.nm**2./u.m**2./u.s
                
        k_EM = round(-5.*k_RN/np.log(0.9), 2)
        L_CR = 0.0323*k_EM + 133.5
        print("k_RN: " + str(k_RN))
        print("k_EM: " + str(k_EM))
        
        k_e = t_f*r_ph
        print("r_ph: " + str(r_ph.decompose()))
        print("t_f: " + str(t_f)) #frameTime SNR!T42
        eta_PC = 0.9
        eta_HP = 1. - t_MF/20.
        eta_CR = 1. - (8.5/u.s*t_f).decompose().value*L_CR/1024.**2.
        print("L_CR: " + str(L_CR))
        dqeFluxSlope = 3.24 #(e/pix/fr)^-1
        dqeKnee = 0.858
        dqeKneeFlux = 0.089 #e/pix/fr
        #DOUBLE CHECK THE PROPER APPLICATION OF CTE_DERATE I ADDED THE '[0.5*1.'... AND '0.5*CTE'...
        eta_NCT = [0.5*1.+0.5*CTE_derate*max(0., min(1. + t_MF*(dqeKnee - 1.), 1. + t_MF*(dqeKnee - 1.) +\
                 t_MF*dqeFluxSlope*(i.decompose().value - dqeKneeFlux))) for i in k_e]
        print("CTE_derate: " + str(CTE_derate))
        print("eta_PC: " + str(eta_PC))
        print("eta_HP: " + str(eta_HP))
        print("eta_CR: " + str(eta_CR))
        print("eta_NCT: " + str(eta_NCT))
        print("k_e: " + str(k_e))

        deta_QE = eta_QE*eta_PC*eta_HP*eta_CR*eta_NCT
        
        f_b = 10.**(0.4*dmag_s)
        print("f_b: " + str(f_b))
        
        try:
            k_sp = 1. + 1./(f_ref*f_b)
            k_det = 1. + 1./(f_ref*f_b**2.)
        except:
            k_sp = 1.
            k_det = 1.
        print("k_sp: " + str(k_sp))
        print("k_det: " + str(k_det))

        k_CIC = k_d*(k_EM*4.337e-6 + 7.6e-3)
        
        i_d = k_d*(1.5 + t_MF/2)/u.s/3600.
                
        #ORIGINALr_dir = 625.*m_pix*(pixel_size/(0.2*u.m))**2*u.ph/u.s
        GCRFlux = 5./u.cm**2./u.s #evants/cm^2/s, StrayLight!G36, relativistic event rate
        photons_per_relativistic_event = 250.*u.ph/u.mm #ph/event/mm, StrayLight!G37, Cherenkov Ceiling assuming no CaF2 BaF2 from graph in paper  by Viehman & Eubanks 1976
        lumrateperSolidAng = photons_per_relativistic_event/(2.*np.pi) #39.8 #ph/Sr/event/mm StrayLight!G38
        luminescingOpticalArea = 0.785*u.cm**2. #cm^2, StrayLight!G39, The beam diameter at the color filter and imaging lens is 5mm.
            #the imaging lens is an achromatic doublet. The thickness is 4mm BK7 glass and 2 mm SF2 glass. The polarized imaging
            # has additional up to 10mm thick glass (quartz) before the lens.
        OpticalThickness = 4.0*u.mm #mm
        luminescingOpticalDistance = 0.1*u.m #m, StrayLight!G41
        Omega_Signal = m_pix*pixel_size**2./luminescingOpticalDistance**2. #2.88*10.**-7. #Sr, StrayLight!G42,
        print("Omega_Signal: " + str(Omega_Signal))
        r_dir = (GCRFlux*lumrateperSolidAng*luminescingOpticalArea*OpticalThickness*Omega_Signal).decompose()

        r_indir = (1.25*np.pi*m_pix/n_pix*u.ph/u.s).decompose()
        print('r_dir: ' + str(r_dir))
        print('r_indir: ' + str(r_indir))

        eta_ph = r_dir + r_indir
        print('eta_ph: ' + str(eta_ph))
        eta_e = eta_ph*deta_QE
        print('eta_e: ' + str(eta_e))
        
        r_DN = ENF**2.*i_d*m_pix
        r_CIC = ENF**2.*k_CIC*m_pix/t_f
        r_lum = ENF**2.*eta_e
        r_RN = (k_RN/k_EM)**2.*m_pix/t_f
        print("r_DN: " + str(r_DN))
        print("r_CIC: " + str(r_CIC))
        print("r_lum: " + str(r_lum))
        print("r_RN: " + str(r_RN))
        
        print("k_pp: " + str(k_pp))
        print("C_CG: " + str(C_CG))
        if mode['instName'] == 'amici-spec':
            dC_CG = 8.7471*10.**-12. #temporary
            #TODO REPLACE THIS BIT WITH SOMETHING MORE REALSITIC based on CStability!M34
            modeNames = ['Z2','Z3','Z4','Z5','Z6','Z7','Z8','Z9','Z10','Z11','Gain Err Z5','Gain Err Z6','Gain Err Z7','Gain Err Z8',\
                            'Gain Err Z9','Gain Err Z10','Gain Err Z11','Pupil X','Pupil Y','DM Settle','DM Therm']
                        #Corresponds to CStability!K9-29, T6-T29, T40-60
            disturbanceMults = np.asarray([2.87,2.87,1.00e-03,1.00e-03,1.00e-03,1.00e-03,1.00e-03,1.00e-03,1.00e-03,\
                                    1.00e-03,1.00e-03,1.00e-03,1.00e-03,1.00e-03,1.00e-03,1.00e-03,1.00e-03,1.,1.,1.,1.])
            disturbanceMultUnits = ['mas rms','mas rms','pm rms','pm rms','pm rms','pm rms','pm rms','pm rms','pm rms','pm rms','pm rms',\
                                        'pm rms','pm rms','pm rms','pm rms','pm rms','pm rms','um rms','um rms','hrs after 2 wks * %/dec','mK']
                                    #Corresponding to CStability!AW40-60
            disturbanceCategories = ["ThermalHP", "ThermalLP", "Pupil Shear", "RWA TT", "RWA WFE", "DM Settle", "DM Therm"] #Corresponds to CStability G38-G45

            #Disturbance Column for specific scenario
            #EXAMPLE mode['DisturbXSens_DisturbanceTable'][:,mode['DisturbanceCaseInd']]
            #DisturbanceCase is SNR!T73 for each table column in SensitivityMUF
            DisturbanceCaseInd = mode['DisturbanceCases'].index(mode['DisturbanceCase'])
            CS_NmodelCoef = mode['modeCoeffs'] # CStability!E23 CS_NmodelCoef
            CS_Nstat = mode['statistics'] # CStability!E24 CS_Nstat
            CS_Nmech = mode['mechanisms'] # CStability!E25 CS_Nmech
            MUFindex = mode['MUFcases'].index(mode['MUFCase']) # CStability!E26 MUFindex

            #i is the Mode Number in ModeNames, j is the Disturbance Table Category 
            modeNameNumbers = np.arange(len(modeNames)) #CStability!U40 Table rows. TODO double check if there are other modes possible...

            #### Annular Zone List 
            #mode['DisturbXSens_AnnZoneMasterTable']
            AnnZoneTableCol = np.where(np.asarray(mode['AnnZoneMasterColLabels']) == mode['SenseCaseSel'])[0][0]
            planetWAListMin = mode['DisturbXSens_AnnZoneMasterTable'][0:5,AnnZoneTableCol]
            planetWAListMax = mode['DisturbXSens_AnnZoneMasterTable'][5:,AnnZoneTableCol]

            #TODO figure out why the 1e-XX is used in the spreadsheet
            planetPositionalWA = 0.0000000001+WA/(mode['lam']/self.pupilDiam*u.rad).to('mas').decompose() #The correction from  SNR!T51 #in units of lam/D
            DarkHoleIWA = mode['IWA']/(mode['lam']/self.pupilDiam*u.rad).to('mas').decompose() #in units of lam/D
            DarkHoleOWA = mode['OWA']/(mode['lam']/self.pupilDiam*u.rad).to('mas').decompose() #in units of lam/D
            planetObservingWA = np.asarray([planetPositionalWA[i].value if planetPositionalWA[i] < DarkHoleOWA else (DarkHoleIWA+0.8*(DarkHoleOWA-DarkHoleIWA)).value for i in np.arange(len(planetPositionalWA))]) #Based on cell SNR!T52
            planetAnnZones =   np.asarray([np.where((planetWAListMin <= planetObservingWA[i])*(planetWAListMax > planetObservingWA[i]))[0][0] for i in np.arange(len(planetObservingWA))])
            #planetAnnZones is an array of AnnZones the size of WA
            #NEED TO CHECK IF PLANET ANNZONES ARE IN PROPER RANGE

            #M, V, dM, and dV all belong to CStability Disturbance Table, this section converts from the DisturbXSens_DisturbanceTable to the CStability Disturbance Table CStability!U34
            self.M_ij_disturbance = np.zeros((len(modeNames),len(disturbanceCategories)*4))
            self.V_ij_disturbance = np.zeros((len(modeNames),len(disturbanceCategories)*4))
            self.dM_ij_disturbance = np.zeros((len(modeNames),len(disturbanceCategories)*4))
            self.dV_ij_disturbance = np.zeros((len(modeNames),len(disturbanceCategories)*4))
            for i in np.arange(len(modeNames)): #Iterate down rows of mode names
                for j in np.arange(len(disturbanceCategories)):
                    disturbanceSheetRow = modeNameNumbers[i] + CS_NmodelCoef*( 4*j + CS_Nstat*CS_Nmech*MUFindex)
                    self.M_ij_disturbance[i,4*j] = mode['DisturbXSens_DisturbanceTable'][disturbanceSheetRow,DisturbanceCaseInd]
                    disturbanceSheetRow = modeNameNumbers[i] + CS_NmodelCoef*( 4*j+1 + CS_Nstat*CS_Nmech*MUFindex)
                    self.V_ij_disturbance[i,4*j+1] = mode['DisturbXSens_DisturbanceTable'][disturbanceSheetRow,DisturbanceCaseInd]
                    disturbanceSheetRow = modeNameNumbers[i] + CS_NmodelCoef*( 4*j+2 + CS_Nstat*CS_Nmech*MUFindex)
                    self.dM_ij_disturbance[i,4*j+2] = mode['DisturbXSens_DisturbanceTable'][disturbanceSheetRow,DisturbanceCaseInd]
                    disturbanceSheetRow = modeNameNumbers[i] + CS_NmodelCoef*( 4*j+3 + CS_Nstat*CS_Nmech*MUFindex)
                    self.dV_ij_disturbance[i,4*j+3] = mode['DisturbXSens_DisturbanceTable'][disturbanceSheetRow,DisturbanceCaseInd]

            #### Sensitivity Table from CStability!J8
            #Calculate column L of the Sensitivity Table in CStability tab, has length CS_NmodelCoef
            #DELETEContrastSensitivityTableCol = np.where(mode['DisturbXSens_ContrastSensitivityVectorsColLabels'] == mode['systName'])[0] #Finds appropriate column mactching the system
            ContrastSensitivityTableCol = mode['DisturbXSens_ContrastSensitivityVectorsColLabels'].index(mode['SensitivityCase']) #Finds appropriate column mactching the system
            #DELETEDisturbaceCaseInd = np.where(mode['SensitivityCases'] == DisturbanceCase)[0] #I used toe incorrect thing here
            SensitivityTableL = mode['DisturbXSens_SensitivityMUF'][:,MUFindex] #CStability!L9-29
            SensitivityTableO = np.asarray([mode['DisturbXSens_ContrastSensitivityVectorsTable'][i + planetAnnZones*CS_NmodelCoef,ContrastSensitivityTableCol] for i in np.arange(len(modeNames))])
                                    #Refers to column O of the Sensitivity Table in CStability tab, has length CS_NmodelCoef

            MUFSensitivities = np.asarray([np.multiply(SensitivityTableL,SensitivityTableO[:,i]) for i in np.arange(SensitivityTableO.shape[1])])[0] #CStability!M9-29

            #M, V, dM, and dV all belong to CStability NI Contribution Table CStability!U6. All have units 10^9 NI
            self.M_ij_NIcon = np.zeros((len(modeNames),len(disturbanceCategories)*4))
            self.V_ij_NIcon = np.zeros((len(modeNames),len(disturbanceCategories)*4))
            self.dM_ij_NIcon = np.zeros((len(modeNames),len(disturbanceCategories)*4))
            self.dV_ij_NIcon = np.zeros((len(modeNames),len(disturbanceCategories)*4))
            for i in np.arange(len(modeNames)): #Iterate down rows of mode names
                for j in np.arange(len(disturbanceCategories)):
                    self.M_ij_NIcon[i,4*j] = MUFSensitivities[i]*(self.M_ij_disturbance[i,4*j]*disturbanceMults[i])**2.
                    self.V_ij_NIcon[i,4*j+1] = MUFSensitivities[i]*(self.V_ij_disturbance[i,4*j+1]*disturbanceMults[i])**2.
                    self.dM_ij_NIcon[i,4*j+2] = MUFSensitivities[i]*(self.dM_ij_disturbance[i,4*j+2]*disturbanceMults[i])**2.
                    self.dV_ij_NIcon[i,4*j+3] = MUFSensitivities[i]*(self.dV_ij_disturbance[i,4*j+3]*disturbanceMults[i])**2.

            #Total NI Contributions
            self.M_j = np.zeros(len(disturbanceCategories))
            self.V_j = np.zeros(len(disturbanceCategories))
            self.dM_j = np.zeros(len(disturbanceCategories))
            self.dV_j = np.zeros(len(disturbanceCategories))
            for j in np.arange(len(disturbanceCategories)):
                self.M_j[j] = np.sum(self.M_ij_NIcon[:,4*j]) #CStability!U31 and every "M" summation of that table
                self.V_j[j] = np.sum(self.V_ij_NIcon[:,4*j+1]) #CStability!V31 and every "V" summation of that table
                self.dM_j[j] = np.sum(self.dM_ij_NIcon[:,4*j+2]) #CStability!W31 and every "dM" summation of that table
                self.dV_j[j] = np.sum(self.dV_ij_NIcon[:,4*j+3]) #CStability!X31 and every "dV" summation of that table

            #### Initial Raw Contrast Table
            IntialRawContrastColInd = mode['DisturbXSens_InitialRawContrastColLabels'].index(mode['SenseCaseSel'])
            mode['DisturbXSens_InitialRawContrastTable'][:,IntialRawContrastColInd]
            #AnnZone #a single value of planetAnnZones
            #MUFindex #Same as DisturbanceCaseInd #index from MUFcase
            M_CGI_initial_NI = mode['DisturbXSens_InitialRawContrastTable'][0+2*planetAnnZones+2*5*MUFindex,IntialRawContrastColInd]#CStability!D46
            #DELETE=OFFSET(InitialRawContrastOrigin, 0+2*AnnZone+2*5*MUFindex, 1+MATCH($O$8, InitialRawConrast!$A$1:$AC$1,0) - (COLUMN(InitialRawContrastOrigin)+1))
            V_CGI_initial_NI = mode['DisturbXSens_InitialRawContrastTable'][1+2*planetAnnZones+2*5*MUFindex,IntialRawContrastColInd]#CStability!E46

            sumM = np.sum(self.M_j)*0.000000001 + M_CGI_initial_NI
            sumV = np.sum(self.V_j)*0.000000001 + V_CGI_initial_NI
            sumdM = np.sqrt(2.*sumM*np.sum(self.dM_j*0.000000001))
            sumdV = np.linalg.norm(self.dV_j*0.000000001)#np.sqrt(np.sum(dV_j**2.))###Check if these are equivalent

            NI_to_Contrast = mode['DisturbXSens_NItoContrastTable'][planetAnnZones,IntialRawContrastColInd]

            dC_CG = np.sqrt(sumdM**2. + sumdV**2.)/NI_to_Contrast#k_pp #CStability!E5 and CStability!H34
            print("dC_CG: " + str(dC_CG))
            print("M_CGI_initial_NI: " + str(M_CGI_initial_NI))
            print("V_CGI_initial_NI: " + str(V_CGI_initial_NI))
            print("sumM: " + str(sumM))
            print("sumV: " + str(sumV))
            print("sumdM: " + str(sumdM))
            print("sumdV: " + str(sumdV))
            print("dC_CG: " + str(dC_CG))
        else:
            dC_CG = C_CG/(5.*k_pp) #SNR!E6
        print("dC_CG: " + str(dC_CG))


        
        print("f_SR: " + str(f_SR))
        print("A_col: " + str(A_col))
        print("tau_PS: " + str(tau_PS))
        print("deta_QE: " + str(deta_QE))
        C_pmult = f_SR*A_col*tau_PS*deta_QE
       
        print("C_pmult: " + str(C_pmult))
        print("F_p: " + str(F_p))
        C_p = F_p*C_pmult
        
        C_b = ENF**2.*(r_pl + k_sp*(r_sp + r_ezo) + k_det*(r_lzo + r_DN + r_CIC + r_lum + r_RN))
        
        C_sp = f_SR*F_s*dC_CG*I_pk*m_pixCG*tau_refl*A_col*deta_QE
        
        if returnExtra:    
            return C_p, C_b, C_sp, C_pmult, F_s        
        
        else:
            return C_p, C_b, C_sp

    def calc_dMag_per_intTime(self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None, TK=None):
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
            TK (TimeKeeping object):
                Optional TimeKeeping object (default None), used to model detector
                degradation effects where applicable.
            
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
        
        # get signal to noise ratio
        SNR = mode['SNR']
        
        # calculate planet delta magnitude
        dMagLim = np.zeros(len(sInds)) + 25
        if (C_b is None) or (C_sp is None):
            _, C_b, C_sp, C_pmult, F_s = self.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dMagLim, WA, mode, TK=TK, returnExtra=True)
        dMag = -2.5*np.log10( SNR/(F_s*C_pmult) * np.sqrt(C_sp**2 + C_b/intTimes) )
        
        return dMag
