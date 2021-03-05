from EXOSIMS.Prototypes.OpticalSystem import OpticalSystem
from EXOSIMS.OpticalSystem.Nemati import Nemati
import astropy.units as u
from astropy.io import fits
import astropy.constants as const
import numpy as np
import scipy.stats as st
import scipy.optimize as opt
import os
from scipy import interpolate
from scipy.optimize import fsolve

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
        #DELETE amici_mode = [self.observingModes[i] for i in np.arange(len(self.observingModes)) if self.observingModes[i]['instName'] == 'amici-spec']
        ContrastScenario = [self.observingModes[i]['ContrastScenario'] for i in np.arange(len(self.observingModes)) if 'ContrastScenario' in self.observingModes[i].keys()]
        ContrastScenarioIndex = [i for i in np.arange(len(self.observingModes)) if 'ContrastScenario' in self.observingModes[i].keys()]
        if np.any(np.asarray(ContrastScenario)=='DisturbXSens'): #DELETElen(amici_mode) > 0:
            import csv
            #find index of amici_mode
            #DELETEamici_mode_index = [i for i in np.arange(len(self.observingModes)) if self.observingModes[i]['instName'] == 'amici-spec'][0] #take first amici-spec instName found
            amici_mode_index = [i for i in ContrastScenarioIndex if self.observingModes[i]['ContrastScenario'] == 'DisturbXSens'][0] #take first amici-spec instName found
            
            #Specifically for the Disturb X Sens observing mode (STILL NOT SURE HOW TO DECIDED IT IS DISTURB X SENS MODE)
            
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

            #### LOAD IN Disturbance Table from Disturbance Tab in Bijan2019 spreadsheet
            #I copied Disturbance!H6-U1181 to a text file and converted the range into CSV
            fname = self.observingModes[amici_mode_index]['DisturbXSens_DisturbanceTable']

            self.observingModes[amici_mode_index]['DisturbXSens_DisturbanceTable'] = extractedCSVTable(fname) #Disturbance table on the Disturbance Sheet in Bijan2019 model
            self.observingModes[amici_mode_index]['DisturbanceCases'] = ['rqt10hr', 'rqt10hr_1mas', 'rqt171012', 'live', 'rqt10hr171212', 'rqt40hr171212', 'rqt10hr171221', 'rqt40hr171221',\
                    'cbe10hr171221', 'rqt10hr180109', 'rqt40hr180109', 'cbe10hr180109', 'cbe10hr180130', 'cbe10hr180306'] #Disturbance!H4-U4
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
            self.observingModes[amici_mode_index]['DisturbXSens_ContrastSensitivityVectorsColLabels'] #Sensitivities!D4-P4

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
            t_EOL = 63. # mission total lifetime in months taken from the Spreadsheet
        else:
            t_now = (TK.currentTimeNorm.to(u.d)).value/30.4375 # current time in units of months
            t_EOL = TK.missionLife.to('d').value/30.4375

        f_ref = self.ref_Time # fraction of time spent on ref star for RDI
        if float(f_ref) == 0:
            # if f_ref isn't set then assume it's 0.2
            f_ref = 0.2
        dmag_s = self.ref_dMag # reference star dMag for RDI
        ppFact = TL.PostProcessing.ppFact(WA) # post processing factor

        # This will match the value of 2 in the spreadsheet and not raise the
        # assertion error of ppFact being between 0 and 1
        k_pp = 1/ppFact

        m_s = TL.Vmag # V magnitude

        D_PM = self.pupilDiam # primary mirror diameter in units of m
        f_o = self.obscurFac # obscuration due to secondary mirror and spiders
        f_s = self.shapeFac # aperture shape factor

        lam = mode['lam'] # wavelenght in units of nm
        inst_name = mode['instName'] # instrument name
        BW = mode['BW'] # bandwidth
        syst = mode['syst'] # starlight suppression system
        inst = mode['inst'] # instrument dictionary

        lam_D = lam.to(u.m)/(D_PM*u.mas.to(u.rad)) # Diffraction limit

        F_0 = TL.starF0(sInds,mode)*BW*lam

        # Setting in the json file that differentiates between MCBE, ICBE, REQ
        try:
            CS_setting = syst['core_stability_setting']
        except:
            CS_setting = 'MCBE'

        #Contrast Scenario related to DisturbXSens
        if mode['ContrastScenario'] == 'DisturbXSens':
            if 'C_CG' in mode.keys() and 'dC_CG' in mode.keys():
                C_CG = mode['C_CG'] #SNR!T45
                dC_CG = mode['dC_CG']
            elif 'sumM' in mode.keys() and 'sumV' in mode.keys() and 'NI_to_Contrast' in mode.keys() and \
                    'sumdM' in mode.keys() and 'sumdV' in mode.keys():
                NI_to_Contrast = mode['NI_to_Contrast']
                sumM = mode['sumM']
                sumV = mode['sumV']
                C_CG = (sumM + sumV)/NI_to_Contrast
                sumdM = mode['sumdM']
                sumdV = mode['sumdV']
                dC_CG = np.sqrt(sumdM**2. + sumdV**2.)/NI_to_Contrast
            else: #Load all the csv files
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
                MUFindex = mode['MUFcases'].index(mode['MUFCase']) # CStability!E26 MUFindex, MUFcases Sensitivity!C1-G1

                #i is the Mode Number in ModeNames, j is the Disturbance Table Category 
                modeNameNumbers = np.arange(len(modeNames)) #CStability!U40 Table rows. TODO double check if there are other modes possible...

                #### Annular Zone List 
                #mode['DisturbXSens_AnnZoneMasterTable']
                #AnnZoneMasterColLabels AnnZoneList!C1-O1
                #SenseCaseSel SNR!T31
                AnnZoneTableCol = np.where(np.asarray(mode['AnnZoneMasterColLabels']) == mode['SenseCaseSel'])[0][0]
                planetWAListMin = mode['DisturbXSens_AnnZoneMasterTable'][0:5,AnnZoneTableCol]
                planetWAListMax = mode['DisturbXSens_AnnZoneMasterTable'][5:,AnnZoneTableCol]

                #TODO figure out why the 1e-XX below is used in the spreadsheet
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
                ContrastSensitivityTableCol = mode['DisturbXSens_ContrastSensitivityVectorsColLabels'].index(mode['SensitivityCase']) #Finds appropriate column mactching the system
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
                V_CGI_initial_NI = mode['DisturbXSens_InitialRawContrastTable'][1+2*planetAnnZones+2*5*MUFindex,IntialRawContrastColInd]#CStability!E46
                NI_to_Contrast = mode['DisturbXSens_NItoContrastTable'][planetAnnZones,IntialRawContrastColInd]#CStability!D20

                sumM = np.sum(self.M_j)*0.000000001 + M_CGI_initial_NI
                sumV = np.sum(self.V_j)*0.000000001 + V_CGI_initial_NI
                sumdM = np.sqrt(2.*sumM*np.sum(self.dM_j*0.000000001))
                sumdV = np.linalg.norm(self.dV_j*0.000000001)#np.sqrt(np.sum(dV_j**2.))###Check if these are equivalent

                C_CG = (sumM + sumV)/NI_to_Contrast
                dC_CG = np.sqrt(sumdM**2. + sumdV**2.)/NI_to_Contrast #k_pp #CStability!E5 and CStability!H34
        elif mode['ContrastScenario'] == '2019_PDR_Update':
            # This is the new contrast scenario from the spreadsheet
            # Draw the values for the coronagraph contrast from the csv files
            positional_WA = (WA.to(u.mas)/lam_D).value

            # Draw the necessary values from the csv files
            core_stability_x, C_CG_y, C_extsta_y, C_intsta_y = self.get_csv_values(syst['core_stability'], 'r_lam_D', CS_setting + '_AvgRawContrast', 
                CS_setting + '_ExtContStab', CS_setting + '_IntContStab')
            C_CG_interp = interpolate.interp1d(core_stability_x, C_CG_y, kind='linear', fill_value=0., bounds_error=False)
            C_CG = C_CG_interp(positional_WA)*1e-9

            # Get values for dC_CG
            C_extsta_interp = interpolate.interp1d(core_stability_x, C_extsta_y, kind='linear', fill_value=0., bounds_error=False)
            C_extsta = C_extsta_interp(positional_WA)

            C_intsta_interp = interpolate.interp1d(core_stability_x, C_intsta_y, kind='linear', fill_value=0., bounds_error=False)
            C_intsta = C_intsta_interp(positional_WA)

            dC_CG = np.sqrt(C_extsta**2 + C_intsta**2)*10**(-9) #SNR!E6
        else: #use default CGDesignPerf
            C_CG = syst['core_contrast'](lam, WA) # coronagraph contrast
            dC_CG = C_CG/(5.*k_pp) #SNR!E6

        # THIS IS FOR TESTING THE DIFFERENCE BETWEEN INTERPOLATION AND THE SPREADSHEET'S FLOORING
        # cgperf_WA = np.arange(5.9, 20.1, 0.3)*lam_D/10**3*u.arcsec
        # # print(WA)
        # WA = min(cgperf_WA, key = lambda x:abs(x-WA))
        # print(f'WA: {WA}')

        A_PSF = syst['core_area'](lam, WA) # PSF area
        # This filter will set the PSF area when core_area is not given or the lambda function is
        # outside the given range
        # Check to see if it's an array or a single value
        if type(A_PSF) == np.float64:
            if A_PSF == 0:
                A_PSF = np.pi*(np.sqrt(2.)/2.*lam/self.pupilDiam)**2.
        else:
            A_PSF[A_PSF == 0] = np.pi*(np.sqrt(2.)/2.*lam/self.pupilDiam)**2.
        I_pk = syst['core_mean_intensity'](lam, WA) # peak intensity
        tau_core = syst['core_thruput'](lam, WA)*inst['MUF_thruput'] # core thruput
        tau_occ = syst['occ_trans'](lam, WA) # Occular transmission

        R = inst['Rs'] # resolution


        # THIS IS FOR TESTING THE DIFFERENCE BETWEEN INTERPOLATION AND THE SPREADSHEET'S FLOORING
        # QE_lambdas = np.arange(300, 1000, 10)*u.nm
        # lam = min(QE_lambdas, key = lambda x:abs(x-lam))

        eta_QE = inst['QE'](lam)[0].value # quantum efficiency
        refl_derate = inst['refl_derate']

        Nlensl = inst['Nlensl']
        lenslSamp = inst['lenslSamp']
        lam_c = inst['lam_c']
        lam_d = inst['lam_d'] # AK7
        k_s = inst['k_samp'] # AK19
        CTE_derate = inst['CTE_derate']
        ENF = inst['ENF'] # excess noise factor
        k_d = inst['dark_derate']
        pixel_size = inst['pixelSize']
        n_pix = inst['pixelNumber']**2.

        try:
            tau_pol = mode['tau_pol']
        except:
            tau_pol = 1

        t_MF = t_now/t_EOL #Mission fraction = (Radiation Exposure)/EOL

        if 'amici' in inst_name.lower():
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
            f_SR = 1./(BW*R)
            m_pix = Nlensl*(lam/lam_c)**2*lenslSamp**2.
        else: #Imaging Mode
            f_SR = 1.0
            m_pix = A_PSF*(np.pi/180./3600.)**2.*(lam/lam_d)**2*(2*D_PM/lam_c)**2

        # Get the file that has throughput information if a file is given
        try:
            thput_filename = inst['THPUT']
            OTA_TCA, CGI = self.get_csv_values(thput_filename, 'CBE_OTAplusTCA', 'CBE_CGI')
        except:
            OTA_TCA = 0.751
            CGI = 0.425

        tau_refl = OTA_TCA*CGI

        A_col = f_s*D_PM**2.*(1. - f_o)

        for i in sInds:
            F_s = F_0*10.**(-0.4*m_s[i])
        F_P_s = 10.**(-0.4*dMag)
        F_p = F_P_s*F_s

        #ORIGINALm_pixCG = A_PSF*(D_PM/(lam_d*k_s))**2.*(np.pi/180./3600.)**2.
        m_pixCG = A_PSF*(np.pi/180./3600.)**2./((lam_d*k_s)/D_PM)**2.

        # Calculations of the local and extra zodical flux
        F_ezo = F_0*fEZ*u.arcsec**2.# U63
        F_lzo = F_0*fZ*u.arcsec**2. # U64

        tau_unif = tau_occ*tau_refl*tau_pol

        tau_psf = tau_core/tau_occ
        # Set all values where tau_occ was calculated outside the working angles to a value of 0 for
        # the psf function
        tau_PS = tau_unif*tau_psf #SNR!AB82
        tau_sp = tau_refl*tau_pol # tau_pol is the polarizer thruput SNR!AB43. tau_sp is the speckle throughput

        r_pl_ia = f_SR*F_p*A_col*tau_PS*eta_QE #SNR!AB5

        #ORIGINALr_sp = f_SR*F_s*C_CG*I_pk*m_pixCG*tau_refl*A_col*eta_QE
        r_sp_ia = f_SR*F_s*C_CG*I_pk*m_pixCG*tau_sp*A_col*eta_QE #Dean replaces with tau_sp as in Bijan latex doc and  excel sheet

        ezo_inc = f_SR*F_ezo*A_PSF*A_col*tau_unif #U66

        lzo_inc = f_SR*F_lzo*A_PSF*A_col*tau_unif #U67
        r_zo_ia = (ezo_inc+lzo_inc) * eta_QE


        try:
            # Dark current
            det_filename = inst['DET']
            dark1, dark2, detEOL = self.get_csv_values(det_filename, 'Dark1', 'Dark2', 'DetEOL_mos')
        except:
            # Standard values
            dark1 = 1.5
            dark2 = 0.5
            detEOL = 63
        darkCurrent = dark1+(t_EOL/detEOL)*dark2
        darkCurrentAdjust = 1 # This is hardcoded in the spreadsheet

        darkCurrentAtEpoch = (darkCurrent*darkCurrentAdjust)*u.ph/3600*(1/u.s)
        r_ph = darkCurrentAtEpoch + (r_pl_ia + r_sp_ia + r_zo_ia)/m_pix # AC8
        t_f = [min(80, max(1, 0.1/i.decompose().value)) for i in r_ph]*u.s # U40

        # print("r_ph: " + str(r_ph.decompose()))
        try:
            k_RN, k_EM, L_CR, PC_threshold, is_PC, CR_1, CR_2, pixels_across = self.get_csv_values(det_filename, 'ReadNoise_e', 'EM_gain', 'CRtailLen_gain', 'PCThresh_nsigma', 'isPC_bool', 'CRtailLen1', 'CRtailLen2', 'PixelsAcross_pix')
        except:
            k_RN = 100
            k_EM = 1900
            L_CR = 195
            PC_threshold = 5
            is_PC = 1
            CR_1 = 0.01615
            CR_2 = 66.75
            pixels_across = 1024
        if is_PC: # SNR AK28
            k_ERN = 0
        else:
            k_ERN = k_RN/k_EM

        if 'REQ' not in CS_setting:
            L_CR = CR_1*k_EM+CR_2

        signal_pix_frame = t_f*r_ph # AC9

        PC_eff_loss = 1-np.exp(-PC_threshold*k_RN/k_EM)
        eta_PC = 1-PC_eff_loss # PC Threshold Efficiency SNR!AJK45
        eta_HP = 1. - t_MF/20. #SNR!AJ39
        eta_CR = 1. - (5*(1/u.s)*1.7*t_f)*L_CR/pixels_across**2 #SNR!AJ48
        try:
            dqeFluxSlope, dqeKnee, dqeKneeFlux = self.get_csv_values(det_filename, 'CTE_dqeFluxSlope', 'CTE_dqeKnee', 'CTE_dqeKneeFlux')
        except:
            dqeFluxSlope = 3.24
            dqeKnee = 0.858
            dqeKneeFlux = 0.089
        # Now uses the fudgeFactor instead of .5*CTE_derate
        eta_NCT = [max(0., min(1. + t_MF*(dqeKnee - 1.), 1. + t_MF*(dqeKnee - 1.) +\
                 t_MF*dqeFluxSlope*(i.decompose().value - dqeKneeFlux))) for i in signal_pix_frame][0] #SNR!AJ41

        tf_cts_pix_frame = t_f*r_ph*eta_NCT # Counts per pixel per frame after transfer
        eta_CL = (1-np.exp(-tf_cts_pix_frame.value))/tf_cts_pix_frame.value # PC Coincidence Efficiency or Coincidence Loss SNR!AJ46

        deta_QE = eta_QE*eta_PC*eta_HP*eta_CL*eta_CR*eta_NCT
        r_ezo = ezo_inc*deta_QE
        r_lzo = lzo_inc*deta_QE

        f_b = 10.**(0.4*dmag_s)

        k_sp = 1. + 1./(f_ref*f_b)
        k_det = 1. + 1./(f_ref*f_b**2.)
        # Get the CIC info from the csv file and use it to compute the CIC at epoch
        try:
            det_CIC1, det_CIC2, det_CIC3, det_CIC4 = self.get_csv_values(det_filename, 'CIC1', 'CIC2', 'CIC3', 'CIC4')
        except:
            det_CIC1 = 0.8
            det_CIC2 = 0.005
            det_CIC3 = 4500
            det_CIC4 = 0.01
        k_CIC = det_CIC1 * (det_CIC2 + (k_EM/det_CIC3)*det_CIC2 + t_MF*det_CIC4)
        r_CIC = ENF**2 * k_CIC * (m_pix/t_f)

        dark_current = (dark1 + t_MF*(dark2 - dark1))/(3600*u.s) # AK 34

        #ORIGINALr_dir = 625.*m_pix*(pixel_size/(0.2*u.m))**2*u.ph/u.s
        #ORIGINAL GCRFlux = 5./u.cm**2./u.s #evants/cm^2/s, StrayLight!G36, relativistic event rate
        try:
            GCRFlux = mode['GCRFlux']/u.cm**2./u.s #evants/cm^2/s, StrayLight!G36, relativistic event rate
        except:
            GCRFlux = 5 /u.cm**2/u.s
        #ORIGINAL photons_per_relativistic_event = 250.*u.ph/u.mm #ph/event/mm, StrayLight!G37, Cherenkov Ceiling assuming no CaF2 BaF2 from graph in paper  by Viehman & Eubanks 1976
        try:
            photons_per_relativistic_event = mode['photons_per_relativistic_event']*u.ph/u.mm #ph/event/mm, StrayLight!G37, Cherenkov Ceiling assuming no CaF2 BaF2 from graph in paper  by Viehman & Eubanks 1976
        except:
            photons_per_relativistic_event = 250*u.ph/u.mm
        lumrateperSolidAng = photons_per_relativistic_event/(2.*np.pi) #39.8 #ph/Sr/event/mm StrayLight!G38
        #ORIGINAL luminescingOpticalArea = 0.785*u.cm**2. #cm^2, StrayLight!G39, The beam diameter at the color filter and imaging lens is 5mm.
        #     #the imaging lens is an achromatic doublet. The thickness is 4mm BK7 glass and 2 mm SF2 glass. The polarized imaging
        #     # has additional up to 10mm thick glass (quartz) before the lens.
        try:
            luminescingOpticalArea = mode['luminescingOpticalArea']*u.cm**2. #cm^2, StrayLight!G39, The beam diameter at the color filter and imaging lens is 5mm.
        except:
            luminescingOpticalArea = 0.7854*u.cm**2
            #the imaging lens is an achromatic doublet. The thickness is 4mm BK7 glass and 2 mm SF2 glass. The polarized imaging
            # has additional up to 10mm thick glass (quartz) before the lens.
        #ORIGINALOpticalThickness = 4.0*u.mm #mm
        try:
            OpticalThickness = mode['OpticalThickness']*u.mm #mm StrayLight!G40
        except:
            OpticalThickness = 4 * u.mm
        #luminescingOpticalDistance = 0.1*u.m #m, StrayLight!G41
        try:
            luminescingOpticalDistance = mode['luminescingOpticalDistance']*u.m #StrayLight!G41
        except:
            luminescingOpticalDistance = 0.1 * u.m
        Omega_Signal = m_pix*pixel_size**2./luminescingOpticalDistance**2. #2.88*10.**-7. #Sr, StrayLight!G42,
        r_dir = (GCRFlux*lumrateperSolidAng*luminescingOpticalArea*OpticalThickness*Omega_Signal).decompose() #StrayLight!G44

        #ORIGINALr_indir = (1.25*np.pi*m_pix/n_pix*u.ph/u.s).decompose()
        try:
            s_baffling = mode['s_baffling'] #0.001 StrayLight!G47
        except:
            s_baffling = 0.001
        Omega_Indirect = 2.*np.pi*s_baffling*m_pix/n_pix #StrayLight!G43
        r_indir = (GCRFlux*lumrateperSolidAng*luminescingOpticalArea*OpticalThickness*Omega_Indirect).decompose() #StrayLight!G45

        r_stray = r_dir + r_indir #StrayLight!G46
        eta_e = r_stray*deta_QE #StrayLight!G51

        r_DN = ENF**2.*dark_current*m_pix*u.ph
        r_CIC = ENF**2.*k_CIC*m_pix*u.ph/t_f
        r_lum = ENF**2.*eta_e
        r_RN = k_ERN**2.*m_pix*u.ph/t_f


        C_pmult = f_SR*A_col*tau_PS*deta_QE

        C_p = (F_p*C_pmult)/u.ph

        C_b = ((ENF**2.*(r_pl_ia + k_sp*(r_sp_ia + r_ezo) + k_det*(r_lzo + r_DN + r_CIC + r_lum)) +
                k_det*r_RN)).decompose()/u.ph

        C_sp = (f_SR*F_s*(dC_CG/k_pp)*I_pk*m_pixCG*tau_sp*A_col*deta_QE).decompose()/u.ph

        # Check for the values that are given when the planet is
        # outside of working angle values and set them to 0
        C_p[np.isnan(C_p)] = 0
        C_sp[np.isnan(C_sp)] = 0
        C_b[np.isnan(C_b)] = 0
        if returnExtra:
            return C_p, C_b, C_sp, C_pmult, F_s

        else:
            return C_p, C_b, C_sp

    def calc_dMag_per_intTime(self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None, TK=None):
        """Finds achievable dMag for one integration time per star in the input
        list at one working angle. Uses scipy's root-finding function fsolve

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
        args = (TL, sInds, fZ, fEZ, WA, mode, TK, intTimes)
        x0 = np.zeros(len(intTimes))+20
        dMag = fsolve(self.dMag_per_intTime_obj, x0=x0, args=(TL, sInds, fZ, fEZ, WA, mode, TK, intTimes))
        return dMag

    def dMag_per_intTime_obj(self, dMag, *args):
        '''
        Objective function for calc_dMag_per_intTime's fsolve function that uses calc_intTime from
        Nemati and then compares the value to the true intTime value

        Args:
            dMag (ndarray):
                dMags being tested in root-finding
            *args:
                all the other arguments that calc_intTime needs
        '''
        TL, sInds, fZ, fEZ, WA, mode, TK, true_intTime = args
        est_intTime = self.calc_intTime(TL, sInds, fZ, fEZ, dMag, WA, mode, TK)
        return true_intTime - est_intTime

    def get_csv_values(self, csv_file, *headers):
        '''
        This takes in a csv file and returns the values in the columns associated with the headers
        given as args

        Arguments:
            csv_file (str or Path):
                location of the csv file to read
            *headers (str):
                The headers that correspond to the columns of data to be returned

        Returns:
            return_vals (list):
                The values in the columns for every given header. Ordered the same way they were
                given as inputs
        '''
        filename = os.path.normpath(os.path.expandvars(csv_file))
        try:
            csv_vals = np.genfromtxt(filename, delimiter=',', skip_header=1)
        except:
            print('Error when reading csv file:')
            print(filename)
            csv_vals = np.genfromtxt(filename, delimiter=',', skip_header=1)
        # Get the number of rows, accounting for the fact that 1D numpy arrays behave different than 2D arrays
        # when calling the len() function
        if len(np.shape(csv_vals)) == 1:
            footer_len = 1
        else:
            footer_len = len(csv_vals)
        csv_headers = np.genfromtxt(filename, delimiter=',', skip_footer = footer_len, dtype=str)

        # Delete any extra rows at the end of the csv files, such as ones labeled "Comments:"
        if footer_len != 1:
            csv_vals = csv_vals[~np.isnan(csv_vals).any(axis=1)]

        # List to be appended to that gets
        return_vals = []
        for header in headers:
            if footer_len == 1:
                header_location = np.where(csv_headers == header)[0]
                return_vals.append(csv_vals[header_location][0])
            else:
                header_location = np.where(csv_headers == header)[0][0]
                return_vals.append(csv_vals[:, header_location])
        return return_vals
