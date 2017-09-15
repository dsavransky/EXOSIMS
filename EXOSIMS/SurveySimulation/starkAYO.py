from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import itertools
import datetime
import time
import json
from scipy.optimize import fsolve
import scipy
import timeit#used for debugging and timing my functions
from astropy.time import Time
import scipy
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer #debugging
import csv #saving completenesses to file
import os
import os.path
import scipy.integrate as integrate
import logging
import pdb

# the EXOSIMS logger
Logger = logging.getLogger(__name__)

class deanAYO(SurveySimulation):
    """deanAYO 
    
    This class implements a Scheduler that selects the current highest Completeness/Integration Time.

    
    """

    def __init__(self, **specs):
        
        SurveySimulation.__init__(self, **specs)
        
        # bring inherited class objects to top level of Survey Simulation
        SU = self.SimulatedUniverse
        self.PlanetPopulation = SU.PlanetPopulation
        self.PlanetPhysicalModel = SU.PlanetPhysicalModel
        self.OpticalSystem = SU.OpticalSystem
        self.ZodiacalLight = SU.ZodiacalLight
        self.BackgroundSources = SU.BackgroundSources
        self.PostProcessing = SU.PostProcessing
        self.Completeness = SU.Completeness
        self.TargetList = SU.TargetList

        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        ZL = self.ZodiacalLight
        
        self.starVisits = np.zeros(TL.nStars,dtype=int)
        self.fullSpectra = np.zeros(SU.nPlans, dtype=int)
        self.partialSpectra = np.zeros(SU.nPlans, dtype=int)
        self.starTimes = np.zeros(TL.nStars)*u.d
        self.starRevisit = np.array([])
        self.starExtended = np.array([])
        self.lastDetected = np.empty((TL.nStars, 4), dtype=object)

        detMode = filter(lambda mode: mode['detectionMode'] == True, OS.observingModes)[0]
        spectroModes = filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes)
        self.mode = detMode
        
        #Create and start Schedule
        self.schedule = np.arange(TL.nStars)#self.schedule is meant to be editable
        self.schedule_startSaved = np.arange(TL.nStars)#preserves initial list of targets
        sInd = None
        
        DRM = {}
        
        TK.allocate_time(Obs.settlingTime + self.mode['syst']['ohTime'])
        
        
        # 0/ initialize arrays
        slewTime = np.zeros(TL.nStars)*u.d#
        fZs = np.zeros(TL.nStars)/u.arcsec**2#
        t_dets = np.zeros(TL.nStars)*u.d
        tovisit = np.zeros(TL.nStars, dtype=bool)
        sInds = self.schedule_startSaved#the list of potential targets sInds is schedule_startSaved

        ##Removed 1 Occulter Slew Time
        startTime = TK.currentTimeAbs + slewTime#create array of times the length of the number of stars (length of slew time)
        
        #The below has all been taken verbatim (almost) from completeness. They are necessary to compute with f_s
        self.Completeness.f_dmagsv = np.vectorize(self.Completeness.f_dmags)
        dMagLim = self.Completeness.dMagLim
        self.dmag_startSaved = np.linspace(1, dMagLim, num=1500,endpoint=True)
        dmag = self.dmag_startSaved
        amax = self.PlanetPopulation.arange.max().value
        emax = self.PlanetPopulation.erange.max()
        rmax = amax*(1+emax)
        SMH = self.SimulatedUniverse.Completeness#using this because I had to port over a lot of code. Equivalent to self.Completeness...
        beta = np.linspace(0.0,np.pi,1000)*u.rad
        Phis = SMH.PlanetPhysicalModel.calc_Phi(beta)
        SMH.Phi = scipy.interpolate.InterpolatedUnivariateSpline(beta.value,Phis,k=3,ext=1)
        SMH.val = np.sin(SMH.bstar)**2*SMH.Phi(SMH.bstar)
        b1 = np.linspace(0.0, SMH.bstar, 25000)
        SMH.binv1 = scipy.interpolate.InterpolatedUnivariateSpline(np.sin(b1)**2*SMH.Phi(b1), b1, k=3, ext=1)
        b2 = np.linspace(SMH.bstar, np.pi, 25000)
        b2val = np.sin(b2)**2*SMH.Phi(b2)
        SMH.binv2 = scipy.interpolate.InterpolatedUnivariateSpline(b2val[::-1], b2[::-1], k=3, ext=1)
        z = np.linspace(SMH.pmin*SMH.Rmin**2, SMH.pmax*SMH.Rmax**2, 1000)
        fz = np.zeros(z.shape)
        for i in xrange(len(z)):
            fz[i] = SMH.f_z(z[i])
        print('Done fz')
        SMH.dist_z = scipy.interpolate.InterpolatedUnivariateSpline(z, fz, k=3, ext=1)
        SMH.rgrand2v = np.vectorize(SMH.rgrand2)
        r = np.linspace(SMH.rmin, SMH.rmax, 1000)
        fr = np.zeros(r.shape)
        for i in xrange(len(r)):
            fr[i] = SMH.f_r(r[i])#this is really slow. On the order of 30 sec or so
        print('Done fr')
        SMH.dist_r = scipy.interpolate.InterpolatedUnivariateSpline(r, fr, k=3, ext=1)

        dir_path = os.path.dirname(os.path.realpath(__file__))#find current directory of survey Simualtion
        fname = '/starComps.csv'
        self.starComps_startSaved = list()#contains all comps vs dmag for stars
        self.compData = list()#contains all data from completeness csv file
        
        IWA = self.OpticalSystem.IWA.value#of telescope. to be used for min s
        OWA = self.OpticalSystem.OWA.value#of telescope to be used for max s

        lastTime = timeit.default_timer()

        ##NEED SOME CHECKS TO DETRMINE WHETHER COMPLETENESS IS OKAY OR NOT

        #IF Completeness Distribution has not been calculated Generate Completeness and Generate that File
        if not os.path.isfile(dir_path+fname):# or len(self.compData) != 402:#If this file does not exist or the length of the file is not appropriate 
            #Calculate Completeness for Each Star##############################
            print('Calculate Completeness 1')
            starSmin = np.zeros(len(sInds))#contains Smin for each star
            starSmax = np.zeros(len(sInds))#contains Smax for each star
            #Calculate Smin and Smax for all Stars in List
            for q in range(len(sInds)):#iterate through each star
                star_dist = self.TargetList.dist[sInds[q]].value
                starSmin[q] = star_dist*IWA
                starSmax[q] = star_dist*OWA
            absSmax = np.amin(starSmin)#find largest Smax of entire list
            absSmin = np.amax(starSmax)#find smallest Smin of entire list

            #Calculate S range to calculate over
            Smat = np.linspace(absSmin,absSmax,400)
            fComp = np.zeros((len(Smat),len(dmag)))
            for j in range(len(dmag)):#for each arbitrary dmaglim between 0 and dMagLim
                dmaglim = dmag[j]#artificial dmaglim to be set by a smaller integration time
                #fComp = range(len(Smat))#same lengt of smat and contains all fComp
                for i in range(len(Smat)):#iterate through star separations
                        fComp[i][j] = self.Completeness.f_s(Smat[i],dmaglim)#calculates fComp
                #myComp[j] = (starS[q][-1]-starS[q][0])*sum(fComp)/40#sum fComp over S and normalize
                print('Calculating fComp for dmag ' + str(j) + ' of ' + str(len(dmag)))
            #generates fComp which is a fComp[Smat][dmag] matrix and holds the completeness value for each one of those
            #Aug 28, 2017 execution time 3701 for 400x5000... depends on size...
            print('Calculating Completeness PDF time = '+str(timeit.default_timer() - lastTime))
            lastTime = timeit.default_timer()
            print('Done Calculating Completeness 1')

            #Save Completeness to File######################################
            try:#Here we delete the previous completeness file
                print('Trying to save Completeness to File')
                timeNow = datetime.datetime.now()
                timeString = str(timeNow.year)+'_'+str(timeNow.month)+'_'+str(timeNow.day)+'_'+str(timeNow.hour)+'_'+str(timeNow.minute)+'_'
                fnameNew = '/' + timeString +  'moved_starComps.csv'
                os.rename(dir_path+fname,dir_path+fnameNew)
            except OSError:
                pass
            with open(dir_path+fname, "wb") as fo:
                wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
                wr.writerow(Smat)#writes the 1st row Separations
                wr.writerow(dmag)#writes the 2nd row dmags
                for i in range(len(fComp)):
                    wr.writerow(fComp[i])#self.starComps_startSaved)#writes the completeness
                fo.close()
                #Aug 28, 2017 execution time 
                print('Saving Completeness PDF To File time = '+str(timeit.default_timer() - lastTime))
                lastTime = timeit.default_timer()
                print('Finished Saving Completeness To File')

        #Load Completeness From File######################################
        print('Load Completeness pdf from File')
        SmatRead = list()
        dmagRead = list()
        with open(dir_path+fname, 'rb') as f:
            reader = csv.reader(f)
            your_list = list(reader)
            f.close()
        for i in range(len(your_list)):
            if i == 0:
                SmatRead = np.asarray(your_list[i]).astype(np.float)
            elif i == 1:
                dmagRead = np.asarray(your_list[i]).astype(np.float)
            else:
                tmp = np.asarray(your_list[i]).astype(np.float)
                self.compData.append(tmp)#Unfiltered Completeness Lists
        #Sept 6, 2017 execution time 0.0922 sec
        #print('Load Completeness File time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        print('Finished Loading Completeness pdf from File')
        #Iterate through each star, interpolate which completeness to start summing at do for all stars.
        #if not os.path.isfile(dir_path+fname) or len(self.starComps_startSaved) != sInds.shape[0]:#If this file does not exist or the length of the file is not appropriate 
        starS = list()#contains separation amtrix for each star
        compvsdmagDat = np.zeros((len(sInds),len(dmagRead)))


        fname = '/starCompsAllStars.csv'
        #IF the Completeness vs dMag for Each Star File Does Not Exist, Calculate It
        if not os.path.isfile(dir_path+fname):# or len(self.compData) != 402:#If this file does not exist or the length of the file is not appropriate 
            print('Calculating Completeness for Each Star vs dmag')
            for q in range(len(sInds)):#iterates through each star
                star_dist = self.TargetList.dist[sInds[q]].value
                Smin = star_dist*IWA
                Smax = star_dist*OWA
                #find index of Smin
                SminIndex = np.argmin(np.abs([(x - Smin) for x in SmatRead]))
                #find index of Smax
                SmaxIndex = np.argmin(np.abs([(Smax - x) for x in SmatRead]))
                for i in range(len(dmagRead)):
                    #Iterate through each column and sum down the column over the range specified
                    compvsdmagDat[q][i] = sum([self.compData[x][i] for x in range(SmaxIndex,SminIndex)])*np.abs(SmatRead[SminIndex]-SmatRead[SmaxIndex])/np.abs(SmaxIndex-SminIndex)
                self.starComps_startSaved.append(compvsdmagDat[q])
            del compvsdmagDat
            #Save Completeness to File######################################
            try:#Here we delete the previous completeness file
                print('Trying to save Completeness vs dMag for Each Star to File')
                timeNow = datetime.datetime.now()
                timeString = str(timeNow.year)+'_'+str(timeNow.month)+'_'+str(timeNow.day)+'_'+str(timeNow.hour)+'_'+str(timeNow.minute)+'_'
                fnameNew = '/' + timeString +  'moved_starCompsAllStars.csv'
                os.rename(dir_path+fname,dir_path+fnameNew)
            except OSError:
                print('There was an error writing the file')
                pass
            with open(dir_path+fname, "wb") as fo:
                wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
                for i in range(len(sInds)):#iterate through all stars
                    wr.writerow(self.starComps_startSaved[i])#self.starComps_startSaved)#writes the completeness
                fo.close()
                #Aug 28, 2017 execution time 
                print('Saving Completeness vs dMag for Each Star to File time = '+str(timeit.default_timer() - lastTime))
                lastTime = timeit.default_timer()
                print('Finished Saving Completeness vs dMag for Each Star to File')

        #Load Completeness vs dMag for Each Star From File######################################
        print('Load Completeness for Each Star from File')
        with open(dir_path+fname, 'rb') as f:
            reader = csv.reader(f)
            your_list = list(reader)
            f.close()
        for i in range(len(your_list)):
            tmp = np.asarray(your_list[i]).astype(np.float)
            self.starComps_startSaved.append(tmp)#Unfiltered Completeness Lists
        #Sept 6, 2017 execution time  sec
        #print('Load Completeness File time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        print('Finished Loading Completeness vs dMags for Each Star from File')
        #Iterate through each star, interpolate which completeness to start summing at do for all stars.



        #MAKE THIS FASTER BY LOADING FROM FILE.... THIS TAKES MAYBE 30 SECONDS FOR CALCULATION OR SOMETHING.... JUST A TAD BIT TOO LONG
        #Sept 6, 2017 execution time 70.629 sec
        print('CalcEach Star Completeness time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        print('Done Calculating Completeness for Each Star vs dmag')

        #Calculate myTint for an initial startimg position#########################
        slewTime2 = np.zeros(sInds.shape[0])*u.d#initialize slewTime for each star
        startTime2 = TK.currentTimeAbs + slewTime2#calculate slewTime
        
        #update self.Tint, self.rawTint
        self.calcTint(None)#Calculate Integration Time for all Stars (relatively quick process)

        print("Done starkAYO init")
        #END INIT##################################################################

    def next_target(self, sInds, mode):
        start_time_AlgorithmSpeeds = timeit.default_timer()
        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        ZL = self.ZodiacalLight
        dmag = self.dmag_startSaved
        WA = OS.WA0
        slewTime = np.zeros(TL.nStars)*u.d
        startTime = TK.currentTimeAbs+slewTime
        tovisit = np.zeros(TL.nStars, dtype=bool)
        """Generate Schedule based off of AYO at this instant in time

        Args:
            sInds (integer array):
                Array of all indexes of stars from sInds = np.array(TL.nStars)
            dmag ():


            WA ():

            dir_path (string):
                string containing the path to starComps.csv. This should be the EXOSIMS SurveySimulation folder
            fname (string):
                string containing starComps.csv (aside: This name should be improved to encode the list of stars being used)
            startTime (float array):
                observation start time of each observation
            tovisit (idk):
                uhhhhh

        Returns:
            DRM
            schedule
            sInd ():
                detection
            t_det (float):
                detection time of next observation
        """
        mode = self.mode
        SU = self.SimulatedUniverse
        self.PlanetPopulation = SU.PlanetPopulation
        self.PlanetPhysicalModel = SU.PlanetPhysicalModel
        self.OpticalSystem = SU.OpticalSystem
        self.ZodiacalLight = SU.ZodiacalLight
        self.BackgroundSources = SU.BackgroundSources
        self.PostProcessing = SU.PostProcessing
        self.Completeness = SU.Completeness
        self.TargetList = SU.TargetList

        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        ZL = self.ZodiacalLight
        TK.obsStart = TK.currentTimeNorm.to('day')

        #Create DRM
        DRM = {}

        # Aluserslocate settling time + overhead time
        TK.allocate_time(Obs.settlingTime + mode['syst']['ohTime'])
        #shouldn't need this??? maybe its here because of DRM...
        lastTime = start_time_AlgorithmSpeeds
        #print(str(timeit.default_timer()-lastTime))
        #Calculate Tint at this time#####################################
        self.calcTint(None)#self.schedule)#updates self.Tint and self.rawTint
        myTint = self.Tint
        #Aug 28, 2017 execution time 3.27
        #print('calcTint time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        
    
        #Calculate C vs T spline######################################
        self.splineCvsTau()
        spl = self.spl_startSaved
        print('SPL 1 is ' + str(len(spl)))
        starComps = self.starComps
        Tint = self.Tint
        #Aug 28, 2017 execution time 3.138
        #print('splineCvsTau time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        #A Note: Technically starComps is the completeness of each star at each dmag specified in the init section of starkAYO
        #the myTint is evaluated at each one of these dmags.
        ##############################################################
    
        #Calculate C/T vs T spline####################################
        self.splineCbyTauvsTau()
        spl2 = self.spl2_startSaved
        #Aug 28, 2017 execution time 0.0909
        #print('splineCbyTauvsTau time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        ##############################################################

        #Calculate dC/dTau############################################
        self.splinedCbydTauvsTau()
        splDeriv = self.splDeriv_startSaved
        #Aug 28, 2017 execution time 0.024
        #print('splinedCbydTauvsTau time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        ##############################################################

        ## Run Dmitry's Code....

        ######################

        ##Plot All the Things
        #self.plotCompvsTau()
        #self.plotCompbyTauvsTau()
        #self.plotdCompbydTauvsTau()
        #print(saltyburrito)
        #I HAVE CONFIRMED AT THIS POINT THAT spl2 is a positive spline and ok.
        ####

        #Pull Untouched Star List#####################################
        sInds = self.schedule_startSaved
        ##############################################################
        """###FILTERING IS TEMPORARILY REMOVED FOR RUNNING DMITRY'S TEST DO NOT DELETE
        #FILTER OUT STARS########################################################
        #start with self.schedule_startSaved

        #Filter KOGOODSTART..... There is a better implementation of this.....
        #We will not filter KOGOODEND using the logic that any star close enough to the keepout region of the sun will have poor C/Tau performance... This assumption should be revaluated at a future date.
        kogoodStart = Obs.keepout(TL, sInds, startTime[sInds], mode)#outputs array where 0 values are good keepout stars
        sInds = sInds[np.where(kogoodStart)[0]]
        
        sInds1 = sInds
        #Filter out previously visited stars#######################################
        if np.any(sInds):
            tovisit[sInds] = (self.starVisits[sInds] == self.starVisits[sInds].min())
            if self.starRevisit.size != 0:
                dt_max = 1.*u.week
                dt_rev = np.abs(self.starRevisit[:,1]*u.day - TK.currentTimeNorm)
                ind_rev = [int(x) for x in self.starRevisit[dt_rev < dt_max,0] if x in sInds]
                tovisit[ind_rev] = True
            sInds = np.where(tovisit)[0]
        print('NumStars Before ' + str(len(sInds1)) + ' After Filter ' + str(len(sInds)))
        #Aug 28, 2017 execution time 0.235
        #print('KOGOODSTART Filter time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        #########################################################################
        """

        #Define Intial Integration Times#########################################
        missionLength = TK.missionLife.to(u.day).value

        #Initial list of integration times for each star
        t_dets = np.zeros(sInds.shape[0]) + missionLength/float(sInds.shape[0]) #the list of integration times for each star
        dCbydT = np.zeros(sInds.shape[0])#initialize dCbydT matrix
        CbyT = np.zeros(sInds.shape[0])#initialize CbyT matrix
        Comp00 = np.zeros(sInds.shape[0])#initialize Comp00 matrix

        #Here we create the filtered splDeriv etc
        fspl = list()
        fstarComps = list()
        fTint = list()
        fspl2 = list()
        fsplDeriv = list()
        fsplDeriv = [splDeriv[i] for i in sInds]#these filter down into only the splDeriv in sInds
        fspl = [spl[i] for i in sInds]
        fTint = [Tint[i] for i in sInds]
        fspl2 = [spl2[i] for i in sInds]
        #fstarComps = [starComps[i] for i in sInds]

        #Initialize dC/dT##########################
        for n in range(sInds.shape[0]):
            dCbydT[n] = fsplDeriv[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            #calculates initial completeness at initial integration times...
        ###########################################

        lastTime = timeit.default_timer()
        #Initialize C/T############################
        for n in range(sInds.shape[0]):
            CbyT[n] = fspl2[n](t_dets[n])#CbyT is the editable list of CbyT for each star
        ###########################################
        print('Initialization Stuffs time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        print(CbyT)
        lastTime = timeit.default_timer()
        fZ = 0./u.arcsec**2#ZL.fZ(Obs, TL, sInds, startTime, self.mode)#self.mode['lam'])
        fEZ = 0./u.arcsec**2# ZL.fEZ0
        CbyTtmp = self.Completeness.dcomp_dt(t_dets*u.d, TL, sInds, fZ, fEZ, WA, mode)
        print('Initialization Stuffs time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        print(CbyT-CbyTtmp)
        print(CbyT)
        print(saltyburrito)

        #Initialize Comp############################
        for n in range(sInds.shape[0]):
            Comp00[n] = fspl[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
        ###########################################
        #Aug 28, 2017 execution time 0.020
        #print('Initialization Stuffs time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()



        #print(saltyburrito1)
        #I confirm spl2 is still valid at this point. fslp2 is also good at this point

        #Update maxIntTime#########################
        #Calculate the Maximum Integration Time and dmag for each star
        maxTint, maxdmag = self.calcMaxTint(sInds, fspl2)
        self.maxTint = maxTint
        self.maxdmag = maxdmag
        assert maxTint[0] != 0#i think unnecessary at this point
        maxIntTime = maxTint
        #Aug 28, 2017 execution time 3.213
        #print('calcMaxTint time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()

        #Check maxIntTime Validity
        for i in range(maxIntTime.shape[0]):
            if(maxIntTime[i] < 0):
                print('ERROR in MaxIntTime Calculation. A Value is Negative!')
        ###########################################


        #Check Validity of Star Integration Times####################################
        ##THE PROBLEM CAUSING HIGH COMP00 VALUES STEMS FROM INITIAL TIMES ASSIGNED BEING SUBSTANTIALLY LARGER THAN MAXIMUM INTEGRATION TIMES
        #find problematic t_dets
        ok_t_dets = [maxIntTime[x] < t_dets[x] for x in range(len(t_dets))]#Find T_dets greater than the maximum allowed and assign them to TRUE in matrix form
        #Sum Excess Time
        excessTime = 0
        for i in range(t_dets.shape[0]):
            if(ok_t_dets[i]):#This star was initially assigned an invalid t_dets
                #print('too large t_dets ' + str(t_dets[i]) + ' Its maxIntTime ' + str(maxIntTime[i]))
                excessTime = excessTime + (t_dets[i] - maxIntTime[i])#sum excess time off this observation
                t_dets[i] = maxIntTime[i]#set t_dets to the maxIntTime
        #Redistribute excessTime
        dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime = self.distributedt(excessTime, dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime)
        #Sept 6, 2017 execution time 0.00155 sec
        #print('Distribute Initial Integration Times = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        #All stars should have valid assigned integration times given the schedule...
        #####################################################################

        #REORDER SO SINDS ARE IN ORDER FOR COMPAIRISON TO DIMITRY'S CODE
        #CAN DELETE AFTER DONE WITH DMITRY'S TESTING
        sortIndex = np.argsort(-sInds,axis=-1)[::-1]#sorts indicies and spits out list containing the indicies of the sorted list from smallest to largest. argsort sorts from smallest to largest, [::-1] flips array
        #index list from highest dCbydT to lowest dCbydT....

        #2 Reorder data#######################
        #sort star index list, integration time list, dC/dT list, splDeriv list, myTint list (contains integrationt time splines for each star)
        sInds = sInds[sortIndex]
        #if numits > 200:
        #    pdb.set_trace()
        t_dets = t_dets[sortIndex]
        dCbydT = dCbydT[sortIndex]
        fsplDeriv = [fsplDeriv[i] for i in sortIndex]
        tmp2 = list()#this tmp2 operation efficienct might be able to be increased
        tmp2[:] = [Tint[i] for i in sortIndex]#there must be a better single line way to do this...
        Tint = tmp2
        del tmp2
        fspl2 = [fspl2[i] for i in sortIndex]
        fspl = [fspl[i] for i in sortIndex]
        CbyT = CbyT[sortIndex]
        Comp00 = Comp00[sortIndex]
        maxIntTime = maxIntTime[sortIndex]
        ###################################################################

        #print(saltyburrito2)
        #I confirm that spl2 is still valid at this point. fspl2 is also good

        firstIteration = 1
        numits = 0
        lastIterationSumComp  = -10000000 #this is some ludacrisly negative number to ensure sumcomp runs. All sumcomps should be positive
        while numits < 100000 and sInds is not None:
            #print('numits ' + str(numits))#we impose numits to limit the total number of iterations of this loop. This may be depreicated later
            numits = numits+1#we increment numits each loop iteration

            #Update dC/dT#################
            for n in range(sInds.shape[0]):
                dCbydT[n] = fsplDeriv[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###############################

            #Update C/T############################
            for n in range(sInds.shape[0]):
                CbyT[n] = fspl2[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###########################################

            #Update Comp############################
            for n in range(sInds.shape[0]):
                Comp00[n] = fspl[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###########################################
            #print('splinedCbydTauvsTau time = '+str(timeit.default_timer() - lastTime))
            lastTime = timeit.default_timer()

            #Sort Descending by dC/dT###########################################################################
            #1 Find Indexes to sort by############
            #print(saltyburrito)

            dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime, sacrificedStarTime = self.sacrificeStarCbyT(dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime)
            
            ################################################
            #THIS SECTION OF CODE ACCOUNTS FOR OVERHEAD IN THE SYSTEM
            if(sInds.shape[0] <= missionLength-7):#condition where the observation schedule could be done within the mission time
                if(firstIteration == 1):#redistribute dt on first Iteration
                    #print('We now have a realizable mission schedule!')
                    firstIteration = 0
                    t_dets = np.zeros(sInds.shape[0]) + (missionLength-sInds.shape[0]*1)/float(sInds.shape[0])
                    #print('Sum t_dets ' + str(sum(t_dets)) + ' miss-sindshape ' + str(missionLength-sInds.shape[0]*1))
                    #print(t_dets)
                    sacrificedStarTime = 0
                else:
                    sacrificedStarTime = sacrificedStarTime + float(1)#we add 1 day to account for the overhead gained by dropping a star
            else:
                sacrificedStarTime = sacrificedStarTime
            ##################################################
            #Aug 28, 2017 execution time ????infinitesimally small
            #print('sacrifice time = '+str(timeit.default_timer() - lastTime))
            lastTime = timeit.default_timer()

            #5 Distribute Sacrificed Time to new star observations
            #print('Last Time to Redistribute ' + str(sacrificedStarTime))
            dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime = self.distributedt(sacrificedStarTime, dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime)
            #Aug 28, 2017 execution time 9e-6
            #Sept 13, 2017 execution time ~0.2 depending on schedule length for dt/10
            #print('distributedt time = '+str(timeit.default_timer() - lastTime))
            lastTime = timeit.default_timer()

            #TECHNICALLY THESE HAVE ALREADY BEEN UPDATED YB DISTRIBUTEDT
            #6 Update Lists #########################################################3
            #Update dC/dT#################
            for n in range(sInds.shape[0]):
                dCbydT[n] = fsplDeriv[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###############################

            #Update C/T############################
            for n in range(sInds.shape[0]):
                CbyT[n] = fspl2[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###########################################

            #Update Comp############################
            for n in range(sInds.shape[0]):
                Comp00[n] = fspl[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###########################################

            #print(saltyturrito)

            #Sort Descending by dC/dT#######################################
            #dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime= self.reOrder(dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime)
            # #1 Find Indexes to sort by############
            # sortIndex = np.argsort(dCbydT,axis=-1)[::-1]#sorts indicies and spits out list containing the indicies of the sorted list from largest to smallest. argsort sorts from smallest to largest, [::-1] flips array
            # #index list from highest dCbydT to lowest dCbydT....


            #AYO Termination Conditions
            #1 If the nthElement that time was added to has dCbydT less than the last star (star to be taken from next)
            #2 If the nthElement is the same as the last element

            ### IF ALL TARGETS ARE MAXIMALLY SORTED, STOP STARKayo PROCESS. I.E. IF THEY ALL HAVE MAXINTTIME
            
            #if any(i < 0.05 for i in spl2[sInds](t_dets)):
            #    continue#run next iteration
            print(str(numits) + ' SumComp ' + str(sum(Comp00)) + ' Sum(t_dets) ' + str(sum(t_dets)) + ' sInds ' + str(sInds.shape[0]*float(1)) + ' TimeConservation ' + str(sum(t_dets)+sInds.shape[0]*float(1)) + ' Avg C/T ' + str(np.average(CbyT)))
            if(sum(CbyT)<0):
                print('CbyT<0')
                print(saltyburrito)


            if 1 >= len(dCbydT):#if this is the last ement in the list
                print('dCbydT maximally sorted (This probably indicates an error)')
                print(saltyburrito)
                break
            maxDeltaIndex = np.argmax(abs(maxIntTime-t_dets))#finds the index where the two are maximally different
            if(abs(t_dets[maxDeltaIndex]-maxIntTime[maxDeltaIndex])<0.00001):
                print('THIS SHOULD JUSTIFY A TERMINATION CONDITION')
                print(saltyburrito)
            # #If the total sum of completeness at this moment is less than the last sum, then exit
            # if(sum(Comp00) < lastIterationSumComp):# and (len(sInds) < 20)):
            #     print('Successfully Sorted List!!')
            #     print('SumComp is ' + str(sum(Comp00)))
            #     print(len(sInds))
            #     #Define Output of AYO Process
            #     sInd = sInds[0]
            #     t_det = t_dets[0]*u.d
            #     #Update List of Visited stars########
            #     self.starVisits[sInd] += 1
            #     self.schedule = sInds
            #     del lastIterationSumComp
            #     return DRM, sInd, t_det #sInds
            # else:#else set lastIterationSumComp to current sum Comp00
            #     lastIterationSumComp = sum(Comp00)
            #     print('SumComp '+str(lastIterationSumComp) + ' with sInds left '+str(len(sInds)))

        ##Define Output of AYO Process
        sInd = sInds[0]
        t_det = t_dets[0]*u.d

        ##### Delete Some Terms    for use in next iteration        
        #del splDeriv
        #del spl

        #Update List of Visited stars########
        self.starVisits[sInd] += 1
        #####################################

        self.schedule = sInds
        return DRM, sInd, t_det #sInds

    def choose_next_target(self,old_sInd,sInds,slewTime):
        """Choose next target based on truncated depth first search 
        of linear cost function.
        
        Args:
            old_sInd (integer):
                Index of the previous target star
            sInds (integer array):
                Indices of available targets
            slewTime (float array):
                slew times to all stars (must be indexed by sInds)
                
        Returns:
            sInd (integer):
                Index of next target star
        
        """
    
        #
        OS = self.OpticalSystem
        Comp = self.Completeness
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        
        comps = TL.comp0[sInds]#completeness of each star in TL
        updated = (self.starVisits[sInds] > 0)#what does starVisits contain?
        comps[updated] = Comp.completeness_update(TL, sInds[updated],TK.currentTimeNorm)
        tint = TL.tint0[sInds]
        
        selMetric=comps/tint#selMetric is the selection metric being used. Here it is Completeness/integration time

        #Here I select the target star to observe
        tmp = sInds[selMetric == max(selMetric)]#this selects maximum completeness/integration time
        sInd = tmp[0]#casts numpy array to single integer
        
        return sInd

    def f_dmags2(self, s, dmag):
        """Calculates the joint probability density of dMag and projected
        separation
        
        Created to put s first in the function (necessary since first term is integrated over)

        Args:
            dmag (float):
                Value of dMag
            s (float):
                Value of projected separation (AU)
        
        Returns:
            f (float):
                Value of joint probability density
        
        """
        self.mindmag = self.Completeness.mindmag
        self.maxdmag = self.Completeness.maxdmag
        if (dmag < self.mindmag(s)) or (dmag > self.maxdmag(s)):
            f = 0.0
        else:
            ztest = (s/self.x)**2*10.**(-0.4*dmag)/self.val
            if ztest >= self.zmax:
                f = 0.0
            elif (self.pconst & self.Rconst):
                f = self.f_dmagsz(self.zmin,dmag,s)
            else:
                if ztest < self.zmin:
                    f = integrate.fixed_quad(self.f_dmagsz, self.zmin, self.zmax, args=(dmag, s), n=61)[0]
                else:
                    f = integrate.fixed_quad(self.f_dmagsz, ztest, self.zmax, args=(dmag, s), n=61)[0]
        return f

        
        #def f_dmag2(self,dmag,star_dist):
        #"""Marginalizes the joint probability density function of dMag and projected separation
        #over projected separation ****This produces Completeness as a Function of dmag
        #
        #Args:
        #    dmag (float):
        #        Value of dMag
        #    s (float):
        #        Value of projected separation (AU)
        #
        #Returns:
        #    f (float):
        #        Value of joint probability density
        #
        #"""
        #IWA = self.OpticalSystem.IWA.value
        #OWA = self.OpticalSystem.OWA.value
        #Smin = star_dist*IWA
        #Smax = star_dist*OWA
        #f = integrate.fixed_quad(self.f_dmags2, Smin, Smax, args=(dmag,), n=61)[0]
        #return f

    def f_dmag2(self, dmag, star_dist):
        """Calculates probability density of dmag marginalized
        from Smin to Smax #note this is similat to the f_s function...
        
        Args:
            dmag (float):
                dMag to evaluate completeness as
            star_dist ():
                determines min and max separation marginalizing over...
                
        Returns:
            f (float):
                Probability density
        
        """

        if (s == 0.0) or (s == self.rmax):
            f = 0.0
        else:
            d1 = self.mindmag(s)
            d2 = self.maxdmag(s)
            if d2 > dmaglim:
                d2 = dmaglim
            if d1 > d2:
                f = 0.0
            else:
                f = integrate.fixed_quad(self.f_dmagsv, d1, d2, args=(s,), n=31)[0]

        return f

    def calcTint(self, sInds):
        """Calculates integration times for all stars in sInds given a dmag. If None is passed to sInds,
        the integration times for all stars in self.schedule_startSaved will be calculated.
        This could be made more efficient bt calculating integration times for only those
        stars still in the schedule but that would elimintate the possibility of reading 
        those stars to the schedule...

        Intended to be called from starkAYO only***

        Args:
            sInds (None or np.ndarray or shape #stars)
            calculates orbital position of SC at currentTime

        Returns:
            This function updates self.rawTint
            rawTint contains all Tint over all dmag_startSaved. We must do this in order to filter future values.
            rawTint has dimensions rawTint[nStars][len(dmag_startSaved)]

            This function updates self.Tint
            Tint contains Tint greater than 10**-10
            Dimensions are Tint[nStars][Tint > 10**-10]
        """
        dmag = self.dmag_startSaved
        newTint, newRawTint = self.calcTint_core(sInds,dmag)
        self.rawTint = newRawTint
        self.Tint = newTint

    def calcMaxTint(self, sInds, fspl2):
        """calcMaxTint estimates the maximum integration time allowed and the associated maximum dmag allowed for a given star
        NOTE that this only returns an approximate... The resolution is determined by the fidelity of the dmag_startSaved array

        Inputs:
        sInds. requires a list of stars to calculate MaxTint for

        Returns:
        maxTint
            The maximum integration times for stars in sInds?... might not be in sInds but Tint...
        maxdmag
            The maximum dmag achievable (dmag where maxTint occurs)
        """

        Tint, rawTint = self.calcTint_core(sInds, self.dmag_startSaved)
        #Tint is of form Tint[#stars][#dmag] for stars in sInds. rawTint has all stars
        #print(len(Tint))
        #print(len(rawTint))
        #pdb.set_trace()
        #print(saltyburrito)
        #Situation where
        #Select Maximum Tint based on calculated values
        maxTint = np.zeros(len(Tint[:]))#declare array
        maxdmag = np.zeros(len(Tint[:]))#declare array
        for i in xrange(sInds.shape[0]):#0, len(Tint[:])):#iterate through all stars
            occur = np.argmax(Tint[i][:])#Find index of maxTint
            maxTint[i] = Tint[i][occur]
            maxdmag[i] = self.dmag_startSaved[occur]

        #Calculate MaxTint based on Maximum positive CbyT (still connected to expected CbyT peak) 
        #import matplotlib.pyplot as plt
        numInSeq = np.zeros([sInds.shape[0],100])
        seqEndIndex = np.zeros([sInds.shape[0],100])
        seqStartIndex = np.zeros([sInds.shape[0],100])
        for i in xrange(sInds.shape[0]):
            intTime = np.arange(100,dtype=np.float)/100*maxTint[i]
            tmpCbyT = fspl2[i](intTime)

            tmpCbyTbool = [fspl2[i](time) > 0 for time in intTime]
            CbyTindicies = [j for j, x in enumerate(tmpCbyTbool) if x]#pull out all indicies where CbyT is less than 0
            #find indicies
            for j in range(len(CbyTindicies)-1):
                #iterate through all indicies
                seqNum = 0
                if(CbyTindicies[j+1]-CbyTindicies[j] > 1.1):#there is more than 1 index between these points
                    seqNum = seqNum+1
                    numInSeq[i,seqNum] = 1
                    seqEndIndex[i] = j
                    seqStartIndex[i] = j+1
                else:
                    numInSeq[i,seqNum] = numInSeq[i,seqNum] + 1
            #Now we have Number of points in a sequence for each star, the start values, and the end values
            seqOfMaxLength = np.argmax(numInSeq[i,:])
            if(intTime[int(seqEndIndex[i,seqOfMaxLength])] < maxTint[i]):
                #print('We are swapping ' + str(maxTint[i]) + ' With ' + str(intTime[int(seqEndIndex[i,seqOfMaxLength])]) + ' for sInd ' + str(sInds[i]))
                maxTint[i] = intTime[int(seqEndIndex[i,seqOfMaxLength])]

            #plt.plot(intTime,tmpCbyT)
            #plt.axis([0,max(intTime),0,max(tmpCbyT)])
            #plt.show()
        return maxTint, maxdmag

    def calcTint_core(self, sInds, dmag):
        """Calculates integration times for all stars in sInds given a dmag. If None is passed to sInds,
        the integration times for all stars in self.schedule_startSaved will be calculated.
        This could be made more efficient bt calculating integration times for only those
        stars still in the schedule but that would elimintate the possibility of reading 
        those stars to the schedule...

        Intended to be called from starkAYO only***

        Args:
            sInds (None or np.ndarray or shape #stars)
            calculates orbital position of SC at currentTime

        Returns:
            This function updates self.rawTint
            rawTint contains all Tint over all dmag_startSaved. We must do this in order to filter future values.
            rawTint has dimensions rawTint[nStars][len(dmag_startSaved)]

            This function updates self.Tint
            Tint contains Tint greater than 10**-10
            Dimensions are Tint[nStars][Tint > 10**-10]
        """
        if sInds is None:
            sInds = self.schedule_startSaved
        #else:
        #    sInds = sInds

        #dmag = self.dmag_startSaved
        OS = self.OpticalSystem
        #WA = OS.WAint
        WA = OS.WA0
        ZL = self.ZodiacalLight
        TL = self.TargetList
        Obs = self.Observatory
        startTime = np.zeros(sInds.shape[0])*u.d + self.TimeKeeping.currentTimeAbs
        r_sc_list = Obs.orbit(startTime)#list of orbital positions the length of startTime
        fZ = 0./u.arcsec**2#ZL.fZ(Obs, TL, sInds, startTime, self.mode)#self.mode['lam'])
        fEZ = 0./u.arcsec**2# ZL.fEZ0
        print(' Note The fZ and fEZ are spoofed')
        Tint = np.zeros((sInds.shape[0],dmag.shape[0]))#array of #stars by dmags(500)
        #Tint=[[0 for j in range(sInds.shape[0])] for i in range(len(dmag))]
        for i in xrange(dmag.shape[0]):#len(dmag)):#Tint of shape Tint[StarInd, dmag]
                Tint[:,i] = OS.calc_intTime(TL, sInds, fZ, fEZ, dmag[i], WA, self.mode).value#it is in units of days
        
        newRawTint = list()#initialize list to store all integration times
        newTint = list()#initialize list to store integration times greater tha 10**-10
        for j in range(sInds.shape[0]):
                newRawTint.append(Tint[j,:])
                newTint.append(Tint[j][np.where(Tint[j] > 10**-10)[0]])

        #Here is where we add in Dmitry's overhead time...
        #for j in range(sInds.shape[0]):
        #    newTint[j] = newTint[j]+1

        #self.rawTint = newRawTint
        #self.Tint = newTint
        return newTint, newRawTint#in units of days

    def splineCvsTau(self):
        """Calculate the spline fit to Completeness vs Tau at simulation start time

        Returns:
            updates self.spl_startSaved
            Dimensions are spl_startSaved[nStars][Tint > 10**-10]
            updates self.starComps_startSaved
            Dimensions are starComps_startSaved[nStars][Tint > 10**-10]
            updates self.Tint for any Completeness exceeding 100
        """
        schedule = self.schedule_startSaved

        #we must remove any 0's that occur above the max value of completeness for the star
        #to do this, we filter any myTint that is less than some threshold value...
        ##Remove all Tints that are 0 and save to list
        self.calcTint(None)#self.schedule)#,self.mode,self.startTime2)
        rawTint = self.rawTint
        Tint = self.Tint

        starComps = list()
        for j in schedule:#xrange(schedule.shape[0]):
            starComps.append(self.starComps_startSaved[j][np.where(rawTint[j] > 10**-10)[0]])


        #HERE IS A TEMPORARY FIX FOR A PROBLEM IN COMPLETENESS CALCULATION########################
        #The root problem is that the current completeness calculation in this module (as of 5/22/2017)
        #produces completness values at dmags near/in excess of 1. Indicating that the completeness calculation is incorrect
        
        #if completeness has discontinuity, 
        #then all values above discontinuity must be truncated
        #and all corresponding Tint need to be truncated....
        index = None#will be used later
        for j in schedule:
            if starComps[j][-1] < 10**-15:#If the last value of starComps is a 0. We assume this means there exists a discontinuity in completeness
                #find How many Indicies are 0
                for k in range(len(starComps[j]))[::-1]:#Iterate through completeness from last index to first
                    #find first index of value greater than 10 **-15
                    if starComps[j][k] > 10**-15:
                        index = k
                        break
            if index is not None:
                starComps[j] = starComps[j][0:k]#starComps is now from 0 to max point
                Tint[j] = Tint[j][0:k]
                index = None
            else:#if index is None
                index = None
        ##########################################################################################

        spl = list()
        for q in range(schedule.shape[0]):#len(schedule)):
            spl.append(UnivariateSpline(Tint[q], starComps[q], k=4, s=0))#Finds star Comp vs Tint SPLINE
        self.spl_startSaved = spl#updates self.spl
        self.starComps = starComps#updates self.starComps
        self.Tint = Tint#update self.Tint

    def splineCbyTauvsTau(self):
        """Calculate the spline fit to Completeness/Tau vs Tau

        Args:

        Returns:
            updates self.spl2_startSaved (spl2 is the spline fitting C/T vs T)
            Dimensions are spl2_startSaved[nStars][Tint > 10**-15]
        """

        #self.calcTint(self.schedule)#,self.mode,self.startTime2)#I think we can remove this
        rawTint = self.rawTint
        Tint = self.Tint
        sInds = self.schedule_startSaved
        starComps = self.starComps

        #I think we can remove this
        #starComps = list()
        #for j in xrange(sInds.shape[0]):
        #        starComps.append(self.starComps_startSaved[j][np.where(rawTint[j] > 10**-15)[0]])

        #spl = self.spl_startSaved
        sInds = self.schedule_startSaved
        spl2 = list()
        for x in range(sInds.shape[0]):#len(sInds)):
            #spl2.append(UnivariateSpline(Tint[x],spl[x](Tint[x])/Tint[x], k=4, s=0))#I think we can remove this
            spl2.append(UnivariateSpline(Tint[x],starComps[x]/Tint[x], k=4, s=0))
        self.spl2_startSaved = spl2

    def splinedCbydTauvsTau(self):#,spl2,myTint,sInds):
        """Calculates the spline for dC/dT vs T

        Returns:
            self.splDeriv_startSaved
            Dimensions are splDeriv_startSaved[nStars][Tint > 10**-10]
        """
        sInds = self.schedule_startSaved
        spl2 = self.spl2_startSaved

        splDeriv = list()
        for x in range(len(sInds)):
            splDeriv.append(spl2[x].derivative())
        self.splDeriv_startSaved = splDeriv

    def plotCompvsTau(self):
        """Plots Completeness vs Integration Time for every star
        """
        print('Plot C vs T')
        #plot of Comp vs Tint for a few stars##################################################
        self.splineCvsTau()#Update self.spl and self.starComps
        starComps = self.starComps
        spl = self.spl_startSaved
        Tint = self.Tint
        TL = self.TargetList

        ##FOR CONFIRMATION OF DMITRY'S CODE####################
        tintRange = np.linspace(0.1,365,num=2000)*u.d
        DmitryCompDatamat = np.zeros((TL.nStars,len(tintRange)))
        self.DmitryCompData = list()
        with open('/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/Scripts/compAtdMaglimSavransky.csv', 'rb') as f:
            reader = csv.reader(f)
            your_list = list(reader)
            f.close()
        for i in range(len(your_list)):
            tmp = np.asarray(your_list[i]).astype(np.float)
            self.DmitryCompData.append(tmp)#Unfiltered Completeness Lists
        for i in range(len(tintRange)):#here we need to convert dmitry's comp for each star vs tint into a matrix
            for j in range(TL.nStars):
                DmitryCompDatamat[j][i] = self.DmitryCompData[i][j]
        ##############################################################
        
        #print(saltyburrito)
        print('The length of Tint is ' + str(len(Tint)))
        for m in xrange(len(Tint)):
            dtos = 24*60*60#days to seconds
            #PLOT COMPLETENESS VS INT TIME
            fig = plt.figure()
            plt.plot(Tint[m]*dtos,starComps[m],'o',Tint[m]*dtos,spl[m](Tint[m]),'-',tintRange.value*dtos,DmitryCompDatamat[m],'--')
            plt.xscale('log')
            axes = plt.gca()
            axes.set_xlim([10e-3,10e8])
            #axes.set_ylim([0,0.2])
            plt.grid(True)
            plt.ylabel('Completeness')
            plt.xlabel('Integration Time (s)')
            plt.title(TL.Name[m])
            folderfig = os.path.normpath(os.path.expandvars('$HOME/Pictures/CompletenessFigs'))
            filenamefig = 'CompvTint' + str(m) + '.png'
            figscriptfile = os.path.join(folderfig,filenamefig)
            fig.savefig(figscriptfile)
            plt.close()
        print('Done C vs T')
        print saltyburrito

    def plotCompbyTauvsTau(self):
        """Plots Completeness/Integration Time vs Integration Time for every star
        """
        #Update splines
        print('Start Plot C/T vs T')
        self.splineCvsTau()
        self.splineCbyTauvsTau()
        spl2 = self.spl2_startSaved
        Tint = self.Tint
        TL = self.TargetList

        for m in xrange(len(Tint)):
            dtos = 24*60*60
            #PLOT COMPLETENESS/INT TIME vs INT TIME
            fig = plt.figure()
            plt.plot(Tint[m]*dtos,spl2[m](Tint[m])/(Tint[m]*dtos),'-')
            plt.xscale('log')
            axes = plt.gca()
            axes.set_xlim([10e-4,10e6])
            plt.grid(True)
            plt.ylabel('Completeness/Tint')
            plt.xlabel('Integration Time (s)')
            plt.title(TL.Name[m])
            folderfig = os.path.normpath(os.path.expandvars('$HOME/Pictures/CompletenessFigs'))
            filenamefig = 'CompbyTintvTint' + str(m) + '.png'
            figscriptfile = os.path.join(folderfig,filenamefig)
            fig.savefig(figscriptfile)
            plt.close()
        print('Done C/T vs T')

    def plotdCompbydTauvsTau(self):
        """Plots dCompleteness/dTint vs Tint for every star
        """
        print('Plot dC/dT vs T')
        self.splineCvsTau()
        self.splineCbyTauvsTau()
        self.splinedCbydTauvsTau()
        splDeriv = self.splDeriv_startSaved
        Tint = self.Tint#myTint_startSaved
        TL = self.TargetList
        for m in xrange(len(Tint)):
            dtos = 24*60*60
            #PLOT COMPLETENESS VS INT TIME
            fig = plt.figure()
            plt.plot(Tint[m]*dtos,splDeriv[m](Tint[m]),'-')
            plt.xscale('log')
            axes = plt.gca()
            axes.set_xlim([10e-4,10e7])
            plt.grid(True)
            plt.ylabel('dC/dTau')
            plt.xlabel('Integration Time (s)')
            #plt.show()
            plt.title(TL.Name[m])
            folderfig = os.path.normpath(os.path.expandvars('$HOME/Pictures/CompletenessFigs'))
            filenamefig = 'dCompbydTintvTint' + str(m) + '.png'
            figscriptfile = os.path.join(folderfig,filenamefig)
            fig.savefig(figscriptfile)
            plt.close()
        print('Done dC/dT vs T')

    def saveComp(self, starComps):
        """Saves Completeness vs dmag for each star to folder

        Args:
            starComps (List)
                List of form Completeness = starComps[star][dmag index]
        Returns:

        """
        dir_path = os.path.dirname(os.path.realpath(__file__))#find current directory of survey Simualtion
        fname = '/starComps.csv'
        with open(dir_path+fname, "wb") as fo:
            wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
            wr.writerows(starComps)
            fo.close()

    def loadComp(self):
        """Loads Completeness from starComps.csv on this path

        Args:

        Returns:
            starComps (list):
                List of Completeness at dmags
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))#find current directory of survey Simualtion
        fname = '/starComps.csv'
        with open(dir_path+fname,'rb') as f:
            reader = csv.reader(f)
            your_list = list(reader)
            f.close()
        starComps = list()
        for i in range(len(your_list)):
            tmp = np.asarray(your_list[i])
            tmp = tmp.astype(np.float)
            starComps.append(tmp)
        return starComps

    def distributedt(self,sacrificedStarTime, dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime):#distributing the sacrificed time
        """#here we want to add integration time to the highest performing stars in our selection box
        #returns nTint for additional integration time added to each star.
        Args:
            sacrificedStarTime is the total amount of time to redistribute amoung the other stars
            maxIntTime is the list of approximate maximum integration times

        """
        timeToDistribute = sacrificedStarTime
        dt_static = sacrificedStarTime/50#1
        dt = dt_static
        #Now decide where to put dt
        numItsDist = 0
        while(timeToDistribute > 0):
            numItsDist = numItsDist + 1
            if(numItsDist > 100000):#this is an infinite loop check
                print('numItsDist>100000')
                break
            if(len(t_dets) <=1):
                break
            if(timeToDistribute < dt):#if the timeToDistribute is smaller than dt
                #check if timeToDistribute <=0 (distributedt Termination conditions!) NOTE I SHOULDN'T NEED THIS CODE HERE...
                #if(timeToDistribute <= 0):
                #    if(timeToDistribute < 0):#Check if less than 0. If so there is a serious problem
                #        print('ERROR: Time To Distribute is less than 0')
                #    #elif(timeToDistribute == 0):#this constitutes nominal operating conditions
                #    #    print('Time To Distribute is 0')#this is the value timeToDistribute should be otherwise there is a leak
                #    break
                #if no problems occur set dt
                dt = timeToDistribute#dt is now the timeToDistribute
                #print('dt starts as ' + str(dt))
            else:#timeToDistribute >= dt under nominal conditions, this is dt to use
                dt = dt_static#this is the maximum quantity of time to distribute at a time.  
                #THERE IS SOME UNNECESSARY CODE HERE BETWEEN THESE FIRST CHECKS AND THE CHECKS IN THE FOLLOWING FOR LOOP         
            ##
            #Find next star that is not at or above maxIntTime
            numStarsAtMaxIntTime = 0#initialize this
            for i in xrange(maxIntTime.shape[0]):#len(maxIntTime)):#iterate through each star from top to bottom
                #check if timeToDistribute is smaller than dt
                if(timeToDistribute <= dt):#in the case where
                    if(timeToDistribute > 0):
                        dt = timeToDistribute#dt needs to be set to the time remaining
                    elif(timeToDistribute == 0):
                        #print('Time To Distribute is 0')#we're done with these iterations
                        break
                    elif(timeToDistribute < 0):
                        print('ERROR Time To Distribute is less than 0')
                        print(saltyburrito)
                ##
                #Some checks on the value of dt
                if(dt <= 0):#There is either no more time to be assigned or the time to be assigned is negative
                    if(dt == 0):
                        dt=dt#DO NOTHING
                        #print('dt is 0 and we are done with these iterations')
                    elif(dt < 0):
                        print('ERROR dt is less than zero')
                    break#we break from the for loop before anything can be assigned
                ##Distribution of Time
                if(t_dets[i] >= maxIntTime[i]):#if we cannot add time to this star...
                    numStarsAtMaxIntTime = numStarsAtMaxIntTime + 1
                    if(numStarsAtMaxIntTime >= maxIntTime.shape[0]):#len(maxIntTime)):
                        #print('All Stars At Max Int Time')
                        return dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime
                    else:
                        dt=dt#DO NOTHING... (We need to skip this star so we assign no dt.)
                        continue
                elif(t_dets[i] < maxIntTime[i]):#Then we will add time to this star BUT WE MUST DECIDE HOW MUCH
                    #Check that added time will not Exceed maxIntTime
                    if(t_dets[i]+dt <= maxIntTime[i]):#If added dt will keep t_dets less than maxIntTime
                        t_dets[i] = t_dets[i]+dt#we add the time to the star
                        timeToDistribute = timeToDistribute - dt#we subtract from the total time to redistribute
                        dt = 0#we set dt to 0 indicating we are done distributing this dt
                    else:#(t_dets[i]+dt > maxIntTime[i]):#added dt would be greater than maxIntTime
                        #Add the most time you can to the first star
                        dt = dt - (maxIntTime[i]-t_dets[i])
                        timeToDistribute = timeToDistribute - (maxIntTime[i]-t_dets[i])
                        t_dets[i] = maxIntTime[i]
                #Update Lists #########################################################3
                #Update dC/dT#################
                for n in range(sInds.shape[0]):
                    dCbydT[n] = fsplDeriv[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
                ###############################

                #Update C/T############################
                for n in range(sInds.shape[0]):
                    CbyT[n] = fspl2[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
                ###########################################

                #Update Comp############################
                for n in range(sInds.shape[0]):
                    Comp00[n] = fspl[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
                ###########################################

                #reorder lists. will be used to determine most deserving star to give time to
                #Sept 13, runs in 0.0005 sec approx
                #lastTime = timeit.default_timer()
                dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime = self.reOrder(dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime)
                #print('distributedt time = '+str(timeit.default_timer() - lastTime))
                #lastTime = timeit.default_timer()
            #END For LOOP
            #print('Time to Distribute is ' + str(timeToDistribute))
        #END While LOOP
        return dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime

    def reOrder(self, dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime):
        sortIndex = np.argsort(dCbydT,axis=-1)[::-1]#sorts indicies and spits out list containing the indicies of the sorted list from largest to smallest. argsort sorts from smallest to largest, [::-1] flips array
        #index list from highest dCbydT to lowest dCbydT....

        #2 Reorder data#######################
        #sort star index list, integration time list, dC/dT list, splDeriv list, myTint list (contains integrationt time splines for each star)
        sInds = sInds[sortIndex]
        #if numits > 200:
        #    pdb.set_trace()
        t_dets = t_dets[sortIndex]
        dCbydT = dCbydT[sortIndex]
        fsplDeriv = [fsplDeriv[i] for i in sortIndex]
        tmp2 = list()#this tmp2 operation efficienct might be able to be increased
        tmp2[:] = [Tint[i] for i in sortIndex]#there must be a better single line way to do this...
        Tint = tmp2
        del tmp2
        fspl2 = [fspl2[i] for i in sortIndex]
        fspl = [fspl[i] for i in sortIndex]
        CbyT = CbyT[sortIndex]
        Comp00 = Comp00[sortIndex]
        maxIntTime = maxIntTime[sortIndex]

        #FOR DEBUGGING
        #print('sortNum sInd dCbydT maxIntTime CbyT Comp00 t_dets')
        #for i in range(len(dCbydT)):
        #    print(str(i) + ' ' + str(sInds[i]) + ' ' + str(dCbydT[i]) + ' ' + str(maxIntTime[i]) + ' ' + str(CbyT[i]) + ' ' + str(Comp00[i]) + ' ' + str(t_dets[i]))
        #print(saltyburrito)

        return dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime

    def reOrderCbyT(self, dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime):
        sortIndex = np.argsort(CbyT,axis=-1)[::-1]#sorts indicies and spits out list containing the indicies of the sorted list from largest to smallest. argsort sorts from smallest to largest, [::-1] flips array
        #index list from highest dCbydT to lowest dCbydT....

        #2 Reorder data#######################
        #sort star index list, integration time list, dC/dT list, splDeriv list, myTint list (contains integrationt time splines for each star)
        sInds = sInds[sortIndex]
        #if numits > 200:
        #    pdb.set_trace()
        t_dets = t_dets[sortIndex]
        dCbydT = dCbydT[sortIndex]
        fsplDeriv = [fsplDeriv[i] for i in sortIndex]
        tmp2 = list()#this tmp2 operation efficienct might be able to be increased
        tmp2[:] = [Tint[i] for i in sortIndex]#there must be a better single line way to do this...
        Tint = tmp2
        del tmp2
        fspl2 = [fspl2[i] for i in sortIndex]
        fspl = [fspl[i] for i in sortIndex]
        CbyT = CbyT[sortIndex]
        Comp00 = Comp00[sortIndex]
        maxIntTime = maxIntTime[sortIndex]

        #FOR DEBUGGING
        #print('sortNum sInd dCbydT maxIntTime CbyT Comp00 t_dets')
        #for i in range(len(dCbydT)):
        #    print(str(i) + ' ' + str(sInds[i]) + ' ' + str(dCbydT[i]) + ' ' + str(maxIntTime[i]) + ' ' + str(CbyT[i]) + ' ' + str(Comp00[i]) + ' ' + str(t_dets[i]))
        #print(saltyburrito)

        return dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime

    def reOrderMaxComp(self, dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime):
        sortIndex = np.argsort(Comp00,axis=-1)[::-1]#sorts indicies and spits out list containing the indicies of the sorted list from largest to smallest. argsort sorts from smallest to largest, [::-1] flips array
        #index list from highest dCbydT to lowest dCbydT....

        #2 Reorder data#######################
        #sort star index list, integration time list, dC/dT list, splDeriv list, myTint list (contains integrationt time splines for each star)
        sInds = sInds[sortIndex]
        #if numits > 200:
        #    pdb.set_trace()
        t_dets = t_dets[sortIndex]
        dCbydT = dCbydT[sortIndex]
        fsplDeriv = [fsplDeriv[i] for i in sortIndex]
        tmp2 = list()#this tmp2 operation efficienct might be able to be increased
        tmp2[:] = [Tint[i] for i in sortIndex]#there must be a better single line way to do this...
        Tint = tmp2
        del tmp2
        fspl2 = [fspl2[i] for i in sortIndex]
        fspl = [fspl[i] for i in sortIndex]
        CbyT = CbyT[sortIndex]
        Comp00 = Comp00[sortIndex]
        maxIntTime = maxIntTime[sortIndex]

        #FOR DEBUGGING
        #print('sortNum sInd dCbydT maxIntTime CbyT Comp00 t_dets')
        #for i in range(len(dCbydT)):
        #    print(str(i) + ' ' + str(sInds[i]) + ' ' + str(dCbydT[i]) + ' ' + str(maxIntTime[i]) + ' ' + str(CbyT[i]) + ' ' + str(Comp00[i]) + ' ' + str(t_dets[i]))
        #print(saltyburrito)

        return dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime

    def sacrificeStarCbyT(self, dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime):
        maxIntTimeCbyT, OneFifthmaxIntTimeCbyT, OneHalfmaxIntTimeCbyT, OneDayIntTimeCbyT = self.calcMaxCbyT(sInds, fspl2, maxIntTime)
        maxIntTimeC, OneFifthmaxIntTimeC, OneHalfmaxIntTimeC, OneDayIntTimeC = self.calcMaxComps(sInds, fspl, maxIntTime)
        sacrificeIndex = np.argmin(CbyT)#finds index of star to sacrifice

        #print(saltyburrito)


        #Need index of sacrificed star by this point
        sacrificedStarTime = t_dets[sacrificeIndex]#saves time being sacrificed
        dCbydT = np.delete(dCbydT,sacrificeIndex)
        sInds = np.delete(sInds,sacrificeIndex)
        t_dets = np.delete(t_dets,sacrificeIndex)
        fsplDeriv = np.delete(fsplDeriv,sacrificeIndex)
        Tint = np.delete(Tint,sacrificeIndex)
        fspl2 = np.delete(fspl2,sacrificeIndex)
        CbyT = np.delete(CbyT,sacrificeIndex)
        Comp00 = np.delete(Comp00,sacrificeIndex)
        fspl = np.delete(fspl,sacrificeIndex)
        maxIntTime = np.delete(maxIntTime,sacrificeIndex)
        return dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime, sacrificedStarTime

    def calcMaxCbyT(self, sInds, fspl2, maxIntTime):
        maxIntTimeCbyT = np.zeros(sInds.shape[0])
        OneFifthmaxIntTimeCbyT = np.zeros(sInds.shape[0])
        OneHalfmaxIntTimeCbyT  = np.zeros(sInds.shape[0])
        OneDayIntTimeCbyT  = np.zeros(sInds.shape[0])
        for n in range(sInds.shape[0]):
            maxIntTimeCbyT[n] = fspl2[n](maxIntTime[n])#calculate the CbyT at the maxIntTime
            OneFifthmaxIntTimeCbyT[n] = fspl2[n](maxIntTime[n]/5)#calculate the CbyT at the maxIntTime/5
            OneHalfmaxIntTimeCbyT[n] = fspl2[n](maxIntTime[n]/2)#calculate the CbyT at the maxIntTime/5
            OneDayIntTimeCbyT[n] = fspl2[n](maxIntTime[n]/5)#calculate the CbyT at the maxIntTime/5
        return maxIntTimeCbyT, OneFifthmaxIntTimeCbyT, OneHalfmaxIntTimeCbyT, OneDayIntTimeCbyT

    def calcMaxComps(self, sInds, fspl, maxIntTime):
        maxIntTimeC = np.zeros(sInds.shape[0])
        OneFifthmaxIntTimeC = np.zeros(sInds.shape[0])
        OneHalfmaxIntTimeC  = np.zeros(sInds.shape[0])
        OneDayIntTimeC  = np.zeros(sInds.shape[0])
        for n in range(sInds.shape[0]):
            maxIntTimeC[n] = fspl[n](maxIntTime[n])#calculate the CbyT at the maxIntTime
            OneFifthmaxIntTimeC[n] = fspl[n](maxIntTime[n]/5)#calculate the CbyT at the maxIntTime/5
            OneHalfmaxIntTimeC[n] = fspl[n](maxIntTime[n]/2)#calculate the CbyT at the maxIntTime/5
            OneDayIntTimeC[n] = fspl[n](maxIntTime[n]/5)#calculate the CbyT at the maxIntTime/5
        return maxIntTimeC, OneFifthmaxIntTimeC, OneHalfmaxIntTimeC, OneDayIntTimeC

    def calc_maxdMag(self, TL, sInds, fZ, fEZ, dMag, WA, mode, returnExtra=False):
        """ DEPRICATED... ORIGINALLY THIS WAS GOING TO SOLVE FOR THE MAXIMUM DMAG FOR OBSERVATION OF ANY GIVEN STAR BUT THE SOLVE FUNCTION IS TOO SLOW AS IT CURRENTLY STANDS
        Calculates electron count rates for planet signal, background noise, 
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
        lam_min = mode['lam'] - mode['deltaLam']/2.
        # get mode bandwidth (including any IFS spectral resolving power)
        deltaLam = lam/inst['Rs'] if 'spec' in inst['name'].lower() else mode['deltaLam']
        
        # if the mode wavelength is different than the wavelength at which the system 
        # is defined, we need to rescale the working angles
        if lam != syst['lam']:
            WA = WA*lam/syst['lam']
        
        # solid angle of photometric aperture, specified by core_area(optional), 
        # otherwise obtained from (lambda/D)^2
        Omega = syst['core_area'](lam,WA)*u.arcsec**2 if syst['core_area'] else \
                np.pi*(np.sqrt(2)/2*lam/self.pupilDiam*u.rad)**2
        # number of pixels in the photometric aperture = Omega / theta^2 
        Npix = (Omega/inst['pixelScale']**2).decompose().value
        
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
            if syst['core_platescale'] != None:
                core_mean_intensity *= (inst['pixelScale']/syst['core_platescale'] \
                        /(lam/self.pupilDiam)).decompose().value
            core_intensity = core_mean_intensity * Npix
        
        # get star magnitude
        sInds = np.array(sInds,ndmin=1)
        mV = TL.starMag(sInds,lam)
        
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
        C_rn = Npix*inst['sread']/inst['texp']
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

        #if returnExtra:
        #    return C_p.to('1/s'), C_b.to('1/s'), C_sp.to('1/s'), C_extra
        #else:
        #   return C_p.to('1/s'), C_b.to('1/s'), C_sp.to('1/s')
                # for characterization, Cb must include the planet
        
        if mode['detectionMode'] == False:
            C_b = C_b + C_p*mode['inst']['ENF']**2
        
        # get SNR threshold
        SNR = mode['SNR']
        # calculate integration time based on Nemati 2014
        with np.errstate(divide='ignore',invalid='ignore'):
            intTime = np.true_divide(SNR**2*C_b, (C_p**2 - (SNR*C_sp)**2))

        #Solve when numerator is zero
        from sympy.solvers import Symbol
        from sympy import Symbol
        dMag2 = Symbol('dMag2')
        C_p = C_F0*10.**(-0.4*(mV + dMag2))*core_thruput
        dMags = solve((C_p**2 - (SNR*C_sp)**2),dMag2)

        C_p = C_F0*10.**(-0.4*(mV + dMags))*core_thruput
        # calculate integration time based on Nemati 2014
        with np.errstate(divide='ignore',invalid='ignore'):
            intTime = np.true_divide(SNR**2*C_b, (C_p**2 - (SNR*C_sp)**2))

        # infinite and NAN are set to zero
        intTime[np.isinf(intTime) | np.isnan(intTime)] = 0.*u.d
        # negative values are set to zero
        intTime[intTime < 0] = 0.*u.d
        
        return intTime.to('day')
