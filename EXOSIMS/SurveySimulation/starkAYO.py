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
#from EXOSIMS.util.get_module import get_module
#import matlab_wrapper
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

class starkAYO(SurveySimulation):
    """starkAYO 
    
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
        slewTime = np.zeros(TL.nStars)*u.d#549
        fZs = np.zeros(TL.nStars)/u.arcsec**2#549
        t_dets = np.zeros(TL.nStars)*u.d
        tovisit = np.zeros(TL.nStars, dtype=bool)
        sInds = self.schedule_startSaved#the list of potential targets sInds is schedule_startSaved

        ##Removed 1 Occulter Slew Time

        startTime = TK.currentTimeAbs + slewTime#549 create array of times the length of the number of stars (length of slew time)
        
        #The below has all been taken verbatim (almost) from completeness. They are necessary to compute with f_s
        self.Completeness.f_dmagsv = np.vectorize(self.Completeness.f_dmags)
        dMagLim = self.TargetList.OpticalSystem.dMagLim
        #self.dmag = 22.5 - (np.exp(np.linspace(3.157,0.1,100))-1)#nonlinear so we can get more datapoints at higher dmaglim
        self.dmag_startSaved = np.linspace(1, 22.5, num=500,endpoint=True)
        dmag = self.dmag_startSaved
        amax = self.PlanetPopulation.arange.max().value
        emax = self.PlanetPopulation.erange.max()
        rmax = amax*(1+emax)
        SMH = self.SimulatedUniverse.Completeness#using this because I had to port over a lot of code. Equivalent to self.Completeness...
        beta = np.linspace(0.0,np.pi,1000)*u.rad
        Phis = SMH.PlanetPhysicalModel.calc_Phi(beta).value
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
        self.starComps_startSaved = list()#contains all comps
    

        #if not os.path.isfile('/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/SurveySimulation/starComps.csv'):#the file doesn't exist Generate Completeness
        if not os.path.isfile(dir_path+fname):#If this file does not exist    
            #Calculate Completeness for Each Star##############################
            print('Calculate Completeness 1')
            IWA = self.OpticalSystem.IWA.value#of telescope. to be used for min s
            OWA = self.OpticalSystem.OWA.value#of telescope to be used for max s
            starS = list()#contains separation amtrix for each star
            for q in range(len(sInds)):#iterates through each star
                star_dist = self.TargetList.dist[sInds[q]].value
                Smin = star_dist*IWA
                Smax = star_dist*OWA
                starS.append(np.linspace(Smin,Smax,400))#maybe I can use a smaller number than 400 here...
                myComp = np.zeros(len(dmag))#initialize Completeness Array
                for j in range(len(dmag)):#for each arbitrary dmaglim between 0 and dMagLim(22.5)
                        dmaglim = dmag[j]#artificial dmaglim to be set by a smaller integration time
                        fComp = range(len(starS[q]))#same lengt of smat and contains all fComp
                        for i in range(len(starS[q])):#iterate through star separations
                                fComp[i] = self.Completeness.f_s(starS[q][i],dmaglim)#calculates fComp
                        myComp[j] = sum(fComp)#sum fComp over 
                        #print('status: ' + str(j))
                myComp = np.asarray(myComp)#turn Completeness list into numpy array
                myComp[np.where(myComp>=1)]=0#sets all values greater than 1 completeness to 0 (out of feasible bounds)
                self.starComps_startSaved.append(myComp)
                #starComps.append(myComp[np.where(Tint[q,:] > 10**-10)[0]])#adds Completeness for star q to starComps. list of numpy arrays starComps[Star][list Element] filtered by Tint greater than 10**-15
                #spl.append(UnivariateSpline(myTint[q], starComps[q], k=4, s=0))#Finds star Comp vs Tint SPLINE
                timeNow = datetime.datetime.now()
                timeString = ' Day '+str(timeNow.day)+' hour '+str(timeNow.hour)+' min '+str(timeNow.minute)
                print(str(q) + ' num of ' + str(len(sInds)) + timeString)
                del myComp#delete a temporary variable
            #starComps[i][j] accessescompleteness for ith star and jth dmaglim
            print('Done Calculating Completeness')
            ################################################################
            #Save Completeness to File######################################
            with open(dir_path+fname, "wb") as fo:
                wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
                wr.writerows(self.starComps_startSaved)
                fo.close()
        else:#we will load the completeness file
            #Load Completeness From File######################################
            print('Load Completeness File')
            with open(dir_path+fname, 'rb') as f:
                reader = csv.reader(f)
                your_list = list(reader)
                f.close()
            #self.starComps_startSaved = list()
            for i in range(len(your_list)):
                tmp = np.asarray(your_list[i])
                tmp = tmp.astype(np.float)
                self.starComps_startSaved.append(tmp)#Unfiltered Completeness Lists
            #################################################################

        if len(self.starComps_startSaved) != sInds.shape[0]:#the number of stars in sInds and fromt the generated starComps are not equivalent
            print('Calculate Completeness 2')
            del self.starComps_startSaved[:]#delete all the old completeness garbage. I need new ones. ASIDE I could salvage and save some computation time if I leave star names in the lists...
            #Calculate Completeness for Each Star##############################
            IWA = self.OpticalSystem.IWA.value#of telescope. to be used for min s
            OWA = self.OpticalSystem.OWA.value#of telescope to be used for max s
            #starComps = list()#contains all comps
            starS = list()#contains separation matrix for each star
            spl = list()
            for q in range(len(sInds)):#iterates through each star
                star_dist = self.TargetList.dist[sInds[q]].value
                Smin = star_dist*IWA
                Smax = star_dist*OWA
                starS.append(np.linspace(Smin,Smax,400))#maybe I can use a smaller number than 400 here...
                myComp = np.zeros(len(dmag))#initialize Completeness Array
                for j in range(len(dmag)):#for each arbitrary dmaglim between 0 and dMagLim(22.5)
                        dmaglim = dmag[j]#artificial dmaglim to be set by a smaller integration time
                        fComp = range(len(starS[q]))#same lengt of smat and contains all fComp
                        for i in range(len(starS[q])):#iterate through star separations
                                fComp[i] = self.Completeness.f_s(starS[q][i],dmaglim)#calculates fComp
                        myComp[j] = sum(fComp)#sum fComp over 
                        #print('status: ' + str(j))
                myComp = np.asarray(myComp)#turn Completeness list into numpy array
                myComp[np.where(myComp>=1)]=0#sets all values greater than 1 completeness to 0 (out of feasible bounds)
                self.starComps_startSaved.append(myComp)

                #Print a record to screen to show how long this has been calculating. Also demonstrate that this is still running to user.
                timeNow = datetime.datetime.now()
                timeString = ' Day '+str(timeNow.day)+' hour '+str(timeNow.hour)+' min '+str(timeNow.minute)
                print(str(q) + ' num of ' + str(len(sInds)) + timeString)
                del myComp#delete a temporary variable
            #starComps[i][j] accessescompleteness for ith star and jth dmaglim
            print('Done Calculating Completeness')
            ################################################################
            #Save Completeness to File######################################
            #DO I NEED TO DELETE THE COMPLETENESS FILE too?
            try:#Here we delete the previous completeness file
                timeNow = datetime.datetime.now()
                timeString = str(timeNow.year)+'_'+str(timeNow.month)+'_'+str(timeNow.day)+'_'+str(timeNow.hour)+'_'+str(timeNow.minute)+'_'
                fnameNew = '/' + timeString +  'moved_starComps.csv'
                os.rename(dir_path+fname,dir_path+fnameNew)
            except OSError:
                pass
            with open(dir_path+fname, "wb") as fo:
                wr = csv.writer(fo, quoting=csv.QUOTE_ALL)
                wr.writerows(self.starComps_startSaved)
                fo.close()

        #Calculate myTint for an initial startimg position#########################
        slewTime2 = np.zeros(sInds.shape[0])*u.d#initialize slewTime for each star
        startTime2 = TK.currentTimeAbs + slewTime2#calculate slewTime
        
        #update self.Tint, self.rawTint
        self.calcTint(None)#Calculate Integration Time for all Stars (relatively quick process)

        print("Done starkAYO init")
        #END INIT##################################################################


    """def run_sim(self):
        print('Starting Run Sim')#This is how I know it is running my module's code

        #Create simplifying phrases
        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        ZL = self.ZodiacalLight
        dmag = self.dmag_startSaved
        WA = OS.WAint


        detMode = filter(lambda mode: mode['detectionMode'] == True, OS.observingModes)[0]#Set same as in other runsim???
        spectroModes = filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes)
        mode = self.mode#detMode
        if np.any(spectroModes):
            charMode = spectroModes[0]
        # if no spectro mode, default char mode is first observing mode
        else:
            charMode = OS.observingModes[0]


        sInds = np.arange(TL.nStars)#create list of valid target stars
        schedule = np.arange(TL.nStars)#create schedlue containing all possible target stars

        Logger.info('OB%s: survey beginning.'%(TK.OBnumber+1))
        print 'OB%s: survey beginning.'%(TK.OBnumber+1)
        sInd = None
        
        #dir_path = os.path.dirname(os.path.realpath(__file__))#find current directory of survey Simualtion
        #fname = '/starComps.csv'
        cnt = 0#this is a count for 
        while not TK.mission_is_over():#we start the mission
            TK.obsStart = TK.currentTimeNorm.to('day')
            tovisit = np.zeros(TL.nStars, dtype=bool)

            slewTime = np.zeros(TL.nStars)*u.d
            startTime = TK.currentTimeAbs + slewTime


            ######################################################################################
            Tstart = timeit.timeit()
            DRM, schedule, sInd, t_det = self.nextSchedule(sInds, dmag, mode, WA, startTime, tovisit)# dir_path, fname
            Tfin = timeit.timeit()
            print('Time1 is ' + str(Tfin-Tstart))
            #We should have a DRM, sInd, t_det, and schedule at this moment
            assert t_det != 0, "Integration Time Cannot be 0."
    
            if sInd is not None:
                cnt += 1
                # get the index of the selected target for the extended list
                if TK.currentTimeNorm > TK.missionLife and self.starExtended.shape[0] == 0:
                    for i in range(len(self.DRM)):
                        try:
                            if np.any([x == 1 for x in self.DRM[i]['plan_detected']]):
                                self.starExtended = np.hstack((self.starExtended, self.DRM[i]['star_ind']))
                                self.starExtended = np.unique(self.starExtended)
                        except:
                            print('i is ' + str(i))
                            print(DRM[i])
                            print('*****************' + str(i))

                # Beginning of observation, start to populate DRM
                DRM['OB'] = TK.OBnumber
                DRM['star_ind'] = sInd
                DRM['arrival_time'] = TK.currentTimeNorm.to('day').value
                pInds = np.where(SU.plan2star == sInd)[0]
                DRM['plan_inds'] = pInds.astype(int).tolist()
                Logger.info('  Observation #%s, target #%s/%s with %s planet(s), mission time: %s'\
                        %(cnt, sInd, TL.nStars, len(pInds), TK.obsStart.round(2)))
                print '  Observation #%s, target #%s/%s with %s planet(s), mission time: %s'\
                        %(cnt, sInd, TL.nStars, len(pInds), TK.obsStart.round(2))
                #print('TACO')
                # PERFORM DETECTION and populate revisit list attribute.
                # First store fEZ, dMag, WA
                if np.any(pInds):
                    DRM['det_fEZ'] = SU.fEZ[pInds].to('1/arcsec2').value.tolist()
                    DRM['det_dMag'] = SU.dMag[pInds].tolist()
                    DRM['det_WA'] = SU.WA[pInds].to('mas').value.tolist()
                detected, detSNR, FA = self.observation_detection(sInd, t_det, detMode)
                # Update the occulter wet mass
                if OS.haveOcculter == True:
                    DRM = self.update_occulter_mass(DRM, sInd, t_det, 'det')
                # Populate the DRM with detection results
                DRM['det_time'] = t_det.to('day').value
                DRM['det_status'] = detected
                DRM['det_SNR'] = detSNR

                # PERFORM CHARACTERIZATION and populate spectra list attribute.
                # First store fEZ, dMag, WA, and characterization mode
                if np.any(pInds):
                    DRM['char_fEZ'] = SU.fEZ[pInds].to('1/arcsec2').value.tolist()
                    DRM['char_dMag'] = SU.dMag[pInds].tolist()
                    DRM['char_WA'] = SU.WA[pInds].to('mas').value.tolist()
                DRM['char_mode'] = dict(charMode)
                del DRM['char_mode']['inst'], DRM['char_mode']['syst']
                characterized, charSNR, t_char = self.observation_characterization(sInd, charMode)
                assert t_char !=0, "Integration time can't be 0."
                # Update the occulter wet mass
                if OS.haveOcculter == True and t_char is not None:
                    DRM = self.update_occulter_mass(DRM, sInd, t_char, 'char')
                # if any false alarm, store its characterization status, fEZ, dMag, and WA
                if FA == True:
                    DRM['FA_status'] = characterized.pop()
                    DRM['FA_SNR'] = charSNR.pop()
                    DRM['FA_fEZ'] = self.lastDetected[sInd,1][-1]
                    DRM['FA_dMag'] = self.lastDetected[sInd,2][-1]
                    DRM['FA_WA'] = self.lastDetected[sInd,3][-1]
                # Populate the DRM with characterization results
                DRM['char_time'] = t_char.to('day').value if t_char else 0.
                DRM['char_status'] = characterized
                DRM['char_SNR'] = charSNR

                # Append result values to self.DRM
                self.DRM.append(DRM)

                # Calculate observation end time, and update target time
                #print('tmp3')
                #TK.obsEnd = TK.currentTimeNorm.to('day')
                #self.starTimes[sInd] = TK.obsEnd
                #print('tmp4')
                ## With prototype TimeKeeping, if no OB duration was specified, advance
                ## to the next OB with timestep equivalent to time spent on one target
                #if np.isinf(TK.OBduration):
                #    obsLength = (TK.obsEnd-TK.obsStart).to('day')
                #    TK.next_observing_block(dt=obsLength)
                #print('tmp5')
                # With occulter, if spacecraft fuel is depleted, exit loop
                if OS.haveOcculter and Obs.scMass < Obs.dryMass:
                    print 'Total fuel mass exceeded at %r' %TK.currentTimeNorm
                    break

        mission_end = "Simulation finishing OK. Results stored in SurveySimulation.DRM"
        Logger.info(mission_end)
        print mission_end
        return mission_end
        """

    def next_target(self, sInds, mode):
        start_time = timeit.default_timer()
        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        ZL = self.ZodiacalLight
        dmag = self.dmag_startSaved
        WA = OS.WAint
        slewTime = np.zeros(TL.nStars)*u.d
        startTime = TK.currentTimeAbs+slewTime
        tovisit = np.zeros(TL.nStars, dtype=bool)
        #def nextSchedule(self, sInds, mode, dmag, WA, startTime, tovisit):# dir_path, fname,
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
        lastTime = start_time
        print(str(timeit.default_timer()-lastTime))
        #Calculate Tint at this time#####################################
        self.calcTint(None)#self.schedule)#updates self.Tint and self.rawTint
        myTint = self.Tint
        #Aug 28, 2017 execution time 3.27
        print('calcTint time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        

        #Integration time for each star
        ##################################################################
    
        #Calculate C vs T spline######################################
        self.splineCvsTau()
        spl = self.spl_startSaved
        starComps = self.starComps
        Tint = self.Tint
        #Aug 28, 2017 execution time 3.138
        print('splineCvsTau time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        #A Note: Technically starComps is the completeness of each star at each dmag specified in the init section of starkAYO
        #the myTint is evaluated at each one of these dmags.
        ##############################################################
    
        #Calculate C/T vs T spline####################################
        self.splineCbyTauvsTau()
        spl2 = self.spl2_startSaved
        #Aug 28, 2017 execution time 0.0909
        print('splineCbyTauvsTau time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        ##############################################################

        #Calculate dC/dTau############################################
        self.splinedCbydTauvsTau()
        splDeriv = self.splDeriv_startSaved
        #Aug 28, 2017 execution time 0.024
        print('splinedCbydTauvsTau time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        ##############################################################

        #Pull Untouched Star List#####################################
        sInds = self.schedule_startSaved
        ##############################################################

        #FILTER OUT STARS########################################################
        #start with self.schedule_startSaved

        #Filter KOGOODSTART..... There is a better implementation of this.....
        #We will not filter KOGOODEND using the logic that any star close enough to the keepout region of the sun will have poor C/Tau performance... This assumption should be revaluated at a future date.
        kogoodStart = Obs.keepout(TL, sInds, startTime[sInds], mode)#outputs array where 0 values are good keepout stars
        #pdb.set_trace()
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
            #pdb.set_trace()
            sInds = np.where(tovisit)[0]
        print('NumStars Before ' + str(len(sInds1)) + ' After Filter ' + str(len(sInds)))
        #pdb.set_trace()
        #Aug 28, 2017 execution time 0.235
        print('KOGOODSTART Filter time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        #########################################################################

        #Define Intial Integration Times#########################################
        missionLength = 365.0#need to get this from .json file

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
        fsplDeriv = [splDeriv[i] for i in sInds]
        fspl = [spl[i] for i in sInds]
        fstarComps = [starComps[i] for i in sInds]
        fTint = [Tint[i] for i in sInds]
        fspl2 = [spl2[i] for i in sInds]

        #Initialize dC/dT##########################
        for n in range(sInds.shape[0]):
            dCbydT[n] = fsplDeriv[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            #calculates initial completeness at initial integration times...
        ###########################################

        #Initialize C/T############################
        for n in range(sInds.shape[0]):
            CbyT[n] = fspl2[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
        ###########################################

        #Initialize Comp############################
        for n in range(sInds.shape[0]):
            Comp00[n] = fspl[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
        ###########################################
        #Aug 28, 2017 execution time 0.020
        print('Initialization Stuffs time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()


        #Update maxIntTime#########################
        #Calculate the Maximum Integration Time and dmag for each star
        maxTint, maxdmag = self.calcMaxTint(sInds)
        self.maxTint = maxTint
        self.maxdmag = maxdmag
        assert maxTint[0] != 0#i think unnecessary at this point
        maxIntTime = maxTint
        #Aug 28, 2017 execution time 3.213
        print('calcMaxTint time = '+str(timeit.default_timer() - lastTime))
        lastTime = timeit.default_timer()
        ###########################################


        numits = 0
        lastIterationSumComp  = 0
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

            #2 Reorder data#######################
            dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime= self.reOrder(dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime)
            #Aug 28, 2017 execution time 0.00039
            #print('Reorder time = '+str(timeit.default_timer() - lastTime))
            lastTime = timeit.default_timer()

            #3 Sacrifice Worst Performing Star####
            #We now perform AYO selection. We add the integration time of the last star to the highest ranking dC/dT star that will not exceed the maximum integration time.
            lastTime = t_dets[-1]#retrieves last Int time from sorted sched_Tdet list
            
            #4 Remove last Observation from schedule################################
            t_dets = t_dets[:-1]#remove last element from Tdet
            dCbydT = dCbydT[:-1]#remove last element from List of Targets
            sInds = sInds[:-1]#sacrifice last element from List of Targets
            fsplDeriv = fsplDeriv[:-1]#remove last element from dCbydT curves
            Tint = Tint[:-1]#removes laste element from Integration Times
            fspl2 = fspl2[:-1]
            CbyT = CbyT[:-1]
            Comp00 = Comp00[:-1]
            fspl = fspl[:-1]
            maxIntTime = maxIntTime[:-1]
            #Aug 28, 2017 execution time ????infinitesimally small
            #print('sacrifice time = '+str(timeit.default_timer() - lastTime))
            lastTime = timeit.default_timer()

            #5 Identify Star Observation to Add Sacrificed Time To##############################
            #nthElement = 0#starIndex integration time was added to
            #dt = 0.01#in days
            #for n in range(sInds.shape[0]):#iterate through all stars ranked by dC/dT until integration time (dt) can be added to a star
            #    #Add t_det from lastStar to First possible star
            #    if (t_dets[n] + lastTime) < max(Tint[n]):#we can add integration time without exceeding maximum calculated integration time value.
            #        t_dets[n] = t_dets[n] + lastTime#adds sacrificed time to top of list time
            #        nthElement = n#this is the star Index time was added to.
            #        break#we have added the lastTime to another star integration time so we break the loop
            
            #calculate Maximum Integration Times
            #tmp6 = Obs.orbit(TK.currentTimeAbs)
            #r_sc = np.repeat(tmp6,len(sInds),axis=0)
            #fZ = ZL.fZ(TL, sInds, mode['lam'],r_sc)
            #fEZ = ZL.fEZ0
            #WA = self.OpticalSystem.WAint
            #dMagLim = self.TargetList.OpticalSystem.dMagLim
            #print(dMagLim)#this is a constant 22.5
            #print(WA)
            #print(mode)
            #C_p, C_b, C_sp = self.OpticalSystem.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dMagLim, WA, mode)
            #print('C_p')
            #print(C_p)
            #print('C_b')
            #print(C_b)
            #print('C_sp')
            #print(C_sp)
            #print(mode['detectionMode'])
            #SNR = mode['SNR']
            #intTime = np.true_divide(SNR**2*C_b, (C_p**2 - (SNR*C_sp)**2))
            #print(intTime)

        
            dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl = self.distributedt(lastTime, dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxTint)
            #Aug 28, 2017 execution time 9e-6
            #print('distributedt time = '+str(timeit.default_timer() - lastTime))
            lastTime = timeit.default_timer()

                #1 Calculate dC/dT of star n+1.
                #2 Solve for Tint that makes dC/dT of star n equal star n+1
                #3 Calculate dC/dT of star n+2
                #4 Solve for Tint that makes dC/dT of star n = star n+2 and when star n+1 = star n+2
                #5 If a Tint is greater than MaxTint for that star, add sufficient time to make that star at Max Tint and continue with the distribution as normal
                #6 If any addition of Tint makes the total Tint added greater than the sacrificed time, then the time needs to be distributed carefully... i don't know what this means yet.
                #6 cont. I can't just divided the remaining time and add them to each star observation because then they would be unequal.
            
            #6 Update Lists #########################################################3
            #Update dC/dT#################
            for n in range(sInds.shape[0]):
                dCbydT[n] = fsplDeriv[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###############################
            #THIS SHOULD BE A MORE EFFICIENT UPDATE BUT DDINT SEEM TO WORK RIGHTupdate dCbydT.... We only do this for the element with integration time added to it.
            #dCbydT[nthElement] = splDeriv[nthElement](t_dets[nthElement])

            #Update C/T############################
            for n in range(sInds.shape[0]):
                CbyT[n] = fspl2[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###########################################

            #Update Comp############################
            for n in range(sInds.shape[0]):
                Comp00[n] = fspl[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###########################################

            #Sort Descending by dC/dT#######################################
            dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime= self.reOrder(dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime)
            # #1 Find Indexes to sort by############
            # sortIndex = np.argsort(dCbydT,axis=-1)[::-1]#sorts indicies and spits out list containing the indicies of the sorted list from largest to smallest. argsort sorts from smallest to largest, [::-1] flips array
            # #index list from highest dCbydT to lowest dCbydT....


            #AYO Termination Conditions
            #1 If the nthElement that time was added to has dCbydT less than the last star (star to be taken from next)
            #2 If the nthElement is the same as the last element
            
            #if any(i < 0.05 for i in spl2[sInds](t_dets)):
            #    continue#run next iteration
            if 1 >= len(dCbydT):#if this is the last ement in the list
                print('dCbydT maximally sorted')
                break
            #print(saltyburrito)
            #If the total sum of completeness at this moment is less than the last sum, then exit
            if(sum(Comp00) < lastIterationSumComp and (len(sInds) < 20)):
                print('Successfully Sorted List!!')
                print(len(sInds))
                #Define Output of AYO Process
                sInd = sInds[0]
                t_det = t_dets[0]*u.d
                #Update List of Visited stars########
                self.starVisits[sInd] += 1
                self.schedule = sInds
                return DRM, sInd, t_det #sInds
            else:#else set lastIterationSumComp to current sum Comp00
                lastIterationSumComp = sum(Comp00)
                print('SumComp '+str(lastIterationSumComp) + ' with sInds left '+str(len(sInds)))

            # if (dCbydT[nthElement+1] <= dCbydT[-1]) or all(i < 0 for i in dCbydT):#logic here is that we just added time to nth element \ if next element to add time to (nth element + 1) is less than last elements, then do not take from last element anymore
            #     #print(t_dets)
            #     #print(dCbydT)
            #     #print(CbyT)
            #     print('#####################HOW WE DID')
            #     #print(str(nthElement))
            #     #print(t_dets[0])
            #     #print(dCbydT[0])
            #     #print(CbyT[0])
            #     #print(Comp00[0])
            #     #print(str(dCbydT[nthElement]))
            #     #print(str(dCbydT[-1]))
            #     #print('numits ' + str(numits) + ' dC/dT Chosen ' + str(dCbydT[0]))#we impose numits to limit the total number of iterations of this loop. This may be depreicated later
            #     #print('No More Attractive Trades!')#
            #     #???Does this also determine if ther an insufficient number of elements left??
            #     break
            # elif len(dCbydT) <= 1:
            #     print('There was a problem')
            #     break#there are an insufficient number of elements to test
            # #There needs to be a lot more error handling here
            #print(numits)

        ##Define Output of AYO Process
        #sInd = sInds[0]
        #t_det = t_dets[0]*u.d

        ##### Delete Some Terms    for use in next iteration        
        del splDeriv
        del spl

        #Update List of Visited stars########
        self.starVisits[sInd] += 1
        #self.Completeness.visits[sInd] += 1
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
        # if sInds is None:
        #     sInds = self.schedule_startSaved
        # #else:
        # #    sInds = sInds

        # OS = self.OpticalSystem
        # WA = OS.WAint
        # ZL = self.ZodiacalLight
        # TL = self.TargetList
        # Obs = self.Observatory
        # startTime = np.zeros(len(sInds))*u.d + self.TimeKeeping.currentTimeAbs
        # r_sc_list = Obs.orbit(startTime)#list of orbital positions the length of startTime
        # fZ = ZL.fZ(TL, sInds, self.mode['lam'], r_sc_list)#378
        # fEZ = ZL.fEZ0
        # Tint = np.zeros((sInds.shape[0],len(self.dmag_startSaved)))#array of #stars by dmags(500)
        # #Tint=[[0 for j in range(sInds.shape[0])] for i in range(len(dmag))]
        # for i in xrange(len(self.dmag_startSaved)):#Tint of shape Tint[StarInd, dmag]
        #         Tint[:,i] = OS.calc_intTime(TL, sInds, fZ, fEZ, self.dmag_startSaved[i], WA, self.mode).value#it is in units of days
        
        # newRawTint = list()#initialize list to store all integration times
        # newTint = list()#initialize list to store integration times greater tha 10**-10
        # for j in range(sInds.shape[0]):
        #         newRawTint.append(Tint[j,:])
        #         newTint.append(Tint[j][np.where(Tint[j] > 10**-10)[0]])
        dmag = self.dmag_startSaved
        newTint, newRawTint = self.calcTint_core(sInds,dmag)
        self.rawTint = newRawTint
        self.Tint = newTint

    def calcMaxTint(self, sInds):
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
        maxTint = np.zeros(len(Tint[:]))#declare array
        maxdmag = np.zeros(len(Tint[:]))#declare array
        for i in range(0, len(Tint[:])):
            maxTint[i] = max(Tint[i][:])
            occur = [k for k, j in enumerate(Tint[i][:]) if j == maxTint[i]]
            maxdmag[i] = self.dmag_startSaved[occur[0]]

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
        WA = OS.WAint
        ZL = self.ZodiacalLight
        TL = self.TargetList
        Obs = self.Observatory
        startTime = np.zeros(len(sInds))*u.d + self.TimeKeeping.currentTimeAbs
        r_sc_list = Obs.orbit(startTime)#list of orbital positions the length of startTime
        fZ = ZL.fZ(TL, sInds, self.mode['lam'], r_sc_list)#378
        fEZ = ZL.fEZ0
        Tint = np.zeros((sInds.shape[0],len(dmag)))#array of #stars by dmags(500)
        #Tint=[[0 for j in range(sInds.shape[0])] for i in range(len(dmag))]
        for i in xrange(len(dmag)):#Tint of shape Tint[StarInd, dmag]
                Tint[:,i] = OS.calc_intTime(TL, sInds, fZ, fEZ, dmag[i], WA, self.mode).value#it is in units of days
        
        newRawTint = list()#initialize list to store all integration times
        newTint = list()#initialize list to store integration times greater tha 10**-10
        for j in range(sInds.shape[0]):
                newRawTint.append(Tint[j,:])
                newTint.append(Tint[j][np.where(Tint[j] > 10**-10)[0]])

        #self.rawTint = newRawTint
        #self.Tint = newTint
        return newTint, newRawTint

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
            if starComps[j][-1] < 10**-10:#If the last value of starComps is a 0. We assume this means there exists a discontinuity in completeness
                #find How many Indicies are 0
                for k in range(len(starComps[j]))[::-1]:#Iterate through completeness from last index to first
                    #find first index of value greater than 10 **-10
                    if starComps[j][k] > 10**-10:
                        index = k
                        break
            if index is not None:
                starComps[j] = starComps[j][0:k]
                Tint[j] = Tint[j][0:k]
                index = None
            else:#if index is None
                index = None
        ##########################################################################################

        spl = list()
        for q in range(len(schedule)):
            spl.append(UnivariateSpline(Tint[q], starComps[q], k=4, s=0))#Finds star Comp vs Tint SPLINE
        self.spl_startSaved = spl#updates self.spl
        self.starComps = starComps#updates self.starComps
        self.Tint = Tint#update self.Tint

    def splineCbyTauvsTau(self):
        """Calculate the spline fit to Completeness/Tau vs Tau

        Args:

        Returns:
            updates self.spl2_startSaved (spl2 is the spline fitting C/T vs T)
            Dimensions are spl2_startSaved[nStars][Tint > 10**-10]
        """

        #self.calcTint(self.schedule)#,self.mode,self.startTime2)
        rawTint = self.rawTint
        Tint = self.Tint
        sInds = self.schedule_startSaved

        starComps = list()
        for j in xrange(sInds.shape[0]):
                starComps.append(self.starComps_startSaved[j][np.where(rawTint[j] > 10**-10)[0]])

        spl = self.spl_startSaved
        sInds = self.schedule_startSaved
        spl2 = list()
        for x in range(len(sInds)):
            spl2.append(UnivariateSpline(Tint[x],spl[x](Tint[x])/Tint[x], k=4, s=0))
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
        spl = self.spl
        Tint = self.Tint
        TL = self.TargetList

        for m in xrange(len(Tint)):
            dtos = 24*60*60
            #PLOT COMPLETENESS VS INT TIME
            fig = plt.figure()
            plt.plot(Tint[m]*dtos,starComps[m],'o',Tint[m]*dtos,spl[m](Tint[m]),'-')
            plt.xscale('log')
            axes = plt.gca()
            axes.set_xlim([10e-4,10e6])
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

    def plotCompbyTauvsTau(self):
        """Plots Completeness/Integration Time vs Integration Time for every star
        """
        #Update splines
        print('Start Plot C/T vs T')
        self.splineCvsTau()
        self.splineCbyTauvsTau()
        spl2 = self.spl2
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
        splDeriv = self.splDeriv
        Tint = self.Tint#myTint_startSaved
        TL = self.TargetList
        for m in xrange(len(Tint)):
            dtos = 24*60*60
            #PLOT COMPLETENESS VS INT TIME
            fig = plt.figure()
            plt.plot(Tint[m]*dtos,splDeriv[m](Tint[m]),'-')
            plt.xscale('log')
            axes = plt.gca()
            axes.set_xlim([10e-4,10e6])
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

    def distributedt(self,lastTime, dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime):#distributing the sacrificed time
        """#here we want to add integration time to the highest performing stars in our selection box
        #returns nTint for additional integration time added to each star.
        Args:
            maxIntTime is the list of approximate maximum integration times

        """
        dt = 0.001#this is the maximum quantity of time to distribute at a time.
        #timeDistributed = 0#this is the total amount of time distributed so far.
        #while(timeDistributed < lastTime):#while the time distributed is less than the total time to be distributed

            #Check Conservation of Time. Ensure Time is conserved.
            #if((lastTime-timeDistributed) < dt):#if the time left to be distributed is less than standard dt
            #    dt = lastTime-timeDistributed#make dt equal to the leftover time

        #Now decide where to put dt
        #Find next star that is not at or above maxIntTime
        tmp = 0
        for i in xrange(len(maxIntTime)):#iterate through each star from top to bottom
            if(t_dets[i] >= maxIntTime[i]):
                dt=dt#DO NOTHING
            elif(t_dets[i] < maxIntTime[i]):#Then we will add time to this star.
                #Check that added time will not Exceed maxIntTime
                if(t_dets[i]+dt <= maxIntTime[i]):
                    t_dets[i] = t_dets[i]+dt
                    dt = 0
                else:#(t_dets[i]+dt > maxIntTime[i]):#added dt would be greater than maxIntTime
                    #Add the most time you can to the first star
                    dt = dt - (maxIntTime[i]-t_dets[i])
                    t_dets[i] = maxIntTime[i]

            if(dt == 0):#if there is no more time to distribute from the sacrificed star
                break#we break out of for loop


                    #tmp = t_dets[i] - maxIntTime[i]
                    #t_dets[1] = t_dets[1] + tmp#the addition of time to t_dets[2] is unchecked
                    #t_dets[0] = maxIntTime[0]  
        #End for Loop

            #t_dets[0] = t_dets[0] + dt#add dt to the first star
            #timeDistriubted = timeDistributed + dt



            #there is a problem in here. I think I need to check
            #if(t_dets[0] > maxIntTime[0]):
            #    tmp = t_dets[0] - maxIntTime[0]
            #    t_dets[1] = t_dets[1] + tmp#the addition of time to t_dets[2] is unchecked
            #    t_dets[0] = maxIntTime[0]
            #print('In distributedt'+)

            #ADD MAXINTTIME TO REORDER OTHERWISE THERE WILL BE A DATA MISSMATCH!!!!!!
            #reorder lists
            dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime = self.reOrder(dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime)

            #Print top 10 targets


        #for i in sInds:
        #    if((maxIntTime[i] - Tint[i])  > 0.001):
        #        ObsOvermaxTint[i] = i
        #    elif():

        #return list of values not at maxTint and not at the same value as the first observation not at maxTint.

        #k is the index of the star with completeness performing lower than the group.
        #j is the index of each star a part of the higher performing group.
        #we will iterate over each star in the higher performing group to calculate the integration time of each observation
        # for j in sInds:#iterate through highest performing stars Note: k=max(hpStars)+1
        #     #where hpStars are the indicies of the highest performing stars
        #     intTimeFunc = lambda tau: fsplDeriv[k](t_dets[k])-fsplDeriv[j](tau)#function setting dC/dT[star1]=dC/dT[star2]
        #     nTint[j] = fsolve(intTimeFunc,t_det[j])#solves for integration time of Star 1 that would make the two dC/dT's equal
        #     if():
        #         break
        # #Aha!now I have the integration times for all the stars that would make each observation have a dC/dT equal to the next star
        
        # #check if the total integration time to achieve this would be less than the integration time we have to redistribute
        # if sum(nTint) > lastTime:
        #     #simple fix and not sophisticated
        #     norm = np.linalg.norm(nTint)
        #     nTint = nTint*lastTime/norm
        #     #just assume everything is linear and redistribute time across all in these ratios

        # #check if the integration time of any star is greater than maxTint
        # maxTint = maxIntTime
        # dt=0
        # while any(nTint - maxTint > 0):


        #     #find the index of occurence
        #     tmp = nTint-maxTint
        #     i = tmp.index(max(tmp))#index of occurence. Note I could be multiple values
        #     dt = nTint[i[0]] - maxTint[i[0]]#calculate differential available
        #     nTint[i[0]] = maxTint[i[0]]#set nTint to maxTint
        #     if not any(x for x in range(nTint) if maxTint[x] - nTint[x] > 0.00001):#if every star is at maxTint, then we will pass dt to the next star
        #         #if all items in list are at maxTint
        #         #we will pass dt as an output for appending to the next star
        #         break#we don't need to keep doing this whole looping thing
        #     else:
        #         tmp = nTint-maxTint#recalculate tmp
        #         indicies = tmp.index(tmp<0)#find indicies of tmp less than 0
        #         tmpindex = tmp.index(max(tmp[indicies]))#find the tmp index where values closest to maxTint occurs
        #         nTint[tmpindex] = nTint[tmpindex] + dt#add to the index where the value closes to maxTint occurs
        #         dt = 0#set dt back to zero because we redistributed it
            


        #     nTint[tmp.index(min(tmp[indicies]))]#find minimum value in tmp to add to
        #     j2 = [i for i in j if i >= 5]

        #     #give to min but also in list greater than 0
        # #for i in nTint:
        #     #check if any obs times exceed maxTint
        #     if nTint[i] > maxTint[i]:
        #         #calculate differential
        #         dt = nTint[i]-maxTint[i]
        #         #set nTint equal to maxTint
        #         nTint[i] = maxTint[i]

        #     #now redistribute dt
        #     if not any(x for x in range(nTint) if maxTint[x] - nTint[x] > 0.00001):#if every star is at maxTint, then we will pass dt to the next star
        #         #if all items in list are at maxTint
        #         #we will pass dt as an output for appending to the next star
        #         break#we don't need to keep doing this whole looping thing
        #     else:#there is one star that is not at Tint
        #         tmp = maxTint-nTint#calculate the difference
        #         maxTint[tmp.index(max(tmp))] = maxTint[tmp.index(max(tmp))] + dt#add dt to the nTint with the maximum space available
        #         #aside: assigning to maxTint isn't right, we probably want to distribute it to a Tint that has the shortest room to reach maximum...
        #         #we can't just use min because that will give us our 0 value... we need the index with smallest value greater than approx 0.

        return dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl

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
        return dCbydT, sInds, t_dets, fsplDeriv, Tint, fspl2, CbyT, Comp00, fspl, maxIntTime

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
