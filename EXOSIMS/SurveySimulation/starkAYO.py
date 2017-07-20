from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import itertools
import datetime
import time
import json
import scipy
#from EXOSIMS.util.get_module import get_module
#import matlab_wrapper
import datetime
import time
#import json
from astropy.time import Time
import scipy
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from IPython.core.debugger import Tracer
import csv
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
    
       CHANGE Args:
        as (iterable 3x1):
            Cost function coefficients: slew distance, completeness, target list coverage
        
        \*\*specs:
            user specified values
    
    """

    def __init__(self, coeffs=[1,1,2], **specs):
        
        SurveySimulation.__init__(self, **specs)
        
        #verify that coefficients input is iterable 6x1
        if not(isinstance(coeffs,(list,tuple,np.ndarray))) or (len(coeffs) != 3):
            raise TypeError("coeffs must be a 3 element iterable")
        
        #normalize coefficients
        coeffs = np.array(coeffs)
        coeffs = coeffs/np.linalg.norm(coeffs)
        
        self.coeffs = coeffs
        #Do I need the stuff above???? I dont actually know what coeffs does 5/23/2017

        #self = sim#this needs to change
        #self.SimulatedUniverse = get_module(specs['modules']['SimulatedUniverse'],'SimulatedUniverse')(**specs)
        #self.Observatory = get_module(specs['modules']['Observatory'],'Observatory')(**specs)
        #self.TimeKeeping = get_module(specs['modules']['TimeKeeping'],'TimeKeeping')(**specs)

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
        self.propagTimes = np.zeros(TL.nStars)*u.d
        self.starRevisit = np.array([])
        self.starExtended = np.array([])
        self.lastDetected = np.empty((TL.nStars, 4), dtype=object)

        detMode = filter(lambda mode: mode['detectionMode'] == True, OS.observingModes)[0]
        spectroModes = filter(lambda mode: 'spec' in mode['inst']['name'], OS.observingModes)
        self.mode = detMode
        
        #Create and start Schedule
        self.schedule = np.arange(TL.nStars)#self.schedule is meant to be editable
        self.schedule_startSaved = np.arange(TL.nStars)#preserves initial list of targets
        #for myi in np.arange(int(TK.missionStart.value),int(TK.missionFinishAbs.value),1):#iterate through each 1 day
        #       tmpTime = Time(myi, format='mjd', scale='tai')
        #        kogoodStart = Obs.keepout(TL, schedule, tmpTime, OS.telescopeKeepout)#check of keepout is good
        #        schedule = schedule[np.where(kogoodStart)[0]]
        #        print('Current Number ',myi,' Max Number ',int(TK.missionFinishAbs.value))
        #schedule should now be evacuated of all stars that are in the keepout region
        #np.savetxt("schedule.csv", schedule, delimiter=",")
        sInd = None
        
        DRM = {}
        
        TK.allocate_time(Obs.settlingTime + self.mode['syst']['ohTime'])
        
        #while not TK.mission_is_over():
            # 0/ initialize arrays
        slewTimes = np.zeros(TL.nStars)*u.d#549
        fZs = np.zeros(TL.nStars)/u.arcsec**2#549
        t_dets = np.zeros(TL.nStars)*u.d
        tovisit = np.zeros(TL.nStars, dtype=bool)
        sInds = self.schedule_startSaved#np.arange(TL.nStars)#here we just use the schedule passed

        ##Removed 1 Occulter Slew Time

        startTime = TK.currentTimeAbs + slewTimes#549 create array of times the length of the number of stars (length of slew time)
            # 5/ Choose best target from remaining
            #Calculate Completeness
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
        self.starComps_startSaved = list()#contains all comps
        #self.spl_startSaved = list()
        if not os.path.isfile('/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/SurveySimulation/starComps.csv'):#the file doesn't exist Generate Completeness
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
        #self.sInds_startSaved = sInds#Saves the sInds to begin the mission. This is used when dealing directly with starComps and myTint as it is the original list of star indicies
        #WA = OS.WALim
        slewTimes2 = np.zeros(sInds.shape[0])*u.d#initialize slewTimes for each star
        startTime2 = TK.currentTimeAbs + slewTimes2#calculate slewTimes
        #fZ = ZL.fZ(Obs, TL, sInds, startTime2, self.mode['lam'])#378
        #fEZ = ZL.fEZ0
        #Tint = np.zeros((sInds.shape[0],len(dmag)))#array of #stars by dmags(225)
        ##Tint=[[0 for j in range(sInds.shape[0])] for i in range(len(dmag))]
        #for i in xrange(len(dmag)):#Tint of shape Tint[StarInd, dmag]
        #        Tint[:,i] = OS.calc_intTime(TL, sInds, fZ, fEZ, dmag[i], WA, mode).value#it is in units of days
        # 
        #myTint = list()#initialize list to store all integration times
        #for j in xrange(sInds.shape[0]):
        #        myTint.append(Tint[j,:])
        
        #update self.Tint, self.rawTint
        self.calcTint(None)#self.schedule)


        ##Remove all Tints that are 0 and save to list
        #myTint = list()#initialize list to store all integration times
        #for j in xrange(sInds.shape[0]):
        #        myTint.append(Tint[j,np.where(Tint[j,:] > 10**-10)[0]])
        #self.myTint_startSaved = myTint
        ###########################################################################

        print("Done starkAYO init")
        #END INIT##################################################################


    def run_sim(self):
        print('Starting Run Sim')#This is how I know it is running my module's code

        #Create simplifying phrases
        OS = self.OpticalSystem
        TL = self.TargetList
        SU = self.SimulatedUniverse
        Obs = self.Observatory
        TK = self.TimeKeeping
        ZL = self.ZodiacalLight
        dmag = self.dmag_startSaved
        WA = OS.WALim


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
        #aside... the starComps.csv file should be saved to the SurveySimulation folder in exosims and is necessary for expediting computations by orders of magnitude
        dir_path = os.path.dirname(os.path.realpath(__file__))#find current directory of survey Simualtion
        fname = '/starComps.csv'
        cnt = 0#this is a count for 
        while not TK.mission_is_over():#we start the mission
            TK.obsStart = TK.currentTimeNorm.to('day')
            tovisit = np.zeros(TL.nStars, dtype=bool)

            slewTimes = np.zeros(TL.nStars)*u.d
            startTime = TK.currentTimeAbs + slewTimes


            ######################################################################################
            DRM, schedule, sInd, t_det = self.nextSchedule(sInds, dmag, WA, dir_path, fname, startTime, tovisit)
            #We should have a DRM, sInd, t_det, and schedule at this moment
            assert t_det != 0, "Integration Time Cannot be 0."
    
            if sInd is not None:
                cnt += 1
                # get the index of the selected target for the extended list
                if TK.currentTimeNorm > TK.missionLife and self.starExtended.shape[0] == 0:
                    for i in range(len(self.DRM)):
                        if np.any([x == 1 for x in self.DRM[i]['plan_detected']]):
                            self.starExtended = np.hstack((self.starExtended, self.DRM[i]['star_ind']))
                            self.starExtended = np.unique(self.starExtended)

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
                if len(pInds) > 0:
                    DRM['det_fEZ'] = SU.fEZ[pInds].to('1/arcsec2').value.tolist()
                    DRM['det_dMag'] = SU.dMag[pInds].tolist()
                    DRM['det_WA'] = SU.WA[pInds].to('arcsec').value.tolist()
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
                if len(pInds) > 0:
                    DRM['char_fEZ'] = SU.fEZ[pInds].to('1/arcsec2').value.tolist()
                    DRM['char_dMag'] = SU.dMag[pInds].tolist()
                    DRM['char_WA'] = SU.WA[pInds].to('arcsec').value.tolist()
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
                #self.propagTimes[sInd] = TK.obsEnd
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

    def nextSchedule(self, sInds, dmag, WA, dir_path, fname, startTime, tovisit):
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

        #Calculate Tint at this time#####################################
        self.calcTint(None)#self.schedule)#updates self.Tint and self.rawTint
        myTint = self.Tint

        #Integration time for each star
        ##################################################################
    

        #I dont think this is necessary
        #Load Completeness From File######################################
        #The issue here is starComps is of length TL.nStars where sInds has already been filtered down based on KOGOODSTART and VISITS
        #The question is whether I do the filtering in nextSchedule or prior to nextSchedule. Either way, I need an array of length TL.nStars
        #How will I do this........ I could jut remake the array and filter again.... lets not... lets pass a list of stars containing all possible stars...
        
        #with open(dir_path+fname, 'rb') as f:#uses pathname from start of run_sim
        #    reader = csv.reader(f)
        #    your_list = list(reader)
        #    f.close()
        #self.starComps_startSaved = list()
        #for i in range(len(your_list)):
        #    tmp = np.asarray(your_list[i])
        #    tmp = tmp.astype(np.float)
        #    self.starComps_startSaved.append(tmp[np.where(myTint[i] > 10**-10)[0]])#Should filter out starComps list
        #################################################################
        
        #Calculate C vs T spline######################################
        self.splineCvsTau()
        spl = self.spl_startSaved
        starComps = self.starComps
        Tint = self.Tint
        #A Note: Technically starComps is the completeness of each star at each dmag specified in the init section of starkAYO
        #the myTint is evaluated at each one of these dmags.
        ##############################################################
    
        #Calculate C/T vs T spline####################################
        self.splineCbyTauvsTau()
        spl2 = self.spl2_startSaved
        ##############################################################

        #Calculate dC/dTau############################################
        self.splinedCbydTauvsTau()
        splDeriv = self.splDeriv_startSaved
        ##############################################################

        #Pull Untouched Star List#####################################
        sInds = self.schedule_startSaved
        ##############################################################

        #FILTER OUT STARS########################################################
        #start with self.schedule_startSaved

        #Filter KOGOODSTART..... There is a better implementation of this.....
        #We will not filter KOGOODEND using the logic that any star close enough to the keepout region of the sun will have poor C/Tau performance... This assumption should be revaluated at a future date.
        kogoodStart = Obs.keepout(TL, sInds, startTime[sInds], OS.telescopeKeepout)#outputs array where 0 values are good keepout stars
        #pdb.set_trace()
        sInds = sInds[np.where(kogoodStart)[0]]
        
        sInds1 = sInds
        #Filter out previously visited stars#######################################
        if len(sInds) > 0:
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
        #########################################################################

        #Define Intial Integration Times#########################################
        missionLength = 365.0#need to get this from .json file

        #Initial list of integration times for each star
        t_dets = np.zeros(sInds.shape[0]) + missionLength/float(sInds.shape[0]) #the list of integration times for each star
        dCbydT = np.zeros(sInds.shape[0])#sInds.shape[0])#initialize dCbydT matrix
        CbyT = np.zeros(sInds.shape[0])#sInds.shape[0])#initialize CbyT matrix
        Comp00 = np.zeros(sInds.shape[0])#sInds.shape[0])#initialize Comp00 matrix
        
        #Initialize dC/dT##########################
        for n in range(sInds.shape[0]):
            dCbydT[n] = splDeriv[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            #calculates initial completeness at initial integration times...
        ###########################################

        #Initialize C/T############################
        for n in range(sInds.shape[0]):
            CbyT[n] = spl2[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
        ###########################################

        #Initialize Comp############################
        for n in range(sInds.shape[0]):
            Comp00[n] = spl[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
        ###########################################

        numits = 0
        while numits < 10000 and sInds is not None:
            #print('numits ' + str(numits))#we impose numits to limit the total number of iterations of this loop. This may be depreicated later
            numits = numits+1#we increment numits each loop iteration

            #Update dC/dT#################
            for n in range(sInds.shape[0]):
                dCbydT[n] = splDeriv[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###############################

            #Initialize C/T############################
            for n in range(sInds.shape[0]):
                CbyT[n] = spl2[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###########################################

            #Initialize Comp############################
            for n in range(sInds.shape[0]):
                Comp00[n] = spl[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###########################################

            #Sort Descending by dC/dT###########################################################################
            #1 Find Indexes to sort by############
            sortIndex = np.argsort(dCbydT,axis=-1)[::-1]#sorts indicies and spits out list containing the indicies of the sorted list from largest to smallest. argsort sorts from smallest to largest, [::-1] flips array
            #index list from highest dCbydT to lowest dCbydT....

            #2 Reorder data#######################
            #sort star index list, integration time list, dC/dT list, splDeriv list, myTint list (contains integrationt time splines for each star)
            sInds = sInds[sortIndex]
            #if numits > 200:
            #    pdb.set_trace()
            t_dets = t_dets[sortIndex]
            dCbydT = dCbydT[sortIndex]
            splDeriv = [splDeriv[i] for i in sortIndex]
            tmp2 = list()
            tmp2[:] = [Tint[i] for i in sortIndex]#there must be a better single line way to do this...
            Tint = tmp2
            del tmp2
            spl2 = [spl2[i] for i in sortIndex]
            CbyT = CbyT[sortIndex]
            Comp00 = Comp00[sortIndex]
            
            #Sacrifice Worst Performing Star####
            #We now perform AYO selection. We add the integration time of the last star to the highest ranking dC/dT star that will not exceed the maximum integration time.
            lastTime = t_dets[-1]#retrieves last Int time from sorted sched_Tdet list

            #Identify Star Observation to Add Sacrificed Time To##############################
            nthElement = 0#starIndex integration time was added to
            for n in range(sInds.shape[0]):#iterate through all stars ranked by dC/dT until integration time can be added to a star
                #Add t_det from lastStar to First possible star
                if (t_dets[n] + lastTime) < max(Tint[n]):#we can add integration time without exceeding maximum calculated integration time value.
                    t_dets[n] = t_dets[n] + lastTime#adds sacrificed time to top of list time
                    nthElement = n#this is the star Index time was added to.
                    break#we have added the lastTime to another star integration time so we break the loop

            #Remove last Observation from schedule################################
            t_dets = t_dets[:-1]#remove last element from Tdet
            dCbydT = dCbydT[:-1]#remove last element from List of Targets
            sInds = sInds[:-1]#sacrifice last element from List of Targets
            splDeriv = splDeriv[:-1]#remove last element from dCbydT curves
            Tint = Tint[:-1]#removes laste element from Integration Times
            spl2 = spl2[:-1]
            CbyT = CbyT[:-1]
            Comp00 = Comp00[:-1]
            
            #Update dC/dT#################
            for n in range(sInds.shape[0]):
                dCbydT[n] = splDeriv[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###############################
            #THIS SHOULD BE A MORE EFFICIENT UPDATE BUT DDINT SEEM TO WORK RIGHTupdate dCbydT.... We only do this for the element with integration time added to it.
            #dCbydT[nthElement] = splDeriv[nthElement](t_dets[nthElement])

            #Initialize C/T############################
            for n in range(sInds.shape[0]):
                CbyT[n] = spl2[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###########################################

            #Initialize Comp############################
            for n in range(sInds.shape[0]):
                Comp00[n] = spl[n](t_dets[n])#dCbydT is the editable list of dCbydT for each star
            ###########################################

            #Sort Descending by dC/dT#######################################
            #1 Find Indexes to sort by############
            sortIndex = np.argsort(dCbydT,axis=-1)[::-1]#sorts indicies and spits out list containing the indicies of the sorted list from largest to smallest. argsort sorts from smallest to largest, [::-1] flips array
            #index list from highest dCbydT to lowest dCbydT....

            #2 Reorder data#######################
            #sort star index list, integration time list, dC/dT list, splDeriv list, myTint list (contains integrationt time splines for each star)
            sInds = sInds[sortIndex]
            #if numits > 200:
            #    pdb.set_trace()
            t_dets = t_dets[sortIndex]
            dCbydT = dCbydT[sortIndex]
            splDeriv = [splDeriv[i] for i in sortIndex]
            tmp2 = list()
            tmp2[:] = [Tint[i] for i in sortIndex]#there must be a better single line way to do this...
            Tint = tmp2
            del tmp2
            spl2 = [spl2[i] for i in sortIndex]
            CbyT = CbyT[sortIndex]
            Comp00 = Comp00[sortIndex]
            ###########################################################################################################


            #AYO Termination Conditions
            #1 If the nthElement that time was added to has dCbydT less than the last star (star to be taken from next)
            #2 If the nthElement is the same as the last element
            
            #if any(i < 0.05 for i in spl2[sInds](t_dets)):
            #    continue#run next iteration
            if nthElement+1 >= len(dCbydT):#if condition
                print('dCbydT maximally sorted')
                break
            if (dCbydT[nthElement+1] <= dCbydT[-1]) or all(i < 0 for i in dCbydT):#logic here is that we just added time to nth element \ if next element to add time to (nth element + 1) is less than last elements, then do not take from last element anymore
                #print(t_dets)
                #print(dCbydT)
                #print(CbyT)
                print('#####################HOW WE DID')
                #print(str(nthElement))
                #print(t_dets[0])
                #print(dCbydT[0])
                #print(CbyT[0])
                #print(Comp00[0])
                #print(str(dCbydT[nthElement]))
                #print(str(dCbydT[-1]))
                #print('numits ' + str(numits) + ' dC/dT Chosen ' + str(dCbydT[0]))#we impose numits to limit the total number of iterations of this loop. This may be depreicated later
                #print('No More Attractive Trades!')#
                #???Does this also determine if ther an insufficient number of elements left??
                break
            elif len(dCbydT) <= 1:
                print('There was a problem')
                break#there are an insufficient number of elements to test
            #There needs to be a lot more error handling here


        #Define Output of AYO Process
        sInd = sInds[0]
        t_det = t_dets[0]*u.d

        ##### Delete Some Terms    for use in next iteration        
        del splDeriv
        del spl

        #Update List of Visited stars########
        self.starVisits[sInd] += 1
        self.Completeness.visits[sInd] += 1
        #####################################

        self.schedule = sInds
        return DRM, sInds, sInd, t_det

    def choose_next_target(self, old_sInd, sInds, slewTimes):
        """Choose next target based on truncated depth first search 
        of linear cost function.
        
        Args:
            old_sInd (integer):
                Index of the previous target star
            sInds (integer array):
                Indices of available targets
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            t_dets (astropy Quantity array):
                Integration times for detection in units of day
        
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
        """Calculates integration times for all stars in sInds. If None is passed to sInds,
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

        OS = self.OpticalSystem
        WA = OS.WALim
        ZL = self.ZodiacalLight
        TL = self.TargetList
        Obs = self.Observatory
        startTime = np.zeros(len(sInds))*u.d + self.TimeKeeping.currentTimeAbs
        fZ = ZL.fZ(Obs, TL, sInds, startTime, self.mode['lam'])#378
        fEZ = ZL.fEZ0
        Tint = np.zeros((sInds.shape[0],len(self.dmag_startSaved)))#array of #stars by dmags(500)
        #Tint=[[0 for j in range(sInds.shape[0])] for i in range(len(dmag))]
        for i in xrange(len(self.dmag_startSaved)):#Tint of shape Tint[StarInd, dmag]
                Tint[:,i] = OS.calc_intTime(TL, sInds, fZ, fEZ, self.dmag_startSaved[i], WA, self.mode).value#it is in units of days
        
        newRawTint = list()#initialize list to store all integration times
        newTint = list()#initialize list to store integration times greater tha 10**-10
        for j in range(sInds.shape[0]):
                newRawTint.append(Tint[j,:])
                newTint.append(Tint[j][np.where(Tint[j] > 10**-10)[0]])

        self.rawTint = newRawTint
        self.Tint = newTint

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


