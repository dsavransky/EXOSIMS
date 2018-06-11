import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/EXOSIMS/EXOSIMS/Scripts'))
filename = 'Dean29May18RS09CXXfZ01OB54PP01SU01.json'#'./TestScripts/04_KeplerLike_Occulter_linearJScheduler.json'#'Dean13May18RS09CXXfZ01OB01PP03SU01.json'#'sS_AYO7.json'#'ICDcontents.json'###'sS_protoTimeKeeping.json'#'sS_AYO3.json'#sS_SLSQPstatic_parallel_ensembleJTWIN.json'#'sS_JTwin.json'#'sS_AYO4.json'#'sS_AYO3.json'
#filename = 'sS_intTime6_KeplerLike2.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile,nopar=True)

try:
    import cPickle as pickle
except:
    import pickle
import os
import numpy as np
from pylab import *
from numpy import nan
import matplotlib.pyplot as plt
import argparse
import json

##################
runDir = '/home/dean/Documents/SIOSlab/Dean29May18RS09CXXfZ01OB54PP01SU01/'


#Given Filepath for pklfile, Plot a pkl from each testrun in subdir
pklPaths = list()
pklfname = list()


#Look for all directories in specified path with structured folder name
dirs = runDir

pklFiles = [myFileName for myFileName in os.listdir(dirs) if 'run' in myFileName and '.pkl' in myFileName]  # Get names of all pkl files in path
for i in range(len(pklFiles)):
    pklPaths.append(dirs + pklFiles[i])  # append a random pkl file to path



#Iterate over all pkl files
alpha = list()#parallactic Angle
for cnt in np.arange(len(pklPaths)):
    try:
        with open(pklPaths[cnt], 'rb') as f:#load from cache
            DRM = pickle.load(f)
    except:
        print('Failed to open pklfile %s'%pklPaths[cnt])
        pass

    #Extract System Properties
    I = DRM['systems']['I']
    M0 = DRM['systems']['M0']
    Mp = DRM['systems']['Mp']
    O = DRM['systems']['O']
    Rp = DRM['systems']['Rp']
    a = DRM['systems']['a']
    e = DRM['systems']['e']
    mu = DRM['systems']['mu']
    p = DRM['systems']['p']
    plan2star = DRM['systems']['plan2star']
    star = DRM['systems']['star']
    w = DRM['systems']['w']
    MsTrue = DRM['systems']['MsTrue']
    MsEst = DRM['systems']['MsEst']

    #Overwrite SimulatedUniverse System Properties
    sim.SimulatedUniverse.I = I
    sim.SimulatedUniverse.M0 = M0
    sim.SimulatedUniverse.Mp = Mp
    sim.SimulatedUniverse.O = O
    sim.SimulatedUniverse.Rp = Rp
    sim.SimulatedUniverse.a = a
    sim.SimulatedUniverse.e = e
    sim.SimulatedUniverse.mu = mu
    sim.SimulatedUniverse.p = p
    sim.SimulatedUniverse.plan2star = plan2star
    #sim.SimulatedUniverse.star = star
    sim.SimulatedUniverse.w = w
    sim.TargetList.MsTrue = MsTrue
    sim.TargetList.MsEst = MsEst

    #Initialize SimulatedUniverse Properties (like r)
    sim.SimulatedUniverse.init_systems()

    #Extract Obs Where detection occurs
    iWhereDet = [i for i in range(len(DRM['DRM'])) if (DRM['DRM'][i]['det_status']==1).tolist().count(True) > 0] #list of DRM indicies where detection occurs
    #Extract arrival_times
    arrival_times = [DRM['DRM'][i]['arrival_time'] for i in iWhereDet]#each list item is the arrival_time for that detection observation
    #Extract inds
    sInds = [DRM['DRM'][i]['star_ind'] for i in iWhereDet]#each list item is the star index for that detection observation
    #Extract pInds
    pInds = [DRM['DRM'][i]['plan_inds'] for i in iWhereDet]#each list item contains the pInds for that star
    #Extract det_status
    det_status = [DRM['DRM'][i]['det_status'] for i in iWhereDet]#each list item contains the det_status for that star

    #Calculate r
    #r = list()

    for atind in range(len(arrival_times)):#atind is the index for iWhereDet, arrival_times, sInds, pInds, det_status
        if atind == 0:#For the first one, we must calculate the dt for propag_systems differently
            if arrival_times[atind] > 0:#don't [propagate if this is teh first observation]
                for ind in range(len(iWhereDet)):#Propagate all planetary systems where a detection occured
                    sim.SimulatedUniverse.propag_system(sInds[ind], arrival_times[atind])#arrival_times is dependent on detection, iterate over all sInds where detection occured
        else:#For subsequent detections, we calculate the difference to propagate by as the time when the last arrival_time occured
            for ind in range(len(iWhereDet)):#propagate all planetary systems where a detection occured
                sim.SimulatedUniverse.propag_system(sInds[ind],arrival_times[atind]-arrival_times[atind-1]) 

        #Extract r
        r = sim.SimulatedUniverse.r

        for iind in np.where(det_status[atind] == 1)[0]:#For all planets where a detection occured
            pInd = pInds[atind][iind]#extract out the planet indicie pInd

            #Extract ParallacticAngle
            alpha.append(np.arctan2(r[pInd,1],r[pInd,0]).value)


    print "%d/%d"%(cnt,len(pklPaths))#, meanNumDetsTMP, meanNumDets[cnt])

plt.close('all')
fig = figure(8000)
rc('axes',linewidth=2)
rc('lines',linewidth=2)
rcParams['axes.linewidth']=2
rc('font',weight='bold') 
hist(alpha)
ylabel("Frequency", weight='bold')
xlabel("Parallactic Angle (radians)", weight='bold')
show(block=False)

savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'ParallacticAngleHistogram' + '.png')
savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'ParallacticAngleHistogram' + '.svg')
savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'ParallacticAngleHistogram' + '.eps')

#ParallacticAngleHistogram.eps