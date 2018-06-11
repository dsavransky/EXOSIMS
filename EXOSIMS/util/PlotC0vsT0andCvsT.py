"""
Purpose: To Plot C_0 vs T_0 and C_actual vs T_actual
Written by: Dean Keithly on 5/17/2018
"""
"""Example 1
I have 1000 pkl files in /home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/run146279583107.pkl and
1qty outspec file in /home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/outspec.json

To generate timelines for these run the following code from an ipython session
from ipython
%run PlotC0vsT0andCvsT.py '/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/run146279583107.pkl' \
'/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/outspec.json'
"""
"""Example 2
I have several folders with foldernames /home/dean/Documents/SIOSlab/*fZ*OB*PP*SU*/
each containing ~1000 pkl files and 1 outspec.json file

To plot a random Timeline from each folder, from ipython
%run PlotC0vsT0andCvsT.py '/home/dean/Documents/SIOSlab/' None
"""
#%run PlotC0vsT0andCvsT.py '/home/dean/Documents/SIOSlab/Dean6May18RS09CXXfZ01OB09PP01SU01.json/run95764934358.pkl' '/home/dean/Documents/SIOSlab/Dean6May18RS09CXXfZ01OB09PP01SU01.json/outspec.json'
#%run PlotC0vsT0andCvsT.py '/home/dean/Documents/SIOSlab/Dean6May18RS09CXXfZ01OB13PP01SU01/run295219944902.pkl' '/home/dean/Documents/SIOSlab/Dean6May18RS09CXXfZ01OB13PP01SU01/outspec.json'
#%run PlotC0vsT0andCvsT.py '/home/dean/Documents/SIOSlab/Dean21May18RS09CXXfZ01OB01PP01SU01/run6012655441614.pkl' '/home/dean/Documents/SIOSlab/Dean21May18RS09CXXfZ01OB01PP01SU01/outspec.json'
#%run PlotC0vsT0andCvsT.py '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/Scripts/Dean21May18RS09CXXfZ01OB01PP01SU01/run3492624809.pkl' '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/Scripts/Dean21May18RS09CXXfZ01OB01PP01SU01/outspec.json'
#%run PlotC0vsT0andCvsT.py '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/Scripts/Dean19May18RS09CXXfZ01OB56PP01SU01/run1636735874.pkl' '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/Scripts/Dean19May18RS09CXXfZ01OB56PP01SU01/outspec.json'

#%run PlotC0vsT0andCvsT.py '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/Scripts/Dean6June18RS09CXXfZ01OB56PP01SU01/run5442111239.pkl' '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/Scripts/Dean6June18RS09CXXfZ01OB56PP01SU01/outspec.json'
#%run PlotC0vsT0andCvsT.py '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/Scripts/Dean6June18RS09CXXfZ01OB01PP01SU01/run7000640433.pkl' '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/Scripts/Dean6June18RS09CXXfZ01OB01PP01SU01/outspec.json'



#%run PlotC0vsT0andCvsT.py '/home/dean/Documents/SIOSlab/Dean6June18RS09CXXfZ01OB56PP01SU01/run254150360189.pkl' '/home/dean/Documents/SIOSlab/Dean6June18RS09CXXfZ01OB56PP01SU01/outspec.json'
#Dean6June18RS09CXXfZ01OB56PP01SU01.json  run245043802546.pkl
#Dean6June18RS09CXXfZ01OB01PP01SU01.json

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
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import astropy.units as u
import copy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Mission Timeline Figures")
    parser.add_argument('pklfile', nargs=1, type=str, help='Full path to pkl file (string).')
    parser.add_argument('outspecfile', nargs=1, type=str, help='Full path to outspec file (string).')


    args = parser.parse_args()
    pklfile = args.pklfile[0]
    outspecfile = args.outspecfile[0]

    if not os.path.exists(pklfile):
        raise ValueError('%s not found'%pklfile)


    #Given Filepath for pklfile, Plot a pkl from each testrun in subdir
    pklPaths = list()
    pklfname = list()
    outspecPaths = list()
    if(os.path.isdir(pklfile)):
        #Look for all directories in specified path with structured folder name
        fp1 = pklfile
        dirs = [myString for myString in next(os.walk(fp1))[1] if 'SU' in myString \
            and 'PP' in myString \
            and 'OB' in myString \
            and 'fZ' in myString \
            and 'RS' in myString]  # Folders containing Monte Carlo Runs

        for i in np.arange(len(dirs)):
            pklFiles = [myFileName for myFileName in os.listdir(fp1+dirs[i]) if 'run' in myFileName and '.pkl' in myFileName]  # Get names of all pkl files in path
            pklfname.append(np.random.choice(pklFiles))
            pklPaths.append(fp1 + dirs[i] + '/' + pklfname[i])  # append a random pkl file to path
            outspecPaths.append(fp1 + dirs[i] + '/' + 'outspec.json')
    elif(os.path.isfile(pklfile)):
        dirs = [pklfile.split('/')[-2]]
        pklfname.append(pklfile.split('/')[-1].split('.')[0])
        pklPaths.append(pklfile)#append filepath provided in args
        outspecPaths.append(outspecfile)#append filepath provided in args       

    fig = list()#list containing all figures
    #Iterate over all pkl files
    for cnt in np.arange(len(pklPaths)):
        try:
            with open(pklPaths[cnt], 'rb') as f:#load from cache
                DRM = pickle.load(f)
        except:
            print('Failed to open pklfile %s'%pklPaths[cnt])
            pass
        try:
            with open(outspecPaths[cnt], 'rb') as g:
                outspec = json.load(g)
        except:
            print('Failed to open outspecfile %s'%outspecPaths[cnt])
            pass

        #extract mission information from DRM
        arrival_times = [DRM['DRM'][i]['arrival_time'].value for i in np.arange(len(DRM['DRM']))]
        star_inds = [DRM['DRM'][i]['star_ind'] for i in np.arange(len(DRM['DRM']))]
        sumOHTIME = outspec['settlingTime'] + outspec['starlightSuppressionSystems'][0]['ohTime']
        raw_det_time = [DRM['DRM'][i]['det_time'].value for i in np.arange(len(DRM['DRM']))]#DOES NOT INCLUDE overhead time
        det_times = [DRM['DRM'][i]['det_time'].value+sumOHTIME for i in np.arange(len(DRM['DRM']))]#includes overhead time
        det_timesROUNDED = [round(DRM['DRM'][i]['det_time'].value+sumOHTIME,1) for i in np.arange(len(DRM['DRM']))]
        ObsNums = [DRM['DRM'][i]['ObsNum'] for i in np.arange(len(DRM['DRM']))]
        y_vals = np.zeros(len(det_times)).tolist()
        char_times = [DRM['DRM'][i]['char_time'].value*(1+outspec['charMargin'])+sumOHTIME for i in np.arange(len(DRM['DRM']))]
        OBdurations = np.asarray(outspec['OBendTimes'])-np.asarray(outspec['OBstartTimes'])
        #sumOHTIME = [1 for i in np.arange(len(DRM['DRM']))]
        print(sum(det_times))
        print(sum(char_times))


        #Create Simulation Object
        sim = EXOSIMS.MissionSim.MissionSim(scriptfile=None, nopar=True, **outspec)
        SS = sim.SurveySimulation
        ZL = SS.ZodiacalLight
        COMP = SS.Completeness
        OS = SS.OpticalSystem
        Obs = SS.Observatory
        TL = SS.TargetList
        TK = SS.TimeKeeping

        close('all')
        #IF SurveySimulation module is SLSQPScheduler
        initt0 = None
        comp0 = None
        if 'SLSQPScheduler' in outspec['modules']['SurveySimulation']:
            #Extract Initial det_time and scomp0
            initt0 = sim.SurveySimulation.t0#These are the optmial times generated by SLSQP
            numObs0 = initt0[initt0.value>1e-10].shape[0]
            timeConservationCheck = numObs0*(outspec['settlingTime'] + outspec['starlightSuppressionSystems'][0]['ohTime'].value) + sum(initt0).value # This assumes a specific instrument for ohTime
            assert abs(timeConservationCheck-outspec['missionLife']*outspec['missionPortion']*365.25) < 0.1, 'total instrument time not consistent with initial calculation'
            #THIS IS JUST SUMCOMP initscomp0 = sim.SurveySimulation.scomp0

            _, Cbs, Csps = OS.Cp_Cb_Csp(TL, range(TL.nStars), ZL.fZ0, ZL.fEZ0, 25.0, SS.WAint, SS.detmode)

            #find baseline solution with dMagLim-based integration times
            #self.vprint('Finding baseline fixed-time optimal target set.')
            # t0 = OS.calc_intTime(TL, range(TL.nStars),  
            #         ZL.fZ0, ZL.fEZ0, SS.dMagint, SS.WAint, SS.detmode)
            comp0 = COMP.comp_per_intTime(initt0, TL, range(TL.nStars), 
                    ZL.fZ0, ZL.fEZ0, SS.WAint, SS.detmode, C_b=Cbs, C_sp=Csps)#Integration time at the initially calculated t0
            sumComp0 = sum(comp0)

            #Plot t0 vs c0
            fig.append(figure(cnt))
            rc('axes',linewidth=2)
            rc('lines',linewidth=2)
            rcParams['axes.linewidth']=2
            rc('font',weight='bold')
            #scatter(initt0.value, comp0, label='SLSQP $C_0$ ALL')
            plt.scatter(initt0[initt0.value > 1e-10].value, comp0[initt0.value > 1e-10], label=r'SLSQP $C_0$, $\sum C_0$' + "=%0.2f"%sumComp0, alpha=0.5, color='blue')


            #This is a calculation check to ensure the targets at less than 1e-10 d are trash
            sIndsLT1us = np.arange(TL.nStars)[initt0.value < 1e-10]
            t0LT1us = initt0[initt0.value < 1e-10].value + 0.1
            comp02 = COMP.comp_per_intTime(t0LT1us*u.d, TL, sIndsLT1us.tolist(), 
                    ZL.fZ0, ZL.fEZ0, SS.WAint[sIndsLT1us], SS.detmode, C_b=Cbs[sIndsLT1us], C_sp=Csps[sIndsLT1us])

        #calculate completeness at the time of each star observation
        slewTimes = np.zeros(len(star_inds))
        fZ = ZL.fZ(Obs, TL, star_inds, TK.missionStart + (arrival_times + slewTimes)*u.d, SS.detmode)
        comps = COMP.comp_per_intTime(raw_det_time*u.d, TL, star_inds, fZ, 
                ZL.fEZ0, SS.WAint[star_inds], SS.detmode)
        sumComps = sum(comps)


        figure(cnt)
        rc('axes',linewidth=2)
        rc('lines',linewidth=2)
        rcParams['axes.linewidth']=2
        rc('font',weight='bold')
        plt.scatter(raw_det_time, comps, label=r'SLSQP $C_{t_{Obs}}$, $\sum C_{t_{Obs}}$' + "=%0.2f"%sumComps, alpha=0.5, color='black')
        plt.xlim([0, 1.1*max(raw_det_time)])
        plt.ylim([0, 1.1*max(comps)])
        xlabel(r'Integration Time, $\tau_i$, in (days)',weight='bold')
        ylabel(r'Target Completeness, $C_i$',weight='bold')
        legend_properties = {'weight':'bold'}
        legend(prop=legend_properties)
        show(block=False)
        #Done plotting Comp vs intTime of Observations
        saveFolder = '/home/dean/Documents/SIOSlab/SPIE2018Journal/'
        savefig(saveFolder + pklfile.split('/')[-2] + 'C0vsT0andCvsT' + '.png')
        savefig(saveFolder + pklfile.split('/')[-2] + 'C0vsT0andCvsT' + '.svg')
        savefig(saveFolder + pklfile.split('/')[-2] + 'C0vsT0andCvsT' + '.eps')


        #Manually Calculate the difference to veryify all det_times are the same
        tmpdiff = np.asarray(initt0[star_inds]) - np.asarray(raw_det_time)
        print(max(tmpdiff))


        
        print -2.5*np.log10(ZL.fZ0.value) # This is 23
        print -2.5*np.log10(mean(fZ).value)