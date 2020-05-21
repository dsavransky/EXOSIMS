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
import os, inspect
if not 'DISPLAY' in os.environ.keys(): #Check environment for keys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
else:
    import matplotlib.pyplot as plt 
import numpy as np
from numpy import nan
import argparse
import json
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import astropy.units as u
import copy
import random
import datetime
import re
from EXOSIMS.util.vprint import vprint
from scipy.optimize import minimize,minimize_scalar
from matplotlib.ticker import NullFormatter, MaxNLocator
import matplotlib.gridspec as gridspec
from EXOSIMS.util.get_dirs import get_cache_dir
try:
    import urllib2
except:
    import urllib3
#from EXOSIMS.SurveySimulation import array_encoder

class plotC0vsT0andCvsT(object):
    """Designed to plot Planned Completeness and Observed Completeness
    """
    _modtype = 'util'

    def __init__(self, args=None):
        """
        Args:
            args (dict) - 'file' keyword specifies specific pkl file to use
        """
        self.args = args
        pass

    def singleRunPostProcessing(self, PPoutpath, folder):
        """Generates a single yield histogram for the run_type
        Args:
            PPoutpath (string) - output path to place data in
            folder (string) - full filepath to folder containing runs
        """
        if not os.path.exists(folder):#Folder must exist
            raise ValueError('%s not found'%folder)
        if not os.path.exists(PPoutpath):#PPoutpath must exist
            raise ValueError('%s not found'%PPoutpath) 
        outspecfile = os.path.join(folder,'outspec.json')
        if not os.path.exists(outspecfile):#outspec file not found
            raise ValueError('%s not found'%outspecfile) 

        #Get name of pkl file
        if isinstance(self.args,dict):
            if 'file' in self.args.keys():
                file = self.args['file']
            else:
                file = self.pickPKL(folder)
        else:
            file = self.pickPKL(folder)
        fullPathPKL = os.path.join(folder,file) # create full file path
        if not os.path.exists(fullPathPKL):
            raise ValueError('%s not found'%fullPathPKL)

        #Load pkl and outspec files
        try:
            with open(fullPathPKL, 'rb') as f:#load from cache
                DRM = pickle.load(f, encoding='latin1')
        except:
            vprint('Failed to open fullPathPKL %s'%fullPathPKL)
            pass
        outspecPath = os.path.join(folder,'outspec.json')
        try:
            with open(outspecPath, 'rb') as g:
                outspec = json.load(g)
        except:
            vprint('Failed to open outspecfile %s'%outspecPath)
            pass

        #Create Simulation Object
        sim = EXOSIMS.MissionSim.MissionSim(scriptfile=None, nopar=True, **outspec)
        SS = sim.SurveySimulation
        ZL = SS.ZodiacalLight
        COMP = SS.Completeness
        OS = SS.OpticalSystem
        Obs = SS.Observatory
        TL = SS.TargetList
        TK = SS.TimeKeeping

        plt.close('all')
        plt.figure(2, figsize=(8.5,6))
        gs = gridspec.GridSpec(2,2, width_ratios=[6,1], height_ratios=[1,4])#DELETE ,0.3,6,1.25
        gs.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 

        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rcParams['axes.linewidth']=2
        plt.rc('font',weight='bold')

        #What the plot layout looks like
        ###---------------
        # | gs[0]  gs[1] |
        # | gs[2]  gs[3] |
        ###---------------
        ax0 = plt.subplot(gs[0])#1D histogram of intTimes
        ax1 = plt.subplot(gs[1])#BLANK
        ax2 = plt.subplot(gs[2])#CvsT lines
        ax3 = plt.subplot(gs[3])#1D histogram of Completeness

        ax1 = plt.subplot(gs[1])#BLANK
        ax1.xaxis.set_visible(False)
        ax1.yaxis.set_visible(False)


        #IF SurveySimulation module is SLSQPScheduler
        initt0 = None
        comp0 = None
        numObs0 = 0
        fZ = ZL.fZ0
        if 'SLSQPScheduler' in outspec['modules']['SurveySimulation']:
            #Extract Initial det_time and scomp0
            initt0 = sim.SurveySimulation.t0#These are the optmial times generated by SLSQP
            numObs0 = initt0[initt0.value>1e-10].shape[0]
            timeConservationCheck = numObs0*(outspec['settlingTime'] + outspec['starlightSuppressionSystems'][0]['ohTime'].value) + sum(initt0).value # This assumes a specific instrument for ohTime
            #assert abs(timeConservationCheck-outspec['missionLife']*outspec['missionPortion']*365.25) < 0.1, 'total instrument time not consistent with initial calculation'
            if not abs(timeConservationCheck-outspec['missionLife']*outspec['missionPortion']*365.25) < 0.1:
                vprint('total instrument time used is not within total allowed time with 0.1d')
            assert abs(timeConservationCheck-outspec['missionLife']*outspec['missionPortion']*365.25) < 0.5, 'total instrument time not consistent with initial calculation'
            #THIS IS JUST SUMCOMP initscomp0 = sim.SurveySimulation.scomp0

            if 'Izod' in outspec.keys():
                if outspec['Izod'] == 'fZ0': # Use fZ0 to calculate integration times
                    vprint('fZ0 in Izod of outspec')
                    pass # Keep ZL.fZ0... fZ = np.array([self.ZodiacalLight.fZ0.value]*len(sInds))*self.ZodiacalLight.fZ0.unit
                elif outspec['Izod'] == 'fZmin': # Use fZmin to calculate integration times
                    vprint('fZmin in Izod of outspec')
                    fZ = SS.valfZmin
                elif outspec['Izod'] == 'fZmax': # Use fZmax to calculate integration times
                    vprint('fZmax in Izod of outspec')
                    fZ = SS.valfZmax
                elif self.Izod == 'current': # Use current fZ to calculate integration times
                    vprint('current in Izod of outspec')
                    pass # keep ZL.fZ0.... fZ = self.ZodiacalLight.fZ(self.Observatory, self.TargetList, sInds, self.TimeKeeping.currentTimeAbs.copy()+np.zeros(self.TargetList.nStars)*u.d, self.detmode)

            WA = SS.WAint
            _, Cbs, Csps = OS.Cp_Cb_Csp(TL, np.arange(TL.nStars), fZ, ZL.fEZ0, 25.0, WA, SS.detmode)

            #find baseline solution with dMagLim-based integration times
            #self.vprint('Finding baseline fixed-time optimal target set.')
            # t0 = OS.calc_intTime(TL, range(TL.nStars),  
            #         ZL.fZ0, ZL.fEZ0, SS.dMagint, SS.WAint, SS.detmode)
            comp0 = COMP.comp_per_intTime(initt0, TL, np.arange(TL.nStars), 
                    fZ, ZL.fEZ0, SS.WAint, SS.detmode, C_b=Cbs, C_sp=Csps)#Integration time at the initially calculated t0
            sumComp0 = sum(comp0)

            #Plot t0 vs c0
            #scatter(initt0.value, comp0, label='SLSQP $C_0$ ALL')
            ax2.scatter(initt0[initt0.value > 1e-10].value, comp0[initt0.value > 1e-10], label=r'$c_{3,i}$,' + '' + r'$\sum c_{3,i}$' + "=%0.2f"%sumComp0, alpha=0.5, color='red', zorder=2, s=45, marker='s')

            #This is a calculation check to ensure the targets at less than 1e-10 d are trash
            sIndsLT1us = np.arange(TL.nStars)[initt0.value < 1e-10]
            t0LT1us = initt0[initt0.value < 1e-10].value + 0.1
            if len(fZ) == 1:
                tmpfZ = fZ
            else:
                tmpfZ = fZ[sIndsLT1us]
            comp02 = COMP.comp_per_intTime(t0LT1us*u.d, TL, sIndsLT1us.tolist(), 
                    tmpfZ, ZL.fEZ0, SS.WAint[sIndsLT1us], SS.detmode, C_b=Cbs[sIndsLT1us], C_sp=Csps[sIndsLT1us])

            #Overwrite DRM with DRM just calculated
            res = sim.run_sim()
            DRM['DRM'] = sim.SurveySimulation.DRM
        elif 'starkAYO' in outspec['modules']['SurveySimulation']:
            #TODO
            initt0 = np.zeros(sim.SurveySimulation.TargetList.nStars)
            initt0[sim.SurveySimulation.schedule] = sim.SurveySimulation.t_dets


        #extract mission information from DRM
        arrival_times = [DRM['DRM'][i]['arrival_time'].value for i in np.arange(len(DRM['DRM']))]
        star_inds = [DRM['DRM'][i]['star_ind'] for i in np.arange(len(DRM['DRM']))]
        sumOHTIME = outspec['settlingTime'] + outspec['starlightSuppressionSystems'][0]['ohTime'].value
        raw_det_time = [DRM['DRM'][i]['det_time'].value for i in np.arange(len(DRM['DRM']))]#DOES NOT INCLUDE overhead time
        det_times = [DRM['DRM'][i]['det_time'].value+sumOHTIME for i in np.arange(len(DRM['DRM']))]#includes overhead time
        det_timesROUNDED = [round(DRM['DRM'][i]['det_time'].value+sumOHTIME,1) for i in np.arange(len(DRM['DRM']))]
        ObsNums = [DRM['DRM'][i]['ObsNum'] for i in np.arange(len(DRM['DRM']))]
        y_vals = np.zeros(len(det_times)).tolist()
        char_times = [DRM['DRM'][i]['char_time'].value*(1.+outspec['charMargin'])+sumOHTIME for i in np.arange(len(DRM['DRM']))]
        OBdurations = np.asarray(outspec['OBendTimes'])-np.asarray(outspec['OBstartTimes'])
        #sumOHTIME = [1 for i in np.arange(len(DRM['DRM']))]
        vprint(sum(det_times))
        vprint(sum(char_times))

        #DIRECT COMPARISON BETWEEN RAW_DET_TIME and initt0
        # print(sum(initt0[initt0.value>0].value))
        # print(sum(np.asarray(raw_det_time)))
        # print(initt0[initt0.value>0].value - np.asarray(raw_det_time))
        # print(np.mean(initt0[initt0.value>0].value - np.asarray(raw_det_time)))

        #Display Text
        #Observations
        #Planned: num
        #Actual: num
        ax1.text(0.1, 0.4, 'Observations\nPlanned:%s\nActual:%s'%("{:,}".format(numObs0),"{:,}".format(len(raw_det_time))), weight='bold', horizontalalignment='left', fontsize=8)
        #TXT1.text(0.5, 0.4, '# Universe\nPlanets:\n%s'%("{:,}".format(len(x))), weight='bold', horizontalalignment='center', fontsize=8)
        #TXT1.text(0.5, -0.1, '# Sims\n%s'%("{:,}".format(len(out['Rps']))), weight='bold', horizontalalignment='center', fontsize=8)

        #calculate completeness at the time of each star observation
        slewTimes = np.zeros(len(star_inds))
        fZ_obs = ZL.fZ(Obs, TL, star_inds, TK.missionStart + (arrival_times + slewTimes)*u.d, SS.detmode)
        _, Cb, Csp = OS.Cp_Cb_Csp(TL, star_inds, fZ_obs, ZL.fEZ0, 25.0, SS.WAint[star_inds], SS.detmode)

        comps = COMP.comp_per_intTime(raw_det_time*u.d, TL, star_inds, fZ_obs, 
                ZL.fEZ0, SS.WAint[star_inds], SS.detmode, C_b=Cb, C_sp=Csp)
        sumComps = sum(comps)

        xlims = [10.**-6, 1.1*max(raw_det_time)]
        ylims = [10.**-6, 1.1*max(comps)]
        #if not plt.get_fignums(): # there is no figure open
        #    plt.figure()
        ax2.scatter(raw_det_time, comps, label=r'$c_{t_{Obs},i}$,' + '' + r'$\sum c_{t_{Obs},i}$' + "=%0.2f"%sumComps, alpha=0.5, color='blue', zorder=2)
        ax2.set_xlim(xlims)
        ax2.set_ylim(ylims)
        ax2.set_xlabel(r'Integration Time, $t_i$, in (days)',weight='bold')
        ax2.set_ylabel(r'Target Completeness, $c_i$',weight='bold')
        legend_properties = {'weight':'bold'}
        ax2.legend(prop=legend_properties)
        ax0.set_xlim(xlims)
        ax3.set_ylim(ylims)
        #ax2.set_xscale('log')
        ax0.set_xscale('log')
        ax0.set_xticks([])
        ax3.set_yticks([])
        nullfmt = NullFormatter()
        ax0.xaxis.set_major_formatter(nullfmt)
        ax1.xaxis.set_major_formatter(nullfmt)
        ax1.yaxis.set_major_formatter(nullfmt)
        ax3.yaxis.set_major_formatter(nullfmt)
        ax0.axis('off')
        ax1.axis('off')
        ax3.axis('off')


        #Done plotting Comp vs intTime of Observations
        date = str(datetime.datetime.now())
        date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
        fname = 'C0vsT0andCvsT_' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'))
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))

        #plt.show(block=False)

        ax0.set_ylabel(r'$\frac{{t_i\ Freq.}}{{{}\ Targets}}$'.format(numObs0),weight='bold', multialignment='center')
        ax3.set_xlabel(r'$\frac{{c_i\ Freq.}}{{{}\ Targets}}$'.format(numObs0),weight='bold', multialignment='center')


        #Manually Calculate the difference to veryify all det_times are the same
        tmpdiff = np.asarray(initt0[star_inds]) - np.asarray(raw_det_time)
        vprint(max(tmpdiff))

        vprint(-2.5*np.log10(ZL.fZ0.value)) # This is 23
        vprint(-2.5*np.log10(np.mean(fZ).value))

        



        ###### Plot C vs T Lines
        #self.plotCvsTlines(TL, Obs, TK, OS, SS, ZL, sim, COMP, PPoutpath, folder, date, ax2)
        """ Plots CvsT with Lines
        #From starkAYO_staticSchedule_withPlotting_copy_Feb6_2018.py
        #Lines 1246-1313, 1490-1502
        """
        ax2.set_xscale('log')
        sInds = np.arange(TL.nStars)
        #DELETE mode = filter(lambda mode: mode['detectionMode'] == True, OS.observingModes)[0]
        mode = [mode for mode in OS.observingModes if mode['detectionMode'] == True][0]#assuming first detection mode
        #fZ, fZabsTime = ZL.calcfZmin(sInds, Obs, TL, TK, mode, SS.cachefname)
        fEZ = ZL.fEZ0
        #WA = OS.WA0
        WA = SS.WAint
        dmag = np.linspace(1, COMP.dMagLim, num=1500,endpoint=True)
        Cp = np.zeros([sInds.shape[0],dmag.shape[0]])
        Cb = np.zeros(sInds.shape[0])
        Csp = np.zeros(sInds.shape[0])
        for i in np.arange(dmag.shape[0]):
            Cp[:,i], Cb[:], Csp[:] = OS.Cp_Cb_Csp(TL, sInds, fZ, fEZ, dmag[i], WA, mode)
        Cb = Cb[:]#Cb[:,0]/u.s#note all Cb are the same for different dmags. They are just star dependent
        Csp = Csp[:]#Csp[:,0]/u.s#note all Csp are the same for different dmags. They are just star dependent
        #self.Cp = Cp[:,:] #This one is dependent upon dmag and each star
        
        cmap = plt.cm.get_cmap('autumn_r')
        intTimes = np.logspace(-6,3,num=400,base=10.0)#define integration times we will evaluate at
        actualComp = np.zeros([sInds.shape[0],intTimes.shape[0]])
        for j in np.arange(intTimes.shape[0]):
            actualComp[:,j] = COMP.comp_per_intTime((intTimes[j]+np.zeros([sInds.shape[0]]))*u.d, TL, sInds, fZ, fEZ, WA, mode, Cb/u.s, Csp/u.s)
        
        #Plot Top 10 black Lines
        compObs = COMP.comp_per_intTime(initt0, TL, sInds, fZ, fEZ, WA, mode, Cb/u.s, Csp/u.s)#integration time at t0
        compObs2 = np.asarray([gg for gg in compObs if gg > 0.])
        tmpI = np.asarray([gg for gg in sInds if compObs[gg] > 0.]) #Inds of sInds with positive Complateness
        maxCI = np.argmax(compObs) # should return ind of max C0
        minCI = tmpI[np.argmin(compObs2)] # should return ind of min C0
        tmpI2 = np.argsort(compObs)[-10:]
        middleCI = compObs.tolist().index(np.percentile(compObs2,50,interpolation='nearest'))

        for l in np.arange(10):
            ax2.plot(intTimes,actualComp[tmpI2[l],:],color='k',zorder=1)
        ax2.plot(intTimes,actualComp[middleCI,:],color='k',zorder=1)
        ax2.plot(intTimes,actualComp[minCI,:],color='k',zorder=1)
        #plt.show(block=False)
        ###############################

        #ax2.set_xscale('log')
        #plt.rcParams['axes.linewidth']=2
        #plt.rc('font',weight='bold') 
        #plt.title('Generic Title I Forgot to Update',weight='bold')
        #plt.xlabel(r'Integration Time, $\tau$ (days)',weight='bold',fontsize=14)
        #plt.ylabel('Completeness',weight='bold',fontsize=14)
        #plt.rc('axes',linewidth=2)
        #plt.rc('lines',linewidth=2)
        #Plot Colorbar
        cmap = plt.cm.get_cmap('autumn_r')

        compatt0 = np.zeros([sInds.shape[0]])
        for j in np.arange(sInds.shape[0]):
            if len(fZ) == 1:
                tmpfZ = fZ
            else:
                tmpfZ = fZ[j]
            compatt0[j] = COMP.comp_per_intTime(initt0[j], TL, sInds[j], tmpfZ, fEZ, WA[j], mode, Cb[j]/u.s, Csp[j]/u.s)
        #ax2.scatter(initt0,compatt0,color='k',marker='o',zorder=3,label=r'$C_{i}(\tau_{0})$')
        #plt.show(block=False)

        def plotSpecialPoints(ind, TL, OS, fZ, fEZ, COMP, WA, mode, sim):
            #### Plot Top Performer at dMagLim, max(C/t)
            if not len(fZ) == 1:
                fZ = fZ[ind]
            if not len(WA) == 1:
                WA = WA[ind]
            tCp, tCb, tCsp = OS.Cp_Cb_Csp(TL, ind, fZ, fEZ, COMP.dMagLim, WA, mode)
            tdMaglim = OS.calc_intTime(TL, ind, fZ, fEZ, COMP.dMagLim, WA, mode)
            Cdmaglim = COMP.comp_per_intTime(tdMaglim, TL, ind, fZ, fEZ, WA, mode, tCb[0], tCsp[0])
            #ax2.scatter(tdMaglim,Cdmaglim,marker='x',color='red',zorder=3)
            def objfun(t, TL, tmpI, fZ, fEZ, WA, mode, OS):
                dmag = OS.calc_dMag_per_intTime(t*u.d, TL, tmpI, fZ, fEZ, WA, mode)#We must calculate a different dmag for each integraiton time
                Cp, Cb, Csp = OS.Cp_Cb_Csp(TL, tmpI, fZ, fEZ, dmag, WA, mode)#We must recalculate Cb and Csp at each dmag
                return -COMP.comp_per_intTime(t*u.d, TL, tmpI, fZ, fEZ, WA, mode, Cb, Csp)/t
            out = minimize_scalar(objfun,method='bounded',bounds=[0,10**3.], args=(TL, ind, fZ, fEZ, WA, mode, OS))#, options={'disp': 3, 'xatol':self.ftol, 'maxiter': self.maxiter}) 
            tMaxCbyT = out['x']
            CtMaxCbyT = COMP.comp_per_intTime(tMaxCbyT*u.d, TL, ind, fZ, fEZ, WA, mode, tCb[0], tCsp[0])
            #ax2.scatter(tMaxCbyT,CtMaxCbyT,marker='D',color='blue',zorder=3)
            return tdMaglim, Cdmaglim, tMaxCbyT, CtMaxCbyT
        ax2.scatter(10**0.,-1.,marker='o',facecolors='white', edgecolors='black',zorder=3,label=r'$c_{\Delta mag_{lim}}$')
        ax2.scatter(10**0.,-1.,marker='D',color='blue',zorder=3,label=r'Max $c_i/t_i$')
        #plt.show(block=False)


        #tdMaglim, Cdmaglim, tMaxCbyT, CtMaxCbyT = plotSpecialPoints(maxCI, TL, OS, fZ, fEZ, COMP, WA, mode, sim)
        #ax2.scatter(tdMaglim,Cdmaglim,marker='o',facecolors='white', edgecolors='black',zorder=3)
        #ax2.scatter(tMaxCbyT,CtMaxCbyT,marker='D',color='blue',zorder=3)
        for l in np.arange(10):
            tmptdMaglim, tmpCdmaglim, tmptMaxCbyT, tmpCtMaxCbyT = plotSpecialPoints(tmpI2[l], TL, OS, fZ, fEZ, COMP, WA, mode, sim)
            ax2.scatter(tmptdMaglim,tmpCdmaglim,marker='o',facecolors='white', edgecolors='black',zorder=3)
            ax2.scatter(tmptMaxCbyT,tmpCtMaxCbyT,marker='D',color='blue',zorder=3)
        tdMaglim, Cdmaglim, tMaxCbyT, CtMaxCbyT = plotSpecialPoints(middleCI, TL, OS, fZ, fEZ, COMP, WA, mode, sim)
        ax2.scatter(tdMaglim,Cdmaglim,marker='o',facecolors='white', edgecolors='black',zorder=3)
        ax2.scatter(tMaxCbyT,CtMaxCbyT,marker='D',color='blue',zorder=3)
        tdMaglim, Cdmaglim, tMaxCbyT, CtMaxCbyT = plotSpecialPoints(minCI, TL, OS, fZ, fEZ, COMP, WA, mode, sim)
        ax2.scatter(tdMaglim,Cdmaglim,marker='o',facecolors='white', edgecolors='black',zorder=3)
        ax2.scatter(tMaxCbyT,CtMaxCbyT,marker='D',color='blue',zorder=3)
        #plt.show(block=False)

        ax2.plot([1e-5,1e-5],[0,0],color='k',label=r'Numerical $c_{i}(t)$',zorder=1)
        ax2.legend(loc=2)
        ax2.set_xlim([1e-6,10.*max(initt0.value)])
        ax0.set_xlim([1e-6,10.*max(initt0.value)])
        ax2.set_ylim([1e-6,1.1*max(compatt0)])
        ax3.set_ylim([1e-6,1.1*max(compatt0)])

        #plt.show(block=False)
        fname = 'CvsTlines_' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'))
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))
        ##################

        #### Plot Axis Histograms
        ax0.axis('on')
        ax3.axis('on')
        #ax0.set_xlim(xlims)
        #ax3.set_ylim(ylims)
        ax0.set_xlim([1e-6,10.*max(initt0.value)])
        ax3.set_ylim([1e-6,1.1*max(compatt0)])
        ax0.set_xscale('log')
        #ax3.set_yscale('log')
        ax0.set_xticks([])
        ax3.set_yticks([])
        nullfmt = NullFormatter()
        ax0.xaxis.set_major_formatter(nullfmt)
        ax1.xaxis.set_major_formatter(nullfmt)
        ax1.yaxis.set_major_formatter(nullfmt)
        ax3.yaxis.set_major_formatter(nullfmt)
        
        xmin = xlims[0]
        xmax = xlims[1]
        ymin = ylims[0]
        ymax = ylims[1]
        # Make the 'main' temperature plot
        # Define the number of bins
        #Base on number of targets???
        nxbins = 50# a bins
        nybins = 50# Rp bins
        nbins = 100
        xbins = np.logspace(start = np.log10(xmin), stop = np.log10(xmax), num = nxbins)
        ybins = np.linspace(start = ymin, stop = ymax, num = nybins)
        xcenter = (xbins[0:-1]+xbins[1:])/2.0
        ycenter = (ybins[0:-1]+ybins[1:])/2.0
        aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)
         
        x = np.asarray(raw_det_time)
        y = comps
        H, xedges,yedges = np.histogram2d(x,y,bins=(xbins,ybins))#,normed=True)
        X = xcenter
        Y = ycenter
        Z = H

        n0, bins0, patches0 = plt.subplot(gs[1]).hist(x, bins=xbins, color = 'black', alpha = 0., fill='black', histtype='step')#,normed=True)#, hatch='-/')#1D histogram of universe a
        center0 = (bins0[:-1] + bins0[1:]) / 2.
        width0=np.diff(bins0)
        ax0.bar(center0, n0/float(numObs0), align='center', width=width0, color='black', fill='black')

        n3, bins3, patches3 = plt.subplot(gs[1]).hist(y, bins=ybins, color = 'black', alpha = 0., fill='black', histtype='step')#,normed=True)#, hatch='-/')#1D histogram of universe a
        center3 = (bins3[:-1] + bins3[1:]) / 2.
        width3=np.diff(bins3)
        ax3.barh(center3, np.asarray(n3/float(numObs0)), align='center', height=width3, color='black', fill='black')
        plt.show(block=False)

        fname = 'CvsTlinesAndHists_' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'))
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))

        #self.plotTauHist()
        #self.plotCompHist()
        plt.close('all')#required before next plotting utility runs


        #### Loading ALIAS FILE ##################################
        #OLD aliasname = 'alias_4_11_2019.pkl'
        aliasname = 'alias_10_07_2019.pkl'
        self.classpath = os.path.split(inspect.getfile(self.__class__))[0]
        vprint(inspect.getfile(self.__class__))
        self.alias_datapath = os.path.join(self.classpath, aliasname)
        #Load pkl and outspec files
        try:
            with open(self.alias_datapath, 'rb') as f:#load from cache
                alias = pickle.load(f, encoding='latin1')
        except:
            vprint('Failed to open fullPathPKL %s'%self.alias_datapath)
            pass
        ##########################################################

        #TODO DOWNLOAD LIST OF STARS WITH DETECTED EXOPLANETS
        data = self.constructIPACurl()
        starsWithPlanets = self.setOfStarsWithKnownPlanets(data)


        outspec = sim.SurveySimulation.genOutSpec()

        OBdurations = np.asarray(outspec['OBendTimes'])-np.asarray(outspec['OBstartTimes'])
        lines = self.writeDATAtoLines(initt0, numObs0, sumOHTIME, raw_det_time, PPoutpath, folder, date, outspec, sim,\
            tmpI, maxCI, minCI, tmpI2, middleCI, comp0, DRM, star_inds, intTimes, actualComp, comps, alias, data, starsWithPlanets)
        self.lines = lines

        #print(saltyburrito)

        #### Save Data File
        fname = 'C0vsT0andCvsTDATA_' + folder.split('/')[-1] + '_' + date
        with open(os.path.join(PPoutpath, fname + '.txt'), 'w') as g:
            g.write("\n".join(lines))
        #end main

    def writeDATAtoLines(self, initt0, numObs0, sumOHTIME, raw_det_time, PPoutpath, folder, date, outspec, sim,\
        tmpI, maxCI, minCI, tmpI2, middleCI, comp0, DRM, star_inds, intTimes, actualComp, comps, alias, data, starsWithPlanets):
        ############################################
        #### Calculate Lines for Data Output
        lines = []
        lines.append('Planned Sum Integration Time: ' + str(sum(initt0[initt0.value>1e-10])))
        lines.append('Planned Number Observations: ' + str(numObs0))
        lines.append('Planned Tsettling+Toh: ' + str(numObs0*sumOHTIME))
        RDT = [rdt for rdt in raw_det_time if rdt>1e-10]
        sumrdt = sum(RDT)
        lines.append('Obs Sum Integration Time: ' + str(sumrdt))
        lines.append('Obs Number Made: ' + str(len(RDT)))
        lines.append('Obs Tsettling+Toh: ' + str(len(RDT)*sumOHTIME))

        #Dump Outspec
        lines.append(json.dumps(outspec,sort_keys=True, indent=4, ensure_ascii=False,
                        separators=(',', ': '), default=array_encoder))

        #Dump Actual DRM
        sumOHTIME = outspec['settlingTime'] + outspec['starlightSuppressionSystems'][0]['ohTime']
        DRMactual = [', '.join([str(DRM['DRM'][i]['arrival_time'].value),\
                                str(DRM['DRM'][i]['star_ind']),\
                                str(DRM['DRM'][i]['det_time'].value),\
                                str(DRM['DRM'][i]['det_time'].value+sumOHTIME),\
                                str(round(DRM['DRM'][i]['det_time'].value+sumOHTIME,1)),\
                                str(DRM['DRM'][i]['ObsNum']),\
                                str(DRM['DRM'][i]['char_time'].value*(1.+outspec['charMargin'])+sumOHTIME)\
                                ]) for i in np.arange(len(DRM['DRM']))]
        lines.append('arrival_time, star_ind, det_time, det_time+sumOHtime, det_time+sumOHTimerounded, ObsNum, totalCharTime\n')
        lines.append('\n'.join(DRMactual))

        lines.append('Seed: ' + str(DRM['seed']) + '\n')

        sim.SurveySimulation.TargetList.ra = sim.SurveySimulation.TargetList.coords.ra
        sim.SurveySimulation.TargetList.dec = sim.SurveySimulation.TargetList.coords.dec
        sim.SurveySimulation.TargetList.distance = sim.SurveySimulation.TargetList.coords.distance
        del sim.SurveySimulation.TargetList.coords
        listOfAtts = sim.SurveySimulation.TargetList.catalog_atts + ['ra','dec','distance']
        listOfAtts.remove('coords')

        #Create a lines of all target stars
        unittedListOfAtts = [att + ' (' + str(getattr(sim.SurveySimulation.TargetList,att).unit) + ')' if 'unit' in dir(getattr(sim.SurveySimulation.TargetList,att)) else att for att in listOfAtts]
        lines.append(', & , '.join(['sInd'] + unittedListOfAtts + ['Observed'] + ['initt0 (d)'] + ['comp0'] + ['KnownPlanet']))
        for i in np.arange(len(tmpI)):
            #### Does the Star Have a Known Planet
            starName = sim.TargetList.Name[tmpI[i]]#Get name of the current star
            if starName in alias[:,1]:
                indWhereStarName = np.where(alias[:,1] == starName)[0][0]# there should be only 1
                starNum = alias[indWhereStarName,3]#this number is identical for all names of a target
                aliases = [alias[j,1] for j in np.arange(len(alias)) if alias[j,3]==starNum] # creates a list of the known aliases
                if np.any([True if aliases[j] in starsWithPlanets else False for j in np.arange(len(aliases))]):
                    KnownPlanet = '1'
                else:
                    KnownPlanet = '0'
            else:
                KnownPlanet = '-2' # this star was not in the alias list
            #### END does a star have a known planet
            lines.append(', & , '.join([str(tmpI[i])] + [str(getattr(sim.SurveySimulation.TargetList,att)[tmpI[i]].value) if 'value' in dir(getattr(sim.SurveySimulation.TargetList,att)) else str(getattr(sim.SurveySimulation.TargetList,att)[tmpI[i]]) for att in listOfAtts] + ['1' if tmpI[i] in star_inds else '0'] + [str(initt0[tmpI[i]].value)] + [str(comp0[tmpI[i]])] + [KnownPlanet]))


        # print(saltyburrito)

        lines.append('Sum Max Completeness Observed Targets: ' + str(sum(actualComp[star_inds,-1])))
        lines.append('Sum Max Completeness Filtered Targets: ' + str(sum(actualComp[:,-1])))
        self.actualComp = actualComp
        lines.append('\% of Max Completeness Observed Targets:')
        self.compDepth = list()
        for i in star_inds:
            tmpInd = np.where(star_inds == i)[0]
            lines.append('sInd: ' + str(i) + ' Max Comp: ' + str(actualComp[i,-1]) + ' Actual Comp: ' + str(comps[tmpInd]) + ' \% of Max C: ' + str(comps[tmpInd]/actualComp[i,-1]*100.))
            self.compDepth.append({'sInd':i, 'maxComp':actualComp[i,-1], 'observedComp':comps[tmpInd], 'percentMaxC':comps[tmpInd]/actualComp[i,-1]*100.})

        #TODO ADD compDepth to lines    
        return lines
    



    def multiRunPostProcessing(self, PPoutpath, folders):
        """Does Nothing
        Args:
            PPoutpath (string) - output path to place data in
            folders (string) - full filepaths to folders containing runs of each run_type
        """
        pass

    def pickPKL(self,folder):
        """Picks a PKL file from the provided folder
        """
        assert os.path.isdir(folder), 'The provided folder %s is not a folder'
        files = os.listdir(folder) # get files located in the provided folder
        assert len(files) > 0, 'There are no files in %s' %(folder)
        assert any('.pkl' in mystring for mystring in files), 'no files in folder are .pkl'
        return random.choice([file for file in files if '.pkl' in file])

    def constructIPACurl(self, tableInput="exoplanets", columnsInputList=['pl_hostname','ra','dec','pl_discmethod','pl_pnum','pl_orbper','pl_orbsmax','pl_orbeccen',\
        'pl_orbincl','pl_bmassj','pl_radj','st_dist','pl_tranflag','pl_rvflag','pl_imgflag',\
        'pl_astflag','pl_omflag','pl_ttvflag', 'st_mass', 'pl_discmethod'],\
        formatInput='json'):
        """
        Extracts Data from IPAC
        Instructions for to interface with ipac using API
        https://exoplanetarchive.ipac.caltech.edu/applications/DocSet/index.html?doctree=/docs/docmenu.xml&startdoc=item_1_01
        Args:
            tableInput (string) - describes which table to query
            columnsInputList (list) - List of strings from https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html 
            formatInput (string) - string describing output type. Only support JSON at this time
        """
        baseURL = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?"
        tablebaseURL = "table="
        # tableInput = "exoplanets" # exoplanets to query exoplanet table
        columnsbaseURL = "&select=" # Each table input must be separated by a comma
        # columnsInputList = ['pl_hostname','ra','dec','pl_discmethod','pl_pnum','pl_orbper','pl_orbsmax','pl_orbeccen',\
        #                     'pl_orbincl','pl_bmassj','pl_radj','st_dist','pl_tranflag','pl_rvflag','pl_imgflag',\
        #                     'pl_astflag','pl_omflag','pl_ttvflag', 'st_mass', 'pl_discmethod']
                            #https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html for explanations

        """
        pl_hostname - Stellar name most commonly used in the literature.
        ra - Right Ascension of the planetary system in decimal degrees.
        dec - Declination of the planetary system in decimal degrees.
        pl_discmethod - Method by which the planet was first identified.
        pl_pnum - Number of planets in the planetary system.
        pl_orbper - Time the planet takes to make a complete orbit around the host star or system.
        pl_orbsmax - The longest radius of an elliptic orbit, or, for exoplanets detected via gravitational microlensing or direct imaging,\
                    the projected separation in the plane of the sky. (AU)
        pl_orbeccen - Amount by which the orbit of the planet deviates from a perfect circle.
        pl_orbincl - Angular distance of the orbital plane from the line of sight.
        pl_bmassj - Best planet mass estimate available, in order of preference: Mass, M*sin(i)/sin(i), or M*sin(i), depending on availability,\
                    and measured in Jupiter masses. See Planet Mass M*sin(i) Provenance (pl_bmassprov) to determine which measure applies.
        pl_radj - Length of a line segment from the center of the planet to its surface, measured in units of radius of Jupiter.
        st_dist - Distance to the planetary system in units of parsecs. 
        pl_tranflag - Flag indicating if the planet transits its host star (1=yes, 0=no)
        pl_rvflag -     Flag indicating if the planet host star exhibits radial velocity variations due to the planet (1=yes, 0=no)
        pl_imgflag - Flag indicating if the planet has been observed via imaging techniques (1=yes, 0=no)
        pl_astflag - Flag indicating if the planet host star exhibits astrometrical variations due to the planet (1=yes, 0=no)
        pl_omflag -     Flag indicating whether the planet exhibits orbital modulations on the phase curve (1=yes, 0=no)
        pl_ttvflag -    Flag indicating if the planet orbit exhibits transit timing variations from another planet in the system (1=yes, 0=no).\
                        Note: Non-transiting planets discovered via the transit timing variations of another planet in the system will not have\
                         their TTV flag set, since they do not themselves demonstrate TTVs.
        st_mass - Amount of matter contained in the star, measured in units of masses of the Sun.
        pl_discmethod - Method by which the planet was first identified.
        """

        columnsInput = ','.join(columnsInputList)
        formatbaseURL = '&format='
        # formatInput = 'json' #https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html#format

        # Different acceptable "Inputs" listed at https://exoplanetarchive.ipac.caltech.edu/applications/DocSet/index.html?doctree=/docs/docmenu.xml&startdoc=item_1_01

        myURL = baseURL + tablebaseURL + tableInput + columnsbaseURL + columnsInput + formatbaseURL + formatInput
        try:
            response = urllib2.urlopen(myURL)
            data = json.load(response)
        except:
            http = urllib3.PoolManager()
            r = http.request('GET', myURL)
            data = json.loads(r.data.decode('utf-8'))
        return data

    def setOfStarsWithKnownPlanets(self, data):
        """ From the data dict created in this script, this method extracts the set of unique star names
        Args:
            data (dict) - dict containing the pl_hostname of each star
        """
        starNames = list()
        for i in np.arange(len(data)):
            starNames.append(data[i]['pl_hostname'])
        return list(set(starNames))


def array_encoder(obj):
    r"""Encodes numpy arrays, astropy Times, and astropy Quantities, into JSON.
    
    Called from json.dump for types that it does not already know how to represent,
    like astropy Quantity's, numpy arrays, etc.  The json.dump() method encodes types
    like integers, strings, and lists itself, so this code does not see these types.
    Likewise, this routine can and does return such objects, which is OK as long as 
    they unpack recursively into types for which encoding is known.th
    
    """
    
    from astropy.time import Time
    from astropy.coordinates import SkyCoord
    if isinstance(obj, Time):
        # astropy Time -> time string
        return obj.fits # isot also makes sense here
    if isinstance(obj, u.quantity.Quantity):
        # note: it is possible to have a numpy ndarray wrapped in a Quantity.
        # NB: alternatively, can return (obj.value, obj.unit.name)
        return obj.value
    if isinstance(obj, SkyCoord):
        return dict(lon=obj.heliocentrictrueecliptic.lon.value,
                    lat=obj.heliocentrictrueecliptic.lat.value,
                    distance=obj.heliocentrictrueecliptic.distance.value)
    if isinstance(obj, (np.ndarray, np.number)):
        # ndarray -> list of numbers
        return obj.tolist()
    if isinstance(obj, (complex, np.complex)):
        # complex -> (real, imag) pair
        return [obj.real, obj.imag]
    if callable(obj):
        # this case occurs for interpolants like PSF and QE
        # We cannot simply "write" the function to JSON, so we make up a string
        # to keep from throwing an error.
        # The fix is simple: when generating the interpolant, add a _outspec attribute
        # to the function (or the lambda), containing (e.g.) the fits filename, or the
        # explicit number -- whatever string was used.  Then, here, check for that 
        # attribute and write it out instead of this dummy string.  (Attributes can
        # be transparently attached to python functions, even lambda's.)
        return 'interpolant_function'
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode()
    # an EXOSIMS object
    if hasattr(obj, '_modtype'):
        return obj.__dict__
    # an object for which no encoding is defined yet
    #   as noted above, ordinary types (lists, ints, floats) do not take this path
    raise ValueError('Could not JSON-encode an object of type %s' % type(obj))
