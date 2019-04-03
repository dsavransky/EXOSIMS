# -*- coding: utf-8 -*-
"""Purpose: Plot Observation Timeline for mission

Written by Dean Keithly on 23 Apr, 2018
"""
"""Example 1
I have 1000 pkl files in /home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/run146279583107.pkl and
1qty outspec file in /home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/outspec.json

To generate timelines for these run the following code from an ipython session
from ipython
%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/run146279583107.pkl' \
'/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/outspec.json'
"""
"""Example 2
I have several folders with foldernames /home/dean/Documents/SIOSlab/*fZ*OB*PP*SU*/
each containing ~1000 pkl files and 1 outspec.json file

To plot a random Timeline from each folder, from ipython
%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/' None
"""
#%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/run136726516274.pkl' '/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/outspec.json'
#%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/Dean21May18RS09CXXfZ01OB56PP03SU03/run5991056964408.pkl' '/home/dean/Documents/SIOSlab/Dean21May18RS09CXXfZ01OB56PP03SU03/outspec.json'
#%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/Dean19May18RS09CXXfZ01OB57PP01SU01/run5152585081560.pkl' '/home/dean/Documents/SIOSlab/Dean19May18RS09CXXfZ01OB57PP01SU01/outspec.json'
#%run DRMtoTimelinePlot.py '/home/dean/Documents/SIOSlab/Dean2June18RS26CXXfZ01OB65PP01SU01/run9239688957.pkl' '/home/dean/Documents/SIOSlab/Dean2June18RS26CXXfZ01OB65PP01SU01/outspec.json'
#%run DRMtoTimelinePlot.py '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/run/run1324783950.pkl' '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/run/outspec.json'


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
from EXOSIMS.util.vprint import vprint
import re
import datetime

class plotTimeline(object):
    """
    """
    _modtype = 'util'

    def __init__(self, args=None):
        vprint(args)
        self.args = args
        vprint('plotTimeline Initialization done')
        pass

    def singleRunPostProcessing(self, PPoutpath, folder):
        """This is called by runPostProcessing
        """
        if self.args == None: # Nothing was provided as input
            # grab random pkl file from folder
            pklfiles_in_folder = [myFileName for myFileName in os.listdir(folder) if 'run' in myFileName and '.pkl' in myFileName]  # Get names of all pkl files in path
            pklfname = np.random.choice(pklfiles_in_folder)
            pklfile = os.path.join(folder,pklfname)
        elif 'pklfile' in self.args.keys(): # specific pklfile was provided for analysis
            pklfile = self.args['pklfile']
        else: # grab random pkl file from folder
            pklfiles_in_folder = [myFileName for myFileName in os.listdir(folder) if 'run' in myFileName and '.pkl' in myFileName]  # Get names of all pkl files in path
            pklfname = np.random.choice(pklfiles_in_folder)
            pklfile = os.path.join(folder,pklfname)
        outspecfile = os.path.join(folder,'outspec.json')

        self.plotTimelineWithOB(pklfile=pklfile, outspecfile=outspecfile, PPoutpath=PPoutpath, folder=folder)
        self.plotSnakingTimeline(pklfile=pklfile, outspecfile=outspecfile, PPoutpath=PPoutpath, folder=folder)



    def plotTimelineWithOB(self, pklfile='./', outspecfile='./', PPoutpath='./', folder='./'):
        """
        Args:
            pklfile (string) - full path to pkl file
            outspecfile (string) - full path to outspec file
            PPoutpath (string) - full path to output directory of file
        Return:
        """
        #Error check to ensure provided pkl file exists
        assert os.path.isfile(pklfile), '%s not found' %pklfile
        assert os.path.isfile(outspecfile), '%s not found' %outspecfile
        
        pkldir = [pklfile.split('/')[-2]]
        pklfname = pklfile.split('/')[-1].split('.')[0]

        DRM, outspec = self.loadFiles(pklfile, outspecfile)

        arrival_times = [DRM['DRM'][i]['arrival_time'].value for i in np.arange(len(DRM['DRM']))]
        sumOHTIME = outspec['settlingTime'] + outspec['starlightSuppressionSystems'][0]['ohTime']
        det_times = [DRM['DRM'][i]['det_time'].value+sumOHTIME for i in np.arange(len(DRM['DRM']))]
        det_timesROUNDED = [round(DRM['DRM'][i]['det_time'].value+sumOHTIME,1) for i in np.arange(len(DRM['DRM']))]
        ObsNums = [DRM['DRM'][i]['ObsNum'] for i in np.arange(len(DRM['DRM']))]
        y_vals = np.zeros(len(det_times)).tolist()
        char_times = [DRM['DRM'][i]['char_time'].value*(1+outspec['charMargin'])+sumOHTIME*(DRM['DRM'][i]['char_time'].value > 0.) for i in np.arange(len(DRM['DRM']))]
        OBdurations = np.asarray(outspec['OBendTimes'])-np.asarray(outspec['OBstartTimes'])
        #sumOHTIME = [1 for i in np.arange(len(DRM['DRM']))]
        vprint(sum(det_times))
        vprint(sum(char_times))


        #Check if plotting font #########################################################
        tmpfig = plt.figure(figsize=(30,3.5),num=0)
        ax = tmpfig.add_subplot(111)
        t = ax.text(0, 0, "Obs#   ,  d", ha='center',va='center',rotation='vertical', fontsize=8)
        r = tmpfig.canvas.get_renderer()
        bb = t.get_window_extent(renderer=r)
        Obstxtwidth = bb.width#Width of text
        Obstxtheight = bb.height#height of text
        FIGwidth, FIGheight = tmpfig.get_size_inches()*tmpfig.dpi
        plt.show(block=False)
        plt.close()
        daysperpixelapprox = max(arrival_times)/FIGwidth#approximate #days per pixel
        if mean(det_times)*0.8/daysperpixelapprox > Obstxtwidth:
            ObstextBool = True
        else:
            ObstextBool = False

        tmpfig = plt.figure(figsize=(30,3.5),num=0)
        ax = tmpfig.add_subplot(111)
        t = ax.text(0, 0, "OB#  , dur.=    d", ha='center',va='center',rotation='horizontal', fontsize=12)
        r = tmpfig.canvas.get_renderer()
        bb = t.get_window_extent(renderer=r)
        OBtxtwidth = bb.width#Width of text
        OBtxtheight = bb.height#height of text
        FIGwidth, FIGheight = tmpfig.get_size_inches()*tmpfig.dpi
        plt.show(block=False)
        plt.close()
        if mean(OBdurations)*0.8/daysperpixelapprox > OBtxtwidth:
            OBtextBool = True
        else:
            OBtextBool = False
        #################################################################################



        colors = 'rb'#'rgbwmc'
        patch_handles = []
        fig = plt.figure(figsize=(30,3.5),num=0)
        ax = fig.add_subplot(111)

        # Plot All Detection Observations
        ind = 0
        obs = 0
        for (det_time, l, char_time) in zip(det_times, ObsNums, char_times):
            #print det_time, l
            patch_handles.append(ax.barh(0, det_time, align='center', left=arrival_times[ind],
                color=colors[int(obs) % len(colors)]))
            if not char_time == 0.:
                ax.barh(0, char_time, align='center', left=arrival_times[ind]+det_time,color=(0./255.,128/255.,0/255.))
            ind += 1
            obs += 1
            patch = patch_handles[-1][0] 
            bl = patch.get_xy()
            x = 0.5*patch.get_width() + bl[0]
            y = 0.5*patch.get_height() + bl[1]
            self.prettyPlot()
            if ObstextBool: 
                ax.text(x, y, "Obs#%d, %dd" % (l,det_time), ha='center',va='center',rotation='vertical', fontsize=8)

        # Plot Observation Blocks
        patch_handles2 = []
        for (OBnum, OBdur, OBstart) in zip(xrange(len(outspec['OBendTimes'])), OBdurations, np.asarray(outspec['OBstartTimes'])):
            patch_handles2.append(ax.barh(1, OBdur, align='center', left=OBstart, hatch='//',linewidth=2.0, edgecolor='black'))
            patch = patch_handles2[-1][0] 
            bl = patch.get_xy()
            x = 0.5*patch.get_width() + bl[0]
            y = 0.5*patch.get_height() + bl[1]
            if OBtextBool:
                ax.text(x, y, "OB#%d, dur.= %dd" % (OBnum,OBdur), ha='center',va='center',rotation='horizontal',fontsize=12)

        #Set Plot Xlimit so the end of the timeline is at the end of the figure box
        ax.set_xlim([None, outspec['missionLife']*365.25])


        # Plot Asthetics
        y_pos = np.arange(2)#Number of xticks to have

        self.prettyPlot()
        ax.set_yticks(y_pos)
        ax.set_yticklabels(('Obs','OB'),fontsize=12)
        ax.set_xlabel('Current Normalized Time (days)', weight='bold',fontsize=12)
        title('Mission Timeline for runName: ' + pkldir[0] + '\nand pkl file: ' + pklfname, weight='bold',fontsize=12)
        plt.tight_layout()
        plt.show(block=False)

        date = unicode(datetime.datetime.now())
        date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
        fname = 'Timeline_' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath,fname+'.png'))
        plt.savefig(os.path.join(PPoutpath,fname+'.svg'))
        plt.savefig(os.path.join(PPoutpath,fname+'.eps'))
        plt.savefig(os.path.join(PPoutpath,fname+'.pdf'))

    def prettyPlot(self):
        """ Makes Plots Pretty
        """
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rcParams['axes.linewidth']=2
        plt.rc('font',weight='bold')

    def loadFiles(self,pklfile,outspecfile):
        """ loads pkl and outspec files
        Args:
            pklfile (string) - full filepath to pkl file to load
            outspecfile (string) - fille filepath to outspec.json file
        Return:
            DRM (dict) - a dict containing seed, DRM, system
            outspec (dict) - a dict containing input instructions
        """
        try:
            with open(pklfile, 'rb') as f:#load from cache
                DRM = pickle.load(f)
        except:
            print('Failed to open pklfile %s'%pklfile)
            pass
        try:
            with open(outspecfile, 'rb') as g:
                outspec = json.load(g)
        except:
            print('Failed to open outspecfile %s'%outspecfile)
            pass
        return DRM, outspec

    def plotSnakingTimeline(self, pklfile='./', outspecfile='./', PPoutpath='./',folder='./'):
        """Plots a Timeline where each year is a "new line" on the chart
        Args:
            pklfile (string) - full filepath to pkl file to load
            outspecfile (string) - fille filepath to outspec.json file
            PPoutpath (string) - 
        """
        DRM, outspec = self.loadFiles(pklfile, outspecfile)

        allModes = outspec['observingModes']
        mode1 = [mode for mode in allModes if 'detectionMode' in mode.keys() or 'detection' in mode.keys()]
        assert len(mode1) >= 1, 'This needs to be enhanced'
        mode = mode1[0]
        if not 'timeMultiplier' in mode.keys():
            mode['timeMultiplier'] = 1.

        LD = np.arange(len(DRM['DRM']))
        arrival_times = [DRM['DRM'][i]['arrival_time'].value for i in LD]
        sumOHTIME = outspec['settlingTime'] + outspec['starlightSuppressionSystems'][0]['ohTime']
        det_times = [DRM['DRM'][i]['det_time'].value*(mode['timeMultiplier'])+sumOHTIME for i in LD]
        det_timesROUNDED = [round(DRM['DRM'][i]['det_time'].value*(mode['timeMultiplier'])+sumOHTIME,1) for i in LD]
        ObsNums = [DRM['DRM'][i]['ObsNum'] for i in LD]
        y_vals = np.zeros(len(det_times)).tolist()
        char_times = [DRM['DRM'][i]['char_time'].value*(1.+outspec['charMargin'])+sumOHTIME*(DRM['DRM'][i]['char_time'].value > 0.) for i in LD]
        OBdurations = np.asarray(outspec['OBendTimes'])-np.asarray(outspec['OBstartTimes'])
        
        #print(sum(det_times))
        #print(sum(char_times))
        #This is just testing stuff for now
        from pylab import *
        arr = [DRM['DRM'][i]['arrival_time'].value for i in np.arange(len(DRM['DRM']))]
        dt = [DRM['DRM'][i]['det_time'].value + 1. for i in np.arange(len(DRM['DRM']))]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors=['red','blue']
        for i in np.arange(len(arr)):
            ax.barh(1., dt[i], align='center', left=arr[i],color=colors[int(i) % len(colors)])
        plt.show(block=False)


        #Check if plotting font #########################################################
        tmpfig = plt.figure(figsize=(20,5),num=0)
        ax = tmpfig.add_subplot(111)
        t = ax.text(0, 0, "Obs#   ,  d", ha='center',va='center',rotation='vertical', fontsize=8)
        r = tmpfig.canvas.get_renderer()
        bb = t.get_window_extent(renderer=r)
        Obstxtwidth = bb.width#Width of text
        Obstxtheight = bb.height#height of text
        FIGwidth, FIGheight = tmpfig.get_size_inches()*tmpfig.dpi
        #plt.show(block=False)
        plt.close()
        daysperpixelapprox = max(arrival_times)/FIGwidth#approximate #days per pixel
        if mean(det_times)*0.8/daysperpixelapprox > Obstxtwidth:
            ObstextBool = True
        else:
            ObstextBool = False

        tmpfig = plt.figure(figsize=(25,5),num=0)
        ax = tmpfig.add_subplot(111)
        t = ax.text(0, 0, "OB#  , dur.=    d", ha='center',va='center',rotation='horizontal', fontsize=12)
        r = tmpfig.canvas.get_renderer()
        bb = t.get_window_extent(renderer=r)
        OBtxtwidth = bb.width#Width of text
        OBtxtheight = bb.height#height of text
        FIGwidth, FIGheight = tmpfig.get_size_inches()*tmpfig.dpi
        #plt.show(block=False)
        plt.close()
        if mean(OBdurations)*0.8/daysperpixelapprox > OBtxtwidth:
            OBtextBool = True
        else:
            OBtextBool = False
        #################################################################################


        ######################################################################
        #Finds arrival times that occur within that year
        ObsNumsL = list()
        det_timesL = list()
        char_timesL = list()
        arrival_timesL = list()
        truthArr = list()
        for i in np.arange(int(np.ceil(max(arrival_times)/365.25))):
            truthArr = (np.asarray(arrival_times) >= 365.25*np.float(i))*(np.asarray(arrival_times) < 365.25*np.float(i+1.))
            arrival_timesL.append([arrival_times[ii] for ii in np.where(truthArr)[0]])
            det_timesL.append([det_times[ii] for ii in np.where(truthArr)[0]])
            char_timesL.append([char_times[ii] for ii in np.where(truthArr)[0]])
            ObsNumsL.append([ObsNums[ii] for ii in np.where(truthArr)[0]])
        #######################################################################

        #######################################################################
        # Plotting 
        colors = 'rb'#'rgbwmc'
        patch_handles = []
        fig = plt.figure(figsize=(20,3+int(np.ceil(max(arrival_times)/365.25))/2.))
        self.prettyPlot()
        ax = fig.add_subplot(111)

        char_color=(0./255.,128/255.,0/255.)

        #Plot individual blocks
        # Plot All Detection Observations for Year
        for iyr in np.arange(int(np.ceil(max(arrival_times)/365.25))):
            ind = 0
            obs = 0
            for (det_time, l, char_time, arrival_times_yr) in zip(det_timesL[iyr], ObsNumsL[iyr], char_timesL[iyr], arrival_timesL[iyr]):
                #print det_time, l
                patch_handles.append(ax.barh(int(np.ceil(max(arrival_times)/365.25))-iyr, det_time, align='center', left=arrival_times_yr-365.25*iyr,
                    color=colors[int(obs) % len(colors)]))
                if not char_time == 0.:
                    ax.barh(int(np.ceil(max(arrival_times)/365.25))-iyr, char_time, align='center', left=arrival_times_yr+det_time-365.25*iyr,color=char_color)
                ind += 1
                obs += 1
                patch = patch_handles[-1][0]
                bl = patch.get_xy()
                x = 0.5*patch.get_width() + bl[0]
                y = 0.5*patch.get_height() + bl[1]
                if ObstextBool: 
                    ax.text(x, y, "Obs#%d, %dd" % (l,det_time), ha='center',va='center',rotation='vertical', fontsize=8)

        #Set Plot Xlimit so the end of the timeline is at the end of the figure box
        ax.set_xlim([None, 365.25])


        # Plot Asthetics
        y_pos = np.arange(int(np.ceil(max(arrival_times)/365.25)))+1#Number of xticks to have

        yticklabels = list()
        for i in np.arange(int(np.ceil(max(arrival_times)/365.25))):
            yticklabels.append(str(int(np.ceil(max(arrival_times)/365.25)) - i) + 'yr')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(yticklabels,fontsize=30)
        #ax.set_yticklabels(('3yr','2yr','1yr'),fontsize=30)
        ax.xaxis.set_tick_params(labelsize=30)
        ax.set_xlabel('Time Since Start of Mission Year (days)', weight='bold',fontsize=30)
        plt.title('Mission Timeline for runName: ' + folder.split('/')[-1] + '\nand pkl file: ' + os.path.basename(pklfile).split('.')[0], weight='bold',fontsize=12)
        plt.tight_layout()
        plt.show(block=False)

        date = unicode(datetime.datetime.now())
        date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
        fname = 'TimelineSnake_' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath,fname+'.png'))
        plt.savefig(os.path.join(PPoutpath,fname+'.svg'))
        plt.savefig(os.path.join(PPoutpath,fname+'.eps'))
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))



