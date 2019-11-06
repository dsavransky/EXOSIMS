import random as myRand
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
try:
    import cPickle as pickle
except:
    import pickle
import os
import numpy as np
from numpy import nan
if not 'DISPLAY' in os.environ.keys(): #Check environment for keys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import json
from EXOSIMS.util.read_ipcluster_ensemble import gen_summary
from EXOSIMS.util.read_ipcluster_ensemble import read_all
from numpy import linspace
from matplotlib.ticker import NullFormatter, MaxNLocator
from matplotlib import ticker
import astropy.units as u
import matplotlib.patheffects as PathEffects
import datetime
import re
from EXOSIMS.util.vprint import vprint
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import glob


class kopparapuPlot(object):#RpLBins(object):
    """Designed to replicate the Kopparapu plot
    """
    _modtype = 'util'

    def __init__(self, args=None):
        """
        Args:
            args (dict) - 'file' keyword specifies specific pkl file to use
        """
        self.args = args


        r'''Dummy class to hold Rp - Luminosity bin properties.'''
        # Bin the detected planets into types
        # 1: planet-radius bin-edges  [units = Earth radii]
        # Old (early 2018, 3 bins):
        #   Rp_bins = np.array([0.5, 1.4, 4.0, 14.3])
        # New (May 2018, 5 bins x 3 bins, see Kopparapu et al, arxiv:1802.09602v1,
        # Table 1 and in particular Table 3 column 1, column 2 and Fig. 2):
        self.Rp_bins = np.array([0.5, 1.0, 1.75, 3.5, 6.0, 14.3])
        # 1b: bin lo/hi edges, same size as the resulting histograms
        # TODO: Luminosity should perhaps be in increasing order.  Plot how you want, but
        #       compute, store, and exchange in increasing order.
        self.Rp_lo = self.Rp_bins[:-1]
        self.Rp_hi = self.Rp_bins[1:]
        #self.Rp_lo = np.outer(self.Rp_bins[:-1], np.ones((3,1))).ravel()
        #self.Rp_hi = np.outer(self.Rp_bins[1:],  np.ones((3,1))).ravel()
        
        # 2: stellar luminosity bins, in hot -> cold order
        #    NB: decreasing ordering.
        # Old (as above):
        # L_bins = np.array([
        #    [185, 1.5,  0.38, 0.0065],
        #    [185, 1.6,  0.42, 0.0065],
        #    [185, 1.55, 0.40, 0.0055]])
        # New (as above):
        self.L_bins = np.array([
            [182, 1.0,  0.28, 0.0035],
            [187, 1.12, 0.30, 0.0030],
            [188, 1.15, 0.32, 0.0030],
            [220, 1.65, 0.45, 0.0030],
            [220, 1.65, 0.40, 0.0025],
            ])
        # self.L_bins = np.array([
        #     [0.0035, 0.28,  1.0, 182.],
        #     [0.0030, 0.30, 1.12, 187.],
        #     [0.0030, 0.32, 1.15, 188.],
        #     [0.0030, 0.45, 1.65, 220.],
        #     [0.0025, 0.40, 1.65, 220.],
        #     ])
        # the below : selectors are correct for increasing ordering
        self.L_lo = self.L_bins[:,:-1]
        self.L_hi = self.L_bins[:,1:]
        # a bins: Unused
        #a_bins = [1./sqrt([185, 1.5, .38, .0065]),1./sqrt([185, 1.6, 0.42, .0065]),1./sqrt([185, 1.55, .4, .0055])]
        # total number of bins (e.g., 9 = 3*4-3, for a 3x3 histogram)
        RpL_bin_count = self.L_bins.size - (self.Rp_bins.size - 1)
        # # radius/luminosity bin boundaries
        # #   if there are 9 bins, there are 10 bin-edges, at 0.5, 1.5, ..., 9.5.
        # #   this histogram drops the "0", or out-of-range, RpL region
        # RpL_bin_edge_list = np.arange(0, RpL_bin_count+1) + 0.5

        # # set up the bin-number map from (Rp_bin, L_bin) -> RpL_bin
        # # this is a map from (int,int) -> int:
        # #   yields 0 for input pairs outside the allowed range
        # #        [namely, 1..len(Rp_bins) and 1..len(L_bins[i])]
        # #   yields 1...9 otherwise, with 1, 2, 3 for the small planets (Rp bin number = 1).
        # Rp_L_to_RpL_bin = defaultdict(int)
        # # there are many ways to set this up: here is one.
        # # (the below lines are enough for the old 3x3 setup, and they work for the new 5x3 setup too)
        # Rp_L_to_RpL_bin[(1,1)] = 1 # smallest radius, highest luminosity => L bin number 1
        # Rp_L_to_RpL_bin[(1,2)] = 2
        # Rp_L_to_RpL_bin[(1,3)] = 3
        # Rp_L_to_RpL_bin[(2,1)] = 4
        # Rp_L_to_RpL_bin[(2,2)] = 5
        # Rp_L_to_RpL_bin[(2,3)] = 6
        # Rp_L_to_RpL_bin[(3,1)] = 7
        # Rp_L_to_RpL_bin[(3,2)] = 8
        # Rp_L_to_RpL_bin[(3,3)] = 9
        # # New setup has 15 bins due to two new radius bins, so just add them here
        # Rp_L_to_RpL_bin[(4,1)] = 10
        # Rp_L_to_RpL_bin[(4,2)] = 11
        # Rp_L_to_RpL_bin[(4,3)] = 12
        # Rp_L_to_RpL_bin[(5,1)] = 13
        # Rp_L_to_RpL_bin[(5,2)] = 14
        # Rp_L_to_RpL_bin[(5,3)] = 15

    def singleRunPostProcessing(self, PPoutpath, folder):
        """Generates a single yield histogram for the run_type
        Args:
            PPoutpath (string) - output path to place data in
            folder (string) - full filepath to folder containing runs
        """
        #Get name of pkl file
        if not os.path.exists(folder):
            raise ValueError('%s not found'%folder)
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

        out = self.gen_summary_kopparapu(folder)#out contains information on the detected planets
        self.out = out

        # Put Planets into appropriate bins
        aggbins, earthLikeBins = self.putPlanetsInBoxes(out,TL)
        self.aggbins = aggbins
        self.earthLikeBins = earthLikeBins

        # Plot Data
        figVio = plt.figure(figsize=(8.5,4.5))
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rcParams['axes.linewidth']=2
        plt.rc('font',weight='bold')
        gs1 = gridspec.GridSpec(3,16, height_ratios=[8,1,1])#, width_ratios=[1]
        gs1.update(wspace=0.06, hspace=0.2) # set the spacing between axes. 
        ax1 = plt.subplot(gs1[:-2,:])
        figBar = plt.figure(figsize=(8.5,4.5))
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rcParams['axes.linewidth']=2
        plt.rc('font',weight='bold')
        gs2 = gridspec.GridSpec(3,16, height_ratios=[8,1,1])#, width_ratios=[1]
        gs2.update(wspace=0.06, hspace=0.2) # set the spacing between axes. 
        ax2 = plt.subplot(gs2[:-2,:])
        ymaxVio = 0
        ymaxBar = 0

        #INSERT VIOLIN PLOT STUFF HERE
        plt.figure(figVio.number)
        parts = ax1.violinplot(dataset=np.transpose(np.asarray(earthLikeBins)), positions=[0], showmeans=False, showmedians=False, showextrema=False, widths=0.75)
        parts['bodies'][0].set_facecolor('green')
        parts['bodies'][0].set_edgecolor('black')
        parts['bodies'][0].set_alpha(1.0)
        ymaxVio = max([ymaxVio,max(earthLikeBins)])
        ax1.scatter([0], np.mean(earthLikeBins), marker='o', color='k', s=30, zorder=3)
        ax1.vlines([0], min(earthLikeBins), max(earthLikeBins), color='k', linestyle='-', lw=2)
        ax1.vlines([0], np.mean(earthLikeBins)-np.std(earthLikeBins), np.mean(earthLikeBins)+np.std(earthLikeBins), color='purple', linestyle='-', lw=5)
        #### Plot Kopparapu Bar Chart
        plt.figure(figBar.number)
        ax2.bar(0,np.mean(earthLikeBins),width=0.8,color='green')
        ax2.scatter([0], np.mean(earthLikeBins), marker='o', color='k', s=30, zorder=3)
        ax2.vlines([0], min(earthLikeBins), max(earthLikeBins), color='k', linestyle='-', lw=2)
        ax2.vlines([0], np.mean(earthLikeBins)-np.std(earthLikeBins), np.mean(earthLikeBins)+np.std(earthLikeBins), color='purple', linestyle='-', lw=5)
        ymaxBar = max([ymaxBar,max(earthLikeBins)])


        ### Calculate Stats of Bins
        binMeans = np.zeros((5,3)) # planet type, planet temperature
        binUpperQ = np.zeros((5,3)) # planet type, planet temperature
        binLowerQ = np.zeros((5,3)) # planet type, planet temperature
        fifthPercentile = np.zeros((5,3)) # planet type, planet temperature
        twentyfifthPercentile = np.zeros((5,3)) # planet type, planet temperature
        fiftiethPercentile = np.zeros((5,3)) # planet type, planet temperature
        seventyfifthPercentile = np.zeros((5,3)) # planet type, planet temperature
        ninetiethPercentile = np.zeros((5,3)) # planet type, planet temperature
        nintyfifthPercentile = np.zeros((5,3)) # planet type, planet temperature
        minNumDetected = np.zeros((5,3)) # planet type, planet temperature
        percentAtMinimum = np.zeros((5,3)) # planet type, planet temperature
        maxNumDetected = np.zeros((5,3)) # planet type, planet temperature
        #                HOT,  WARM,        COLD
        colorViolins = ['red','royalblue','skyblue']
        for i in np.arange(len(self.Rp_hi)): # iterate over Rp sizes
            for j in np.arange(len(self.L_hi[0])): # iterate over Luminosities
                # Create array of bin counts for this specific bin type
                counts = np.asarray([aggbins[k][i][j] for k in np.arange(len(out['detected']))]) # create array of counts
                binMeans[i][j] = np.mean(counts)
                binUpperQ[i][j] = np.mean(counts) + np.std(counts)
                binLowerQ[i][j] = np.mean(counts) - np.std(counts)
                #stdUniqueDetections[i][j].append(np.std(el))
                fifthPercentile[i][j] = np.percentile(counts,5)
                twentyfifthPercentile[i][j] = np.percentile(counts,25)
                fiftiethPercentile[i][j] = np.percentile(counts,50)
                seventyfifthPercentile[i][j] = np.percentile(counts,75)
                ninetiethPercentile[i][j] = np.percentile(counts,90)
                nintyfifthPercentile[i][j] = np.percentile(counts,95)
                minNumDetected[i][j] = min(counts)
                percentAtMinimum[i][j] = float(counts.tolist().count(min(counts)))/len(counts)
                maxNumDetected[i][j] = max(counts)
                fiftiethPercentile[i][j] = np.percentile(counts,50)
                seventyfifthPercentile[i][j] = np.percentile(counts,75)


                #INSERT VIOLIN PLOT STUFF HERE
                plt.figure(figVio.number)
                parts = ax1.violinplot(dataset=np.transpose(np.asarray(counts)), positions=[3*i+j+1], showmeans=False, showmedians=False, showextrema=False, widths=0.75)
                parts['bodies'][0].set_facecolor(colorViolins[j])
                parts['bodies'][0].set_edgecolor('black')
                parts['bodies'][0].set_alpha(1.0)
                ymaxVio = max([ymaxVio,max(counts)])
                ax1.scatter([3*i+j+1], binMeans[i][j], marker='o', color='k', s=30, zorder=3)
                ax1.vlines([3*i+j+1], min(counts), max(counts), color='k', linestyle='-', lw=2)
                ax1.vlines([3*i+j+1], twentyfifthPercentile[i][j], seventyfifthPercentile[i][j], color='purple', linestyle='-', lw=5)
                #### Plot Kopparapu Bar Chart
                plt.figure(figBar.number)
                ax2.bar(3*i+j+1,binMeans[i][j],width=0.8,color=colorViolins[j])
                ymaxBar = max([ymaxBar,max(counts)])
                ax2.scatter([3*i+j+1], binMeans[i][j], marker='o', color='k', s=30, zorder=3)
                ax2.vlines([3*i+j+1], min(counts), max(counts), color='k', linestyle='-', lw=2)
                ax2.vlines([3*i+j+1], twentyfifthPercentile[i][j], seventyfifthPercentile[i][j], color='purple', linestyle='-', lw=5)

        #Limits Touch Up and Labels
        plt.figure(figVio.number)
        #axes = ax1.gca()
        ax1.set_ylim([0,1.05*np.amax(ymaxVio)])
        ax1.set_xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        ax1.set_xticklabels(('','','','','','','','','','','','','','','',''))
        #ax1.set_xticklabels(('','Hot','Warm','Cold','Hot','Warm','Cold','Hot','Warm','Cold','Hot','Warm','Cold','Hot','Warm','Cold'))
        #ax1.tick_params(axis='x',labelrotation=60)
        ax1.set_ylabel('Unique Detection Yield',weight='bold',fontsize=12)
        plt.figure(figBar.number)
        #axes2 = ax2.gca()
        ax2.set_ylim([0,1.05*np.amax(ymaxBar)])
        ax2.set_xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        ax2.set_xticklabels(('','','','','','','','','','','','','','','',''))
        #ax2.set_xticklabels(('','Hot','Warm','Cold','Hot','Warm','Cold','Hot','Warm','Cold','Hot','Warm','Cold','Hot','Warm','Cold'))
        #ax2.tick_params(axis='x',labelrotation=60)
        ax2.set_ylabel('Unique Detection Yield',weight='bold',fontsize=12)
        
        #Add Hot Warm Cold Labels
        plt.figure(figVio.number)
        axHC = plt.subplot(gs1[-2,:]) # subplot for Plant classification Labels
        ht = 0.9
        xstart = 0.1
        Labels = ['Hot','Warm','Cold']
        for i in np.arange(5):
            axHC.text(xstart+np.float(i)*0.175+0., ht, 'Hot', weight='bold', rotation=60, fontsize=12, color=colorViolins[0])
            axHC.text(xstart+np.float(i)*0.175+0.04, ht, 'Warm', weight='bold', rotation=60, fontsize=12, color=colorViolins[1])
            axHC.text(xstart+np.float(i)*0.175+0.11, ht, 'Cold', weight='bold', rotation=60, fontsize=12, color=colorViolins[2])
        axHC.axis('off')
        plt.figure(figBar.number)
        axHC2 = plt.subplot(gs2[-2,:]) # subplot for Plant classification Labels
        for i in np.arange(5):
            axHC2.text(xstart+np.float(i)*0.175+0., ht, 'Hot', weight='bold', rotation=60, fontsize=12, color=colorViolins[0])
            axHC2.text(xstart+np.float(i)*0.175+0.04, ht, 'Warm', weight='bold', rotation=60, fontsize=12, color=colorViolins[1])
            axHC2.text(xstart+np.float(i)*0.175+0.11, ht, 'Cold', weight='bold', rotation=60, fontsize=12, color=colorViolins[2])
        axHC2.axis('off')

        #Add Planet Type Labels
        #ax1
        plt.figure(figVio.number)
        axEarthLike = plt.subplot(gs1[-1,0]) # subplot for Plant classification Labels
        axEarthLike.text(0.8, 0.3, 'Earth-\nLike', weight='bold', horizontalalignment='center', fontsize=12)
        axEarthLike.axis('off')
        axRocky = plt.subplot(gs1[-1,1:3]) # subplot for Plant classification Labels
        axRocky.text(0.9, 0.2, 'Rocky', weight='bold', horizontalalignment='center', fontsize=12)
        axRocky.axis('off')
        axSEarth = plt.subplot(gs1[-1,4:6]) # subplot for Plant classification Labels
        axSEarth.text(0.7, 0.1, 'Super\nEarth', weight='bold', horizontalalignment='center', fontsize=12)
        axSEarth.axis('off')
        axSNept = plt.subplot(gs1[-1,7:9]) # subplot for Plant classification Labels
        axSNept.text(0.85, 0.1, 'Sub-\nNeptune', weight='bold', horizontalalignment='center', fontsize=12)
        axSNept.axis('off')
        axSJov = plt.subplot(gs1[-1,10:12]) # subplot for Plant classification Labels
        axSJov.text(0.7, 0.1, 'Sub-\nJovian', weight='bold', horizontalalignment='center', fontsize=12)
        axSJov.axis('off')
        axJov = plt.subplot(gs1[-1,13:15]) # subplot for Plant classification Labels
        axJov.text(0.5, 0.3, 'Jovian', weight='bold', horizontalalignment='center', fontsize=12)
        axJov.axis('off')
        #ax2
        plt.figure(figBar.number)
        axEarthLike = plt.subplot(gs2[-1,0]) # subplot for Plant classification Labels
        axEarthLike.text(0.8, 0.3, 'Earth-\nLike', weight='bold', horizontalalignment='center', fontsize=12)
        axEarthLike.axis('off')
        axRocky = plt.subplot(gs2[-1,1:3]) # subplot for Plant classification Labels
        axRocky.text(0.9, 0.2, 'Rocky', weight='bold', horizontalalignment='center', fontsize=12)
        axRocky.axis('off')
        axSEarth = plt.subplot(gs2[-1,4:6]) # subplot for Plant classification Labels
        axSEarth.text(0.7, 0.1, 'Super\nEarth', weight='bold', horizontalalignment='center', fontsize=12)
        axSEarth.axis('off')
        axSNept = plt.subplot(gs2[-1,7:9]) # subplot for Plant classification Labels
        axSNept.text(0.85, 0.1, 'Sub-\nNeptune', weight='bold', horizontalalignment='center', fontsize=12)
        axSNept.axis('off')
        axSJov = plt.subplot(gs2[-1,10:12]) # subplot for Plant classification Labels
        axSJov.text(0.7, 0.1, 'Sub-\nJovian', weight='bold', horizontalalignment='center', fontsize=12)
        axSJov.axis('off')
        axJov = plt.subplot(gs2[-1,13:15]) # subplot for Plant classification Labels
        axJov.text(0.5, 0.3, 'Jovian', weight='bold', horizontalalignment='center', fontsize=12)
        axJov.axis('off')


        plt.show(block=False)
        self.aggbins = aggbins

        #Save Plots
        # Save to a File
        date = str(datetime.datetime.now())
        date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date

        plt.figure(figBar.number)
        fname = 'KopparapuBar_' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)
        plt.figure(figVio.number)
        fname = 'KopparapuVio_' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)

        ###########################################################################
        #### Save Bins to File
        lines = []
        lines.append('#################################################################################')
        lines.append('Rp_Lo: 0.90 , Rp_hi: 1.4 , L_lo: 0.3586 , L_hi: 1.1080 , Planet Type: EarthLike')
        #planet type, planet temperature
        lines.append('   mean: ' + str(np.mean(earthLikeBins)))
        lines.append('   upper STD: ' + str(np.mean(earthLikeBins)+np.std(earthLikeBins)))
        lines.append('   lower STD: ' + str(np.mean(earthLikeBins)-np.std(earthLikeBins)))
        lines.append('   5th percentile: ' + str(np.percentile(earthLikeBins,5)))
        lines.append('   25th percentile: ' + str(np.percentile(earthLikeBins,25)))
        lines.append('   50th percentile: ' + str(np.percentile(earthLikeBins,50)))
        lines.append('   75th percentile: ' + str(np.percentile(earthLikeBins,75)))
        lines.append('   90th percentile: ' + str(np.percentile(earthLikeBins,90)))
        lines.append('   95th percentile: ' + str(np.percentile(earthLikeBins,95)))
        lines.append('   min #: ' + str(min(earthLikeBins)))
        lines.append('   \% at min percentile: ' + str(float(earthLikeBins.count(min(earthLikeBins)))/len(earthLikeBins)))
        lines.append('   max #: ' + str(max(earthLikeBins)))
        lines.append('   50th percentile: ' + str(np.percentile(earthLikeBins,50)))
        lines.append('   75th percentile: ' + str(np.percentile(earthLikeBins,75)))

        #### Plotting Types
        pTypesLabels = ['Rocky','Super Earth','Sub-Neptune','Sub-Jovian','Jovian']
        for i in np.arange(len(self.Rp_hi)): # iterate over Rp sizes
            for j in np.arange(len(self.L_hi[0])): # iterate over Luminosities
                lines.append('#################################################################################')
                lines.append('Rp_Lo: ' + str(self.Rp_lo[i]) + ' , Rp_hi: ' + str(self.Rp_hi[i]) + \
                        ' , L_lo: ' + str(self.L_lo[i][j]) + ' , L_hi: ' + str(self.L_hi[i][j]) + \
                        ' , Planet Type: ' + pTypesLabels[i] + ' , Temperatures: ' + Labels[j])
                #planet type, planet temperature
                lines.append('   mean: ' + str(binMeans[i][j]))
                lines.append('   upper STD: ' + str(binUpperQ[i][j]))
                lines.append('   lower STD: ' + str(binLowerQ[i][j]))
                lines.append('   5th percentile: ' + str(fifthPercentile[i][j]))
                lines.append('   25th percentile: ' + str(twentyfifthPercentile[i][j]))
                lines.append('   50th percentile: ' + str(fiftiethPercentile[i][j]))
                lines.append('   75th percentile: ' + str(seventyfifthPercentile[i][j]))
                lines.append('   90th percentile: ' + str(ninetiethPercentile[i][j]))
                lines.append('   95th percentile: ' + str(nintyfifthPercentile[i][j]))
                lines.append('   min #: ' + str(minNumDetected[i][j]))
                lines.append('   \% at min percentile: ' + str(percentAtMinimum[i][j]))
                lines.append('   max #: ' + str(maxNumDetected[i][j]))
                lines.append('   50th percentile: ' + str(fiftiethPercentile[i][j]))
                lines.append('   75th percentile: ' + str(seventyfifthPercentile[i][j]))

        fname = 'KopparapuDATA_' + folder.split('/')[-1] + '_' + date
        with open(os.path.join(PPoutpath, fname + '.txt'), 'w') as g:
            g.write("\n".join(lines))

    def gen_summary_kopparapu(self,folder):
        """
        """
        pklfiles = glob.glob(os.path.join(folder,'*.pkl'))

        out = {'fname':[],
               'detected':[],
               #'fullspectra':[],
               #'partspectra':[],
               'Rps':[],
               #'Mps':[],
               #'tottime':[],
               'starinds':[],
               'smas':[],
               #'ps':[],
               'es':[],
               #'WAs':[],
               #'SNRs':[],
               #'fZs':[],
               #'fEZs':[],
               #'allsmas':[],
               #'allRps':[],
               #'allps':[],
               #'alles':[],
               #'allMps':[],
               #'dMags':[],
               #'rs':[]}
               }

        for counter,f in enumerate(pklfiles):
            vprint("%d/%d"%(counter,len(pklfiles)))
            with open(f, 'rb') as g:
                res = pickle.load(g, encoding='latin1')

            out['fname'].append(f)
            dets = np.hstack([row['plan_inds'][row['det_status'] == 1]  for row in res['DRM']])
            out['detected'].append(dets) # planet inds

            #out['WAs'].append(np.hstack([row['det_params']['WA'][row['det_status'] == 1].to('arcsec').value for row in res['DRM']]))
            #out['dMags'].append(np.hstack([row['det_params']['dMag'][row['det_status'] == 1] for row in res['DRM']]))
            #out['rs'].append(np.hstack([row['det_params']['d'][row['det_status'] == 1].to('AU').value for row in res['DRM']]))
            #out['fEZs'].append(np.hstack([row['det_params']['fEZ'][row['det_status'] == 1].value for row in res['DRM']]))
            #out['fZs'].append(np.hstack([[row['det_fZ'].value]*len(np.where(row['det_status'] == 1)[0]) for row in res['DRM']]))
            #out['fullspectra'].append(np.hstack([row['plan_inds'][row['char_status'] == 1]  for row in res['DRM']]))
            #out['partspectra'].append(np.hstack([row['plan_inds'][row['char_status'] == -1]  for row in res['DRM']]))
            #out['tottime'].append(np.sum([row['det_time'].value+row['char_time'].value for row in res['DRM']]))
            #out['SNRs'].append(np.hstack([row['det_SNR'][row['det_status'] == 1]  for row in res['DRM']]))
            out['Rps'].append((res['systems']['Rp'][dets]/u.R_earth).decompose().value)
            out['smas'].append(res['systems']['a'][dets].to(u.AU).value)
            #out['ps'].append(res['systems']['p'][dets])
            out['es'].append(res['systems']['e'][dets])
            #out['Mps'].append((res['systems']['Mp'][dets]/u.M_earth).decompose())
            out['starinds'].append(np.hstack([[row['star_ind']]*len(np.where(row['det_status'] == 1)[0]) for row in res['DRM']]))
            #DELETE out['starinds'].append(np.hstack([row['star_ind'][row['det_status'] == 1] for row in res['DRM']]))

            #if includeUniversePlanetPop == True:
            #  out['allRps'].append((res['systems']['Rp']/u.R_earth).decompose().value)
            #  out['allMps'].append((res['systems']['Mp']/u.M_earth).decompose())
            #  out['allsmas'].append(res['systems']['a'].to(u.AU).value)
            #  out['allps'].append(res['systems']['p'])
            #  out['alles'].append(res['systems']['e'])
            del res
            
        return out

    def putPlanetsInBoxes(self,out,TL):
        """ Classifies planets in a gen_summary out file by their hot/warm/cold and rocky/superearth/subneptune/subjovian/jovian bins
        Args:
            out () - a gen_sumamry output list
            TL () - 
        Returns:
            aggbins (list) - dims [# simulations, 5x3 numpy array]
            earthLikeBins (list) - dims [# simulations]
        """
        aggbins = list()
        earthLikeBins = list()
        bins = np.zeros((5,3)) # planet type, planet temperature
        #planet types: rockey, super-Earths, sub-Neptunes, sub-Jovians, Jovians
        #planet temperatures: cold, warm, hot
        for i in np.arange(len(out['starinds'])): # iterate over simulations
            bins = np.zeros((5,3)) # planet type, planet temperature
            earthLike = 0
            starinds = out['starinds'][i]#inds of the stars
            plan_inds = out['detected'][i] # contains the planet inds
            Rps = out['Rps'][i]
            smas = out['smas'][i]
            es = out['smas'][i]
            for j in np.arange(len(plan_inds)): # iterate over targets
                Rp = Rps[j]
                starind = int(starinds[j])
                sma = smas[j]
                ej = es[j]

                bini, binj, earthLikeBool = self.classifyPlanet(Rp, TL, starind, sma, ej)
                if earthLikeBool:
                    earthLike += 1 # just increment count by 1

                #DELETE functionified planet classification
                # bini = np.where((self.Rp_lo < Rp)*(Rp < self.Rp_hi))[0] # index of planet size, rocky,...,jovian
                # if bini.size == 0: # correction for if planet is outside planet range
                #     if Rp < 0:
                #         bini = 0
                #     elif Rp > max(self.Rp_hi):
                #         bini = len(self.Rp_hi)-1
                # else:
                #     bini = bini[0]

                # L_star = TL.L[starind] # grab star luminosity
                # #OLDL_plan = L_star/sma**2. # adjust star luminosity by distance^2 in AU
                # L_plan = L_star/(sma*(1.+(ej**2.)/2.))**2. # adjust star luminosity by distance^2 in AU
                # #CONTENTION FOR USING /SMA**2. HERE.
                # """
                # Averaging orbital radius:
                #     over eccentric anomaly gives the semi-major axis
                #     over true anomaly gives semi-minor axis b=a(1-e**2.)**(1/2)
                #     over mean anomaly gives the time averaged a(1+(e**2.)/2.) 
                # """
                # L_lo = self.L_lo[bini] # lower bin range of luminosity
                # L_hi = self.L_hi[bini] # upper bin range of luminosity
                # binj = np.where((L_lo > L_plan)*(L_plan > L_hi))[0] # index of planet temp. cold,warm,hot
                # if binj.size == 0: # correction for if planet luminosity is out of bounds
                #     if L_plan > max(L_lo):
                #         binj = 0
                #     elif L_plan < min(L_hi):
                #         binj = len(L_hi)-1
                # else:
                #     binj = binj[0]
                # #NEED CITATION ON THIS
                # if (Rp >= 0.90 and Rp <= 1.4) and (L_plan >= 0.3586 and L_plan <= 1.1080):
                #     earthLike += 1

                bins[bini,binj] += 1 # just increment count by 1
                del bini
                del binj


            earthLikeBins.append(earthLike)
            aggbins.append(bins) # aggrgate the bin count for each simulation
        return aggbins, earthLikeBins

    def classifyPlanet(self, Rp, TL, starind, sma, ej):
        """ Determine Kopparapu bin of an individual planet
        Args:
            Rp (float) - planet radius in Earth Radii
            TL (object) - EXOSIMS target list object
            sma (float) - planet semi-major axis in AU
            ej (float) - planet eccentricity
        Returns:
            bini (int) - planet size-type: 0-rocky, 1- Super-Earths, 2- sub-Neptunes, 3- sub-Jovians, 4- Jovians
            binj (int) - planet incident stellar-flux: 0- hot, 1- warm, 2- cold
            earthLike (bool) - boolean indicating whether the planet is earthLike or not earthLike
        """
        bini = np.where((self.Rp_lo < Rp)*(Rp < self.Rp_hi))[0] # index of planet size, rocky,...,jovian
        if bini.size == 0: # correction for if planet is outside planet range
            if Rp < 0:
                bini = 0
            elif Rp > max(self.Rp_hi):
                bini = len(self.Rp_hi)-1
        else:
            bini = bini[0]

        L_star = TL.L[starind] # grab star luminosity
        L_plan = L_star/(sma*(1.+(ej**2.)/2.))**2. # adjust star luminosity by distance^2 in AU
        #*uses true anomaly average distance

        L_lo = self.L_lo[bini] # lower bin range of luminosity
        L_hi = self.L_hi[bini] # upper bin range of luminosity
        binj = np.where((L_lo > L_plan)*(L_plan > L_hi))[0] # index of planet temp. cold,warm,hot
        if binj.size == 0: # correction for if planet luminosity is out of bounds
            if L_plan > max(L_lo):
                binj = 0
            elif L_plan < min(L_hi):
                binj = len(L_hi)-1
        else:
            binj = binj[0]

        #NEED CITATION ON THIS
        earthLike = False
        if (Rp >= 0.90 and Rp <= 1.4) and (L_plan >= 0.3586 and L_plan <= 1.1080):
            earthLike = True

        return bini, binj, earthLike

    def quantize(self, specs, plan_id, star_ind):
        r'''Compute the radius, luminosity, and combined bins for a given planet and star.
        This is Rhonda's original code. It is here to allow cross-checks but is not used now.
        Returns None if the planet/star lies outside the bin boundaries.'''

        # return value indicating error
        error_rval = (None, None, None)
        # extract planet values
        Rp_single = strip_units(specs['Rp'][plan_id])
        a_single = strip_units(specs['a'][plan_id])
        L_star = specs['L'][star_ind]
        L_plan = L_star/a_single**2
        # bin them
        tmpbinplace_Rp = np.digitize(Rp_single, self.Rp_bins)
        if tmpbinplace_Rp >= len(self.Rp_bins):
            return error_rval # out of range
        tmpbinplace_L = np.digitize(L_plan, self.L_bins[tmpbinplace_Rp-1])
        if tmpbinplace_L >= len(self.L_bins[tmpbinplace_Rp-1]):
            return error_rval # out of range
        Rp_bin = tmpbinplace_Rp.item()
        L_bin = tmpbinplace_L.item()
        RpL_bin = tmpbinplace_L.item() + 10*tmpbinplace_Rp.item()
        return Rp_bin, L_bin, RpL_bin

    def quantize_final(self, specs, plan_id, star_ind):
        r'''Compute the final radius/luminosity bin, an integer, for a given planet and star.
        Returns 0 if the planet/star lies outside the bin boundaries.  Returns 1..15 otherwise.'''
        # extract planet and star properties
        Rp_plan = strip_units(specs['Rp'][plan_id])
        a_plan = strip_units(specs['a'][plan_id])
        L_star = specs['L'][star_ind]
        L_plan = L_star / (a_plan**2) # adjust star luminosity by distance^2 in AU
        # Bin by Rp.  "too low" maps to 0, and "too high" maps to len(Rp_bins).
        Rp_bin = np.digitize(Rp_plan, self.Rp_bins)
        # index into L_bins array: if Rp is out-of-range, index is irrelevant
        Rp_bin_index = Rp_bin-1 if (Rp_bin > 0) and (Rp_bin < len(self.Rp_bins)) else 0
        # bin by L
        L_bin = np.digitize(L_plan, self.L_bins[Rp_bin_index])
        # map the pair (Rp,L) -> RpL
        # Rp_bin and L_bin are np arrays, so need to cast to integers
        return self.Rp_L_to_RpL_bin[(int(Rp_bin), int(L_bin))]

    def is_earthlike(self, specs, plan_id, star_ind):
        """Depricated Determine if this planet is Earth-Like or Not, given specs/star id/planet id
        """
        # extract planet and star properties
        Rp_plan = strip_units(specs['Rp'][plan_id])
        a_plan = strip_units(specs['a'][plan_id])
        L_star = specs['L'][star_ind]
        L_plan = L_star / (a_plan**2) # adjust star luminosity by distance^2 in AU
        # its radius (in earth radii) and solar-equivalent luminosity must be
        # between given bounds.  The lower Rp bound is not axis-parallel, but
        # the best axis-parallel bound is 0.90, so that's what we use.
        return (Rp_plan >= 0.90 and Rp_plan <= 1.4) and (L_plan >= 0.3586 and L_plan <= 1.1080)

    def is_earthlike2(self, Rp, L_plan):
        """ Determine if this planet is Earth-Like or Not, given Rp & L_plan
        NEED CITATION ON THESE RANGES
        Args:
            Rp (float) - planet radius in Earth-Radii
            L_plan (float) - adjusted stellar flux on planet
        Returns:
            earthLike (boolean) - True if planet is earth-like, False o.w.
        """
        # its radius (in earth radii) and solar-equivalent luminosity must be
        # between given bounds.  The lower Rp bound is not axis-parallel, but
        # the best axis-parallel bound is 0.90, so that's what we use.
        return (Rp_plan >= 0.90 and Rp_plan <= 1.4) and (L_plan >= 0.3586 and L_plan <= 1.1080)
