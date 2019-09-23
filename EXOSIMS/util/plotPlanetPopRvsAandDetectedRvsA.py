"""
This plotting utility creates joint probability distributions of
planetary radius vs semi-major axis for the SimulatedUniverse
specified by outspec.json and the detected planet population
aggregated from all .pkl files in 'folder'.
Side histograms represent occurence frequency of the parameter
per simulation.
In the grid version, the number represents the summation of
values in each cell.
Plot Planet Population Radius vs a AND Detected Planet Rp vs a
Plot will be saved to the directory specified by PPoutpath

Written by Dean Keithly on 5/6/2018
Updated 2/7/2019
"""

import random as myRand
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
try:
    import cPickle as pickle
except:
    import pickle
import os
import numpy as np
#from pylab import *
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
from matplotlib.colors import LogNorm

class plotPlanetPopRvsAandDetectedRvsA(object):
    """Designed to plot Rp vs a of Planet Population Generated and Planet Population Observed
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

        #Extract Data from folder containing pkl files
        out = gen_summary(folder)#out contains information on the detected planets
        allres = read_all(folder)# contains all drm from all missions in folder

        #Convert Extracted Data to x,y
        x, y, z = self.extractXY(out, allres)
        
        # Define the x and y data for detected planets
        det_Rps = np.concatenate(out['Rps']).ravel() # Planet Radius in Earth Radius of detected planets
        det_smas = np.concatenate(out['smas']).ravel()
        det_eccens = np.concatenate(out['es']).ravel()

        #Create Mission Object To Extract Some Plotting Limits
        sim = EXOSIMS.MissionSim.MissionSim(outspecfile, nopar=True)
        ymax = np.nanmax(sim.PlanetPhysicalModel.ggdat['radii']).to('earthRad').value


        ################ 
        #Create Figure and define gridspec
        fig2 = plt.figure(2, figsize=(8.5,4.5))
        gs = gridspec.GridSpec(3,5, width_ratios=[6,1,0.3,6,1.25], height_ratios=[0.2,1,4])
        gs.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rcParams['axes.linewidth']=2
        plt.rc('font',weight='bold')

        #What the plot layout looks like
        ###-----------------------------------
        # | gs[0]  gs[1]  gs[2]  gs[3]  gs[4] |
        # | gs[5]  gs[6]  gs[7]  gs[8]  gs[9] |
        # | gs[10] gs[11] gs[12] gs[13] gs[14]|
        ###-----------------------------------
        ax1 = plt.subplot(gs[5+5])#2D histogram of planet pop
        ax2 = plt.subplot(gs[0+5])#1D histogram of a
        ax3 = plt.subplot(gs[6+5])#1D histogram of Rp
        ax4 = plt.subplot(gs[8+5])#2D histogram of detected Planet Population
        ax5 = plt.subplot(gs[3+5])#1D histogram of detected planet a
        ax6 = plt.subplot(gs[9+5])#1D histogram of detected planet Rp
        TXT1 = plt.subplot(gs[1+5])
        TXT4 = plt.subplot(gs[4+5])
        axCBAR = plt.subplot(gs[0:5])

        # Set up default x and y limits
        print(min(x))
        xlims = [sim.PlanetPopulation.arange[0].value, sim.PlanetPopulation.arange[1].value]#[min(x),max(x)]# of aPOP
        ylims = [min(y),ymax]#max(y)]# of RpPOp
        xmin = xlims[0]
        xmax = xlims[1]
        ymin = ylims[0]
        ymax = ylims[1]

        # Make the 'main' temperature plot
        # Define the number of bins
        nxbins = 50# a bins
        nybins = 50# Rp bins
        nbins = 100
        xbins = np.logspace(start = np.log10(xmin), stop = np.log10(xmax), num = nxbins)
        ybins = np.logspace(start = np.log10(ymin), stop = np.log10(ymax), num = nybins)
        xcenter = (xbins[0:-1]+xbins[1:])/2.0
        ycenter = (ybins[0:-1]+ybins[1:])/2.0
        aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)
         
        H, xedges,yedges = np.histogram2d(x,y,bins=(xbins,ybins),density=True)#normed=True)
        X = xcenter
        Y = ycenter
        Z = H

        #To calculate area under H
        # tmpx = np.diff(xedges)
        # tmpy = np.diff(yedges)
        # tmpmat = np.transpose(np.asarray(tmpx,ndmin))*tmpy
        # #test to be sure it is correct
        # tmpmat[1,2] == tmpx[1]*tmpy[2] #this should be true
        # np.sum(tmpmat*H) #this should equal1


        # Plot the temperature data
        xcents = np.diff(xbins)/2.+xbins[:-1]
        ycents = np.diff(ybins)/2.+ybins[:-1]

        #Plots the contour lines for ax1
        tmpH = H
        tmpH[H==0.] = np.nan
        cscaleMin = np.floor(np.nanmin(np.log10(tmpH))) # 10**min, min order of magnitude
        cscaleMax = np.ceil(np.nanmax(np.log10(tmpH))) # 10**max, max order of magnitude
        levels = 10.**np.arange(cscaleMin,cscaleMax+1)
        cax = ax1.contourf(xcents, ycents, H.T, extent=[xmin, xmax, ymin, ymax], cmap='jet', levels=levels, norm = LogNorm())#locator=ticker.LogLocator())
        CS4 = ax1.contour(cax, colors=('k',), linewidths=(1,), origin='lower', levels=levels, norm = LogNorm())#locator=ticker.LogLocator())

        #Add Colorbar
        cbar = fig2.colorbar(cax, cax=axCBAR, orientation='horizontal')#pad=0.05,
        plt.rcParams['axes.titlepad']=-10
        axCBAR.set_xlabel('Joint Probability Density: Universe (Left) Detected Planets (Right)', weight='bold', labelpad=-35)
        axCBAR.tick_params(axis='x',direction='in',labeltop=True,labelbottom=False)#'off'
        cbar.add_lines(CS4)

        HDET, xedgesDET, yedgesDET = np.histogram2d(det_smas,det_Rps,bins=(xbins,ybins),density=True)#,normed=True)
        caxDET = ax4.contourf(xcents,ycents,HDET.T, extent=[xmin, xmax, ymin, ymax], cmap='jet', levels=levels, norm = LogNorm())#locator=ticker.LogLocator())
        CS42 = ax4.contour(caxDET, colors=('k',), linewidths=(1,), origin='lower', levels=levels, norm = LogNorm())#locator=ticker.LogLocator())


        #Set axes scales to log
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax2.set_xscale('log')
        ax3.set_yscale('log')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax5.set_xscale('log')
        ax6.set_yscale('log')

        #Plot the axes labels
        ax1.set_xlabel('Universe Pop.\nSemi-Major Axis, $a$, in ($AU$)',weight='bold', multialignment='center')
        ax1.set_ylabel('Planet Radius $R_{p}$, in ($R_{\oplus}$)',weight='bold', multialignment='center')
        ax4.set_xlabel('Detected Planet Pop.\nSemi-Major Axis, $a$, in ($AU$)',weight='bold', multialignment='center')

        #Set up the histogram bins
        xbins = np.logspace(start = np.log10(xmin), stop = np.log10(xmax), num = nxbins)
        ybins = np.logspace(start = np.log10(ymin), stop = np.log10(ymax), num = nybins)
         
        #Plot the universe planet pop histograms
        #*note len(out) should equal len(all_res)
        #Universe SMA Hist
        n2, bins2, patches2 = plt.subplot(gs[4+5]).hist(x, bins=xbins, color = 'black', alpha=0., histtype='step',density=True)#,normed=True)#,density=True)#1D histogram of universe a
        center2 = (bins2[:-1] + bins2[1:]) / 2
        width2=np.diff(bins2)
        ax2.bar(center2, n2*(len(x)/float(len(out['smas']))), align='center', width=width2, color='black', fill='black')
        #Detected SMA Hist
        n5, bins5, patches5 = plt.subplot(gs[4+5]).hist(det_smas, bins=xbins, color = 'black', alpha=0., histtype='step',density=True)#,normed=True)#,density=True)#1D histogram of detected planet a
        center5 = (bins5[:-1] + bins5[1:]) / 2
        width5=np.diff(bins5)
        ax5.bar(center5, n5*(len(det_smas)/float(len(out['smas']))), align='center', width=width5, color='black', fill='black')
        #Universe Rp Hist
        n3, bins3, patches3 = plt.subplot(gs[4+5]).hist(y, bins=ybins, color = 'black', alpha=0., histtype='step',density=True)#,normed=True)#,density=True)#1D histogram of detected planet a
        center3 = (bins3[:-1] + bins3[1:]) / 2
        width3=np.diff(bins3)
        ax3.barh(center3, n3*(len(y)/float(len(out['Rps']))), width3, align='center', color='black')
        #aDetected Rp Hist
        n6, bins6, patches6 = plt.subplot(gs[4+5]).hist(det_Rps, bins=ybins, color = 'black', alpha=0., histtype='step',density=True)#,normed=True)#,density=True)#1D histogram of detected planet a
        center6 = (bins6[:-1] + bins6[1:]) / 2
        width6=np.diff(bins6)
        ax6.barh(center6, n6*(len(det_Rps)/float(len(out['Rps']))), width6, align='center', color='black')
        #Label Histograms
        ax2.set_ylabel(r'$\frac{{a\ Freq.}}{{{}\ sims}}$'.format(len(out['Rps'])),weight='bold', multialignment='center')
        ax3.set_xlabel(r'$\frac{{R_P\ Freq.}}{{{}\ sims}}$'.format(len(out['Rps'])),weight='bold', multialignment='center')
        ax6.set_xlabel(r'$\frac{{R_P\ Freq.}}{{{}\ sims}}$'.format(len(out['Rps'])),weight='bold', multialignment='center')

        #Set plot limits
        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims)
        ax2.set_xlim(xlims)
        ax3.set_ylim(ylims)
        ax4.set_xlim(xlims)
        ax4.set_ylim(ylims)
        ax5.set_xlim(xlims)
        ax6.set_ylim(ylims)

        #Remove xticks on x-histogram and remove yticks on y-histogram
        ax2.set_xticks([])
        ax3.set_yticks([])
        ax4.set_yticks([])
        ax5.set_xticks([])
        ax6.set_yticks([])

        # Remove the inner axes numbers of the histograms
        nullfmt = NullFormatter()
        ax2.xaxis.set_major_formatter(nullfmt)
        ax3.yaxis.set_major_formatter(nullfmt)
        ax4.yaxis.set_major_formatter(nullfmt)
        ax5.xaxis.set_major_formatter(nullfmt)
        ax6.yaxis.set_major_formatter(nullfmt)
        axCBAR.yaxis.set_major_formatter(nullfmt)

        fig2.subplots_adjust(bottom=0.15, top=0.75)
        
        TXT1.text(0.0, 0.15, 'Num.\nUniverse\nPlanets:\n%s'%("{:,}".format(len(x))), weight='bold', horizontalalignment='left', fontsize=6)
        TXT4.text(0.0, 0.15, 'Num.\nDetected\nPlanets:\n%s'%("{:,}".format(len(det_Rps))), weight='bold', horizontalalignment='left', fontsize=6)
        
        TXT1.axis('off')
        TXT4.axis('off')
        TXT1.xaxis.set_visible(False)
        TXT1.yaxis.set_visible(False)
        TXT4.xaxis.set_visible(False)
        TXT4.yaxis.set_visible(False)

        # Save to a File
        date = str(datetime.datetime.now())
        date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
        fname = 'RpvsSMAdetections_' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)

        #### Apply Grid to Detected Planet Pop
        #create coarse grid and calculate total numbers in each bin
        acoarse1 = np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),6)
        Rcoarse1 = np.logspace(np.log10(ylims[0]),np.log10(ylims[1]),6)

        #Calculate 2d Histogram for input bins
        hcoarse1 = np.histogram2d(np.hstack(det_smas), np.hstack(det_Rps),bins=[acoarse1,Rcoarse1],normed=False)[0]

        #Plot Vertical and Horizontal Lines
        for R in Rcoarse1:
            ax4.plot(xlims,[R]*2,'k--')
        for a in acoarse1:
            ax4.plot([a]*2,ylims,'k--')

        accents1 = np.sqrt(acoarse1[:-1]*acoarse1[1:])#SMA centers for text
        Rccents1 = np.sqrt(Rcoarse1[:-1]*Rcoarse1[1:])#Rp centers for text

        #Plot Text
        for i in np.arange(len(Rccents1)):
            for j in range(len(accents1)):
                tmp1 = ax4.text(accents1[j],Rccents1[i],u'%2.2f'%(hcoarse1[j,i]/len(out['smas'])),horizontalalignment='center',verticalalignment='center',weight='bold', color='white', fontsize=8)
                tmp1.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])


        #### Apply Grid to Universe Planet Population
        acoarse2 = np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),6)
        Rcoarse2 = np.logspace(np.log10(ylims[0]),np.log10(ylims[1]),6)

        #Calculate 2d Histogram for input bins
        hcoarse2 = np.histogram2d(np.hstack(x), np.hstack(y),bins=[acoarse2,Rcoarse2],normed=False)[0]

        #Plot Vertical and Horizontal Lines
        for R in Rcoarse2:
            ax1.loglog(xlims,[R]*2,'k--')
        for a in acoarse2:
            ax1.loglog([a]*2,ylims,'k--')

        accents2 = np.sqrt(acoarse2[:-1]*acoarse2[1:])#SMA centers for text
        Rccents2 = np.sqrt(Rcoarse2[:-1]*Rcoarse2[1:])#Rp centers for text

        #Plot Text
        for i in np.arange(len(Rccents2)):
            for j in np.arange(len(accents2)):
                tmp2 = ax1.text(accents2[j],Rccents2[i],u'%2.2f'%(hcoarse2[j,i]/len(out['smas'])),horizontalalignment='center',verticalalignment='center',weight='bold', color='white', fontsize=8)
                tmp2.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])

        # Save to a File
        fname = 'RpvsSMAdetectionsGridOverlay_' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500, bbox_inches='tight')
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'), bbox_inches='tight')
        plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500, bbox_inches='tight')
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500, bbox_inches='tight')
        plt.show(block=False)


        ###### Write Data File On This Plot
        lines = []
        lines.append("Number of Simulations (Count): " + str(len(out['Rps'])))
        # Universe Plot Limits
        lines.append('Universe plot xmin (AU): ' + str(xlims[0]) + '\n')
        lines.append('Universe plot xmax (AU): ' + str(xlims[1]) + '\n')
        lines.append('Universe plot ymin (Re): ' + str(ylims[0]) + '\n')
        lines.append('Universe plot ymax (Re): ' + str(ylims[1]) + '\n')
        lines.append('Universe plot xmin (AU): ' + str(xlims[0]) + '\n')
        lines.append('Universe plot xmax (AU): ' + str(xlims[1]) + '\n')
        lines.append('Universe plot ymin (Re): ' + str(ylims[0]) + '\n')
        lines.append('Universe plot ymax (Re): ' + str(ylims[1]) + '\n')
        lines.append('Universe plot max text grid val (Count/Sim): ' + str(np.amax(hcoarse1)) + '\n')
        lines.append('Universe plot Total Detections (Count): ' + str(len(x)) + '\n')
        lines.append('Universe plot Total Detections per Sim (Count/Sim): ' + str(len(x)/len(out['Rps'])) + '\n')
        lines.append('Universe Grid Data: xmin_grid (AU), xmax_grid (AU), ymin_grid (Re), ymax_grid (Re), grid_value (Count/Sim)\n')
        maxBinij = [0,0]
        maxBinVal = 0.
        for i in np.arange(len(Rccents2)):
            for j in np.arange(len(accents2)):
                lines.append(", ".join([str(acoarse2[i]),str(acoarse2[i+1]),str(Rcoarse2[j]),str(Rcoarse2[j+1]),str(hcoarse2[j,i]/len(out['smas']))]) + '\n')
                if hcoarse2[j,i]/len(out['smas']) > maxBinVal:
                    maxBinVal = hcoarse2[j,i]/len(out['smas'])
                    maxBinij = [i,j]
        lines.append('Universe plot Maximum Grid Value (Count/Sim): ' + str(maxBinVal))
        lines.append('Universe plot Maximum Grid i,j (index, index): ' + str(maxBinij[0]) + ', ' + str(maxBinij[1]))


        # Detected Planet Population Limits
        lines.append('Detected plot xmin (AU): ' + str(xlims[0]) + '\n')
        lines.append('Detected plot xmax (AU): ' + str(xlims[1]) + '\n')
        lines.append('Detected plot ymin (Re): ' + str(ylims[0]) + '\n')
        lines.append('Detected plot ymax (Re): ' + str(ylims[1]) + '\n')
        lines.append('Detected plot xmin (AU): ' + str(xlims[0]) + '\n')
        lines.append('Detected plot xmax (AU): ' + str(xlims[1]) + '\n')
        lines.append('Detected plot ymin (Re): ' + str(ylims[0]) + '\n')
        lines.append('Detected plot ymax (Re): ' + str(ylims[1]) + '\n')
        lines.append('Detected plot max text grid val (Count/Sim): ' + str(np.amax(hcoarse2)) + '\n')
        lines.append('Detected plot Total Detections (Count): ' + str(len(det_Rps)) + '\n')
        lines.append('Detected plot Total Detections per Sim (Count/Sim): ' + str(len(x)/len(out['Rps'])) + '\n')
        lines.append('Detected Grid Data: xmin_grid (AU), xmax_grid (AU), ymin_grid (Re), ymax_grid (Re), grid_value (Count/Sim)\n')
        maxBinij = [0,0]
        maxBinVal = 0.
        for i in np.arange(len(Rccents1)):
            for j in np.arange(len(accents1)):
                lines.append(", ".join([str(acoarse1[i]),str(acoarse1[i+1]),str(Rcoarse1[j]),str(Rcoarse1[j+1]),str(hcoarse1[j,i]/len(out['smas']))]) + '\n')
                if hcoarse2[j,i]/len(out['smas']) > maxBinVal:
                    maxBinVal = hcoarse1[j,i]/len(out['smas'])
                    maxBinij = [i,j]
        lines.append('Detected plot Maximum Grid Value (Count/Sim): ' + str(maxBinVal))
        lines.append('Detected plot Maximum Grid i,j (index, index): ' + str(maxBinij[0]) + ', ' + str(maxBinij[1]))


        #### Save Data File
        fname = 'RpvsSMAdetectionsDATA_' + folder.split('/')[-1] + '_' + date
        with open(os.path.join(PPoutpath, fname + '.txt'), 'w') as g:
            g.write("\n".join(lines))

        #### Plot

        self.plotEccenHist(PPoutpath, folder, fname, date, z, det_eccens)

        del out
        del allres

    def extractXY(self, out, allres):
        """
        Simply pulls out the Rp, SMA, and e data for each star in the pkl file
        Args:
        Returns:
            SMA () - SMA of all Stars
            Rp () - Rp of all Stars
            E () - E of all Stars
        """
        Rpunits = allres[0]['systems']['Rp'].unit
        allres_Rp = np.concatenate([allres[i]['systems']['Rp'].value for i in range(len(allres))])
        smaunits = allres[0]['systems']['a'].unit
        allres_sma = np.concatenate([allres[i]['systems']['a'].value for i in range(len(allres))])
        allres_E = np.concatenate([allres[i]['systems']['e'] for i in range(len(allres))])
        SMA = allres_sma
        Rp = allres_Rp
        E = allres_E
        return SMA, Rp, E

    def plotEccenHist(self, PPoutpath, folder, fname, date, uni_eccens, det_eccens):
        """ Plots the input population eccentricity histogram and detected planet eccentricity distribution
        """
        #### Calculate universe planet pop eccen CDF
        uni_bins = np.linspace(start = 0., stop = 1., num = 1001., endpoint=True)
        plt.close(68132188463517733654)
        figH = plt.figure(68132188463517733654)
        uni_n, uni_bins, uni_patches = plt.hist(uni_eccens, bins=uni_bins, alpha=0.3, color='red', label='Universe')
        plt.xlabel('Planet Eccentricity, ' + r"$e$", weight='bold')
        plt.ylabel('Frequency (count)', weight='bold')
        uni_cdf = np.cumsum(uni_n)#cumtrapz(n, bins[:-1], initial=0.)
        uni_cdf_norm = uni_cdf/np.max(uni_cdf)
        #### Calculate detected planet pop eccen CDF
        det_n, det_bins, det_patches = plt.hist(det_eccens, bins=uni_bins, alpha=0.3, color='blue', label='Detected')
        det_cdf = np.cumsum(det_n)#cumtrapz(n, bins[:-1], initial=0.)
        det_cdf_norm = det_cdf/np.max(det_cdf)
        plt.xlim([0.,1.])
        plt.legend()
        plt.show(block=False)
        #not saving

        plt.close(63548643515)
        figI = plt.figure(63548643515)
        plt.plot(uni_bins[:-1],uni_cdf_norm*100.,color='red',linestyle='--', label='Universe CDF')
        plt.plot(uni_bins[:-1],det_cdf_norm*100.,color='blue',linestyle='--', label='Detected CDF')
        plt.ylabel('Percent (%)', weight='bold')
        plt.xlabel('Planet Eccentricity, ' + r"$e$", weight='bold')
        plt.xlim([0.,1.])
        plt.legend()
        plt.show(block=False)
        #not saving

        plt.close(278989879863)
        figJ = plt.figure(278989879863)
        plt.plot(uni_bins[:-1],det_n/uni_n,color='k',alpha=0.5, label='Count ratio')
        plt.plot(uni_bins[:-1],det_cdf/uni_cdf,color='purple', alpha=0.5, label='CDF ratio')
        plt.ylabel('Fraction of Planets Detected in Universe', weight='bold')
        plt.xlabel('Planet Eccentricity, ' + r"$e$", weight='bold')
        plt.ylim([0.,1.])
        plt.xlim([0.,1.])
        plt.legend()
        plt.show(block=False)
        fname = 'kop_EccenHistFractionDetected_1' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500, bbox_inches='tight')
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'), bbox_inches='tight')
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500, bbox_inches='tight')
        plt.ylim([0.,1.1*np.nanmax(det_n/uni_n)])
        fname = 'kop_EccenHistFractionDetected_2' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500, bbox_inches='tight')
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'), bbox_inches='tight')
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500, bbox_inches='tight')
        plt.show(block=False)

        #### Plot histogram of detected
        plt.close(231635)
        fig_eccenHist = plt.figure(231635)
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rcParams['axes.linewidth']=2
        plt.rc('font',weight='bold')
        ax2 = fig_eccenHist.add_subplot(1,1,1)
        uni_bins_coarse = np.linspace(start = 0., stop = 1., num = 21., endpoint=True)
        uni_n2, uni_bins2 = np.histogram(uni_eccens, bins=uni_bins_coarse)
        det_n2, det_bins2 = np.histogram(det_eccens, bins=uni_bins_coarse)
        xcents_uni = (uni_bins_coarse[:-1]+uni_bins_coarse[1:])/2.
        width = np.diff(uni_bins_coarse)
        ax2.bar(xcents_uni, uni_n2, width=width, zorder=8,color='red',alpha=0.3, label='Universe: '+str(int(np.sum(uni_n))))
        ax2.bar(xcents_uni, det_n2, width=width, zorder=8,color='blue',alpha=0.3, label='Detected: '+str(int(np.sum(det_n))))
        ax2.set_xlabel('Oribital Eccentricity', weight='bold')
        ax3 = ax2.twinx()
        ax3.plot(uni_bins[:-1],uni_cdf_norm*100.,color='red',linestyle='--', label='Universe CDF')
        ax3.plot(uni_bins[:-1],det_cdf_norm*100.,color='blue',linestyle='--', label='Detected CDF')
        ax2.set_ylabel('Occurence Frequency (counts)', weight='bold')
        ax3.set_ylabel('Eccentricty CDF (%)', weight='bold')
        ax2.set_xlim(left=0.,right=1.)
        #ax2.set_ylim(bottom=1e-1,top=100.) #np.sum(uni_n))
        ax3.set_ylim(bottom=0.,top=100.)
        ax2.legend(loc='upper right')
        ax3.legend(loc='lower right')
        ax2.ticklabel_format(style='sci', axis='y',scilimits=(0,5))
        plt.show(block=False)

        fname = 'kop_DetectedEccenHist' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500, bbox_inches='tight')
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'), bbox_inches='tight')
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500, bbox_inches='tight')
