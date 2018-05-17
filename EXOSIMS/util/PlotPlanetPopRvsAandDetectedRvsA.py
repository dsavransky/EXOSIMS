"""
Plot Planet Population Radius vs a AND Detected Planet Rp vs a
Plot will be shown and saved to the directory specified by runPath

#Call this function by 
python PlotPlanetPopRvsAandDetectedRvsA.py --runPath '/dir/containing/pkl/files/'

#A specific example calling this function
python PlotPlanetPopRvsAandDetectedRvsA.py --runPath '/home/dean/Documents/SIOSlab/Dean2May18RS12CXXfZ01OB01PP01SU01/'

Written by Dean Keithly on 5/6/2018
"""


#RUN LOAD MISSION FROM SEED OPERATION...
import random as myRand
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
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
from EXOSIMS.util.read_ipcluster_ensemble import gen_summary
from numpy import linspace
from matplotlib.ticker import NullFormatter, MaxNLocator
import matplotlib.pyplot as plt

### FilePath specification
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create MonteCarlo Planet Radius vs Semi-Major Axis Distribution Figures")
    parser.add_argument('--runPath', nargs=1, type=str, help='Full path to directory containing pkl Files')

    args = parser.parse_args()
    runPath = args.runPath[0]
    #outspecfile = args.outspecfile[0]

    if not os.path.exists(runPath):
        raise ValueError('%s not found'%runPath)

    out = gen_summary(runPath)#out contains information on the detected planets
     
    #Load and plot generated planets
    myF = 'a'
    for i in range(100):#Iterate for 100 or until pkl file has been found
        myF = myRand.choice(os.listdir(runPath)) #change dir name to whatever
        if os.path.splitext(myF)[1] == '.pkl':
            break#we found a pkl file break from loop
        assert i < 100, "Could not find a pkl file in runPath: %s"%runPath
    with open(runPath+myF, 'rb') as f:#load from cache
        DRM = pickle.load(f)
    aPOP = DRM['systems']['a'].value
    RpPOP = DRM['systems']['Rp'].value
    x = aPOP
    y = RpPOP




    # Define the x and y data for detected planets
    det_Rps = np.concatenate(out['Rps']).ravel() # Planet Radius in Earth Radius of detected planets
    det_smas = np.concatenate(out['smas']).ravel()
     
    #Create Figure and define gridspec
    fig2 = figure(2, figsize=(8.5,4.5))
    gs = GridSpec(2,4, width_ratios=[4,1,4,1], height_ratios=[1,4])
    gs.update(wspace=0.03, hspace=0.03) # set the spacing between axes. 
    #What the plot layout looks like
    ###----------------------------
    # | gs[0]  gs[1]  gs[2]  gs[3] |
    # | gs[4]  gs[5]  gs[6]  gs[7] |
    ###----------------------------
    ax1 = plt.subplot(gs[4])#2D histogram of planet pop
    ax2 = plt.subplot(gs[0])#1D histogram of a
    ax3 = plt.subplot(gs[5])#1D histogram of Rp
    ax4 = plt.subplot(gs[6])#2D histogram of detected Planet Population
    ax5 = plt.subplot(gs[2])#1D histogram of detected planet a
    ax6 = plt.subplot(gs[7])#1D histogram of detected planet Rp


    # Set up default x and y limits
    xlims = [min(x),max(x)]# of aPOP
    ylims = [min(y),max(y)]# of RpPOP
    # Find the min/max of the POP data and APPLY LIMITS
    xmin = min(xlims)#min of a
    xmax = max(xlims)#max of a
    ymin = min(ylims)#min of Rp
    ymax = max(y)#max of Rp
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
    ax2.set_xlim(xlims)
    ax3.set_ylim(ylims)
    ax4.set_xlim(xlims)
    ax4.set_ylim(ylims)
    ax5.set_xlim(xlims)
    ax6.set_ylim(ylims)

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
     
    H, xedges,yedges = np.histogram2d(x,y,bins=(xbins,ybins))
    X = xcenter
    Y = ycenter
    Z = H

    # Plot the temperature data
    # cax = (ax1.imshow(H, extent=[xmin,xmax,ymin,ymax],
    #     interpolation='nearest', origin='lower',aspect="auto"))#aspectratio))
    xcents = np.diff(xbins)/2.+xbins[:-1]
    ycents = np.diff(ybins)/2.+ybins[:-1]
    cax = ax1.contourf(xcents,ycents,np.log10(H.T/float(len(x))), extent=[xmin, xmax, ymin, ymax], intepolation='nearest')

    HDET, xedgesDET, yedgesDET = np.histogram2d(det_smas,det_Rps,bins=(xbins,ybins))
    caxDET = ax4.contourf(xcents,ycents,np.log10(HDET.T/float(len(det_smas))), extent=[xmin, xmax, ymin, ymax], intepolation='nearest')

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
    ax1.set_xlabel('Universe Pop.\nSemi-Major Axis ($a$) in $AU$',weight='bold', multialignment='center')
    ax1.set_ylabel('Planet Radius ($R_{p}$) in $R_{\oplus}$',weight='bold', multialignment='center')
    ax4.set_xlabel('Detected Planet Pop.\nSemi-Major Axis ($a$) in $AU$',weight='bold', multialignment='center')

    #Set up the histogram bins
    xbins = np.logspace(start = np.log10(xmin), stop = np.log10(xmax), num = nxbins)
    ybins = np.logspace(start = np.log10(ymin), stop = np.log10(ymax), num = nybins)
    #xbins = np.arange(xmin, xmax, (xmax-xmin)/nbins)
    #ybins = np.arange(ymin, ymax, (ymax-ymin)/nbins)
     
    #Plot the universe planet pop histograms
    ax2.hist(x, bins=xbins, color = 'blue')#1D histogram of universe a
    ax5.hist(det_smas, bins=xbins, color = 'blue')#1D histogram of detected planet a
    ax3.hist(y, bins=ybins, orientation='horizontal', color = 'red')#1D histogram of detected planet a
    ax6.hist(det_Rps, bins=ybins, orientation='horizontal', color = 'red')#1D histogram of detected planet Rp
    ax2.set_ylabel('$a$\nFreq.',weight='bold', multialignment='center')
    ax3.set_xlabel('$R_{P}$\nFreq.',weight='bold', multialignment='center')
    ax6.set_xlabel('$R_{P}$\nFreq.',weight='bold', multialignment='center')

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
    
    #plot the detected planet Rp and a #make this a separate plot... or the same plot..... yess lets use subplots
    #ax1.scatter(det_smas, det_Rps, marker='o', color='red', alpha=0.1)

    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    #plt.rc('axes',prop_cycle=(cycler('color',['red','purple'])))#,'blue','black','purple'])))
    rcParams['axes.linewidth']=2
    rc('font',weight='bold')

    #plt.tight_layout({"pad":.0})
    #plt.axis('tight')
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    show(block=False)
    # Save to a File
    filename = 'RpvsSMAdetections'
    savefig(runPath + os.path.splitext(myF)[0] + 'filename' + '.png')
    savefig(runPath + os.path.splitext(myF)[0] + 'filename' + '.svg')
    savefig(runPath + os.path.splitext(myF)[0] + 'filename' + '.eps')






    #Dmitry's Code
    aedges = np.logspace(np.log10(0.2),np.log10(25),101)
    Redges = np.logspace(0,np.log10(16),31)

    acents = np.diff(aedges)/2.+aedges[:-1]
    Rcents = np.diff(Redges)/2.+Redges[:-1]


    h = np.histogram2d(np.hstack(out['smas']), np.hstack(out['Rps']),bins=[aedges,Redges])[0]

    plt.figure()
    plt.clf()
    plt.contourf(acents,Rcents,np.log10(h.T/float(len(out['smas']))))
    gca().set_xscale('log')
    gca().set_yscale('log')
    plt.xlabel('a (AU)')
    plt.ylabel('R ($R_\\oplus$)')
    c = plt.colorbar()

    show(block=False)