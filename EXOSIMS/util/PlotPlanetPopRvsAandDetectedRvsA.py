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

    # Define the x and y data 
    det_Rps = np.concatenate(out['Rps']).ravel() # Planet Radius in Earth Radius of detected planets
    det_smas = np.concatenate(out['smas']).ravel()
     
    fig2 = figure(2)

    gs = GridSpec(2,2, width_ratios=[4,1], height_ratios=[1,4])
    gs.update(wspace=0.03, hspace=0.03) # set the spacing between axes. 
    ax1 = plt.subplot(gs[2])#2Dhistogram
    ax2 = plt.subplot(gs[0])#, sharex=ax1)#x histogram
    ax3 = plt.subplot(gs[3])#, sharey=ax1)#y histogram

    # Set up default x and y limits
    xlims = [min(x),max(x)]
    ylims = [min(y),max(y)]
    # Find the min/max of the data
    xmin = min(xlims)
    xmax = max(xlims)
    ymin = min(ylims)
    ymax = max(y)
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
    ax2.set_xlim(xlims)
    ax3.set_ylim(ylims)

    # Make the 'main' temperature plot
    # Define the number of bins
    nxbins = 50
    nybins = 50
    nbins = 100
     
    xbins = linspace(start = xmin, stop = xmax, num = nxbins)
    ybins = linspace(start = ymin, stop = ymax, num = nybins)
    xcenter = (xbins[0:-1]+xbins[1:])/2.0
    ycenter = (ybins[0:-1]+ybins[1:])/2.0
    aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)
     
    H, xedges,yedges = np.histogram2d(y,x,bins=(ybins,xbins))
    X = xcenter
    Y = ycenter
    Z = H

    # Plot the temperature data
    cax = (ax1.imshow(H, extent=[xmin,xmax,ymin,ymax],
        interpolation='nearest', origin='lower',aspect="auto"))#aspectratio))

    #Plot the axes labels
    ax1.set_xlabel('$\mathrm{Semi-Major\\ Axis\\ (a)\\ in\\ AU}$',fontsize=14,weight='bold')
    ax1.set_ylabel('$\mathrm{Planet Radius\\ (R_{p})\\ in\\ R_{\oplus}}$',fontsize=14,weight='bold')

    #Set up the histogram bins
    xbins = np.arange(xmin, xmax, (xmax-xmin)/nbins)
    ybins = np.arange(ymin, ymax, (ymax-ymin)/nbins)
     
    #Plot the histograms
    ax2.hist(x, bins=xbins, color = 'blue')
    ax3.hist(y, bins=ybins, orientation='horizontal', color = 'red')
    ax2.set_ylabel(r'a Frequency',weight='bold')
    ax3.set_xlabel(r'$R_{P}$ Frequency',weight='bold')

    #Remove xticks on x-histogram and remove yticks on y-histogram
    ax2.set_xticks([])
    ax3.set_yticks([])

    # Remove the inner axes numbers of the histograms
    nullfmt = NullFormatter()
    ax2.xaxis.set_major_formatter(nullfmt)
    ax3.yaxis.set_major_formatter(nullfmt)

    #plot the detected planet Rp and a
    ax1.scatter(det_smas, det_Rps, marker='o', color='red', alpha=0.1)

    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    #plt.rc('axes',prop_cycle=(cycler('color',['red','purple'])))#,'blue','black','purple'])))
    rcParams['axes.linewidth']=2
    rc('font',weight='bold')

    show(block=False)
    # Save to a File
    filename = 'RpvsSMAdetections'
    savefig(runPath + os.path.splitext(myF)[0] + 'filename' + '.png')
    savefig(runPath + os.path.splitext(myF)[0] + 'filename' + '.svg')
    savefig(runPath + os.path.splitext(myF)[0] + 'filename' + '.eps')
