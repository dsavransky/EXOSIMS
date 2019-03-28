# -*- coding: utf-8 -*-
"""
Plotting planet population Joint PDF

Written By: Dean Keithly
2/1/2019
"""

try:
    import cPickle as pickle
except:
    import pickle
import os
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
from copy import deepcopy

from astropy.io import fits
import scipy.interpolate
import astropy.units as u
import numpy as np
from EXOSIMS.MissionSim import MissionSim
import numbers
from scipy import interpolate
from matplotlib import ticker, cm


class plotCompletenessJointPDFs(object):
    """Plotting utility to reproduce Completeness Joint PDF
    """
    _modtype = 'util'

    def __init__(self, args=None):
        vprint(args)
        vprint('plotCompletenessJointPDFs done')
        pass

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

        self.plotJointPDF(sim,PPoutpath,folder)


    def plotJointPDF(self, sim, PPoutpath, folder):
        """
        Args:
            sim
            PPoutpath
            folder
        Returns:
            None
        """

        xnew = sim.SurveySimulation.Completeness.xnew #this pulls an array of star-planet distances based on rrange
        dMag = np.linspace(start=10.,stop=50.,num=200)
        xmin = np.min(xnew)
        xmax = np.max(xnew)
        ymin = np.min(dMag)
        ymax = np.max(dMag)

        f = list()
        for k, dm in enumerate(dMag):
            f.append(sim.SurveySimulation.Completeness.EVPOCpdf(xnew,dm)[:,0])
        f = np.asarray(f)
        f[ 10**-5. >= f] = np.nan
        maxf = np.ceil(np.log10(np.nanmax(f)))
        minf = np.floor(np.log10(np.nanmin(f)))
        levelList = [10**x for x in np.linspace(start=minf,stop=maxf,num=maxf-minf+1, endpoint=True)]

        #xlims = [xmin,sim.SurveySimulation.PlanetPopulation.rrange[1].to('AU').value] # largest possible planet orbital radius
        maxXIndinRows = [np.max(np.where(f[i,:]>=1e-5)) for i in np.arange(len(f)) if np.any(f[i,:]>=1e-5)]
        maxYIndinCols = [np.max(np.where(f[:,j]>=1e-5)) for j in np.arange(len(f[0,:]))  if np.any(f[:,j]>=1e-5)]
        xlims = [xmin,xnew[np.max(maxXIndinRows)]] # based on where furthest right of 1e-5 occurs
        ylims = [ymin,dMag[np.max(maxYIndinCols)]]#ymax]

        plt.close(351687)
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rcParams['axes.linewidth']=2
        plt.rc('font',weight='bold')
        fig = plt.figure(351687)
        ax1 = plt.subplot(111)

        CS = ax1.contourf(xnew,dMag,f, levels=levelList, extent=[xlims[0], xlims[1], ylims[0], ylims[1]], cmap='bwr', intepolation='nearest', locator=ticker.LogLocator())
        CS2 = ax1.contour(CS, levels=levelList, extent=[xlims[0], xlims[1], ylims[0], ylims[1]], linewidths=2.0,colors='k')
        #ATTEMPTING TO ADD CONTOUR LABELS plt.clabel(CS2, fmt='%2.1f', colors='k', fontsize=12)

        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims)
        cbar = fig.colorbar(CS)
        plt.xlabel(r'$s$ (AU)',weight='bold')
        plt.ylabel(r'$\Delta$mag',weight='bold')
        plt.show(block=False)

        # Save to a File
        date = unicode(datetime.datetime.now())
        date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
        fname = 'completenessJoinfPDF_' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'), format='png', dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.eps'), format='eps', dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'), format='pdf', dpi=500)
