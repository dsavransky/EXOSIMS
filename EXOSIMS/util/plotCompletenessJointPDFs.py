# -*- coding: utf-8 -*-
"""
Plotting planet population Joint PDF

Written By: Dean Keithly
2/1/2019
"""

import pickle
import os

if not "DISPLAY" in os.environ.keys():  # Check environment for keys
    import matplotlib

    matplotlib.use("Agg")
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
    """Plotting utility to reproduce Completeness Joint PDF"""

    _modtype = "util"

    def __init__(self, args=None):
        vprint(args)
        vprint("plotCompletenessJointPDFs done")
        pass

    def singleRunPostProcessing(self, PPoutpath, folder):
        """Generates a single yield histogram for the run_type
        Args:
            PPoutpath (string) - output path to place data in
            folder (string) - full filepath to folder containing runs
        """
        # Get name of pkl file
        if not os.path.exists(folder):
            raise ValueError("%s not found" % folder)
        outspecPath = os.path.join(folder, "outspec.json")
        try:
            with open(outspecPath, "rb") as g:
                outspec = json.load(g)
        except:
            vprint("Failed to open outspecfile %s" % outspecPath)
            pass

        # Create Simulation Object
        sim = EXOSIMS.MissionSim.MissionSim(scriptfile=None, nopar=True, **outspec)

        self.plotJointPDF(sim, PPoutpath, folder)
        # self.plotJointPDFwithCompRectDetBoxes(sim, PPoutpath, folder, IWA, OWA, dMag,s0,dMag0,s1,dMag1,sigma_s,sigma_dmag)

    def plotJointPDF(self, sim, PPoutpath, folder):
        """
        Args:
            sim
            PPoutpath
            folder
        Returns:
            None
        """

        xnew = (
            sim.SurveySimulation.Completeness.xnew
        )  # this pulls an array of star-planet distances based on rrange
        dMag = np.linspace(start=10.0, stop=50.0, num=200)
        xmin = np.min(xnew)
        xmax = np.max(xnew)
        ymin = np.min(dMag)
        ymax = np.max(dMag)

        f = list()
        for k, dm in enumerate(dMag):
            if hasattr(sim.SurveySimulation.Completeness, "EVPOCpdf"):
                f.append(sim.SurveySimulation.Completeness.EVPOCpdf(xnew, dm)[:, 0])
            else:
                f.append(sim.SurveySimulation.Completeness.EVPOCpdf_pop(xnew, dm)[:, 0])
        f = np.asarray(f)
        f[10**-5.0 >= f] = np.nan
        maxf = int(np.ceil(np.log10(np.nanmax(f))))
        minf = int(np.floor(np.log10(np.nanmin(f))))
        levelList = [
            10**x
            for x in np.linspace(
                start=minf, stop=maxf, num=maxf - minf + 1, endpoint=True
            )
        ]

        # xlims = [xmin,sim.SurveySimulation.PlanetPopulation.rrange[1].to('AU').value] # largest possible planet orbital radius
        maxXIndinRows = [
            np.max(np.where(f[i, :] >= 1e-5))
            for i in np.arange(len(f))
            if np.any(f[i, :] >= 1e-5)
        ]
        maxYIndinCols = [
            np.max(np.where(f[:, j] >= 1e-5))
            for j in np.arange(len(f[0, :]))
            if np.any(f[:, j] >= 1e-5)
        ]
        xlims = [
            xmin,
            xnew[np.max(maxXIndinRows)],
        ]  # based on where furthest right of 1e-5 occurs
        ylims = [ymin, dMag[np.max(maxYIndinCols)]]  # ymax]

        plt.close(351687)
        plt.rc("axes", linewidth=2)
        plt.rc("lines", linewidth=2)
        plt.rcParams["axes.linewidth"] = 2
        plt.rc("font", weight="bold")
        fig = plt.figure(351687)
        ax1 = plt.subplot(111)

        CS = ax1.contourf(
            xnew,
            dMag,
            f,
            levels=levelList,
            extent=[xlims[0], xlims[1], ylims[0], ylims[1]],
            cmap="bwr",
            intepolation="nearest",
            locator=ticker.LogLocator(),
        )
        CS2 = ax1.contour(
            CS,
            levels=levelList,
            extent=[xlims[0], xlims[1], ylims[0], ylims[1]],
            linewidths=2.0,
            colors="k",
        )
        # ATTEMPTING TO ADD CONTOUR LABELS plt.clabel(CS2, fmt='%2.1f', colors='k', fontsize=12)

        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims)
        cbar = fig.colorbar(CS)
        plt.xlabel(r"$s$ (AU)", weight="bold")
        plt.ylabel(r"$\Delta$mag", weight="bold")
        plt.show(block=False)
        plt.gcf().canvas.draw()

        # Save to a File
        DT = datetime.datetime
        date = str(DT.now())  # ,"utf-8")
        date = "".join(
            c + "_" for c in re.split("-|:| ", date)[0:-1]
        )  # Removes seconds from date
        fname = "completenessJoinfPDF_" + folder.split("/")[-1] + "_" + date
        plt.savefig(os.path.join(PPoutpath, fname + ".png"), format="png", dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + ".svg"))
        plt.savefig(os.path.join(PPoutpath, fname + ".eps"), format="eps", dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + ".pdf"), format="pdf", dpi=500)

    def plotJointPDFwithCompRectDetBoxes(
        self,
        sim,
        PPoutpath,
        folder,
        IWA,
        OWA,
        dMagLim,
        s0,
        dMag0,
        s1,
        dMag1,
        sigma_s,
        sigma_dmag,
    ):
        """
        Args:
            sim
            PPoutpath
            folder
            IWA in pc
            OWA in pc
            dMag
            s0 (float): separation of measurement 1 in AU
            dMag0 (float): dmag of measurement 1
            s1 (float): separation of measurement 1 in AU
            dMag1 (float): dmag of measurement 1
            sigma_s (float): uncertainty in separation measurement in AU
            sigma_dmag (float): uncertainty in dmag measurement
        Returns:
            None
        """

        xnew = (
            sim.SurveySimulation.Completeness.xnew
        )  # this pulls an array of star-planet distances based on rrange
        dMag = np.linspace(start=10.0, stop=50.0, num=200)
        xmin = np.min(xnew)
        xmax = np.max(xnew)
        ymin = np.min(dMag)
        ymax = np.max(dMag)

        f = list()
        for k, dm in enumerate(dMag):
            if hasattr(sim.SurveySimulation.Completeness, "EVPOCpdf"):
                f.append(sim.SurveySimulation.Completeness.EVPOCpdf(xnew, dm)[:, 0])
            else:
                f.append(sim.SurveySimulation.Completeness.EVPOCpdf_pop(xnew, dm)[:, 0])
        f = np.asarray(f)
        f[10**-5.0 >= f] = np.nan
        # this scaling makes it 1 level per log scale
        # maxf = int(np.ceil(np.log10(np.nanmax(f))))
        # minf = int(np.floor(np.log10(np.nanmin(f))))
        # maxf = 200
        # minf = 0
        # levelList = [10**x for x in np.linspace(start=minf,stop=maxf,num=maxf-minf+1, endpoint=True)]
        levelList = np.logspace(start=-5.0, stop=0.0, num=200)

        # xlims = [xmin,sim.SurveySimulation.PlanetPopulation.rrange[1].to('AU').value] # largest possible planet orbital radius
        maxXIndinRows = [
            np.max(np.where(f[i, :] >= 1e-5))
            for i in np.arange(len(f))
            if np.any(f[i, :] >= 1e-5)
        ]
        maxYIndinCols = [
            np.max(np.where(f[:, j] >= 1e-5))
            for j in np.arange(len(f[0, :]))
            if np.any(f[:, j] >= 1e-5)
        ]
        xlims = [
            xmin,
            xnew[np.max(maxXIndinRows)],
        ]  # based on where furthest right of 1e-5 occurs
        ylims = [ymin, dMag[np.max(maxYIndinCols)]]  # ymax]

        plt.close(351687)
        plt.rc("axes", linewidth=2)
        plt.rc("lines", linewidth=2)
        plt.rcParams["axes.linewidth"] = 2
        plt.rc("font", weight="bold")
        fig = plt.figure(351687)
        ax1 = plt.subplot(111)

        CS = ax1.contourf(
            xnew,
            dMag,
            f,
            levels=levelList,
            extent=[xlims[0], xlims[1], ylims[0], ylims[1]],
            cmap="bwr",
            intepolation="nearest",
            locator=ticker.LogLocator(),
        )
        # THIS IS THE BLACK OUTLINE OF EACH CONTOUR CS2 = ax1.contour(CS, levels=levelList, extent=[xlims[0], xlims[1], ylims[0], ylims[1]], linewidths=2.0,colors='k')
        # ATTEMPTING TO ADD CONTOUR LABELS plt.clabel(CS2, fmt='%2.1f', colors='k', fontsize=12)

        ax1.set_xlim(xlims)
        ax1.set_ylim(ylims)
        # cbar = fig.colorbar(CS)
        plt.xlabel(r"$s$ (AU)", weight="bold")
        plt.ylabel(r"$\Delta$mag", weight="bold")
        plt.show(block=False)
        plt.gcf().canvas.draw()

        # Save to a File
        DT = datetime.datetime
        date = str(DT.now())  # ,"utf-8")
        date = "".join(
            c + "_" for c in re.split("-|:| ", date)[0:-1]
        )  # Removes seconds from date
        fname = "completenessJoinfPDF_" + folder.split("/")[-1] + "_" + date
        plt.savefig(os.path.join(PPoutpath, fname + ".png"), format="png", dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + ".svg"))
        plt.savefig(os.path.join(PPoutpath, fname + ".eps"), format="eps", dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + ".pdf"), format="pdf", dpi=500)

        # Add Comp Rectangle
        smax = 10.0 * u.pc.to("AU") * OWA.to("rad").value  # OWA in AU at 10 parsecs
        smin = 10.0 * u.pc.to("AU") * IWA.to("rad").value  # IWA in AU at 10 parsecs
        plt.plot(
            [smin, smin, smax, smax],
            [0.0, dMagLim, dMagLim, 0.0],
            color="black",
            linewidth=2,
        )
        fname = "completenessJoinfPDFwithCompRect_" + folder.split("/")[-1] + "_" + date
        plt.savefig(os.path.join(PPoutpath, fname + ".png"), format="png", dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + ".svg"))
        plt.savefig(os.path.join(PPoutpath, fname + ".eps"), format="eps", dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + ".pdf"), format="pdf", dpi=500)

        # Add Det0
        plt.plot(
            [s0 - sigma_s, s0 - sigma_s, s0 + sigma_s, s0 + sigma_s, s0 - sigma_s],
            [
                dMag0 - sigma_dmag,
                dMag0 + sigma_dmag,
                dMag0 + sigma_dmag,
                dMag0 - sigma_dmag,
                dMag0 - sigma_dmag,
            ],
            color="black",
            linewidth=2,
        )
        fname = (
            "completenessJoinfPDFwithCompRectdet0_" + folder.split("/")[-1] + "_" + date
        )
        plt.savefig(os.path.join(PPoutpath, fname + ".png"), format="png", dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + ".svg"))
        plt.savefig(os.path.join(PPoutpath, fname + ".eps"), format="eps", dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + ".pdf"), format="pdf", dpi=500)

        # Add Det1
        plt.plot(
            [s1 - sigma_s, s1 - sigma_s, s1 + sigma_s, s1 + sigma_s, s1 - sigma_s],
            [
                dMag1 - sigma_dmag,
                dMag1 + sigma_dmag,
                dMag1 + sigma_dmag,
                dMag1 - sigma_dmag,
                dMag1 - sigma_dmag,
            ],
            color="black",
            linewidth=2,
        )
        fname = (
            "completenessJoinfPDFwithCompRectdet0det1_"
            + folder.split("/")[-1]
            + "_"
            + date
        )
        plt.savefig(os.path.join(PPoutpath, fname + ".png"), format="png", dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + ".svg"))
        plt.savefig(os.path.join(PPoutpath, fname + ".eps"), format="eps", dpi=500)
        plt.savefig(os.path.join(PPoutpath, fname + ".pdf"), format="pdf", dpi=500)


"""
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import json
import astropy.units as u  
from EXOSIMS.util import plotCompletenessJointPDFs
pjf = plotCompletenessJointPDFs.plotCompletenessJointPDFs()                                           
OWA = 6*u.arcsec                                                        
OWA = 2.*u.arcsec                                                  
IWA = 0.045*u.arcsec                                                   
dMagLim = 27.5                                                             
IWA = 0.2*u.arcsec                                                     
s0 = 5.                                                                
dMag0 = 22.5                                                           
dMag1 = 26.                                                            
s1=12.                                                                 
sigma_s = 0.25                                                         
sigma_dmag=0.5 
folder = "/home/dean/Documents/exosims/EXOSIMSres/HabExCompSpecPriors_HabEx_4m_50119/HabEx_CSAG13_PPSAG13" 
PPoutpath = "/home/dean/Documents/exosims/EXOSIMSres/HabExCompSpecPriors_HabEx_4m_50119"
outspecPath = os.path.join(folder,'outspec.json')
with open(outspecPath, 'rb') as g:
    outspec = json.load(g)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile=None, nopar=True, **outspec)
pjf.plotJointPDFwithCompRectDetBoxes(sim, PPoutpath, folder, IWA, OWA, dMagLim,s0,dMag0,s1,dMag1,sigma_s,sigma_dmag)
pjf.plotJointPDF(sim, PPoutpath, folder)
"""
