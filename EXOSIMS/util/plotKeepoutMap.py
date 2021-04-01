# -*- coding: utf-8 -*-
"""
Plotting utility for the production of Keepout Map Related Products
Generalized from makeKeepoutMap.py (authored by Gabriel Soto)
Written by: Dean Keithly
Written on: 3/6/2019
"""

import os
from EXOSIMS.util.vprint import vprint
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
from matplotlib import colors
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

import astropy.units as u
from astropy.time import Time
import time


class plotKeepoutMap(object):
    """ This plotting utility plots anything pertaining to keepout maps
    """
    _modtype = 'util'

    def __init__(self, args=None):
        vprint(args)
        vprint('initialize plotKeepoutMap done')
        pass

    def singleRunPostProcessing(self, PPoutpath=None, folder=None):
        """This is called by runPostProcessing
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



        #Create Mission Object To Extract Some Plotting Limits
        sim = EXOSIMS.MissionSim.MissionSim(outspecfile, nopar=True)
        obs = sim.Observatory
        TL  = sim.TargetList   #target list 
        missionStart = sim.TimeKeeping.missionStart  #Time Object
        TK = sim.TimeKeeping

        ##########################################################################################

        #### Generate Keepout map #array of Target List star indeces
        N = np.arange(0,TL.nStars)

        #Generate Keepout over Time
        koEvaltimes = np.arange(TK.missionStart.value, TK.missionStart.value+TK.missionLife.to('day').value,1) #2year mission, I guess
        koEvaltimes = Time(koEvaltimes,format='mjd')

        #initial arrays
        koGood  = np.zeros([TL.nStars,len(koEvaltimes)])      #keeps track of when a star is in keepout or not (True = observable)
        culprit = np.zeros([TL.nStars,len(koEvaltimes),11])   #keeps track of whose keepout the star is under

        #calculating keepout angles for all stars
        tic = time.clock()
        for n in np.arange(TL.nStars):
            koGood[n,:],r_body, r_targ, culprit[n,:,:], koangles = obs.keepout(TL,n,koEvaltimes,True)
        toc = time.clock()

        print('This took %s seconds' %(str(toc-tic)))



        # Define Colors
        #green:#00802b
        #purplish:7F7FFF
        #crap taupe:DEDE7F
        #GOLD: FFD500
        #GREY:747783
        cmap = colors.ListedColormap(['white','#FFD500', 'blue', '#747783','red','m','red']) #colors for used to indicate a culprit behind keepout
        bounds=[0,1,2,3,4,5,6,7]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        #creating an array of colors based on culprit
        koColor = np.zeros([TL.nStars,len(koEvaltimes)])
        for t in np.arange(0,len(koEvaltimes)):
            sunFault   = [bool(culprit[x,t,0]) for x in np.arange(TL.nStars)]
            earthFault = [bool(culprit[x,t,2]) for x in np.arange(TL.nStars)]
            moonFault  = [bool(culprit[x,t,1]) for x in np.arange(TL.nStars)]
            mercFault  = [bool(culprit[x,t,3]) for x in np.arange(TL.nStars)]
            venFault   = [bool(culprit[x,t,4]) for x in np.arange(TL.nStars)]
            marsFault  = [bool(culprit[x,t,5]) for x in np.arange(TL.nStars)]
            
            koColor[marsFault ,t] = 4#red
            koColor[venFault  ,t] = 5#m
            koColor[mercFault ,t] = 6#red
            koColor[moonFault ,t] = 3#747783
            koColor[earthFault,t] = 2#blue
            koColor[sunFault  ,t] = 1#FFD500


        #plotting colors on a 2d map
        plt.close(546832183)
        fig = plt.figure(546832183, figsize=(10,5))
        fig.subplots_adjust(bottom=0.15)
        gs = gridspec.GridSpec(1,2, width_ratios=[6,1], height_ratios=[1])
        gs.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rcParams['axes.linewidth']=2
        plt.rc('font',weight='bold') 
        ax2 = plt.subplot(gs[1])
        ax = plt.subplot(gs[0])


        #I'm plotting a subset of koColor here that looked good to me for half the mission time (1yr)
        if koColor.shape[0] > 100: #Determine maximum number of stars to track keepouts for
            NUMBER_Y = 100
        else:
            NUMBER_Y = koColor.shape[0]
        sInds = np.linspace(0, koColor.shape[0], num=NUMBER_Y, endpoint=False, dtype=int).tolist()
        img = plt.imshow(koColor[sInds,0:int(np.floor(len(koEvaltimes)))], aspect='auto',#4,
                            cmap=cmap,interpolation='none',origin='lower',norm=norm)

        ax.set_xlabel('Mission Elapsed Time (d), Mission Start %s UTC MJD' %(str(TK.missionStart.value)), weight='bold')
        ax.set_ylabel(r'Target Star, $i$', weight='bold')
        ax.set_xlim(left=0.,right=np.max(koEvaltimes).value-TK.missionStart.value)
        ax.set_ylim(bottom=0.,top=NUMBER_Y)

        outline=PathEffects.withStroke(linewidth=5, foreground='black')
        plt.plot([-1.,-1.],[-1.,-1.],color=cmap.colors[0],label='Visible',path_effects=[outline])
        plt.plot([-1.,-1.],[-1.,-1.],color=cmap.colors[1],label=ur"$\u2609$")
        plt.plot([-1.,-1.],[-1.,-1.],color=cmap.colors[2],label=ur'$\oplus$')##\u2641$')
        plt.plot([-1.,-1.],[-1.,-1.],color=cmap.colors[3],label=ur'$\u263D$')
        plt.plot([-1.,-1.],[-1.,-1.],color=cmap.colors[4],label=ur'$\u2642\u263F$')
        plt.plot([-1.,-1.],[-1.,-1.],color=cmap.colors[5],label=ur'$\u2640$')
        #plt.plot([-1.,-1.],[-1.,-1.],color=cmap.colors[6],label=ur'$\u263F$')  duplicate color so appended above
        leg = plt.legend(framealpha=1.0)
        # get the lines and texts inside legend box
        leg_lines = leg.get_lines()
        leg_texts = leg.get_texts()
        # bulk-set the properties of all lines and texts
        plt.setp(leg_lines, linewidth=4)
        plt.setp(leg_texts, fontsize='x-large')
        nullfmt = NullFormatter()
        ax2.yaxis.set_major_formatter(nullfmt)

        #Plot horizontal histogram
        tTotal = np.max(koEvaltimes).value-TK.missionStart.value # Calculate total time width of koMap
        tVis = list() # stores time visible of each star
        for i in np.arange(len(sInds)):#iterate over all stars and append amount of time each star is visible
            tVis.append(len(np.where(koColor[sInds[i],:]==0)[0]))
         
        width = np.zeros(len(tVis))+1.
        ax2.barh(np.arange(len(sInds))+0.5,np.asarray(tVis,dtype=float)/tTotal*100., width, align='center', color='black')
        ax2.set_xlim(left=0.,right=100.)
        ax2.set_ylim(bottom=0.,top=NUMBER_Y)
        ax2.set_xlabel('% Time\n Visible', weight='bold')
        plt.show(block=False)   

        date = unicode(datetime.datetime.now())
        date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
        fname = 'koMap_' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'))
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))


        #### Plot a koMap scaled down to 1 year
        if TK.missionLife.to('year').value > 1.0:# years
            plt.close(56846512161)
            fig = plt.figure(56846512161)
            fig.subplots_adjust(bottom=0.15)
            gs = gridspec.GridSpec(1,2, width_ratios=[6,1], height_ratios=[1])
            gs.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 
            plt.rc('axes',linewidth=2)
            plt.rc('lines',linewidth=2)
            plt.rcParams['axes.linewidth']=2
            plt.rc('font',weight='bold') 
            ax2 = plt.subplot(gs[1])
            ax = plt.subplot(gs[0])

            #I'm plotting a subset of koColor here that looked good to me for half the mission time (1yr)
            if koColor.shape[0] > 100: #Determine maximum number of stars to track keepouts for
                NUMBER_Y = 100
            else:
                NUMBER_Y = koColor.shape[0]
            sInds = np.linspace(0, koColor.shape[0], num=NUMBER_Y, endpoint=False, dtype=int).tolist()
            img = plt.imshow(koColor[sInds,0:365], aspect='auto',#4,
                                cmap=cmap,interpolation='none',origin='lower',norm=norm)

            ax.set_xlabel('Mission Elapsed Time (d)\nMission Start %s UTC MJD' %(str(TK.missionStart.value)), weight='bold')
            ax.set_ylabel(r'Target Star, $i$', weight='bold')
            ax.set_xlim(left=0.,right=365.)
            ax.set_ylim(bottom=0.,top=NUMBER_Y)

            outline=PathEffects.withStroke(linewidth=5, foreground='black')
            plt.plot([-1.,-1.],[-1.,-1.],color=cmap.colors[0],label='Visible',path_effects=[outline])
            plt.plot([-1.,-1.],[-1.,-1.],color=cmap.colors[1],label=ur'$\u2609$')
            plt.plot([-1.,-1.],[-1.,-1.],color=cmap.colors[2],label=ur'$\oplus$')#\u2641$')
            plt.plot([-1.,-1.],[-1.,-1.],color=cmap.colors[3],label=ur'$\u263D$')
            plt.plot([-1.,-1.],[-1.,-1.],color=cmap.colors[4],label=ur'$\u2642\u263F$')
            plt.plot([-1.,-1.],[-1.,-1.],color=cmap.colors[5],label=ur'$\u2640$')
            #plt.plot([-1.,-1.],[-1.,-1.],color=cmap.colors[6],label=ur'$\u263F$')
            leg = plt.legend(framealpha=1.0)
            # get the lines and texts inside legend box
            leg_lines = leg.get_lines()
            leg_texts = leg.get_texts()
            # bulk-set the properties of all lines and texts
            plt.setp(leg_lines, linewidth=4)
            plt.setp(leg_texts, fontsize='x-large')
            nullfmt = NullFormatter()
            ax2.yaxis.set_major_formatter(nullfmt)

            #Plot horizontal histogram
            tTotal = np.max(koEvaltimes).value-TK.missionStart.value # Calculate total time width of koMap
            tVis = list() # stores time visible of each star
            for i in np.arange(len(sInds)):#iterate over all stars and append amount of time each star is visible
                tVis.append(len(np.where(koColor[sInds[i],:]==0)[0]))
             
            width = np.zeros(len(tVis))+1.
            ax2.barh(np.arange(len(sInds))+0.5,np.asarray(tVis,dtype=float)/tTotal*100., width, align='center', color='black')
            ax2.set_xlim(left=0.,right=100.)
            ax2.set_ylim(bottom=0.,top=NUMBER_Y)
            ax2.set_xlabel('% Time\n Visible', weight='bold')
            plt.show(block=False) 

            date = unicode(datetime.datetime.now())
            date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
            fname = 'koMapScaled_' + folder.split('/')[-1] + '_' + date
            plt.savefig(os.path.join(PPoutpath, fname + '.png'))
            plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
            plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
            plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))


            #### Plot a Histogram of Percent Time Visible 10 bins
            plt.close(65685621)
            fig = plt.figure(65685621)
            bins = np.linspace(start=0,stop=100,num=11)
            plt.hist(np.asarray(tVis)/tTotal*100., bins=bins, color='black', alpha=1., histtype='bar', ec='black')
            plt.ylabel('Target Count', weight='bold')
            plt.xlabel('Time Visible (%)', weight='bold')
            plt.xlim((0,100))
            plt.show(block=False)

            date = unicode(datetime.datetime.now())
            date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
            fname = 'koMapHist10_' + folder.split('/')[-1] + '_' + date
            plt.savefig(os.path.join(PPoutpath, fname + '.png'))
            plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
            plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
            plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))


            #### Plot as Histogram of Percent Time Visible Many Bins
            plt.close(98735654)
            fig = plt.figure(98735654)
            bins = np.linspace(start=0,stop=np.round(np.max(tVis)/tTotal*100.),num=np.round(np.max(tVis)/tTotal*100.)+1)
            plt.hist(np.asarray(tVis)/tTotal*100., bins=bins, color='black', alpha=1., histtype='bar', ec='black')
            plt.ylabel('Target Count', weight='bold')
            plt.xlabel('Time Visible (%)', weight='bold')
            plt.xlim((0,np.ceil(np.max(tVis)/tTotal*100.)))
            plt.show(block=False)

            date = unicode(datetime.datetime.now())
            date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
            fname = 'koMapHistDetail_' + folder.split('/')[-1] + '_' + date
            plt.savefig(os.path.join(PPoutpath, fname + '.png'))
            plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
            plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
            plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))


        #### Plot Hist and CDF of % time ######################################
        tVis2 = list() # stores time visible of each star
        for i in np.arange(len(sInds)):#iterate over all stars and append amount of time each star is visible
            tVis2.append(len(np.where(koColor[i,:]==0)[0]))
            if tVis2[-1] > tTotal:
                tVis2[-1] = tTotal
        bins = np.linspace(start=0,stop=np.round(np.max(tVis2)/tTotal*100.),num=np.round(np.max(tVis2)/tTotal*100.)+1)
        n, bins, patches = plt.figure(665465461286584).add_subplot(1,1,1).hist(np.asarray(tVis2)/tTotal*100., bins=bins)
        plt.show(block=False)
        plt.close(665465461286584) # doing this just to descroy above plot Replace with numpy.histogram in future
        cdf = np.cumsum(n)#cumtrapz(n, bins[:-1], initial=0.)
        cdf = cdf/np.max(cdf)

        plt.close(23623)
        fig2 = plt.figure(23623)
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rcParams['axes.linewidth']=2
        plt.rc('font',weight='bold')
        ax2 = fig2.add_subplot(1,1,1)
        bins = np.linspace(start=0,stop=np.round(np.max(tVis2)/tTotal*100.),num=np.round(np.max(tVis2)/tTotal*100.)+1)
        n2, bins2, patches2 = ax2.hist(np.asarray(tVis2)/tTotal*100.,zorder=8,color='black', bins=bins[1:])
        ax2.set_xlabel('Percent Time Visible (%)', weight='bold')
        ax3 = ax2.twinx()
        ax3.plot(bins[:-1],cdf*100.,zorder=10, color='red')
        #DELETEax2.spines['right'].set_color('red') # setting the right side axis to red
        ax3.spines['right'].set_color('red') # setting the right side axis to red
        ax3.xaxis.label.set_color('red')
        ax3.tick_params(axis='y', colors='red')
        ax2.set_ylabel('Target Count', weight='bold')
        ax3.set_ylabel('CDF (%)', weight='bold', color='red')
        ax2.set_xlim(left=0.,right=100.)
        ax2.set_ylim(bottom=0.,top=1.1*np.max(n2))
        ax3.set_ylim(bottom=0.,top=100.)
        plt.show(block=False)

        fname = 'koMapHIST_CDF_' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'))
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))
        ###########################################################
