# -*- coding: utf-8 -*-
""" Plot Convergence vs Number of Runs

Written by: Dean Keithly on 5/29/2018
Updated on: 3/4/2019
"""

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
import scipy.stats as st
import os
from EXOSIMS.util.vprint import vprint
import datetime
import re

class plotConvergencevsNumberofRuns(object):
    """Template format for adding singleRunPostProcessing to any plotting utility
    singleRunPostProcessing method is a required method with the below inputs to work with runPostProcessing.py
    """
    _modtype = 'util'

    def __init__(self, args=None):
        vprint(args)
        vprint('plotConvergencevsNumberofRuns done')
        pass

    def singleRunPostProcessing(self, PPoutpath=None, folder=None):
        """This is called by runPostProcessing
        Args:
            PPoutpath (string) - output path to place data in
            folder (string) - full filepath to folder containing runs
        """
        #runDir = '/home/dean/Documents/SIOSlab/EXOSIMSres/Dean22May18RS09CXXfZ01OB01PP01SU01/'


        #Given Filepath for pklfile, Plot a pkl from each testrun in subdir
        pklPaths = list()
        pklfname = list()


        #Look for all directories in specified path with structured folder name
        dirs = folder

        pklFiles = [myFileName for myFileName in os.listdir(dirs) if 'run' in myFileName and '.pkl' in myFileName]  # Get names of all pkl files in path
        for i in np.arange(len(pklFiles)):
            pklPaths.append(os.path.join(dirs,pklFiles[i]))  # append a random pkl file to path


        #Iterate over all pkl files
        meanNumDets = list()
        numDetsInSim = list()
        incVariance = list()
        ci90u = list()
        ci95u = list()
        ci99u = list()
        ci99p5u = list()
        ci90l = list()
        ci95l = list()
        ci99l = list()
        ci99p5l = list()
        ci1su = list()
        ci2su = list()
        ci3su = list()
        ci1sl = list()
        ci2sl = list()
        ci3sl = list()
        #CIpmlZ = list() # Incremental plus/minus to add to mean for confidence interval, CIpmlZ*Z = CI
        for cnt in np.arange(len(pklPaths)):
            try:
                with open(pklPaths[cnt], 'rb') as f:#load from cache
                    DRM = pickle.load(f)
            except:
                print('Failed to open pklfile %s'%pklPaths[cnt])
                pass

            #Calculate meanNumDets #raw detections, not unique detections
            AllDetsInPklFile = [(DRM['DRM'][i]['det_status'] == 1).tolist().count(True) for i in np.arange(len(DRM['DRM']))]
            meanNumDetsTMP = sum(AllDetsInPklFile)
            numDetsInSim.append(meanNumDetsTMP)

            if len(numDetsInSim) > 10:
                #was using st.t, now using st.chi2
                l1s, u1s = st.t.interval(0.6827, len(numDetsInSim)-1, loc=np.mean(numDetsInSim), scale=st.sem(numDetsInSim))
                l2s, u2s = st.t.interval(0.9545, len(numDetsInSim)-1, loc=np.mean(numDetsInSim), scale=st.sem(numDetsInSim))
                l3s, u3s = st.t.interval(0.9937, len(numDetsInSim)-1, loc=np.mean(numDetsInSim), scale=st.sem(numDetsInSim))
                l90, u90 = st.t.interval(0.90, len(numDetsInSim)-1, loc=np.mean(numDetsInSim), scale=st.sem(numDetsInSim))
                l95, u95 = st.t.interval(0.95, len(numDetsInSim)-1, loc=np.mean(numDetsInSim), scale=st.sem(numDetsInSim))
                l99, u99 = st.t.interval(0.99, len(numDetsInSim)-1, loc=np.mean(numDetsInSim), scale=st.sem(numDetsInSim))
                l995, u995 = st.t.interval(0.995, len(numDetsInSim)-1, loc=np.mean(numDetsInSim), scale=st.sem(numDetsInSim))
            else:
                u90, l90, u95, l95, u99, l99, u995, l995 = (0., 0., 0., 0., 0., 0., 0., 0.)
                l1s, u1s, l2s, u2s, l3s, u3s = (0., 0., 0., 0., 0., 0.)
            ci1su.append(u1s)
            ci2su.append(u2s)
            ci3su.append(u3s)
            ci1sl.append(l1s)
            ci2sl.append(l2s)
            ci3sl.append(l3s)
            ci90u.append(u90)
            ci95u.append(u95)
            ci99u.append(u99)
            ci99p5u.append(u995)
            ci90l.append(l90)
            ci95l.append(l95)
            ci99l.append(l99)
            ci99p5l.append(l995)
            
            #Append to list and incrementally update the new mean
            if cnt == 0:
                meanNumDets.append(float(meanNumDetsTMP))
            else:
                meanNumDets.append((meanNumDets[cnt-1]*float(cnt-1+1) + meanNumDetsTMP)/float(cnt+1))

            print("%d/%d  %d %f"%(cnt,len(pklPaths), meanNumDetsTMP, meanNumDets[cnt]))
        ci1su = np.asarray(ci1su)
        ci2su = np.asarray(ci2su)
        ci3su = np.asarray(ci3su)
        ci1sl = np.asarray(ci1sl)
        ci2sl = np.asarray(ci2sl)
        ci3sl = np.asarray(ci3sl)
        ci90u = np.asarray(ci90u)
        ci95u = np.asarray(ci95u)
        ci99u = np.asarray(ci99u)
        ci99p5u = np.asarray(ci99p5u)
        ci90l = np.asarray(ci90l)
        ci95l = np.asarray(ci95l)
        ci99l = np.asarray(ci99l)
        ci99p5l = np.asarray(ci99p5l)




        plt.close('all')
        fig = plt.figure(8000)
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rcParams['axes.linewidth']=2
        plt.rc('font',weight='bold') 
        #rcParams['axes.titlepad']=-50
        plt.plot(np.arange(len(meanNumDets))+1, abs(np.asarray(meanNumDets) - meanNumDets[-1]), color='purple', zorder=1, label='|error|')
        tmp1 = [np.abs(ci1su[i]-meanNumDets[i]) for i in np.arange(len(meanNumDets))]
        tmp2 = [np.abs(ci1sl[i]-meanNumDets[i]) for i in np.arange(len(meanNumDets))]
        errorCI1s = [np.max([tmp1[i],tmp2[i]]) for i in np.arange(len(meanNumDets)) if i > 30]
        tmp1 = [np.abs(ci2su[i]-meanNumDets[i]) for i in np.arange(len(meanNumDets))]
        tmp2 = [np.abs(ci2sl[i]-meanNumDets[i]) for i in np.arange(len(meanNumDets))]
        errorCI2s = [np.max([tmp1[i],tmp2[i]]) for i in np.arange(len(meanNumDets)) if i > 30]
        tmp1 = [np.abs(ci3su[i]-meanNumDets[i]) for i in np.arange(len(meanNumDets))]
        tmp2 = [np.abs(ci3sl[i]-meanNumDets[i]) for i in np.arange(len(meanNumDets))]
        errorCI3s = [np.max([tmp1[i],tmp2[i]]) for i in np.arange(len(meanNumDets)) if i > 30]
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], errorCI1s, linestyle=(0,(1,5)),color='black', label=r'1$\sigma$ CI')
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], errorCI2s, linestyle=(0,(5,10)),color='black', label=r'2$\sigma$ CI')
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], errorCI3s, linestyle='-',color='black', label=r'3$\sigma$ CI')

        plt.xscale('log')
        plt.yscale('log')

        plt.xlim([1,len(meanNumDets)])
        plt.ylim([0,np.max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))])
        plt.ylabel("Mean # of Detections Error\n$|\mu_{det_i}-\mu_{det_{10000}}|$", weight='bold')
        plt.xlabel("# of Simulations, i", weight='bold')
        plt.legend()
        plt.show(block=False)


        date = unicode(datetime.datetime.now())
        date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
        fname = 'meanNumDetectionDiffConvergenceLOG' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'))
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))


        plt.yscale('linear')
        DetsDeltas = abs(np.asarray(meanNumDets) - meanNumDets[-1])
        #ci1sel, ci1seu = st.t.interval(0.6827, len(numDetsInSim)-1, loc=np.mean(numDetsInSim), scale=st.sem(numDetsInSim)) # not sure this is the right approach for CI on error

        #http://circuit.ucsd.edu/~yhk/ece250-win17/pdfs/lect07.pdf
        #Example 7.7
        # varMU = np.var(numDetsInSim)
        # N90 = varMU/(1.-0.90)
        # N95 = varMU/(1.-0.95)
        # N99 = varMU/(1.-0.99)
        # N99p5 = varMU/(1.-0.995)

        # inds90 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x]/meanNumDets[-1] < (1.-0.90) and i > 50][0]
        # inds95 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x]/meanNumDets[-1] < (1.-0.95) and i > 50][0]
        # inds99 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x]/meanNumDets[-1] < (1.-0.99) and i > 50][0]
        # inds99p5 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x]/meanNumDets[-1] < (1.-0.995) and i > 50][0]


        #### Call out when we have reached within XX % of mu_10000 at 3sigma% confidence
        XX = 0.05 # within 1% of meanNumDets
        ind99p = np.where(errorCI3s < XX)[0]



        inds90 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x] < ci90u[-1] and i > 50][0]
        inds95 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x] < ci95u[-1] and i > 50][0]
        inds99 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x] < ci99u[-1] and i > 50][0]
        inds99p5 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x] < ci99p5u[-1] and i > 50][0]

        maxDetsDelta = np.max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))
        plt.plot([100,100],[abs(np.asarray(meanNumDets[99]) - meanNumDets[-1]),maxDetsDelta], linewidth=1, color='k')
        plt.text(90,1.3*maxDetsDelta,r"$\mu_{det_{100\ }}=$" + ' %2.1f'%(meanNumDets[99]/meanNumDets[-1]*100.) + '%', rotation=45)
        plt.plot([1000,1000],[abs(np.asarray(meanNumDets[999]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))], linewidth=1, color='k')
        plt.text(900,1.3*maxDetsDelta,r"$\mu_{det_{1000}}=$" + ' %2.1f'%(meanNumDets[999]/meanNumDets[-1]*100.) + '%', rotation=45)
        #plot([10000,10000],[abs(np.asarray(meanNumDets[-1]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))], linewidth=1, color='k')

        plt.plot([inds90,inds90],[abs(np.asarray(meanNumDets[inds90]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))],linestyle='--', linewidth=1, color='k')
        plt.text(inds90-12,1.3*maxDetsDelta,r"$\mu_{det_{82\ \ }}=$" + ' %2.1f'%(meanNumDets[inds90]/meanNumDets[-1]*100.) + '%', rotation=45)
        plt.plot([inds95,inds95],[abs(np.asarray(meanNumDets[inds95]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))],linestyle='--', linewidth=1, color='k')
        plt.text(inds95-12,1.3*maxDetsDelta,r"$\mu_{det_{132\ }}=$" + ' %2.1f'%(meanNumDets[inds95]/meanNumDets[-1]*100.) + '%', rotation=45)
        plt.plot([inds99,inds99],[abs(np.asarray(meanNumDets[inds99]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))],linestyle='--', linewidth=1, color='k')
        plt.text(inds99-12,1.3*maxDetsDelta,r"$\mu_{det_{363\ }}=$" + ' %2.1f'%(meanNumDets[inds99]/meanNumDets[-1]*100.) + '%', rotation=45)
        plt.plot([inds99p5,inds99p5],[abs(np.asarray(meanNumDets[inds99p5]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))],linestyle='--', linewidth=1, color='k')
        plt.text(inds99p5-12,1.3*maxDetsDelta,r"$\mu_{det_{1550}}=$" + ' %2.1f'%(meanNumDets[1555]/meanNumDets[-1]*100.) + '%', rotation=45)
        #gca().text(9000,4,r"$\mu_{det_{1000}}=$" + ' %2.1f'%(meanNumDets[999]/meanNumDets[-1]*100.))

        plt.xlim([1,len(meanNumDets)])
        plt.ylim([0,np.max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))])
        plt.ylabel("Mean # of Detections Error\n$|\mu_{det_i}-\mu_{det_{10000}}|$", weight='bold')
        plt.xlabel("# of Simulations, i", weight='bold')
        #tight_layout()
        #margins(1)
        plt.gcf().subplots_adjust(top=0.75, left=0.15)
        plt.show(block=False)

        fname = 'meanNumDetectionDiffConvergence' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'))
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))


        # Plot Convergence as a percentage of total
        fig = plt.figure(9000)
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rcParams['axes.linewidth']=2
        plt.rc('font',weight='bold') 
        plt.plot(np.arange(len(meanNumDets))+1, abs(np.asarray(meanNumDets) - meanNumDets[-1])/meanNumDets[-1]*100., color='purple', zorder=1, label='|error|')
        tmp1 = [np.abs(ci1su[i]-meanNumDets[i]) for i in np.arange(len(meanNumDets))]
        tmp2 = [np.abs(ci1sl[i]-meanNumDets[i]) for i in np.arange(len(meanNumDets))]
        errorCI1s = [np.max([tmp1[i],tmp2[i]]) for i in np.arange(len(meanNumDets)) if i > 30]
        tmp1 = [np.abs(ci2su[i]-meanNumDets[i]) for i in np.arange(len(meanNumDets))]
        tmp2 = [np.abs(ci2sl[i]-meanNumDets[i]) for i in np.arange(len(meanNumDets))]
        errorCI2s = [np.max([tmp1[i],tmp2[i]]) for i in np.arange(len(meanNumDets)) if i > 30]
        tmp1 = [np.abs(ci3su[i]-meanNumDets[i]) for i in np.arange(len(meanNumDets))]
        tmp2 = [np.abs(ci3sl[i]-meanNumDets[i]) for i in np.arange(len(meanNumDets))]
        errorCI3s = [np.max([tmp1[i],tmp2[i]]) for i in np.arange(len(meanNumDets)) if i > 30]
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], np.asarray(errorCI1s)/meanNumDets[-1]*100., linestyle=(0,(1,5)),color='black', label=r'1$\sigma$ CI')
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], np.asarray(errorCI2s)/meanNumDets[-1]*100., linestyle=(0,(5,10)),color='black', label=r'2$\sigma$ CI')
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], np.asarray(errorCI3s)/meanNumDets[-1]*100., linestyle='-',color='black', label=r'3$\sigma$ CI')

        plt.xscale('log')
        plt.yscale('log')

        plt.xlim([1,len(meanNumDets)])
        plt.ylim([0,np.max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))/meanNumDets[-1]*100.])
        plt.ylabel("Mean # of Detections Error\n$|\mu_{det_i}-\mu_{det_{10000}}|$", weight='bold')
        plt.xlabel("# of Simulations, i", weight='bold')
        plt.legend()
        plt.show(block=False)

        fname = 'meanNumDetectionDiffConvergenceLOGscalePERCENT' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'))
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))


        plt.yscale('linear')
        DetsDeltas = abs(np.asarray(meanNumDets) - meanNumDets[-1])
        #ci1sel, ci1seu = st.t.interval(0.6827, len(numDetsInSim)-1, loc=np.mean(numDetsInSim), scale=st.sem(numDetsInSim)) # not sure this is the right approach for CI on error

        #http://circuit.ucsd.edu/~yhk/ece250-win17/pdfs/lect07.pdf
        #Example 7.7
        # varMU = np.var(numDetsInSim)
        # N90 = varMU/(1.-0.90)
        # N95 = varMU/(1.-0.95)
        # N99 = varMU/(1.-0.99)
        # N99p5 = varMU/(1.-0.995)

        # inds90 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x]/meanNumDets[-1] < (1.-0.90) and i > 50][0]
        # inds95 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x]/meanNumDets[-1] < (1.-0.95) and i > 50][0]
        # inds99 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x]/meanNumDets[-1] < (1.-0.99) and i > 50][0]
        # inds99p5 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x]/meanNumDets[-1] < (1.-0.995) and i > 50][0]


        #### Call out when we have reached within XX % of mu_10000 at 3sigma% confidence
        XX = 0.05 # within 1% of meanNumDets
        ind99p = np.where(errorCI3s < XX)[0]



        inds90 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x] < ci90u[-1] and i > 50][0]
        inds95 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x] < ci95u[-1] and i > 50][0]
        inds99 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x] < ci99u[-1] and i > 50][0]
        inds99p5 = [x for x in np.arange(len(DetsDeltas)) if DetsDeltas[x] < ci99p5u[-1] and i > 50][0]

        maxDetsDelta = np.max(np.abs(np.asarray(meanNumDets) - meanNumDets[-1]))/meanNumDets[-1]*100.
        plt.plot([100,100],[np.abs(np.asarray(meanNumDets[99]) - meanNumDets[-1])/meanNumDets[-1]*100.,maxDetsDelta], linewidth=1, color='k')
        plt.text(90,1.3*maxDetsDelta,r"$\mu_{det_{100\ }}=$" + ' %2.1f'%(meanNumDets[99]/meanNumDets[-1]*100.) + '%', rotation=45)
        plt.plot([1000,1000],[np.abs(np.asarray(meanNumDets[999]) - meanNumDets[-1])/meanNumDets[-1]*100.,np.max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))/meanNumDets[-1]*100.], linewidth=1, color='k')
        plt.text(900,1.3*maxDetsDelta,r"$\mu_{det_{1000}}=$" + ' %2.1f'%(meanNumDets[999]/meanNumDets[-1]*100.) + '%', rotation=45)
        #plot([10000,10000],[abs(np.asarray(meanNumDets[-1]) - meanNumDets[-1]),max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))], linewidth=1, color='k')

        plt.plot([inds90,inds90],[np.abs(np.asarray(meanNumDets[inds90]) - meanNumDets[-1])/meanNumDets[-1]*100.,np.max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))/meanNumDets[-1]*100.],linestyle='--', linewidth=1, color='k')
        plt.text(inds90-12,1.3*maxDetsDelta,r"$\mu_{det_{82\ \ }}=$" + ' %2.1f'%(meanNumDets[inds90]/meanNumDets[-1]*100.) + '%', rotation=45)
        plt.plot([inds95,inds95],[np.abs(np.asarray(meanNumDets[inds95]) - meanNumDets[-1])/meanNumDets[-1]*100.,np.max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))/meanNumDets[-1]*100.],linestyle='--', linewidth=1, color='k')
        plt.text(inds95-12,1.3*maxDetsDelta,r"$\mu_{det_{132\ }}=$" + ' %2.1f'%(meanNumDets[inds95]/meanNumDets[-1]*100.) + '%', rotation=45)
        plt.plot([inds99,inds99],[np.abs(np.asarray(meanNumDets[inds99]) - meanNumDets[-1])/meanNumDets[-1]*100.,np.max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))/meanNumDets[-1]*100.],linestyle='--', linewidth=1, color='k')
        plt.text(inds99-12,1.3*maxDetsDelta,r"$\mu_{det_{363\ }}=$" + ' %2.1f'%(meanNumDets[inds99]/meanNumDets[-1]*100.) + '%', rotation=45)
        plt.plot([inds99p5,inds99p5],[np.abs(np.asarray(meanNumDets[inds99p5]) - meanNumDets[-1])/meanNumDets[-1]*100.,np.max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))/meanNumDets[-1]*100.],linestyle='--', linewidth=1, color='k')
        plt.text(inds99p5-12,1.3*maxDetsDelta,r"$\mu_{det_{1550}}=$" + ' %2.1f'%(meanNumDets[1555]/meanNumDets[-1]*100.) + '%', rotation=45)
        #gca().text(9000,4,r"$\mu_{det_{1000}}=$" + ' %2.1f'%(meanNumDets[999]/meanNumDets[-1]*100.))

        plt.xlim([1,len(meanNumDets)])
        plt.ylim([0,np.max(abs(np.asarray(meanNumDets) - meanNumDets[-1]))/meanNumDets[-1]*100.])
        plt.ylabel(r"Mean # of Detections Error in % of $\mu_{det_{10000}}$" + "\n" + r"$|\mu_{det_i}/\mu_{det_{10000}}-1| \times 100$", weight='bold')
        plt.xlabel("# of Simulations, i", weight='bold')
        #tight_layout()
        #margins(1)
        plt.gcf().subplots_adjust(top=0.75, left=0.15)
        plt.show(block=False)

        fname = 'meanNumDetectionDiffConvergencePERCENT' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'))
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))





        fig = plt.figure(8001)
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rcParams['axes.linewidth']=2
        plt.rc('font',weight='bold')
        plt.plot(np.arange(len(meanNumDets))+1, meanNumDets, color='purple', zorder=10)
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], [ci1su[i] for i in np.arange(len(meanNumDets)) if i > 30], linestyle=(0,(1,5)),color='black', label=r'1$\sigma$ CI')
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], [ci2su[i] for i in np.arange(len(meanNumDets)) if i > 30], linestyle=(0,(5,10)),color='black', label=r'2$\sigma$ CI')
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], [ci3su[i] for i in np.arange(len(meanNumDets)) if i > 30], linestyle='-',color='black', label=r'3$\sigma$ CI')
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], [ci1sl[i] for i in np.arange(len(meanNumDets)) if i > 30], linestyle=(0,(1,5)),color='black')
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], [ci2sl[i] for i in np.arange(len(meanNumDets)) if i > 30], linestyle=(0,(5,10)),color='black')
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], [ci3sl[i] for i in np.arange(len(meanNumDets)) if i > 30], linestyle='-',color='black')
        # plt.plot(np.arange(len(ci90u))+1, ci90u, linestyle=(0,(1,5)), color='black', label='90% CI')
        # plt.plot(np.arange(len(ci90u))+1, ci95u, linestyle=(0,(5,10)), color='black', label='95% CI')
        # plt.plot(np.arange(len(ci90u))+1, ci99u, linestyle=(0,(5,5)), color='black', label='99% CI')
        # plt.plot(np.arange(len(ci90u))+1, ci99p5u, linestyle=(0,(5,1)), color='black', label='99.5% CI')
        # plt.plot(np.arange(len(ci90u))+1, ci90l, linestyle=(0,(1,5)), color='black')
        # plt.plot(np.arange(len(ci90u))+1, ci95l, linestyle=(0,(5,10)), color='black')
        # plt.plot(np.arange(len(ci90u))+1, ci99l, linestyle=(0,(5,5)), color='black')
        # plt.plot(np.arange(len(ci90u))+1, ci99p5l, linestyle=(0,(5,1)), color='black')
        plt.plot([0.,len(np.asarray(meanNumDets))],[meanNumDets[-1],meanNumDets[-1]],linestyle='--',color='black', zorder=1)
        plt.xscale('log')
        plt.xlim([1,len(meanNumDets)])
        plt.ylim([0,np.max(ci99p5u)*1.05])
        plt.ylabel("Mean # of Unique Detections", weight='bold')
        plt.xlabel("# of Simulations, i", weight='bold')
        plt.legend()
        plt.show(block=False)

        fname = 'meanNumDetectionConvergence' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'))
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))


        fig = plt.figure(8002)
        #ax = fig.add_subplot(111)
        plt.rc('axes',linewidth=2)
        plt.rc('lines',linewidth=2)
        plt.rcParams['axes.linewidth']=2
        plt.rc('font',weight='bold') 
        plt.plot(np.arange(len(meanNumDets))+1, np.asarray(meanNumDets)/meanNumDets[-1]*100., color='purple', zorder=10)
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], [(ci1su/meanNumDets[-1]*100.)[i] for i in np.arange(len(meanNumDets)) if i > 30], linestyle=(0,(1,5)),color='black', label=r'1$\sigma$ CI')
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], [(ci2su/meanNumDets[-1]*100.)[i] for i in np.arange(len(meanNumDets)) if i > 30], linestyle=(0,(5,10)),color='black', label=r'2$\sigma$ CI')
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], [(ci3su/meanNumDets[-1]*100.)[i] for i in np.arange(len(meanNumDets)) if i > 30], linestyle='-',color='black', label=r'3$\sigma$ CI')
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], [(ci1sl/meanNumDets[-1]*100.)[i] for i in np.arange(len(meanNumDets)) if i > 30], linestyle=(0,(1,5)),color='black')
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], [(ci2sl/meanNumDets[-1]*100.)[i] for i in np.arange(len(meanNumDets)) if i > 30], linestyle=(0,(5,10)),color='black')
        plt.plot([i for i in np.arange(len(meanNumDets)-1)+1 if i > 30], [(ci3sl/meanNumDets[-1]*100.)[i] for i in np.arange(len(meanNumDets)) if i > 30], linestyle='-',color='black')
        # plt.plot(np.arange(len(meanNumDets))+1, ci90u/meanNumDets[-1]*100., linestyle=(0,(1,5)),color='black', label='90% CI')
        # plt.plot(np.arange(len(meanNumDets))+1, ci95u/meanNumDets[-1]*100., linestyle=(0,(5,10)),color='black', label='95% CI')
        # plt.plot(np.arange(len(meanNumDets))+1, ci99u/meanNumDets[-1]*100., linestyle=(0,(5,5)),color='black', label='99% CI')
        # plt.plot(np.arange(len(meanNumDets))+1, ci99p5u/meanNumDets[-1]*100., linestyle=(0,(5,1)),color='black', label='99.5% CI')
        # plt.plot(np.arange(len(meanNumDets))+1, ci90l/meanNumDets[-1]*100., linestyle=(0,(1,5)),color='black')
        # plt.plot(np.arange(len(meanNumDets))+1, ci95l/meanNumDets[-1]*100., linestyle=(0,(5,10)),color='black')
        # plt.plot(np.arange(len(meanNumDets))+1, ci99l/meanNumDets[-1]*100., linestyle=(0,(5,5)),color='black')
        # plt.plot(np.arange(len(meanNumDets))+1, ci99p5l/meanNumDets[-1]*100., linestyle=(0,(5,1)),color='black')
        plt.plot([0.,len(np.asarray(meanNumDets))],[100.,100.],linestyle='--',color='black', zorder=1)
        plt.xscale('log')
        plt.xlim([1,1e4])
        plt.ylim([0,np.max(ci99p5u/meanNumDets[-1]*100.)*1.05])
        plt.ylabel(r"Percentage of $\mu_{det_{10000}}$, $\frac{\mu_{det_i}}{\mu_{det_{10000}}} \times 100$", weight='bold')
        plt.xlabel("# of Simulations, i", weight='bold')
        plt.legend()
        plt.show(block=False)

        fname = 'percentErrorFromMeanConvergence' + folder.split('/')[-1] + '_' + date
        plt.savefig(os.path.join(PPoutpath, fname + '.png'))
        plt.savefig(os.path.join(PPoutpath, fname + '.svg'))
        plt.savefig(os.path.join(PPoutpath, fname + '.eps'))
        plt.savefig(os.path.join(PPoutpath, fname + '.pdf'))


        # $\mu_1000=9.82700$ which is 99.27\% of the $\mu_10000$.
        # $\mu_10000=9.898799$.
        # $\mu_100=9.100$ which is 91.9\% of the $\mu_10000$. 
        # 90\% of the $\mu_10000$ is achieved at 82sims.
        # 95\% of the $\mu_10000$ is achieved at 132sims.
        # 99\% of the $\mu_10000$ is achieved at 363sims.
        # 99.5\% of the $\mu_10000$ is achieved at 1550sims.


        #### Calculate sigma at 1000
        FirstsAt1000 = (np.asarray(errorCI1s)/meanNumDets[-1]*100.)[1000-30]
        SecondsAt1000 = (np.asarray(errorCI2s)/meanNumDets[-1]*100.)[1000-30]
        ThirdsAt1000 = (np.asarray(errorCI3s)/meanNumDets[-1]*100.)[1000-30]

        #### Calculate sigma at 100
        FirstsAt100 = (np.asarray(errorCI1s)/meanNumDets[-1]*100.)[100-30]
        SecondsAt100 = (np.asarray(errorCI2s)/meanNumDets[-1]*100.)[100-30]
        ThirdsAt100 = (np.asarray(errorCI3s)/meanNumDets[-1]*100.)[100-30]

        lines = list()
        lines.append('Data from folder: ' + folder)
        lines.append('Data created on: ' + date)
        lines.append('1sigma CI at 1000 sims: ' + str(FirstsAt1000))
        lines.append('2sigma CI at 1000 sims: ' + str(SecondsAt1000))
        lines.append('3sigma CI at 1000 sims: ' + str(ThirdsAt1000))
        lines.append('1sigma CI at 100 sims: ' + str(FirstsAt100))
        lines.append('2sigma CI at 100 sims: ' + str(SecondsAt100))
        lines.append('3sigma CI at 100 sims: ' + str(ThirdsAt100))

        #### Save Data File
        fname = 'convergenceDATA_' + folder.split('/')[-1] + '_' + date
        with open(os.path.join(PPoutpath, fname + '.txt'), 'w') as g:
            g.write("\n".join(lines))
