"""
Plot Planet Population Radius vs a AND Detected Planet Rp vs a
Plot will be shown and saved to the directory specified by runPath

#Call this function by 
python PlotPlanetPopRvsAandDetectedRvsA.py --runPath '/dir/containing/pkl/files/'

#A specific example calling this function
python PlotPlanetPopRvsAandDetectedRvsA.py --runPath '/home/dean/Documents/SIOSlab/Dean2May18RS12CXXfZ01OB01PP01SU01/'
Where '/home/dean/Documents/SIOSlab/Dean2May18RS12CXXfZ01OB01PP01SU01/' contains 1000 pkl files from a simulation run
%run PlotPlanetPopRvsAandDetectedRvsA.py --runPath '/home/dean/Documents/SIOSlab/Dean19May18RS09CXXfZ01OB54PP01SU01/'
python PlotPlanetPopRvsAandDetectedRvsA.py --runPath '/home/dean/Documents/SIOSlab/Dean21May18RS09CXXfZ01OB54PP03SU03/' #Baseline SAG13 Case
python PlotPlanetPopRvsAandDetectedRvsA.py --runPath '/home/dean/Documents/SIOSlab/Dean19May18RS09CXXfZ01OB54PP01SU01/' #Baseline KeplerLike2 Case
python PlotPlanetPopRvsAandDetectedRvsA.py --runPath '/home/dean/Documents/SIOSlab/Dean25May18RS09CXXfZ01OB56PP06SU01/' #SAG13 CompSpecs, PP-KeplerLike2 Case
python PlotPlanetPopRvsAandDetectedRvsA.py --runPath '/home/dean/Documents/SIOSlab/Dean25May18RS09CXXfZ01OB56PP07SU03/' #KeplerLike2 CompSpecs, PP-SAG13

Written by Dean Keithly on 5/6/2018
"""

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
from EXOSIMS.util.read_ipcluster_ensemble import read_all
from numpy import linspace
from matplotlib.ticker import NullFormatter, MaxNLocator
import matplotlib.pyplot as plt
from matplotlib import ticker
import astropy.units as u
import matplotlib.patheffects as PathEffects

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
    allres = read_all(runPath)# contains all drm from all missions in runPath

    Rpunits = allres[0]['systems']['Rp'].unit
    allres_Rp = np.concatenate([allres[i]['systems']['Rp'].value for i in range(len(allres))])
    smaunits = allres[0]['systems']['a'].unit
    allres_sma = np.concatenate([allres[i]['systems']['a'].value for i in range(len(allres))])
    x = allres_sma
    y = allres_Rp

    # Define the x and y data for detected planets
    det_Rps = np.concatenate(out['Rps']).ravel() # Planet Radius in Earth Radius of detected planets
    det_smas = np.concatenate(out['smas']).ravel()


    #Create Mission Object To Extract Some Plotting Limits
    outspecfile = runPath + 'outspec.json'
    sim = EXOSIMS.MissionSim.MissionSim(outspecfile, nopar=True)
    ymax = np.nanmax(sim.PlanetPhysicalModel.ggdat['radii']).to('earthRad').value


    ################ 
    #Create Figure and define gridspec
    fig2 = figure(2, figsize=(8.5,4.5))
    gs = GridSpec(2,5, width_ratios=[4,1,0.3,4,1.25], height_ratios=[1,4])
    gs.update(wspace=0.06, hspace=0.06) # set the spacing between axes. 
    plt.rc('axes',linewidth=2)
    plt.rc('lines',linewidth=2)
    #plt.rc('axes',prop_cycle=(cycler('color',['red','purple'])))#,'blue','black','purple'])))
    rcParams['axes.linewidth']=2
    rc('font',weight='bold')

    #What the plot layout looks like
    ###----------------------------
    # | gs[0]  gs[1]  gs[2]  gs[3] |
    # | gs[4]  gs[5]  gs[6]  gs[7] |
    ###----------------------------
    # ax1 = plt.subplot(gs[4])#2D histogram of planet pop
    # ax2 = plt.subplot(gs[0])#1D histogram of a
    # ax3 = plt.subplot(gs[5])#1D histogram of Rp
    # ax4 = plt.subplot(gs[6])#2D histogram of detected Planet Population
    # ax5 = plt.subplot(gs[2])#1D histogram of detected planet a
    # ax6 = plt.subplot(gs[7])#1D histogram of detected planet Rp
    ax1 = plt.subplot(gs[5])#2D histogram of planet pop
    ax2 = plt.subplot(gs[0])#1D histogram of a
    ax3 = plt.subplot(gs[6])#1D histogram of Rp
    ax4 = plt.subplot(gs[8])#2D histogram of detected Planet Population
    ax5 = plt.subplot(gs[3])#1D histogram of detected planet a
    ax6 = plt.subplot(gs[9])#1D histogram of detected planet Rp
    TXT1 = plt.subplot(gs[1])
    TXT4 = plt.subplot(gs[4])

    # Set up default x and y limits
    xlims = [min(x),max(x)]# of aPOP
    ylims = [min(y),max(y)]# of RpPOP
    # Find the min/max of the POP data and APPLY LIMITS
    xmin = 0.1#min(xlims)#min of a
    xmax = 30#max(xlims)#max of a
    ymin = 1#min(ylims)#min of Rp
    #ymax = 22.6#max(y)#max of Rp
    xlims = [xmin, xmax]#sma range
    ylims = [ymin, ymax]#Rp range

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
     
    H, xedges,yedges = np.histogram2d(x,y,bins=(xbins,ybins),normed=True)
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
    # cax = (ax1.imshow(H, extent=[xmin,xmax,ymin,ymax],
    #     interpolation='nearest', origin='lower',aspect="auto"))#aspectratio))
    xcents = np.diff(xbins)/2.+xbins[:-1]
    ycents = np.diff(ybins)/2.+ybins[:-1]
    #levels = np.logspace(start = np.log10(1e-8), stop = np.log10(np.amax(H.T/float(len(x)))), num = 8)#[1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3, 3.125e-3, 1.5625e-3, 7.1825e-4]# 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    #/len(out['smas'])
    cax = ax1.contourf(xcents, ycents, H.T, extent=[xmin, xmax, ymin, ymax], cmap='jet', intepolation='nearest',locator=ticker.LogLocator())#, 
    # fmt = ticker.LogFormatterMathtext()
    # plt.clabel(cax, fmt=fmt, colors='k', fontsize=8, inline=1)#OK
    # levels = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    
    CS4 = ax1.contour(cax, colors=('k',), linewidths=(1,), origin='lower', locator=ticker.LogLocator())
    colorbar_ax = fig2.add_axes([0.125, 0.775, 0.775, 0.025])#[left, bottom, width, height]
    cbar = fig2.colorbar(cax, cax=colorbar_ax, orientation='horizontal')#pad=0.05,
    rcParams['axes.titlepad']=-10
    #plt.text(0.5, 1, 'Normalized Planet Joint Probability Density', weight='bold', horizontalalignment='center')
    #cbar.ax.set_title(label='Normalized Planet Joint Probability Density', weight='bold')
    cbar.ax.set_xlabel('Joint Probability Density: Universe (Left) Detected Planets (Right)', weight='bold', labelpad=-35)
    #cbar.set_label('Normalized Planet Joint Probability Density', weight='bold')#, rotation=270)
    cbar.ax.tick_params(axis='x',direction='in',labeltop='on',labelbottom='off')
    cbar.add_lines(CS4)
    


    HDET, xedgesDET, yedgesDET = np.histogram2d(det_smas,det_Rps,bins=(xbins,ybins),normed=True)
    caxDET = ax4.contourf(xcents,ycents,HDET.T, extent=[xmin, xmax, ymin, ymax], cmap='jet', intepolation='nearest',locator=ticker.LogLocator())
    # fmt2 = ticker.LogFormatterMathtext()
    # plt.clabel(caxDET, fmt=fmt2, colors='k', fontsize=8, inline=1)#OK
    #levels2 = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    CS42 = ax4.contour(caxDET, colors=('k',), linewidths=(1,), origin='lower', locator=ticker.LogLocator())
    #colorbar_ax = fig2.add_axes([0.125, 0.9, 0.775, 0.025])#[left, bottom, width, height]
    #cbar = fig2.colorbar(caxDET, cax=colorbar_ax, orientation='horizontal')#pad=0.05,
    #cbar.ax.tick_params(axis='x',direction='in',labeltop='on',labelbottom='off')
    #cbar.add_lines(CS4)

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
    #*note len(out) should equal len(all_res)
    #ax2
    n2, bins2, patches2 = plt.subplot(gs[5]).hist(x, bins=xbins, color = 'black', fill='black', histtype='step',normed=True)#, hatch='-/')#1D histogram of universe a
    center2 = (bins2[:-1] + bins2[1:]) / 2
    width2=np.diff(bins2)
    ax2.bar(center2, n2*(len(x)/float(len(out['smas']))), align='center', width=width2, color='black', fill='black')
    #ax5
    n5, bins5, patches5 = plt.subplot(gs[5]).hist(det_smas, bins=xbins, color = 'black', fill='black', histtype='step',normed=True)#, hatch='+x')#1D histogram of detected planet a
    center5 = (bins5[:-1] + bins5[1:]) / 2
    width5=np.diff(bins5)
    ax5.bar(center5, n5*(len(det_smas)/float(len(out['smas']))), align='center', width=width5, color='black', fill='black')
    #ax3
    n3, bins3, patches3 = plt.subplot(gs[5]).hist(y, bins=ybins, color = 'black', fill='black', histtype='step',normed=True)#, hatch='+x')#1D histogram of detected planet a
    center3 = (bins3[:-1] + bins3[1:]) / 2
    width3=np.diff(bins3)
    ax3.barh(center3, n3*(len(y)/float(len(out['Rps']))), width3, align='center', color='black')
    #ax3.barh(y=center3, width=width3, height=n3*(len(det_Rps)/float(len(out['Rps']))), align='center', color='black')#, fill='black')#, orientation='horizontal')
    #ax3.hist(y, bins=ybins, orientation='horizontal', color = 'black', fill='black', histtype='step')#, hatch='///')#1D histogram of detected planet a
    #ax6
    n6, bins6, patches6 = plt.subplot(gs[5]).hist(det_Rps, bins=ybins, color = 'black', fill='black', histtype='step',normed=True)#, hatch='+x')#1D histogram of detected planet a
    center6 = (bins6[:-1] + bins6[1:]) / 2
    width6=np.diff(bins6)
    ax6.barh(center6, n6*(len(det_Rps)/float(len(out['Rps']))), width6, align='center', color='black')#, fill='black')#, orientation='horizontal')
    #ax6.hist(det_Rps, bins=ybins, orientation='horizontal', color = 'black', fill='black', histtype='step')#, hatch='x')#1D histogram of detected planet Rp
    ax2.set_ylabel(r'$\frac{a\ Freq.}{1000\ sims}$',weight='bold', multialignment='center')
    ax3.set_xlabel(r'$\frac{R_P\ Freq.}{1000\ sims}$',weight='bold', multialignment='center')
    ax6.set_xlabel(r'$\frac{R_P\ Freq.}{1000\ sims}$',weight='bold', multialignment='center')

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
    #(horizontalalignment="right")
    #ax4.set_xticklabels(multialignment='right')

    # Remove the inner axes numbers of the histograms
    nullfmt = NullFormatter()
    ax2.xaxis.set_major_formatter(nullfmt)
    ax3.yaxis.set_major_formatter(nullfmt)
    ax4.yaxis.set_major_formatter(nullfmt)
    ax5.xaxis.set_major_formatter(nullfmt)
    ax6.yaxis.set_major_formatter(nullfmt)
    
    #plot the detected planet Rp and a #make this a separate plot... or the same plot..... yess lets use subplots
    #ax1.scatter(det_smas, det_Rps, marker='o', color='red', alpha=0.1)
    plt.gcf().subplots_adjust(bottom=0.15, top=0.75)
    #plt.tight_layout({"pad":.0})
    #plt.axis('tight')
    
    #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    rc('axes', linewidth=0)
    TXT1.text(0.5, 0.4, '# Universe\nPlanets:\n%s'%("{:,}".format(len(x))), weight='bold', horizontalalignment='center', fontsize=8)
    TXT4.text(0.5, 0.0, '# Detected\nPlanets:\n%s'%("{:,}".format(len(det_Rps))), weight='bold', horizontalalignment='center', fontsize=8)
    TXT1.text(0.5, -0.1, '# Sims\n%s'%("{:,}".format(len(out['Rps']))), weight='bold', horizontalalignment='center', fontsize=8)
    #plt.subplot(gs[1]).axhline(linewidth=0, color="k",alpha=0)
    setp(TXT1.spines.values(),linewidth=0)
    setp(TXT4.spines.values(),linewidth=0)
    TXT1.xaxis.set_visible(False)
    TXT1.yaxis.set_visible(False)
    TXT4.xaxis.set_visible(False)
    TXT4.yaxis.set_visible(False)
    #plt.subplot(gs[4]).axhline(linewidth=0, color="k",alpha=0)

    show(block=False)
    # Save to a File
    filename = 'RpvsSMAdetections'
    savefig(runPath + os.path.basename(os.path.normpath(runPath)) + filename + '.png', format='png', dpi=500)
    savefig(runPath + os.path.basename(os.path.normpath(runPath)) + filename + '.svg')
    savefig(runPath + os.path.basename(os.path.normpath(runPath)) + filename + '.eps', format='eps', dpi=500)


    #Apply Grid to Detected Planet Pop
    #create coarse grid and calculate total numbers in each bin
    acoarse1 = np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),6)
    Rcoarse1 = np.logspace(np.log10(ylims[0]),np.log10(ylims[1]),6)

    hcoarse1 = np.histogram2d(np.hstack(det_smas), np.hstack(det_Rps),bins=[acoarse1,Rcoarse1],normed=False)[0]

    for R in Rcoarse1:
        ax4.plot(xlims,[R]*2,'k--')

    for a in acoarse1:
        ax4.plot([a]*2,ylims,'k--')

    accents1 = np.sqrt(acoarse1[:-1]*acoarse1[1:])
    Rccents1 = np.sqrt(Rcoarse1[:-1]*Rcoarse1[1:]) 

    for i in range(len(Rccents1)):
        for j in range(len(accents1)):
            tmp1 = ax4.text(accents1[j],Rccents1[i],u'%2.2f'%(hcoarse1[j,i]/len(out['smas'])),horizontalalignment='center',verticalalignment='center',weight='bold', color='white', fontsize=8)
            tmp1.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])


    #Apply Grid to Universe Planet Population
    acoarse2 = np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),6)
    Rcoarse2 = np.logspace(np.log10(ylims[0]),np.log10(ylims[1]),6)

    hcoarse2 = np.histogram2d(np.hstack(x), np.hstack(y),bins=[acoarse2,Rcoarse2],normed=False)[0]

    for R in Rcoarse2:
        ax1.plot(xlims,[R]*2,'k--')

    for a in acoarse2:
        ax1.plot([a]*2,ylims,'k--')

    accents2 = np.sqrt(acoarse2[:-1]*acoarse2[1:])
    Rccents2 = np.sqrt(Rcoarse2[:-1]*Rcoarse2[1:]) 

    for i in range(len(Rccents2)):
        for j in range(len(accents2)):
            tmp2 = ax1.text(accents2[j],Rccents2[i],u'%2.2f'%(hcoarse2[j,i]/len(out['smas'])),horizontalalignment='center',verticalalignment='center',weight='bold', color='white', fontsize=8)
            tmp2.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='k')])


    show(block=False)
    # Save to a File
    filename = 'RpvsSMAdetectionsGridOverlay'
    savefig(runPath + os.path.basename(os.path.normpath(runPath)) + filename + '.png', format='png', dpi=500)
    savefig(runPath + os.path.basename(os.path.normpath(runPath)) + filename + '.svg')
    savefig(runPath + os.path.basename(os.path.normpath(runPath)) + filename + '.eps', format='eps', dpi=500)



    # fig3 = figure(3)
    # #create coarse grid and calculate total numbers in each bin
    # acoarse1 = np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),6)
    # Rcoarse1 = np.logspace(np.log10(ylims[0]),np.log10(ylims[1]),6)

    # hcoarse1 = np.histogram2d(np.hstack(out['smas']), np.hstack(out['Rps']),bins=[acoarse1,Rcoarse1],normed=False)[0]

    # for R in Rcoarse1:
    #     plot(xlims,[R]*2,'k--')

    # for a in acoarse1:
    #     plot([a]*2,ylims,'k--')

    # accents1 = np.sqrt(acoarse1[:-1]*acoarse1[1:])
    # Rccents1 = np.sqrt(Rcoarse1[:-1]*Rcoarse1[1:]) 

    # for i in range(len(Rccents1)):
    #     for j in range(len(accents1)):
    #         text(accents1[j],Rccents1[i],u'%2.2f'%hcoarse1[j,i],horizontalalignment='center',verticalalignment='center')




    # show(block=False)


    # #Dmitry's Code
    # aedges = np.logspace(np.log10(0.2),np.log10(25),101)
    # Redges = np.logspace(0,np.log10(16),31)

    # acents = np.diff(aedges)/2.+aedges[:-1]
    # Rcents = np.diff(Redges)/2.+Redges[:-1]


    # h = np.histogram2d(np.hstack(out['smas']), np.hstack(out['Rps']),bins=[aedges,Redges])[0]

    # plt.figure()
    # plt.clf()
    # plt.contourf(acents,Rcents,np.log10(h.T/float(len(out['smas']))))
    # gca().set_xscale('log')
    # gca().set_yscale('log')
    # plt.xlabel('a (AU)')
    # plt.ylabel('R ($R_\\oplus$)')
    # c = plt.colorbar()

    # show(block=False)

#DMITRY's most recent version of this plotting function
# #create true normed joint PDF
# alim = [0.2,25]
# Rlim = [1,16]

# aedges = np.logspace(np.log10(alim[0]),np.log10(alim[1]),101)
# Redges = np.logspace(np.log10(Rlim[0]),np.log10(Rlim[1]),31)

# acents = np.diff(aedges)/2.+aedges[:-1]
# Rcents = np.diff(Redges)/2.+Redges[:-1]

# h = np.histogram2d(np.hstack(out['smas']), np.hstack(out['Rps']),bins=[aedges,Redges],normed=True)[0]

# plt.figure()
# plt.clf()
# plt.contourf(acents,Rcents,np.log10(h.T))
# gca().set_xscale('log')
# gca().set_yscale('log')
# gca().set_xlim(alim)
# gca().set_ylim(Rlim)
# plt.xlabel('a (AU)')
# plt.ylabel('R ($R_\\oplus$)')
# c = plt.colorbar()

# #create coarse grid and calculate total numbers in each bin
# acoarse = np.logspace(np.log10(alim[0]),np.log10(alim[1]),6)
# Rcoarse = np.logspace(np.log10(Rlim[0]),np.log10(Rlim[1]),6)

# hcoarse = np.histogram2d(np.hstack(out['smas']), np.hstack(out['Rps']),bins=[acoarse,Rcoarse],normed=False)[0]

# for R in Rcoarse:
#     plot(alim,[R]*2,'k--')

# for a in acoarse:
#     plot([a]*2,Rlim,'k--')

# accents = np.sqrt(acoarse[:-1]*acoarse[1:])
# Rccents = np.sqrt(Rcoarse[:-1]*Rcoarse[1:]) 

# for i in range(len(Rccents)):
#     for j in range(len(accents)):
#         text(accents[j],Rccents[i],u'%2.2f'%hcoarse[j,i],horizontalalignment='center',verticalalignment='center')


#         





#Stuff To Keep but probably delete in the near future
    # CS = plt.contourf(X, Y, Z, 10,
    #               #[-1, -0.1, 0, 0.1],
    #               #alpha=0.5,
    #               cmap=plt.cm.bone,
    #               origin=origin)


    # # Note that in the following, we explicitly pass in a subset of
    # # the contour levels used for the filled contours.  Alternatively,
    # # We could pass in additional levels to provide extra resolution,
    # # or leave out the levels kwarg to use all of the original levels.

    # CS2 = plt.contour(CS, levels=CS.levels[::2],
    #                   colors='r',
    #                   origin=origin)

    # plt.title('Nonsense (3 masked regions)')
    # plt.xlabel('word length anomaly')
    # plt.ylabel('sentence length anomaly')

    # # Make a colorbar for the ContourSet returned by the contourf call.

    # cbar.ax.set_ylabel('verbosity coefficient')
    # # Add the contour line levels to the colorbar
    # cbar.add_lines(CS2)