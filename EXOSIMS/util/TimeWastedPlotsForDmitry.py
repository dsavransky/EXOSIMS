import os
import numpy as np
from pylab import *
from numpy import nan
import matplotlib.pyplot as plt
import argparse
import json


QFile = '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/util/PlotTimeWastedQueueDmitryRuns.json'
try:
    with open(os.getcwd() + '/' + os.path.basename(QFile).split('.')[0] + 'TimeWastedRunQueue.json') as outfile:
        Qdata = json.load(outfile)
except:
    print('Failed to open outspecfile %s'%os.getcwd() + '/' + os.path.basename(QFile).split('.')[0] + 'TimeWastedRunQueue.json')
    pass

allFolderDirs = Qdata['allFolderDirs']
runMeanEngineeringTimeWasted = Qdata['runmeanEngineeringTimeWasted']
runMeanTimeDetecting = Qdata['runMeanTimeDetecting']
runMeanTimeCharacterizing = Qdata['runMeanTimeCharacterizing']
runMeanTimeObserving = Qdata['runMeanTimeObserving']
runMeanTimeWastedFromOB = Qdata['runMeanTimeWastedFromOB']
runMeanMissionPercentTimeWastedFromOB = Qdata['runMeanMissionPercentTimeWastedFromOB']
runMeanNumCharacterizations = Qdata['runMeanNumCharacterizations']
runMeanNumUniqDetections = Qdata['runMeanNumUniqDetections']
runMeanTimeWastedOnTargetsWithNoDet = Qdata['runMeanTimeWastedOnTargetsWithNoDet']

plt.close('all')

#Plot Mean Engineering Time Wasted
METW = figure(2000)
rc('axes',linewidth=2)
rc('lines',linewidth=2)
rcParams['axes.linewidth']=2
rc('font',weight='bold') 
#marker='s' => 7d OB
#marker='o' => 3.5d OB
#marker='v' => Infinite OB
#c='r' => KeplerLike2
#c='b' => SAG13
scatter(1, runMeanEngineeringTimeWasted[0], marker='s', c='r')
scatter(2, runMeanEngineeringTimeWasted[2], marker='s', c='r')
scatter(3, runMeanEngineeringTimeWasted[4], marker='s', c='r')
scatter(1, runMeanEngineeringTimeWasted[1], marker='o', c='r')
scatter(2, runMeanEngineeringTimeWasted[3], marker='o', c='r')
scatter(3, runMeanEngineeringTimeWasted[5], marker='o', c='r')
scatter(1, runMeanEngineeringTimeWasted[15], marker='v', c='r')
scatter(2, runMeanEngineeringTimeWasted[13], marker='v', c='r')
scatter(3, runMeanEngineeringTimeWasted[17], marker='v', c='r')

scatter(1, runMeanEngineeringTimeWasted[7], marker='s', c='b')
scatter(2, runMeanEngineeringTimeWasted[10], marker='s', c='b')
scatter(3, runMeanEngineeringTimeWasted[8], marker='s', c='b')
scatter(1, runMeanEngineeringTimeWasted[11], marker='o', c='b')
scatter(2, runMeanEngineeringTimeWasted[9], marker='o', c='b')
scatter(3, runMeanEngineeringTimeWasted[12], marker='o', c='b')
scatter(1, runMeanEngineeringTimeWasted[16], marker='v', c='b')
scatter(2, runMeanEngineeringTimeWasted[14], marker='v', c='b')
scatter(3, runMeanEngineeringTimeWasted[18], marker='v', c='b')
xticks([1,2,3], [1,2,3])
xlim([0,4])
ylim([0,max(runMeanEngineeringTimeWasted)*1.25])
h1 = scatter(-1,-1, marker='s', c='k', label='7d OB duration')
h2 = scatter(-1,-1, marker='o', c='k', label='3.5d OB duration')
h3 = scatter(-1,-1, marker='v', c='k', label='Proportional\nOB duration')
c1 = scatter(-1,-1, marker='_', c='r', s=30, label='KeplerLike2')
c2 = scatter(-1,-1, marker='_', c='b', s=30, label='SAG13')
leg1 = legend(prop={'weight':'bold'}, handles=[h1,h2,h3],loc=2)
ax = plt.gca().add_artist(leg1)
leg2 = legend(prop={'weight':'bold'}, handles=[c1,c2],loc=1)
xlabel('Mission Length (years)', weight='bold')
ylabel('Mean Engineering Time Wasted (d)', weight='bold')
title('Constant exoplanetObsTime=3mo')
show(block=False)

savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'meanEngineeringTimeWasted' + '.png')
savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'meanEngineeringTimeWasted' + '.svg')
savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'meanEngineeringTimeWasted' + '.eps')

#Plot numCharacterizations
NC = figure(2001)
rc('axes',linewidth=2)
rc('lines',linewidth=2)
rcParams['axes.linewidth']=2
rc('font',weight='bold') 
scatter(1, runMeanNumCharacterizations[0], marker='s', c='r')
scatter(2, runMeanNumCharacterizations[2], marker='s', c='r')
scatter(3, runMeanNumCharacterizations[4], marker='s', c='r')
scatter(1, runMeanNumCharacterizations[1], marker='o', c='r')
scatter(2, runMeanNumCharacterizations[3], marker='o', c='r')
scatter(3, runMeanNumCharacterizations[5], marker='o', c='r')
scatter(1, runMeanNumCharacterizations[15], marker='v', c='r')
scatter(2, runMeanNumCharacterizations[13], marker='v', c='r')
scatter(3, runMeanNumCharacterizations[17], marker='v', c='r')
scatter(1, runMeanNumCharacterizations[7], marker='s', c='b')
scatter(2, runMeanNumCharacterizations[10], marker='s', c='b')
scatter(3, runMeanNumCharacterizations[8], marker='s', c='b')
scatter(1, runMeanNumCharacterizations[11], marker='o', c='b')
scatter(2, runMeanNumCharacterizations[9], marker='o', c='b')
scatter(3, runMeanNumCharacterizations[12], marker='o', c='b')
scatter(1, runMeanNumCharacterizations[16], marker='v', c='b')
scatter(2, runMeanNumCharacterizations[14], marker='v', c='b')
scatter(3, runMeanNumCharacterizations[18], marker='v', c='b')
xticks([1,2,3], [1,2,3])
xlim([0,4])
ylim([0,max(runMeanNumCharacterizations)*1.3])
h1 = scatter(-1,-1, marker='s', c='k', label='7d OB duration')
h2 = scatter(-1,-1, marker='o', c='k', label='3.5d OB duration')
h3 = scatter(-1,-1, marker='v', c='k', label='Proportional\nOB duration')
c1 = scatter(-1,-1, marker='_', c='r', s=30, label='KeplerLike2')
c2 = scatter(-1,-1, marker='_', c='b', s=30, label='SAG13')
leg1 = legend(prop={'weight':'bold'}, handles=[h1,h2,h3],loc=2)
ax = plt.gca().add_artist(leg1)
leg2 = legend(prop={'weight':'bold'}, handles=[c1,c2],loc=1)
xlabel('Mission Length (years)', weight='bold')
ylabel('Mean Number of Characterizations', weight='bold')
title('Constant exoplanetObsTime=3mo')
show(block=False)

savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'MeanNumCharacterizations' + '.png')
savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'MeanNumCharacterizations' + '.svg')
savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'MeanNumCharacterizations' + '.eps')

#Plot runMeanNumUniqDetections
NUD = figure(2002)
rc('axes',linewidth=2)
rc('lines',linewidth=2)
rcParams['axes.linewidth']=2
rc('font',weight='bold') 
scatter(1, runMeanNumUniqDetections[0], marker='s', c='r')
scatter(2, runMeanNumUniqDetections[2], marker='s', c='r')
scatter(3, runMeanNumUniqDetections[4], marker='s', c='r')
scatter(1, runMeanNumUniqDetections[1], marker='o', c='r')
scatter(2, runMeanNumUniqDetections[3], marker='o', c='r')
scatter(3, runMeanNumUniqDetections[5], marker='o', c='r')
scatter(1, runMeanNumUniqDetections[15], marker='v', c='r')
scatter(2, runMeanNumUniqDetections[13], marker='v', c='r')
scatter(3, runMeanNumUniqDetections[17], marker='v', c='r')
scatter(1, runMeanNumUniqDetections[7], marker='s', c='b')
scatter(2, runMeanNumUniqDetections[10], marker='s', c='b')
scatter(3, runMeanNumUniqDetections[8], marker='s', c='b')
scatter(1, runMeanNumUniqDetections[11], marker='o', c='b')
scatter(2, runMeanNumUniqDetections[9], marker='o', c='b')
scatter(3, runMeanNumUniqDetections[12], marker='o', c='b')
scatter(1, runMeanNumUniqDetections[16], marker='v', c='b')
scatter(2, runMeanNumUniqDetections[14], marker='v', c='b')
scatter(3, runMeanNumUniqDetections[18], marker='v', c='b')
xticks([1,2,3], [1,2,3])
xlim([0,4])
ylim([0,max(runMeanNumUniqDetections)*1.35])
h1 = scatter(-1,-1, marker='s', c='k', label='7d OB duration')
h2 = scatter(-1,-1, marker='o', c='k', label='3.5d OB duration')
h3 = scatter(-1,-1, marker='v', c='k', label='Proportional\nOB duration')
c1 = scatter(-1,-1, marker='_', c='r', s=30, label='KeplerLike2')
c2 = scatter(-1,-1, marker='_', c='b', s=30, label='SAG13')
leg1 = legend(prop={'weight':'bold'}, handles=[h1,h2,h3],loc=2)
ax = plt.gca().add_artist(leg1)
leg2 = legend(prop={'weight':'bold'}, handles=[c1,c2],loc=1)
xlabel('Mission Length (years)', weight='bold')
ylabel('Mean Number of Unique Detections', weight='bold')
title('Constant exoplanetObsTime=3mo')
show(block=False)

savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'runMeanNumUniqDetections' + '.png')
savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'runMeanNumUniqDetections' + '.svg')
savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'runMeanNumUniqDetections' + '.eps')


#Plot runMeanTimeWastedOnTargetsWithNoDet
MTWnoDet= figure(2003)
rc('axes',linewidth=2)
rc('lines',linewidth=2)
rcParams['axes.linewidth']=2
rc('font',weight='bold') 
scatter(1, runMeanTimeWastedOnTargetsWithNoDet[0], marker='s', c='r')
scatter(2, runMeanTimeWastedOnTargetsWithNoDet[2], marker='s', c='r')
scatter(3, runMeanTimeWastedOnTargetsWithNoDet[4], marker='s', c='r')
scatter(1, runMeanTimeWastedOnTargetsWithNoDet[1], marker='o', c='r')
scatter(2, runMeanTimeWastedOnTargetsWithNoDet[3], marker='o', c='r')
scatter(3, runMeanTimeWastedOnTargetsWithNoDet[5], marker='o', c='r')
scatter(1, runMeanTimeWastedOnTargetsWithNoDet[15], marker='v', c='r')
scatter(2, runMeanTimeWastedOnTargetsWithNoDet[13], marker='v', c='r')
scatter(3, runMeanTimeWastedOnTargetsWithNoDet[17], marker='v', c='r')
scatter(1, runMeanTimeWastedOnTargetsWithNoDet[7], marker='s', c='b')
scatter(2, runMeanTimeWastedOnTargetsWithNoDet[10], marker='s', c='b')
scatter(3, runMeanTimeWastedOnTargetsWithNoDet[8], marker='s', c='b')
scatter(1, runMeanTimeWastedOnTargetsWithNoDet[11], marker='o', c='b')
scatter(2, runMeanTimeWastedOnTargetsWithNoDet[9], marker='o', c='b')
scatter(3, runMeanTimeWastedOnTargetsWithNoDet[12], marker='o', c='b')
scatter(1, runMeanTimeWastedOnTargetsWithNoDet[16], marker='v', c='b')
scatter(2, runMeanTimeWastedOnTargetsWithNoDet[14], marker='v', c='b')
scatter(3, runMeanTimeWastedOnTargetsWithNoDet[18], marker='v', c='b')
xticks([1,2,3], [1,2,3])
xlim([0,4])
ylim([0,max(runMeanTimeWastedOnTargetsWithNoDet)*1.2])
h1 = scatter(-1,-1, marker='s', c='k', label='7d OB duration')
h2 = scatter(-1,-1, marker='o', c='k', label='3.5d OB duration')
h3 = scatter(-1,-1, marker='v', c='k', label='Proportional\nOB duration')
c1 = scatter(-1,-1, marker='_', c='r', s=30, label='KeplerLike2')
c2 = scatter(-1,-1, marker='_', c='b', s=30, label='SAG13')
leg1 = legend(prop={'weight':'bold'}, handles=[h1,h2,h3],loc=2)
ax = plt.gca().add_artist(leg1)
leg2 = legend(prop={'weight':'bold'}, handles=[c1,c2],loc=1)
xlabel('Mission Length (years)', weight='bold')
ylabel('Mean Science Time Wasted (d)', weight='bold')
title('Constant exoplanetObsTime=3mo')
show(block=False)

savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'runMeanTimeWastedOnTargetsWithNoDet' + '.png')
savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'runMeanTimeWastedOnTargetsWithNoDet' + '.svg')
savefig('/home/dean/Desktop/ContentsForDmitryPresentationToWFIRSTPeople/' + 'runMeanTimeWastedOnTargetsWithNoDet' + '.eps')