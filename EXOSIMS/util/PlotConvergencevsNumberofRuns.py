""" Plot Convergence vs Number of Runs

Written by: Dean Keithly on 5/29/2018
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


runDir = '/home/dean/Documents/SIOSlab/Dean22May18RS09CXXfZ01OB01PP01SU01/'
saveFolder = '/home/dean/Documents/SIOSlab/SPIE2018Journal/'


#Given Filepath for pklfile, Plot a pkl from each testrun in subdir
pklPaths = list()
pklfname = list()


#Look for all directories in specified path with structured folder name
dirs = runDir

pklFiles = [myFileName for myFileName in os.listdir(dirs) if 'run' in myFileName and '.pkl' in myFileName]  # Get names of all pkl files in path
for i in range(len(pklFiles)):
    pklPaths.append(dirs + pklFiles[i])  # append a random pkl file to path



#Iterate over all pkl files
meanNumDets = list()
for cnt in np.arange(len(pklPaths)):
    try:
        with open(pklPaths[cnt], 'rb') as f:#load from cache
            DRM = pickle.load(f)
    except:
        print('Failed to open pklfile %s'%pklPaths[cnt])
        pass

    #Calculate meanNumDets #raw detections, not unique detections
    AllDetsInPklFile = [(DRM['DRM'][i]['det_status'] == 1).tolist().count(True) for i in range(len(DRM['DRM']))]
    meanNumDetsTMP = sum(AllDetsInPklFile)

    
    #Append to list
    if cnt == 0:
        meanNumDets.append(float(meanNumDetsTMP))
    else:
        meanNumDets.append((meanNumDets[cnt-1]*float(cnt-1+1) + meanNumDetsTMP)/float(cnt+1))

    print "%d/%d  %d %f"%(cnt,len(pklPaths), meanNumDetsTMP, meanNumDets[cnt])

plt.close('all')
fig = figure(8000)
rc('axes',linewidth=2)
rc('lines',linewidth=2)
rcParams['axes.linewidth']=2
rc('font',weight='bold') 
#rcParams['axes.titlepad']=-50
plot(abs(np.asarray(meanNumDets) - meanNumDets[9999]), color='purple')

plot([100,100],[abs(np.asarray(meanNumDets[99]) - meanNumDets[9999]),max(abs(np.asarray(meanNumDets) - meanNumDets[9999]))], linewidth=1, color='k')
gca().text(90,5,r"$\mu_{det_{100}}=$" + ' %2.1f'%(meanNumDets[99]/meanNumDets[9999]*100.) + '%', rotation=45)
plot([1000,1000],[abs(np.asarray(meanNumDets[999]) - meanNumDets[9999]),max(abs(np.asarray(meanNumDets) - meanNumDets[9999]))], linewidth=1, color='k')
gca().text(900,5,r"$\mu_{det_{1000}}=$" + ' %2.1f'%(meanNumDets[999]/meanNumDets[9999]*100.) + '%', rotation=45)
#plot([10000,10000],[abs(np.asarray(meanNumDets[9999]) - meanNumDets[9999]),max(abs(np.asarray(meanNumDets) - meanNumDets[9999]))], linewidth=1, color='k')
#ADD LABEL
xscale('log')
#yscale('log')

plot([82,82],[abs(np.asarray(meanNumDets[82]) - meanNumDets[9999]),max(abs(np.asarray(meanNumDets) - meanNumDets[9999]))],linestyle='--', linewidth=1, color='k')
gca().text(60,4.85,r"$\mu_{det_{82}}=$" + ' %2.0f'%(meanNumDets[81]/meanNumDets[9999]*100.) + '%', rotation=45)
plot([132,132],[abs(np.asarray(meanNumDets[132]) - meanNumDets[9999]),max(abs(np.asarray(meanNumDets) - meanNumDets[9999]))],linestyle='--', linewidth=1, color='k')
gca().text(130,4.85,r"$\mu_{det_{132}}=$" + ' %2.0f'%(meanNumDets[131]/meanNumDets[9999]*100.) + '%', rotation=45)
plot([363,363],[abs(np.asarray(meanNumDets[363]) - meanNumDets[9999]),max(abs(np.asarray(meanNumDets) - meanNumDets[9999]))],linestyle='--', linewidth=1, color='k')
gca().text(310,4.85,r"$\mu_{det_{363}}=$" + ' %2.0f'%(meanNumDets[362]/meanNumDets[9999]*100.) + '%', rotation=45)
plot([1550,1550],[abs(np.asarray(meanNumDets[1550]) - meanNumDets[9999]),max(abs(np.asarray(meanNumDets) - meanNumDets[9999]))],linestyle='--', linewidth=1, color='k')
gca().text(1400,5,r"$\mu_{det_{1550}}=$" + ' %2.1f'%(meanNumDets[1555]/meanNumDets[9999]*100.) + '%', rotation=45)
#gca().text(9000,4,r"$\mu_{det_{1000}}=$" + ' %2.1f'%(meanNumDets[999]/meanNumDets[9999]*100.))

xlim([1,1e4])
ylim([0,max(abs(np.asarray(meanNumDets) - meanNumDets[9999]))])
ylabel("Mean # of Detections Error\n$|\mu_{det_i}-\mu_{det_{10000}}|$", weight='bold')
xlabel("# of Simulations, i", weight='bold')
#tight_layout()
#margins(1)
gcf().subplots_adjust(top=0.75)
show(block=False)

savefig(saveFolder + 'meanNumDetectionDiffConvergence' + '.png')
savefig(saveFolder + 'meanNumDetectionDiffConvergence' + '.svg')
savefig(saveFolder + 'meanNumDetectionDiffConvergence' + '.eps')


fig = figure(8001)
rc('axes',linewidth=2)
rc('lines',linewidth=2)
rcParams['axes.linewidth']=2
rc('font',weight='bold') 
plot(meanNumDets, color='purple')
xscale('log')
xlim([1,1e4])
ylim([0,meanNumDets[9999]*1.05])
ylabel("Mean # of Detections", weight='bold')
xlabel("# of Simulations, i", weight='bold')
show(block=False)
savefig(saveFolder + 'meanNumDetectionConvergence' + '.png')
savefig(saveFolder + 'meanNumDetectionConvergence' + '.svg')
savefig(saveFolder + 'meanNumDetectionConvergence' + '.eps')


fig = figure(8002)
#ax = fig.add_subplot(111)
rc('axes',linewidth=2)
rc('lines',linewidth=2)
rcParams['axes.linewidth']=2
rc('font',weight='bold') 
plot(np.asarray(meanNumDets)/meanNumDets[9999]*100., color='purple')
xscale('log')
xlim([1,1e4])
ylim([0,100*1.05])
ylabel(r"Percentage of $\mu_{det_{10000}}$, $\frac{\mu_{det_i}}{\mu_{det_{10000}}} \times 100$", weight='bold')
xlabel("# of Simulations, i", weight='bold')
show(block=False)
savefig(saveFolder + 'percentErrorFromMeanConvergence' + '.png')
savefig(saveFolder + 'percentErrorFromMeanConvergence' + '.svg')
savefig(saveFolder + 'percentErrorFromMeanConvergence' + '.eps')



# $\mu_1000=9.82700$ which is 99.27\% of the $\mu_10000$.
# $\mu_10000=9.898799$.
# $\mu_100=9.100$ which is 91.9\% of the $\mu_10000$. 
# 90\% of the $\mu_10000$ is achieved at 82sims.
# 95\% of the $\mu_10000$ is achieved at 132sims.
# 99\% of the $\mu_10000$ is achieved at 363sims.
# 99.5\% of the $\mu_10000$ is achieved at 1550sims.