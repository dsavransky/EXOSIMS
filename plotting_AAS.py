"""
Plot Yield Plot Histograms
Written by: Dean Keithly
Written on: 11/28/2018
"""
from EXOSIMS.util.read_ipcluster_ensemble import gen_summary
import os
if not 'DISPLAY' in os.environ.keys(): #Check environment for keys
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
import math
import datetime
import re



plotmeans=True
folder = '/home/corey/Documents/GitHub/EXOSIMS/simulations/AASDC'
PPoutpath = '/home/corey/Documents/GitHub/EXOSIMS/simulations'
base_path = '/home/corey/Documents/GitHub/EXOSIMS/simulations/'
runs = ['AASStatic2', 'AASPD', 'AASDC', 'AASDC3']
legtext = ['Static', 'Partially dynamic', 'Dynamic MaxC', 'Dynamic PriorityObs']

# unique_text

res = []
for run_name in runs:
    folder = os.path.join(base_path, run_name)
    res.append(gen_summary(folder)['detected'])

#Set linewidth and color cycle
plt.rc('axes',linewidth=2)
plt.rc('lines',linewidth=2)
plt.rc('axes',prop_cycle=(cycler('color',['red','purple','blue','black','darkorange','forestgreen'])))

rcounts = []
unique_rcounts = []
for el in res:
    rcounts.append(np.array([len(r) for r in el]))
    unique_rcounts.append(np.array([np.unique(r).size for r in el]))

bins = range(np.min(np.hstack(rcounts).astype(int)),np.max(np.hstack(rcounts).astype(int))+2)
bcents = np.diff(bins)/2. + bins[:-1]


pdfs = []
unique_pdfs = []
for j in range(len(res)):
    # Total detections
    pdfs.append(np.histogram(rcounts[j],bins=bins,density=True)[0].astype(float))

for j in range(len(res)):
    # Unique detections
    unique_pdfs.append(np.histogram(unique_rcounts[j],bins=bins,density=True)[0].astype(float))

mx = 1.1*np.max(pdfs) #math.ceil(np.max(pdfs)*10)/10#np.round(np.max(pdfs),decimals=1)
print(mx)

syms = 'ospx^*<>h'
total_lstyle = '--'
unique_lstyle = ':'


# Total detections loop
plt.figure(figsize=(20,5))
for j in range(len(res)):
    leg = legtext[j]
    c = plt.gca()._get_lines.prop_cycler.__next__()['color']#after 3.6

    if plotmeans:
        mn = np.mean(rcounts[j])
        plt.plot([mn]*2,[0,mx],'--',color=c)
        if leg is not None:
            leg += ' mission ($\\mu = %2.2f$)'%mn
    plt.plot(bcents, pdfs[j], syms[np.mod(j,len(syms))]+total_lstyle,color=c,label=leg, lw=4, ms=15)
plt.ylim([0,mx])
plt.legend(prop={'size': 26})
plt.xlabel('Total Exoplanet Detections', weight='bold', size=20)
plt.ylabel('Normalized Yield Frequency', weight='bold', size=20)
plt.xticks(np.arange(0, 76, step=5), size=20)
plt.yticks(size=20)
plt.title('Results from 1000 WFIRST mission simulations', size=35, weight='bold')

date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'Total_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath,fname+'.png'), bbox_inches='tight')
plt.savefig(os.path.join(PPoutpath,fname+'.svg'), bbox_inches='tight')
plt.savefig(os.path.join(PPoutpath,fname+'.eps'), bbox_inches='tight')
plt.savefig(os.path.join(PPoutpath,fname+'.pdf'), bbox_inches='tight')

# Unique detections loop
plt.figure(figsize=(20,5))
for j in range(len(res)):
    leg = legtext[j]
    c = plt.gca()._get_lines.prop_cycler.__next__()['color']#after 3.6

    if plotmeans:
        mn = np.mean(unique_rcounts[j])
        plt.plot([mn]*2,[0,mx],'--',color=c, ms=5)
        if leg is not None:
            leg += ' mission ($\\mu = %2.2f$)'%mn
    plt.plot(bcents, unique_pdfs[j], syms[np.mod(j,len(syms))]+total_lstyle,color=c,label=leg, lw=4, ms=15)

plt.ylim([0,mx])
plt.legend(prop={'size': 26})
plt.xlabel('Unique Exoplanet Detections', weight='bold', size=20)
plt.ylabel('Normalized Yield Frequency', weight='bold', size=20)
plt.xticks(np.arange(0, 76, step=5), size=20)
plt.yticks(size=20)
plt.title('Results from 1000 WFIRST mission simulations', size=35, weight='bold')

date = str(datetime.datetime.now())
date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
fname = 'Unique_' + folder.split('/')[-1] + '_' + date
plt.savefig(os.path.join(PPoutpath,fname+'.png'), bbox_inches='tight')
plt.savefig(os.path.join(PPoutpath,fname+'.svg'), bbox_inches='tight')
plt.savefig(os.path.join(PPoutpath,fname+'.eps'), bbox_inches='tight')
plt.savefig(os.path.join(PPoutpath,fname+'.pdf'), bbox_inches='tight')