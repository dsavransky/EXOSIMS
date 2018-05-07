"""
Plot Planet Population Radius vs a AND Detected Planet R vs a
Written by Dean Keithly on 5/6/2018
"""


#RUN LOAD MISSION FROM SEED OPERATION...

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

pklPaths = list()
outspecPaths = list()
pklPaths.append('/home/dean/Documents/SIOSlab/Dean2May18RS15CXXfZ01OB01PP01SU01/run875756922387.pkl')
outspecPaths.append('/home/dean/Documents/SIOSlab/Dean2May18RS15CXXfZ01OB01PP01SU01/outspec.json')
cnt=0
try:
    with open(pklPaths[cnt], 'rb') as f:#load from cache
        DRM = pickle.load(f)
except:
    print('Failed to open pklfile %s'%pklPaths[cnt])
    pass
try:
    with open(outspecPaths[cnt], 'rb') as g:
        outspec = json.load(g)
except:
    print('Failed to open outspecfile %s'%outspecPaths[cnt])
    pass


#Load DRM from pkl file and compare to sim run with same seed
# pklfile = "/home/dean/Documents/SIOSlab/Dean2May18RS16CXXfZ01OB01PP01SU01/run979296152263.pkl"
# outspecfile = "/home/dean/Documents/SIOSlab/Dean2May18RS16CXXfZ01OB01PP01SU01/outspec.json"
# with open(pklfile, 'rb') as f:#load from cache
#     DRM3tmp = pickle.load(f)

DRMseed = DRM['seed']
newDRM = DRM['DRM']

# with open(outspecPaths[cnt], 'rb') as f:#load from cache
#     newjson = json.load(f)
# newjson['seed'] = DRMseed
# newjson['modules']['SurveyEnsemble'] = "SurveyEnsemble"
#with open(os.path.dirname(outspecPaths[cnt])+'tmp.json', 'wb') as f:
#    json.dump(newjson, f)
#sim4 = EXOSIMS.MissionSim.MissionSim(os.path.dirname(outspecPaths[cnt]) + 'tmp.json')

#Collect all detected Rp and a
detRp = list()
deta = list()
for i in range(len(DRM['DRM'])):
    if 1 in DRM['DRM'][i]['det_status']:#Was there a detection
        star_ind = DRM['DRM'][i]['star_ind']
        plan_inds = DRM['DRM'][i]['plan_inds']
        #iterate over detected planet inds
        for j in np.where(DRM['DRM'][i]['det_status'] == 1):
            deta.append(DRM['systems']['a'][plan_inds[j]])
            detRp.append(DRM['systems']['Rp'][plan_inds[j]])

fig1 = figure(1)
a = DRM['systems']['a']#All generated Rp and a
Rp = DRM['systems']['Rp']
scatter(a,Rp,marker='o',color='b')
scatter(deta,detRp,marker='x',color='r')
xlabel('a')
ylabel(r'$R_{p}$')
show(block=False)

