"""Load Mission From Seed
Written by Dean Keithly on 5/6/2018
"""
#### CONFIRM DRM COMPARISON WORKS
import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import json
import copy
from copy import deepcopy

# folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/EXOSIMS/EXOSIMS/Scripts'))
# filename = 'sS_AYO7.json'
# scriptfile = os.path.join(folder,filename)
# #run simulation #1
# sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
# sim.run_sim()
# DRM = sim.SurveySimulation.DRM

# #run simulation #2
# sim2 = EXOSIMS.MissionSim.MissionSim(scriptfile)
# sim2.run_sim()
# DRM2 = sim2.SurveySimulation.DRM

# import numpy as np
# for i in np.arange(len(DRM)):
#     for key1 in DRM[i].keys():
#         if type(DRM[i][key1]) == type(1) or type(DRM[i][key1]) == type(1.0) or type(DRM[i][key1]) == type(np.int64(1)):#is an integer of float
#             assert(DRM[i][key1] == DRM2[i][key1])
#         elif type(DRM[i][key1]) == type(np.asarray([1,2,3,4])):#is numpy array
#             assert(all(DRM[i][key1] == DRM2[i][key1]))
#         elif hasattr(DRM[i][key1],'value'):
#             assert(DRM[i][key1].value == DRM2[i][key1].value)
#         elif hasattr(DRM[i][key1], 'keys'):
#             for key2 in DRM[i][key1]:
#                 if type(DRM[i][key1][key2]) == type(np.asarray([1,2,3,4])):#is numpy array
#                     assert(all(DRM[i][key1][key2] == DRM2[i][key1][key2]))
#                 elif hasattr(DRM[i][key1][key2],'value'):
#                     if type(DRM[i][key1][key2].value) == type(np.asarray([1,2,3,4])):#is numpy array
#                         assert(all(DRM[i][key1][key2].value == DRM2[i][key1][key2].value))
#                     else:
#                         assert(DRM[i][key1][key2].value == DRM2[i][key1][key2].value)
#                 else:
#                     assert(DRM[i][key1][key2] == DRM2[i][key1][key2])
#         elif type(DRM[i][key1]) == type('taco'):
#             assert(DRM[i][key1] == DRM2[i][key1])
#         else:
#             print DRM[i][key1]
#             print i
#             print key1

def DRMcomparator(DRM1, DRM2, exact=True):
    """
    Args:
        DRM3 (list of dict):
            DRM of mission 1 to compare to mission 2
        DRM2 (list of dict):
            DRM of mission2 to compare to mission 1
        exact (boolean):
            boolean indicating whether to be exact in comparison (default) (all keys and values are identical), or
            approximately equal (useful for comparing DRM from pkl file and freshly simulated DRM)
    Returns:
        success (boolean):
            boolean indicating they are identical (true) or different (False)
    """
    if exact:#if they must be equal
        ae = 0
    else:
        ae = 1e-10 #arbitrarily set allowed error

    for i in np.arange(len(DRM1)):#Iterate over all elements of list
        for key1 in DRM1[i].keys():#iterate over all key1
            #Comparison 1
            if type(DRM1[i][key1]) == type(1) or type(DRM1[i][key1]) == type(np.int64(1)):  # key value is an integer or int64
                if not DRM1[i][key1] == DRM2[i][key1]:
                    return False
            #Comparison 2
            elif type(DRM1[i][key1]) == type(1.0):  # key value is float 
                if not abs(DRM1[i][key1] - DRM2[i][key1]) <= ae:
                    return False

            #Comparison 3
            elif type(DRM1[i][key1]) == type(np.asarray([1,2,3,4])) and (DRM1[i][key1].dtype == np.int64(1).dtype):# or DRM1[i][key1].dtype == 1.dtype):  # is numpy array of ints
                if not all(DRM1[i][key1] == DRM2[i][key1]):
                    return False

            #Comparison 4
            elif type(DRM1[i][key1]) == type(np.asarray([1,2,3,4])) and (DRM1[i][key1].dtype == np.float64(1).dtype):# or type(DRM1[i][key1][0]) == type(1.)):  # is numpy array of floats
                if not all(abs(DRM1[i][key1] - DRM2[i][key1]) <= ae):
                    return False

            #Comparison 5
            elif hasattr(DRM1[i][key1],'value'):
                if not DRM1[i][key1].value == DRM2[i][key1].value:
                    print "key: %s, i: %d, values: %f, %f"%(key1, i, DRM1[i][key1].value, DRM2[i][key1].value)
                    return False

            #Explore all dict elements with dicts
            elif hasattr(DRM1[i][key1], 'keys'):
                for key2 in DRM1[i][key1]:#iterate over all key2

                    #Comparison 6
                    if type(DRM1[i][key1][key2]) == type(np.asarray([1,2,3,4])) and (DRM1[i][key1][key2].dtype == np.int64(1).dtype):# or type(DRM1[i][key1][key2][0]) == type(1)):  # is numpy array of ints
                        if not all(DRM1[i][key1][key2] == DRM2[i][key1][key2]):
                            return False

                    #Comparison 7
                    elif type(DRM1[i][key1][key2]) == type(np.asarray([1,2,3,4])) and (DRM1[i][key1][key2].dtype == np.float64(1).dtype):# or type(DRM1[i][key1][key2][0]) == type(1.)):  # is numpy array of floats
                        if not all(abs(DRM1[i][key1][key2] - DRM2[i][key1][key2]) <= ae):
                            return False

                    #If elements have units
                    elif hasattr(DRM1[i][key1][key2],'value'):
                        #Comparison 8
                        if type(DRM1[i][key1][key2].value) == type(np.asarray([1,2,3,4])) and (DRM1[i][key1][key2].value.dtype == np.int64(1).dtype):# or type(DRM1[i][key1][key2][0].value) == type(1)):  # is numpy array of ints
                            if not all(DRM1[i][key1][key2].value == DRM2[i][key1][key2].value):
                                return False

                        #Comparison 9
                        elif type(DRM1[i][key1][key2].value) == type(np.asarray([1,2,3,4])) and (DRM1[i][key1][key2].value.dtype == np.float64(1).dtype):# or type(DRM1[i][key1][key2][0].value) == type(1.)):  # is numpy array of floats
                            if not all(abs(DRM1[i][key1][key2].value - DRM2[i][key1][key2].value) <= ae):
                                return False

                        #Comparison 10
                        else:
                            if not DRM1[i][key1][key2].value == DRM2[i][key1][key2].value:
                                return False
                    else:
                        #Comparison 11
                        if not DRM1[i][key1][key2] == DRM2[i][key1][key2]:
                            return False
            elif type(DRM1[i][key1]) == type('taco'):
                #Comparison 12
                if not DRM1[i][key1] == DRM2[i][key1]:
                    return False
            else:
                print("i: %d, key1: %s, value DRM1[i][key1]: %s"%(i, key1, str(DRM1[i][key1])))
                print DRM1[i][key1]
                #print DRM1[i][key1]
                #print i
                #print key1

    return True


######################################################################
#Load DRM from pkl file and compare to sim run with same seed
pklfile =     "/home/dean/Documents/SIOSlab/Dean6May18RS09CXXfZ01OB09PP01SU01/run9333229941.pkl"
outspecfile = "/home/dean/Documents/SIOSlab/Dean6May18RS09CXXfZ01OB09PP01SU01/outspec.json"
with open(pklfile, 'rb') as f:#load pkl to extract mission specific SEED
    DRM3tmp = pickle.load(f)

DRM3seed = DRM3tmp['seed']
DRM3 = DRM3tmp['DRM']
DRM3systems = DRM3tmp['systems']
#Spot Logic Check to ensure DRM3systems plan_inds have the correct star
#planet indices for DRM3
pInds = DRM3[0]['plan_inds']
#All plan_inds should have the same star. Confirm with this
DRM3systems['star'][pInds] #all names should be identical
#########################################################################


#Load the Outspec File for Modification#######################
with open(outspecfile, 'rb') as f:#load from cache
    newjson = json.load(f)
newjson['seed'] = DRM3seed#Modify the outspec file
newjson['modules']['SurveyEnsemble'] = "SurveyEnsemble"#modify the outspec file
#Save newjson script
with open(os.path.dirname(outspecfile)+'tmp.json', 'wb') as f:
    json.dump(newjson, f)
##############################################################

######################################
#Comparing two freshly run simulations
#This validates that the DRM can be replicated under some circumstances
######################################
sim4 = EXOSIMS.MissionSim.MissionSim(os.path.dirname(outspecfile) + 'tmp.json') #initializing random mission
sim4.SurveySimulation.reset_sim()
seed1 = sim4.SurveySimulation._outspec['seed']
sim4.SurveySimulation.run_sim()
DRM1 = deepcopy(sim4.SurveySimulation.DRM)

#here is where we separately duplicat DRM1
sim4.SurveySimulation.reset_sim(seed=seed1)
sim4.SurveySimulation.run_sim()
DRM2 = sim4.SurveySimulation.DRM

success1 = DRMcomparator(DRM1, DRM2)
print("#11111111111111111111111111111111111111")
print('Freshly Run Simulation Comparison')
print(success1)
#This operation succeeded on #5/9/2018
######################################


#################################################
#Comparing freshly run simulation to pkl file DRM
#################################################
sim4.SurveySimulation.reset_sim(seed=DRM3seed)#I must make this method work
#sim4.SurveySimulation.reset_sim(seed=newjson['seed'])
sim4.SurveySimulation.run_sim()
DRM4 = deepcopy(sim4.SurveySimulation.DRM)
success2 = DRMcomparator(DRM4, DRM3, exact=False)
print('Freshly Run Simulation To Loaded pkl File Comparison')
print(success2)


#########################################
#Compairing Freshly Made Sim to DRM output
#########################################
sim5 = EXOSIMS.MissionSim.MissionSim(os.path.dirname(outspecfile) + 'tmp.json') #initializing random mission
sim5.SurveySimulation.run_sim()
DRM5 = deepcopy(sim5.SurveySimulation.DRM)


pInds2 = DRM5[0]['plan_inds']
#All plan_inds should have the same star. Confirm with this
DRM3systems['star'][pInds2] #all nam
success3 = DRMcomparator(DRM5, DRM3, exact=False)
print('Freshly Run Simulation To Loaded pkl File Comparison')
print(success3)

sim5.SurveySimulation.reset_sim(seed=sim5.SurveySimulation.seed)
sim5.SurveySimulation.run_sim()
DRM5a = deepcopy(sim5.SurveySimulation.DRM)
success5 = DRMcomparator(DRM5, DRM5a)
print('Freshly Run Simulation Comparison')
print(success5)


#################################################
tmp = [all(DRM3[x]['plan_inds'] == DRM5[x]['plan_inds']) for x in range(len(DRM3))]


#DELETE
# from pylab import *
# fig = figure(1)
# error = list()
# atime = list()
# EOT3 = list()
# for i in np.arange(len(DRM1)):
#     EOT3.append(DRM1[i]['exoplanetObsTime'].value)
#     error.append(DRM1[i]['exoplanetObsTime'].value-DRM2[i]['exoplanetObsTime'].value)
#     atime.append(DRM1[i]['arrival_time'].value-DRM2[i]['arrival_time'].value)
# plot([0 for i in range(len(error))],error,marker='o')
# #plot(EOT3)
# show(block=False)

