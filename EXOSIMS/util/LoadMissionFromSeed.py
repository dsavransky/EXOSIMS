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



######################################################################
#Load DRM from pkl file and compare to sim run with same seed
pklfile = "/home/dean/Documents/SIOSlab/Dean2May18RS16CXXfZ01OB01PP01SU01/run979296152263.pkl"
outspecfile = "/home/dean/Documents/SIOSlab/Dean2May18RS16CXXfZ01OB01PP01SU01/outspec.json"
with open(pklfile, 'rb') as f:#load pkl to extract mission specific SEED
    DRM3tmp = pickle.load(f)

DRM3seed = DRM3tmp['seed']
DRM3 = DRM3tmp['DRM']
DRM3systems = DRM3tmp['systems']
#Spot Logic Check to ensure DRM3systems plan_inds have the correct star##############
#planet indices for DRM3
pInds = DRM3[0]['plan_inds']
#All plan_inds should have the same star. Confirm with this
DRM3systems['star'][pInds] #all names should be identical


#Load the Outspec File for Modification#######################
with open(outspecfile, 'rb') as f:#load from cache
    newjson = json.load(f)
newjson['seed'] = DRM3seed#Modify the outspec file
newjson['modules']['SurveyEnsemble'] = "SurveyEnsemble"#modify the outspec file
#Save newjson script
with open(os.path.dirname(outspecfile)+'tmp.json', 'wb') as f:
    json.dump(newjson, f)

#Initialize sim4 from modified json script
sim4 = EXOSIMS.MissionSim.MissionSim(os.path.dirname(outspecfile) + 'tmp.json')
SU1 = sim4.SurveySimulation.SimulatedUniverse
TK = sim4.SurveySimulation.TimeKeeping
Obs = sim4.SurveySimulation.Observatory
SS = sim4.SurveySimulation

with open(os.path.dirname(outspecfile) + 'tmp.json', 'rb') as f:#load from cache
    newjson2 = json.load(f)

TK.__init__(**newjson2.copy())
Obs.__init__(**newjson2.copy())
SS.__init__(**newjson2.copy())#The scienceInstruments become populated with units....
print(newjson2)#For some reason, newjson2 has been populated with units... gross
with open(os.path.dirname(outspecfile) + 'tmp.json', 'rb') as f:#load from cache
    newjson2 = json.load(f)
SU1.__init__(**newjson2.copy())


print saltyburrito

SU1.I = DRM3systems['I']
SU1.M0 = DRM3systems['M0']
SU1.Mp = DRM3systems['Mp']
SU1.O = DRM3systems['O']
SU1.Rp = DRM3systems['Rp']
SU1.a = DRM3systems['a']
SU1.e = DRM3systems['e']
SU1.mu = DRM3systems['mu']
SU1.p = DRM3systems['p']
SU1.plan2star = DRM3systems['plan2star']
SU1.star = DRM3systems['star']
SU1.w = DRM3systems['w']

#absT1 = TK.currentTimeAbs.copy()
#SCpos1 = Obs.orbit(TK.currentTimeAbs.copy())
#fZ1 = sim4.ZodiacalLight.fZ(Obs, sim4.TargetList, range(sim4.TargetList.nStars), absT1, sim4.SurveySimulation.mode)


sim4.SurveySimulation.run_sim()
SS = sim4.SurveySimulation
DRM4 = SS.DRM

#FACT This statement must be if both DRM are equivalent
if all(DRM4[0]['plan_inds'] == pInds):
    print('okay1')
else:
    print('not okay1')


ind = DRM4[0]['star_ind']
DRM4[0]['det_fZ']
fZ1[ind]




# #sim4.SurveySimulation._outspec['seed'] = DRM3seed

# #sim4.reset_sim(genNewPlanets=False)
# SU = sim4.SurveySimulation.SimulatedUniverse
# TK = sim4.SurveySimulation.TimeKeeping
# Obs = sim4.SurveySimulation.Observatory
# SS = sim4.SurveySimulation
# # re-initialize SurveySimulation arrays
# specs = SS._outspec
# specs.pop('seed')
# specs['seed'] = DRM3seed
# specs['modules'] = sim4.SurveySimulation.modules
# # if 'seed' in specs:
# #     specs.pop('seed')

# #sim4.SurveySimulation.__init__(**specs)
# TK.__init__(**TK._outspec)
# Obs.__init__(**Obs._outspec)
# SS.__init__(**specs)
# # reset mission time and observatory parameters


# #SS.__init__(**SS._outspec)
# # generate new planets if requested (default)
# # if genNewPlanets:
# #     SU.gen_physical_properties(**SU._outspec)
# #     rewindPlanets = True
# # re-initialize systems if requested (default)
# #if rewindPlanets:

# #SU.init_systems()


# SU.I = DRM3systems['I']
# SU.M0 = DRM3systems['M0']
# SU.Mp = DRM3systems['Mp']
# SU.O = DRM3systems['O']
# SU.Rp = DRM3systems['Rp']
# SU.a = DRM3systems['a']
# SU.e = DRM3systems['e']
# SU.mu = DRM3systems['mu']
# SU.p = DRM3systems['p']
# SU.plan2star = DRM3systems['plan2star']
# SU.star = DRM3systems['star']
# SU.w = DRM3systems['w']

# #reset helper arrays
# SS.initializeStorageArrays()
# SS.vprint("Simulation reset.")
# SS.run_sim()
# DRM4 = SS.DRM

from pylab import *
fig = figure(1)
error = list()
atime = list()
EOT3 = list()
for i in np.arange(len(DRM3)):
    EOT3.append(DRM3[i]['exoplanetObsTime'].value)
    error.append(DRM3[i]['exoplanetObsTime'].value-DRM4[i]['exoplanetObsTime'].value)
    atime.append(DRM3[i]['arrival_time'].value-DRM4[i]['arrival_time'].value)
plot([0 for i in range(len(error))],error,marker='o')
#plot(EOT3)
show(block=False)



for i in np.arange(len(DRM3)):
    for key1 in DRM3[i].keys():
        if type(DRM3[i][key1]) == type(1) or type(DRM3[i][key1]) == type(1.0) or type(DRM3[i][key1]) == type(np.int64(1)):#is an integer of float
            assert(DRM3[i][key1] == DRM4[i][key1])
        elif type(DRM3[i][key1]) == type(np.asarray([1,2,3,4])):#is numpy array
            assert(all(DRM3[i][key1] == DRM4[i][key1]))
        elif hasattr(DRM3[i][key1],'value'):
            assert DRM3[i][key1].value == DRM4[i][key1].value, "key: %s, i: %d, values: %f, %f"%(key1, i, DRM3[i][key1].value, DRM4[i][key1].value)
        elif hasattr(DRM3[i][key1], 'keys'):
            for key2 in DRM3[i][key1]:
                if type(DRM3[i][key1][key2]) == type(np.asarray([1,2,3,4])):#is numpy array
                    assert(all(DRM3[i][key1][key2] == DRM4[i][key1][key2]))
                elif hasattr(DRM3[i][key1][key2],'value'):
                    if type(DRM3[i][key1][key2].value) == type(np.asarray([1,2,3,4])):#is numpy array
                        assert(all(DRM3[i][key1][key2].value == DRM4[i][key1][key2].value))
                    else:
                        assert(DRM3[i][key1][key2].value == DRM4[i][key1][key2].value)
                else:
                    assert(DRM3[i][key1][key2] == DRM4[i][key1][key2])
        elif type(DRM3[i][key1]) == type('taco'):
            assert(DRM3[i][key1] == DRM4[i][key1])
        else:
            print DRM3[i][key1]
            print i
            print key1


