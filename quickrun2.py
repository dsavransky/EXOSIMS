import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
import numpy as np
import json
import pickle
# folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts/HabExCompSpecPriors_HabEx_4m_TSDD_pop100DD_revisit_20180424/'))#EXOSIMS/EXOSIMS/Scripts'))#EXOSIMS/EXOSIMS/Scripts'))
# filename = 'HabEx_CKL2_PPKL2.json'#'Dean3June18RS26CXXfZ01OB66PP01SU01.json'#'Dean1June18RS26CXXfZ01OB56PP01SU01.json'#'./TestScripts/04_KeplerLike_Occulter_linearJScheduler.json'#'Dean13May18RS09CXXfZ01OB01PP03SU01.json'#'sS_AYO7.json'#'ICDcontents.json'###'sS_protoTimeKeeping.json'#'sS_AYO3.json'#sS_SLSQPstatic_parallel_ensembleJTWIN.json'#'sS_JTwin.json'#'sS_AYO4.json'#'sS_AYO3.json'
#filename = 'sS_intTime6_KeplerLike2.json'
# folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/Scripts/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_40519/'))
# filename = 'WFIRSTcycle6core_CKL2_PPKL2.json'
folder = os.path.normpath(os.path.expandvars('$HOME/Documents/GitHub/EXOSIMS/scripts/WFIRSTCompSpecPriors_WFIRSTcycle6core_3mo_40519'))
filename = 'WFIRSTcycle6core_CKL2_PPKL2.json'

scriptfile = os.path.join(folder,filename)

base_folder = '$HOME/Documents/GitHub/EXOSIMS/simulations'
dynamic_folder = 'dynamic'

folder_name = os.path.join(os.path.normpath(os.path.expandvars(base_folder)),dynamic_folder)

sim = EXOSIMS.MissionSim.MissionSim(scriptfile,nopar=True)
# sims = 100
# for i in range(0,sims):
#     DRM_filename = 'sim_'+str(i)+'.p'
#     sim_file = os.path.join(folder_name,DRM_filename)
#     sim.run_sim()
#     DRM = sim.SurveySimulation.DRM
#     pickle.dump(DRM, open(sim_file, 'wb'))
#     sim.reset_sim(genNewPlanets = True, rewindPlanets = True)
# sim.run_ensemble(sims, genNewPlanets = True, rewindPlanets = True, outpath=folder_name)

# DRM = sim.SurveySimulation.DRM
# DRM2 = sim.DRM2array

# # sim.genOutSpec(tofile='outspec.json')

# # Look for planets with the most revisits
# visits = sim.SurveySimulation.starVisits
# max_visits = np.max(visits)
# sInd = np.where(visits == max_visits)[0][0]

# obs_dict = {}
# det_dict = {}

# for i in range(len(DRM)):
#     star_ind = DRM[i]['star_ind']
#     if DRM[i]['det_status'].size == 0:
#         # If the star has no planets around it
#         star_det = 0
#     else:
#         if 1 in DRM[i]['det_status']:    
#             # If a detection was made
#             star_det = 1
#         else:
#             star_det = 0
#     if star_ind not in obs_dict:
#         # Initializing the values for the dictionaries
#         obs_dict[star_ind] = 0
#         det_dict[star_ind] = 0
    
#     obs_dict[star_ind] += 1 # Counts observations per star
#     det_dict[star_ind] += star_det # Counts detections per star
    
# total_detections = sum(det_dict.values())
# total_observations = sum(obs_dict.values())

# # Calculate the number of stars visited multiple times
# revisited_stars = 0
# revisit_detections = 0
# for key in obs_dict:
#     if obs_dict[key] > 1:
#         revisited_stars +=1
        

#FOR TESTING RESET ISSUES############################
# import numpy as np
# for i in range(1,100):
#     sim.run_sim()
#     sim.reset_sim()
#     print 'Print Current Abs Time: ' + str(sim.TimeKeeping.currentTimeAbs)
#     print 'AbsTimefZmin: ' + str(min(sim.SurveySimulation.absTimefZmin))
#     #sInds = np.arange(sim.TargetList.nStars)
#     #tmp, sim.SurveySimulation.absTimefZmin = sim.ZodiacalLight.calcfZmin(sInds, sim.Observatory, sim.TargetList, sim.TimeKeeping, sim.SurveySimulation.mode, sim.SurveySimulation.cachefname) # find fZmin to use in intTimeFilter
#     print 'AbsTimefZmin: ' + str(min(sim.SurveySimulation.absTimefZmin))



# #Sum total mission time...########
# import numpy as np
# import astropy.units as u
# DRM = sim.SurveySimulation.DRM

# det_times = list()
# eOt = list()
# arrival_times = list()
# for i in np.arange(len(DRM)):
#     det_times.append(DRM[i]['det_time'].value)
#     eOt.append(DRM[i]['exoplanetObsTime'].value)
#     arrival_times.append(DRM[i]['arrival_time'].value)

# sum(det_times) + len(DRM)
# ####################

# from pylab import *
# try:
#     plt.close('all')
# except:
#     pass
# fig = figure(num=1)
# plot(arrival_times[1:], eOt[:-1])
# axis('equal')
# ylabel('exoplanetObsTime')
# xlabel('arrivalTime')
# plt.show(block=False)


# ###CHECK IF ANY CHARACTERIZATIONS WERE MADE
# chars = [x for x in DRM if x['char_time'].value > 0]
# print chars







# arrival_times = [DRM[i]['arrival_time'].value for i in np.arange(len(DRM))]
# sumOHTIME = 1
# det_times = [DRM[i]['det_time'].value+sumOHTIME for i in np.arange(len(DRM))]
# det_timesROUNDED = [round(DRM[i]['det_time'].value+sumOHTIME,1) for i in np.arange(len(DRM))]
# ObsNums = [DRM[i]['ObsNum'] for i in np.arange(len(DRM))]
# y_vals = np.zeros(len(det_times)).tolist()
# char_times = [DRM[i]['char_time'].value*(1+sim.SurveySimulation.charMargin)+sumOHTIME for i in np.arange(len(DRM))]
# OBdurations = np.asarray(sim.TimeKeeping.OBendTimes-sim.TimeKeeping.OBstartTimes)
# #sumOHTIME = [1 for i in np.arange(len(DRM))]
# print(sum(det_times))
# print(sum(char_times))


# #Check if plotting font #########################################################
# tmpfig = plt.figure(figsize=(30,3.5),num=0)
# ax = tmpfig.add_subplot(111)
# t = ax.text(0, 0, "Obs#   ,  d", ha='center',va='center',rotation='vertical', fontsize=8)
# r = tmpfig.canvas.get_renderer()
# bb = t.get_window_extent(renderer=r)
# Obstxtwidth = bb.width#Width of text
# Obstxtheight = bb.height#height of text
# FIGwidth, FIGheight = tmpfig.get_size_inches()*tmpfig.dpi
# plt.show(block=False)
# plt.close()
# daysperpixelapprox = max(arrival_times)/FIGwidth#approximate #days per pixel
# if mean(det_times)*0.8/daysperpixelapprox > Obstxtwidth:
#     ObstextBool = True
# else:
#     ObstextBool = False

# tmpfig = plt.figure(figsize=(30,3.5),num=0)
# ax = tmpfig.add_subplot(111)
# t = ax.text(0, 0, "OB#  , dur.=    d", ha='center',va='center',rotation='horizontal', fontsize=12)
# r = tmpfig.canvas.get_renderer()
# bb = t.get_window_extent(renderer=r)
# OBtxtwidth = bb.width#Width of text
# OBtxtheight = bb.height#height of text
# FIGwidth, FIGheight = tmpfig.get_size_inches()*tmpfig.dpi
# plt.show(block=False)
# plt.close()
# if mean(OBdurations)*0.8/daysperpixelapprox > OBtxtwidth:
#     OBtextBool = True
# else:
#     OBtextBool = False
# #################################################################################



# colors = 'rb'#'rgbwmc'
# patch_handles = []
# fig = plt.figure(figsize=(30,3.5),num=1)

# # Plot All Detection Observations
# ind = 0
# obs = 0
# for (det_time, l, char_time) in zip(det_times, ObsNums, char_times):
#     #print det_time, l
#     patch_handles.append(ax.barh(0, det_time, align='center', left=arrival_times[ind],
#         color=colors[int(obs) % len(colors)]))
#     if not char_time == 0:
#         ax.barh(0, char_time, align='center', left=arrival_times[ind]+det_time,color=(255/255.,69/255.,0/255.))
#     ind += 1
#     obs += 1
#     patch = patch_handles[-1][0] 
#     bl = patch.get_xy()
#     x = 0.5*patch.get_width() + bl[0]
#     y = 0.5*patch.get_height() + bl[1]
#     plt.rc('axes',linewidth=2)
#     plt.rc('lines',linewidth=2)
#     rcParams['axes.linewidth']=2
#     rc('font',weight='bold')
#     if ObstextBool: 
#         ax.text(x, y, "Obs#%d, %dd" % (l,det_time), ha='center',va='center',rotation='vertical', fontsize=8)

# # Plot Observation Blocks
# patch_handles2 = []
# for (OBnum, OBdur, OBstart) in zip(xrange(len(OBdurations)), OBdurations, np.asarray(sim.TimeKeeping.OBstartTimes)):
#     patch_handles2.append(ax.barh(1, OBdur, align='center', left=OBstart, hatch='//',linewidth=2.0, edgecolor='black'))
#     patch = patch_handles2[-1][0] 
#     bl = patch.get_xy()
#     x = 0.5*patch.get_width() + bl[0]
#     y = 0.5*patch.get_height() + bl[1]
#     if OBtextBool:
#         ax.text(x, y, "OB#%d, dur.= %dd" % (OBnum,OBdur), ha='center',va='center',rotation='horizontal',fontsize=12)

# # Plot Asthetics
# y_pos = np.arange(2)#Number of xticks to have
# plt.rc('axes',linewidth=2)
# plt.rc('lines',linewidth=2)
# rcParams['axes.linewidth']=2
# rc('font',weight='bold') 
# ax.set_yticks(y_pos)
# ax.set_yticklabels(('Obs','OB'),fontsize=12)
# ax.set_xlabel('Current Normalized Time (days)', weight='bold',fontsize=12)
# #title('Mission Timeline for runName: ' + dirs[cnt] + '\nand pkl file: ' + pklfname[cnt], weight='bold',fontsize=12)
# plt.tight_layout()
# plt.show(block=False)
# #savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'Timeline' + '.png')
# #savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'Timeline' + '.svg')
# #savefig('/'.join(pklPaths[cnt].split('/')[:-1]) + '/' + dirs[cnt] + 'Timeline' + '.eps')

