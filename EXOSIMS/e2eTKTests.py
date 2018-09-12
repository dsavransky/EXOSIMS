import sys, os.path, EXOSIMS, EXOSIMS.MissionSim
basedir = EXOSIMS.__path__[0]
folder = os.path.join(basedir, 'Scripts', 'TestScripts')
#folder = os.path.normpath(os.path.expandvars('$HOME/Documents/exosims/EXOSIMS/EXOSIMS/Scripts'))
#filename = 'sS_AYO6.json'#'sS_protoTimeKeeping.json'#'sS_AYO3.json'#sS_SLSQPstatic_parallel_ensembleJTWIN.json'#'sS_JTwin.json'#'sS_AYO4.json'#'sS_AYO3.json'
#filename = 'sS_intTime6_KeplerLike2.json'

#Test OB loaded from File0
filename = 'TKtestingOBfromFile0.json' #Load from File sampleOB.csv
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
sim.run_sim()
print('ran TKtestingOBfromFile0')

#Test OB loaded from File1
filename = 'TKtestingOBfromFile1.json' #Load from File sampleOB.csv
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
sim.run_sim()
print('ran TKtestingOBfromFile1')

#Test OB loaded from File2
filename = 'TKtestingOBfromFile2.json' #Load from File sampleOB.csv
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
sim.run_sim()
print('ran TKtestingOBfromFile2')

#Test OB loaded from File3
filename = 'TKtestingOBfromFile3.json' #Load from File sampleOB.csv
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
sim.run_sim()
print('ran TKtestingOBfromFile3')

#Test OB loaded from File4
filename = 'TKtestingOBfromFile4.json' #Load from File sampleOB.csv
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
sim.run_sim()
print('ran TKtestingOBfromFile4')

#Test OB from missionPortion and missionLife
filename = 'TKtestingOBfromJSON.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
sim.run_sim()
print('ran TKtestingOBfromJSON')

#Test Single Block mission full length of mission
filename = 'TKtestingSingleOBFullPortion.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
sim.run_sim()
print('ran TKtestingSingleOBFullPortion')

#Test Single Block mission Portion of mission
filename = 'TKtestingSingleOBHalfPortion.json'
scriptfile = os.path.join(folder,filename)
sim = EXOSIMS.MissionSim.MissionSim(scriptfile)
sim.run_sim()
print('ran TKtestingSingleOBHalfPortion')

#print(saltyburrito)
