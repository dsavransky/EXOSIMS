"""
Top-level run script for IPCluster parallel Queue implementation
Run as:
    python runQueue.py --outpath '/dirToCreateAllOutputFolders/'

This script is designed to load the queue.json file located in the "run" folder.
queue.json contains the names of .json files located in ../Scripts/. to be run by this script
This script then runs .json file named in queue.json for the number of runs specified in queue.json
This script should be executed from the "run" folder
Written by Dean Keithly 4/27/2018
"""
import json
from subprocess import Popen
import subprocess
import os
import numpy as np
import EXOSIMS
import EXOSIMS.MissionSim
import os
import os.path
import cPickle
import time
import random
import argparse
import traceback

def run_one(genNewPlanets=True, rewindPlanets=True, outpath='.'):
    # wrap the run_sim in a try/except loop
    nbmax = 10
    for attempt in range(nbmax):
        try:
            # run one survey simulation
            SS.run_sim()
            DRM = SS.DRM[:]
            systems = SS.SimulatedUniverse.dump_systems()
            systems['MsTrue'] = SS.TargetList.MsTrue
            systems['MsEst'] = SS.TargetList.MsEst
            seed = SS.seed
        except Exception as e:
            # if anything goes wrong, log the error and reset simulation
            with open(os.path.join(outpath,'log.err'), 'ab') as f:
                f.write(repr(e))
                f.write('\n')
                f.write(traceback.format_exc())
                f.write('\n\n')
            
            SS.reset_sim(genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets)
        else:
            break
    else:
        raise ValueError("Unsuccessful run_sim after %s reset_sim attempts"%nbmax)
    
    # reset simulation at the end of each simulation
    SS.reset_sim(genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets)  

    pklname = 'run'+str(int(time.clock()*100))+''.join(["%s" % random.randint(0, 9) for num in range(5)]) + '.pkl'
    pklpath = os.path.join(outpath, pklname)
    with open(pklpath, 'wb') as f:
        cPickle.dump({'DRM':DRM,'systems':systems,'seed':seed}, f)
        
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an ipcluster parallel ensemble job queue.")
    parser.add_argument('--outpath',nargs=1,type=str, help='Full path to output directory where each job directory will be saved (string).')

    args = parser.parse_args()

    queueName = 'queue.json'
    with open('queue.json') as queueFile:
        queueData = json.load(queueFile)

    error = False
    #TODO ERROR CHECKING
    
    #Check if Any of the Scripts have already been run...
    with open("runLog.csv") as myfile:
        scriptsRun = map(lambda s: s.strip(), myfile.readlines())
        for scriptRun in scriptsRun:
            if scriptRun in queueData['scriptNames']:
                tmpIndex = queueData['scriptNames'].index(scriptRun)
                queueData['scriptNames'].remove(scriptRun)#remove scriptfile from list
                queueData['numRuns'].remove(queueData['numRuns'][tmpIndex])#remove numRuns from list


    if error == False:
        while(len(queueData['scriptNames']) > 0):#Iterate until there are no more 
            outpath = args.outpath[0] + str(queueData['scriptNames'][0].split('.')[0])
            if not os.path.isdir(outpath):#IF the directory doesn't exist
                os.makedirs(outpath)#make directory

            scriptfile = queueData['scriptNames'][0]
            numRuns = queueData['numRuns'][0]
            sim = EXOSIMS.MissionSim.MissionSim('../Scripts/' + scriptfile)
            res = sim.genOutSpec(tofile = os.path.join(outpath,'outspec.json'))
            kwargs = {'outpath':outpath}
            numRuns = queueData['numRuns'][0]
            res = sim.run_ensemble(numRuns, run_one=run_one, kwargs=kwargs)

            #Append ScriptName to logFile.csv
            with open("runLog.csv", "a") as myfile:
                myfile.write(queueData['scriptNames'][0] + '\n')

            queueData['scriptNames'].remove(queueData['scriptNames'][0])#remove scriptfile from list
            queueData['numRuns'].remove(queueData['numRuns'][0])#remove numRuns from list
        print('Done running all jobs')
    else:
        print('There was an error and the runs did not execute')
        pass