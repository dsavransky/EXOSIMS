"""
Top-level run script for IPCluster parallel Queue implementation
Run as:
    python runQueue.py --qFPath '/Full path/queuefile.json'                 (optional but strongly suggested)
        --outpath '/dirToCreateAllOutputFolders/'                           (optional)
        --ScriptsPath '/Full path to directory containing Scripts/*.json')  (optional)
        --runLogPath '/Full path to directory to/runLog.csv')               (optional)


This script is designed to load the queue.json file located in the "run" folder.
queue.json contains the names of .json files located in ../Scripts/. to be run by this script
This script then runs .json file named in queue.json for the number of runs specified in queue.json
This script should be executed from the "run" folder
Written by Dean Keithly 4/27/2018
"""
import json
import os
import numpy as np
import EXOSIMS
import EXOSIMS.MissionSim
import os
import os.path
import sys
import cPickle
import time
import random
import argparse
import traceback
from EXOSIMS.util.vprint import vprint as tvprint

vprint = tvprint(True)

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

def parse_qPath(args):
    if args.qFPath is None:
        qPath = '../../../cache/queue.json'#Default behavior
    else:
        qPath = args.qPath[0]
    assert os.path.isfile(qPath), 'Queue Path: %s does not exist' %qPath
    qfname = qPath.split('/')[-1]
    #Load Queue File
    with open(qPath) as queueFile:
        queueData = json.load(queueFile)
    return qPath, qfname, queueData

def parse_ScriptsPath(args, queueData):
    if args.ScriptsPath is None:
        ScriptsPath = '../../../Scripts/'#Default. Explicitly when run from run folder
        if queueData.has_key('ScriptsPath'):
            ScriptsPath = queueData['ScriptsPath'] #extract from queue Folder
    else:
        ScriptsPath = args.ScriptsPath[0]
    return ScriptsPath

def parse_runLogPath(args,queueData):
    if args.runLogPath is None:
        runLogPath = '../../../cache/'#Default
        if queueData.has_key('runLogPath'):
            runLogPath = queueData['runLogPath'] #extract from queue Folder
    else:
        runLogPath = args.runLogPath[0]
    assert os.path.isfile(runLogPath), 'runLog Path: %s does not exist' %runLogPath
    return runLogPath

def scriptNamesInScriptPath(queueData, ScriptsPath):
    #This function searches the ScriptsPath to determine if files are at the current level or 1 level down
    scriptfile = queueData['scriptNames'][0] #just grab first script file in list of .json files
    makeSimilar_TemplateFolder = ''
    if not os.path.isfile(ScriptsPath + scriptfile):
        dirsFolderDown = [x[0].split('/')[-1] for x in os.walk(ScriptsPath)] #Get all directories in ScriptsPath
        #print(dirsFolderDown)
        for tmpFolder in dirsFolderDown:
            if os.path.isfile(ScriptsPath + tmpFolder + '/' + scriptfile):#We found the Scripts folder containing scriptfile
                #print(ScriptsPath + tmpFolder + '/' + scriptfile)
                makeSimilar_TemplateFolder = tmpFolder + '/'
                break
    assert os.path.isfile(ScriptsPath + makeSimilar_TemplateFolder + scriptfile), 'Scripts Path: %s does not exist' %ScriptsPath
    return makeSimilar_TemplateFolder, scriptfile

def outpathCore(args,queueData):
    if args.outpathCore is None:
        outpathCore = '../../../cache/'#Default
        if queueData.has_key('outpath'):
            outpathCore = queueData['outpath'] #extract from queue Folder
    else:
        outpathCore = args.outpathCore[0]
    assert os.path.isfile(outpathCore), 'runLog Path: %s does not exist' %outpathCore
    return outpathCore

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an ipcluster parallel ensemble job queue.")
    parser.add_argument('--outpath',nargs=1,type=str, help='Full path to output directory where each job directory will be saved (string).')
    parser.add_argument('--ScriptsPath',nargs=1,type=str, help='Full path to directory containing run .json scripts (string).')
    parser.add_argument('--runLogPath',nargs=1,type=str, help='Full path to directory to log simulations run (string).')
    parser.add_argument('--qFPath',nargs=1,type=str, help='Full path to the queue file (string).')
    args = parser.parse_args()

    qPath, qfname, queueData = parse_qPath(args) # Parse the queue full filepath, filename, and Json Data
    ScriptsPath = parse_ScriptsPath(args, queueData) # ScriptsPath (folder containing all scripts)
    runLogPath = parse_runLogPath(args,queueData) # runLogPath (full path to folder containing runLog.csv)
    makeSimilar_TemplateFolder, scriptfile = scriptNamesInScriptPath(queueData, ScriptsPath) # Check if scriptNames in ScriptsPath
    outpathCore = outpathCore(args, queueData) # parse the outpath from user input or queuefile

    ####Check if Any of the Scripts have already been run... and remove from scriptNames list ##########
    try:#check through log file if it exists
        with open(runLogPath + "runLog.csv","w+") as myfile:
            scriptsRun = map(lambda s: s.strip(), myfile.readlines())
            for scriptRun in scriptsRun:
                if scriptRun in queueData['scriptNames']:
                    tmpIndex = queueData['scriptNames'].index(scriptRun)
                    queueData['scriptNames'].remove(scriptRun)#remove scriptfile from list
                    queueData['numRuns'].remove(queueData['numRuns'][tmpIndex])#remove numRuns from list
    except:
        pass
    ####################################################################################################

    #### Run over all Scripts in Queue #################################################################
    while(len(queueData['scriptNames']) > 0): # Iterate until there are no more 
        outpath = args.outpath[0] + str(queueData['scriptNames'][0].split('.')[0])
        if not os.path.isdir(outpath): # IF the directory doesn't exist
            os.makedirs(outpath) # make directory

        scriptfile = queueData['scriptNames'][0] # pull first script name (will remove from list at end)
        numRuns = queueData['numRuns'][0] # pull first number of runs
        sim = EXOSIMS.MissionSim.MissionSim(ScriptsPath + makeSimilar_TemplateFolder + scriptfile)
        res = sim.genOutSpec(tofile = os.path.join(outpath,'outspec.json'))
        kwargs = {'outpath':outpath}
        numRuns = queueData['numRuns'][0]
        res = sim.run_ensemble(numRuns, run_one=run_one, kwargs=kwargs)

        #Append ScriptName to logFile.csv
        with open(runLogPath + "runLog.csv", "a") as myfile:
            myfile.write(queueData['scriptNames'][0] + '\n')

        queueData['scriptNames'].remove(queueData['scriptNames'][0])#remove scriptfile from list
        queueData['numRuns'].remove(queueData['numRuns'][0])#remove numRuns from list
    vprint('Done running all jobs')
