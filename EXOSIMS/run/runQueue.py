#This script is designed to load the queue.json file located in the "run" folder.
#queue.json contains the names of .json files located in ../Scripts/. to be run by this script
#This script then runs .json file named in queue.json for 1000 runs
#This script should be executed from the "run" folder
#Written by Dean Keithly 4/27/2018
"""
Top-level run script for IPCluster parallel implementation.
Run as:
    python run_ipcluster_ensemble scriptname #runs
Run:
    python run_ipcluster_ensemble --help 
for detailed usage.

Notes:  
1) It is always advisable to run a script with the prototype 
SurveyEnsemble BEFORE running a parallel job to allow EXOSIMS to
cache all pre-calcualted products.
2) An ipcluster instance must be running and accessible in order
to use this script.  If everything is already set up and configured
properly, this is usually a matter of just executing:
    ipcluster start
from the command line.
3) The emailing setup assumes a gmail address.  If you're using a 
different SMTP, or wish to send/receive on different accounts, modify
the email setup.
4) If an output directory is reused, new run files will be added, but
the outspec.json file will be ovewritten.  Any generated errors will be
appended to any exisitng log.err file.
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
            seed = SS.seed
        except Exception as e:
            # if anything goes wrong, log the error and reset simulation
            with open(os.path.join(outpath,'log.err'), 'ab') as f:
                f.write(repr(e))
                f.write('\n')
                f.write(traceback.format_exc())
                f.write('\n\n')
            
            SS.reset_sim()
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
    #parser.add_argument('scriptfile', nargs=1, type=str, help='Full path to scriptfile (string).')
    # parser.add_argument('numruns', nargs=1, type=int, help='Number of runs (int).')
    parser.add_argument('--outpath',nargs=1,type=str, help='Full path to output directory where each job directory will be saved (string).')
    # parser.add_argument('--email',nargs=1,type=str,help='Email address to notify when run is complete.')
    # parser.add_argument('--toemail',nargs=1,type=str,help='Additional email to notify when run is complete.')

    args = parser.parse_args()

    queueName = 'queue.json'
    with open('queue.json') as queueFile:
        queueData = json.load(queueFile)

    error = False
    #DO SOME ERROR CHECKING

    if error == False:
        while(len(queueData['scriptNames']) > 0):#Iterate until there are no more 
            outpath = args.outpath[0] + str(queueData['scriptNames'][0].split('.')[0])
            #outpath = '/home/dean/' + queueData['scriptNames'][0].split('.')[0]
            #on atuin outpath = '/data2/extmount/EXOSIMSres/' + queueData['scriptNames'][0].split('.')[0]
            if not os.path.isdir(outpath):#IF the directory doesn't exist
                os.makedirs(outpath)#make directory

            scriptfile = queueData['scriptNames'][0]
            numRuns = queueData['numRuns'][0]
            sim = EXOSIMS.MissionSim.MissionSim('../Scripts/' + scriptfile)
            res = sim.genOutSpec(tofile = os.path.join(outpath,'outspec.json'))
            kwargs = {'outpath':outpath}
            numRuns = queueData['numRuns'][0]
            res = sim.run_ensemble(numRuns, run_one=run_one, kwargs=kwargs)

            queueData['scriptNames'].remove(queueData['scriptNames'][0])#remove scriptfile from list
            queueData['numRuns'].remove(queueData['numRuns'][0])#remove numRuns from list
        print('Done running all jobs')
    else:
        print('There was an error and the runs did not execute')
        pass