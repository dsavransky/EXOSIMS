"""
Top-level run script for IPCluster parallel Queue implementation
Run as:
    python runQueue.py --numCores 16 --EXOSIMS_QUEUE_FILE_PATH '/Full path/queuefile.json'                 (optional but strongly suggested)
        --outpath '/dirToCreateAllOutputFolders/'                           (optional)
        --EXOSIMS_SCRIPTS_PATH '/Full path to directory containing Scripts/*.json')  (optional)
        --EXOSIMS_RUN_LOG_PATH '/Full path to directory to/runLog.csv')               (optional)


This script is designed to load the JSON file specified in the --qFPath argument.
runQueue will then attempt to find an outpathCore, EXOSIMS_SCRIPTS_PATH, and EXOSIMS_RUN_LOG_PATH in said JSON file.
If additional arguments are passed in, these will overload anything in the JSON file.
If no keys exist in the JSON file and No arguments are passed in, the '../../../cache/' folder will be searched.
The --qFPath file must contain a list of 'scriptNames' and 'numRuns'.
Written by Dean Keithly 4/27/2018
Updated 10/11/2018
Updated 11/26/2018
Updated 5/22/2021
"""
import json
import os
import numpy as np
import EXOSIMS
import EXOSIMS.MissionSim
import os
import os.path
import sys
import pickle
import time
import random
import argparse
import traceback
from EXOSIMS.util.vprint import vprint as tvprint
from EXOSIMS.util.get_dirs import get_paths
import subprocess

vprint = tvprint(True)


def run_one(genNewPlanets=True, rewindPlanets=True, outpath="."):
    # wrap the run_sim in a try/except loop
    # reset simulation at the end of each simulation
    # NOTE: Methods are imported from IPClusterEnsemble sync_imports function line
    SS.reset_sim(genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets)
    nbmax = 10
    for attempt in numpy.arange(nbmax):
        try:
            # run one survey simulation
            SS.run_sim()
            DRM = SS.DRM[:]
            systems = SS.SimulatedUniverse.dump_systems()
            systems["MsTrue"] = SS.TargetList.MsTrue
            systems["MsEst"] = SS.TargetList.MsEst
            seed = SS.seed
        except Exception as e:
            # if anything goes wrong, log the error and reset simulation
            with open(os.path.join(outpath, "log.err"), "ab") as f:
                f.write(repr(e))
                f.write("\n")
                f.write(traceback.format_exc())
                f.write("\n\n")

            SS.reset_sim(genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets)
        else:
            break
    else:
        raise ValueError("Unsuccessful run_sim after %s reset_sim attempts" % nbmax)

    # reset simulation at the end of each simulation
    SS.reset_sim(genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets)

    pklname = (
        "run"
        + str(int(time.clock() * 100))
        + "".join(["%s" % random.randint(0, 9) for num in numpy.arange(5)])
        + ".pkl"
    )
    pklpath = os.path.join(outpath, pklname)
    with open(os.path.join(outpath, "pickles.txt"), "w+") as f:
        f.write(pklpath)
        f.write(pklname)
        f.write("\n")
    with open(pklpath, "wb") as f:
        pickle.dump({"DRM": DRM, "systems": systems, "seed": seed}, f)

    return 0


def scriptNamesInScriptPath(queueData, ScriptsPath):
    # This function searches the ScriptsPath to determine if files are at the current level or 1 level down
    scriptfile = queueData["scriptNames"][
        0
    ]  # just grab first script file in list of .json files
    makeSimilar_TemplateFolder = ""
    if not os.path.isfile(ScriptsPath + scriptfile):
        dirsFolderDown = [
            x[0].split("/")[-1] for x in os.walk(ScriptsPath)
        ]  # Get all directories in ScriptsPath
        # print(dirsFolderDown)
        for tmpFolder in dirsFolderDown:
            if os.path.isfile(
                ScriptsPath + tmpFolder + "/" + scriptfile
            ):  # We found the Scripts folder containing scriptfile
                # print(ScriptsPath + tmpFolder + '/' + scriptfile)
                makeSimilar_TemplateFolder = tmpFolder + "/"
                break
    assert os.path.isfile(ScriptsPath + makeSimilar_TemplateFolder + scriptfile), (
        "Scripts Path: %s does not exist" % ScriptsPath
    )
    return makeSimilar_TemplateFolder, scriptfile


def extractArgs(args):
    """Convert from args to a dict of parsed arguments of form {'EXOSIMS_RUN_SAVE_PATH':'/home/user/Doc...'}
    Args:
        args (parser.parse_args()) - the output from parser.parse_args
    Returns:
        EXOSIMS_QUEUE_FILE_PATH (string) - full file path to the queue file
        numCoresString (string) - string of the number of cores to run ipcluster with
        qFargs (dict) - dictionary of paths from parsed runQueue arguments of form {'EXOSIMS_RUN_SAVE_PATH':'/home/user/Doc...'}
    """
    myArgs = [
        arg
        for arg in args.__dict__.keys()
        if "EXOSIMS" in arg and not args.__getattribute__(arg) == None
    ]
    paths = dict()
    for arg in myArgs:
        paths[arg] = args.__dict__[arg][0]

    EXOSIMS_QUEUE_FILE_PATH = paths["EXOSIMS_QUEUE_FILE_PATH"]
    if args.numCores == None:
        args.numCores == ["1"]
    numCoresString = str(int(args.numCores[0]))
    return EXOSIMS_QUEUE_FILE_PATH, numCoresString, paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an ipcluster parallel ensemble job queue."
    )
    parser.add_argument(
        "--EXOSIMS_RUN_SAVE_PATH",
        nargs=1,
        type=str,
        help="Full path to output directory where each job directory will be created and pkl saved (string).",
    )
    parser.add_argument(
        "--EXOSIMS_SCRIPTS_PATH",
        nargs=1,
        type=str,
        help="Full path to directory containing run .json scripts (string).",
    )
    parser.add_argument(
        "--EXOSIMS_RUN_LOG_PATH",
        nargs=1,
        type=str,
        help="Full path to directory to log simulations run (string).",
    )
    parser.add_argument(
        "--EXOSIMS_QUEUE_FILE_PATH",
        nargs=1,
        type=str,
        help="Full path to the queue file (string).",
    )
    parser.add_argument(
        "--numCores",
        nargs=1,
        type=str,
        help="Number of Cores to run ipcluster with (int).",
    )
    args = parser.parse_args()

    EXOSIMS_QUEUE_FILE_PATH, numCoresString, qFargs = extractArgs(args)

    #### Parse queue file
    assert os.path.isfile(EXOSIMS_QUEUE_FILE_PATH), (
        "Queue File Path: %s does not exist" % EXOSIMS_QUEUE_FILE_PATH
    )
    with open(EXOSIMS_QUEUE_FILE_PATH) as queueFile:
        queueData = json.load(queueFile)

    #### Get all paths
    paths = get_paths(qFile=queueData, specs=None, qFargs=qFargs)

    makeSimilar_TemplateFolder, scriptfile = scriptNamesInScriptPath(
        queueData, paths["EXOSIMS_SCRIPTS_PATH"]
    )  # Check if scriptNames in EXOSIMS_SCRIPTS_PATH

    ####Check if Any of the Scripts have already been run... and remove from scriptNames list ##########
    try:  # check through log file if it exists
        with open(
            os.path.join(paths["EXOSIMS_RUN_LOG_PATH"], "runLog.csv"), "w+"
        ) as myfile:
            scriptsRun = map(lambda s: s.strip(), myfile.readlines())
            for scriptRun in scriptsRun:
                if scriptRun in queueData["scriptNames"]:
                    tmpIndex = queueData["scriptNames"].index(scriptRun)
                    queueData["scriptNames"].remove(
                        scriptRun
                    )  # remove scriptfile from list
                    queueData["numRuns"].remove(
                        queueData["numRuns"][tmpIndex]
                    )  # remove numRuns from list
    except:
        pass
    ####################################################################################################

    #### Run over all Scripts in Queue #################################################################
    while len(queueData["scriptNames"]) > 0:  # Iterate until there are no more
        # TODO Check if ipcluster is running
        # Start IPCluster
        startIPClusterCommand = subprocess.Popen(
            ["ipcluster", "start", "-n", numCoresString]
        )
        time.sleep(80)
        vprint(startIPClusterCommand.stdout)

        outpath = paths["EXOSIMS_RUN_SAVE_PATH"] + str(
            queueData["scriptNames"][0].split(".")[0]
        )
        if not os.path.isdir(outpath):  # IF the directory doesn't exist
            os.makedirs(outpath)  # make directory

        scriptfile = queueData["scriptNames"][
            0
        ]  # pull first script name (will remove from list at end)
        vprint(scriptfile)
        numRuns = queueData["numRuns"][0]  # pull first number of runs
        sim = EXOSIMS.MissionSim.MissionSim(
            paths["EXOSIMS_SCRIPTS_PATH"] + makeSimilar_TemplateFolder + scriptfile
        )
        res = sim.genOutSpec(tofile=os.path.join(outpath, "outspec.json"))
        vprint(res)
        del res
        vprint(outpath)
        kwargs = {"outpath": outpath}
        numRuns = queueData["numRuns"][0]
        res = sim.run_ensemble(numRuns, run_one=run_one, kwargs=kwargs)

        # Append ScriptName to logFile.csv
        with open(
            os.path.join(paths["EXOSIMS_RUN_LOG_PATH"], "runLog.csv"), "a"
        ) as myfile:
            myfile.write(queueData["scriptNames"][0] + "\n")

        queueData["scriptNames"].remove(
            queueData["scriptNames"][0]
        )  # remove scriptfile from list
        queueData["numRuns"].remove(queueData["numRuns"][0])  # remove numRuns from list
        del sim  # required otherwise data can be passed between sim objects (observed when running e2eTests.py)
        del res, scriptfile, numRuns, kwargs  # deleting these as well as a percaution

    # Stop IPCluster
    stopIPClusterCommand = subprocess.Popen(["ipcluster", "stop"])
    stopIPClusterCommand.wait()
    time.sleep(80)

    vprint("Done running all jobs")
