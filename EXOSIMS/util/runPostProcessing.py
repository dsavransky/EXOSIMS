"""runPostProcessing.py


#To Run singleRunPostProcessing.py within an ipython session
%run singeRunPostPorcessing.py 
--queueFileFolder '/pathToDirContainingQueue*.json/' OR --queueFileFolder '/pathTo/queue*.json/'
####FIX --onMultiRun '/pathToDirContainingRunTypesFolders/'
####FIX --onSingleRun '/pathToDirContainingRunsTypesAndsinglePlotInstJSON/'

Note: the onMultiRun path should contain a series of run_type folders containing individual runs

Written By: Dean Keithly
Written On: 9/10/2018
"""
import os
if not 'DISPLAY' in os.environ.keys(): #Check environment for keys
    from matplotlib import *
    use('Agg')
from pylab import *
import shutil
import numpy as np
import argparse
import json
import datetime
import glob
import importlib
import EXOSIMS.util.get_module as get_module



def singleRunPostProcessing(SRPPdat,PPoutpath,outpath):
    """singelRunPostProcessing
    1. Imports all analysisName modules
    2. Creates instances of modules
    3. Runs all instances over all subfolders of outpath
    """
    #### Get File Path of All Folders to run PP on ########
    folders = glob.glob(os.path.join(PPoutpath,'*'))#List of all full folder filepaths of type queue in queueFileFolder
    #######################################################

    #### Import Single Run PP Analysis Modules and Create Instances ##################
    module = {}
    instance = {}
    for i in range(len(SRPPdat)):
        analysisScriptName = SRPPdat[i]['analysisName']
        analysisScriptNameCORE = analysisScriptName.split('.')[0]
        module[analysisScriptName] = get_module.get_module(analysisScriptName,'util')
        #DELETE module['analysisScriptNameCORE'] = importlib.import_module(analysisScriptNameCORE)
        if SRPPdat[i].has_key('args'):
            args = SRPPdat[i]['args']
        else:
            args = {}
        instance[analysisScriptName] = module[analysisScriptName](args)
    ##################################################################################

    #### Run Instances Over Each Run Type Folder #####################################
    for folder in folders:#iterate over each run
        for i in range(len(SRPPdat)):#iterate over each analysis
            analysisScriptName = SRPPdat[i]['analysisName']
            instance[analysisScriptName].singleRunPostProcessing(PPoutpath,folder) #singleRunPostProcessing method in plotting utility
    ##################################################################################
    return True

def multiRunPostProcessing(MRPPdat,PPoutpath,outpath):
    """singelRunPostProcessing
    1. Imports all analysisName modules
    2. Creates instances of modules
    3. Runs all instances over all subfolders of outpath
    """
    #### Get File Path of All Folders to run PP on ########
    folders = glob.glob(os.path.join(PPoutpath,'*'))#List of all full folder filepaths of type queue in queueFileFolder
    #######################################################

    #### Import Multi Run PP Analysis Modules and Create Instances ##################
    module = {}
    instance = {}
    for i in range(len(MRPPdat)):
        analysisScriptName = MRPPdat[i]['analysisName']
        analysisScriptNameCORE = analysisScriptName.split('.')[0]
        module[analysisScriptName] = get_module.get_module(analysisScriptName,'util')
        #DELETE module['analysisScriptNameCORE'] = importlib.import_module(analysisScriptNameCORE)
        if MRPPdat[i].has_key('args'):
            args = MRPPdat[i]['args']
        else:
            args = {}
        instance[analysisScriptName] = module[analysisScriptName](args)
    ##################################################################################

    #### Run Instance Once #####################################
    for i in range(len(MRPPdat)):#Iterate over each analysis
        analysisScriptName = MRPPdat[i]['analysisName']
        instance[analysisScriptName].multiRunPostProcessing(PPoutpath,folders) #multiRunPostProcessing method in plotting utility
    ##################################################################################
    return True

def queuePostProcessing(queueFileFolder):
    """
    Args:
        queueFileFolder (string) - Path to Folder containing queue*.json string
        alternate queueFileFolder (string) - Full Path to json file to use for determining post processing
    Returns:
        True - returns True if all analysis scripts executed
    """
    #### Extract queue Full File Path and queue File Name#################################################
    if os.path.isfile(queueFileFolder):#CASE 1
        queueFFP = queueFileFolder#queue Full File Path
        queueFile = queueFileFolder.split('/')[-1]#queue File
    elif os.path.isfolder(queueFileFolder):#CASE 2
        # Do a search of this folder for a queue*.json file
        files = glob.glob(os.path.join(queueFileFolder,'queue*'))#List of all full filepaths of type queue in queueFileFolder
        if len(files) == 1:#There is only 1 entry
            queueFFP = files[0]#queue Full File Path
            queueFile = files[0].split('/')[-1]#queue File
        else:
            #TODO add handling here
            print('There is more than one queue*.json file at this location')
            return False
    else: #queueFileFolder must be a valid path to a file or folder
        assert os.path.isfile(queueFileFolder) or os.path.isfolder(queueFileFolder),'queueFileFolder is neither a file nor folder'
        return False
    #####################################################################################################

    #### Load Queue File ################################################################################
    with open(queueFFP) as Q:
        queueData = json.load(Q)
    assert not queueData is None, 'No %s data was loaded' %queueFFP
    assert "singleRunPostProcessing" in queueData or "multiRunPostProcessing" in queueData, 'There are no specified scripts in %s' %queueFFP
    
    if "outpath" in queueData: #assign outpath if specified in queue file. Otherwise None
        outpath = queueData["outpath"]
    else:
        outpath = None

    if "PPoutpath" in queueData: #assign PPoutpath if specified in queue file. Otherwise None
        PPoutpath = queueData["PPoutpath"]
    else:
        PPoutpath = None
    #####################################################################################################

    #### Run SRPP and MRPP ##############################################################################
    SRPPsuccess = singleRunPostProcessing(queueData["singleRunPostProcessing"],PPoutpath,outpath)
    MRPPsuccess = multiRunPostProcessing(queueData["multiRunPostProcessing"],PPoutpath,outpath)

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a set of scripts and a queue. all files are relocated to a new folder.")
    parser.add_argument('--queueFileFolder',nargs=1,type=str, help='Path to Folder containing queue.json OR full filepath to queue.json (string).')
    parser.add_argument('--onSingleRun',nargs=1,type=str, help='Folder to run Single Run Post Porcessing On (string).')
    parser.add_argument('--onMultiRun',nargs=1,type=str, help='Folder to run Multi Run Post Porcessing On (string).')

    args = parser.parse_args()
    # TODO ADD CHECK ENSURING AT LEAST 1 ARGUMENT WAS INPUT




    if not args.onSingleRun is None:#Check that The argument was passed in
        onSingleRun = args.onSingleRun[0]
    if not args.onMultiRun is None:
        onMultiRun = args.onMultiRun[0]#Check that The argument was passed in


    if not args.queueFileFolder is None: #Check that The argument was passed in
        #### Behavior if queueFileFolder specified #################
        """The queue*.json file specified in any multirun (or single run) contains the dict keys:
        "singleRunPostProcessing":{"yieldPlotHistogram.py":True,...},"multiRunPostProcessing":{"yieldVsMissionTime.py":True,...}, "PPoutpath":"/pathToOutputDir/"
        IF queue*.json is specified the "singelRunPostProcessing" and "multiRunPostProcessing" will be run on all files in outpath.
        """
        queueFileFolder = args.queueFileFolder[0]
        status = queuePostProcessing(queueFileFolder)
        #return True
        ############################################################
