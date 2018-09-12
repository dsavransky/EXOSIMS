"""
The purpose of this script is to take a template json script and make a series of "similar" json scripts from an original json script

This script is designed for:
	Sweep Single Parameter
	Sweep Multiple Parameters over Multiple Values
    Sweep Multiple Combinations of Parameters over 

#makeSimilarScripts.py is designed to run from the 'EXOSIMS/Scripts/' Folder

Another example
 %run makeSimilarScripts.py


 
Written by Dean Keithly on 6/28/2018
"""

import os
import numpy as np
import argparse
import json
import re
import ntpath
import ast
import string
import copy
import datetime
from itertools import combinations
import shutil

def createScriptFolder(makeSimilarInst,sourcefile):
    """This method creates a 'Script Folder' - a new folder with name 'makeSimilarInst_sourcefile' in 'EXOSIMS/Scripts/'
    returns folderName
    """
    myString = os.getcwd() + '/' + makeSimilarInst + '_' + sourcefile
    try:
        os.mkdir(myString)#will fail if directory exists
        print('MADE DIR: ' + myString)
    except:
        print('DID NOT MAKE DIR: ' + myString + ' It already exists.')
    return myString.split('/')[-1]

def createScriptName(prepend,makeSimilarInst,sourcefile,ind):
    """ This Script creates the ScriptName
    """
    date = unicode(datetime.datetime.now())
    date = ''.join(c + '_' for c in re.split('-|:| ',date)[0:-1])#Removes seconds from date
    scriptName = prepend+ '_' + date + '_' + makeSimilarInst + '_' + sourcefile.split('.')[0] + '_' + str(ind) + '.json'
    return scriptName



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a set of scripts and a queue. all files are relocated to a new folder.")
    parser.add_argument('--makeSimilarInst',nargs=1,type=str, help='Full path to the makeSimilar.json instruction script (string).')

    args = parser.parse_args()

    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    valid_sweepTypes = ['SweepParameters','SweepParametersPercentages']
    
    #(default) If no makeSimilarScripts instruction file is provided, default use makeSimilar.json
    if args.makeSimilarInst is None:
        makeSimilarInst = './makeSimilar.json'
        assert os.path.exists(makeSimilarInst), "%s is not a valid filepath" % (makeSimilarInst)
        with open(makeSimilarInst) as f:#Load variational instruction script
            jsonDataInstruction = json.load(f)#This script contains the instructions for precisely how to modify the base file
    else:#else: use the provided instructions
        makeSimilarInst = args.makeSimilarInst[0]
        assert os.path.exists(makeSimilarInst), "%s is not a valid filepath" % (makeSimilarInst)
        with open(makeSimilarInst) as f:#Load variational instruction script
            jsonDataInstruction = json.load(f)#This script contains the instructions for precisely how to modify the base file
    sourceFolderCore = (makeSimilarInst.split('/')[-1]).split('.')[0]

    #Load source File
    sourcefile = jsonDataInstruction['scriptName']#the filename of the script to be copied
    sourceFileCore = sourcefile.split('.')[0]#strips the .json part of the filename
    with open('./' + sourcefile) as f:#Load source script json file
        jsonDataSource = json.load(f)#This script contains the information to be slightly modified

    #Error Checking
    assert jsonDataInstruction['sweepType'] in valid_sweepTypes, 'sweepType %s not in valid_sweepTypes' %(jsonDataInstruction['sweepType'])

    #Create Script Folder
    folderName = createScriptFolder(sourceFolderCore,sourceFileCore)#Create 'Script Folder' - a new folder with name 'makeSimilarInst_sourcefile' in 'EXOSIMS/Scripts/'

    namesOfScriptsCreated = list()
    # Case 1
    """
    Here we want to sweep parameters A and B such that  A = [1,2,3] and B = [4,5,6]
    In this case, the first script will have A=1, B=4. The second script will have A=2, B=5.
    It is required that len(A) == len(B).
    You can sweep an arbitrarily large number of parameters A,B,C,D,...,Z,AA,... so long as the number of values you have for each is constant
    """
    if jsonDataInstruction['sweepType'] == "SweepParameters":
        sweepParameters = jsonDataInstruction['sweepParameters']# Grab Parameter to Sweep
        sweepValues = jsonDataInstruction['sweepValues']# retrieve manually defined sweep values

        #Error Checking
        for ind in range(len(sweepValues)-1):#Check Each Parameter has the same number of values to sweep
            assert(len(sweepValues[ind]) == len(sweepValues[ind+1]))

        #Create each Script
        for ind in range(len(sweepValues[0])):#Number of values to sweep over
            #Create Filename Substring using parameters and values
            paramNameSet = ''
            for ind2 in range(len(sweepParameters)):#Iterate over all parameters
                paramNameSet = paramNameSet + sweepParameters[ind2] + str(sweepValues[ind2][ind])
            
            scriptName = createScriptName('auto',sourceFolderCore,sourceFileCore,ind)#create script name
            namesOfScriptsCreated.append(scriptName)#Append to master list of all scripts created
            jsonDataOutput = copy.deepcopy(jsonDataSource)#Create a deepCopy of the original json script

            for ind3 in range(len(sweepParameters)):#Iterate over all parameters
                jsonDataOutput[sweepParameters[ind3]] = sweepValues[ind3][ind] #replace value

            #Write out json file
            with open('./' + folderName + '/' + scriptName, 'w') as g:
                json.dump(jsonDataOutput, g, indent=1)


        # Create queue.json script from namesOfScriptsCreated
        queueOut = {}
        queueName = createScriptName('queue',sourceFolderCore,sourceFileCore,'')
        with open('./' + folderName + '/' + queueName, 'w') as g:
            queueOut['scriptNames'] = namesOfScriptsCreated
            queueOut['numRuns'] = [jsonDataInstruction['numRuns'] for i in range(len(namesOfScriptsCreated))]
            json.dump(queueOut, g, indent=1)
        if os.path.isdir('../run'):#If the relative path exists for run, create it there
            with open('../run/' + 'queue.json', 'w') as g:
                queueOut['scriptNames'] = namesOfScriptsCreated
                queueOut['numRuns'] = [jsonDataInstruction['numRuns'] for i in range(len(namesOfScriptsCreated))]
                json.dump(queueOut, g, indent=1)
        else:#Otherwise create queue.json in current directory
            with open('../cache/' + 'queue.json', 'w') as g:
                queueOut['scriptNames'] = namesOfScriptsCreated
                queueOut['numRuns'] = [jsonDataInstruction['numRuns'] for i in range(len(namesOfScriptsCreated))]
                json.dump(queueOut, g, indent=1)



        # Case 2
        """
        Here we want to take a set of parameters A,B,C,...,Z
        and set them at +/- a,b,c,...,z% from theic current value
        """
    elif (jsonDataInstruction['sweepType'] == "SweepParametersPercentages"):
        sweepPercentages = jsonDataInstruction['sweepPercentages']# retrieve manually defined sweep percentage
        sweepParameters = jsonDataInstruction['sweepParameters']# Grab Parameter to Sweep
        if 'sweepCombNums' in jsonDataInstruction:
            sweepCombNums = jsonDataInstruction['sweepCombNums']# Combinations of parameters to iterate over
        else:
            sweepCombNums = [1]

        #Error Checking
        assert max(sweepCombNums) <= len(sweepParameters), "sweepCombNums: %d > len(sweepParameters): %d" %(max(sweepCombNums),len(sweepParameters)) #check the sweepCombNum is valid

        #Combination Number Loop
        cnt = 0
        for cInd in range(len(sweepCombNums)):

            paramInds = range(len(sweepParameters)) # inds for parameters to sweep
            allIndCombs = list(combinations(paramInds,sweepCombNums[cInd])) # all combinations of paramInds for sweepCombNums

            #Iterate over Combinations and Create Script Loop
            for ind in range(len(allIndCombs)):#Iterate over allIndCombs
                comb = allIndCombs[ind]#The combination under consideration

                #Iterate over sweepPercentages
                for pInd in range(len(sweepPercentages)):

                    scriptName = createScriptName('auto',sourceFolderCore,sourceFileCore,cnt)#create script name
                    namesOfScriptsCreated.append(scriptName)#Append to master list of all scripts created
                    jsonDataOutput = copy.deepcopy(jsonDataSource)#Create a deepCopy of the original json script

                    #Iterate over all Parameters and update based on percentage
                    for jnd in range(len(comb)):
                        paramInd = comb[jnd]#The specific parameter index being modified in this script
                        jsonDataOutput[sweepParameters[paramInd]] = jsonDataOutput[sweepParameters[paramInd]]*(1. + sweepPercentages[pInd]) #replace value
                    
                    #Write Out Script
                    with open('./' + folderName + '/' + scriptName, 'w') as g:
                        json.dump(jsonDataOutput, g, indent=1)
                        cnt += 1

        # Create queue.json script from namesOfScriptsCreated
        queueOut = {}
        queueName = createScriptName('queue',sourceFolderCore,sourceFileCore,'')
        with open('./' + folderName + '/' + queueName, 'w') as g:
            queueOut['scriptNames'] = namesOfScriptsCreated
            queueOut['numRuns'] = [jsonDataInstruction['numRuns'] for i in range(len(namesOfScriptsCreated))]
            json.dump(queueOut, g, indent=1)
        if os.path.isdir('../run'):#If the relative path exists for run, create it there
            with open('../run/'  'queue.json', 'w') as g:
                queueOut['scriptNames'] = namesOfScriptsCreated
                queueOut['numRuns'] = [jsonDataInstruction['numRuns'] for i in range(len(namesOfScriptsCreated))]
                json.dump(queueOut, g, indent=1)
        else:#Otherwise create queue.json in current directory
            with open('./' + 'queue.json', 'w') as g:
                queueOut['scriptNames'] = namesOfScriptsCreated
                queueOut['numRuns'] = [jsonDataInstruction['numRuns'] for i in range(len(namesOfScriptsCreated))]
                json.dump(queueOut, g, indent=1)

        #Create Filename Substring using parameters and values
        # paramNameSet = ''
        # for ind2 in range(len(sweepParameters)):#Iterate over all parameters
        #     paramNameSet = paramNameSet + sweepParameters[ind2] + str(1. + sweepPercentage[ind2])
        #Now strip all invalid filename parts
        # scriptName = os.path.splitext(sourcefile)[0] + ''.join(c for c in paramNameSet if c in valid_chars and not(c == '.')) + '.json'
        # namesOfScriptsCreated.append(scriptName)#Append to master list of all scripts created
        # jsonDataOutput = copy.deepcopy(jsonDataSource)#Create a deepCopy of the original json script

        # for ind3 in range(len(sweepParameters)):#Iterate over all parameters
        #     jsonDataOutput[sweepParameters[ind3]] = 1. + sweepPercentage[ind3] #replace value

        # #Write out json file
        # with open('./' + folderName + '/' + scriptName, 'w') as g:
        #     json.dump(jsonDataOutput, g, indent=1)
    else:
        print('not a valid instruction script')

    #Copy MakeSimilarInst to directory containing scripts
    shutil.copy2(makeSimilarInst,'./' + folderName)
    #Copy ScriptAAA.json to directory containing scripts
    shutil.copy2(sourcefile,'./' + folderName)