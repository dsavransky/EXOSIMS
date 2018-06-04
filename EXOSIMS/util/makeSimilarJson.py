"""
The purpose of this script is to take a template json script and make a series of "similar" json scripts from an original json script

From within an ipython session issue the command
%run makeSimilarJson.py --sourcefile '/Path/To/Source/Jsonscript.json' \
 --outpath '/Output/Path/To/New/JsonscriptX.json' \
 --runParam 'paramString' \
 --repKey ['jsonDictKey1','jsonDictKey2'] \
 --repVals [jsonDictVal,jsonDictVal2] \
 --runNums [paramNum,paramNum]

Specific Example:
%run makeSimilarJson.py --sourcefile '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/Scripts/Dean2May18RS09CXXfZ01OB01PP01SU01.json' \
 --outpath '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/Scripts/' \
 --runParam 'OB' \
 --repKey ['settlingTime','ohTime'] \
 --repVals [0.025,0.05] \
 --runNums [100,101]

Another example
 %run makeSimilarJson.py --sourcefile '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/Scripts/Dean2June18RS07CXXfZ01OB02PP01SU01.json' --outpath '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/Scripts/' --runParam 'OB' --repKey ['OBduration'] --repVals [15,10,5,20,25] --runNums [03,04,05,06,07]

Written by Dean Keithly on 5/15/2018
"""

import os
import numpy as np
import argparse
import json
import re
import ntpath
import ast


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="make a series of similar json scripts")
    parser.add_argument('--sourcefile', nargs=1, type=str, help='Full Path to initial json script (string)')
    parser.add_argument('--outpath', nargs=1, type=str, help='Full path to output directory of new file(s) (string).')
    parser.add_argument('--runParam', nargs=1, type=str, help='Run Parameter to vary (string).')#like "OB or PP, RS, SU, C, fZ"
    parser.add_argument('--repKey', nargs=1, type=list, help='json script Key(s) to replace (list).')
    parser.add_argument('--repVals', nargs=1, type=list, help='json script value(s) to replace (list).')
    parser.add_argument('--runNums', nargs=1, type=list, help='min runNum and max runNum (list).')


    args = parser.parse_args()
    sourcefile = args.sourcefile[0]
    outpath = args.outpath[0]
    runParam = args.runParam[0]
    repKeys = ''.join(args.repKey[0]).strip(']').strip('[').split(',')
    repVals = ast.literal_eval(''.join([n.strip() for n in args.repVals[0]]))
    runNums = ast.literal_eval(''.join([n.strip() for n in args.runNums[0]]))

    if not os.path.exists(sourcefile):#check whether the input file exists
        raise ValueError('%s not found'%sourcefile)

    if not os.path.exists(outpath):#Check whether the output directory exists
        raise ValueError('%s not found'%outpath)

    with open(sourcefile) as f:#Load json file
        jsonData = json.load(f)

    #doubleCheck values to replace are as long as maxRun-minRun
    assert len(runNums) == len(repVals)
    for runNum in range(0,len(runNums)):#iterate over the number of runs to write out
        for key1 in jsonData.keys():#iterate over all top level keys
            if key1 in repKeys:#If the top level key is in the list of keys to replace
                jsonData[key1] = repVals[runNum]#replace the key
            if type(jsonData[key1]) == list:#if the key is a list
                for listInd in range(len(jsonData[key1])):#iterate through list
                    if hasattr(jsonData[key1][listInd],'keys'):#check if the list has key attributes
                        for key2 in jsonData[key1][listInd].keys():#iterate through keys of each element in list
                            if key2 in repKeys:#if key2 is in repKeys
                                jsonData[key1][listInd][key2] = repVals[runNum]#replaces value with key
        #write out the json file
        fullInputFilename = ntpath.basename(sourcefile)#removes path from filename
        coreFilename, coreExtension = os.path.splitext(fullInputFilename)#removes extension from filename to give be the 
        splitCoreFilename = re.split('(\d+)',coreFilename)
        for i in range(len(splitCoreFilename)):
            #TODO make runParam specified by user. Should be "OB" or "PP" etc...
            if splitCoreFilename[i] == runParam:
                splitCoreFilename[i+1] = "%02d" %runNums[runNum]

        with open(outpath + ''.join(splitCoreFilename) + '.json', 'w') as g:
            json.dump(jsonData, g, indent=1)

