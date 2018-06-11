"""Purpose: Plot Time Wasted for a mission
Should be run from util folder

Written by Dean Keithly on 23 May, 2018
"""
"""Example 1
I have 1000 pkl files in /home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/run146279583107.pkl and
1qty outspec file in /home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/outspec.json

To generate timelines for these run the following code from an ipython session
from ipython
%run PlotTimeWasted.py '/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/run146279583107.pkl' \
'/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/outspec.json'
"""
"""Example 2
I have several folders with foldernames /home/dean/Documents/SIOSlab/*fZ*OB*PP*SU*/
each containing ~1000 pkl files and 1 outspec.json file

To plot a random Timeline from each folder, from ipython
%run PlotTimeWasted.py '/home/dean/Documents/SIOSlab/' None
"""
#%run PlotTimeWasted.py '/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/run136726516274.pkl' '/home/dean/Documents/SIOSlab/Dean17Apr18RS01C01fZ01OB01PP01SU01/outspec.json'
#%run PlotTimeWasted.py '/home/dean/Documents/SIOSlab/Dean21May18RS09CXXfZ01OB56PP03SU03/run5991056964408.pkl' '/home/dean/Documents/SIOSlab/Dean21May18RS09CXXfZ01OB56PP03SU03/outspec.json'
#%run PlotTimeWasted.py --runDirs '/home/dean/Documents/SIOSlab/' --QFile '/home/dean/Documents/exosims/EXOSIMS/EXOSIMS/util/PlotTimeWastedQueueDmitryRuns.json'

#TODO You cannot escape out of this analysis script with ctrl-c

try:
    import cPickle as pickle
except:
    import pickle
import os
import numpy as np
from pylab import *
from numpy import nan
import matplotlib.pyplot as plt
import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Mission Timeline Figures")
    parser.add_argument('--runDirs', nargs=1, type=str, help='directory containing all runs (string).')
    parser.add_argument('--QFile', nargs=1, type=str, help='full path to JSON file containing queue of runs to do statistics for (string).')
    #parser.add_argument('outspecfile', nargs=1, type=str, help='Full path to outspec file (string).')

    args = parser.parse_args()
    runDirs = args.runDirs[0]
    QFile = args.QFile[0]
    #outspecfile = args.outspecfile[0]

    if not os.path.exists(runDirs):
        raise ValueError('%s not found'%runDirs)

    with open(QFile) as queueFile:
        queueData = json.load(queueFile)
    RunsToPlot = queueData['folderNames']

    #Create all folderPaths
    allFolderDirs = [runDirs + n for n in RunsToPlot] 





    #Analysis for comparing this queue of Runs######################## (to be saved in data3)
    runMeanEngineeringTimeWasted = list()
    runMeanTimeDetecting = list()
    runMeanTimeCharacterizing = list()
    runMeanTimeObserving = list()
    runMeanTimeWastedFromOB = list()
    runMeanMissionPercentTimeWastedFromOB = list()
    runMeanNumCharacterizations = list()
    runMeanNumUniqDetections = list()
    runMeanTimeWastedOnTargetsWithNoDet = list()
    for i in np.arange(len(allFolderDirs)):#Iterate over each run directory #5/27/2018 Single Iteration takes 52 sec
        print "Working on Run %s, %d/%d" %(str(allFolderDirs[i]),i+1,len(allFolderDirs))
        #get all files and filter those containing .pkl
        allFiles = [fileName for fileName in next(os.walk(allFolderDirs[i]))[2] if '.pkl' in fileName]
        #DELETE print(allFiles)
        #Load outspec
        try:
            with open(allFolderDirs[i] + '/outspec.json', 'rb') as g:
                outspec = json.load(g)
        except:
            print('Failed to open outspecfile %s'%outspecPaths[cnt])
            pass

        #Extract all stars in run that were never observed, observed but never yielded detection
        AllsIndsNoDet = np.arange(10000).tolist()#arbitrarily large number
        maxInd = 0
        for j in np.arange(len(allFiles)):#iterate over all pkl files in this run directory
            #load pkl file
            try:
                with open(allFolderDirs[i] + '/' + allFiles[j], 'rb') as f:#load from cache
                    DRM = pickle.load(f)
            except:
                print('Failed to open pklfile %s'%allFiles[j])
                pass

            inds = [DRM['DRM'][ii]['star_ind'] for ii in np.arange(len(DRM['DRM']))]
            if max(inds) > maxInd:
                maxInd = max(inds)#replace maximum
            #Extract Obs Where detection occurs
            iWhereDet = [ii for ii in range(len(DRM['DRM'])) if (DRM['DRM'][ii]['det_status']==1).tolist().count(True) > 0] #list of DRM indicies where detection occurs
            # #Extract arrival_times
            # arrival_times = [DRM['DRM'][i]['arrival_time'] for i in iWhereDet]#each list item is the arrival_time for that detection observation
            #Extract inds
            sInds = [DRM['DRM'][ii]['star_ind'] for ii in iWhereDet]#each list item is the star index for that detection observation
            # #Extract pInds
            # pInds = [DRM['DRM'][i]['plan_inds'] for i in iWhereDet]#each list item contains the pInds for that star
            # #Extract det_status
            # det_status = [DRM['DRM'][i]['det_status'] for i in iWhereDet]#each list item contains the det_status for that star
            for tmpsInd in sInds:#look over all sInds
                if tmpsInd in AllsIndsNoDet:#If sInd was observed
                    AllsIndsNoDet.remove(tmpsInd)#Remove from AllsIndsObserved where a detection occured

        for tmpsInd in np.arange(maxInd+1, 10000):
            AllsIndsNoDet.remove(tmpsInd)#Remove all sInds used as padding
        #We now have AllsIndsNoDet, a listing of all planet indicies where no detections have occured in the set of simulations


        #Load and analyze all pkl files
        totalTimeDetecting = list()
        totalTimeCharacterizing = list()
        totalTimeObserving = list()
        totalMissionLife = list()
        totalObservingTime = list()
        totalMissionTimeWasted = list()
        totalMissionPercentTimeWasted = list()
        meanMissionTimeWastedFromOB = list()
        stdMissionTimeWastedFromOB = list()
        meanMissionPercentTimeWastedFromOB = list()
        numCharacterizations = list()
        numUniqDetections = list()
        totalMissionTimeWastedOnTargetsWithNoDet = list()

        firstRun=True
        #plt.close('all')
        for j in np.arange(len(allFiles)):#iterate over all pkl files in this run directory
            #load pkl file
            try:
                with open(allFolderDirs[i] + '/' + allFiles[j], 'rb') as f:#load from cache
                    DRM = pickle.load(f)
            except:
                print('Failed to open pklfile %s'%allFiles[j])
                pass

            #extract information from pkl files
            arrival_times = [DRM['DRM'][ii]['arrival_time'].value for ii in np.arange(len(DRM['DRM']))]
            sumOHTIME = outspec['settlingTime'] + outspec['starlightSuppressionSystems'][0]['ohTime']
            det_times = [DRM['DRM'][ii]['det_time'].value+sumOHTIME for ii in np.arange(len(DRM['DRM']))]
            det_timesROUNDED = [round(DRM['DRM'][ii]['det_time'].value+sumOHTIME,1) for ii in np.arange(len(DRM['DRM']))]
            ObsNums = [DRM['DRM'][ii]['ObsNum'] for ii in np.arange(len(DRM['DRM']))]
            y_vals = np.zeros(len(det_times)).tolist()
            char_times = [DRM['DRM'][ii]['char_time'].value*(1+outspec['charMargin'])+sumOHTIME*(DRM['DRM'][ii]['char_time'].value > 0.) for ii in np.arange(len(DRM['DRM']))]
            OBdurations = np.asarray(outspec['OBendTimes'])-np.asarray(outspec['OBstartTimes'])

            dets = np.hstack([row['plan_inds'][row['det_status'] == 1]  for row in DRM['DRM']])
            numUniqDetections.append(len(np.array([np.unique(r).size for r in dets])))

            det_times_onNoDet = [DRM['DRM'][ii]['det_time'].value+sumOHTIME for ii in np.arange(len(DRM['DRM'])) if DRM['DRM'][ii]['star_ind'] in AllsIndsNoDet]
            #sumOHTIME = [1 for i in np.arange(len(DRM['DRM']))]
            # print(sum(det_times))
            # print(sum(char_times))

            #Calculate Engineering Time Wasted (Statistics for the whole mission)
            totalTimeDetecting.append(sum(det_times))
            totalTimeCharacterizing.append(sum(char_times))
            totalTimeObserving.append(totalTimeDetecting[j] + totalTimeCharacterizing[j])#Time spent observing the target
            totalMissionLife.append(outspec['missionLife']*365.25)#in days
            totalObservingTime.append(outspec['missionPortion'] * totalMissionLife[j])#total time you can spend observing the target
            totalMissionTimeWasted.append(float(totalObservingTime[j]) - float(totalTimeObserving[j]))#amount of time spend doing nothing
            totalMissionPercentTimeWasted.append(totalMissionTimeWasted[j]/float(totalObservingTime[j]))
            numCharacterizations.append((np.asarray(char_times) > 0.0).tolist().count(True))
            totalMissionTimeWastedOnTargetsWithNoDet.append(sum(det_times_onNoDet))

            # #Total Observing Time Prints
            # print 'totalTimeDetecting: %f' %(totalTimeDetecting[j])
            # print 'totalTimeCharacterizing: %f' %(totalTimeCharacterizing[j])
            # print 'totalTimeObserving: %f' %(totalTimeObserving[j])
            # print 'totalMissionLife: %f' %(totalMissionLife[j])
            # print 'totalObservingTime: %f' %(totalObservingTime[j])
            # print 'totalMissionTimeWasted: %f' %(totalMissionTimeWasted[j])
            # print 'totalMissionPercentTimeWasted: %f' %(totalMissionPercentTimeWasted[j])


            #Calculate Engineering Time Wasted Per Observing Block
            OBassignment = np.zeros(len(arrival_times))+1000
            for k in np.arange(len(outspec['OBstartTimes'])):#Iterate Over OB
                OBassignment[np.where((np.asarray(arrival_times)>=outspec['OBstartTimes'][k])*(np.asarray(arrival_times)<outspec['OBendTimes'][k]))[0]] = k #Need to correlate Observations to Observing Blocks
            assert not np.any(OBassignment==1000), "An Observation Occured outside of an OB... or this is a bad check"


            OBtimeDetecting = list()
            OBtimeCharacterizing = list()
            totalTimeUsedInOB = list()
            totalTimeWastedInOB = list()
            percentOfOBWasted = list()
            totalOBTime = np.asarray(outspec['OBendTimes'])-np.asarray(outspec['OBstartTimes'])#This is the total time you can spend in an OB
            cumSumOBTimes = np.cumsum(totalOBTime)#calculate cumulativesum
            if len(cumSumOBTimes) > 1:
                index = np.where(np.asarray(cumSumOBTimes) > totalObservingTime[j])[0][0]#return index of OB where exoplanetObsTime would be exceeded
                remainderTimeInOB = (totalObservingTime[j] - cumSumOBTimes[index - 1])#find remainder time in final OB
            else:
                remainderTimeInOB = totalObservingTime[j]
            assert remainderTimeInOB >= 0, "remaining time in OB is less than 0... error?"
            totalOBTime[-1] = remainderTimeInOB #We correct for the fact that the addition time from the final OB could exceed the exoplanetObsTime
            for k in np.arange(len(outspec['OBstartTimes'])):#Iterate Over OB
                OBtimeDetecting.append(sum([det_times[cnt] for cnt in range(len(det_times)) if OBassignment[cnt] == k]))#total OB Time Detecting
                OBtimeCharacterizing.append(sum([char_times[cnt] for cnt in range(len(char_times)) if OBassignment[cnt] == k]))#total OB Time characterizing
                totalTimeUsedInOB.append(OBtimeDetecting[k] + OBtimeCharacterizing[k])#summation of time detecting and time characterizing in an OB
                totalTimeWastedInOB.append(float(totalOBTime[k]) - float(totalTimeUsedInOB[k]))#Total time wasted in OB
                percentOfOBWasted.append(totalTimeWastedInOB[k]/totalOBTime[k])
            # #Post Observing Block Prints
            # print 'OBtimeDetecting: %s' %str(OBtimeDetecting)
            # print 'OBtimeCharacterizing: %s' %str(OBtimeCharacterizing)
            # print 'totalTimeUsedInOB: %s' %str(totalTimeUsedInOB)
            # print 'totalTimeWastedInOB: %s' %str(totalTimeWastedInOB)
            # print 'totalOBTime: %s' %str(totalOBTime)
            # print 'cumSumOBTimes: %s' %str(cumSumOBTimes)
            #Post Observing Block Analysis Asserts
            assert abs(sum(totalOBTime) - (sum(totalTimeUsedInOB) + sum(totalTimeWastedInOB))) < 1e-3

            #Analysis of OB Engineering Time Wasted
            meanMissionTimeWastedFromOB.append(mean(totalTimeWastedInOB))
            stdMissionTimeWastedFromOB.append(std(totalTimeWastedInOB))
            meanMissionPercentTimeWastedFromOB.append(meanMissionTimeWastedFromOB[j]/sum(totalOBTime))#This should match up with the totalMissionPercentTimeWasted
            assert abs(sum(totalTimeWastedInOB) - totalMissionTimeWasted[j]) < 1e-3, "totalTimeWastedInOB != totalMissionTimeWasted"
            #assert totalMissionPercentTimeWasted[j] == meanMissionPercentTimeWastedFromOB[j], "mean times wasted do not match"



            #Plots of Single Run##################
            if firstRun == True:
                #det_times histogram
                fig_intTimes = figure(1000)
                hist(np.asarray(det_times)-sumOHTIME)
                title('Folder Dir: ' + os.path.basename(allFolderDirs[i]) + ' pkl File: ' + allFiles[j])
                xlim([0,1.1*max(np.asarray(det_times)-sumOHTIME)])
                xlabel('Detection Integration Times (d)')
                show(block=False)

                #%OB wasted vs OB Number
                fig_pOBWasted = figure(1001)
                scatter(range(len(percentOfOBWasted)), np.asarray(percentOfOBWasted)*100)
                ylim([0,100])
                title('Folder Dir: ' + os.path.basename(allFolderDirs[i]) + ' pkl File: ' + allFiles[j])
                xlabel('OB Number')
                ylabel('$\%$ of OB Wasted')
                show(block=False)

                #Time of OB wasted vs OB Number
                fig_TOBWasted = figure(1002)
                scatter(range(len(totalTimeWastedInOB)), np.asarray(totalTimeWastedInOB))
                #ylim([0,100])
                title('Folder Dir: ' + os.path.basename(allFolderDirs[i]) + ' pkl File: ' + allFiles[j])
                xlabel('OB Number')
                ylabel('OB Time Wasted')
                show(block=False)

                #Write Out Information From the First Run Analyzed of This Runtype
                data = {}
                data['TotalTimeWastedInOB'] = totalTimeWastedInOB
                data['PercentEngineeringTimeWastedInOB'] = percentOfOBWasted
                data['TotalOBTime'] = totalOBTime.copy().tolist()
                data['TotalMissionTimeWasted'] = totalMissionTimeWasted[j]
                with open(allFolderDirs[i] + '/' + os.path.basename(allFolderDirs[i]) + 'engineeringTimeWastedSingleRun.json', 'w') as outfile:
                    json.dump(data, outfile)

                firstRun = False
            #End Single Run Plots#####################

        #For this RunType##################
        data2 = {}
        data2['totalTimeDetecting'] = totalTimeDetecting
        data2['totalTimeCharacterizing'] = totalTimeCharacterizing
        data2['totalTimeObserving'] = totalTimeObserving
        data2['totalMissionLife'] = totalMissionLife
        data2['totalObservingTime'] = totalObservingTime
        data2['totalMissionTimeWasted'] = totalMissionTimeWasted
        data2['totalMissionPercentTimeWasted'] = totalMissionPercentTimeWasted
        data2['meanMissionTimeWastedFromOB'] = meanMissionTimeWastedFromOB
        data2['stdMissionTimeWastedFromOB'] = stdMissionTimeWastedFromOB
        data2['meanMissionPercentTimeWastedFromOB'] = meanMissionPercentTimeWastedFromOB
        data2['numCharacterizations'] = numCharacterizations
        data2['numUniqDetections'] = numUniqDetections
        data2['totalMissionTimeWastedOnTargetsWithNoDet'] = totalMissionTimeWastedOnTargetsWithNoDet
        with open(allFolderDirs[i] + '/' + os.path.basename(allFolderDirs[i]) + 'engineeringTimeWastedRunType.json', 'w') as outfile:
            json.dump(data2, outfile)

        #Plot Histogram of Number of Characterizations Distribution for RunType
        #%OB wasted vs OB Number
        fig_charHist = figure(10000)
        hist(numCharacterizations)
        #ylim([0,100])
        title(os.path.basename(allFolderDirs[i]))
        xlabel('Number of Characterizations')
        ylabel('Frequency')
        show(block=False)





        #Analysis for comparing this queue of Runs######################## (to be saved in Qdata)
        runMeanEngineeringTimeWasted.append(mean(totalMissionTimeWasted))
        runMeanTimeDetecting.append(mean(totalTimeDetecting))
        runMeanTimeCharacterizing.append(mean(totalTimeCharacterizing))
        runMeanTimeObserving.append(mean(totalTimeObserving))
        runMeanTimeWastedFromOB.append(mean(meanMissionTimeWastedFromOB))
        runMeanMissionPercentTimeWastedFromOB.append(mean(meanMissionPercentTimeWastedFromOB))
        runMeanNumCharacterizations.append(mean(numCharacterizations))
        runMeanNumUniqDetections.append(mean(numUniqDetections))
        runMeanTimeWastedOnTargetsWithNoDet.append(mean(totalMissionTimeWastedOnTargetsWithNoDet))

    #Writing Out Qdata#############################
    Qdata = {}
    Qdata['allFolderDirs'] = allFolderDirs
    Qdata['runmeanEngineeringTimeWasted'] = runMeanEngineeringTimeWasted
    Qdata['runMeanTimeDetecting'] = runMeanTimeDetecting
    Qdata['runMeanTimeCharacterizing'] = runMeanTimeCharacterizing
    Qdata['runMeanTimeObserving'] = runMeanTimeObserving
    Qdata['runMeanTimeWastedFromOB'] = runMeanTimeWastedFromOB
    Qdata['runMeanMissionPercentTimeWastedFromOB'] = runMeanMissionPercentTimeWastedFromOB
    Qdata['runMeanNumCharacterizations'] = runMeanNumCharacterizations
    Qdata['runMeanNumUniqDetections'] = runMeanNumUniqDetections
    Qdata['runMeanTimeWastedOnTargetsWithNoDet'] = runMeanTimeWastedOnTargetsWithNoDet
    with open(os.getcwd() + '/' + os.path.basename(QFile).split('.')[0] + 'TimeWastedRunQueue.json', 'w') as outfile:
        json.dump(Qdata, outfile)
    ################################################



