"""
End to End Test Suite for EXOSIMS

Run as:
    >python e2eTests.py

This code will sequentially execute all script files found in:
EXOSIMS_ROOT/EXOSIMS/Scripts/TestScripts
and print a summary of the results.  A script execution includes
instantiating a MissionSim object using the script, running a 
simulation via MissionSim.run_sim(), and resetting the simulation
using MissionSim.reset_sim(). Possible outcomes for each test
are:
    
    PASS
    FAIL - Instantiation
    FAIL - Execution
    FAIL - Reset


"""

import EXOSIMS
import os
import glob
import sys
import EXOSIMS.MissionSim
import numpy as np


def run_e2e_tests():
    basedir = EXOSIMS.__path__[0]
    testdir = os.path.join(basedir, 'Scripts', 'TestScripts')
    
    if not os.path.isdir(testdir):
        print("Cannot find test script directory in " \
                + "EXOSIMS_ROOT/EXOSIMS/Scripts/TestScripts")
        return
    
    scripts = glob.glob(os.path.join(testdir, "*.json"))
    
    if not scripts:
        print("No test scripts found in %s"%testdir)
        return
    
    print("%d test scripts found"%len(scripts))
    
    scripts = np.sort(scripts)
    
    results = []
    n = 0
    
    for j,script in enumerate(scripts):
        print("Running script: %s (%d/%d)"%(os.path.basename(script), j+1, len(scripts)))
        if len(os.path.basename(script)) > n:
            n = len(os.path.basename(script)) 
        
        try:
            sim = EXOSIMS.MissionSim.MissionSim(script)
        except:
            print("Instantiation failed.")
            print(sys.exc_info()[0])
            print("\n\n\n")
            results.append("FAIL - Instantiation")
            continue
        
        try:
            res = sim.run_sim()
        except:
            print("Run failed.")
            print(sys.exc_info()[0])
            print("\n\n\n")
            results.append("FAIL - Execution")
            continue

        try:
            sim.reset_sim()
        except:
            print("Reset failed.")
            print(sys.exc_info()[0])
            print("\n\n\n")
            results.append("FAIL - Reset")
            continue

        try:
            res2 = sim.run_sim()
        except:
            print("Second run failed.")
            print(sys.exc_info()[0])
            print("\n\n\n")
            results.append("FAIL - Execution after Reset")
            continue

        del sim
        
        results.append("PASS")
        print("\n\n\n")
    
    #results
    print("Summary")
    print("-"*80)
    for script,result in zip(scripts, results):
        tmp = '{0:' + str(n + 5) + '} ==> {1}'
        print(tmp.format(os.path.basename(script), result))
    
    print("-"*80)


if __name__ == '__main__':
    run_e2e_tests()
