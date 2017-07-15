import numpy as np
import EXOSIMS
import EXOSIMS.MissionSim
import os
import os.path
import cPickle
import time
import random
import argparse

def run_one():    
    # wrap the run_sim in a try/except loop
    nbmax = 10
    for attempt in range(nbmax):
        try:
            # run one survey simulation
            SS.run_sim()
            DRM = SS.DRM[:]
            systems = SS.SimulatedUniverse.dump_systems()
        except:
            # if anything goes wrong, reset simulation
            SS.reset_sim()
        else:
            break
    else:
        raise ValueError("Unsuccessful run_sim after %s reset_sim attempts"%nbmax)
    
    # reset simulation at the end of each simulation
    SS.reset_sim(genNewPlanets=True, rewindPlanets=True)

    basename = 'wfirst_nom1'
    savepath = os.path.join(os.path.expandvars('$HOME/Documents/AFTA-coronagraph/EXOSIMSres'),basename)
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    pklname = 'run'+str(int(time.clock()*100))+''.join(["%s" % random.randint(0, 9) for num in range(5)]) + '.pkl'
    pklpath = os.path.join(savepath, pklname)
    with open(pklpath, 'wb') as f:
        cPickle.dump({'DRM':DRM,'systems':systems}, f)
        
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an ipcluster parallel ensemble.")
    parser.add_argument('scriptfile', nargs=1, type=str, help='Full path to scriptfile.')
    parser.add_argument('numruns', nargs=1, type=int, help='Number of runs.')
    parser.add_argument('--outpath',nargs=1,type=str, help='Full path to output directory. Defaults to the basename of the scriptfile in the working directory, otherwise is created if it does not exist.') 

    args = parser.parse_args()

    scriptfile = args.scriptfile[0]

    if not os.path.exists(scriptfile):
        raise ValueError('%s not found'%scriptfile)

    if args.outpath is None:
        outpath = os.path.join(os.path.abspath('.'),os.path.splitext(os.path.basename(scriptfile))[0])
    else:
        outpath = args.outpath[0]

    if not os.path.exists(outpath):
        print "Creating output path %s"%outpath
        os.makedirs(outpath)



    #sim = EXOSIMS.MissionSim.MissionSim('parscript.json')
    #res = sim.run_ensemble(8,run_one=run_one)

    #print res
