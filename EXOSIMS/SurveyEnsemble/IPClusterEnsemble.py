from __future__ import print_function

from ipyparallel import Client
from EXOSIMS.Prototypes.SurveyEnsemble import SurveyEnsemble 
from EXOSIMS.util.get_module import get_module
import time
from IPython.core.display import clear_output
import sys


class IPClusterEnsemble(SurveyEnsemble):
    """Parallelized suvey ensemble based on IPython parallel (ipcluster)
    
    """

    def __init__(self, **specs):
        
        SurveyEnsemble.__init__(self, **specs)

        self.verb = specs.get('verbose', True)
        
        # access the cluster
        self.rc = Client()
        self.dview = self.rc[:]
        self.dview.block = True
        with self.dview.sync_imports(): import EXOSIMS, EXOSIMS.util.get_module, \
                os, os.path, time, random, cPickle, traceback
        if specs.has_key('logger'):
            specs.pop('logger')
        if specs.has_key('seed'):
            specs.pop('seed')
        self.dview.push(dict(specs=specs))
        res = self.dview.execute("SS = EXOSIMS.util.get_module.get_module(specs['modules'] \
                ['SurveySimulation'], 'SurveySimulation')(**specs)")

        self.vprint("Created SurveySimulation objects on %d engines."%len(self.rc.ids))
        #for row in res.stdout:
        #    self.vprint(row)

        self.lview = self.rc.load_balanced_view()

    def run_ensemble(self, sim, nb_run_sim, run_one=None, genNewPlanets=True,
            rewindPlanets=True, kwargs={}):
        
        t1 = time.time()
        async_res = []
        for j in range(nb_run_sim):
            ar = self.lview.apply_async(run_one, genNewPlanets=genNewPlanets,
                    rewindPlanets=rewindPlanets, **kwargs)
            async_res.append(ar)
        
        print("Submitted %d tasks."%len(async_res))
        
        ar = self.rc._asyncresult_from_jobs(async_res)
        while not ar.ready():
            ar.wait(10.)
            clear_output(wait=True)
            if ar.progress > 0:
                timeleft = ar.elapsed/ar.progress * (nb_run_sim - ar.progress)
                if timeleft > 3600.:
                    timeleftstr = "%2.2f hours"%(timeleft/3600.)
                elif timeleft > 60.:
                    timeleftstr = "%2.2f minutes"%(timeleft/60.)
                else:
                    timeleftstr = "%2.2f seconds"%timeleft
            else:
                timeleftstr = "who knows"

            print("%4i/%i tasks finished after %4i s. About %s to go." % (ar.progress, nb_run_sim, ar.elapsed, timeleftstr), end="")
            sys.stdout.flush()

        #self.rc.wait(async_res)
        #self.rc.wait_interactive(async_res)
        t2 = time.time()
        print("\nCompleted in %d sec" % (t2 - t1))
        
        res = [ar.get() for ar in async_res]
        
        return res
