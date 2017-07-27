from ipyparallel import Client
from EXOSIMS.Prototypes.SurveyEnsemble import SurveyEnsemble 
from EXOSIMS.util.get_module import get_module
import time

class IPClusterEnsemble(SurveyEnsemble):
    """Parallelized suvey ensemble based on IPython parallel (ipcluster)
    
    """

    def __init__(self, **specs):
        
        SurveyEnsemble.__init__(self, **specs)
        
        # access the cluster
        self.rc = Client()
        self.dview = self.rc[:]
        self.dview.block = True
        with self.dview.sync_imports(): import EXOSIMS, EXOSIMS.util.get_module, \
                os, os.path, time, random, cPickle, traceback
        specs.pop('logger')
        self.dview.push(dict(specs=specs))
        self.dview.execute("SS = EXOSIMS.util.get_module.get_module(specs['modules'] \
                ['SurveySimulation'], 'SurveySimulation')(**specs)")
        self.lview = self.rc.load_balanced_view()

    def run_ensemble(self, sim, nb_run_sim, run_one=None, genNewPlanets=True,
            rewindPlanets=True, kwargs={}):
        
        t1 = time.time()
        async_res = []
        for j in range(nb_run_sim):
            ar = self.lview.apply_async(run_one, genNewPlanets=genNewPlanets,
                    rewindPlanets=rewindPlanets, **kwargs)
            async_res.append(ar)
        
        print "Submitted tasks: ", len(async_res)
        
        self.rc.wait(async_res)
        t2 = time.time()
        print "Completed in %d sec" % (t2 - t1)
        
        res = [ar.get() for ar in async_res]
        
        return res
