from ipyparallel import Client
from EXOSIMS.Prototypes.SurveyEnsemble import SurveyEnsemble 
from EXOSIMS.util.get_module import get_module
import time

class IPClusterEnsemble(SurveyEnsemble):
    """
    Parallelized suvey ensemble based on IPython parallal (ipcluster)

    Args: 
        \*\*specs: 
            user specified values
            
    Attributes: 
        

    Notes:  

    """

    def __init__(self, **specs):

        SurveyEnsemble.__init__(self, **specs)

        # access the cluster
        self.rc = Client()
        self.dview = self.rc[:]
        self.dview.block = True
        with self.dview.sync_imports(): import EXOSIMS,EXOSIMS.util.get_module
        r1 = self.dview.execute("SurveySim = EXOSIMS.util.get_module.get_module('%s', 'SurveySimulation')"%specs['modules']['SurveySimulation'])
        self.dview.push(dict(specs=specs))
        r2 = self.dview.execute("sim = SurveySim(**specs)")
        self.lview = self.rc.load_balanced_view()

    def run_ensemble(self,run_one,N=10):
        t1 = time.time()
        async_res = []
        for j in range(N):
            ar = self.lview.apply_async(run_one)
            async_res.append(ar)

        print "Submitted tasks: ", len(async_res)
        
        self.rc.wait(async_res)
        t2 = time.time()
        print "Completed in %d sec" %(t2-t1)

        res = [ar.get() for ar in async_res]

        return res



        
        
