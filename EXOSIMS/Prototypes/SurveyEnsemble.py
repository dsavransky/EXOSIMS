import time

class SurveyEnsemble(object):
    """Survey Ensemble prototype
    
        
    Args:
        \*\*specs:
            user specified values
            
    Attributes:
        
    """

    _modtype = 'SurveyEnsemble'
    _outspec = {}

    def __init__(self, **specs):
        #currently nothing to do here
        return

    def run_ensemble(self, sim, nb_run_sim, run_one=None, genNewPlanets=True, rewindPlanets=True):
        
        t1 = time.time()
        res = []
        for j in range(nb_run_sim):
            print '\nSurvey simulation number %s/%s' %(j+1, int(nb_run_sim))
            ar = self.run_one(sim)
            res.append(ar)
        t2 = time.time()
        print "%s survey simulations, completed in %d sec" %(int(nb_run_sim), t2-t1)
        
        return res

    def run_one(self, sim):
        
        sim.run_sim()
        res = sim.SurveySimulation.DRM[:]
        sim.reset_sim()
        
        return res
