from EXOSIMS.util.vprint import vprint
import time

class SurveyEnsemble(object):
    """Survey Ensemble prototype
    
    Args:
        \*\*specs:
            user specified values
            
    Attributes:
        
    """

    _modtype = 'SurveyEnsemble'

    def __init__(self, **specs):

        #start the outspec
        self._outspec = {}
        
        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))

    def run_ensemble(self, sim, nb_run_sim, run_one=None, genNewPlanets=True,
            rewindPlanets=True, kwargs={}):
        
        SS = sim.SurveySimulation
        t1 = time.time()
        res = []
        for j in range(nb_run_sim):
            print('\nSurvey simulation number %s/%s'%(j + 1, int(nb_run_sim)))
            ar = self.run_one(SS, genNewPlanets=genNewPlanets, 
                    rewindPlanets=rewindPlanets)
            res.append(ar)
        t2 = time.time()
        self.vprint("%s survey simulations, completed in %d sec"%(int(nb_run_sim), t2 - t1))
        
        return res

    def run_one(self, SS, genNewPlanets=True, rewindPlanets=True):
        SS.run_sim()
        res = SS.DRM[:]
        SS.reset_sim(genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets)
        return res
