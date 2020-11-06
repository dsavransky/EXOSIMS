from EXOSIMS.util.vprint import vprint
from EXOSIMS.util.get_dirs import get_cache_dir
import time

class SurveyEnsemble(object):
    """Survey Ensemble prototype
    
    Args:
        specs:
            user specified values
            
    Attributes:
        cachedir (str):
            Path to cache directory
        
    """

    _modtype = 'SurveyEnsemble'

    def __init__(self, cachedir=None, **specs):

        #start the outspec
        self._outspec = {}

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec['cachedir'] = self.cachedir
        specs['cachedir'] = self.cachedir 

        
        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get('verbose', True))

    def run_ensemble(self, sim, nb_run_sim, run_one=None, genNewPlanets=True,
        rewindPlanets=True, kwargs={}):
        """
        Args:
            sim (MissionSim Object):
                EXOSIMS.MissionSim.MissionSim object
            nb_run_sim (integer):
                number of simulations to run
            run_one (method):
                method to call for each simulation
            genNewPlanets (boolean):
                flag indicating whether to generate new planets each simulation (True means new planets will be generated)
            rewindPlanets (boolean):
                flag indicating whether planets will be rewound (True means planets will be rewound)
            kwargs ():
                -???
        Return:
            res (list dict):
                simulation list of dictionaries
        """
        
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
        """
        Args:
            SS (SurveySimulation Object):
                SurveySimulation object
            genNewPlanets (boolean):
                flag indicating whether to generate new planets each simulation (True means new planets will be generated)
            rewindPlanets (boolean):
                flag indicating whether planets will be rewound (True means planets will be rewound)
        Return:
            res (list dict):
                simulation list of dictionaries
        """
        SS.reset_sim(genNewPlanets=genNewPlanets, rewindPlanets=rewindPlanets)
        SS.run_sim()
        res = SS.DRM[:]
        return res
