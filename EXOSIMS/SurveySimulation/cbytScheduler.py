from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import itertools
from IPython.core.debugger import Tracer


class cbytScheduler(SurveySimulation):
    """startAYO2014 
    
    This class implements the AYO discussed in stark2014.
    
       CHANGE Args:
        as (iterable 3x1):
            Cost function coefficients: slew distance, completeness, target list coverage
        
        \*\*specs:
            user specified values
    
    """

    def __init__(self, coeffs=[1,1,2], **specs):
        
        SurveySimulation.__init__(self, **specs)
        
        #verify that coefficients input is iterable 6x1
        if not(isinstance(coeffs,(list,tuple,np.ndarray))) or (len(coeffs) != 3):
            raise TypeError("coeffs must be a 3 element iterable")
        
        #normalize coefficients
        coeffs = np.array(coeffs)
        coeffs = coeffs/np.linalg.norm(coeffs)
        
        self.coeffs = coeffs

    def choose_next_target(self,old_sInd,sInds,slewTime):
        """Choose next target based on truncated depth first search 
        of linear cost function.
        
        Args:
            old_sInd (integer):
                Index of the previous target star
            sInds (integer array):
                Indices of available targets
            slewTime (float array):
                slew times to all stars (must be indexed by sInds)
                
        Returns:
            sInd (integer):
                Index of next target star
        
        """
        
        OS = self.OpticalSystem
        Comp = self.Completeness
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        
<<<<<<< HEAD

=======
>>>>>>> 63ec20907afba61ad4e193f80e833fa4a900ec68
        comps = TL.comp0[sInds]#completeness of each star in TL
        updated = (self.starVisits[sInds] > 0)#what does starVisits contain?
        comps[updated] = Comp.completeness_update(TL, sInds[updated],TK.currentTimeNorm)
        tint = TL.tint0[sInds]
        
        selMetric=comps/tint#selMetric is the selection metric being used. Here it is Completeness/integration time

        #Here I select the target star to observe
        tmp = sInds[selMetric == max(selMetric)]#this selects maximum completeness/integration time
        sInd = tmp[0]#casts numpy array to single integer
        #sInd = np.random.choice(sInds[comps == max(comps)])#this selects the maximum completeness

        ## reshape sInds
        #sInds = np.array(sInds,ndmin=1)
        # 
        # # current star has to be in the adjmat
        # if (old_sInd is not None) and (old_sInd not in sInds):
        #     sInds = np.append(sInds, old_sInd)
        # 
        # # get completeness values
        # comps = TL.comp0[sInds]
        # updated = (self.starVisits[sInds] > 0)
        # comps[updated] =  Comp.completeness_update(TL, sInds[updated], TK.currentTimeNorm)
        # 
        # # if first target, or if only 1 available target, choose highest available completeness
        # nStars = len(sInds)
        # if (old_sInd is None) or (nStars == 1):
        #     sInd = np.random.choice(sInds[comps == max(comps)])
        #     return sInd
        #Tracer()()
        
        return sInd

