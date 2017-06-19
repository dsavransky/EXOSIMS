from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import numpy as np


class cbytScheduler(SurveySimulation):
    """C-by-t Scheduler 
    
    This class implements a Scheduler that selects the current highest 
    Completeness/Integration Time.
    
    """

    def __init__(self, **specs):
        
        SurveySimulation.__init__(self, **specs)

    def choose_next_target(self, old_sInd, sInds, slewTimes, t_dets):
        """Choose next target based on truncated depth first search 
        of linear cost function.
        
        Args:
            old_sInd (integer):
                Index of the previous target star
            sInds (integer array):
                Indices of available targets
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            t_dets (astropy Quantity array):
                Integration times for detection in units of day
        
        Returns:
            sInd (integer):
                Index of next target star
        
        """
        
        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping
        
        # reshape sInds
        sInds = np.array(sInds, ndmin=1)
        # get dynamic completeness values
        comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], TK.currentTimeNorm)
        
        # Selection metric being used: completeness/integration time
        selMetric = comps/t_dets
        
        # Selecting the target star to observe
        sInd = sInds[selMetric == max(selMetric)][0]
        
        return sInd

