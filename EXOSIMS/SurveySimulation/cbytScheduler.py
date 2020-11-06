from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import numpy as np

class cbytScheduler(SurveySimulation):
    """cbytScheduler - Completeness-by-time Scheduler 
    
    This class implements a Scheduler that selects the current highest 
    Completeness/Integration Time.

    Args:
        \*\*specs:
            user specified values
    
    """
    
    def __init__(self, **specs):
        
        SurveySimulation.__init__(self, **specs)

    def choose_next_target(self, old_sInd, sInds, slewTimes, intTimes):
        """Choose next target based on truncated depth first search 
        of linear cost function.
        
        Args:
            old_sInd (integer):
                Index of the previous target star
            sInds (integer array):
                Indices of available targets
            slewTimes (astropy quantity array):
                slew times to all stars (must be indexed by sInds)
            intTimes (astropy Quantity array):
                Integration times for detection in units of day
        
        Returns:
            sInd (integer):
                Index of next target star
            waitTime (astropy Quantity):
                the amount of time to wait (this method returns None)
        
        """
        
        Comp = self.Completeness
        TL = self.TargetList
        TK = self.TimeKeeping
        
        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        # calculate dt since previous observation
        dt = TK.currentTimeNorm + slewTimes[sInds] - self.lastObsTimes[sInds]
        # get dynamic completeness values
        comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], dt)
        
        # selection metric being used: completeness/integration time
        selMetric = comps/intTimes
        
        # selecting the target star to observe
        sInd = sInds[selMetric == max(selMetric)][0]
        
        return sInd, None
