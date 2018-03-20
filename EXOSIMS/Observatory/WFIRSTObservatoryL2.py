from EXOSIMS.Observatory.ObservatoryL2Halo import ObservatoryL2Halo


class WFIRSTObservatoryL2(ObservatoryL2Halo):
    """ WFIRST Observatory at L2 implementation. 
    Contains methods and parameters unique to the WFIRST mission.
    """

    def __init__(self, **specs):
        
        # run prototype constructor __init__ 
        ObservatoryL2Halo.__init__(self,**specs)
        
        
