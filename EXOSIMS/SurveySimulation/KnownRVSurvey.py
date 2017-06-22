from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
from EXOSIMS.util.deltaMag import deltaMag
import numpy as np
import astropy.units as u

class KnownRVSurvey(SurveySimulation):
    """Survey Simulation module based on Know RV planets
    
    This class uses estimates of flux ratios (dMag) and working angle 
    (separations) for the known RV planets
    
    """

    def __init__(self, **specs):
        
        # call prototype constructor
        SurveySimulation.__init__(self, **specs)
        
        TL = self.TargetList
        SU = self.SimulatedUniverse
        
        # reinitialize separations and flux ratios used for integration
        self.WAint = np.zeros(TL.nStars)*u.arcsec
        self.dMagint = np.zeros(TL.nStars)
        
        # calculate estimates of shortest WAint and largest dMagint for each target
        for sInd in range(TL.nStars):
            pInds = np.where(SU.plan2star == sInd)[0]
            self.WAint[sInd] = np.arctan(np.min(SU.a[pInds])/TL.dist[sInd]).to('mas')
            phis = np.array([np.pi/2]*pInds.size)
            dMags = deltaMag(SU.p[pInds], SU.Rp[pInds], SU.a[pInds], phis)
            self.dMagint[sInd] = np.min(dMags)
