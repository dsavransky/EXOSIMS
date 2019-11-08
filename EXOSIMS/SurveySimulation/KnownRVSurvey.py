from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
from EXOSIMS.util.deltaMag import deltaMag
import numpy as np
import astropy.units as u

class KnownRVSurvey(SurveySimulation):
    """KnownRVSurvey

    Survey Simulation module based on Know RV planets
    
    This class uses estimates of delta magnitude (dMagint) and instrument 
    working angle (WAint) for integration time calculation, specific to 
    the known RV planets.

    Args:
        \*\*specs:
            user specified values
    
    """

    def __init__(self, **specs):
        
        # call prototype constructor
        SurveySimulation.__init__(self, **specs)
        
        TL = self.TargetList
        SU = self.SimulatedUniverse
        
        # reinitialize working angles and delta magnitudes used for integration
        self.WAint = np.zeros(TL.nStars)*u.arcsec
        self.dMagint = np.zeros(TL.nStars)
        
        # calculate estimates of shortest WAint and largest dMagint for each target
        for sInd in range(TL.nStars):
            pInds = np.where(SU.plan2star == sInd)[0]
            self.WAint[sInd] = np.arctan(np.min(SU.a[pInds])/TL.dist[sInd]).to('arcsec')
            phis = np.array([np.pi/2]*pInds.size)
            dMags = deltaMag(SU.p[pInds], SU.Rp[pInds], SU.a[pInds], phis)
            self.dMagint[sInd] = np.min(dMags)
        
        # populate outspec with arrays
        self._outspec['WAint'] = self.WAint.value
        self._outspec['dMagint'] = self.dMagint
