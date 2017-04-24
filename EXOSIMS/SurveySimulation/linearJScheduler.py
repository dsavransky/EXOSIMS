from EXOSIMS.Prototypes.SurveySimulation import SurveySimulation
import astropy.units as u
import numpy as np
import itertools

class linearJScheduler(SurveySimulation):
    """linearJScheduler 
    
    This class implements the linear cost function scheduler described
    in Savransky et al. (2010).
    
        Args:
        as (iterable 3x1):
            Cost function coefficients: slew distance, completeness, target list coverage
        
        \*\*specs:
            user specified values
    
    """

    def __init__(self, coeffs=[1,1,2,1], **specs):
        
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
        
        # reshape sInds
        sInds = np.array(sInds,ndmin=1)
        
        # current star has to be in the adjmat
        if (old_sInd is not None) and (old_sInd not in sInds):
            sInds = np.append(sInds, old_sInd)
        
        # get completeness values
        comps = TL.comp0[sInds]
        updated = (self.starVisits[sInds] > 0)
        comps[updated] = Comp.completeness_update(TL, sInds[updated], TK.currentTimeNorm)
        
        # if first target, or if only 1 available target, choose highest available completeness
        nStars = len(sInds)
        if (old_sInd is None) or (nStars == 1):
            sInd = np.random.choice(sInds[comps == max(comps)])
            return sInd
        
        # define adjacency matrix
        A = np.zeros((nStars,nStars))

        # only consider slew distance when there's an occulter
        if OS.haveOcculter:
            r_ts = Obs.starprop(TL, sInds, TK.currentTimeAbs)
            u_ts = (r_ts.value.T/np.linalg.norm(r_ts,axis=1)).T
            angdists = np.arccos(np.clip(np.dot(u_ts,u_ts.T),-1,1))
            A[np.ones((nStars),dtype=bool)] = angdists
            A = self.coeffs[0]*(A)/np.pi

        # add factor due to completeness
        A = A + self.coeffs[1]*(1-comps)
        
        # add factor due to unvisited ramp
        f_uv = np.zeros(nStars)
        f_uv[self.starVisits[sInds] == 0] = ((TK.currentTimeNorm / TK.missionFinishNorm)\
                .decompose().value)**2
        A = A - self.coeffs[2]*f_uv

        # add factor due to revisited ramp
        f2_uv = np.where(SSim.starVisits[sInds] > 0, SSim.starVisits[sInds], 0) *\
                (1 - (np.in1d(sInds, self.starRevisit[:,0],invert=True)))
        A = A - self.coeffs[3]*f2_uv
        
        # kill diagonal
        A = A + np.diag(np.ones(nStars)*np.Inf)
        
        # take two traversal steps
        step1 = np.tile(A[sInds==old_sInd,:],(nStars,1)).flatten('F')
        step2 = A[np.array(np.ones((nStars,nStars)),dtype=bool)]
        tmp = np.argmin(step1+step2)
        sInd = sInds[int(np.floor(tmp/float(nStars)))]
        
        return sInd

