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
            
        #current star has to be in the adjmat
        if (old_sInd is not None) and (old_sInd not in sInds):
            sInds = np.append(sInds,old_sInd)

        comps = self.TargetList.comp0[sInds]
        updated = (self.starVisits[sInds] > 0)
        comps[updated] =  self.Completeness.completeness_update(self.TargetList, \
                sInds[updated], self.TimeKeeping.currentTimeNorm)

        #for first target, just choose highest available completeness
        if old_sInd is None:
            sInd = np.random.choice(sInds[comps == max(comps)])
            return sInd
        
        #define adjacency matrix
        A = np.zeros((len(sInds),len(sInds)))

        # only consider slew distance when there's an occulter
        if self.OpticalSystem.haveOcculter:
            combs = np.array([np.array(x) for x in  itertools.combinations(range(len(sInds)),2)]) 
            r_ts = Obs.starprop(self.TargetList, sInds, self.TimeKeeping.currentTimeAbs)
            u_ts = r_ts/(np.tile(np.linalg.norm(r_ts,axis=1),(3,1)).T*r_ts.unit)

            angdists = np.arccos(np.sum(u_ts[combs[:,0],:]*u_ts[combs[:,1],:],1))
            A[np.tril(np.ones((len(sInds),len(sInds)),dtype=bool),-1)] = angdists
            A = self.coeffs[0]*(A+A.T)/np.pi
        
                
        #add factor due to completeness
        A = A + self.coeffs[1]*np.tile(np.array(1-comps,ndmin=2),(len(sInds),1))
       
        #add factor due to unvisited ramp
        f_uv = np.array(np.ones(len(sInds)),ndmin=2)
        f_uv[0,self.starVisits[sInds] != 0] = 0 
        f_uv = f_uv * ((self.TimeKeeping.currentTimeNorm/self.TimeKeeping.missionFinishNorm).value)**2.
        A = A - self.coeffs[2]*np.tile(f_uv,(len(sInds),1))

        #kill diagonal
        A = A + np.diag(np.ones(len(sInds))*np.Inf)

        #take two traversal steps
        lc = len(sInds)-1        
        step1 = np.tile(A[sInds==old_sInd,:],(lc,1)).flatten('F')
        step2 = A[np.array(np.ones((len(sInds),len(sInds))) - np.eye(len(sInds)),dtype=bool)]

        tmp = np.argmin(step1+step2)
        sInd = sInds[int(np.ceil(tmp/float(lc)))]

        return sInd

    

        



