from EXOSIMS.SurveySimulation.linearJScheduler import linearJScheduler
import astropy.units as u
import numpy as np

class occulterJScheduler(linearJScheduler):
    """occulterJScheduler 
    
    This class inherits linearJScheduler and works best when paired with the 
    SotoStarshade Observatory class. 
    
    Args:
        nSteps (integer 1x1):
            Number of steps to take when calculating the cost function.
        useAngles (bool):
            Use interpolated dV angles.
        \*\*specs:
            user specified values
    
    """
    
    def __init__(self, nSteps=1, useAngles=False, **specs):
        
        linearJScheduler.__init__(self, **specs)
        
        if nSteps < 1:
            raise TypeError("nSteps must be 1 or greater")
        
        nSteps = int(nSteps)
        self.nSteps = nSteps
        self.useAngles = useAngles

    
    def choose_next_target(self, old_sInd, sInds, slewTimes, intTimes):
        """Helper method for method next_target to simplify alternative implementations.
        
        Given a subset of targets (pre-filtered by method next_target or some 
        other means), select the best next one. The prototype uses completeness 
        as the sole heuristic.
        
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
                some strategic amount of time to wait in case an occulter slew is desired (default is None)
        
        """
        
        OS = self.OpticalSystem
        Comp = self.Completeness
        TL = self.TargetList
        Obs = self.Observatory
        TK = self.TimeKeeping
        

        # cast sInds to array
        sInds = np.array(sInds, ndmin=1, copy=False)
        
        # calculate dt since previous observation
        dt = TK.currentTimeAbs.copy() + slewTimes[sInds]
        # get dynamic completeness values
        comps = Comp.completeness_update(TL, sInds, self.starVisits[sInds], dt)
        
        # if first target, or if only 1 available target, 
        # choose highest available completeness
        nStars = len(sInds)
        if (old_sInd is None) or (nStars == 1):
            sInd = np.random.choice(sInds[comps == max(comps)])
            return sInd, slewTimes[sInd]
        
        else:
            # define adjacency matrix
            A = np.zeros(nStars)

            # only consider slew distance when there's an occulter
            if OS.haveOcculter:
                angdists = [Obs.star_angularSep(TL, old_sInd, s, t) for s,t in zip(sInds,dt)]
                
                try:
                    Obs.__getattribute__('dV_interp')
                except:
                    self.useAngles = True   
                
                if self.useAngles:
                    A[np.ones((nStars), dtype=bool)] = angdists
                    A = self.coeffs[0]*(A)/np.pi
                else:
                    dVs = np.array([Obs.dV_interp(slewTimes[sInds[s]],angdists[s].to('deg'))[0] for s in range(len(sInds))])
                    A[np.ones((nStars), dtype=bool)] = dVs
                    A = self.coeffs[0]*(A)/(0.025*Obs.dVtot.value)
            
            # add factor due to completeness
            A = A + self.coeffs[1]*(1 - comps)
            
            # add factor due to unvisited ramp
            f_uv = np.zeros(nStars)
            unvisited = self.starVisits[sInds]==0
            f_uv[unvisited] = float(TK.currentTimeNorm.copy()/TK.missionLife)**2
            A = A - self.coeffs[2]*f_uv
    
            # add factor due to revisited ramp
            f2_uv = np.where(self.starVisits[sInds] > 0, 1, 0) *\
                    (1 - (np.in1d(sInds, self.starRevisit[:,0],invert=True)))
            A = A + self.coeffs[3]*f2_uv
            
            if self.nSteps > 1:
                A_ = np.zeros((nStars,nStars))
                # only consider slew distance when there's an occulter
                if OS.haveOcculter:
                    angdists_ = np.array([Obs.star_angularSep(TL, s, sInds, t) for s,t in zip(sInds,dt)])
                    dVs_= np.array([Obs.dV_interp(slewTimes[sInds[s]],angdists_[s,:]) for s in range(len(sInds))])
                    A_ = self.coeffs[0]*dVs_.reshape(nStars,nStars)/(0.025*Obs.dVtot.value)
                # add factor due to completeness
                A_ = A_ + self.coeffs[1]*(1 - comps)
                
                # add factor due to unvisited ramp
                f_uv = np.zeros(nStars)
                unvisited = self.starVisits[sInds]==0
                f_uv[unvisited] = float(TK.currentTimeNorm.copy()/TK.missionLife)**2
                A_ = A_ - self.coeffs[2]*f_uv
        
                # add factor due to revisited ramp
                f2_uv = np.where(self.starVisits[sInds] > 0, 1, 0) *\
                        (1 - (np.in1d(sInds, self.starRevisit[:,0],invert=True)))
                A_ = A_ + self.coeffs[3]*f2_uv
                
                step1 = np.tile(A, (nStars, 1)).flatten('F')
                stepN = A_.flatten()
                tmp = np.argmin( step1 + stepN*(self.nSteps-1) )
                sInd = sInds[int(np.floor(tmp/float(nStars)))]
                
            else:
                # take just one step
                tmp = np.argmin(A)
                sInd = sInds[int(tmp)]
                

            return sInd, slewTimes[sInd] #if coronagraph or first sInd, waitTime will be 0 days