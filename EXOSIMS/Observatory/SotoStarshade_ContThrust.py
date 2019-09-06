from EXOSIMS.Observatory.SotoStarshade import SotoStarshade
import numpy as np
import astropy.units as u
from scipy.integrate import solve_ivp
import astropy.constants as const
import hashlib
import scipy.optimize as optimize
from scipy.optimize import basinhopping
import scipy.interpolate as interp
import scipy.integrate as intg
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
from copy import deepcopy
import time
import os
try:
    import _pickle as pickle
except:
    import pickle

EPS = np.finfo(float).eps


class SotoStarshade_ContThrust(SotoStarshade):
    """ StarShade Observatory class
    This class is implemented at L2 and contains all variables, functions, 
    and integrators to calculate occulter dynamics. 
    """
    
    def __init__(self,orbit_datapath=None,**specs): 

        SotoStarshade.__init__(self,**specs)  
        
        #to convert from dimensionalized to normalized, (Dimension) / self.(Dimension)U
        self.mass = 6000*u.kg
        
        Tmax = 1040*u.mN
        Isp  = 3000*u.s
        ve   = const.g0 * Isp
        
        self.Tmax = Tmax
        self.ve   = self.convertVel_to_canonical(ve)
        
        # smoothing factor (eps = 1 means min energy, eps = 0 means min fuel)
        self.epsilon = 1 
        
        self.lagrangeMults = self.lagrangeMult()
        
# =============================================================================
# Miscellaneous        
# =============================================================================

    def unitVector(self,p):
        """ returns unit vector of p with same dimensions (3xn)
        """
        
        pnorm = np.linalg.norm(p,axis=0)
        p_ = p/pnorm
        
        return p_,pnorm
    

    def DCM_r2i(self,t):
        """ Direction cosine matrix to rotate from Rotating Frame to Inertial Frame
        """
        
        Ts  = np.array([[ np.cos(t) ,-np.sin(t) , 0],\
                        [ np.sin(t) , np.cos(t) , 0],\
                        [   0       ,    0      , 1]])
        
        dTs = np.array([[-np.sin(t) ,-np.cos(t) , 0],\
                        [ np.cos(t) ,-np.sin(t) , 0],\
                        [   0       ,    0      , 0]])
    
        Qu = np.hstack( [ Ts , np.zeros([3,3])])
        Ql = np.hstack( [dTs , Ts])
        
        return np.vstack([Qu,Ql])


    def DCM_i2r(self,t):
        """ Direction cosine matrix to rotate from Inertial Frame to Rotating Frame
        """
        
        Ts  = np.array([[ np.cos(t) ,-np.sin(t) , 0],\
                        [ np.sin(t) , np.cos(t) , 0],\
                        [   0       ,    0      , 1]])
        
        dTs = np.array([[-np.sin(t) ,-np.cos(t) , 0],\
                        [ np.cos(t) ,-np.sin(t) , 0],\
                        [   0       ,    0      , 0]])
        
        argD = np.matmul( np.matmul(-Ts.T,dTs) , Ts.T )
        Qu = np.hstack( [  Ts.T , np.zeros([3,3])])
        Ql = np.hstack( [  argD , Ts.T])
        
        return np.vstack([Qu,Ql])

    def lagrangeMult(self):
        """ Generate a random lagrange multiplier for initial guess (6x1)
        """
        
        Lr = np.random.uniform(1,5)
        Lv = np.random.uniform(1,5)
        
        alpha_r = np.random.uniform(0,2*np.pi)
        alpha_v = np.random.uniform(0,2*np.pi)
        
        delta_r = np.random.uniform(-np.pi/2,np.pi/2)
        delta_v = np.random.uniform(-np.pi/2,np.pi/2)
        
        L = np.array([Lr*np.cos(alpha_r)*np.cos(delta_r),
                 Lr*np.sin(alpha_r)*np.cos(delta_r),
                 np.sin(delta_r),
                 Lv*np.cos(alpha_v)*np.cos(delta_v),
                 Lv*np.sin(alpha_v)*np.cos(delta_v),
                 np.sin(delta_v)
                 ])
        
        return L

# =============================================================================
# Unit conversions
# =============================================================================
        
    # converting time 
    def convertTime_to_canonical(self,normTime):
        """ Convert time to canonical units
        """
        normTime = normTime.to('yr')
        return normTime.value * (2*np.pi)

    def convertTime_to_dim(self,normTime):
        """ Convert time to years
        """
        normTime = normTime / (2*np.pi) 
        return normTime * u.yr

    # converting length
    def convertPos_to_canonical(self,pos):
        """ Convert position to canonical units
        """
        pos = pos.to('au')
        return pos.value
    
    def convertPos_to_dim(self,pos):
        """ Convert position to canonical units
        """
        return pos * u.au 

    # converting velocity
    def convertVel_to_canonical(self,vel):
        """ Convert velocity to canonical units
        """
        vel = vel.to('au/yr')
        return vel.value / (2*np.pi)

    def convertVel_to_dim(self,vel):
        """ Convert velocity to canonical units
        """
        vel = vel * (2*np.pi)
        return vel * u.au / u.yr

    # converting acceleration
    def convertAcc_to_canonical(self,acc):
        """ Convert velocity to canonical units
        """
        acc = acc.to('au/yr**2')
        return acc.value / (2*np.pi)**2

    def convertAcc_to_dim(self,acc):
        """ Convert velocity to canonical units
        """
        acc = acc * (2*np.pi)**2
        return acc * u.au / u.yr**2

# =============================================================================
# Helper functions
# =============================================================================
        
    def determineThrottle(self,state):
        """ Determines throttle based on instantaneous switching function value
        """
        
        eps = self.epsilon
        x,y,z,dx,dy,dz,m,L1,L2,L3,L4,L5,L6,L7 = state
        _,n = state.shape
        
        Lv_, lv = self.unitVector( np.array([L4,L5,L6]) )
        
        S = -lv*self.ve/m - L7 + 1
        
        throttle = np.zeros(n)
        for i,s in enumerate(S):
            if eps > 0:
                midthrottle = (eps - s)/(2*eps)
                throttle[i] = 0 if s > eps else 1 if s < -eps else midthrottle
            else:
                throttle[i] = 0 if s > eps else 1
        
        return throttle

# =============================================================================
# Equations of Motion and Boundary Conditions
# =============================================================================
    def boundary_conditions_thruster(self,sA,sB,constrained=False):
        """ Creates boundary conditions for solving a boundary value problem
        """
    
        BCo1 = sA[0] - self.sA[0]
        BCo2 = sA[1] - self.sA[1]
        BCo3 = sA[2] - self.sA[2]
        BCo4 = sA[3] - self.sA[3]
        BCo5 = sA[4] - self.sA[4]
        BCo6 = sA[5] - self.sA[5]
        
        BCf1 = sB[0] - self.sB[0]
        BCf2 = sB[1] - self.sB[1]
        BCf3 = sB[2] - self.sB[2]
        BCf4 = sB[3] - self.sB[3]
        BCf5 = sB[4] - self.sB[4]
        BCf6 = sB[5] - self.sB[5]
        
        if constrained:
            BCo7 = sA[6] - self.sA[6]
            BCf7 = sB[-1]
            BC = np.array([BCo1,BCo2,BCo3,BCo4,BCo5,BCo6,BCo7,BCf1,BCf2,BCf3,BCf4,BCf5,BCf6,BCf7])
        else:
            BC = np.array([BCo1,BCo2,BCo3,BCo4,BCo5,BCo6,BCf1,BCf2,BCf3,BCf4,BCf5,BCf6])

        return BC   
     
        
    def EoM_Adjoint(self,t,state,constrained=False,amax=False,):
        """ Equations of Motion with costate vectors
        """
        
        mu = self.mu
        ve   = self.ve
        if amax:
            x,y,z,dx,dy,dz,m,L1,L2,L3,L4,L5,L6,L7 = state
        else:
            x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6 = state
        _,n = state.shape
        
        # vector distances from primaries
        r1   = np.array([x-(-mu),y,z])
        r2   = np.array([x-(1-mu),y,z])
        # norms of distances from primaries
        R1   = np.linalg.norm(r1,axis=0)
        R2   = np.linalg.norm(r2,axis=0)
        
        # Position-dependent acceleration terms
        qx = np.array([ x  - (1-mu)*(x+mu)/R1**3  - mu*(x+mu-1)/R2**3]).reshape(1,n)
        qy = np.array([ y  - (1-mu)*y/R1**3       - mu*y/R2**3]).reshape(1,n)
        qz = np.array([    - (1-mu)*z/R1**3       - mu*z/R2**3]).reshape(1,n)
        q = np.vstack([qx,qy,qz])  #shape of 3xn
        
        # Position partial derivatives
        Q11 = 1 - (1-mu)/R1**3 + 3*(1-mu)*(x+mu)**2/R1**5  - mu/R2**3   + 3*mu*(x+mu-1)**2/R2**5
        Q22 = 1 - (1-mu)/R1**3 + 3*(1-mu)*     y**2/R1**5  - mu/R2**3   + 3*mu*       y**2/R2**5
        Q33 =   - (1-mu)/R1**3 + 3*(1-mu)*     z**2/R1**5  - mu/R2**3   + 3*mu*       z**2/R2**5
        Q12 =                    3*(1-mu)*(x+mu)* y/R1**5               + 3*mu*(x+mu-1)* y/R2**5
        Q13 =                    3*(1-mu)*(x+mu)* z/R1**5               + 3*mu*(x+mu-1)* z/R2**5
        Q23 =                    3*(1-mu)*      y*z/R1**5               + 3*mu*        y*z/R2**5
        Qr   = np.array([[Q11, Q12 , Q13], [Q12, Q22 , Q23], [Q13, Q23 , Q33]]) # shape of 3x3xn

        # Velocity-dependent acceleration terms
        px =  2*dy
        py = -2*dx
        pz = np.zeros([1,n])
        p  = np.vstack([px,py,pz]) #shape of 3xn
        
        # Velocity partial derivatives
        Pv_arr = np.array([[0,2,0],[-2,0,0],[0,0,0]])
        Pv = np.dstack([Pv_arr]*n)
        
        # Costate vectors
        Lr = np.vstack([L1,L2,L3])
        Lv = np.vstack([L4,L5,L6])
        Lr_, lr = self.unitVector(Lr)
        Lv_, lv = self.unitVector(Lv)
        
        # ================================================
        # Equations of Motion
        # ================================================
        dX  = np.vstack([ dx,dy,dz ])
        dV  = q + p
        dLx = -np.vstack( [np.dot(a.T,b) for a,b in zip(Qr.T,Lv.T)] ).T
        dLv = -Lr - np.vstack( [np.dot(a.T,b) for a,b in zip(Pv.T,Lv.T)] ).T
        
        if amax:
            # throttle factor
            throttle = self.determineThrottle(state)
            
            dV -= Lv_ * amax * throttle / m
            dm  = -throttle * amax / ve
            dLm = -lv * throttle * amax / m**2
            # putting them all together, a 14xn array
            f   = np.vstack([ dX, dV, dm, dLx, dLv, dLm ])
        else:
            dV -= Lv
            # putting them all together, a 12xn array
            f   = np.vstack([ dX, dV, dLx, dLv ])
        
        return f
    
# =============================================================================
# Initial conditions
# =============================================================================

    def findInitialTmax(self,TL,nA,nB,tA,dt,s_init=np.array([])):
        """ Finding initial guess for starting Thrust
        """
        
        tB = tA + dt
        angle,uA,uB,r_tscp = self.lookVectors(TL,nA,nB,tA,tB)

        #position vector of occulter in heliocentric frame
        self_rA = uA*self.occulterSep.to('au').value + r_tscp[ 0]
        self_rB = uB*self.occulterSep.to('au').value + r_tscp[-1]
        
        self_vA = self.haloVelocity(tA)[0].value/(2*np.pi)
        self_vB = self.haloVelocity(tB)[0].value/(2*np.pi)
                
        self_sA = np.hstack([self_rA,self_vA])
        self_sB = np.hstack([self_rB,self_vB])
                
        self_fsA = np.hstack([self_sA, self.lagrangeMults])
        self_fsB = np.hstack([self_sB, self.lagrangeMults])
                
        a = ((np.mod(tA.value,self.equinox.value)*u.d)).to('yr') / u.yr * (2*np.pi)
        b = ((np.mod(tB.value,self.equinox.value)*u.d)).to('yr') / u.yr * (2*np.pi)
                
        #running collocation
        tGuess = np.hstack([a,b]).value

        if s_init.size:
            sGuess = np.vstack([s_init[:,0] , s_init[:,-1]])
        else:
            sGuess = np.vstack([self_fsA , self_fsB])
            
        s,t_s,status = self.send_it_thruster(sGuess.T,tGuess,verbose=False)
        
        lv = s[9:,:]

        aNorms0 = np.linalg.norm(lv,axis=0)
        aMax0   = self.convertAcc_to_dim( np.max(aNorms0) ).to('m/s^2') 
        Tmax0   = (aMax0 * self.mass ).to('N')
        
        return Tmax0, s, t_s
    
    def findTmaxGrid(self,TL,tA,dtRange):
        """ Create grid of Tmax values using unconstrained thruster
        """
        
        midInt = int( np.floor( (TL.nStars-1)/2 ) )

        sInds = np.arange(0,TL.nStars)
        ang   =  self.star_angularSep(TL, midInt, sInds, tA) 
        sInd_sorted = np.argsort(ang)
        angles  = ang[sInd_sorted].to('deg').value
        
        
        TmaxMap = np.zeros([len(dtRange) , len(angles)])*u.N
        
        for i,t in enumerate(dtRange):
            for j,n in enumerate(sInd_sorted):
                print(i,j)
                Tmax, s, t_s = self.findInitialTmax(TL,midInt,n,tA,t)
                TmaxMap[i,j] = Tmax
        
        return TmaxMap

    def findTmaxGrid_sequential(self,TL,tA,dtRange):
        """ Create grid of Tmax values using unconstrained thruster
        """
        
        midInt = int( np.floor( (TL.nStars-1)/2 ) )

        sInds = np.arange(0,TL.nStars)
        ang   =  self.star_angularSep(TL, midInt, sInds, tA) 
        sInd_sorted = np.argsort(ang)
        angles  = ang[sInd_sorted].to('deg').value
        
        TmaxMap = np.zeros([len(dtRange) , len(angles)])*u.N
        d = len(dtRange)
        sGuess = []
        
        for i in range(d-1,-1,-1):
            for j,n in enumerate(sInd_sorted):
                print(i,j)
                
                if i == d-1:
                    Tmax, s, t_s = self.findInitialTmax(TL,midInt,n,tA,dtRange[i])
                    TmaxMap[i,j] = Tmax
                    sGuess.append(s)
                else:
                    Tmax, s, t_s = self.findInitialTmax(TL,midInt,n,tA,dtRange[i],sGuess[j])
                    TmaxMap[i,j] = Tmax
                    sGuess[j] = s
        
        return TmaxMap

# =============================================================================
# BVP solvers
# =============================================================================
  
    def send_it_thruster(self,sGuess,tGuess,aMax=False,constrained=False,maxNodes=1e5,verbose=False):
        """ Solving generic bvp from t0 to tF using states and costates
        """
        
        sG = sGuess
        if len(sGuess) == 12:
            x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6 = sGuess

            if aMax:
                mRange  = np.linspace(1, 0.8, len(x))
                lmRange = np.linspace(1, 0, len(x))
                sG = np.vstack([x,y,z,dx,dy,dz,mRange,L1,L2,L3,L4,L5,L6,lmRange])
        
        self.sA = sG[:,0]
        self.sB = sG[:,-1]
        self.sG = sG
        
        EoM = lambda t,s: self.EoM_Adjoint(t,s,constrained,aMax)
        BC  = lambda t,s: self.boundary_conditions_thruster(t,s,constrained)
        sol = solve_bvp(EoM,BC,tGuess,sG,tol=1e-8,max_nodes=int(maxNodes),verbose=0)
        
        if verbose:
            self.vprint(sol.message)
        
        s = sol.y
        t_s = sol.x
            
        return s,t_s,sol.status
    
    
    def collocate_ContThrustTrajectory(self,TL,nA,nB,tA,dt):
        """ Solves minimum energy and minimum fuel cases for continuous thrust
        
        Returns:
            s       - trajectory
            t_s     - time of trajectory
            dm      - mass used
            epsilon - last epsilon that fully converged (2 if minimum energy didn't work)
            
        """
        
        # initializing arrays
        stateLog = []
        timeLog  = []
        
        # solving using unconstrained thruster as initial guess
        Tmax, sTmax, tTmax = self.findInitialTmax(TL,nA,nB,tA,dt)
        aMax = self.convertAcc_to_canonical( (Tmax / self.mass).to('m/s^2') )
        # saving results
        stateLog.append(sTmax)
        timeLog.append(tTmax)
        
        # all thrusts were successful
        e_best   = 2
        s_best   = deepcopy(sTmax)
        t_best   = deepcopy(tTmax)
        u_best   = []
        
        # thrust values
        desiredT = self.Tmax.to('N').value
        currentT = Tmax.value
        # range of thrusts to try
        tMaxRange    = np.linspace(currentT,desiredT,20)
        
        # range of epsilon values to try (e=1 is minimum energy, e=0 is minimum fuel)
        epsilonRange = np.round( np.arange(1,-0.1,-0.1) , decimals = 1)
        
        # Huge loop over all thrust and epsilon values:
        #   we start at the minimum energy case, e=1, using the thrust value from the unconstrained solution
        #     In/De-crement the thrust until we reach the desired thrust level
        #   then we decrease e and repeat process until we get to e=0 (minimum fuel)
        #   saves the last successful result in case collocation fails
        
        # loop over epsilon starting with e=1
        for j,e in enumerate(epsilonRange):
#            print("Epsilon = ",e)
            # initialize epsilon
            self.epsilon = e
            
            # loop over thrust values from current to desired thrusts
            for i,thrust in enumerate(tMaxRange):
#                print("Thrust #",i," / ",len(tMaxRange))
                # convert thrust to canonical acceleration
                aMax = self.convertAcc_to_canonical( (thrust*u.N / self.mass).to('m/s^2') )
                # retrieve state and time initial guesses
                sGuess = stateLog[i]
                tGuess = timeLog[i]
                # perform collocation
                s,t_s,status = self.send_it_thruster(sGuess,tGuess,aMax,constrained=True,maxNodes=1e5,verbose=False)
                throttle     = self.determineThrottle(s)
                
                # collocation failed, exits out of everything
                if status != 0:
                    return s_best, t_best, u_best, e_best

                # collocation was successful!
                if j == 0:
                    # creates log of state and time results for next thrust iteration (at the beginning of the loop)
                    stateLog.append(s)
                    timeLog.append(t_s)
                else:
                    # updates log of state and time results for next thrust iteration
                    stateLog[i] = s
                    timeLog[i]  = t_s
            
            # all thrusts were successful
            e_best   = self.epsilon
            s_best   = deepcopy(s)
            t_best   = deepcopy(t_s)
            u_best   = deepcopy(throttle)
        
        return s_best, t_best, u_best, e_best

    def calculate_dMmap(self,TL,tA,dtRange):
        
        midInt = int( np.floor( (TL.nStars-1)/2 ) )

        sInds       = np.arange(0,TL.nStars)
        ang         = self.star_angularSep(TL, midInt, sInds, tA) 
        sInd_sorted = np.argsort(ang)
        angles      = ang[sInd_sorted].to('deg').value
        
        self.dMmap = np.zeros([len(dtRange) , len(angles)])*u.kg
        self.eMap  = np.zeros([len(dtRange) , len(angles)])
        
        for i,t in enumerate(dtRange):
            for j,n in enumerate(sInd_sorted):
                print(i,j)
                s_best, t_best, u_best, e_best = self.collocate_ContThrustTrajectory(TL, \
                                                  midInt,n,tA,t)
                
                m = s_best[6,:] * self.mass
                dm = m[-1] - m[0]
                self.dMmap[i,j] = m[-1] - m[0]
                self.eMap[i,j]  = e_best
                
                print('Mass - ',dm)
                print('Best Epsilon - ',e_best)
        