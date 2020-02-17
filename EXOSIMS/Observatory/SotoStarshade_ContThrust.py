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
        
        Finds rotation matrix for positions and velocities (6x6)
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
        
        Finds rotation matrix for positions and velocities (6x6)
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
    
    
    def DCM_r2i_9(self,t):
        """
        
        Finds rotation matrix for positions, velocities, and accelerations (9x9)
        """
        
        Ts  = np.array([[ np.cos(t) ,-np.sin(t) , 0],\
                        [ np.sin(t) , np.cos(t) , 0],\
                        [   0       ,    0      , 1]])
        
        dTs = np.array([[-np.sin(t) ,-np.cos(t) , 0],\
                        [ np.cos(t) ,-np.sin(t) , 0],\
                        [   0       ,    0      , 0]])
    
        ddTs = -np.array([[ np.cos(t) ,-np.sin(t) , 0],\
                          [ np.sin(t) , np.cos(t) , 0],\
                          [   0       ,    0      , 0]])
        
        Qu = np.hstack( [  Ts ,  np.zeros([3,6])                 ])
        Qm = np.hstack( [ dTs ,  Ts             , np.zeros([3,3])])
        Ql = np.hstack( [ddTs ,2*dTs             , Ts])
        
        return np.vstack([Qu,Qm,Ql])
        
        

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
        """ Determines throttle based on instantaneous switching function value.
        
        Typically being used during collocation algorithms. A zero-crossing of
        the switching function is highly unlikely between the large number of 
        nodes. 
        """
        
        eps = self.epsilon
        n = 1 if state.size == 14 else state.shape[1]
        
        throttle = np.zeros(n)
        S = self.switchingFunction(state)
        S = S.reshape(n)
        
        for i,s in enumerate(S):
            if eps > 0:
                midthrottle = (eps - s)/(2*eps)
                throttle[i] = 0 if s > eps else 1 if s < -eps else midthrottle
            else:
                throttle[i] = 0 if s > eps else 1
        
        return throttle
    
    
    def switchingFunction(self,state):
        """ Evaluates the switching function at specific states.
        """
        
        x,y,z,dx,dy,dz,m,L1,L2,L3,L4,L5,L6,L7 = state
        
        Lv_, lv = self.unitVector( np.array([L4,L5,L6]) )
        
        S = -lv*self.ve/m - L7 + 1
        
        return S

    def switchingFunctionDer(self,state):
        """ Evaluates the time derivative of the switching function.
        
        Switching function derivative evaluated for specific states. 
        """
        ve  = self.ve
        n = 1 if state.size == 14 else state.shape[1]
        x,y,z,dx,dy,dz,m,L1,L2,L3,L4,L5,L6,L7 = state
        
        Lr = np.array([L1,L2,L3]).reshape(3,n)
        Lv = np.array([L4,L5,L6]).reshape(3,n)
        Lv_, lv = self.unitVector( Lv )
        
        Pv_arr = np.array([[0,2,0],[-2,0,0],[0,0,0]])
        Pv = np.dstack([Pv_arr]*n)
        
        PLdot = np.vstack( [np.dot(a.T,b) for a,b in zip(Pv.T,Lv.T)] ).T
        
        dS = -(ve / m) * np.vstack( [np.dot(a.T,b) for a,b in zip( (-Lr - PLdot).T ,Lv_.T)] ).T 

        return dS
    
    def selectEventFunctions(self,s0):
        """ Selects the proper event function for integration.
        
        This method calculates the switching function and its derivative at a 
        single specific state. It then determines which thrust case it will be 
        in: full, medium, or no thrust. If the value of the switching function
        is within a certain tolerance of the boundaries, it uses the derivative 
        to determine the direction it is heading in. Then the proper event 
        functions are created for the integrator to determine the next crossing
        (i.e. the next case change). 
        
        """
        eps = self.epsilon
        
        S  =  self.switchingFunction(s0)
        dS =  self.switchingFunctionDer(s0)[0]

        # finding which case we are in:
        #   - case 2 if       -eps < S < eps       (medium thrust)
        #   - case 1 if   S < -eps                 (full thrust)
        #   - case 0 if                  eps < S   (no thrust)
        
        case = 0 if S > eps else 1 if S < -eps else 2

        # checking to see if S is within a certain tolerance from epsilon
        withinTol = np.abs( (np.abs(S) - eps) ) < 1e-10
        # determine if there is a case error if within tolerance
        if withinTol:
            # not the minimum fuel case
            if eps != 0:
                # at the upper bound, case determined by derivative
                if S > 0:
                    case = 2 if dS < 0 else 0
                # at the lower bound, case determined by derivative
                else:
                    case = 2 if dS > 0 else 1
            # minimum fuel case, only two cases
            else:
                case = 0 if dS > 0 else 1
            
            
        eventFunctions = []
        CrossingUpperBound = lambda t,s : self.switchingFunction(s) - eps
        CrossingLowerBound = lambda t,s : self.switchingFunction(s) + eps
        
        CrossingUpperBound.terminal = True
        CrossingLowerBound.terminal = True
        
        if case == 0:
            # crossing upper epsilon from above
            CrossingUpperBound.direction = -1 
            # appending event function
            eventFunctions.append(CrossingUpperBound)
        elif case == 1:
            # crossing lower epsilon from below
            CrossingLowerBound.direction = 1 
            # appending event function
            eventFunctions.append(CrossingLowerBound)
        else:
            # can either cross lower epsilon from above or upper from below
            CrossingLowerBound.direction = -1
            CrossingUpperBound.direction = 1
            # appending event function
            eventFunctions.append(CrossingUpperBound)
            eventFunctions.append(CrossingLowerBound)
        
        return eventFunctions,case
        

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
     
        
    def EoM_Adjoint(self,t,state,constrained=False,amax=False,integrate=False):
        """ Equations of Motion with costate vectors
        """
        
        mu = self.mu
        ve = self.ve
        
        n = 1 if state.size == 14 else state.shape[1]

        if amax:
            x,y,z,dx,dy,dz,m,L1,L2,L3,L4,L5,L6,L7 = state
        else:
            x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6 = state
        
        if integrate:
            state = state.T
        
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
        Qr   = np.array([[Q11, Q12 , Q13], [Q12, Q22 , Q23], [Q13, Q23 , Q33]]).reshape(3,3,n) # shape of 3x3xn

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
        
        
        if integrate:
            f = f.flatten()

        return f
    
# =============================================================================
# Initial conditions
# =============================================================================

    def findInitialTmax(self,TL,nA,nB,tA,dt,m0=1,s_init=np.array([])):
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
                
        self_fsA = np.hstack([self_sA, self.lagrangeMult()])
        self_fsB = np.hstack([self_sB, self.lagrangeMult()])
                
        a = ((np.mod(tA.value,self.equinox.value)*u.d)).to('yr') / u.yr * (2*np.pi)
        b = ((np.mod(tB.value,self.equinox.value)*u.d)).to('yr') / u.yr * (2*np.pi)
                
        #running collocation
        tGuess = np.hstack([a,b]).value

        if s_init.size:
            sGuess = np.vstack([s_init[:,0] , s_init[:,-1]])
        else:
            sGuess = np.vstack([self_fsA , self_fsB])
            
        s,t_s,status = self.send_it_thruster(sGuess.T,tGuess,m0=m0,verbose=False)
        
        lv = s[9:,:]

        aNorms0 = np.linalg.norm(lv,axis=0)
        aMax0   = self.convertAcc_to_dim( np.max(aNorms0) ).to('m/s^2') 
        Tmax0   = (aMax0 * self.mass ).to('N')
        
        return Tmax0, s, t_s
    
    
    def findTmaxGrid(self,TL,tA,dtRange):
        """ Create grid of Tmax values using unconstrained thruster
        
        This method is used purely for creating figures. 
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

# =============================================================================
# BVP solvers
# =============================================================================
  
    def send_it_thruster(self,sGuess,tGuess,aMax=False,constrained=False, m0 = 1, maxNodes=1e5,verbose=False):
        """ Solving generic bvp from t0 to tF using states and costates
        """
        
        sG = sGuess
        # unconstrained problem begins with 12 states, rather than 14. checking for that
        if len(sGuess) == 12:
            x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6 = sGuess
            # if unconstrained is initial guess for constrained problem, make 14 state array
            if aMax:
                mRange  = np.linspace(m0, 0.8, len(x))
                lmRange = np.linspace(1, 0, len(x))
                sG = np.vstack([x,y,z,dx,dy,dz,mRange,L1,L2,L3,L4,L5,L6,lmRange])

        # only saves initial and final desired states if first solving unconstrained problem
        if not constrained:
            self.sA = np.hstack([ sG[0:6,0 ], m0   , sG[6:,0]  , 1 ])
            self.sB = np.hstack([ sG[0:6,-1], 0.8 , sG[6:,-1] , 0 ])
            self.sG = sG
        
        # creating equations of motion and boundary conditions functions
        EoM = lambda t,s: self.EoM_Adjoint(t,s,constrained,aMax)
        BC  = lambda t,s: self.boundary_conditions_thruster(t,s,constrained)
        # solving BVP
        sol = solve_bvp(EoM,BC,tGuess,sG,tol=1e-8,max_nodes=int(maxNodes),verbose=0)
        
        if verbose:
            self.vprint(sol.message)
        # saving results
        s = sol.y
        t_s = sol.x
            
        return s,t_s,sol.status
    
    
    def collocate_Trajectory(self,TL,nA,nB,tA,dt):
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
        s_best   = deepcopy(stateLog)
        t_best   = deepcopy(timeLog)
        
        # thrust values
        desiredT = self.Tmax.to('N').value
        currentT = Tmax.value
        # range of thrusts to try
        TmaxRange    = np.linspace(currentT,desiredT,30)
        
        # range of epsilon values to try (e=1 is minimum energy, e=0 is minimum fuel)
        epsilonRange = np.round( np.arange(1,-0.1,-0.1) , decimals = 1)
        
        # Huge loop over all thrust and epsilon values:
        #   we start at the minimum energy case, e=1, using the thrust value from the unconstrained solution
        #     In/De-crement the thrust until we reach the desired thrust level
        #   then we decrease e and repeat process until we get to e=0 (minimum fuel)
        #   saves the last successful result in case collocation fails
        
        # loop over epsilon starting with e=1
        for j,e in enumerate(epsilonRange):
            print("Collocate Epsilon = ",e)
            # initialize epsilon
            self.epsilon = e
            
            # loop over thrust values from current to desired thrusts
            for i,thrust in enumerate(TmaxRange):
                #print("Thrust #",i," / ",len(TmaxRange))
                # convert thrust to canonical acceleration
                aMax = self.convertAcc_to_canonical( (thrust*u.N / self.mass).to('m/s^2') )
                # retrieve state and time initial guesses
                sGuess = stateLog[i]
                tGuess = timeLog[i]
                # perform collocation
                s,t_s,status = self.send_it_thruster(sGuess,tGuess,aMax,constrained=True,maxNodes=1e5,verbose=False)
                
                # collocation failed, exits out of everything
                if status != 0:
                    self.epsilon = e_best
                    if e_best == 2:
                        # if only the unconstrained problem worked, still returns a 14 length array
                        s_out = []
                        length = s_best[0].shape[1]
                        m  = np.linspace(1,0.9,length)
                        lm = np.linspace(0.3,0,length)
                        s_out.append( np.vstack([s_best[0][:6] , m, s_best[0][6:], lm]) )
                        s_best = deepcopy(s_out)
                    return s_best, t_best, e_best, TmaxRange
                
                # collocation was successful!
                if j == 0:
                    # creates log of state and time results for next thrust iteration (at the beginning of the loop)
                    stateLog.append(s)
                    timeLog.append(t_s)
                else:
                    # updates log of state and time results for next thrust iteration
                    stateLog[i] = s
                    timeLog[i]  = t_s
            
            # all thrusts were successful, save results
            e_best   = self.epsilon
            s_best   = deepcopy(stateLog)
            t_best   = deepcopy(timeLog)
        
        return s_best, t_best, e_best, TmaxRange


    def collocate_Trajectory_minEnergy(self,TL,nA,nB,tA,dt,m0=1):
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
        Tmax, sTmax, tTmax = self.findInitialTmax(TL,nA,nB,tA,dt,m0)
        aMax = self.convertAcc_to_canonical( (Tmax / self.mass).to('m/s^2') )
        # saving results
        stateLog.append(sTmax)
        timeLog.append(tTmax)
        
        # all thrusts were successful
        e_best   = 2
        s_best   = deepcopy(sTmax)
        t_best   = deepcopy(tTmax)
        
        # thrust values
        desiredT = self.Tmax.to('N').value
        currentT = Tmax.value
        # range of thrusts to try
        TmaxRange    = np.linspace(currentT,desiredT,30)
        
        # Huge loop over all thrust and epsilon values:
        #   we start at the minimum energy case, e=1, using the thrust value from the unconstrained solution
        #     In/De-crement the thrust until we reach the desired thrust level
        #   then we decrease e and repeat process until we get to e=0 (minimum fuel)
        #   saves the last successful result in case collocation fails
        
        # loop over epsilon starting with e=1
        self.epsilon = 1
        
        # loop over thrust values from current to desired thrusts
        for i,thrust in enumerate(TmaxRange):
            #print("Thrust #",i," / ",len(TmaxRange))
            # convert thrust to canonical acceleration
            aMax = self.convertAcc_to_canonical( (thrust*u.N / self.mass).to('m/s^2') )
            # retrieve state and time initial guesses
            sGuess = stateLog[i]
            tGuess = timeLog[i]
            # perform collocation
            s,t_s,status = self.send_it_thruster(sGuess,tGuess,aMax,constrained=True,m0=m0,maxNodes=1e5,verbose=False)
            
            # collocation failed, exits out of everything
            if status != 0:
                self.epsilon = e_best
                return s_best, t_best, e_best, TmaxRange
            
            # creates log of state and time results for next thrust iteration (at the beginning of the loop)
            stateLog.append(s)
            timeLog.append(t_s)

        # all thrusts were successful, save results
        e_best   = self.epsilon
        s_best   = deepcopy(s)
        t_best   = deepcopy(t_s)
    
        return s_best, t_best, e_best, TmaxRange
    
# =============================================================================
# Shooting Algorithms
# =============================================================================
        
    def integrate_thruster(self,sGuess,tGuess,Tmax,verbose=False):
        """ Integrates thruster trajectory with thrust case switches
        
        This methods integrates an initial guess for the spacecraft state 
        forwards in time. It uses event functions to find the next zero of the
        switching function which means a new thrust case is needed (full, 
        medium or no thrust). 
        
        """
        s0 = sGuess[:,0]
        t0 = tGuess[0]
        tF = tGuess[-1]
        
        # initializing starting time
        tC = deepcopy(t0)
        
        # converting Tmax to a canonical acceleration
        Tmax = Tmax.to('N').value
        aMax = self.convertAcc_to_canonical( (Tmax*u.N / self.mass).to('m/s^2') )
        
        # establishing equations of motion
        EoM  = lambda t,s: self.EoM_Adjoint(t,s,constrained=True,amax=aMax,integrate=True)
        
        # starting integration
        count = 0
        while tC < tF:
            # selecting the switch functions with correct boundaries
            switchFunctions,case = self.selectEventFunctions(s0)
            if verbose:
                print("[%.3f / %.3f] with case %d" % (tC,tF,case) )
            # running integration with event functions
            res = solve_ivp(EoM, [tC , tF], s0, events=switchFunctions)
            
            # saving final integration time, if greater than tF, we completed the trajectory
            tC = deepcopy(res.t[-1])
            s0 = deepcopy(res.y[:,-1])
            
            # saving results in a new array
            if count == 0:
                sLog = deepcopy(res.y)
                tLog = deepcopy(res.t)
            #adding to the results log if there was a previous thrust case switch
            else:
                sLog = np.hstack([sLog,res.y])
                tLog = np.hstack([tLog,res.t])
            count += 1
            
        return sLog,tLog


    def conFun_singleShoot(self,w,t0,tF,Tmax,returnLog=False):
        """ Objective Function for single shooting thruster
        """
        
        sInit  = np.hstack([self.sA[:7] , w]).reshape(14,1)
        tGuess = np.array([t0,tF])
        
        sLog,tLog = self.integrate_thruster(sInit,tGuess,Tmax)
        
        f = self.boundary_conditions_thruster( sLog[:,0], sLog[:,-1], constrained=True)
        fnorm = np.linalg.norm(f)
        
        if returnLog:
            return fnorm,sLog,tLog
        else:
            return fnorm


    def minimize_TerminalState(self,s_best,t_best,Tmax,method):
        """ Minimizes boundary conditions for thruster
        """
        
        w0 = s_best[7:,0]
        t0 = t_best[0]
        tF = t_best[-1]
        
        res = optimize.minimize(self.conFun_singleShoot, w0, method=method, \
                                tol=1e-12, args=(t0,tF,Tmax,) )
#        minimizer_kwargs = {"method":method,"args":(t0,tF,Tmax,)}
#        res = optimize.basinhopping(self.conFun_singleShoot,w0,minimizer_kwargs=minimizer_kwargs)
#        
        fnorm, sLog, tLog = self.conFun_singleShoot(res.x,t0,tF,Tmax,returnLog=True)
        
        return fnorm, sLog, tLog


    def singleShoot_Trajectory(self,stateLog,timeLog,e_best,TmaxRange,method='SLSQP'):
        
        # initializing arrays
        s_best = deepcopy(stateLog)
        t_best = deepcopy(timeLog)
        
        incLogFlag = True if len(stateLog) != len(TmaxRange) else False
        
        # range of epsilon values to try (e=1 is minimum energy, e=0 is minimum fuel)
        e_best = 1 if e_best == 2 else e_best
        epsilonRange = np.round( np.arange(e_best,-0.1,-0.1) , decimals = 1)
        
        # Huge loop over all thrust and epsilon values, just like collocation method
        
        # loop over epsilon starting with e=1
        for j,e in enumerate(epsilonRange):
            print("SS Epsilon = ",e)
            # initialize epsilon
            self.epsilon = e
            
            # loop over thrust values from current to desired thrusts
            for i,thrust in enumerate(TmaxRange):
                #print("Thrust #",i," / ",len(TmaxRange))
                # retrieve state and time initial guesses
                sGuess = stateLog[i]
                tGuess = timeLog[i]
                # perform single shooting
                fnorm,sLog,tLog = self.minimize_TerminalState(sGuess,tGuess,thrust,method)
                print(fnorm)
                # single shooting failed, exits out of everything
                if fnorm > 1e-7:
                    self.epsilon = e_best
                    return s_best, t_best, e_best
                
                # single shooting was successful!
                if incLogFlag:
                    # appends stateLog if the input was incomplete
                    stateLog.append(sLog)
                    timeLog.append(tLog)
                else:
                    # updates log of state and time results for next thrust iteration
                    stateLog[i] = sLog
                    timeLog[i]  = tLog
            
            # all thrusts were successful, save results
            e_best   = self.epsilon
            s_best   = deepcopy(sLog)
            t_best   = deepcopy(tLog)
        
        return s_best, t_best, e_best
    
# =============================================================================
#  Putting it al together
# =============================================================================

    def calculate_dMmap(self,TL,tA,dtRange,filename):

        sInds       = np.arange(0,TL.nStars)
        ang         = self.star_angularSep(TL, 0, sInds, tA) 
        sInd_sorted = np.argsort(ang)
        angles      = ang[sInd_sorted].to('deg').value

        dtFlipped = np.flipud(dtRange)
        
        self.dMmap = np.zeros([len(dtRange) , len(angles)])
        self.eMap  = np.zeros([len(dtRange) , len(angles)])
        
        tic = time.perf_counter()
        for j,n in enumerate(sInd_sorted):
            for i,t in enumerate(dtFlipped):
                print(i,j)
                s_coll, t_coll, e_coll, TmaxRange = \
                            self.collocate_Trajectory(TL,0,n,tA,t)
                
                if e_coll != 0:
                    s_ssm, t_ssm, e_ssm = self.singleShoot_Trajectory(s_coll, \
                                                t_coll,e_coll,TmaxRange*u.N)

                if e_ssm == 2 and t.value < 30:
                    break
                
                m = s_ssm[-1][6,:] 
                dm = m[-1] - m[0]
                self.dMmap[i,j] = dm
                self.eMap[i,j]  = e_ssm
                toc = time.perf_counter()
                
                dmPath = os.path.join(self.cachedir, filename+'.dmmap')
                A = {'dMmap':self.dMmap,'eMap':self.eMap,'angles':angles,'dtRange':dtRange,'time':toc-tic,\
                     'tA':tA,'m0':1,'ra':TL.coords.ra,'dec':TL.coords.dec,'mass':self.mass}
                with open(dmPath, 'wb') as f:
                    pickle.dump(A, f)
                print('Mass - ',dm*self.mass)
                print('Best Epsilon - ',e_ssm)
    
    
    def calculate_dMmap_collocate(self,TL,tA,dtRange,filename):
        
        sInds       = np.arange(0,TL.nStars)
        ang         = self.star_angularSep(TL, 0, sInds, tA) 
        sInd_sorted = np.argsort(ang)
        angles      = ang[sInd_sorted].to('deg').value

        dtFlipped = np.flipud(dtRange)
        
        self.dMmap = np.zeros([len(dtRange) , len(angles)])
        self.eMap  = np.zeros([len(dtRange) , len(angles)])
        
        tic = time.perf_counter()
        for j,n in enumerate(sInd_sorted):
            for i,t in enumerate(dtFlipped):
                print(i,j)
                s_coll, t_coll, e_coll, TmaxRange = \
                            self.collocate_Trajectory(TL,0,n,tA,t)

                if e_coll == 2 and t.value < 30:
                    break
                
                m = s_coll[-1][6,:]
                dm = m[-1] - m[0]
                self.dMmap[i,j] = dm
                self.eMap[i,j]  = e_coll
                toc = time.perf_counter()
                
                dmPath = os.path.join(self.cachedir, filename+'.dmmap')
                A = {'dMmap':self.dMmap,'eMap':self.eMap,'angles':angles,'dtRange':dtRange,'time':toc-tic,\
                     'tA':tA,'ra':TL.coords.ra,'dec':TL.coords.dec,'mass':self.mass}
                with open(dmPath, 'wb') as f:
                    pickle.dump(A, f)
                print('Mass - ',dm*self.mass)
                print('Best Epsilon - ',e_coll)
    
    
    def calculate_dMmap_collocateEnergy(self,TL,tA,dtRange,filename,m0=1,seed=000000000):
        
        sInds       = np.arange(0,TL.nStars)
        ang         = self.star_angularSep(TL, 0, sInds, tA) 
        sInd_sorted = np.argsort(ang)
        angles      = ang[sInd_sorted].to('deg').value
        
        dtFlipped = np.flipud(dtRange)
        
        self.dMmap = np.zeros([len(dtRange) , len(angles)])
        self.eMap  = 2*np.ones([len(dtRange) , len(angles)])
        
        tic = time.perf_counter()
        for j,n in enumerate(sInd_sorted):
            for i,t in enumerate(dtFlipped):
                print(i,j)
                s_coll, t_coll, e_coll, TmaxRange = \
                            self.collocate_Trajectory_minEnergy(TL,0,n,tA,t,m0)
                
                # if unsuccessful, reached min time -> move on to next star
                if e_coll == 2 and t.value < 30:
                    break

                m = s_coll[6,:] 
                dm = m[-1] - m[0]
                self.dMmap[i,j] = dm
                self.eMap[i,j]  = e_coll
                toc = time.perf_counter()
                
                dmPath = os.path.join(self.cachedir, filename+'.dmmap')
                A = {'dMmap':self.dMmap,'eMap':self.eMap,'angles':angles,'dtRange':dtRange,'time':toc-tic,\
                     'tA':tA,'m0':m0,'ra':TL.coords.ra,'dec':TL.coords.dec,'seed':seed,'mass':self.mass}
                with open(dmPath, 'wb') as f:
                    pickle.dump(A, f)
                print('Mass - ',dm*self.mass)
                print('Best Epsilon - ',e_coll)


    def calculate_dMsols_collocateEnergy(self,TL,tStart,tArange,dtRange,N,filename,m0=1,seed=000000000):
        
        self.dMmap = np.zeros(N)
        self.eMap  = 2*np.ones(N)
        iLog   = np.zeros(N)
        jLog   = np.zeros(N)
        dtLog  = np.zeros(N)
        tALog  = np.zeros(N)
        angLog = np.zeros(N)*u.deg

        tic = time.perf_counter()
        for n in range(N):
                print("---------\nIteration",n)

                i = np.random.randint(0,TL.nStars)
                j = np.random.randint(0,TL.nStars)
                dt = np.random.randint(0,len(dtRange))
                tA = np.random.randint(0,len(tArange))
                ang = self.star_angularSep(TL,i,j,tStart+tArange[tA]) 

                print("star pair  :",i,j)
                print("ang  :",ang.to('deg').value)
                print("dt   :",dtRange[dt].to('d').value)
                print("tau  :",tArange[tA].to('d').value,"\n")

                pair = np.array([i,j])
                iLog[n]   = i
                jLog[n]   = j
                dtLog[n]  = dt
                tALog[n]  = tA
                angLog[n] = ang

                s_coll, t_coll, e_coll, TmaxRange = \
                            self.collocate_Trajectory_minEnergy(TL,i,j,tStart+tArange[tA],dtRange[dt],m0)
                
                # if unsuccessful, reached min time -> move on to next star
                if e_coll == 2 and dtRange[dt].value < 30:
                    break

                m = s_coll[6,:] 
                dm = m[-1] - m[0]
                self.dMmap[n] = dm
                self.eMap[n]  = e_coll
                toc = time.perf_counter()
                
                dmPath = os.path.join(self.cachedir, filename+'.dmsols')
                A = {'dMmap':self.dMmap,'eMap':self.eMap,'angLog':angLog,'dtLog':dtLog,'time':toc-tic,\
                     'tArange':tArange,'dtRange':dtRange,'N':N,'tStart':tStart,\
                     'tALog':tALog,'m0':m0,'ra':TL.coords.ra,'dec':TL.coords.dec,'seed':seed,'mass':self.mass}
                with open(dmPath, 'wb') as f:
                    pickle.dump(A, f)
                print('Mass - ',dm*self.mass)
                print('Best Epsilon - ',e_coll)         