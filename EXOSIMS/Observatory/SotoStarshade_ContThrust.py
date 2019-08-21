from EXOSIMS.Observatory.SotoStarshade import SotoStarshade
import numpy as np
import astropy.units as u
from scipy.integrate import solve_ivp
import astropy.constants as const
import hashlib
import scipy.optimize as optimize
from scipy.optimize import basinhopping
import scipy.interpolate as interp
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
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
        
# =============================================================================
# Miscellaneous        
# =============================================================================

    def unitVector(self,p):
        """ returns unit vector of p with same dimensions (3xn)
        """
        
        p_ = p/np.linalg.norm(p,axis=0)
        return p_
    

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
        
    def determineThrottle(self,s):
        """ Determines throttle based on instantaneous switching function value
        """
        
        eps = self.epsilon
        x,y,z,dx,dy,dz,m,L1,L2,L3,L4,L5,L6,L7 = s
        Lv_, lv = self.normalizeVector( np.array([L4,L5,L6]) )
        
        S = -lv*self.ve/m - L7 + 1
        
        if eps > 0:
            midthrottle = (eps - S)/(2*eps)
            throttle = 0 if S > eps else 1 if S < -eps else midthrottle
        else:
            throttle = 0 if S > eps else 1
        
        return throttle

# =============================================================================
# Equations of Motion and Boundary Conditions
# =============================================================================
        
    def boundary_conditions_UT(self,sA,sB):
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
        
        BC = np.array([BCo1,BCo2,BCo3,BCo4,BCo5,BCo6,BCf1,BCf2,BCf3,BCf4,BCf5,BCf6])
        
        return BC    
    
    def boundary_conditions_CT(self,sA,sB):
        """ Creates boundary conditions for solving a boundary value problem
        """
    
        BCo1 = sA[0] - self.sA[0]
        BCo2 = sA[1] - self.sA[1]
        BCo3 = sA[2] - self.sA[2]
        BCo4 = sA[3] - self.sA[3]
        BCo5 = sA[4] - self.sA[4]
        BCo6 = sA[5] - self.sA[5]
        BCo7 = sA[6] - self.sA[6]
        
        BCf1 = sB[0] - self.sB[0]
        BCf2 = sB[1] - self.sB[1]
        BCf3 = sB[2] - self.sB[2]
        BCf4 = sB[3] - self.sB[3]
        BCf5 = sB[4] - self.sB[4]
        BCf6 = sB[5] - self.sB[5]
        BCf7 = sB[6] - self.sB[6]
        
        BC = np.array([BCo1,BCo2,BCo3,BCo4,BCo5,BCo6,BCo7,BCf1,BCf2,BCf3,BCf4,BCf5,BCf6,BCf7])
        
        return BC   


    def EoM_Adjoint_UT(self,t,state):
        """ Equations of Motion with costate vectors
        """
        
        mu = self.mu
        x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6 = state
        f = np.zeros(state.shape)
        
        f[0,:]   = dx
        f[1,:]   = dy
        f[2,:]   = dz
        f[3,:]   = -L4 + 2*dy + mu*(-mu - x + 1)/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) + x + (-mu + 1)*(-mu - x)/(y**2 + z**2 + (mu + x)**2)**(3/2.)
        f[4,:]   = -L5 - 2*dx - mu*y/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) - y*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.) + y
        f[5,:]   = -L6 - mu*z/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) - z*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.)
        
        f[6,:]    = -L4*(mu*(-3*mu - 3*x + 3)*(-mu - x + 1)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - mu/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) + (-3*mu - 3*x)*(-mu + 1)*(-mu - x)/(y**2 + z**2 + (mu + x)**2)**(5/2.) - (-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.) + 1) - L5*(-mu*y*(-3*mu - 3*x + 3)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - y*(-3*mu - 3*x)*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.)) - L6*(-mu*z*(-3*mu - 3*x + 3)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - z*(-3*mu - 3*x)*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.))
        f[7,:]    = -L4*(-3*mu*y*(-mu - x + 1)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - 3*y*(-mu + 1)*(-mu - x)/(y**2 + z**2 + (mu + x)**2)**(5/2.)) - L5*(3*mu*y**2/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - mu/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) + 3*y**2*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.) - (-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.) + 1) - L6*(3*mu*y*z/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) + 3*y*z*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.))
        f[8,:]    = -L4*(-3*mu*z*(-mu - x + 1)/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - 3*z*(-mu + 1)*(-mu - x)/(y**2 + z**2 + (mu + x)**2)**(5/2.)) - L5*(3*mu*y*z/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) + 3*y*z*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.)) - L6*(3*mu*z**2/(y**2 + z**2 + (-mu - x + 1)**2)**(5/2.) - mu/(y**2 + z**2 + (-mu - x + 1)**2)**(3/2.) + 3*z**2*(-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(5/2.) - (-mu + 1)/(y**2 + z**2 + (mu + x)**2)**(3/2.))
        f[9,:]    = -L1 + 2*L5
        f[10,:]   = -L2 - 2*L4
        f[11,:]   = -L3
        
        return f


    def send_it_UT(self,fs0,fsF,t0,dt,maxNodes=1e5,verbose=False):
        """ Solving generic bvp from t0 to tF using states and costates
        """
        
        x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6 = fs0
        self.sA = np.array([x,y,z,dx,dy,dz])

        x,y,z,dx,dy,dz,L1,L2,L3,L4,L5,L6 = fsF
        self.sB = np.array([x,y,z,dx,dy,dz])

        t = np.linspace(t0,t0+dt,2)
        
        sG = np.vstack([ fs0 , fsF ])
        self.sG = sG
        sol = solve_bvp(self.EoM_Adjoint_UT,self.boundary_conditions_UT,t,sG.T,tol=1e-8,max_nodes=int(maxNodes),verbose=0)
        
        if verbose:
            self.vprint(sol.message)
        
        s = sol.y
        t_s = sol.x
            
        return s,t_s,sol.status
    

    def findInitialTmax(self,TL,nA,nB,tA,dt):
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
        s,t_s,status = self.send_it_UT(self_fsA,self_fsB,a,b-a,verbose=True)
        
        lv = s[9:,:]
        aNorms0 = np.linalg.norm(lv,axis=0)
        aMax0   = self.convertAcc_to_dim( np.max(aNorms0) ).to('m/s^2') 
        Tmax0   = (aMax0 * self.mass ).to('N')
        
        return Tmax0