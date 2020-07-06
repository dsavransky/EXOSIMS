from EXOSIMS.Observatory.SotoStarshade_ContThrust import SotoStarshade_ContThrust
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


class SotoStarshade_SKi(SotoStarshade_ContThrust):
    """ StarShade Observatory class
    This class is implemented at L2 and contains all variables, functions, 
    and integrators to calculate occulter dynamics. 
    """
    
    def __init__(self,latDist=0.9,latDistOuter=0.95,latDistFull=1,axlDist=250,**specs): 

        SotoStarshade_ContThrust.__init__(self,**specs)  
        
        self.latDist      = latDist * u.m
        self.latDistOuter = latDistOuter * u.m
        self.latDistFull  = latDistFull * u.m
        self.axlDist      = axlDist * u.km

    # converting angular velocity
    def convertAngVel_to_canonical(self,angvel):
        """ Convert velocity to canonical units
        """
        angvel = angvel.to('rad/yr')
        return angvel.value / (2*np.pi)

    def convertAngVel_to_dim(self,angvel):
        """ Convert velocity to canonical units
        """
        angvel = angvel * (2*np.pi)
        return angvel * u.rad / u.yr
    
    def convertAngAcc_to_canonical(self,angacc):
        """ Convert velocity to canonical units
        """
        angacc = angacc.to('rad/yr^2')
        return angacc.value / (2*np.pi)**2

    def convertAngAcc_to_dim(self,angacc):
        """ Convert velocity to canonical units
        """
        angacc = angacc * (2*np.pi)**2
        return angacc * u.rad / u.yr**2

# =============================================================================
# Kinematics
# =============================================================================

    def EulerAngleAndDerivatives(self,TL,sInd,currentTime,tRange=np.array([0])):
        
        # ecliptic coordinates and parallax of stars
        coords = TL.coords.transform_to('barycentrictrueecliptic')
        lamb = coords.lon[sInd]
        beta = coords.lat[sInd]
        varpi = TL.parx[sInd].to('rad')
        varpiValue = varpi.value   
        
        # absolute times (Note: equinox is start time of Halo AND when inertial frame and rotating frame match)
        absTimes = currentTime + tRange      #mission times  in jd
        modTimes = np.mod(absTimes.value,self.equinox.value)*u.d  #mission times relative to equinox )
        t = self.convertTime_to_canonical(modTimes) * u.rad       #modTimes in canonical units 
        
        # halo kinematics in rotating frame relative to Origin of R-frame (in au)
        haloPos = self.haloPosition(absTimes) + np.array([1,0,0])*self.L2_dist.to('au')
        haloVel = self.haloVelocity(absTimes)
        
        # halo positions and velocities in canonical units
        xTR,   yTR,  zTR = np.array([self.convertPos_to_canonical(haloPos[:,n]) for n in range(3)])
        dxTR, dyTR, dzTR = np.array([self.convertVel_to_canonical(haloVel[:,n]) for n in range(3)])
        
        # converting halo to inertial frame coordinates AND derivatives
        xTI = xTR*np.cos(t) - yTR*np.sin(t)
        yTI = xTR*np.sin(t) + yTR*np.cos(t)
        zTI = zTR
        IdxTI = dxTR*np.cos(t) - dyTR*np.sin(t) - yTI
        IdyTI = dxTR*np.sin(t) + dyTR*np.cos(t) + xTI
        IdzTI = dzTR
        
        x_comp = np.cos(beta)*np.cos(lamb) - varpiValue*xTI 
        y_comp = np.cos(beta)*np.sin(lamb) - varpiValue*yTI 
        z_comp = np.sin(beta) - varpiValue*zTI
        r_comp = np.sqrt( x_comp**2 + y_comp**2 )
        
        theta = np.arctan2( y_comp , x_comp )
        phi   = np.arctan2( r_comp , z_comp)
        
        
        dtheta = varpiValue * (-IdxTI*np.sin(theta) + IdyTI*np.cos(theta))
        dphi   = varpiValue * (np.cos(phi) * (IdxTI*np.cos(theta) + IdyTI*np.sin(theta)) + IdzTI ) / np.sin(phi)
        
        return theta.value, phi.value, dtheta, dphi

    def Bframe(self,TL,sInd,currentTime,tRange=np.array([0])):
        
        theta,phi,dtheta,dphi = self.EulerAngleAndDerivatives(TL,sInd,currentTime,tRange)
        
        b1_I = np.array([np.cos(phi)*np.cos(theta),\
                       np.cos(phi)*np.sin(theta),\
                      -np.sin(phi)])

        b2_I = np.array([-np.sin(theta),\
                       np.cos(theta),\
                       np.zeros(len(theta))])
        
        b3_I = np.array([np.sin(phi)*np.cos(theta),\
                       np.sin(phi)*np.sin(theta),\
                       np.cos(phi)])
            
        return b1_I, b2_I, b3_I
        
    
    def starshadeKinematics(self,TL,sInd,currentTime,tRange=np.array([0])):
        
        
        varpi = TL.parx[sInd].to('rad')
        varpiValue = varpi.value 
        
        # absolute times (Note: equinox is start time of Halo AND when inertial frame and rotating frame match)
        absTimes = currentTime + tRange      #mission times  in jd
        modTimes = np.mod(absTimes.value,self.equinox.value)*u.d  #mission times relative to equinox )
        t = self.convertTime_to_canonical(modTimes) * u.rad       #modTimes in canonical units 
        
        s = self.convertPos_to_canonical( self.occulterSep )
        
        # halo kinematics in rotating frame relative to Origin of R-frame (in au)
        haloPos = self.haloPosition(absTimes) + np.array([1,0,0])*self.L2_dist.to('au')
        haloVel = self.haloVelocity(absTimes)
        
        # halo positions and velocities in canonical units
        xTR,   yTR,  zTR = np.array([self.convertPos_to_canonical(haloPos[:,n]) for n in range(3)])
        dxTR, dyTR, dzTR = np.array([self.convertVel_to_canonical(haloVel[:,n]) for n in range(3)])

        xTI = xTR*np.cos(t) - yTR*np.sin(t)
        yTI = xTR*np.sin(t) + yTR*np.cos(t)
        zTI = zTR
        dxTI = dxTR*np.cos(t) - dyTR*np.sin(t) - yTI
        dyTI = dxTR*np.sin(t) + dyTR*np.cos(t) + xTI
        dzTI = dzTR
        
        rTI = np.vstack([xTI,   yTI,  zTI, dxTI, dyTI, dzTI])
        ddxTI,ddyTI,ddzTI = self.equationsOfMotion_CRTBPInertial(t.value,rTI,TL,sInd)[3:6,:]

        # Euler angles
        theta,phi,dtheta,dphi = self.EulerAngleAndDerivatives(TL,sInd,currentTime,tRange)
        
        ddtheta = varpiValue * (ddyTI*np.cos(theta) - ddxTI*np.sin(theta))
        ddphi =( varpiValue / np.sin(phi)) * (np.cos(phi)*(ddxTI*np.cos(theta) + ddyTI*np.sin(theta) ) + ddzTI  )

        
        # starshade positions
        r_ST_I = np.array([ [ s*np.sin(phi)*np.cos(theta) ],
                            [ s*np.sin(phi)*np.sin(theta) ],
                            [ s*np.cos(phi)] ])[:,0,:]
        
        Iv_ST_I = np.array([ [ s*(dphi*np.cos(phi)*np.cos(theta) - dtheta*np.sin(phi)*np.sin(theta) ) ],
                             [ s*(dphi*np.cos(phi)*np.sin(theta) - dtheta*np.sin(phi)*np.cos(theta) ) ],
                             [-s*dphi*np.sin(phi)] ])[:,0,:]

        Ia_ST_I = np.array([ [ s*(ddphi*np.cos(phi)*np.cos(theta) - ddtheta*np.sin(phi)*np.sin(theta) ) ],
                             [ s*(ddphi*np.cos(phi)*np.sin(theta) - ddtheta*np.sin(phi)*np.cos(theta) ) ],
                             [-s*ddphi*np.sin(phi)] ])[:,0,:]
        
        r_S0_I = np.array([ [ xTI + r_ST_I[0,:] ],
                            [ yTI + r_ST_I[1,:] ],
                            [ zTI + r_ST_I[2,:]] ])[:,0,:]
        
        Iv_S0_I = np.array([ [ dxTI + Iv_ST_I[0,:] ],
                             [ dyTI + Iv_ST_I[1,:] ],
                             [ dzTI + Iv_ST_I[2,:]] ])[:,0,:]

        Ia_S0_I = np.array([ [ ddxTI + Ia_ST_I[0,:] ],
                             [ ddyTI + Ia_ST_I[1,:] ],
                             [ ddzTI + Ia_ST_I[2,:]] ])[:,0,:]
        
        return r_S0_I , Iv_S0_I, Ia_S0_I,r_ST_I , Iv_ST_I, Ia_ST_I
    
    
# =============================================================================
# Ideal Dynamics
# =============================================================================

    def starshadeIdealDynamics(self,TL,sInd,currentTime,tRange=np.array([0]),SRP=False,Moon=False):
        
        # absolute times (Note: equinox is start time of Halo AND when inertial frame and rotating frame match)
        absTimes = currentTime + tRange      #mission times  in jd
        modTimes = np.mod(absTimes.value,self.equinox.value)*u.d  #mission times relative to equinox 
        t = self.convertTime_to_canonical(modTimes) * u.rad       #modTimes in canonical units 
        
        # B-frame unit vectors
        b1, b2, b3 = self.Bframe(TL,sInd,currentTime,tRange)
        
        # full starshade kinematics
        r_S0_I , Iv_S0_I, Ia_S0_I, r_ST_I , Iv_ST_I, Ia_ST_I = self.starshadeKinematics(TL,sInd,currentTime,tRange)
        
        # full state of starshade (pos + vel)
        s0 = np.vstack([r_S0_I,Iv_S0_I])
        
        # gravitational forces on the starshade at nominal position
        f_S0_I = self.equationsOfMotion_CRTBPInertial(t.value,s0,TL,sInd,integrate=False,SRP=SRP,Moon=Moon)[3:6,:]

        # differential force and components (axial and lateral)
        df_S0_I = f_S0_I - Ia_S0_I
        dfA  = np.array([np.matmul(a,b) for a,b in zip(df_S0_I.T,b3.T)])
        dfL_I  = df_S0_I - dfA * b3
        
        # components of lateral differential force on the b1-b2 plane
        dfL_b1  = np.array([np.matmul(a,b) for a,b in zip(dfL_I.T,b1.T)])
        dfL_b2  = np.array([np.matmul(a,b) for a,b in zip(dfL_I.T,b2.T)])
        
        # roll angle so that df_L points in negative 2nd axis in new frame
        psi = np.arctan2( dfL_b1 , -dfL_b2 )
            
        return psi, dfL_I, dfA, df_S0_I, f_S0_I


    def rotateComponents2NewFrame(self,TL,sInd,trajStartTime,s_int,t_int,final_frame='C',SRP=False, Moon=False):
        """
        
        Args:
            s_int (6xn):
                Integrated states in canonical units
            t_int (n):
                Integrated times in canonical units


        """
        # assume that t_int is relative to the given trajStartTime or currentTime
        tRange = self.convertTime_to_dim(t_int)
        # Euler angles
        theta, phi, dtheta, dphi = self.EulerAngleAndDerivatives(TL,sInd,trajStartTime,tRange)
        psi, dfL_I, df_A, df_S0_I, f_S0_I = self.starshadeIdealDynamics(TL,sInd,trajStartTime,tRange,SRP=SRP,Moon=Moon)
        
        # extracting initial states
        x,y,z,dx,dy,dz = s_int
        r_i  = np.vstack([x,y,z])
        Iv_i = np.vstack([dx,dy,dz])
        
        # defining final states
        r_f = np.zeros(r_i.shape)
        Iv_f = np.zeros(Iv_i.shape)
        
        for n in range(len(t_int)):
            AcI = self.rot(theta[n],3)
            BcA = self.rot(phi[n],2)
            CcB = self.rot(psi[n],3)
            
            CcI = np.matmul(  CcB, np.matmul(BcA,AcI)   )
            
            FcI = CcI if final_frame == 'C' else CcI.T
        
            r_f[:,n]  = np.matmul( FcI, r_i[:,n]  )
            Iv_f[:,n] = np.matmul( FcI, Iv_i[:,n]  )
    
        return r_f , Iv_f
# =============================================================================
# Equations of Motion
# =============================================================================

    def equationsOfMotion_CRTBPInertial(self,t,state,TL,sInd,integrate=False,SRP=False, Moon=False):
        """Equations of motion of the CRTBP with Solar Radiation Pressure
        
        Equations of motion for the Circular Restricted Three Body 
        Problem (CRTBP). First order form of the equations for integration, 
        returns 3 velocities and 3 accelerations in (x,y,z) rotating frame.
        All parameters are normalized so that time = 2*pi sidereal year.
        Distances are normalized to 1AU. Coordinates are taken in a rotating 
        frame centered at the center of mass of the two primary bodies. Pitch
        angle of the starshade with respect to the Sun is assumed to be 60 
        degrees, meaning the 1/2 of the starshade cross sectional area is 
        always facing the Sun on average
        
        Args:
            t (float):
                Times in normalized units
            state (float 6xn array):
                State vector consisting of stacked position and velocity vectors
                in normalized units

        Returns:
            ds (integer Quantity 6xn array):
                First derivative of the state vector consisting of stacked 
                velocity and acceleration vectors in normalized units
        """
        
        mu = self.mu
        
        x,y,z,dx,dy,dz = state
        r_P0_I  = np.vstack([x,y,z])   
        Iv_P0_I = np.vstack([dx,dy,dz])  
        
        # positions of the Earth and Sun
        try:
            len(t)
        except:
            t = np.array([t])

        r_10_I = -mu    * np.array([ [np.cos(t)], [np.sin(t)], [np.zeros(len(t))] ])[:,0,:] 
        r_20_I = (1-mu) * np.array([ [np.cos(t)], [np.sin(t)], [np.zeros(len(t))] ])[:,0,:] 


        # relative positions of P
        r_P1_I = r_P0_I - r_10_I
        r_P2_I = r_P0_I - r_20_I
        
        d_P1_I = np.linalg.norm(r_P1_I,axis=0)
        d_P2_I = np.linalg.norm(r_P2_I,axis=0)
        
        # equations of motion
        Ia_P0_I = -(1-mu) * r_P1_I/d_P1_I**3 - mu * r_P2_I/d_P2_I**3
        
        if SRP:
            modTimes = self.convertTime_to_dim(t).to('d')
            absTimes = self.equinox + modTimes
            tRange = absTimes - absTimes[0] if len(t) > 0 else [0]
            fSRP = self.SRPforce(TL,sInd,absTimes[0],tRange)
            Ia_P0_I  += fSRP
            
        if Moon:
            modTimes = self.convertTime_to_dim(t).to('d')
            absTimes = self.equinox + modTimes
            tRange = absTimes - absTimes[0] if len(t) > 0 else [0]
            fMoon = self.lunarPerturbation(TL,sInd,absTimes[0],tRange)
            # import pdb
            # pdb.set_trace()
            Ia_P0_I  += fMoon.value
        
        # full equations
        f_P0_I = np.vstack([ Iv_P0_I, Ia_P0_I ])
        
        if integrate:
            f_P0_I = f_P0_I.flatten()

        return f_P0_I
    
    def SRPforce(self,TL,sInd,currentTime,tRange,radius=36):
        """Equations of motion of the CRTBP with Solar Radiation Pressure

        """
        
        mu = self.mu
        
        # absolute times (Note: equinox is start time of Halo AND when inertial frame and rotating frame match)
        absTimes = currentTime + tRange      #mission times  in jd
        modTimes = np.mod(absTimes.value,self.equinox.value)*u.d  #mission times relative to equinox )
        t = self.convertTime_to_canonical(modTimes) * u.rad       #modTimes in canonical units 
        
        b1, b2, b3 = self.Bframe(TL,sInd,currentTime,tRange)
        r_S0_I , Iv_S0_I, Ia_S0_I, r_ST_I , Iv_ST_I, Ia_ST_I = self.starshadeKinematics(TL,sInd,currentTime,tRange)

        # positions of the Earth and Sun
        r_10_I = -mu  * np.array([ [np.cos(t)], [np.sin(t)], [np.zeros(len(t))] ])[:,0,:]

        # relative positions of P
        r_S1_I = r_S0_I - r_10_I
        u_S1_I , d_S1 = self.unitVector(r_S1_I)
        
        cosA = np.array([np.matmul(a,b) for a,b in zip(u_S1_I.T,b3.T)])
        
        R = radius * u.m
        A = np.pi*R**2.     #starshade cross-sectional area
        PA = self.convertAcc_to_canonical(   (4.473*u.uN/u.m**2.) * A / self.scMass )
        Bf = 0.038                  #non-Lambertian coefficient (front)
        Bb = 0.004                  #non-Lambertian coefficient (back)
        s  = 0.975                  #specular reflection factor
        p  = 0.999                  #nreflection coefficient
        ef = 0.8                    #emission coefficient (front)
        eb = 0.2                    #emission coefficient (back)
        
        # optical coefficients
        a1 = 0.5*(1.-s*p)
        a2 = s*p
        a3 = 0.5*(Bf*(1.-s)*p + (1.-p)*(ef*Bf - eb*Bb) / (ef + eb) ) 
        
        f_SRP = 2*PA * cosA * ( a1 *  u_S1_I +  (a2 * cosA + a3)*b3  )
        
        return f_SRP

    def lunarPerturbation(self,TL,sInd,currentTime,tRange):
        
        mu = self.mu

        # absolute times (Note: equinox is start time of Halo AND when inertial frame and rotating frame match)
        absTimes = currentTime + tRange      #mission times  in jd
        modTimes = np.mod(absTimes.value,self.equinox.value)*u.d  #mission times relative to equinox )
        t = self.convertTime_to_canonical(modTimes) * u.rad       #modTimes in canonical units 
        
        b1, b2, b3 = self.Bframe(TL,sInd,currentTime,tRange)
        r_S0_I , Iv_S0_I, Ia_S0_I, r_ST_I , Iv_ST_I, Ia_ST_I = self.starshadeKinematics(TL,sInd,currentTime,tRange)
        
        # Moon
        mM = ( (7.342e22*u.kg) / (const.M_earth + const.M_sun) ).to('')
        aM = 384748*u.km
        aM = self.convertPos_to_canonical(aM)
        iM = 5.15*u.deg
        TM = 29.53*u.d
        wM = 2*np.pi/self.convertTime_to_canonical(TM)
        
        # positions of the Earth and Sun
        r_20_I = (1-self.mu) * np.array([ [np.cos(t)], [np.sin(t)], [np.zeros(len(t))] ])[:,0,:]
        
        r_32_I = -aM * np.array([ [np.cos(wM*t)], [np.sin(wM*t)*np.cos(iM)], [np.sin(wM*t)*np.sin(iM)] ])[:,0,:] 
        r_30_I = r_32_I + r_20_I
        
        # relative positions of P
        r_S3_I = r_S0_I - r_30_I
        u_S3_I , d_S3 = self.unitVector(r_S3_I)
        
        f_Moon = -mM * r_S3_I / d_S3**3
        
        return f_Moon
    

    def equationsOfMotion_aboutS(self,t,state,TL,sInd,trajStartTime,integrate=False,SRP=False, Moon=False):
        
        # original state of S' wrt to S in R-frame components and derivatives
        x,y,z,dx,dy,dz = state
        
        r_OS_I  = state[0:3]
        Iv_OS_I = state[3:6]
        
        # reference epoch in canonical units
        t0 = self.convertTime_to_canonical(np.mod(trajStartTime.value,self.equinox.value)*u.d)[0]
        tF = t0 + t
    
        # current time in dimensional units
        currentTime = trajStartTime + self.convertTime_to_dim(t)
        
        # retrieving acceleration of point S rel O at currentTime
        r_S0_I, Iv_S0_I, Ia_S0_I, r_ST_I , Iv_ST_I, Ia_ST_I = self.starshadeKinematics(TL,sInd,currentTime)
        
        # determining position and velocity of S' rel O at currentTime
        r_O0_I   = r_S0_I[:,0] + r_OS_I
        Iv_O0_I = Iv_S0_I[:,0] + Iv_OS_I
        s0 = np.hstack([r_O0_I,Iv_O0_I])
        
        # calculating force on S' at currentTime
        f_O0_R = self.equationsOfMotion_CRTBPInertial(tF,s0,TL,sInd,integrate=True,SRP=SRP,Moon=Moon)[3:6]

        # setting final second derivatives and stuff
        dr  = [dx,dy,dz]
        ddr = f_O0_R - Ia_S0_I.flatten()
        ds = np.vstack([dr,ddr])
        ds = ds.flatten()
            
        return ds


# =============================================================================
# Playing Tennis
# =============================================================================

    def crossThreshholdEvent(self,t,s,TL,sInd,trajStartTime,latDist,SRP=False, Moon=False):
        
        currentTime = trajStartTime + self.convertTime_to_dim(t)
        # converting state vectors from R-frame to C-frame
        r_OS_C,Iv_OS_C = self.rotateComponents2NewFrame(TL,sInd,currentTime,s,np.array([0]),final_frame='C',SRP=SRP,Moon=Moon)
        # converting units to meters
        r_OS_C_dim = self.convertPos_to_dim(r_OS_C).to('m')
        
        # distance from S (in units of m)
        delta_r = r_OS_C_dim[0:2,:]
        dr = np.linalg.norm(delta_r,axis=0)
        
        # distance from inner threshhold
        distanceFromLim = (dr - latDist).value
        # print(latDist,distanceFromLim)
        return distanceFromLim


    def drift(self,TL,sInd,trajStartTime,dt=20*u.min,freshStart=True,s0=None,fullSol=False,SRP=False, Moon=False):
        """
        
        return:
            cross - 0,1,or 2 for Tolerance Not Crossed, Lateral Cross, and Axial Cross
        """
        
        # defining equations of motion
        EoM = lambda t,s: self.equationsOfMotion_aboutS(t,s,TL,sInd,trajStartTime,integrate=True,SRP=SRP,Moon=Moon)
        
        # event function for crossing inner boundary
        crossInner = lambda t,s: self.crossThreshholdEvent(t,s,TL,sInd,trajStartTime,self.latDist)
        crossInner.terminal  = False
        crossInner.direction = 1
        
        # event function for crossing outer boundary
        crossOuter = lambda t,s: self.crossThreshholdEvent(t,s,TL,sInd,trajStartTime,self.latDistOuter)
        crossOuter.terminal  = True # this one is terminal, should end integration after trigger
        crossOuter.direction = 1
        
        # defining times
        t0 = self.convertTime_to_canonical(np.mod(trajStartTime.value,self.equinox.value)*u.d)[0]
        tF = t0 + self.convertTime_to_canonical( dt )
        tInt = np.linspace(0,tF-t0,5000)
        
        # remember these states are relative to the nominal track S/0
        # either start exactly on nominal track or else place it yourself
        if freshStart:
            r0_C, v0_C = self.starshadeInjectionVelocity(TL,sInd,trajStartTime,SRP)
            s0_C = np.hstack([r0_C, v0_C])
            # rotate to inertial frame
            r0_I, v0_I = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,s0_C,np.array([0]),SRP=SRP,Moon=Moon,final_frame='I')
            s0 = np.hstack([r0_I.flatten(), v0_I.flatten()])
            
        # integrate forward to t0 + dt
        res = solve_ivp(EoM,[tInt[0],tInt[-1]],s0,t_eval=tInt,events=(crossInner,crossOuter,),rtol=1e-13,atol=1e-13,method='Radau')
        t_int = res.t
        y_int = res.y
        x,y,z,dx,dy,dz = y_int
        
        #gotta rotate trajectory array from R-frame to C-frame components
        r_OS_C,Iv_OS_C = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,y_int,t_int,SRP=SRP,Moon=Moon,final_frame='C')
        
        #there should be 2 events, even if they weren't triggered (they'd be empty)
        t_innerEvent = res.t_events[0]
        t_outerEvent = res.t_events[1]
        s_innerEvent = res.y_events[0][-1] if t_innerEvent.size > 0 else np.zeros(6)
        s_outerEvent = res.y_events[1][0]  if t_outerEvent.size > 0 else np.zeros(6)
        r_C_outer = -np.ones(3)
        if t_outerEvent.size > 0:
            r_C_outer,v_C_outer = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,s_outerEvent,t_outerEvent,SRP=SRP,Moon=Moon,final_frame='C')
        
        runItAgain = False
        #outer event triggered 
        if t_outerEvent.size > 0:
            #outer event triggered in positive y -> we need to burn!
            if r_C_outer[1] > 0:
                print('crossed outer')
                cross = 2
                driftTime = self.convertTime_to_dim(t_outerEvent[0]).to('min')
                # rotate states at crossing event to C frame
                r_cross,v_cross = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,s_outerEvent,t_outerEvent,SRP=SRP,Moon=Moon,final_frame='C')
                # import pdb
                # pdb.set_trace()
                if fullSol:
                    tEndInd = np.where( t_int <= t_outerEvent )[0][-1]
                    t_input = t_int[0:tEndInd]
                    s_input = y_int[:,0:tEndInd]
                    s_interp = interp.interp1d( t_input,s_input )
                    
                    t_full = np.linspace(t_input[0] , t_input[-1], len(tInt) )
                    s_interped = s_interp( t_full )
                    r_full,v_full = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,s_interped,t_full,SRP=SRP,Moon=Moon,final_frame='C')
            
            #outer event triggered in negative y -> let's see what happens....
            else:
                #if we trigger in negative y, we should then just resolve the inner boundary crossing instead
                #inner event triggered AND => outer event triggered only in negative y 
                # import pdb
                # pdb.set_trace()
                if t_innerEvent.size > 0:
                    if np.any(t_innerEvent) > 0:
                         print('crossed inner')
                         cross = 1
                         driftTime = self.convertTime_to_dim(t_innerEvent[-1]).to('min')
                         # rotate states at crossing event to C frame
                         t_inner = t_innerEvent if t_innerEvent.size == 0 else np.array([t_innerEvent[-1]])
                         r_cross,v_cross = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,s_innerEvent,t_inner,SRP=SRP,Moon=Moon,final_frame='C')
                         # import pdb
                         # pdb.set_trace()
                         if fullSol:
                             tEndInd = np.where( t_int <= t_inner )[0][-1]
                             t_input = t_int[0:tEndInd]
                             s_input = y_int[:,0:tEndInd]
                             s_interp = interp.interp1d( t_input,s_input )
                             
                             t_full = np.linspace(t_input[0] , t_input[-1], len(tInt) )
                             s_interped = s_interp( t_full )
                             r_full,v_full = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,s_interped,t_full,SRP=SRP,Moon=Moon,final_frame='C')
                    else:
                        print('ERROR: crossed outer but not inner crossing broke!!')
                        # let's check if the inner boundary was triggered badly by the integrator
                        ds = np.linalg.norm( r_OS_C[0:2,:],axis=0 )
                        ds_dim =self.convertPos_to_dim(ds).to('m')
            
                        if np.any(ds_dim[1:] > self.latDist):
                            print('something DID break')
                            outsideInnerRadius = np.where( ds_dim > self.latDist )[0]
                            
                            if outsideInnerRadius.size > 0:
                                print('crossed inner - fixed')
                                jumpsInInnerRadius = np.diff(outsideInnerRadius)
                                lastInnerCrossing = outsideInnerRadius[int(np.where( jumpsInInnerRadius > 1)[0][-1] + 1)] 
                                # we got em
                                cross = 1
                                y_int = y_int[:,0:lastInnerCrossing]
                                t_int = t_int[0:lastInnerCrossing]
                                t_inner = np.array([t_int[-1]])
                                driftTime = self.convertTime_to_dim(t_int[-1]).to('min')
                                # rotate states at crossing event to C frame
                                r_cross,v_cross = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,y_int[:,-1],t_inner,SRP=SRP,Moon=Moon,final_frame='C')
                                
                                if fullSol:
                                     s_interp = interp.interp1d( t_int,y_int )
                                     
                                     t_full = np.linspace(t_int[0] , t_int[-1], len(tInt) )
                                     s_interped = s_interp( t_full )
                                     r_full,v_full = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,s_interped,t_full,SRP=SRP,Moon=Moon,final_frame='C')
            
                            else:
                                runItAgain = True
                            
                        else:
                            runItAgain = True          
                #something messed up with the event function!
                #we triggered outer function in the negative y direction BUT somehow didnt trigger the inner boundary?!?!
                #I swear this happens.
                else:
                    print('ERROR: crossed outer but not inner!!')
                    # let's check if the inner boundary was supposed to be triggered but wasn't caught by the integrator
                    ds = np.linalg.norm( r_OS_C[0:2,:],axis=0 )
                    ds_dim =self.convertPos_to_dim(ds).to('m')
        
                    if np.any(ds_dim[1:] > self.latDist):
                        print('something DID break')
                        outsideInnerRadius = np.where( ds_dim > self.latDist )[0]
                        
                        if outsideInnerRadius.size > 0:
                            print('crossed inner - fixed')
                            jumpsInInnerRadius = np.diff(outsideInnerRadius)
                            lastInnerCrossing = outsideInnerRadius[int(np.where( jumpsInInnerRadius > 1)[0][-1] + 1)] 
                            # we got em
                            cross = 1
                            y_int = y_int[:,0:lastInnerCrossing]
                            t_int = t_int[0:lastInnerCrossing]
                            t_inner = np.array([t_int[-1]])
                            driftTime = self.convertTime_to_dim(t_int[-1]).to('min')
                            # rotate states at crossing event to C frame
                            r_cross,v_cross = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,y_int[:,-1],t_inner,SRP=SRP,Moon=Moon,final_frame='C')
                            
                            if fullSol:
                                 s_interp = interp.interp1d( t_int,y_int )
                                 
                                 t_full = np.linspace(t_int[0] , t_int[-1], len(tInt) )
                                 s_interped = s_interp( t_full )
                                 r_full,v_full = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,s_interped,t_full,SRP=SRP,Moon=Moon,final_frame='C')
        
                        else:
                            runItAgain = True
                        
                    else:
                        runItAgain = True
        
        #outer event not triggered at all 
        else:
            #BUT inner event WAS triggered.
            if t_innerEvent.size > 0:
                 print('crossed inner')
                 cross = 1
                 driftTime = self.convertTime_to_dim(t_innerEvent[-1]).to('min')
                 # rotate states at crossing event to C frame
                 t_inner = t_innerEvent if t_innerEvent.size == 0 else np.array([t_innerEvent[-1]])
                 r_cross,v_cross = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,s_innerEvent,t_inner,SRP=SRP,Moon=Moon,final_frame='C')
                 
                 if fullSol:
                     tEndInd = np.where( t_int <= t_inner )[0][-1]
                     t_input = t_int[0:tEndInd]
                     s_input = y_int[:,0:tEndInd]
                     s_interp = interp.interp1d( t_input,s_input )
                     
                     t_full = np.linspace(t_input[0] , t_input[-1], len(tInt) )
                     s_interped = s_interp( t_full )
                     r_full,v_full = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,s_interped,t_full,SRP=SRP,Moon=Moon,final_frame='C')
                    
            #no events were triggered, need to run drift for longer
            else:
                runItAgain = True
        
        if runItAgain:
            print('crossed nothing')
            cross = 0
            driftTime = self.convertTime_to_dim(t_int[-1]).to('min')
            r_cross  = r_OS_C[:,-1]
            v_cross  = Iv_OS_C[:,-1]
        
            r_full = r_OS_C
            v_full = Iv_OS_C
            t_full = t_int
            
        if fullSol:
            return cross, driftTime, t_full, r_full, v_full
        else:
            return cross, driftTime, t_int, r_cross, v_cross


    def guessAParabola(self,TL,sInd,trajStartTime,r_OS_C,Iv_OS_C,latDist = 0.9*u.m,fullSol=False,SRP=False, Moon=False,burnHard=False):
        
        psi, dfL_I, df_A, df_S0_I, f_S0_I = self.starshadeIdealDynamics(TL,sInd,trajStartTime,SRP=SRP,Moon=Moon)
        
        a = np.linalg.norm(dfL_I)
        DU = self.convertPos_to_canonical(latDist)
        VU = np.sqrt( a*DU )
        TU = np.sqrt( DU/a )
        
        # converting initial state from canonical units to C-frame units
        x0,y0,z0       = r_OS_C[:,-1] / DU
        dx0_,dy0_,dz0_ = Iv_OS_C[:,-1] / VU
        
        # finding intercept point
        if y0 > 0.5:
            xi = x0
            yi = y0
        else:
            xi = np.sqrt(1+y0)/np.sqrt(2) if x0 > 0 else -np.sqrt(1+y0)/np.sqrt(2)
            yi = np.sqrt(1-y0)/np.sqrt(2)
                    
        # initial velocities of the second parabola
        dx0 = np.sqrt(yi*(1-yi))/np.sqrt(2) if x0 < 0 else -np.sqrt(yi*(1-yi))/np.sqrt(2)
        dy0 = -dx0*xi/yi + (xi-x0)/dx0 if dx0 != 0 else 0

        #tof and velocities if we're really close to the well
        if np.abs(x0) < 0.08:
            dy0 = (3*yi-1)*np.sqrt( (1+yi)/(2*yi) ) if y0 < 0.5 else (1-yi)*np.sqrt( (1+yi)/(2*yi) )
            
            ymax = dy0**2/2 + y0
            tof  = np.sqrt(2) * (np.sqrt(ymax-y0) + np.sqrt(ymax+1))
            dx0  = -x0/tof
        else:
            # time of flight in C-frame units
            tof = np.abs( x0 / dx0 ) 
                    
        # if burnHard:
        #     dy0 = -(3*yi-1)*np.sqrt( (1+yi)/(2*yi) )
        #     dx0 = -np.sign(x0) * (3*yi-1)*np.sqrt( (1+yi)/(2*yi) )
        #     tof = 4
                
        ###planar parabolic trajectories in C-frame units
        # time range from 0 to full time of flight
        t = np.linspace(0,tof,1000)
        
        # x and y throughout entire new parabolic trajectory
        xP  = dx0*t + x0
        yP  = -0.5*t**2 + dy0*t + y0
        
        r_PS_C        = np.vstack([xP,yP])  * DU
        Iv_PS_C_newIC = np.array([dx0,dy0,0]) * VU
        dt_newTOF     = tof * TU
        
        # calculate delta v
        sNew_C = np.hstack([r_OS_C[:,-1] , Iv_PS_C_newIC])
        r0new,v0new_I = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,sNew_C,np.array([0]),SRP=SRP,Moon=Moon,final_frame='I')
        
        sOld_C = np.hstack([r_OS_C[:,-1] , Iv_OS_C[:,-1]])
        r0old,v0old_I = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,sOld_C,np.array([0]),SRP=SRP,Moon=Moon,final_frame='I')
        dv = np.linalg.norm( v0new_I - v0old_I )
        dv_dim = self.convertVel_to_dim( dv ).to('m/s')
                
        if fullSol:
            return dt_newTOF, Iv_PS_C_newIC, dv_dim, r_PS_C
        else:
            return dt_newTOF, Iv_PS_C_newIC, dv_dim
        

    def starshadeInjectionVelocity(self,TL,sInd,trajStartTime,SRP=False,Moon=False):
        
        latDist   = self.latDist
        startingY = self.convertPos_to_canonical( latDist )
        r_OS_C  = np.array([0,-1,0]).reshape(3,1) * startingY
        Iv_OS_C = np.array([1,1,1]).reshape(3,1)  * startingY
        dt, Iv_OS_C_newIC, dv_dim = self.guessAParabola(TL,sInd,trajStartTime,r_OS_C,Iv_OS_C,latDist = latDist,fullSol=False,SRP=SRP,Moon=Moon)
        
        return r_OS_C.flatten(), Iv_OS_C_newIC


    def stationkeep(self,TL,sInd,trajStartTime,dt=30*u.min,simTime=1*u.hr,SRP=False, Moon=False):
        
        
        # drift for the first time, calculates correct injection speed within
        cross, driftTime, t_int, r_OS_C, Iv_OS_C = self.drift(TL,sInd,trajStartTime, dt = dt,freshStart=True,fullSol=True, SRP=SRP,Moon=Moon)
        
        # counter for re-do's
        reDo = 0
        # if we didn't cross any threshold, let's increase the drift time 
        if cross == 0:
            while cross == 0:
                print('redo!')
                # increase time
                dt *= 4
                cross, driftTime, t_int, r_OS_C, Iv_OS_C = self.drift(TL,sInd,trajStartTime, dt = dt,freshStart=True,fullSol=True, SRP=SRP,Moon=Moon)

                # augment counter
                reDo += 1
                # if we've gone through this 5 times, something is probably wrong...
                if reDo > 5:
                    break
        # import pdb
        # pdb.set_trace()
        # initiate arrays to log results and times        
        timeLeft = simTime - driftTime
        nBounces = 1
        dvLog      = np.array([]) 
        dvAxialLog = np.array([]) 
        driftLog   = np.array([driftTime.to('min').value])
        
        # running deadbanding simulation
        while timeLeft.to('min').value > 0:
            print(timeLeft.to('min'))
            trajStartTime += driftTime
            latDist = self.latDist if cross == 1 else self.latDistOuter if cross == 2 else 0

            dt_new, Iv_PS_C_newIC, dv_dim, r_PS_C = self.guessAParabola(TL,sInd,trajStartTime,r_OS_C,Iv_OS_C,latDist,fullSol=True,SRP=SRP,Moon=Moon)
            dt_newGuess = self.convertTime_to_dim(dt_new).to('min')  * 2
            
            s0_Cnew     = np.hstack([ r_OS_C[:,-1] , Iv_PS_C_newIC ])
            r0,v0 = self.rotateComponents2NewFrame(TL,sInd,trajStartTime,s0_Cnew,np.array([0]),SRP=SRP,Moon=Moon,final_frame='I')
            s0_new = np.hstack([r0.flatten(),v0.flatten()])
            
            cross, driftTime, t_int, r_OS_C, Iv_OS_C = self.drift(TL,sInd,trajStartTime, dt = dt_newGuess,freshStart=False,s0=s0_new,fullSol=True,SRP=SRP,Moon=Moon)
            
            reDo = 0
            if cross == 0:
                while cross == 0:
                    print('redo!')
                    dt_newGuess *= 4
                    cross, driftTime, t_int, r_OS_C, Iv_OS_C = self.drift(TL,sInd,trajStartTime, dt = dt_newGuess,\
                                                                         freshStart=False,s0=s0_new,fullSol=True,SRP=SRP,Moon=Moon)
                    
                    reDo += 1
                    if reDo > 5:
                        break
            
            #update everything
            nBounces += 1
            timeLeft -= driftTime
            dvLog    = np.hstack([dvLog,dv_dim.to('m/s').value])
            driftLog = np.hstack([driftLog,driftTime.to('min').value])
            dvAxialLog    = np.hstack([dvAxialLog,self.convertVel_to_dim(np.abs(Iv_OS_C[2,-1])).to('m/s').value])
        
        
        return nBounces, timeLeft, dvLog*u.m/u.s, dvAxialLog*u.m/u.s, driftLog*u.min
    
    
    def globalStationkeep(self,TL,trajStartTime,tau=0*u.d,dt=30*u.min,simTime=1*u.hr,SRP=False, Moon=False):
        
        # drift times
        tDriftMax_Log  = np.zeros(TL.nStars)*u.min
        tDriftMean_Log = np.zeros(TL.nStars)*u.min
        tDriftStd_Log  = np.zeros(TL.nStars)*u.min
        # full delta v's
        dvMean_Log  = np.zeros(TL.nStars)*u.m/u.s
        dvMax_Log = np.zeros(TL.nStars)*u.m/u.s
        dvStd_Log = np.zeros(TL.nStars)*u.m/u.s
        # axial delta v's (counteract axial drifts)
        dvAxlMean_Log  = np.zeros(TL.nStars)*u.m/u.s
        dvAxlMax_Log = np.zeros(TL.nStars)*u.m/u.s
        dvAxlStd_Log = np.zeros(TL.nStars)*u.m/u.s
        # number of burns
        bounce_Log = np.zeros(TL.nStars)
        
        # relative to some trajStartTime
        currentTime = trajStartTime + tau
        
        # just so we can catalogue it
        latDist = self.latDist
        latDistOuter = self.latDistOuter
        
        tic = time.perf_counter()
        for sInd in range(TL.nStars):
            print(tau, " -- start time: ",trajStartTime)
            print(sInd, " / ",TL.nStars)
            
            # let's try to stationkeep!
            good = True
            try:
                nBounces, timeLeft, dvLog, dvAxialLog, driftLog = self.stationkeep(TL,sInd,currentTime,dt=dt,simTime=simTime,SRP=SRP,Moon=Moon)
            except:
                # stationkeeping didn't work! sad. just skip that index, then.
                good = False
            
            # stationkeeping worked!
            if good:
                bounce_Log[sInd] = nBounces
                tDriftMax_Log[sInd]  = np.max(driftLog)
                tDriftMean_Log[sInd] = np.mean(driftLog)
                tDriftStd_Log[sInd]  = np.std(driftLog)
                dvMax_Log[sInd]  = np.max(dvLog)
                dvMean_Log[sInd] = np.mean(dvLog)
                dvStd_Log[sInd]  = np.std(dvLog)
                dvAxlMax_Log[sInd]  = np.max(dvAxialLog)
                dvAxlMean_Log[sInd] = np.mean(dvAxialLog)
                dvAxlStd_Log[sInd]  = np.std(dvAxialLog)
                
            # NOMENCLATURE:
            # m  - model:   (IN - inertial) (RT - rotating)
            # ic - initial conditions: (CNV - centered, neutral velocity) (WIP - well, ideal parabola)
            # n  - number of stars
            # ld - lateral distance (in meters * 10 )
            # ms - reference epoch for mission start (in mjd)
            # t  - time since reference epoch (in days)

            filename = 'skMap_mIN_icWIP_n'+str(int(TL.nStars))+ \
                '_ld' + str(int(latDist.value*10)) + '_ms' + str(int(trajStartTime.value)) + \
                '_t' + str(int((tau).value)) + '_SRP' + str(int(SRP)) + '_Moon' + str(int(Moon))
                
            timePath = os.path.join(self.cachedir, filename+'.skmap')
            toc = time.perf_counter() 
            
            A = { 'simTime':simTime, 'compTime':toc-tic, 'bounces': bounce_Log, 'SRP':SRP, 'Moon':Moon, 'dt':dt, \
                  'tDriftMax': tDriftMax_Log, 'tDriftMean': tDriftMean_Log, 'tDriftStd': tDriftStd_Log,\
                  'dvMax'    : dvMax_Log,     'dvMean'    : dvMean_Log,     'dvStd'    : dvStd_Log,\
                  'dvAxlMax' : dvAxlMax_Log,  'dvAxlMean' : dvAxlMean_Log,  'dvAxlStd' : dvAxlStd_Log,\
                  'dist':TL.dist,'lon':TL.coords.lon,'lat':TL.coords.lat,'missionStart':trajStartTime,'tau':tau,\
                  'latDist':latDist,'latDistOuter':latDistOuter,'trajStartTime':currentTime}
                
            with open(timePath, 'wb') as f:
                pickle.dump(A, f)
                