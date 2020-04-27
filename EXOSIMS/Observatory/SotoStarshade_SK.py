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


class SotoStarshade_SK(SotoStarshade_ContThrust):
    """ StarShade Observatory class
    This class is implemented at L2 and contains all variables, functions, 
    and integrators to calculate occulter dynamics. 
    """
    
    def __init__(self,latDist=0.9,latDistOuter=0.95,axlDist=250,**specs): 

        SotoStarshade_ContThrust.__init__(self,**specs)  
        
        self.latDist      = latDist * u.m
        self.latDistOuter = latDistOuter * u.m
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

# =============================================================================
# Dynamics and Kinematics
# =============================================================================

    def starshadeAcceleration(self,TL,sInd,currentTime,tRange=[0]):
        
        s = self.convertPos_to_canonical(self.occulterSep)
        
        # coordinates of the star from the TL
        lamb = TL.coords.barycentrictrueecliptic.lon[sInd].to('rad')  #RA
        beta = TL.coords.barycentrictrueecliptic.lat[sInd].to('rad')  #DEC
        varpi = TL.parx[sInd].to('rad')                                #parallax angle
        
        nu,gam,dnu,dgam = self.EulerAngles(TL,sInd,currentTime,tRange)
        # dnu = self.convertAngVel_to_dim(dnu_).to('rad/s').value
        # dgam = self.convertAngVel_to_dim(dgam_).to('rad/s').value
        
        # time in canonical units
        absTimes = currentTime + tRange                   #mission times  in jd
        t = self.convertTime_to_canonical(np.mod(absTimes.value,self.equinox.value)*u.d) * u.rad
        
        # halo positions and velocities 
        haloPos = self.haloPosition(absTimes) + np.array([1,0,0])*self.L2_dist.to('au')
        haloVel = self.haloVelocity(absTimes)
        
        # halo positions and velocities in canonical units
        rT0_R     = np.array([self.convertPos_to_canonical(haloPos[:,n]) for n in range(3)])
        RdrT0_R   = np.array([self.convertVel_to_canonical(haloVel[:,n]) for n in range(3)])
        
        x,y,z    = rT0_R
        dx,dy,dz = RdrT0_R
        
        #force/acceleration on Telescope throughout Halo orbit
        sT0_R  = np.vstack([rT0_R,RdrT0_R])
        RddrT0_R = self.equationsOfMotion_CRTBP_noSRP(t,sT0_R)[3:6,:]
        ddx,ddy,ddz = RddrT0_R
        
        # some terms that will make my life easier
        varpiV         = varpi.value
        bigAngle       = -lamb + t + nu
        thatPeskyDenom =  varpiV*z - np.sin(beta)
        
        #ddnu angular acceleration (in canonical units)
        ddnu1 = -varpiV * (-varpiV*dx*np.sin(nu) + varpiV*dy*np.cos(nu) + np.cos(beta)*np.cos(bigAngle)) *dz *np.tan(gam)
        ddnu2 = -np.cos(gam)**(-2) * thatPeskyDenom * (-varpiV*dx*np.sin(nu)+varpiV*dy*np.cos(nu) +np.cos(beta)*np.cos(bigAngle)) * dgam
        ddnu3 = thatPeskyDenom * (-varpiV*dx*np.cos(nu)*dnu - varpiV*dy*np.sin(nu)*dnu - varpiV*np.sin(nu)*ddx + varpiV*np.cos(nu)*ddy - (dnu+1)*np.sin(bigAngle)*np.cos(beta))*np.tan(gam)
        ddnu4 = thatPeskyDenom**2 * np.tan(gam)**2
        
        ddnu = (ddnu1+ddnu2+ddnu3)/ddnu4
        
        #ddgam angular acceleration (in canonical units)
        ddgam1 = -varpiV * (varpiV * (dx*np.cos(gam)*np.cos(nu)+dy*np.sin(nu)*np.cos(gam)-dz*np.sin(gam)) + np.sin(bigAngle)*np.cos(beta)*np.cos(gam))*np.cos(gam)*dz
        ddgam2 = thatPeskyDenom * ( -2*varpiV*dx*np.sin(gam)*np.cos(gam)*np.cos(nu)*dgam - varpiV*dx*np.sin(nu)*np.cos(gam)**2*dnu - 2*varpiV*dy*np.sin(gam)*np.sin(nu)*np.cos(gam)*dgam + varpiV*dy*np.cos(gam)**2*np.cos(nu)*dnu - 2*varpiV*dz*np.cos(gam)**2*dgam + \
                                   varpiV*dz*dgam - varpiV*np.sin(gam)*np.cos(gam)*ddz + varpiV*np.sin(nu)*np.cos(gam)**2*ddy + varpiV*np.cos(gam)**2*np.cos(nu)*ddx - 2*np.sin(bigAngle)*np.sin(gam)*np.cos(beta)*np.cos(gam)*dgam + np.cos(beta)*np.cos(bigAngle)*np.cos(gam)**2*dnu + \
                                   np.cos(beta)*np.cos(bigAngle)*np.cos(gam)**2 )
        ddgam3 = thatPeskyDenom**2
        
        ddgam = (ddgam1+ddgam2)/ddgam3
        
        #required acceleration for starshade S for stationkeeping with given star's LOS
        Ra_S0_Rx = ddx - s*(dgam*np.sin(nu)*np.cos(gam) + dnu*np.sin(gam)*np.cos(nu))*dnu + s*ddgam*np.cos(gam)*np.cos(nu) - s*ddnu*np.sin(gam)*np.sin(nu) - s*dgam**2*np.sin(gam)*np.cos(nu)
        Ra_S0_Ry = ddy + s*(dgam*np.cos(gam)*np.cos(nu) - dnu*np.sin(gam)*np.sin(nu))*dnu + s*ddgam*np.sin(nu)*np.cos(gam) + s*ddnu*np.sin(gam)*np.cos(nu) - s*dgam**2*np.sin(gam)*np.sin(nu)
        Ra_S0_Rz = ddz - s*ddgam*np.sin(gam) - s*dgam**2*np.cos(gam)
        
        Ra_S0_R = np.vstack([Ra_S0_Rx,Ra_S0_Ry,Ra_S0_Rz])
        
        #required position and velocity of point S (starshade) 
        rS_R,RdrS_R,RdrT_R = self.starshadeVelocity(TL,sInd,currentTime,tRange,frame='rot')
        
        #force on starshade at point S 
        sS0_R  = np.vstack([rS_R.T,RdrS_R.T])
        f_S0_R = self.equationsOfMotion_CRTBP_noSRP(t,sS0_R)[3:6,:]
        
        #differential force
        df = f_S0_R - Ra_S0_R
        dfx,dfy,dfz = df #components in R frame
        
        #axial force in the B-frame
        fA_B = dfx * np.sin(gam) *np.cos(nu) + dfy * np.sin(gam) * np.sin(nu) + dfz*np.cos(gam)
        
        #lateral force components in the B-frame
        fL_Bx = dfx*np.cos(gam)*np.cos(nu) + dfy*np.sin(nu)*np.cos(gam)-dfz*np.sin(gam)
        fL_By = -dfx*np.sin(nu) + dfy*np.cos(nu)
        fL_Bz = np.zeros(nu.shape)
        
        #lateral force in the B-frame
        fL_B = np.vstack([fL_Bx, fL_By, fL_Bz])
        
        #final roll angle to get to C-frame
        theta = np.arctan2( fL_B[0,:] , -fL_B[1,:])
        
        return Ra_S0_R.value , f_S0_R, fA_B.value, fL_B.value, theta
    
    
    def rotate_RorC(self,TL,sInd,trajStartTime,s_int,t_int,final_frame='C'):
        """
        
        Args:
            s_int (6xn):
                Integrated states in canonical units
            t_int (n):
                Integrated times in canonical units


        """
        # getting the associated 3-2-3 Euler set
        t_MJD = self.convertTime_to_dim( t_int )
        t_intMJD = t_MJD - t_MJD[0]
        nu,gam,dnu,dgam = self.EulerAngles(TL,sInd,trajStartTime,t_intMJD)
        Ra_S0_R , f_S0_R, fA_B, fL_B, theta = self.starshadeAcceleration(TL,sInd,trajStartTime,t_intMJD)
        
        # sign of rotation angle depending on rotating to C or R frame
        sign = 1 if final_frame == 'C' else -1

        # extracting initial states
        x,y,z,dx,dy,dz = s_int
        r_i  = np.vstack([x,y,z])
        Rv_i = np.vstack([dx,dy,dz])
        # defining final states
        r_f = np.zeros(r_i.shape)
        Rv_f = np.zeros(Rv_i.shape)
        
        for n in range(len(t_int)):
            AcR = self.rot(nu[n],3)
            BcA = self.rot(gam[n],2)
            CcB = self.rot(theta[n],3)
            if sign == 1:
                fci = np.matmul( CcB,  np.matmul(BcA,AcR) )
            else:
                fci = np.matmul( AcR.T,  np.matmul(BcA.T,CcB.T))
            r_f[:,n]  = np.matmul( fci, r_i[:,n]  )
            Rv_f[:,n] = np.matmul( fci, Rv_i[:,n]  )
        
        return r_f , Rv_f
# =============================================================================
# Equations of Motion
# =============================================================================

    def equationsOfMotion_CRTBP_noSRP(self,t,state,integrate=False):
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
        m1 = self.m1
        m2 = self.m2
        
        x,y,z,dx,dy,dz = state
        
        #occulter distance from each of the two other bodies
        r1 = np.sqrt( (x + mu)**2. + y**2. + z**2. )
        r2 = np.sqrt( (1. - mu - x)**2. + y**2. + z**2. )
        
        #equations of motion
        ds1 = x + 2.*dy + m1*(-mu-x)/r1**3. + m2*(1.-mu-x)/r2**3.
        ds2 = y - 2.*dx - m1*y/r1**3. - m2*y/r2**3.
        ds3 = -m1*z/r1**3. - m2*z/r2**3.
        
        dr  = [dx,dy,dz]
        ddr = [ds1,ds2,ds3]
        
        ds = np.vstack([dr,ddr])
        
        if integrate:
            ds = ds.flatten()

        return ds
    
    
    def equationsOfMotion_aboutS(self,t,state,TL,sInd,trajStartTime,integrate=False):
        
        # original state of S' wrt to S in R-frame components and derivatives
        x,y,z,dx,dy,dz = state
        
        # reference epoch in canonical units
        t0 = self.convertTime_to_canonical(np.mod(trajStartTime.value,self.equinox.value)*u.d)[0]
        tF = t0 + t
        
        if integrate:
            # current time in dimensional units
            currentTime = trajStartTime + self.convertTime_to_dim(t)
            
            # retrieving acceleration of point S rel O at currentTime
            Ra_S0_R , f_S0_R, fA_B, fL_B, theta = self.starshadeAcceleration(TL,sInd,currentTime)
            
            # retrieving pos and vel of point S rel O at currentTime
            rS0_R,RdrS0_R,RdrT0_R = self.starshadeVelocity(TL,sInd,currentTime,[0],frame='rot')
            
            # determining position and velocity of S' rel O at currentTime
            rSp0_R   = rS0_R[0] + state[0:3]
            RdrSp0_R = RdrS0_R[0] + state[3:6]
            s0 = np.hstack([rSp0_R,RdrSp0_R])
            
            # calculating force on S' at currentTime
            f_Sp0_R = self.equationsOfMotion_CRTBP_noSRP(tF,s0,integrate=True)[3:6]
        
        else:
            # current time in dimensional units
            tRange = self.convertTime_to_dim(t)
            
            # retrieving acceleration of point S rel O at currentTime
            Ra_S0_R , f_S0_R, fA_B, fL_B, theta = self.starshadeAcceleration(TL,sInd,trajStartTime,tRange)
        
            # retrieving pos and vel of point S rel O at currentTime
            rS0_R,RdrS0_R,RdrT0_R = self.starshadeVelocity(TL,sInd,trajStartTime,tRange,frame='rot')
            
            # determining position and velocity of S' rel O at currentTime
            rSp0_R   = rS0_R.T + state[0:3,:]
            RdrSp0_R = RdrS0_R.T + state[3:6,:]
            s0 = np.vstack([rSp0_R,RdrSp0_R])
            # calculating force on S' at currentTime
            f_Sp0_R = self.equationsOfMotion_CRTBP_noSRP(tF,s0,integrate=False)[3:6]

        # setting final second derivatives and stuff
        if integrate:
            dr  = [dx,dy,dz]
            ddr = f_Sp0_R - Ra_S0_R.flatten()
            ds = np.vstack([dr,ddr])
            ds = ds.flatten()
            
        else:
            dr = np.vstack([dx,dy,dz])
            ddr = f_Sp0_R - Ra_S0_R
            ds = np.vstack([dr,ddr])
        
        return ds
    
    
    def boundary_conditions_SK(self,sA,sB):
        """ Creates boundary conditions for solving a boundary value problem
        """
    
        BCo1 = sA[0] - self.sA[0]
        BCo2 = sA[1] - self.sA[1]
        BCo3 = sA[2] - self.sA[2]

        
        BCf1 = sB[0] - self.sB[0]
        BCf2 = sB[1] - self.sB[1]
        BCf3 = sB[2] - self.sB[2]

        BC = np.array([BCo1,BCo2,BCo3,BCf1,BCf2,BCf3])

        return BC   
# =============================================================================
# Deadbanding
# =============================================================================

    def crossThreshholdEvent(self,t,s,TL,sInd,trajStartTime,latDist):
        
        currentTime = trajStartTime + self.convertTime_to_dim(t)
        # converting state vectors from R-frame to C-frame
        rSpS_C,RdrSpS_C = self.rotate_RorC(TL,sInd,currentTime,s,np.array([0]),final_frame='C')
        # converting units to meters
        rSpS_Cdim = self.convertPos_to_dim(rSpS_C).to('m')
        
        
        # distance from S (in units of m)
        delta_r = rSpS_Cdim[0:2,:]
        dr = np.linalg.norm(delta_r,axis=0) * u.m
        
        # distance from inner threshhold
        distanceFromLim = (dr - latDist).value
        # print(latDist,distanceFromLim)
        return distanceFromLim


    def drift(self,TL,sInd,trajStartTime,dt=20*u.min,neutralStart=True,s0=None,fullSol=False):
        """
        
        return:
            cross - 0,1,or 2 for Tolerance Not Crossed, Lateral Cross, and Axial Cross
        """
        
        # defining equations of motion
        EoM = lambda t,s: self.equationsOfMotion_aboutS(t,s,TL,sInd,trajStartTime,integrate=True)
        
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
        if neutralStart:
            s0 = np.zeros(6)
            
        # integrate forward to t0 + dt
        res = solve_ivp(EoM,[tInt[0],tInt[-1]],s0,t_eval=tInt,events=(crossInner,crossOuter,),rtol=1e-13,atol=1e-13,method='Radau')
        t_int = res.t
        y_int = res.y
        x,y,z,dx,dy,dz = y_int
        
        #gotta rotate trajectory array from R-frame to C-frame components
        rSpS_C,RdrSpS_C = self.rotate_RorC(TL,sInd,trajStartTime,y_int,t_int,final_frame='C')
        rSpS_Cdim = self.convertPos_to_dim(rSpS_C).to('m')
        
        #there should be 2 events, even if they weren't triggered (they'd be empty)
        t_innerEvent = res.t_events[0]
        t_outerEvent = res.t_events[1]
        s_innerEvent = res.y_events[0][-1] if t_innerEvent.size > 0 else np.zeros(6)
        s_outerEvent = res.y_events[1][0]  if t_outerEvent.size > 0 else np.zeros(6)
        r_C_outer = -np.ones(3)
        if t_outerEvent.size > 0:
            r_C_outer,v_C_outer = self.rotate_RorC(TL,sInd,trajStartTime,s_outerEvent,t_outerEvent,final_frame='C')
        
        #outer event triggered in positive y -> we need to burn!
        #(if we trigger in negative y, we should then just resolve the inner boundary crossing instead)
        if t_outerEvent.size > 0 and r_C_outer[1] > 0:
            print('crossed outer')
            cross = 2
            driftTime = self.convertTime_to_dim(t_outerEvent[0]).to('min')
            # rotate states at crossing event to C frame
            r_cross,v_cross = self.rotate_RorC(TL,sInd,trajStartTime,s_outerEvent,t_outerEvent,final_frame='C')
            
            r_full = rSpS_C
            v_full = RdrSpS_C
        
        #inner event triggered AND => outer event triggered only in negative y OR outer event not triggered at all
        #(this means we should look at the last inner event trigger, either way)
        elif t_innerEvent.size > 0:
            print('crossed inner')
            cross = 1
            driftTime = self.convertTime_to_dim(t_innerEvent[-1]).to('min')
            # rotate states at crossing event to C frame
            t_inner = t_innerEvent if t_innerEvent.size == 0 else np.array([t_innerEvent[-1]])
            r_cross,v_cross = self.rotate_RorC(TL,sInd,trajStartTime,s_innerEvent,t_inner,final_frame='C')
            
            if fullSol:
                tEndInd = np.where( t_int <= t_inner )[0][-1]
                t_input = t_int[0:tEndInd]
                s_input = y_int[:,0:tEndInd]
                s_interp = interp.interp1d( t_input,s_input )
                
                t_interped = np.linspace(t_input[0] , t_input[-1], len(tInt) )
                s_interped = s_interp( t_interped )
                r_full,v_full = self.rotate_RorC(TL,sInd,trajStartTime,s_interped,t_interped,final_frame='C')
            
        #no events were triggered, need to run drift for longer
        else:
            print('crossed nothing')
            cross = 0
            driftTime = self.convertTime_to_dim(t_int[-1]).to('min')
            r_cross  = rSpS_C[:,-1]
            v_cross  = RdrSpS_C[:,-1]
        
            r_full = rSpS_C
            v_full = RdrSpS_C
            
        if fullSol:
            return cross, driftTime, t_int, r_full, v_full
        else:
            return cross, driftTime, t_int, r_cross, v_cross


    def guessAParabola(self,TL,sInd,trajStartTime,rSpS_C,RdrSpS_C,cross,latDist = 0.9*u.m,fullSol=False):
        
        
        Ra_S0_R , f_S0_R, fA_B, fL_B, theta = self.starshadeAcceleration(TL,sInd,trajStartTime)


        a = np.linalg.norm(fL_B)
        DU = self.convertPos_to_canonical(latDist)
        VU = np.sqrt( a*DU )
        TU = np.sqrt( DU/a )
    
        # converting initial state from canonical units to C-frame units
        x0,y0,z0       = rSpS_C[:,-1] / DU
        dx0_,dy0_,dz0_ = RdrSpS_C[:,-1] / VU
    
        # finding intercept point
        if y0 > 0.5:
            xi = x0
            yi = y0
        else:
            xi = np.sqrt(1+y0)/np.sqrt(2) if x0 > 0 else -np.sqrt(1+y0)/np.sqrt(2)
            yi = np.sqrt(1-y0)/np.sqrt(2)
            
        # initial velocities of the second parabola
        dx0 = np.sqrt(yi*(1-yi))/np.sqrt(2) if x0 < 0 else -np.sqrt(yi*(1-yi))/np.sqrt(2)
        dy0 = -dx0*xi/yi + (xi-x0)/dx0
        
        #tof and velocities if we're really close to the well
        if np.abs(x0) < 0.05:
            dy0 = (3*yi-1)*np.sqrt( (1+yi)/(2*yi) ) if y0 < 0.5 else (1-yi)*np.sqrt( (1+yi)/(2*yi) )
            
            ymax = dy0**2/2 + y0
            tof  = np.sqrt(2) * (np.sqrt(ymax-y0) + np.sqrt(ymax+1))
            dx0  = -x0/tof
        else:
            # time of flight in C-frame units
            tof = np.abs( x0 / dx0 ) 
        
        ### planar parabolic trajectories in C-frame units
        # time range from 0 to full time of flight
        t = np.linspace(0,tof,1000)
        
        # x and y throughout entire new parabolic trajectory
        xP  = dx0*t + x0
        yP  = -0.5*t**2 + dy0*t + y0
        
        parab   = np.vstack([xP,yP])  * DU
        RdrSpS_C_newIC = np.array([dx0,dy0,0]) * VU
        dt_newTOF = tof * TU
        
        # calculate delta v
        sNew_C = np.hstack([rSpS_C[:,-1] , RdrSpS_C_newIC])
        r0new,v0new_R = self.rotate_RorC(TL,sInd,trajStartTime,sNew_C,np.array([0]),final_frame='R')
        
        sOld_C = np.hstack([rSpS_C[:,-1] , RdrSpS_C[:,-1]])
        r0old,v0old_R = self.rotate_RorC(TL,sInd,trajStartTime,sOld_C,np.array([0]),final_frame='R')
        dv = np.linalg.norm( v0new_R - v0old_R )
        dv_dim = self.convertVel_to_dim( dv ).to('m/s')
        
        if fullSol:
            return dt_newTOF, RdrSpS_C_newIC, dv_dim, parab
        else:
            return dt_newTOF, RdrSpS_C_newIC, dv_dim


    def stationkeep(self,TL,sInd,trajStartTime,dt=30*u.min,simTime=1*u.hr):
        
        
        s0_C = np.array([0,0,0,0,0,0])
        r0,v0 = self.rotate_RorC(TL,sInd,trajStartTime,s0_C,np.array([0]),final_frame='R')
        s0 = np.hstack([r0.flatten(),v0.flatten()])
        
        cross, driftTime, t_int, rSpS_C, RdrSpS_C = self.drift(TL,sInd,trajStartTime, dt = dt,neutralStart=False,s0=s0,fullSol=True)
        
        
        reDo = 0
        if cross == 0:
            while cross == 0:
                print('redo!')
                dt *= 2
                cross, driftTime, t_int, rSpS_C, RdrSpS_C = self.drift(TL,sInd,trajStartTime, dt = dt,neutralStart=False,s0=s0,fullSol=True)
                
                reDo += 1
                if reDo > 5:
                    break
                
        timeLeft = simTime - driftTime
        nBounces = 1
        dvLog    = np.array([]) 
        driftLog = np.array([driftTime.to('min').value])
        
        while timeLeft.to('min').value > 0:
            print(timeLeft.to('min'))
            trajStartTime += driftTime
            latDist = self.latDist if cross == 1 else self.latDistOuter if cross == 2 else 0
            
            dt_new, RdrSpS_C_new, dv, parab = self.guessAParabola(TL,sInd,trajStartTime,rSpS_C,RdrSpS_C,cross,latDist,fullSol=True)
            dt_newGuess = self.convertTime_to_dim(dt_new).to('min')  * 2
            
            s0_Cnew     = np.hstack([ rSpS_C[:,-1] , RdrSpS_C_new ])
            r0,v0 = self.rotate_RorC(TL,sInd,trajStartTime,s0_Cnew,np.array([0]),final_frame='R')
            s0_new = np.hstack([r0.flatten(),v0.flatten()])
            
            cross, driftTime, t_int, rSpS_C, RdrSpS_C = self.drift(TL,sInd,trajStartTime, dt = dt_newGuess,neutralStart=False,s0=s0_new,fullSol=True)
            
            reDo = 0
            if cross == 0:
                while cross == 0:
                    print('redo!')
                    dt_newGuess *= 2
                    cross, driftTime, t_int, rSpS_C, RdrSpS_C = self.drift(TL,sInd,trajStartTime, dt = dt_newGuess, \
                                                                          neutralStart=False,s0=s0_new,fullSol=True)
                    
                    reDo += 1
                    if reDo > 5:
                        break
            
            #update everything
            nBounces += 1
            timeLeft -= driftTime
            dvLog    = np.hstack([dvLog,dv.to('m/s').value])
            driftLog = np.hstack([driftLog,driftTime.to('min').value])
        
        
        return nBounces, timeLeft, dvLog*u.m/u.s, driftLog*u.min

    def globalStationkeep(self,TL,trajStartTime,tau=0*u.d,dt=30*u.min,simTime=1*u.hr):
        
        tDriftMax_Log  = np.zeros(TL.nStars)*u.min
        tDriftMean_Log = np.zeros(TL.nStars)*u.min
        dvMean_Log  = np.zeros(TL.nStars)*u.m/u.s
        dvMax_Log = np.zeros(TL.nStars)*u.m/u.s
        bounce_Log = np.zeros(TL.nStars)
        
        latDist = self.latDist
        latDistOuter = self.latDistOuter
        
        
        tic = time.perf_counter()
        for sInd in range(TL.nStars):
            print(trajStartTime)
            print(sInd, " / ",TL.nStars)
        
            nBounces, timeLeft, dvLog, driftLog = self.stationkeep(TL,sInd,trajStartTime,simTime=3*u.hr)
            
            bounce_Log[sInd] = nBounces
            tDriftMax_Log[sInd]  = np.max(driftLog)
            tDriftMean_Log[sInd] = np.mean(driftLog)
            dvMax_Log[sInd]  = np.max(dvLog)
            dvMean_Log[sInd] = np.mean(dvLog)
        
            #tID = type Initial Drift
            #icCvUL = initial conditions - Centered velocity towards Upper Left
            #icCNV  = initial conditions - Centered Neutral Velocity
            #ms  = missionStart
            #tau   = simulation start time after missionStart
            filename = 'skMap_IDsim_icCNV_n'+str(int(TL.nStars))+ \
                        'ldI' + str(int(latDist.value*10)) + 'ms' + str(int(trajStartTime.value)) + \
                        'tau' + str(int((tau).value)) + 'd'
                       
            timePath = os.path.join(self.cachedir, filename+'.skmap')
            A = { 'bounces': bounce_Log, 'tDriftMax': tDriftMax_Log, 'tDriftMean': tDriftMean_Log,\
                  'dvMax':dvMax_Log, 'dvMean':dvMean_Log,'simTime':simTime,\
                  'dist':TL.dist,'lon':TL.coords.lon,'lat':TL.coords.lat,'missionStart':trajStartTime,'tau':tau,\
                  'latDist':latDist,'latDistOuter':latDistOuter,'trajStartTime':trajStartTime}
            with open(timePath, 'wb') as f:
                pickle.dump(A, f)
        toc = time.perf_counter()       