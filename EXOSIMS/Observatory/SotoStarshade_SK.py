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
    
    def __init__(self,orbit_datapath=None,**specs): 

        SotoStarshade_ContThrust.__init__(self,**specs)  

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
        
        nu,gam,dnu_,dgam_ = self.EulerAngles(TL,sInd,currentTime,tRange)
        dnu = self.convertAngVel_to_dim(dnu_).to('rad/s').value
        dgam = self.convertAngVel_to_dim(dgam_).to('rad/s').value
        
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
        print(sign)
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

    def drift(self,TL,sInd,trajStartTime, latDist = 1*u.m, axDist = 250*u.km,\
                     dt=20*u.min,neutralStart=True,s0=None,fullSol=False):
        """
        
        return:
            cross - 0,1,or 2 for Tolerance Not Crossed, Lateral Cross, and Axial Cross
        """
        
        EoM = lambda t,s: self.equationsOfMotion_aboutS(t,s,TL,sInd,trajStartTime,integrate=True)
        
        # defining times
        t0 = self.convertTime_to_canonical(np.mod(trajStartTime.value,self.equinox.value)*u.d)[0]
        tF = t0 + self.convertTime_to_canonical( dt )
        tInt = np.linspace(0,tF-t0,5000)
                
        # remember these states are relative to the nominal track S/0
        # either start exactly on nominal track or else place it yourself
        if neutralStart:
            s0 = np.zeros(6)
            
        # integrate forward to t0 + dt
        res = solve_ivp(EoM,[tInt[0],tInt[-1]],s0,t_eval=tInt,rtol=1e-13,atol=1e-13,method='Radau')
        t_int = res.t
        y_int = res.y
        x,y,z,dx,dy,dz = y_int
        
        #gotta rotate trajectory array from C-frame to R-frame components
        rSpS_C,RdrSpS_C = self.rotate_RorC(TL,sInd,trajStartTime,y_int,t_int,final_frame='C')
        rSpS_Cdim = self.convertPos_to_dim(rSpS_C).to('m')
        
        #### LATERAL OFFSET ########################
        # lateral distance from nominal point S (in meters)
        delta_r = rSpS_Cdim[0:2,:]
        dr = np.linalg.norm(delta_r,axis=0) * u.m
        
        # finding where lateral distance is greater than threshhold
        latCrossInds = np.where( dr >= latDist )[0]
        
        if latCrossInds.size > 0:
            # we crossed the lateral threshhold
            crossIndL = latCrossInds[0] 
            cross_Lat   = True
        else:
            # we did not cross the lateral threshhold
            cross_Lat   = False
        
        #### AXIAL OFFSET ##########################
        # axial distance from nominal point S (in meters)
        dz = rSpS_Cdim[2,:] 
        
        # finding where axial distance is greater than threshhold
        axlCrossInds = np.where( dz >= axDist )[0]
        
        if axlCrossInds.size > 0:
            # we crossed the axial threshhold
            crossIndA = axlCrossInds[0]
            cross_Axl   = True
        else:
            # we did not cross the axial threshhold
            cross_Axl = False
        
        ### Resolving cross events #################
        if cross_Axl + cross_Lat == 0:
            # nothing happened, WE'RE STILL IN THE CIRCLE MURPH
            cross = 0
            driftTime = None
            # just gonna output the last state
            rSpS_Ccross = rSpS_C[:,-1]
            RdrSpS_Ccross = RdrSpS_C[:,-1]
        else:
            # we did cross the cylinder somewhere (1: laterally, 2:axially)
            if cross_Axl + cross_Lat == 2:
                if crossIndL < crossIndA:
                    cross = 1
                    crossInd = crossIndL
                else:
                    cross = 2
                    crossInd = crossIndA
            else:
                if cross_Lat:
                    cross = 1
                    crossInd = crossIndL
                else:
                    cross = 2
                    crossInd = crossIndA
        
            #MAKE INTO ITS OWN FUNCTION
            # actual positions just before and after crossing boundary
            relevantInds = [crossInd-1,crossInd] #crossInd-1 needs to exist
            rSpS_C_nearBoundary   = rSpS_C[:,relevantInds]
            RdrSpS_C_nearBoundary = RdrSpS_C[:,relevantInds]
            # times just before and after crossing (canonical units)
            tInterpRange = t_int[relevantInds]
            # creating initial interpolant, to find a more precise crossing time
            ri = interp.interp1d( tInterpRange,rSpS_C_nearBoundary )
            vi = interp.interp1d( tInterpRange,RdrSpS_C_nearBoundary )
            
            # interpolate between two relevants points to find better crossing
            tNew = np.linspace( tInterpRange[0] , tInterpRange[-1], 1000)
            rNew = ri(tNew) 
            vNew = vi(tNew)
            
            # find new cross time for either crossing event
            if cross == 1:
                newDelta_r = self.convertPos_to_dim(rNew[0:2,:]).to('m')
                drNew = np.linalg.norm(newDelta_r,axis=0) * u.m
                
                # finding where lateral distance is greater than threshhold
                newLatCrossInds = np.where( drNew >= latDist )[0]
                newCrossInd = newLatCrossInds[0] - 1
                
            else:
                dzNew = self.convertPos_to_dim(rNew[2,:]).to('m')
                
                # finding where lateral distance is greater than threshhold
                newAxlCrossInds = np.where( dzNew >= axDist )[0]
                newCrossInd = newAxlCrossInds[0] - 1
            
            newCrossTime   = tNew[newCrossInd]
            rSpS_Ccross    = rNew[:,newCrossInd]
            RdrSpS_Ccross  = vNew[:,newCrossInd]
            
            driftTime = self.convertTime_to_dim(newCrossTime).to('min')
            rSpS_C   = np.hstack([ rSpS_C[:,:crossInd] , rSpS_Ccross.reshape(3,1)])
            RdrSpS_C = np.hstack([ RdrSpS_C[:,:crossInd] , RdrSpS_Ccross.reshape(3,1)])
        
        
        if fullSol:
            return cross, driftTime, t_int, rSpS_C, RdrSpS_C
        else:
            return cross, driftTime, t_int, rSpS_Ccross, RdrSpS_Ccross
                

    def guessAParabola(self,TL,sInd,trajStartTime,rSpS_C,RdrSpS_C,cross,\
                       latDist = 1*u.m, axDist = 250*u.km):
        
        
        Ra_S0_R , f_S0_R, fA_B, fL_B, theta = self.starshadeAcceleration(TL,sInd,trajStartTime)

        if cross == 1:
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
            
            # time of flight in C-frame units
            tof = np.abs( x0 / dx0 ) if x0 != 0 else 4
            
            ### planar parabolic trajectories in C-frame units
            # time range from 0 to full time of flight
            t = np.linspace(0,tof,1000)
            
            # x and y throughout entire new parabolic trajectory
            xP  = dx0*t + x0
            yP  = -0.5*t**2 + dy0*t + y0
            
            parab   = np.vstack([xP,yP])  * DU
            RdrSpS_C_newIC = np.array([dx0,dy0,dz0_]) * VU
            dt_newTOF = tof * TU
            
        else:
            a = np.abs(fA_B[0])
            DU = self.convertPos_to_canonical(axDist)
            VU = np.sqrt( a*DU )
            TU = np.sqrt( DU/a )
        
            # converting initial state from canonical units to C-frame units
            x0,y0,z0       = rSpS_C[:,-1] / DU
            dx0_,dy0_,dz0_ = RdrSpS_C[:,-1] / VU
            
            sgnA = np.sign(fA_B[0])
            sgnV = np.sign(dz0_)
            sgnD = np.sign(z0)
            
            dz0 = -sgnV * np.sqrt( 4 * np.abs(z0) ) if sgnD * sgnA == 1 else 0
            tof = 2 * np.sqrt( 4 * np.abs(z0) )
            
            t = np.linspace(0,tof,1000)
            
            parab = (sgnA * t**2 + dz0 * t + z0) * DU
            RdrSpS_C_newIC = np.array([dx0_,dy0_,dz0]) * VU
            dt_newTOF = tof * TU
        
        return dt_newTOF, RdrSpS_C_newIC, parab
