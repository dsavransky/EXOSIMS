import numpy as np
import astropy.units as u
from astropy.time import Time
import scipy.integrate as itg
from EXOSIMS.Observatory.WFIRSTObservatoryL2 import WFIRSTObservatoryL2
from scipy.integrate import solve_bvp
import astropy.constants as const
import os, inspect
import scipy.optimize as optimize
import scipy.interpolate as interpolate
try:
    import cPickle as pickle
except:
    import pickle
from scipy.io import loadmat

EPS = np.finfo(float).eps


class SotoStarshade(WFIRSTObservatoryL2):
    """ StarShade Observatory class
    This class is implemented at L2 and contains all variables, functions, 
    and integrators to calculate occulter dynamics. 
    """
    
    def __init__(self, missionStart=60634., maxdVpcnt=0.02,
            setTOF=20,orbit_datapath=None,**specs): 

        WFIRSTObservatoryL2.__init__(self,**specs)

        self.mu = 3.04043263333266026e-6
        
        self.m1 = float(1-self.mu)
        self.m2 = self.mu
        self.setTOF = np.array([setTOF])        

    
        self.dV_tot = self.slewIsp*const.g0*np.log(self.scMass/self.dryMass)
        self.dVmax  = self.dV_tot * maxdVpcnt
        
        
    def halo_pos(self,currentTime):
        """ This method returns the position vector of the WFIRST observatory 
        in the rotating frame of the Earth-Sun system centered at L2.
        """
        # Find the time between Earth equinox and current time(s)
        
        dt = (currentTime - self.equinox).to('yr').value
        t_halo = dt % self.period_halo
        
        # Interpolate to find correct observatory position(s)
        r_halo = self.r_halo_interp_L2(t_halo).T*u.AU
        print 'hello'
        
        return r_halo
    
    
    def halo_vel(self,currentTime):
        """ Finds observatory velocity within its halo orbit about L2
        """
        # Find the time between Earth equinox and current time(s)
        
        dt = (currentTime - self.equinox).to('yr').value
        t_halo = dt % self.period_halo
        
        # Interpolate to find correct observatory velocity(-ies)
        v_halo = self.v_halo_interp(t_halo).T
        v_halo = v_halo*u.au/u.year
        
        return v_halo
    
    
    def equations_of_motion(self,t,s):
        """ Equations of motion for the Circular Restricted Three Body 
        Problem (CRTBP). First order form of the equations for integration, 
        returns 3 velocities and 3 accelerations in (x,y,z) rotating frame
            
        All parameters are normalized so that time = 2*pi sidereal year
        Distances normalized to 1AU
            
        Coordinates are taken in a rotating frame centered at the center of mass
        of the two primary bodies
            
        """
        
        #occulter distance from each of the two other bodies
        r1 = np.sqrt( (self.mu - s[0])**2 + s[1]**2 + s[2]**2 )
        r2 = np.sqrt( (1 - self.mu - s[0])**2 + s[1]**2 + s[2]**2 )
            
        #equations of motion
        ds1 = s[0] + 2*s[4] + self.m1*(-self.mu-s[0])/r1**3 + self.m2*(1-self.mu-s[0])/r2**3
        ds2 = s[1] - 2*s[3] - self.m1*s[1]/r1**3 - self.m2*s[1]/r2**3
        ds3 = -self.m1*s[2]/r1**3 - self.m2*s[2]/r2**3
        
        ds = np.vstack((s[3],s[4],s[5],ds1,ds2,ds3))
        
        return ds
    
    
    def boundary_conditions(self,rA,rB):
    
        BC1 = rA[0] - self.rA[0]
        BC2 = rA[1] - self.rA[1]
        BC3 = rA[2] - self.rA[2]
        
        BC4 = rB[0] - self.rB[0]
        BC5 = rB[1] - self.rB[1]
        BC6 = rB[2] - self.rB[2]
        
        return np.array([BC1,BC2,BC3,BC4,BC5,BC6])
    
    
    def send_it(self,TL,nA,nB,tA,tB):
        """ Taking Occulter from One Target to Another
        Given a target list and indeces of two target stars in it.
        Given initial time tA and final time tB (could be tA + dt)
        Runs shooting algorithm within and returns the desired occulter trajectory.
        """
        
        t = np.linspace(tA.value,tB.value,2)    #discretizing time
        t = Time(t,format='mjd')                #converting time to modified julian date
        
        #position of WFIRST at the given times in rotating frame
        r_halo = self.halo_pos(t).to('au')
        r_WFIRST = (r_halo + np.array([1,0,0])*self.L2_dist).value
        
        #position of stars wrt to WFIRST
        starA = self.eclip2rot(TL,nA,tA).value
        starB = self.eclip2rot(TL,nB,tB).value
        
        starA_wfirst = starA - r_WFIRST[ 0]
        starB_wfirst = starB - r_WFIRST[-1]
        
        #corresponding unit vectors pointing WFIRST -> Target Star
        uA = starA_wfirst / np.linalg.norm(starA_wfirst)
        uB = starB_wfirst / np.linalg.norm(starB_wfirst)
        
        #position vector of occulter in heliocentric frame
        self.rA = uA*self.occulterSep.to('au').value + r_WFIRST[ 0]
        self.rB = uB*self.occulterSep.to('au').value + r_WFIRST[-1]
        
        a = ((np.mod(tA.value,self.equinox.value)*u.d)).to('yr') / u.yr * (2*np.pi)
        b = ((np.mod(tB.value,self.equinox.value)*u.d)).to('yr') / u.yr * (2*np.pi)
        
        #running shooting algorithm
        t = np.linspace(a,b,2)
        
        guess = (self.rB-self.rA)/(b-a)
        sG = np.array([np.full_like(t,self.rA[0]),np.full_like(t,self.rA[1]),np.full_like(t,self.rA[2]), 
                       np.full_like(t,guess[0]),np.full_like(t,guess[1]),np.full_like(t,guess[2])])               
            
        sol = solve_bvp(self.equations_of_motion,self.boundary_conditions,t,sG,tol=1e-8)

        return sol.y.T
    
    
    def eclip2rot(self,TL,sInd,currentTime):
        
        star_pos = TL.starprop(sInd,currentTime)[0].to('au')
        theta    = (np.mod(currentTime.value,self.equinox.value[0])*u.d).to('yr') / u.yr * (2*np.pi)
        
        star_rot = np.array([np.dot(self.rot(theta.value, 3),star_pos.to('AU').value)])*u.AU

        return star_rot[0]
    

    def calculate_dV(self,dt,TL,N1,N2,tA):
        
        if dt.shape:
            dt = dt[0]
        
        tB = tA + dt*u.d
        
        sol = self.send_it(TL,N1,N2,tA,tB)
        
        v_occulter_A = sol[ 0,3:6]*u.AU/u.year*(2*np.pi) #velocity after leaving star A
        v_occulter_B = sol[-1,3:6]*u.AU/u.year*(2*np.pi) #velocity arriving at star B
        
        #portions of the station-keeping trajectories required for stars A and B respectively
        s0 = self.send_it(TL,N1,N1,tA - 15*u.min,tA) 
        sF = self.send_it(TL,N2,N2,tB,tB + 15*u.min)
        
        v0 = s0[-1,3:6]*u.AU/u.year*(2*np.pi) #velocity needed to maintain constant distance with star A
        vF = sF[ 0,3:6]*u.AU/u.year*(2*np.pi) #velocity needed to maintain constant distance with star B
        
        dvA = (v_occulter_A-v0).to('m/s')
        dvB = (v_occulter_B-vF).to('m/s')
        
        dv = np.linalg.norm(dvA) + np.linalg.norm(dvB)
        
        return dv
    
    
    def minimize_slewTimes(self,TL,N1,N2,tA):
        
        dt_guess=20
        percent=0.05        
        Tol=1e-3
        
        t0 = [dt_guess]
        
        res = optimize.minimize(self.slewTime_objFun,t0,method='COBYLA',
                        constraints={'type': 'ineq', 'fun': self.slewTime_constraints,'args':([TL,N1,N2,tA,percent])},
                        tol=Tol,options={'disp': False})
                        
        opt_slewTime = res.x
        opt_dV       = self.calculate_dV(opt_slewTime,TL,N1,N2,tA)
        
        return opt_slewTime,opt_dV
    
    
    def minimize_fuelUsage(self,TL,N1,N2,tA):
        
        dt_guess=20
        dt_min=1
        dt_max=40        
        Tol=1e-3
        
        t0 = [dt_guess]

        res = optimize.minimize(self.calculate_dV,t0,method='COBYLA',
                        constraints={'type': 'ineq', 'fun': self.fuelUsage_constraints,'args':([dt_min,dt_max])},
                        tol=Tol,args=(TL,N1,N2,tA),options={'disp': False})
        opt_slewTime = res.x
        opt_dV   = res.fun
        
        return opt_slewTime,opt_dV
    
    
    def slewTime_objFun(self,dt):
        if dt.shape:
            dt = dt[0]
            
        return dt
    
    
    def slewTime_constraints(self,dt,TL,N1,N2,tA,percent):
        
        dV = self.calculate_dV(dt,TL,N1,N2,tA)
        dV_max = self.DV_tot * percent 
        
        return dV_max.value - dV
    
    
    def fuelUsage_constraints(self,dt,dt_min,dt_max):
        
        return dt_max - dt, dt - dt_min
    
    
    def star_angularSep(self,TL,N1,N2,tA,tB):
        
        t = np.linspace(tA.value,tB.value,2)    #discretizing time
        t = Time(t,format='mjd')                #converting time to modified julian date
        
        #position of WFIRST at the given times in rotating frame
        r_halo = self.halo_pos(t).to('au')
        r_WFIRST = (r_halo + np.array([1,0,0])*self.L2_dist).value
        
        #position of stars wrt to WFIRST
        star1 = self.eclip2rot(TL,N1,tA).value
        star2 = self.eclip2rot(TL,N2,tB).value
        
        star1_wfirst = star1 - r_WFIRST[ 0]
        star2_wfirst = star2 - r_WFIRST[-1]
        
        #corresponding unit vectors pointing WFIRST -> Target Star
        u1 = star1_wfirst / np.linalg.norm(star1_wfirst)
        u2 = star2_wfirst / np.linalg.norm(star2_wfirst)
        
        angle = (np.arccos(np.dot(u1[0],u2[0].T))*u.rad).to('deg')
        
        return angle
    
    
    def integrate(self,s0,t):
        """ Setting up integration using scipy odeint
        Tolerances are lowered and output info from integration is defined
        as an attribute.        
        """
        
        def EoM(y,t):
            """ Equations of motion for the Circular Restricted Three Body 
            Problem (CRTBP). First order form of the equations for integration, 
            returns 3 velocities and 3 accelerations in (x,y,z) rotating frame
            
            All parameters are normalized so that time = 2*pi sidereal year
            Distances normalized to 1AU
            
            Coordinates are taken in a rotating frame centered at the center of mass
            of the two primary bodies
            
            """
            #setting up state vector
            s1,s2,s3,s4,s5,s6 = y
        
            #occulter distance from each of the two other bodies
            r1 = np.sqrt( (self.mu - s1)**2 + s2**2 + s3**2 )
            r2 = np.sqrt( (1 - self.mu - s1)**2 + s2**2 + s3**2 )
            
            #equations of motion
            ds1 = s1 + 2*s5 + self.m1*(-self.mu-s1)/r1**3 + self.m2*(1-self.mu-s1)/r2**3
            ds2 = s2 - 2*s4 - self.m1*s2/r1**3 - self.m2*s2/r2**3
            ds3 = -self.m1*s3/r1**3 - self.m2*s3/r2**3
        
            ds = [s4,s5,s6,ds1,ds2,ds3]
        
            return ds
        
        sol,info = itg.odeint(EoM, s0, t, full_output = 1,rtol=2.5e-14,atol=1e-22)
        self.info = info
        
        return sol
    

    
    def calculate_slewTimes(self,TL,old_sInd,sInds,currentTime,model):
        
        slewTimes = np.zeros(TL.nStars)*u.d
        sInds = np.arange(TL.nStars)
        dV = np.zeros(TL.nStars)*u.d   
        
        if model is "SotoStarshade":
            
            sd = np.arange(TL.nStars)
            
            if old_sInd is None:
                old_sInd = np.random.randint(0,TL.nStars)

            for x in sInds:
                sd[x] = self.star_angularSep(TL,old_sInd,x,currentTime,self.setTOF)
                dV[x] = self.calculate_dV(self.setTOF,TL,old_sInd,x,currentTime)
                
            slewTimes = np.full_like(sd,self.setTOF[0])
            
            sInds = sInds[np.where(dV < self.dVmax)]
            
        else:
    
            sd = None
            dV = None
            self.ao = self.thrust/self.scMass
            slewTime_fac = (2.*self.occulterSep/np.abs(self.ao)/(self.defburnPortion/2. - 
                    self.defburnPortion**2/4.)).decompose().to('d2')

            if old_sInd is None:
                sd = np.array([np.radians(90)]*TL.nStars)*u.rad
            else:
                # position vector of previous target star
                r_old = TL.starprop(old_sInd, currentTime)[0]
                u_old = r_old.value/np.linalg.norm(r_old)
                # position vector of new target stars
                r_new = TL.starprop(sInds, currentTime)
                u_new = (r_new.value.T/np.linalg.norm(r_new, axis=1)).T
                # angle between old and new stars
                sd = np.arccos(np.clip(np.dot(u_old, u_new.T), -1, 1))*u.rad
            # calculate slew time
            slewTimes = np.sqrt(slewTime_fac*np.sin(sd/2.))
        
        return sd,slewTimes,sInds,dV
    
    def log_occulterResults(self,DRM,slewTimes,sInd,sd,dV,model):
        
        DRM['slew_time'] = slewTimes[sInd].to('day')
        DRM['slew_angle'] = sd[sInd].to('deg')
        
        if model is "SotoStarshade":
            dV = dV[sInd]*u.m/u.s
            slew_mass_used = self.scMass * ( np.exp(-dV/(self.Isp*const.g0)) - 1)
            DRM['slew_dV'] = dV
            DRM['slew_mass_used'] = slew_mass_used.to('kg')
            self.scMass = self.scMass - slew_mass_used
            DRM['scMass'] = self.scMass.to('kg')
            
        else:
            slew_mass_used = slewTimes[sInd]*self.defburnPortion*self.flowRate
            DRM['slew_dV'] = (slewTimes[sInd]*self.ao*self.defburnPortion).to('m/s')
            DRM['slew_mass_used'] = slew_mass_used.to('kg')
            self.scMass = self.scMass - slew_mass_used
            DRM['scMass'] = self.scMass.to('kg')
        
        return DRM
    
    def update_occulter_mass(self, TL, DRM, sInd, t_int, currentTime, skMode):
        """Updates the occulter wet mass in the Observatory module, and stores all 
        the occulter related values in the DRM array.
        
        Args:
            DRM (dict):
                Design Reference Mission, contains the results of one complete
                observation (detection and characterization)
            sInd (integer):
                Integer index of the star of interest
            t_int (astropy Quantity):
                Selected star integration time (for detection or characterization)
                in units of day
            skMode (string):
                Station keeping observing mode type ('det' or 'char')
                
        Returns:
            DRM (dict):
                Design Reference Mission, contains the results of one complete
                observation (detection and characterization)
        
        """
        
        assert skMode in ('det', 'char'), "Observing mode type must be 'det' or 'char'."
        
        # find disturbance forces on occulter
        dF_lateral, dF_axial = self.distForces(TL, sInd, currentTime)
        # decrement mass for station-keeping
        intMdot, mass_used, deltaV = self.mass_dec(dF_lateral, t_int)
        DRM[skMode + '_dV'] = deltaV.to('m/s')
        DRM[skMode + '_mass_used'] = mass_used.to('kg')
        DRM[skMode + '_dF_lateral'] = dF_lateral.to('N')
        DRM[skMode + '_dF_axial'] = dF_axial.to('N')
        # update spacecraft mass
        self.scMass = self.scMass - mass_used
        DRM['scMass'] = self.scMass.to('kg')
        
        return DRM
        
    
    
    
        
        
