from EXOSIMS.Observatory.ObservatoryL2Halo import ObservatoryL2Halo
import numpy as np
import astropy.units as u
from astropy.time import Time
from scipy.integrate import solve_bvp
import astropy.constants as const
import scipy.optimize as optimize

EPS = np.finfo(float).eps


class SotoStarshade(ObservatoryL2Halo):
    """ StarShade Observatory class
    This class is implemented at L2 and contains all variables, functions, 
    and integrators to calculate occulter dynamics. 
    """
    
    def __init__(self, missionStart=60634., maxdVpcnt=0.02,
            setTOF=20,orbit_datapath=None,**specs): 

        ObservatoryL2Halo.__init__(self,**specs)

        self.setTOF = np.array([setTOF])        

        self.dV_tot = self.slewIsp*const.g0*np.log(self.scMass/self.dryMass)
        self.dVmax  = self.dV_tot * maxdVpcnt
    
    
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
        r_halo = self.haloPosition(t).to('au')
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
        
        def slewTime_objFun(self,dt):
            if dt.shape:
                dt = dt[0]
            
            return dt
        
        def slewTime_constraints(self,dt,TL,N1,N2,tA,percent):
        
            dV = self.calculate_dV(dt,TL,N1,N2,tA)
            dV_max = self.DV_tot * percent 
        
            return dV_max.value - dV
        
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
        
        def fuelUsage_constraints(self,dt,dt_min,dt_max):
        
            return dt_max - dt, dt - dt_min
        
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
    
    
        
    
    def calculate_slewTimes(self,TL,old_sInd,sInds,currentTime):
        
        sInds = np.arange(TL.nStars)
        dV = np.zeros(TL.nStars)  
        
        sd = np.arange(TL.nStars)
        
        finalTime = currentTime + self.setTOF[0]*u.d
            
        if old_sInd is None:
            old_sInd = np.random.randint(0,TL.nStars)

        for x in sInds:
            sd[x] = self.star_angularSep(TL,old_sInd,x,currentTime,finalTime)
            dV[x] = self.calculate_dV(self.setTOF,TL,old_sInd,x,currentTime)
            
                
        slewTimes = np.full_like(sd,self.setTOF[0])*u.d
            
        sInds = sInds[np.where(dV < self.dVmax)]
            
        return sd,slewTimes,sInds,dV
        
    
    def log_occulterResults(self,DRM,slewTimes,sInd,sd,dV):
        
        DRM['slew_time'] = slewTimes[sInd].to('day')
        DRM['slew_angle'] = sd[sInd].to('deg')
        
        dV = dV[sInd]*u.m/u.s
        slew_mass_used = self.scMass * ( np.exp(-dV/(self.Isp*const.g0)) - 1)
        DRM['slew_dV'] = dV
        DRM['slew_mass_used'] = slew_mass_used.to('kg')
        self.scMass = self.scMass - slew_mass_used
        DRM['scMass'] = self.scMass.to('kg')
        
        return DRM
    
    
