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
    
    def __init__(self, missionStart=60634.,orbit_datapath=None,**specs): 

        ObservatoryL2Halo.__init__(self,**specs)
    
    def boundary_conditions(self,rA,rB):
        """Creates boundary conditions for solving a boundary value problem
        
        This method returns the boundary conditions for the starshade transfer
        trajectory between the lines of sight of two different stars. Point A
        corresponds to the starshade alignment with star A; Point B, with star B.
        
        
        Args:
            rA (float 1x3 ndarray):
                Starshade position vector aligned with current star of interest
            rB (float 1x3 ndarray):
                Starshade position vector aligned with next star of interest
                
        Returns:
            BC (float 1x6 ndarray):
                Star position vector in rotating frame in units of AU
        
        """
    
        BC1 = rA[0] - self.rA[0]
        BC2 = rA[1] - self.rA[1]
        BC3 = rA[2] - self.rA[2]
        
        BC4 = rB[0] - self.rB[0]
        BC5 = rB[1] - self.rB[1]
        BC6 = rB[2] - self.rB[2]
        
        BC = np.array([BC1,BC2,BC3,BC4,BC5,BC6])
        
        return BC
    
    def send_it(self,TL,nA,nB,tA,tB):
        """Solves boundary value problem between starshade star alignments
        
        This method solves the boundary value problem for starshade star alignments
        with two given stars at times tA and tB. It uses scipy's solve_bvp method.
        
        Args:
            TL (float 1x3 ndarray):
                TargetList class object
            nA (integer):
                Integer index of the current star of interest
            nB (integer):
                Integer index of the next star of interest
            tA (astropy Time array):
                Current absolute mission time in MJD
            tB (astropy Time array):
                Absolute mission time for next star alignment in MJD
                
        Returns:
            s (float nx6 ndarray):
                State vectors in rotating frame in normalized units
        """
        
        angle,uA,uB,r_tscp = self.star_angularSep(TL,nA,nB,tA,tB)
        
        #position vector of occulter in heliocentric frame
        self.rA = uA*self.occulterSep.to('au').value + r_tscp[ 0]
        self.rB = uB*self.occulterSep.to('au').value + r_tscp[-1]
        
        a = ((np.mod(tA.value,self.equinox.value)*u.d)).to('yr') / u.yr * (2*np.pi)
        b = ((np.mod(tB.value,self.equinox.value)*u.d)).to('yr') / u.yr * (2*np.pi)
        
        #running shooting algorithm
        t = np.linspace(a,b,2)
        
        guess = (self.rB-self.rA)/(b-a)
        sG = np.array([np.full_like(t,self.rA[0]),np.full_like(t,self.rA[1]),np.full_like(t,self.rA[2]), 
                       np.full_like(t,guess[0]),np.full_like(t,guess[1]),np.full_like(t,guess[2])])               
            
        sol = solve_bvp(self.equationsOfMotion_CRTBP,self.boundary_conditions,t,sG,tol=1e-10)
        
        s = sol.y.T
        
        return s
    
    def calculate_dV(self,dt,TL,nA,nB,tA):
        """Finds the change in velocity needed to transfer to a new star line of sight
        
        This method sums the total delta-V needed to transfer from one star
        line of sight to another. It determines the change in velocity to move from
        one station-keeping orbit to a transfer orbit at the current time, then from
        the transfer orbit to the next station-keeping orbit at currentTime + dt.
        Station-keeping orbits are modeled as discrete boundary value problems.
        
        Args:
            dt (float 1x1 ndarray):
                Number of days corresponding to starshade slew time
            TL (float 1x3 ndarray):
                TargetList class object
            nA (integer):
                Integer index of the current star of interest
            nB (integer):
                Integer index of the next star of interest
            tA (astropy Time array):
                Current absolute mission time in MJD
                
        Returns:
            dV (float nx6 ndarray):
                State vectors in rotating frame in normalized units
        """
        
        if dt.shape:
            dt = dt[0]
            
        tB = tA + dt*u.d
        
        
        sol = self.send_it(TL,nA,nB,tA,tB)
        
        v_occulter_A = sol[ 0,3:6]*u.AU/u.year*(2*np.pi) #velocity after leaving star A
        v_occulter_B = sol[-1,3:6]*u.AU/u.year*(2*np.pi) #velocity arriving at star B
        
        #portions of the station-keeping trajectories required for stars A and B respectively
        s0 = self.send_it(TL,nA,nA,tA - 15*u.min,tA) 
        sF = self.send_it(TL,nB,nB,tB,tB + 15*u.min)
        
        v0 = s0[-1,3:6]*u.AU/u.year*(2*np.pi) #velocity needed to maintain constant distance with star A
        vF = sF[ 0,3:6]*u.AU/u.year*(2*np.pi) #velocity needed to maintain constant distance with star B
        
        dvA = (v_occulter_A-v0).to('m/s')
        dvB = (v_occulter_B-vF).to('m/s')
        
        dv = np.linalg.norm(dvA) + np.linalg.norm(dvB)
        
        return dv
    
    def minimize_slewTimes(self,TL,nA,nB,tA):
        """Minimizes the slew time for a starshade transferring to a new star line of sight
        
        This method uses scipy's optimization module to minimize the slew time for
        a starshade transferring between one star's line of sight to another's under 
        the constraint that the total change in velocity cannot exceed more than a 
        certain percentage of the total fuel on board the starshade. 
        
        Args:
            TL (float 1x3 ndarray):
                TargetList class object
            nA (integer):
                Integer index of the current star of interest
            nB (integer):
                Integer index of the next star of interest
            tA (astropy Time array):
                Current absolute mission time in MJD
                
        Returns:
            opt_slewTime (float):
                Optimal slew time in days for starshade transfer to a new line of sight
            opt_dV (float):
                Optimal total change in velocity in m/s for starshade line of sight transfer
        
        """
        
        def slewTime_objFun(self,dt):
            if dt.shape:
                dt = dt[0]
            
            return dt
        
        def slewTime_constraints(self,dt,TL,nA,nB,tA,percent):
            dV = self.calculate_dV(dt,TL,nA,nB,tA)
            dV_max = self.DV_tot * percent 
        
            return dV_max.value - dV
        
        dt_guess=20
        percent=0.05        
        Tol=1e-3
        
        t0 = [dt_guess]
        
        res = optimize.minimize(self.slewTime_objFun,t0,method='COBYLA',
                        constraints={'type': 'ineq', 'fun': self.slewTime_constraints,'args':([TL,nA,nB,tA,percent])},
                        tol=Tol,options={'disp': False})
                        
        opt_slewTime = res.x
        opt_dV       = self.calculate_dV(opt_slewTime,TL,nA,nB,tA)
        
        return opt_slewTime,opt_dV
    
    def minimize_fuelUsage(self,TL,nA,nB,tA):
        """Minimizes the fuel usage of a starshade transferring to a new star line of sight
        
        This method uses scipy's optimization module to minimize the fuel usage for
        a starshade transferring between one star's line of sight to another's. The 
        total slew time for the transfer is bounded with some dt_min and dt_max.
        
        Args:
            TL (float 1x3 ndarray):
                TargetList class object
            nA (integer):
                Integer index of the current star of interest
            nB (integer):
                Integer index of the next star of interest
            tA (astropy Time array):
                Current absolute mission time in MJD
                
        Returns:
            opt_slewTime (float):
                Optimal slew time in days for starshade transfer to a new line of sight
            opt_dV (float):
                Optimal total change in velocity in m/s for starshade line of sight transfer
        
        """
        
        def fuelUsage_constraints(self,dt,dt_min,dt_max):
            return dt_max - dt, dt - dt_min
        
        dt_guess=20
        dt_min=1
        dt_max=40        
        Tol=1e-3
        
        t0 = [dt_guess]

        res = optimize.minimize(self.calculate_dV,t0,method='COBYLA',
                        constraints={'type': 'ineq', 'fun': self.fuelUsage_constraints,'args':([dt_min,dt_max])},
                        tol=Tol,args=(TL,nA,nB,tA),options={'disp': False})
        opt_slewTime = res.x
        opt_dV   = res.fun
        
        return opt_slewTime,opt_dV

    def calculate_slewTimes(self,TL,old_sInd,sInds,currentTime):
        """Finds slew times and separation angles between target stars
        
        This method determines the slew times of an occulter spacecraft needed
        to transfer from one star's line of sight to all others in a given 
        target list.
        
        Args:
            TL (TargetList module):
                TargetList class object
            old_sInd (integer):
                Integer index of the most recently observed star
            sInds (integer):
                Integer indeces of the star of interest
            currentTime (astropy Time):
                Current absolute mission time in MJD
                
        Returns:
            sInds (integer):
                Integer indeces of the star of interest
            sd (astropy Quantity):
                Angular separation between stars in rad
            slewTimes (astropy Quantity):
                Time to transfer to new star line of sight in units of days
            dV (astropy Quantity):
                Delta-V used to transfer to new star line of sight in units of m/s
        """
        
        sd = np.zeros(TL.nStars)*u.deg
            
        if old_sInd is None:
            sd = np.array([np.radians(0)]*TL.nStars)*u.rad
            slewTimes = np.zeros(TL.nStars)*u.d
        else:
            # position vector of previous target star
            r_old = TL.starprop(old_sInd, currentTime)[0]
            u_old = r_old.value/np.linalg.norm(r_old)
            # position vector of new target stars
            r_new = TL.starprop(sInds, currentTime)
            u_new = (r_new.value.T/np.linalg.norm(r_new, axis=1)).T
            # angle between old and new stars
            sd = np.arccos(np.clip(np.dot(u_old, u_new.T), -1, 1))*u.rad
    
            slewTimes = self.constTOF[0].value*np.ones(TL.nStars)*u.d
            
        return sd,slewTimes
    
    def filter_dV(self,TL,old_sInd,sInds,currentTime):
        """Calculates and filters dV needed to reach line of sight of every star in sInds
        
        This method determines the changes in velocity an occulter spacecraft needs
        to make to transfer from one star's line of sight to all others in a given 
        target list. Trajectories with too large of a delta-V are filtered out. 
        
        Args:
            TL (TargetList module):
                TargetList class object
            old_sInd (integer):
                Integer index of the most recently observed star
            sInds (integer):
                Integer indeces of the star of interest
            currentTime (astropy Time):
                Current absolute mission time in MJD
                
        Returns:
            sInds (integer):
                Integer indeces of the star of interest
            dV (astropy Quantity):
                Delta-V used to transfer to new star line of sight in units of m/s
        """

        dV = np.zeros(TL.nStars)*u.m/u.s
        
        if old_sInd is not None:
            for x in sInds:
                dV[x] = self.calculate_dV(self.constTOF.value,TL,old_sInd,x,currentTime)*u.m/u.s
                
            sInds = sInds[np.where(dV.value < self.dVmax.value)]
        
        return sInds,dV
        
    def log_occulterResults(self,DRM,slewTimes,sInd,sd,dV):
        """Updates the given DRM to include occulter values and results
        
        Args:
            DRM (dict):
                Design Reference Mission, contains the results of one complete
                observation (detection and characterization)
            slewTimes (astropy Quantity):
                Time to transfer to new star line of sight in units of days
            sInd (integer):
                Integer index of the star of interest
            sd (astropy Quantity):
                Angular separation between stars in rad
            dV (astropy Quantity):
                Delta-V used to transfer to new star line of sight in units of m/s
                
        Returns:
            DRM (dict):
                Design Reference Mission, contains the results of one complete
                observation (detection and characterization)
        """
        
        DRM['slew_time'] = slewTimes.to('day')
        DRM['slew_angle'] = sd.to('deg')
        
        dV = dV.to('m/s')
        slew_mass_used = self.scMass * ( 1 - np.exp(-dV.value/(self.slewIsp.value*const.g0.value)))
        DRM['slew_dV'] = dV
        DRM['slew_mass_used'] = slew_mass_used.to('kg')
        self.scMass = self.scMass - slew_mass_used
        DRM['scMass'] = self.scMass.to('kg')
        
        return DRM
    
    
