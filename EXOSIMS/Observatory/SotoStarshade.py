from EXOSIMS.Observatory.ObservatoryL2Halo import ObservatoryL2Halo
from EXOSIMS.Prototypes.TargetList import TargetList
import numpy as np
import astropy.units as u
from scipy.integrate import solve_bvp
import astropy.constants as const
import hashlib
import scipy.optimize as optimize
import scipy.interpolate as interp
import time
import os
try:
    import cPickle as pickle
except:
    import pickle

EPS = np.finfo(float).eps


class SotoStarshade(ObservatoryL2Halo):
    """ StarShade Observatory class
    This class is implemented at L2 and contains all variables, functions, 
    and integrators to calculate occulter dynamics. 
    """
    
    def __init__(self,orbit_datapath=None,f_nStars=10,**specs): 

        ObservatoryL2Halo.__init__(self,**specs)  
        self.f_nStars = int(f_nStars)
        
        # instantiating fake star catalog, used to generate good dVmap
        fTL = TargetList(**{"ntargs":self.f_nStars,'modules':{"StarCatalog": "FakeCatalog", \
                    "TargetList":" ","OpticalSystem": "Nemati", "ZodiacalLight": "Stark", "PostProcessing": " ", \
                    "Completeness": " ","BackgroundSources": "GalaxiesFaintStars", "PlanetPhysicalModel": " ", \
                    "PlanetPopulation": "KeplerLike1"}, "scienceInstruments": [{ "name": "imager"}],  \
                    "starlightSuppressionSystems": [{ "name": "HLC-565"}]   })
        
        f_sInds = np.arange(0,fTL.nStars)
        dV,ang,dt = self.generate_dVMap(fTL,0,f_sInds,self.equinox[0])
        
        # pick out unique angle values
        ang, unq = np.unique(ang, return_index=True)
        dV = dV[:,unq]
        
        #create dV 2D interpolant
        self.dV_interp  = interp.interp2d(dt,ang,dV.T,kind='linear')


    def generate_dVMap(self,TL,old_sInd,sInds,currentTime):
        """Creates dV map for an occulter slewing between targets.
        
        This method returns a 2D array of the dV needed for an occulter
        to slew between all the different stars on the target list. The dV
        map is calculated relative to a reference star and the stars are 
        ordered by their angular separation from the reference star (X-axis).
        The Y-axis represents the time of flight ("slew time") for a
        trajectory between two stars. 
        
        Args:
            TL (TargetList module):
                TargetList class object
            old_sInd (integer):
                Integer index of the last star of interest
            sInds (integer ndarray):
                Integer indices of the stars of interest
            currentTime (astropy Time array):
                Current absolute mission time in MJD
                
        Returns:
            tuple:
            dVMap (float ndarray):
                Map of dV needed to transfer from a reference star to another. 
                Each ordered pair (psi,t) of the dV map corresponds to a 
                trajectory to a star an angular distance psi away with flight
                time of t. units of (m/s)
            angles (float ndarray):
                Range of angles (in deg) used in dVMap as the X-axis
            dt (float ndarray):
                Range of slew times (in days) used in dVMap as the Y-axis
        """
        
        # generating hash name
        filename  = 'dVMap_'
        extstr = ''
        extstr += '%s: ' % 'occulterSep'  + str(getattr(self,'occulterSep'))  + ' '
        extstr += '%s: ' % 'period_halo'  + str(getattr(self,'period_halo'))  + ' '
        extstr += '%s: ' % 'f_nStars'  + str(getattr(self,'f_nStars'))  + ' '
        ext = hashlib.md5(extstr.encode('utf-8')).hexdigest()
        filename += ext
        dVpath = os.path.join(self.cachedir, filename + '.dVmap')
        
        # initiating slew Times for starshade
        dt = np.arange(self.occ_dtmin.value,self.occ_dtmax.value,1)
        
        # angular separation of stars in target list from old_sInd
        ang =  self.star_angularSep(TL, old_sInd, sInds, currentTime) 
        sInd_sorted = np.argsort(ang)
        angles  = ang[sInd_sorted].to('deg').value
        
        # initializing dV map
        dVMap   = np.zeros([len(dt),len(sInds)])
        
        #checking to see if map exists or needs to be calculated
        if os.path.exists(dVpath):
            # dV map already exists for given parameters
            self.vprint('Loading cached Starshade dV map file from %s' % dVpath)
            try:
                with open(dVpath, "rb") as ff:
                    A = pickle.load(ff)
            except UnicodeDecodeError:
                with open(dVpath, "rb") as ff:
                    A = pickle.load(ff,encoding='latin1')
            self.vprint('Starshade dV Map loaded from cache.')
            dVMap = A['dVMap']
        else:
            self.vprint('Cached Starshade dV map file not found at "%s".' % dVpath)
            # looping over all target list and desired slew times to generate dV map
            self.vprint('Starting dV calculations for %s stars.' % TL.nStars)
            tic = time.clock()
            for i in range(len(dt)):
                dVMap[i,:] = self.impulsiveSlew_dV(dt[i],TL,old_sInd,sInd_sorted,currentTime) #sorted
                if not i % 5: self.vprint('   [%s / %s] completed.' % (i,len(dt)))
            toc = time.clock()
            B = {'dVMap':dVMap}
            with open(dVpath, 'wb') as ff:
                pickle.dump(B, ff)
            self.vprint('dV map computation completed in %s seconds.' % (toc-tic))
            self.vprint('dV Map array stored in %r' % dVpath)
            
        return dVMap,angles,dt


    def starshadeVelocity(self,TL,sInd,currentTime,tRange=[0],frame='inertial'):
        """ Calculates starshade and telescope velocities in R- or I-frames during stationkeeping
        """
        
        s = self.convertPos_to_canonical(self.occulterSep)
        
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
        x,y,z    = np.array([self.convertPos_to_canonical(haloPos[:,n]) for n in range(3)])
        RdrT_R   = np.array([self.convertVel_to_canonical(haloVel[:,n]) for n in range(3)])
        dx,dy,dz = RdrT_R

        rS_R = np.zeros([len(tRange),3])

        rS_R[:,0] = s*np.sin(gam)*np.cos(nu) + x
        rS_R[:,1] = s*np.sin(gam)*np.sin(nu) + y
        rS_R[:,2] = s*np.cos(gam) + z
        
        RdrS_R = np.zeros([len(tRange),3])
        
        RdrS_R[:,0] = s*dgam*np.cos(gam)*np.cos(nu) - s*dnu*np.sin(gam)*np.sin(nu) + dx
        RdrS_R[:,1] = s*dgam*np.cos(gam)*np.sin(nu) + s*dnu*np.sin(gam)*np.cos(nu) + dy
        RdrS_R[:,2] = -s*dgam*np.sin(gam) + dz

        if frame == 'inertial':

            rS_I = np.zeros([len(tRange),3])
            IdrS_I = np.zeros([len(tRange),3])
            IdrT_I = np.zeros([len(tRange),3])

            rS_I[:,0] = rS_R[:,0]*np.cos(t) - rS_R[:,1]*np.sin(t)
            rS_I[:,1] = rS_R[:,0]*np.sin(t) + rS_R[:,1]*np.cos(t)
            rS_I[:,2] = rS_R[:,2]
            
            ds1 = RdrS_R[:,0] - s*np.sin(gam)*np.sin(nu) - y
            ds2 = RdrS_R[:,1] + s*np.sin(gam)*np.cos(nu) + x
            ds3 = RdrS_R[:,2]           
            
            IdrS_I[:,0] = ds1*np.cos(t) - ds2*np.sin(t)
            IdrS_I[:,1] = ds1*np.sin(t) + ds2*np.cos(t)
            IdrS_I[:,2] = ds3

            dt1 = dx - y
            dt2 = dy + x
            dt3 = dz

            IdrT_I[:,0] = dt1*np.cos(t) - dt2*np.sin(t)
            IdrT_I[:,1] = dt1*np.sin(t) + dt2*np.cos(t)
            IdrT_I[:,2] = dt3
            
            return rS_I,IdrS_I,IdrT_I
        
        else:
            return rS_R,RdrS_R,RdrT_R
    
    
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
            float 1x6 ndarray:
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
            float nx6 ndarray:
                State vectors in rotating frame in normalized units
        """
        
        self_rA,self_vA,telA = self.starshadeVelocity(TL,nA,tA,frame='rot')
        self_rB,self_vB,telB = self.starshadeVelocity(TL,nB,tB,frame='rot')
           
        vA = self_vA[0]
        vB = self_vB[0]
        
        #position vector of occulter in heliocentric frame
        self.rA = self_rA[0]
        self.rB = self_rB[0]
        
        a = ((np.mod(tA.value,self.equinox[0].value)*u.d)).to('yr').value * (2*np.pi)
        b = ((np.mod(tB.value,self.equinox[0].value)*u.d)).to('yr').value * (2*np.pi)
        
        #running shooting algorithm
        t = np.linspace(a,b,2)
        
        sG = np.array([  [ self.rA[0],self.rB[0] ], \
                         [ self.rA[1],self.rB[1] ], \
                         [ self.rA[2],self.rB[2] ], \
                         [      vA[0],     vB[0] ], \
                         [      vA[1],     vB[1] ], \
                         [      vA[2],     vB[2] ] ])            
            
        sol = solve_bvp(self.equationsOfMotion_CRTBP,self.boundary_conditions,t,sG,tol=1e-10)
        
        s = sol.y.T
        t_s = sol.x
            
        return s,t_s
    
    
    def calculate_dV(self, TL, old_sInd, sInds, sd, slewTimes, tmpCurrentTimeAbs): 
        """Finds the change in velocity needed to transfer to a new star line of sight
        
        This method sums the total delta-V needed to transfer from one star
        line of sight to another. It determines the change in velocity to move from
        one station-keeping orbit to a transfer orbit at the current time, then from
        the transfer orbit to the next station-keeping orbit at currentTime + dt.
        Station-keeping orbits are modeled as discrete boundary value problems.
        This method can handle multiple indeces for the next target stars and calculates
        the dVs of each trajectory from the same starting star.
        
        Args:
            dt (float 1x1 ndarray):
                Number of days corresponding to starshade slew time
            TL (float 1x3 ndarray):
                TargetList class object
            nA (integer):
                Integer index of the current star of interest
            N  (integer):
                Integer index of the next star(s) of interest
            tA (astropy Time array):
                Current absolute mission time in MJD
                
        Returns:
            float nx6 ndarray:
                State vectors in rotating frame in normalized units
        """
        
        if old_sInd is None:
            dV = np.zeros(slewTimes.shape)
        else:
            dV = np.zeros(slewTimes.shape)
            badSlews_i,badSlew_j = np.where(slewTimes.value <  self.occ_dtmin.value)
            for i in range(len(sInds)):
                for t in range(len(slewTimes.T)):
                    dV[i,t] = self.dV_interp(slewTimes[i,t],sd[i].to('deg')) 
            dV[badSlews_i,badSlew_j] = np.Inf
        
        return dV*u.m/u.s
    
    
    def impulsiveSlew_dV(self,dt,TL,nA,N,tA):  
        """Finds the change in velocity needed to transfer to a new star line of sight
        
        This method sums the total delta-V needed to transfer from one star
        line of sight to another. It determines the change in velocity to move from
        one station-keeping orbit to a transfer orbit at the current time, then from
        the transfer orbit to the next station-keeping orbit at currentTime + dt.
        Station-keeping orbits are modeled as discrete boundary value problems.
        This method can handle multiple indeces for the next target stars and calculates
        the dVs of each trajectory from the same starting star.
        
        Args:
            dt (float 1x1 ndarray):
                Number of days corresponding to starshade slew time
            TL (float 1x3 ndarray):
                TargetList class object
            nA (integer):
                Integer index of the current star of interest
            N  (integer):
                Integer index of the next star(s) of interest
            tA (astropy Time array):
                Current absolute mission time in MJD
                
        Returns:
            float nx6 ndarray:
                State vectors in rotating frame in normalized units
        """
        
        if dt.shape:
            dt = dt[0]
            
        if nA is None:
            dV = np.zeros(len(N))
        else:
            # if only calculating one trajectory, this allows loop to run
            if N.size is 1:
                N  = np.array([N])
            
            # time to reach star B's line of sight
            tB = tA + dt*u.d
            
            # initializing arrays for BVP state solutions
            sol_slew = np.zeros([2,len(N),6])
            t_sol    = np.zeros([2,len(N)])
            v_skB    = np.zeros([1,len(N),3])
            for x in range(len(N)):   
                # simulating slew trajectory from star A at tA to star B at tB
                sol,t  = self.send_it(TL,nA,N[x],tA,tB)
                sol_slew[:,x,:] = np.array([sol[0],sol[-1]])
                t_sol[:,x]      = np.array([t[0],t[-1]])
            
                r_skBx,v_skBx,telB = self.starshadeVelocity(TL,N[x],tB,frame='rot')
                v_skB[:,x,:] = v_skBx
            
            r_skA,v_skA,telA = self.starshadeVelocity(TL,nA,tA,frame='rot')
            
            # starshade velocities at both endpoints of the slew trajectory
            v_slewA = sol_slew[ 0,:,3:6]
            v_slewB = sol_slew[-1,:,3:6]
            
            dvA = v_slewA - v_skA.reshape(1,1,3)
            dvB = v_slewB - v_skB
            
            if dvA.size==1:
                dV = self.convertVel_to_dim( np.linalg.norm(dvA) + np.linalg.norm(dvB)  )
                
            else:
                dV = self.convertVel_to_dim( np.linalg.norm(dvA[0],axis=1) + np.linalg.norm(dvB[0],axis=1)  )
                
        return dV.to('m/s')  
    
    
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
            tuple:
            opt_slewTime (float):
                Optimal slew time in days for starshade transfer to a new line of sight
            opt_dV (float):
                Optimal total change in velocity in m/s for starshade line of sight transfer
        
        """
        
        def slewTime_objFun(dt):
            if dt.shape:
                dt = dt[0]
            
            return dt
        
        def slewTime_constraints(dt,TL,nA,nB,tA):
            dV = self.calculate_dV(dt,TL,nA,nB,tA)
            dV_max = self.dVmax
        
            return (dV_max - dV).value, dt - 1
        
        dt_guess=20   
        Tol=1e-3
        
        t0 = [dt_guess]
        
        res = optimize.minimize(slewTime_objFun,t0,method='COBYLA',
                        constraints={'type': 'ineq', 'fun': slewTime_constraints,'args':([TL,nA,nB,tA])},
                        tol=Tol,options={'disp': False})
                        
        opt_slewTime = res.x
        opt_dV       = self.calculate_dV(opt_slewTime,TL,nA,nB,tA)
        
        return opt_slewTime,opt_dV.value
    
    
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
            tuple:
            opt_slewTime (float):
                Optimal slew time in days for starshade transfer to a new line of sight
            opt_dV (float):
                Optimal total change in velocity in m/s for starshade line of sight transfer
        
        """
        
        def fuelUsage_objFun(dt,TL,nA,N,tA):
            dV = self.calculate_dV(dt,TL,nA,N,tA)
            return dV.value
        
        def fuelUsage_constraints(dt,dt_min,dt_max):
            return dt_max - dt, dt - dt_min
        
        dt_guess=20
        dt_min=1
        dt_max=45        
        Tol=1e-5
        
        t0 = [dt_guess]

        res = optimize.minimize(fuelUsage_objFun,t0,method='COBYLA',args=(TL,nA,nB,tA),
                        constraints={'type': 'ineq', 'fun': fuelUsage_constraints,'args':([dt_min,dt_max])},
                        tol=Tol,options={'disp': False})
        opt_slewTime = res.x
        opt_dV   = res.fun
        
        return opt_slewTime,opt_dV


    def calculate_slewTimes(self,TL, old_sInd, sInds, sd, obsTimes, tmpCurrentTimeAbs):
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
            tuple:
            sInds (integer):
                Integer indeces of the star of interest
            sd (astropy Quantity):
                Angular separation between stars in rad
            slewTimes (astropy Quantity):
                Time to transfer to new star line of sight in units of days
            dV (astropy Quantity):
                Delta-V used to transfer to new star line of sight in units of m/s
        """
        if old_sInd is None:
            slewTimes = np.zeros(len(sInds))*u.d
        else:        
            obsTimeRangeNorm = (obsTimes - tmpCurrentTimeAbs).value
            slewTimes = obsTimeRangeNorm[0,:]*u.d
            
        return slewTimes

    
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
            dict:
                Design Reference Mission dicitonary, contains the results of one complete
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
    
