from EXOSIMS.PlanetPopulation.KeplerLike1 import KeplerLike1
import warnings
import astropy
import astropy.units as u
import astropy.constants as const
import numpy as np
import os,inspect
from astropy.io.votable import parse
from astropy.time import Time
from EXOSIMS.util import statsFun 
import pkg_resources

class KnownRVPlanets(KeplerLike1):
    """Population consisting only of known RV planets.  Eccentricity and sma 
    distributions are taken from KeplerLike1 (Rayleigh and power law with 
    exponential decay, respectively).  Mass is sampled from power law and 
    radius is assumed to be calculated from mass via the physical model.
    
    The data file read in by this class also provides all of the information
    about the target stars, and so no StarCatalog object is needed (only the
    KnownRvPlanetsTargetList implementation).
    
    To download a new copy of the data file:
    1. Navigate to the IPAC exoplanet archive at http://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=planets
    2. Type 'radial' (minus quotes) in the 'Discovery Method' search box and
    hit enter.
    3. In the 'Download Table' menu select 'VOTable Format', 'Download all 
    Columns' and 'Download Currently Filtered Rows'.
    4. In the 'Download Table' menu  click 'Download Table'.
    
    Args: 
        \*\*specs: 
            user specified values
            
    Attributes: 
        smaknee (float):
            Location (in AU) of semi-major axis decay point (knee).
            Not an astropy quantity.
        esigma (float):
            Sigma value of Rayleigh distribution for eccentricity.
        rvplanetfilepath (string):
            Full path to RV planet votable file from IPAC. If None,
            assumes default file in PlanetPopulation directory of EXOSIMS.
        period (astropy Quantity array):
            Orbital period in units of day.  Error in perioderr.
        tper (astropy Time):
            Periastron time in units of jd.  Error in tpererr.
        
    
    Notes:  
    
    """

    def __init__(self, smaknee=30, esigma=0.25, rvplanetfilepath=None,planetfile = 'planets_2019.05.31_11.18.02.votable', **specs):
        
        KeplerLike1.__init__(self, smaknee=smaknee, esigma=esigma, **specs)
        
        #default file is ipac_2016-05-15
        if rvplanetfilepath is None:
            rvplanetfilepath = pkg_resources.resource_filename('EXOSIMS.PlanetPopulation',planetfile)

        if not os.path.isfile(rvplanetfilepath):
            raise IOError('RV Planet File %s Not Found.'%rvplanetfilepath)
        
        #read votable
        with warnings.catch_warnings():
            # warnings for IPAC votables are out of control 
            #   they are not moderated by pedantic=False
            #   they all have to do with units, which we handle independently anyway
            warnings.simplefilter('ignore', astropy.io.votable.exceptions.VOTableSpecWarning)
            warnings.simplefilter('ignore', astropy.io.votable.exceptions.VOTableChangeWarning)
            votable = parse(rvplanetfilepath)
        table = votable.get_first_table()
        data = table.array
        
        #we need mass info (either true or m\sin(i)) AND stellar mass AND
        #(sma OR period)
        keep = ~data['pl_bmassj'].mask & \
               ~data['st_mass'].mask & \
               (~data['pl_orbsmax'].mask | ~data['pl_orbper'].mask)
        data = data[keep]
        
        #save masses and determine which masses are *sin(I)
        self.mass = data['pl_bmasse'].data*u.earthMass
        self.masserr = data['pl_bmasseerr1'].data*u.earthMass
        self.msini = data['pl_bmassprov'].data == 'Msini'
        
        #store G x Ms product
        GMs = const.G*data['st_mass'].data*u.solMass # units of solar mass
        p2sma = lambda mu,T: ((mu*T**2/(4*np.pi**2))**(1/3.)).to('AU')
        sma2p = lambda mu,a: (2*np.pi*np.sqrt(a**3.0/mu)).to('day')

        #save semi-major axes
        self.sma = data['pl_orbsmax'].data*u.AU
        mask = data['pl_orbsmax'].mask
        T = data['pl_orbper'].data[mask]*u.day
        self.sma[mask] = p2sma(GMs[mask],T) 
        assert np.all(~np.isnan(self.sma)), 'sma has nan value(s)'
        #sma errors
        self.smaerr = data['pl_orbsmaxerr1'].data*u.AU
        mask = data['pl_orbsmaxerr1'].mask
        T = data['pl_orbper'].data[mask]*u.day
        Terr = data['pl_orbpererr1'].data[mask]*u.day
        self.smaerr[mask] = np.abs(p2sma(GMs[mask],T+Terr) - p2sma(GMs[mask],T))
        self.smaerr[np.isnan(self.smaerr)] = np.nanmean(self.smaerr)
        
        #save eccentricities
        self.eccen = data['pl_orbeccen'].data
        mask = data['pl_orbeccen'].mask
        _, etmp, _, _ = self.gen_plan_params(len(np.where(mask)[0]))
        self.eccen[mask] = etmp
        assert np.all(~np.isnan(self.eccen)), 'eccen has nan value(s)'
        #eccen errors
        self.eccenerr = data['pl_orbeccenerr1'].data
        mask = data['pl_orbeccenerr1'].mask
        self.eccenerr[mask | np.isnan(self.eccenerr)] = np.nanmean(self.eccenerr)
        
        #store available radii for using in KnownRVPlanetsTargetList
        self.radius = data['pl_radj'].data*u.jupiterRad
        self.radiusmask = data['pl_radj'].mask
        self.radiuserr1 = data['pl_radjerr1'].data*u.jupiterRad
        self.radiuserr2 = data['pl_radjerr2'].data*u.jupiterRad
        
        #save the periastron time and period 
        self.period = data['pl_orbper'].data*u.day
        mask = data['pl_orbper'].mask
        self.period[mask] = sma2p(GMs[mask], self.sma[mask])  
        assert np.all(~np.isnan(self.period)), 'period has nan value(s)'
        self.perioderr = data['pl_orbpererr1'].data*u.day
        mask = data['pl_orbpererr1'].mask
        a = data['pl_orbsmax'].data[mask]*u.AU
        aerr = data['pl_orbsmaxerr1'].data[mask]*u.AU
        self.perioderr[mask] = np.abs(sma2p(GMs[mask],a+aerr) - sma2p(GMs[mask],a))
        self.perioderr[np.isnan(self.perioderr)] = np.nanmean(self.perioderr)
        
        #if perisastron time missing, fill in random value
        dat = data['pl_orbtper'].data
        mask = data['pl_orbtper'].mask
        dat[mask] = np.random.uniform(low=np.nanmin(dat), high=np.nanmax(dat),
                size=np.where(mask)[0].size)
        self.tper =  Time(dat, format='jd')
        self.tpererr = data['pl_orbtpererr1'].data*u.day
        self.tpererr[data['pl_orbtpererr1'].mask] = np.nanmean(self.tpererr)
        
        #save host names
        self.hostname = data['pl_hostname'].filled().astype(str)
        
        #save the original data structure
        self.allplanetdata = data

    def gen_radius(self,n):
        """Generate planetary radius values in Earth radius
        
        Samples the mass distribution and then converts to radius using the physical model.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            Rp (astropy Quantity array):
                Planet radius values in units of Earth radius
        
        """
        n = self.gen_input_check(n)
        Mp = self.gen_mass(n)
        Rp = self.PlanetPhysicalModel.calc_radius_from_mass(Mp).to('earthRad')
        
        return Rp

    def gen_mass(self,n):
        """Generate planetary mass values in Earth mass
        
        The mass is determined by sampling the RV mass distribution from
        Cumming et al. 2010
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            Mp (astropy Quantity array):
                Planet mass values in units of Earth mass
        
        """
        n = self.gen_input_check(n)
        # unitless mass range
        Mpr = self.Mprange.to('earthMass').value
        Mp = statsFun.simpSample(self.dist_mass, n, Mpr[0], Mpr[1])*u.earthMass
        
        return Mp
