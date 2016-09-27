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


class KnownRVPlanets(KeplerLike1):
    """
    Population consisting only of known RV planets.  Eccentricity and sma 
    distributions are taken from the same as in KeplerLike1 (Rayleigh and 
    power law with exponential decay, respectively).  Mass is sampled from 
    power law and radius is assumed to be calculated from mass via the 
    physical model.
    
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

    def __init__(self, smaknee=30, esigma=0.25, rvplanetfilepath=None, **specs):
        
        specs['smaknee'] = float(smaknee)
        specs['esigma'] = float(esigma)
        KeplerLike1.__init__(self, **specs)
        
        #default file is ipac_2016-05-15
        if rvplanetfilepath is None:
            classpath = os.path.split(inspect.getfile(self.__class__))[0]
            filename = 'RVplanets_ipac_2016-05-15.votable'
            rvplanetfilepath = os.path.join(classpath, filename)
        if not os.path.exists(rvplanetfilepath):
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
        
        #we need mass info (either true or m\sin(i)) AND
        #(sma OR (period AND stellar mass))
        keep = ~data['pl_bmassj'].mask & \
               (~data['pl_orbsmax'].mask | \
               (~data['pl_orbper'].mask & ~data['st_mass'].mask))
        data = data[keep]
        
        #save masses and determine which masses are *sin(I)
        self.mass = data['pl_bmasse'].data*const.M_earth
        self.masserr = data['pl_bmasseerr1'].data*const.M_earth
        self.msini = data['pl_bmassprov'].data == 'Msini'
        
        #save semi-major axes
        self.sma = data['pl_orbsmax'].data*u.AU
        mask = data['pl_orbsmax'].mask
        Ms = data['st_mass'].data[mask]*const.M_sun # units of kg
        T = data['pl_orbper'].data[mask]*u.d
        self.sma[mask] = ((const.G*Ms*T**2 / (4*np.pi**2))**(1/3.)).to('AU')
        assert np.all(~np.isnan(self.sma)), 'sma has nan value(s)'
        #sma errors
        self.smaerr = data['pl_orbsmaxerr1'].data*u.AU
        mask = data['pl_orbsmaxerr1'].mask
        Ms = data['st_mass'].data[mask]*const.M_sun # units of kg
        T = data['pl_orbpererr1'].data[mask]*u.d
        self.smaerr[mask] = ((const.G*Ms*T**2 / (4*np.pi**2))**(1/3.)).to('AU')
        self.smaerr[np.isnan(self.smaerr)] = np.nanmean(self.smaerr)
        
        #save eccentricities
        self.eccen = data['pl_orbeccen'].data
        mask = data['pl_orbeccen'].mask
        self.eccen[mask] = self.gen_eccen(len(np.where(mask)[0]))
        assert np.all(~np.isnan(self.eccen)), 'eccen has nan value(s)'
        #eccen errors
        self.eccenerr = data['pl_orbeccenerr1'].data
        mask = data['pl_orbeccenerr1'].mask
        self.eccenerr[mask | np.isnan(self.eccenerr)] = np.nanmean(self.eccenerr)
        
        #store available radii for using in KnownRVPlanetsTargetList
        self.radius = data['pl_radj'].data*const.R_jup
        self.radiusmask = data['pl_radj'].mask
        self.radiuserr1 = data['pl_radjerr1'].data*const.R_jup
        self.radiuserr2 = data['pl_radjerr2'].data*const.R_jup

        #save the periastron time and period 
        tmp = data['pl_orbper'].data*u.d
        tmp[data['pl_orbper'].mask] = np.sqrt((4*np.pi**2*self.sma[data['pl_orbper'].mask]**3)\
                /(const.G*data['st_mass'].data[data['pl_orbper'].mask]*const.M_sun)).decompose().to(u.d)
        self.period = tmp
        self.perioderr = data['pl_orbpererr1'].data*u.d
        mask = data['pl_orbpererr1'].mask
        self.perioderr[mask] = np.nanmean(self.perioderr)

        #if perisastron time missing, fill in random value
        tmp = data['pl_orbtper'].data
        tmp[data['pl_orbtper'].mask] = np.random.uniform(low=np.nanmin(tmp),high=np.nanmax(tmp),\
                size=np.where(data['pl_orbtper'].mask)[0].size)
        self.tper =  Time(tmp,format='jd')
        self.tpererr = data['pl_orbtpererr1'].data*u.d
        self.tpererr[data['pl_orbtpererr1'].mask] = np.nanmean(self.tpererr)
        
        #save host names
        self.hostname = data['pl_hostname'].filled().astype(str)
        
        #save the original data structure
        self.allplanetdata = data
        
        #define the mass distribution function (in Jupiter masses)
        self.massdist = lambda x: x**(-1.3)

    def gen_radius(self,n):
        """Generate planetary radius values in km
        
        Samples the mass distribution and then converts to radius using the physical model.
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            Rp (astropy Quantity array):
                Planet radius in units of km
        
        """
        n = self.gen_input_check(n)
        Mtmp = self.gen_mass(n)
        Rp = self.PlanetPhysicalModel.calc_radius_from_mass(Mtmp)
        
        return Rp

    def gen_mass(self,n):
        """Generate planetary mass values in kg
        
        The mass is determined by sampling the RV mass distribution from
        Cumming et al. 2010
        
        Args:
            n (integer):
                Number of samples to generate
                
        Returns:
            Mp (astropy Quantity array):
                Planet mass in units of kg
        
        """
        n = self.gen_input_check(n)
        Mmin = (self.Mprange[0]/const.M_jup).decompose().value
        Mmax = (self.Mprange[1]/const.M_jup).decompose().value
        Mp = statsFun.simpSample(self.massdist, n, Mmin, Mmax)*const.M_jup
        
        return Mp
