from EXOSIMS.PlanetPopulation.KeplerLike1 import KeplerLike1
import astropy.units as u
import astropy.constants as const
import numpy as np
import os,inspect
from astropy.io.votable import parse
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
        smaknee (numeric) 
            Location (in AU) of semi-major axis decay point.
        esigma (numeric)
            Sigma value of Rayleigh distribution for eccentricity.
        rvplanetfilepath (string)
            Full path to RV planet votable file from IPAC. If None,
            assumes default file in PlanetPopulation directory of EXOSIMS.
    
    Notes:  
    
    """

    def __init__(self, smaknee=30., esigma=0.25, rvplanetfilepath=None, **specs):
        
        specs['smaknee'] = smaknee
        specs['esigma'] = esigma
        KeplerLike1.__init__(self, **specs)
        
        #default file is IPAC_032216
        if rvplanetfilepath is None:
            classpath = os.path.split(inspect.getfile(self.__class__))[0]
            filename = 'RVplanets_ipac_032216.votable'
            rvplanetfilepath = os.path.join(classpath, filename)
        if not os.path.exists(rvplanetfilepath):
            raise IOError('RV Planet File %s Not Found.'%rvplanetfilepath)
        
        #read votable
        votable = parse(rvplanetfilepath)
        table = votable.get_first_table()
        self.table = table
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
        
        #self.radius = data['pl_radj']*const.R_jup
        
        #save host names
        self.hostname = data['pl_hostname'].filled().astype(str)
        
        #save the original data structure
        self.allplanetdata = data
        
        #define the mass distribution function (in Jupiter masses)
        self.massdist = lambda x: x**(-1.3)

    def gen_mass(self,n):
        """Generate planetary mass values in kg
        
        The mass is determined by sampling the RV mass distribution from
        Cumming et al. 2010
        
        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            Mp (astropy Quantity units kg)
        
        """
        n = self.gen_input_check(n)
        Mmin = (self.Mprange[0]/const.M_jup).decompose().value
        Mmax = (self.Mprange[1]/const.M_jup).decompose().value
        
        return statsFun.simpSample(self.massdist, n, Mmin, Mmax)*const.M_jup

    def gen_radius(self,n):
        """Generate planetary radius values in km
        
        Samples the mass distribution and then converts to radius using the physical model.
        
        Args:
            n (numeric):
                Number of samples to generate
                
        Returns:
            R (astropy Quantity units m)
        
        """
        n = self.gen_input_check(n)
        Mtmp = self.gen_mass(n)
        
        return self.PlanetPhysicalModel.calc_radius_from_mass(Mtmp)

