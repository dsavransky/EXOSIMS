from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import pickle
import numpy as np
import pandas as pd
import astropy
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
import scipy.interpolate as interpolate
import scipy.integrate as integrate
from EXOSIMS.util import statsFun
import pkg_resources
import os,inspect


class PlandbPlanets(PlanetPopulation):
    """Population of Known RV planets from the plandb database. This module is created with the same 
    framework as that of the existing Known RV module."""

    def __init__(self, smaknee=30, esigma=0.25, planetfilepath=None, planetfile = 'orbfits_2022-03.p', **specs):

        PlanetPopulation.__init__(self, smaknee=smaknee, esigma=esigma, **specs)

        if planetfilepath is None:
            planetfilepath = pkg_resources.resource_filename('EXOSIMS.PlanetPopulation',planetfile)

        if not os.path.isfile(planetfilepath):
                raise IOError('Planet File %s Not Found.'%planetfilepath)

        with open(planetfilepath,'rb') as x:
            db = pickle.load(x)

        #selecting the default fit of the orbit fits from the pickle file
        data = db.loc[db['default_fit']==1]

        #smaknee
        self.smaknee = float(smaknee)
        ar = self.arange.to('AU').value
        # sma dist without normalization
        tmp_dist_sma = lambda x, s0 = self.smaknee: x**(-0.62)*np.exp(-(x/s0)**2)
        self.smanorm = integrate.quad(tmp_dist_sma, ar[0], ar[1])[0]
        
        #esigma, norm for eccentricity Rayleigh distribution
        self.esigma = float(esigma)
        er = self.erange
        self.enorm = np.exp(-er[0]**2/(2.*self.esigma**2)) \
                    - np.exp(-er[1]**2/(2.*self.esigma**2))

        #filtering the data based on the availability of mass info, stellar mass, and (sma or period)
        keep = ~data["pl_bmasse"].isna() & \
               ~data["st_mass"].isna() & \
                (~data["pl_orbsmax"].isna() | ~data["pl_orbper"].isna())
        data = data[keep]

        #storing the planet paramaters - mass, evaluating which data are Msini
        self.mass = data["pl_bmasse"].values*u.earthMass
        self.masserr = data["pl_bmasseerr1"].values*u.earthMass
        self.msini  = data["pl_bmassprov"].values == 'Msini'

        #storing G x Ms product
        GMs = const.G*data["st_mass"].values*u.solMass #units of solar mass
        p2sma = lambda mu,T: ((mu*T**2/(4*np.pi**2))**(1/3.)).to('AU')
        sma2p = lambda mu,a: (2*np.pi*np.sqrt(a**3.0/mu)).to('day')

        #storing radii
        self.radius = data["pl_radj"].values*u.jupiterRad
        self.radiusmask = data["pl_radj"].isna()
        self.radiuserr1 = data["pl_radjerr1"].values*u.jupiterRad
        self.radiuserr2 = data["pl_radjerr2"].values*u.jupiterRad

        #save semi-major axes, replacing nan with values converted from period
        self.sma = data["pl_orbsmax"].values*u.AU
        mask = data["pl_orbsmax"].isna()
        T = data["pl_orbper"].loc[mask].values*u.day
        self.sma[mask] = p2sma(GMs[mask],T)
        assert np.all(~np.isnan(self.sma)), 'sma has nan value(s)'

        #sma errors
        self.smaerr = data["pl_orbsmaxerr1"].values*u.AU
        mask = data["pl_orbsmaxerr1"].isna()
        T = data["pl_orbper"].loc[mask].values*u.day
        Terr = data["pl_orbpererr1"].loc[mask].values*u.day
        self.smaerr[mask] = np.abs(p2sma(GMs[mask],T+Terr) - p2sma(GMs[mask],T))
        self.smaerr[np.isnan(self.smaerr)] = np.nanmean(self.smaerr)

        #save eccentricities
        self.eccen = data["pl_orbeccen"].values
        mask = data["pl_orbeccen"].isna()
        _, etmp, _, _ = self.gen_plan_params(len(np.where(mask)[0]))
        self.eccen[mask] = etmp
        assert np.all(~np.isnan(self.eccen)), 'eccen has nan value(s)'

        #save eccentricities errors
        self.eccenerr = data["pl_orbeccenerr1"].values
        mask = data["pl_orbeccenerr1"].isna()
        self.eccenerr[mask | np.isnan(self.eccenerr)] = np.nanmean(self.eccenerr)

        #save the periastron time and period
        self.period = data["pl_orbper"].values*u.day
        mask = data["pl_orbper"].isna()
        self.period[mask] = sma2p(GMs[mask], self.sma[mask])
        assert np.all(~np.isnan(self.period)), 'period has nan value(s)'

        self.perioderr = data["pl_orbpererr1"].values*u.day
        mask = data["pl_orbpererr1"].isna()
        a = data["pl_orbsmax"].loc[mask].values*u.AU
        aerr = data["pl_orbsmaxerr1"].loc[mask].values*u.AU
        self.perioderr[mask] = np.abs(sma2p(GMs[mask],a+aerr) - sma2p(GMs[mask],a))
        self.perioderr[np.isnan(self.perioderr)] = np.nanmean(self.perioderr)

        #filling in random values if the periastron time is missing
        dat = data["pl_orbtper"].values
        mask = data["pl_orbtper"].isna()
        dat[mask] = np.random.uniform(low=np.nanmin(dat), high=np.nanmax(dat),
                    size = np.where(mask)[0].size)
        self.tper = Time(dat, format ='jd')
        self.tpererr = data["pl_orbtpererr1"].values*u.day
        tpererrmask = data["pl_orbtpererr1"].isna()
        self.tpererr[tpererrmask] = np.nanmean(self.tpererr)

        #saving the host name
        self.hostname = data["hostname"].values.astype(str)

        #saving the original data structure
        self.allplanetdata = data


