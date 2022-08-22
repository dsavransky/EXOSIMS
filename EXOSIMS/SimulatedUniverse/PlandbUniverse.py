from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
import numpy as np
import astropy.units as u
from astropy.time import Time

class PlandbUniverse(SimulatedUniverse):
    """Simulated Universe based on the planet population from the plandb database.
    Should work with the family of PlandbPlanets modules.(e.g. PlandbTargetList)"""
    
    def __init__(self, **specs):

        SimulatedUniverse.__init__(self, **specs)
    
    def gen_physical_properties(self, missionStart=60634, **specs):


        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        TL = self.TargetList

        #picking up planets belonging to each host stars
        starinds = np.array([])
        planinds = np.array([])

        for j,name in enumerate(TL.Name):
            tmp = np.where(PPop.hostname == name)[0]
            planinds = np.hstack((planinds,tmp))
            starinds = np.hstack((starinds,[j]*len(tmp)))
        planinds = planinds.astype(int)
        starinds = starinds.astype(int)

        #mapping planets to stars in standard format
        self.plan2star = starinds
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(planinds)

        #populating parameters
        #semi-major axis
        self.a = PPop.sma[planinds] + np.random.normal(size=self.nPlans)\
                *PPop.smaerr[planinds]
        #ensuring sampling did not make the values negative
        self.a[self.a<=0] = PPop.sma[planinds][self.a<=0]

        #eccentricity
        self.e = PPop.eccen[planinds] + np.random.normal(size=self.nPlans)\
                *PPop.eccenerr[planinds]

        self.e[self.e < 0.] = 0.
        self.e[self.e > 0.9] = 0.9

        Itmp, Otmp, self.w = PPop.gen_angles(self.nPlans)

        #inclination
        self.I = PPop.allplanetdata["pl_orbincl"].iloc[planinds].values + np.random.normal\
                (size=self.nPlans)*PPop.allplanetdata["pl_orbinclerr1"].iloc[planinds].values
        self.I[np.isnan(self.I)] = Itmp[np.isnan(self.I)].to('deg').value
        self.I = self.I.data*u.deg



        lper = PPop.allplanetdata["pl_orblper"].iloc[planinds].values + \
                np.random.normal(size=self.nPlans)*PPop.allplanetdata["pl_orblpererr1"].iloc[planinds].values

        #longitude of ascending nodeexit()
        self.O = lper.data*u.deg - self.w
        self.O[np.isnan(self.O)] =  Otmp[np.isnan(self.O)]

        #albedo
        self.p = PPMod.calc_albedo_from_sma(self.a,PPop.prange)

        #mass
        self.Mp = PPop.mass[planinds]

        #radius
        self.Rp = PPMod.calc_radius_from_mass(self.Mp)
        self.Rmask = ~PPop.radiusmask.values[planinds]
        self.Rp[self.Rmask] = PPop.radius[planinds][self.Rmask]
        self.Rperr1 = PPop.radiuserr1[planinds][self.Rmask]
        self.Rperr2 = PPop.radiuserr2[planinds][self.Rmask]


        #calculate period
        missionStart = Time(float(missionStart), format='mjd', scale='tai')
        T = PPop.period[planinds] + np.random.normal(size=self.nPlans)\
            *PPop.perioderr[planinds]
        T[T <= 0] = PPop.period[planinds][T <= 0]
        # calculate initial mean anomaly
        tper = Time(PPop.tper[planinds].value + (np.random.normal(size=self.nPlans)\
                *PPop.tpererr[planinds]).to('day').value, format='jd', scale='tai')
        self.M0 = ((missionStart - tper)/T % 1)*360*u.deg
