from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse
import numpy as np
import astropy.units as u
from astropy.time import Time
from EXOSIMS.util.deltaMag import deltaMag
from keplertools import fun


class PlandbUniverse(SimulatedUniverse):
    """Simulated Universe based on the planet population from the plandb database.
    Should work with the family of PlandbPlanets modules.(e.g. PlandbTargetList)"""

    def __init__(self, **specs):
        SimulatedUniverse.__init__(self, **specs)

    def gen_physical_properties(self, missionStart=60634, **specs):
        PPop = self.PlanetPopulation
        PPMod = self.PlanetPhysicalModel
        TL = self.TargetList

        # picking up planets belonging to each host stars
        starinds = np.array([])
        planinds = np.array([])

        for j, name in enumerate(TL.Name):
            tmp = np.where(PPop.hostname == name)[0]
            planinds = np.hstack((planinds, tmp))
            starinds = np.hstack((starinds, [j] * len(tmp)))
        self.planinds = planinds.astype(int)
        starinds = starinds.astype(int)

        # mapping planets to stars in standard format
        self.plan2star = starinds
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(planinds)

        # populating parameters
        # semi-major axis
        self.a = (
            PPop.sma[self.planinds]
            + np.random.normal(size=self.nPlans) * PPop.smaerr1[self.planinds]
        )
        # ensuring sampling did not make the values negative
        self.a[self.a <= 0] = PPop.sma[self.planinds][self.a <= 0]

        # eccentricity
        self.e = (
            PPop.eccen[self.planinds]
            + np.random.normal(size=self.nPlans) * PPop.eccenerr1[self.planinds]
        )

        self.e[self.e < 0.0] = 0.0
        self.e[self.e > 0.9] = 0.9

        Itmp, Otmp, self.w = PPop.gen_angles(self.nPlans)

        # inclination
        self.I = (
            PPop.allplanetdata["pl_orbincl"].iloc[self.planinds].values
            + np.random.normal(size=self.nPlans)
            * PPop.allplanetdata["pl_orbinclerr1"].iloc[self.planinds].values
        )
        self.I[np.isnan(self.I)] = Itmp[np.isnan(self.I)].to("deg").value
        self.I = self.I.data * u.deg

        # find the indices of planets with inclination 90 degrees
        ind90deg = np.where(self.I == 90 * u.deg)[0]
        self.Msini = PPop.allplanetdata["pl_msinij"].iloc[self.planinds].values

        Icrit = np.arcsin(
            (self.Msini * u.M_earth).value / ((0.0800 * u.M_sun).to(u.M_earth)).value
        )
        Irange = [Icrit, np.pi - Icrit]
        C = 0.5 * (np.cos(Irange[0]) - np.cos(Irange[1]))
        I = np.arccos(np.cos(Irange[0]) - 2.0 * C * np.random.uniform(size=self.nPlans))
        Ireplace = I * u.deg

        # replacing the 90 with a random value
        self.I[ind90deg] = Ireplace[ind90deg]
        mask = np.isnan(self.I)
        self.I[mask] = 90 * u.deg

        lper = (
            PPop.allplanetdata["pl_orblper"].iloc[self.planinds].values
            + np.random.normal(size=self.nPlans)
            * PPop.allplanetdata["pl_orblpererr1"].iloc[self.planinds].values
        )

        # longitude of ascending nodeexit()
        self.O = lper.data * u.deg - self.w
        self.O[np.isnan(self.O)] = Otmp[np.isnan(self.O)]

        # albedo
        self.p = PPMod.calc_albedo_from_sma(self.a, PPop.prange)

        # mass
        self.Mp = PPop.mass[self.planinds]

        # radius
        self.Rp = PPMod.calc_radius_from_mass(self.Mp)
        self.Rmask = ~PPop.radiusmask.values[self.planinds]
        self.Rp[self.Rmask] = PPop.radius[self.planinds][self.Rmask]
        self.Rperr1 = PPop.radiuserr1[self.planinds][self.Rmask]
        self.Rperr2 = PPop.radiuserr2[self.planinds][self.Rmask]

        # calculate period
        self.missionStart = Time(float(missionStart), format="mjd", scale="tai")
        T = (
            PPop.period[self.planinds]
            + np.random.normal(size=self.nPlans) * PPop.perioderr1[self.planinds]
        )
        T[T <= 0] = PPop.period[self.planinds][T <= 0]
        # calculate initial mean anomaly
        tper = Time(
            PPop.tper[self.planinds].value
            + (np.random.normal(size=self.nPlans) * PPop.tpererr1[self.planinds])
            .to("day")
            .value,
            format="jd",
            scale="tai",
        )
        self.M0 = ((self.missionStart - tper) / T % 1) * 360 * u.deg

    def TimeProbability(self, interval=1, **specs):
        PPop = self.PlanetPopulation
        TL = self.TargetList
        PPMod = self.PlanetPhysicalModel

        # mission end time
        missionLife = 3
        missionStart_year = self.missionStart.decimalyear
        missionEnd_year = missionStart_year + missionLife
        missionEnd_year = Time(missionEnd_year, format="decimalyear")
        missionEnd = missionEnd_year.mjd

        self.missionEnd = Time(float(missionEnd), format="mjd", scale="tai")

        # initialize array of periods
        T = np.zeros([self.nPlans, 1000])

        period = PPop.period[self.planinds].value

        perioderr = PPop.perioderr1[self.planinds].value

        # sampling period for 1000 orbits for each planet
        for inds in range(self.nPlans):
            T[inds] = np.random.normal(period[inds], perioderr[inds], 1000)

        T = T * u.day

        # time of periapsis passage iniatializing array
        tper = np.zeros([self.nPlans, 1000])

        top = PPop.tper[self.planinds].value

        toperr = PPop.tpererr1[self.planinds].value

        # sampling time of periapsis for 1000 orbits for each planet
        for inds in range(self.nPlans):
            tper[inds] = np.random.normal(top[inds], toperr[inds], 1000)

        tper = tper * u.day

        tper = Time(tper.to("day").value, format="jd", scale="tai")

        # Spacing out intervals between missionStart and missionEnd

        num_intervals = int(self.missionEnd.mjd - self.missionStart.mjd) / interval

        t = np.linspace(
            int(self.missionStart.mjd), int(self.missionEnd.mjd), int(num_intervals)
        )
        t = Time(t, format="mjd", scale="tai")
        t = t.jd

        # initialize array of mean anomalies
        M = np.zeros([self.nPlans, 1000, len(t)])

        tper = tper.value
        T = T.value

        # calculating mean anomaly at each time
        for i in range(self.nPlans):
            for j in range(len(t)):
                M[i, :, j] = t[j] - tper[i]

        M = (M / T[:, :, np.newaxis] % 1) * 360 * u.deg

        # initialize array of eccentricity
        e = np.zeros([self.nPlans, 1000])

        eccen = PPop.eccen[self.planinds]

        eccenerr = PPop.eccenerr1[self.planinds]

        # sampling the eccentricity for 1000 orbits for each planet
        for inds in range(self.nPlans):
            e[inds] = np.random.normal(eccen[inds], eccenerr[inds], 1000)

        e[e < 0.0] = 0.0
        e[e > 0.9] = 0.9

        # initialize array of argument of periapsis
        o = np.zeros([self.nPlans, 1000])

        omega = PPop.allplanetdata["pl_orblper"].iloc[self.planinds].values
        mask = np.isnan(omega)
        omega[mask] = self.w[mask]

        omegaerr = PPop.allplanetdata["pl_orblpererr1"].iloc[self.planinds].values
        mask = np.isnan(omegaerr)
        omegaerr[mask] = np.nanmean(omegaerr)

        # sampling the argument of periapsis for each planet
        for inds in range(self.nPlans):
            o[inds] = np.random.normal(omega[inds], omegaerr[inds], 1000)

        o = o * u.deg
        # calculating true anomaly from invKepler fn
        e_re = np.repeat(e[:, :, np.newaxis], M.shape[-1], axis=-1)
        # _, _, nu = fun.invKepler(M, e_re, return_nu=True)

        batch_size = 30
        num_batches = M.shape[0]
        nu = []

        M_batches = [M[i : i + batch_size] for i in range(0, num_batches, batch_size)]

        e_re_batches = [
            e_re[i : i + batch_size] for i in range(0, num_batches, batch_size)
        ]

        nu_batches = [
            fun.invKepler(M_batch, e_re_batch, return_nu=True)
            for M_batch, e_re_batch in zip(M_batches, e_re_batches)
        ]
        nu_list = [i[2] for i in nu_batches]
        nu = np.concatenate(nu_list)
        nu = nu.reshape(self.nPlans, 1000, len(t))
        nu = nu * u.deg
        # initialize array of argument of latitude
        theta = np.zeros([self.nPlans, 1000, len(t)])

        for i in range(self.nPlans):
            for j in range(1000):
                theta[i, j] = nu[i, j] + o[i, j]

        theta = theta * u.deg
        # sampling the inclination for each orbit of each planet
        inc = np.zeros([self.nPlans, 1000])

        Itmp, __, __ = PPop.gen_angles(self.nPlans)

        I = PPop.allplanetdata["pl_orbincl"].iloc[self.planinds].values

        Ierr = PPop.allplanetdata["pl_orbinclerr1"].iloc[self.planinds].values
        mask = np.isnan(Ierr)
        Ierr[mask] = np.nanmean(Ierr)

        I[np.isnan(I)] = Itmp[np.isnan(I)].to("deg").value

        I = I.data * u.deg

        # finding the indices where inclination is 90 degrees
        ind90deg = np.where(I == 90 * u.deg)[0]
        Msini = PPop.allplanetdata["pl_msinij"].iloc[self.planinds].values

        Icrit = np.arcsin(
            (Msini * u.M_earth).value / ((0.0800 * u.M_sun).to(u.M_earth)).value
        )
        Irange = [Icrit, np.pi - Icrit]
        C = 0.5 * (np.cos(Irange[0]) - np.cos(Irange[1]))
        Isamp = np.arccos(
            np.cos(Irange[0]) - 2.0 * C * np.random.uniform(size=self.nPlans)
        )
        Ireplace = Isamp * u.deg

        # replacing the 90 with a random value
        I[ind90deg] = Ireplace[ind90deg]
        mask = np.isnan(I)
        I[mask] = 90 * u.deg
        I = I.value

        for inds in range(self.nPlans):
            inc[inds] = np.random.normal(I[inds], Ierr[inds], 1000)

        inc = inc * u.deg

        # initialize array of semi-major axis
        a = np.zeros([self.nPlans, 1000])

        sma = PPop.sma[self.planinds].value

        smaerr = PPop.smaerr1[self.planinds].value

        # sampling the semi-major axis for each orbit of each planet
        for inds in range(self.nPlans):
            a[inds] = np.random.normal(sma[inds], smaerr[inds], 1000)

        sma = sma * u.AU
        a = a * u.AU

        # initialize array of orbital radius magnitude
        r = np.zeros([self.nPlans, 1000, len(t)])

        # calculating the orbital radius magnitude for each orbit of each planet
        r = (
            a[:, :, np.newaxis]
            * (1 - e[:, :, np.newaxis] ** 2)
            / (1 + e[:, :, np.newaxis] * np.cos(nu[:, :, :]))
        )

        # initialize array of projected seperation
        sep = np.zeros([self.nPlans, 1000, len(t)])

        # calculating the projected seperation for each orbit of each planet
        sep = r[:, :, :] * np.sqrt(
            1 - np.sin(inc[:, :, np.newaxis]) ** 2 * np.sin(theta[:, :, :]) ** 2
        )
        sep = sep.value
        # initialize array of angular seperation
        alpha = np.zeros([self.nPlans, 1000, interval])

        # distance from the star to the observer
        d = PPop.allplanetdata["sy_dist"].values[self.planinds] * u.pc
        d = d.to(u.AU).value

        # calculating the angular seperation for each orbit of each planet
        alpha = np.arctan(sep / d[:, np.newaxis, np.newaxis])[:, :, : len(t)] * u.rad

        # deltaMag calculation
        dMag = np.zeros([self.nPlans, 1000, len(t)])

        beta = np.zeros([self.nPlans, 1000, len(t)])

        beta = np.arccos(np.sin(inc[:, :, np.newaxis]) * np.sin(theta[:, :, : len(t)]))

        Phi = np.zeros([self.nPlans, 1000, interval])

        Phi = PPMod.calc_Phi(beta)

        # albedo
        p = PPMod.calc_albedo_from_sma(sma, PPop.prange)

        # radius of the planet
        Rp = PPop.radius[self.planinds].to(u.km)

        # calculating deltaMag for each orbit of each planet
        dMag = deltaMag(
            p[:, np.newaxis, np.newaxis], Rp[:, np.newaxis, np.newaxis], r, Phi
        )

        # assigning intdMag to each planet based on its host star
        intdMag = TL.int_dMag[self.plan2star]

        # checking the conditions being met
        detbool = np.where(
            (TL.default_mode["IWA"] < alpha)
            & (alpha < TL.default_mode["OWA"])
            & (dMag < intdMag[:, np.newaxis, np.newaxis]),
            1,
            0,
        )

        Pdet = np.sum(detbool, axis=1) / 1000

        return Pdet
