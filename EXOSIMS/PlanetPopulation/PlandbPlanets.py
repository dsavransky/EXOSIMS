from EXOSIMS.Prototypes.PlanetPopulation import PlanetPopulation
import pickle
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.time import Time
import requests
from EXOSIMS.util.get_dirs import get_downloads_dir
import os


class PlandbPlanets(PlanetPopulation):
    """Population of Known RV planets from the plandb database. This module is created
    with the same framework as that of the existing Known RV module."""

    def __init__(self, **specs):
        PlanetPopulation.__init__(self, **specs)

        # read the orbfits file from the EXOSIMS downloads directory
        downloadsdir = get_downloads_dir()
        filename = "orbfits_2022-05.p"
        planetfilepath = os.path.join(downloadsdir, filename)

        # fetching the orbfits file if it doesn't exist already
        if not os.path.exists(planetfilepath) and os.access(
            downloadsdir, os.W_OK | os.X_OK
        ):
            url = "https://plandb.sioslab.com/data/orbfits_2022-05.p"
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 "
                "Safari/537.36"
            }
            filename = "orbfits_2022-05.p"
            planetfilepath = os.path.join(downloadsdir, filename)

            with requests.get(url, stream=True, headers=headers) as r:
                r.raise_for_status()
                with open(planetfilepath, "wb") as f:
                    for c in r.iter_content(chunk_size=8192):
                        f.write(c)

        with open(planetfilepath, "rb") as f:
            db = pickle.load(f)

        # selecting the default fit of the orbit fits from the pickle file
        data = db.loc[db["default_fit"] == 1]

        # filtering the data based on the availability of mass info, stellar mass,
        # and (sma or period)
        keep = (
            ~data["pl_bmasse"].isna()
            & ~data["st_mass"].isna()
            & (~data["pl_orbsmax"].isna() | ~data["pl_orbper"].isna())
        )
        data = data[keep]

        # storing the planet paramaters - mass, evaluating which data are Msini
        self.mass = data["pl_bmasse"].values * u.earthMass
        self.masserr = data["pl_bmasseerr1"].values * u.earthMass
        self.msini = data["pl_bmassprov"].values == "Msini"

        # storing G x Ms product
        GMs = const.G * data["st_mass"].values * u.solMass  # units of solar mass
        p2sma = lambda mu, T: ((mu * T**2 / (4 * np.pi**2)) ** (1 / 3.0)).to("AU")
        sma2p = lambda mu, a: (2 * np.pi * np.sqrt(a**3.0 / mu)).to("day")

        # storing radii
        self.radius = data["pl_radj_forecastermod"].values * u.jupiterRad
        self.radiusmask = data["pl_radj_forecastermod"].isna()
        self.radiuserr1 = data["pl_radj_forecastermoderr1"].values * u.jupiterRad
        self.radiuserr2 = data["pl_radj_forecastermoderr2"].values * u.jupiterRad
        self.radiuserr1[self.radiusmask | np.isnan(self.radiuserr1)] = np.nanmean(
            self.radiuserr1
        )
        self.radiuserr2[self.radiusmask | np.isnan(self.radiuserr2)] = np.nanmean(
            self.radiuserr2
        )
        assert np.all(~np.isnan(self.radius)), "radius has nan values"

        # save semi-major axes, replacing nan with values converted from period
        self.sma = data["pl_orbsmax"].values * u.AU
        mask = data["pl_orbsmax"].isna()
        T = data["pl_orbper"].loc[mask].values * u.day
        self.sma[mask] = p2sma(GMs[mask], T)
        assert np.all(~np.isnan(self.sma)), "sma has nan value(s)"

        # sma errors
        self.smaerr1 = data["pl_orbsmaxerr1"].values * u.AU
        self.smaerr2 = data["pl_orbsmaxerr2"].values * u.AU
        mask = data["pl_orbsmaxerr1"].isna()
        T = data["pl_orbper"].loc[mask].values * u.day
        Terr = data["pl_orbpererr1"].loc[mask].values * u.day
        self.smaerr1[mask] = np.abs(p2sma(GMs[mask], T + Terr) - p2sma(GMs[mask], T))
        self.smaerr1[np.isnan(self.smaerr1)] = np.nanmean(self.smaerr1)

        mask = data["pl_orbsmaxerr2"].isna()
        T = data["pl_orbper"].loc[mask].values * u.day
        Terr = data["pl_orbpererr2"].loc[mask].values * u.day
        self.smaerr2[mask] = np.abs(p2sma(GMs[mask], T + Terr) - p2sma(GMs[mask], T))
        self.smaerr2[np.isnan(self.smaerr2)] = np.nanmean(self.smaerr2)

        # save eccentricities
        self.eccen = data["pl_orbeccen"].values
        mask = data["pl_orbeccen"].isna()
        _, etmp, _, _ = self.gen_plan_params(len(np.where(mask)[0]))
        self.eccen[mask] = etmp
        assert np.all(~np.isnan(self.eccen)), "eccen has nan value(s)"

        # save eccentricities errors
        self.eccenerr1 = data["pl_orbeccenerr1"].values
        mask = data["pl_orbeccenerr1"].isna()
        self.eccenerr1[mask | np.isnan(self.eccenerr1)] = np.nanmean(self.eccenerr1)

        self.eccenerr2 = data["pl_orbeccenerr2"].values
        mask = data["pl_orbeccenerr2"].isna()
        self.eccenerr2[mask | np.isnan(self.eccenerr2)] = np.nanmean(self.eccenerr2)

        # save the periastron time and period
        self.period = data["pl_orbper"].values * u.day
        mask = data["pl_orbper"].isna()
        self.period[mask] = sma2p(GMs[mask], self.sma[mask])
        assert np.all(~np.isnan(self.period)), "period has nan value(s)"

        self.perioderr1 = data["pl_orbpererr1"].values * u.day
        mask = data["pl_orbpererr1"].isna()
        a = data["pl_orbsmax"].loc[mask].values * u.AU
        aerr = data["pl_orbsmaxerr1"].loc[mask].values * u.AU
        self.perioderr1[mask] = np.abs(sma2p(GMs[mask], a + aerr) - sma2p(GMs[mask], a))
        self.perioderr1[np.isnan(self.perioderr1)] = np.nanmean(self.perioderr1)

        self.perioderr2 = data["pl_orbpererr2"].values * u.day
        mask = data["pl_orbpererr2"].isna()
        a = data["pl_orbsmax"].loc[mask].values * u.AU
        aerr = data["pl_orbsmaxerr2"].loc[mask].values * u.AU
        self.perioderr2[mask] = np.abs(sma2p(GMs[mask], a + aerr) - sma2p(GMs[mask], a))
        self.perioderr2[np.isnan(self.perioderr2)] = np.nanmean(self.perioderr2)

        # filling in random values if the periastron time is missing
        dat = data["pl_orbtper"].values
        mask = data["pl_orbtper"].isna()
        dat[mask] = np.random.uniform(
            low=np.nanmin(dat), high=np.nanmax(dat), size=np.where(mask)[0].size
        )
        self.tper = Time(dat, format="jd")
        self.tpererr1 = data["pl_orbtpererr1"].values * u.day
        tpererrmask = data["pl_orbtpererr1"].isna()
        self.tpererr1[tpererrmask] = np.nanmean(self.tpererr1)

        self.tpererr2 = data["pl_orbtpererr2"].values * u.day
        tpererrmask = data["pl_orbtpererr2"].isna()
        self.tpererr2[tpererrmask] = np.nanmean(self.tpererr2)

        # saving the host name
        self.hostname = data["hostname"].values.astype(str)

        # saving the original data structure
        self.allplanetdata = data
