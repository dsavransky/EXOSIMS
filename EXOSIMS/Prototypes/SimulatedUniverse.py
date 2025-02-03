import astropy.constants as const
import astropy.units as u
import numpy as np

from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.eccanom import eccanom
from EXOSIMS.util.get_dirs import get_cache_dir
from EXOSIMS.util.get_module import get_module
from EXOSIMS.util.keplerSTM import planSys
from EXOSIMS.util.vprint import vprint


class SimulatedUniverse(object):
    r""":ref:`SimulatedUniverse` Prototype

    Args:
        fixedPlanPerStar (int, optional):
            If set, every system will have the same number of planets.
            Defaults to None
        Min (float, optional):
            Initial mean anomaly for all planets.  If set, every planet
            has the same mean anomaly at mission start. Defaults to None
        cachedir (str, optional):
            Full path to cachedir.
            If None (default) use default (see :ref:`EXOSIMSCACHE`)
        lucky_planets (bool):
            Used downstream in survey simulation. If True, planets are
            observed at optimal times. Defaults to False
        commonSystemPlane (bool):
            Planet inclinations are sampled as normally distributed about a
            common system plane. Defaults to False
        commonSystemPlaneParams (list(float)):
            [inclination mean, inclination standard deviation, Omega mean, Omega
            standard deviation] defining the normal distribution of
            inclinations and longitudes of the ascending node about a common
            system plane in units of degrees.  Ignored if commonSystemPlane is
            False. Defaults to [0 2.25, 0, 2.25], where the standard deviation
            is approximately the standard deviation of solar system planet
            inclinations.
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        _outspec (dict):
            :ref:`sec:outspec`
        a (astropy.units.quantity.Quantity):
             Planet semi-major axis (length units)
        BackgroundSources (:ref:`BackgroundSources`):
            BackgroundSources object
        cachedir (str):
            Path to the EXOSIMS cache directory (see :ref:`EXOSIMSCACHE`)
        commonSystemPlaneParams (list):
            2 element list of [mean, standard deviation] in units of degrees,
            describing the distribution of inclinations relative to a common orbital
            plane.  Ignored if commonSystemPlane is False.
        commonSystemPlane (bool):
            If False, planet inclinations are independently drawn for all planets,
            including those in the same target system.  If True, inclinations will be
            drawn from a normal distribution defined by
            commonSystemPlaneParams and added to a single inclination value drawn
            for each system.
        Completeness (:ref:`Completeness`):
            Completeness object
        d (astropy.units.quantity.Quantity):
            Current orbital radius magnitude (length units)
        dMag (numpy.ndarray):
            Current planet :math:`\Delta\mathrm{mag}`
        e (numpy.ndarray):
            Planet eccentricity
        fEZ (astropy.units.quantity.Quantity):
            Surface brightness of exozodiacal light in units of 1/arcsec2
        fixedPlanPerStar (int or None):
            If set, every system has the same number of planets, given by
            this attribute
        I (astropy.units.quantity.Quantity):
            Planet inclinations (angle units)
        lucky_planets (bool):
            If True, planets are observed at optimal times.
        M0 (astropy.units.quantity.Quantity):
            Initial planet mean anomaly (at mission start time).
        Min (float or None):
            Input constant initial mean anomaly.  If none, initial
            mean anomaly is randomly distributed from a uniform distribution in
            [0, 360] degrees.
        Mp (astropy.units.quantity.Quantity):
            Planet mass.
        nPlans (int):
            Number of planets in all target systems.
        O (astropy.units.quantity.Quantity):
            Planet longitude of the ascending node (angle units)
        OpticalSystem (:ref:`OpticalSystem`):
            Optical System object
        p (numpy.ndarray):
            Planet geometric albedo
        phi (numpy.ndarray):
            Current value of planet phase function.
        phiIndex (numpy.ndarray):
            Intended for use with input
            'whichPlanetPhaseFunction'='realSolarSystemPhaseFunc'
            When None, the default is the phi_lambert function, otherwise it is Solar
            System Phase Functions
        plan2star (numpy.ndarray):
            Index of host star or each planet.  Indexes attributes of TargetsList.
        planet_atts (list):
            List of planet attributes
        PlanetPhysicalModel (:ref:`PlanetPhysicalModel`):
            Planet physical model object.
        PlanetPopulation (:ref:`PlanetPopulation`):
            Planet population object.
        PostProcessing (:ref:`PostProcessing`):
            Postprocessing object.
        r (astropy.units.quantity.Quantity):
            Current planet orbital radius (3xnPlans). Length units.
        Rp (astropy.units.quantity.Quantity):
            Planet radius (length units).
        s (astropy.units.quantity.Quantity):
            Current planet projected separation. Length units.
        sInds (numpy.ndarray):
            Indices of stars with planets.  Equivalent to unique entries of
            ``plan2star``.
        TargetList (:ref:`TargetList`):
            Target list object.
        v (astropy.units.quantity.Quantity):
            Current orbital velocity vector (3xnPlans). Velocity units.
        w (astropy.units.quantity.Quantity):
            Planet argument of periapsis.
        WA (astropy.units.quantity.Quantity):
            Current planet angular separation (angle units)
        ZodiacalLight (:ref:`ZodiacalLight`):
            Zodiacal light object.


    .. note::

        When generating planets, :ref:`PlanetPopulation` attribute ``eta`` is
        treated as the rate parameter of a Poisson distribution.
        Each target's number of planets is a Poisson random variable
        sampled with :math:`\lambda\equiv\eta`.

    .. warning::

        All attributes described as 'current' are updated only when planets are
        observed.  As such, during mission simulations, these values for different
        planets correspond to different times (bookkept in the survey simulation
        object).

    """

    _modtype = "SimulatedUniverse"

    def __init__(
        self,
        fixedPlanPerStar=None,
        Min=None,
        cachedir=None,
        lucky_planets=False,
        commonSystemPlane=False,
        commonSystemPlaneParams=[0, 2.25, 0, 2.25],
        **specs
    ):

        # start the outspec
        self._outspec = {}

        # load the vprint function (same line in all prototype module constructors)
        self.vprint = vprint(specs.get("verbose", True))
        self.lucky_planets = lucky_planets
        self._outspec["lucky_planets"] = lucky_planets
        self.commonSystemPlane = bool(commonSystemPlane)
        self._outspec["commonSystemPlane"] = commonSystemPlane
        assert (
            len(commonSystemPlaneParams) == 4
        ), "commonSystemPlaneParams must be a four-element list"
        self.commonSystemPlaneParams = commonSystemPlaneParams
        self._outspec["commonSystemPlaneParams"] = commonSystemPlaneParams

        # save fixed number of planets to generate
        self.fixedPlanPerStar = fixedPlanPerStar
        self._outspec["fixedPlanPerStar"] = fixedPlanPerStar

        # get cache directory
        self.cachedir = get_cache_dir(cachedir)
        self._outspec["cachedir"] = self.cachedir
        specs["cachedir"] = self.cachedir

        # check if KnownRVPlanetsUniverse has correct input modules
        if specs["modules"]["SimulatedUniverse"] == "KnownRVPlanetsUniverse":
            val = (
                specs["modules"]["TargetList"] == "KnownRVPlanetsTargetList"
                and specs["modules"]["PlanetPopulation"] == "KnownRVPlanets"
            )
            assert val, (
                "KnownRVPlanetsUniverse must use KnownRVPlanetsTargetList "
                "and KnownRVPlanets"
            )
        else:
            val = (
                specs["modules"]["TargetList"] == "KnownRVPlanetsTargetList"
                or specs["modules"]["PlanetPopulation"] == "KnownRVPlanets"
            )
            assert not (val), (
                "KnownRVPlanetsTargetList or KnownRVPlanets should not be used "
                "with this SimulatedUniverse"
            )

        # import TargetList class
        self.TargetList = get_module(specs["modules"]["TargetList"], "TargetList")(
            **specs
        )

        # bring inherited class objects to top level of Simulated Universe
        TL = self.TargetList
        self.StarCatalog = TL.StarCatalog
        self.PlanetPopulation = TL.PlanetPopulation
        self.PlanetPhysicalModel = TL.PlanetPhysicalModel
        self.OpticalSystem = TL.OpticalSystem
        self.ZodiacalLight = TL.ZodiacalLight
        self.BackgroundSources = TL.BackgroundSources
        self.PostProcessing = TL.PostProcessing
        self.Completeness = TL.Completeness

        # initial constant mean anomaly
        assert isinstance(Min, (int, float)) or (
            Min is None
        ), "Min may be int, float, or None"
        if Min is not None:
            self.Min = float(Min) * u.deg
        else:
            self.Min = Min
        self._outspec["Min"] = Min

        # list of possible planet attributes
        self.planet_atts = [
            "plan2star",
            "a",
            "e",
            "I",
            "O",
            "w",
            "M0",
            "Min",
            "Rp",
            "Mp",
            "p",
            "r",
            "v",
            "d",
            "s",
            "phi",
            "fEZ",
            "dMag",
            "WA",
        ]

        self.phiIndex = (
            None  # Used to switch select specific phase function for each planet
        )

        # generate orbital elements, albedos, radii, and masses
        self.gen_physical_properties(**specs)

        # find initial position-related parameters: position, velocity, planet-star
        # distance, apparent separation, surface brightness of exo-zodiacal light
        self.init_systems()

    def __str__(self):
        """String representation of Simulated Universe object

        When the command 'print' is used on the Simulated Universe object,
        this method will return the values contained in the object

        """

        for att in self.__dict__:
            print("%s: %r" % (att, getattr(self, att)))

        return "Simulated Universe class object attributes"

    def gen_physical_properties(self, **specs):
        """Generates the planetary systems' physical properties.

        Populates arrays of the orbital elements, albedos, masses and radii
        of all planets, and generates indices that map from planet to parent star.

        Args:
            **specs:
                :ref:`sec:inputspec`

        Returns:
            None

        """

        PPop = self.PlanetPopulation
        TL = self.TargetList

        if self.fixedPlanPerStar is not None:  # Must be an int for fixedPlanPerStar
            # Create array of length TL.nStars each w/ value ppStar
            targetSystems = np.ones(TL.nStars).astype(int) * self.fixedPlanPerStar
        else:
            # treat eta as the rate parameter of a Poisson distribution
            targetSystems = np.random.poisson(lam=PPop.eta, size=TL.nStars)

        plan2star = []
        for j, n in enumerate(targetSystems):
            plan2star = np.hstack((plan2star, [j] * n))
        self.plan2star = plan2star.astype(int)
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(self.plan2star)

        # The prototype StarCatalog module is made of one single G star at 1pc.
        # In that case, the SimulatedUniverse prototype generates one Jupiter
        # at 5 AU to allow for characterization testing.
        # Also generates at least one Jupiter if no planet was generated.
        if (TL.Name[0].startswith("Prototype") and (TL.nStars == 1)) or (
            self.nPlans == 0
        ):
            if self.nPlans == 0:
                self.vprint("No planets were generated. Creating single fake planet.")
            else:
                self.vprint(
                    (
                        "Prototype StarCatalog with 1 target. "
                        "Creating single fake planet."
                    )
                )
            self.plan2star = np.array([0], dtype=int)
            self.sInds = np.unique(self.plan2star)
            self.nPlans = len(self.plan2star)
            self.a = np.array([5.0]) * u.AU
            self.e = np.array([0.0])
            self.I = np.array([0.0]) * u.deg
            self.O = np.array([0.0]) * u.deg
            self.w = np.array([0.0]) * u.deg
            self.gen_M0()
            self.Rp = np.array([10.0]) * u.earthRad
            self.Mp = np.array([300.0]) * u.earthMass
            self.p = np.array([0.6])
        else:
            # sample all of the orbital and physical parameters
            self.I, self.O, self.w = PPop.gen_angles(
                self.nPlans,
                commonSystemPlane=self.commonSystemPlane,
                commonSystemPlaneParams=self.commonSystemPlaneParams,
            )
            self.setup_system_planes()

            self.a, self.e, self.p, self.Rp = PPop.gen_plan_params(self.nPlans)
            if PPop.scaleOrbits:
                self.a *= np.sqrt(TL.L[self.plan2star])
            self.gen_M0()  # initial mean anomaly
            self.Mp = PPop.gen_mass(self.nPlans)  # mass

        if self.ZodiacalLight.commonSystemfEZ:
            self.ZodiacalLight.nEZ = self.ZodiacalLight.gen_systemnEZ(TL.nStars)

        self.phiIndex = np.asarray(
            []
        )  # Used to switch select specific phase function for each planet

    def gen_M0(self):
        """Set initial mean anomaly for each planet"""
        if self.Min is not None:
            self.M0 = np.ones((self.nPlans,)) * self.Min
        else:
            self.M0 = np.random.uniform(360, size=int(self.nPlans)) * u.deg

    def init_systems(self):
        """Finds initial time-dependant parameters. Assigns each planet an
        initial position, velocity, planet-star distance, apparent separation,
        phase function, surface brightness of exo-zodiacal light, delta
        magnitude, and working angle.

        This method makes use of the systems' physical properties (masses,
        distances) and their orbital elements (a, e, I, O, w, M0).
        """

        PPMod = self.PlanetPhysicalModel
        ZL = self.ZodiacalLight
        TL = self.TargetList

        a = self.a.to("AU").value  # semi-major axis
        e = self.e  # eccentricity
        I = self.I.to("rad").value  # inclinations #noqa: E741
        O = self.O.to("rad").value  # right ascension of the ascending node #noqa: E741
        w = self.w.to("rad").value  # argument of perigee
        M0 = self.M0.to("rad").value  # initial mean anomany
        E = eccanom(M0, e)  # eccentric anomaly
        Mp = self.Mp  # planet masses

        # This is the a1 a2 a3 and b1 b2 b3 are the euler angle transformation from
        # the I,J,K refernece frame to an x,y,z frame
        a1 = np.cos(O) * np.cos(w) - np.sin(O) * np.cos(I) * np.sin(w)
        a2 = np.sin(O) * np.cos(w) + np.cos(O) * np.cos(I) * np.sin(w)
        a3 = np.sin(I) * np.sin(w)
        A = a * np.vstack((a1, a2, a3)) * u.AU
        b1 = -np.sqrt(1 - e**2) * (
            np.cos(O) * np.sin(w) + np.sin(O) * np.cos(I) * np.cos(w)
        )
        b2 = np.sqrt(1 - e**2) * (
            -np.sin(O) * np.sin(w) + np.cos(O) * np.cos(I) * np.cos(w)
        )
        b3 = np.sqrt(1 - e**2) * np.sin(I) * np.cos(w)
        B = a * np.vstack((b1, b2, b3)) * u.AU
        r1 = np.cos(E) - e
        r2 = np.sin(E)

        mu = const.G * (Mp + TL.MsTrue[self.plan2star])
        v1 = np.sqrt(mu / self.a**3) / (1 - e * np.cos(E))
        v2 = np.cos(E)

        self.r = (A * r1 + B * r2).T.to("AU")  # position
        self.v = (v1 * (-A * r2 + B * v2)).T.to("AU/day")  # velocity
        self.s = np.linalg.norm(self.r[:, 0:2], axis=1)  # apparent separation
        self.d = np.linalg.norm(self.r, axis=1)  # planet-star distance
        try:
            self.phi = PPMod.calc_Phi(
                np.arccos(self.r[:, 2] / self.d), phiIndex=self.phiIndex
            )  # planet phase
        except u.UnitTypeError:
            self.d = self.d * self.r.unit  # planet-star distance
            self.phi = PPMod.calc_Phi(
                np.arccos(self.r[:, 2] / self.d), phiIndex=self.phiIndex
            )  # planet phase

        # self.phi = PPMod.calc_Phi(np.arccos(self.r[:,2]/self.d))    # planet phase
        self.fEZ = ZL.fEZ(TL.MV[self.plan2star], self.I, self.d)  # exozodi brightness

        self.dMag = deltaMag(self.p, self.Rp, self.d, self.phi)  # delta magnitude
        try:
            self.WA = np.arctan(self.s / TL.dist[self.plan2star]).to(
                "arcsec"
            )  # working angle
        except u.UnitTypeError:
            self.s = self.s * self.r.unit
            self.WA = np.arctan(self.s / TL.dist[self.plan2star]).to(
                "arcsec"
            )  # working angle

    def propag_system(self, sInd, dt):
        """Propagates planet time-dependant parameters: position, velocity,
        planet-star distance, apparent separation, phase function, surface brightness
        of exo-zodiacal light, delta magnitude, and working angle.

        This method uses the Kepler state transition matrix to propagate a
        planet's state (position and velocity vectors) forward in time using
        the Kepler state transition matrix.

        Args:
            sInd (int):
                Index of the target system of interest
            dt (~astropy.units.Quantity(float)):
                Time increment in units of day, for planet position propagation

        Returns:
            None
        """

        PPMod = self.PlanetPhysicalModel
        TL = self.TargetList

        assert np.isscalar(
            sInd
        ), "Can only propagate one system at a time, sInd must be scalar."
        # check for planets around this target
        pInds = np.where(self.plan2star == sInd)[0]
        if len(pInds) == 0:
            return
        # check for positive time increment
        assert dt >= 0, "Time increment (dt) to propagate a planet must be positive."
        if dt == 0:
            return

        # Calculate initial positions in AU and velocities in AU/day
        r0 = self.r[pInds].to("AU").value
        v0 = self.v[pInds].to("AU/day").value
        # stack dimensionless positions and velocities
        nPlans = pInds.size
        x0 = np.reshape(np.concatenate((r0, v0), axis=1), nPlans * 6)

        # Calculate vector of gravitational parameter in AU3/day2
        Ms = TL.MsTrue[[sInd]]
        Mp = self.Mp[pInds]
        mu = (const.G * (Mp + Ms)).to("AU3/day2").value

        # use keplerSTM.py to propagate the system
        prop = planSys(x0, mu, epsmult=10.0)
        try:
            prop.takeStep(dt.to("day").value)
        except ValueError:
            # try again with larger epsmult and two steps to force convergence
            prop = planSys(x0, mu, epsmult=100.0)
            try:
                prop.takeStep(dt.to("day").value / 2.0)
                prop.takeStep(dt.to("day").value / 2.0)
            except ValueError:
                raise ValueError("planSys error")

        # split off position and velocity vectors
        x1 = np.array(np.hsplit(prop.x0, 2 * nPlans))
        rind = np.array(range(0, len(x1), 2))  # even indices
        vind = np.array(range(1, len(x1), 2))  # odd indices

        # update planets' position, velocity, planet-star distance, apparent,
        # separation, phase function, exozodi surface brightness, delta magnitude and
        # working angle
        self.r[pInds] = x1[rind] * u.AU
        self.v[pInds] = x1[vind] * u.AU / u.day

        try:
            self.d[pInds] = np.linalg.norm(self.r[pInds], axis=1)
            if len(self.phiIndex) == 0:
                self.phi[pInds] = PPMod.calc_Phi(
                    np.arccos(self.r[pInds, 2] / self.d[pInds]), phiIndex=self.phiIndex
                )
            else:
                self.phi[pInds] = PPMod.calc_Phi(
                    np.arccos(self.r[pInds, 2] / self.d[pInds]),
                    phiIndex=self.phiIndex[pInds],
                )
        except u.UnitTypeError:
            self.d[pInds] = np.linalg.norm(self.r[pInds], axis=1) * self.r.unit
            if len(self.phiIndex) == 0:
                self.phi[pInds] = PPMod.calc_Phi(
                    np.arccos(self.r[pInds, 2] / self.d[pInds]), phiIndex=self.phiIndex
                )
            else:
                self.phi[pInds] = PPMod.calc_Phi(
                    np.arccos(self.r[pInds, 2] / self.d[pInds]),
                    phiIndex=self.phiIndex[pInds],
                )

        # self.fEZ[pInds] = ZL.fEZ(TL.MV[sInd], self.I[pInds], self.d[pInds])
        self.dMag[pInds] = deltaMag(
            self.p[pInds], self.Rp[pInds], self.d[pInds], self.phi[pInds]
        )
        try:
            self.s[pInds] = np.linalg.norm(self.r[pInds, 0:2], axis=1)
            self.WA[pInds] = np.arctan(self.s[pInds] / TL.dist[sInd]).to("arcsec")
        except u.UnitTypeError:
            self.s[pInds] = np.linalg.norm(self.r[pInds, 0:2], axis=1) * self.r.unit
            self.WA[pInds] = np.arctan(self.s[pInds] / TL.dist[sInd]).to("arcsec")

    def set_planet_phase(self, beta=np.pi / 2):
        """Positions all planets at input star-planet-observer phase angle
        where possible. For systems where the input phase angle is not achieved,
        planets are positioned at quadrature (phase angle of 90 deg).

        The position found here is not unique. The desired phase angle will be
        achieved at two points on the planet's orbit (for non-face on orbits).

        Args:
            beta (float):
                star-planet-observer phase angle in radians.

        """

        PPMod = self.PlanetPhysicalModel
        ZL = self.ZodiacalLight
        TL = self.TargetList

        a = self.a.to("AU").value  # semi-major axis
        e = self.e  # eccentricity
        I = self.I.to("rad").value  # noqa: E741 # inclinations
        O = self.O.to("rad").value  # noqa: E741 # long. of the ascending node
        w = self.w.to("rad").value  # argument of perigee
        Mp = self.Mp  # planet masses

        # make list of betas
        betas = beta * np.ones(w.shape)
        mask = np.cos(betas) / np.sin(I) > 1.0
        num = len(np.where(mask)[0])
        betas[mask] = np.pi / 2.0
        mask = np.cos(betas) / np.sin(I) < -1.0
        num += len(np.where(mask)[0])
        betas[mask] = np.pi / 2.0
        if num > 0:
            self.vprint("***Warning***")
            self.vprint(
                (
                    "{} planets out of {} could not be set to phase angle {} radians."
                ).format(num, self.nPlans, beta)
            )
            self.vprint("These planets are set to quadrature (phase angle pi/2)")

        # solve for true anomaly
        nu = np.arcsin(np.cos(betas) / np.sin(I)) - w

        # setup for position and velocity
        a1 = np.cos(O) * np.cos(w) - np.sin(O) * np.cos(I) * np.sin(w)
        a2 = np.sin(O) * np.cos(w) + np.cos(O) * np.cos(I) * np.sin(w)
        a3 = np.sin(I) * np.sin(w)
        A = np.vstack((a1, a2, a3))

        b1 = -(np.cos(O) * np.sin(w) + np.sin(O) * np.cos(I) * np.cos(w))
        b2 = -np.sin(O) * np.sin(w) + np.cos(O) * np.cos(I) * np.cos(w)
        b3 = np.sin(I) * np.cos(w)
        B = np.vstack((b1, b2, b3))

        r = a * (1.0 - e**2) / (1.0 - e * np.cos(nu))
        mu = const.G * (Mp + TL.MsTrue[self.plan2star])
        v1 = -np.sqrt(mu / (self.a * (1.0 - self.e**2))) * np.sin(nu)
        v2 = np.sqrt(mu / (self.a * (1.0 - self.e**2))) * (self.e + np.cos(nu))

        self.r = (A * r * np.cos(nu) + B * r * np.sin(nu)).T * u.AU  # position
        self.v = (A * v1 + B * v2).T.to("AU/day")  # velocity

        try:
            self.d = np.linalg.norm(self.r, axis=1)  # planet-star distance
            self.phi = PPMod.calc_Phi(
                np.arccos(self.r[:, 2].to("AU").value / self.d.to("AU").value) * u.rad,
                phiIndex=self.phiIndex,
            )  # planet phase
        except u.UnitTypeError:
            self.d = (
                np.linalg.norm(self.r, axis=1) * self.r.unit
            )  # planet-star distance
            self.phi = PPMod.calc_Phi(
                np.arccos(self.r[:, 2].to("AU").value / self.d.to("AU").value) * u.rad,
                phiIndex=self.phiIndex,
            )  # planet phase

        self.fEZ = ZL.fEZ(TL.MV[self.plan2star], self.I, self.d)  # exozodi brightness
        self.dMag = deltaMag(self.p, self.Rp, self.d, self.phi)  # delta magnitude

        try:
            self.s = np.linalg.norm(self.r[:, 0:2], axis=1)  # apparent separation
            self.WA = np.arctan(self.s / TL.dist[self.plan2star]).to(
                "arcsec"
            )  # working angle
        except u.UnitTypeError:
            self.s = (
                np.linalg.norm(self.r[:, 0:2], axis=1) * self.r.unit
            )  # apparent separation
            self.WA = np.arctan(self.s / TL.dist[self.plan2star]).to(
                "arcsec"
            )  # working angle

    def dump_systems(self):
        """Create a dictionary of planetary properties for archiving use.

        Args:
            None

        Returns:
            dict:
                Dictionary of planetary properties

        """

        systems = {
            "a": self.a,
            "e": self.e,
            "I": self.I,
            "O": self.O,
            "w": self.w,
            "M0": self.M0,
            "Mp": self.Mp,
            "mu": (
                const.G * (self.Mp + self.TargetList.MsTrue[self.plan2star])
            ).decompose(),
            "Rp": self.Rp,
            "p": self.p,
            "plan2star": self.plan2star,
            "star": self.TargetList.Name[self.plan2star],
        }
        if self.commonSystemPlane:
            systems["systemInclination"] = self.TargetList.systemInclination
            systems["systemOmega"] = self.TargetList.systemOmega
        if self.ZodiacalLight.commonSystemfEZ:
            systems["starnEZ"] = self.ZodiacalLight.nEZ

        return systems

    def load_systems(self, systems):
        """Load a dictionary of planetary properties (nominally created by dump_systems)

        Args:
            systems (dict):
                Dictionary of planetary properties corresponding to the output of
                dump_systems.

            Returns:
                None

        .. note::

            If keyword ``systemInclination`` is present in the dictionary, it
            is assumed that it was generated with ``commonSystemPlane`` set to
            True.  Similarly, if keyword ``starnEZ`` is present, it is assumed
            that ``ZodiacalLight.commonSystemfEZ`` should be true.

        .. warning::

            This method assumes that the exact same targetlist is being used as in the
            object that generated the systems dictionary.  If this assumption is
            violated unexpected results may occur.
        """

        self.a = systems["a"]
        self.e = systems["e"]
        self.I = systems["I"]  # noqa: E741
        self.O = systems["O"]  # noqa: E741
        self.w = systems["w"]
        self.M0 = systems["M0"]
        self.Mp = systems["Mp"]
        self.Rp = systems["Rp"]
        self.p = systems["p"]
        self.plan2star = systems["plan2star"]

        if "systemInclination" in systems:
            self.TargetList.systemInclination = systems["systemInclination"]
            self.commonSystemPlane = True
            if "systemOmega" in systems:
                # leaving as if for backwards compatibility with old dumped
                # params for now
                self.TargetList.systemOmega = systems["systemOmega"]

        if "starnEZ" in systems:
            self.ZodiacalLight.nEZ = systems["starnEZ"]
            self.ZodiacalLight.commonSystemfEZ = True

        self.init_systems()

    def dump_system_params(self, sInd=None):
        """Create a dictionary of time-dependant planet properties for a specific target

        Args:
            sInd (int):
                Index of the target system of interest. Default value (None) will
                return an empty dictionary with the selected parameters and their units.

        Returns:
            dict:
                Dictionary of time-dependant planet properties

        """

        # get planet indices
        if sInd is None:
            pInds = np.array([], dtype=int)
        else:
            pInds = np.where(self.plan2star == sInd)[0]

        # build dictionary
        system_params = {
            "d": self.d[pInds],
            "phi": self.phi[pInds],
            "fEZ": self.fEZ[pInds],
            "dMag": self.dMag[pInds],
            "WA": self.WA[pInds],
        }

        return system_params

    def revise_planets_list(self, pInds):
        """Replaces Simulated Universe planet attributes with filtered values,
        and updates the number of planets.

        Args:
            pInds (~numpy.ndarray(int)):
                Planet indices to keep

        Returns:
            None

        .. warning::

            Throws AssertionError if all planets are removed

        """

        # planet attributes which are floats and should not be filtered
        bad_atts = ["Min"]

        if len(pInds) == 0:
            raise IndexError("Planets list filtered to empty.")

        for att in self.planet_atts:
            if att not in bad_atts:
                if getattr(self, att).size != 0:
                    setattr(self, att, getattr(self, att)[pInds])
        self.nPlans = len(pInds)
        assert self.nPlans, "Planets list is empty: nPlans = %r" % self.nPlans

    def revise_stars_list(self, sInds):
        """Revises the TargetList with filtered values, and updates the
        planets list accordingly.

        Args:
            sInds (~numpy.ndarray(int)):
                Star indices to keep

        Returns:
            None

        """
        self.TargetList.revise_lists(sInds)
        pInds = np.sort(
            np.concatenate([np.where(self.plan2star == x)[0] for x in sInds])
        )
        self.revise_planets_list(pInds)
        for i, ind in enumerate(sInds):
            self.plan2star[np.where(self.plan2star == ind)[0]] = i

    def setup_system_planes(self):
        """
        Helper function that augments the system planes if
        commonSystemPlane is true

        Args:
            None

        Returns:
            None
        """
        if self.commonSystemPlane:
            self.I += self.TargetList.systemInclination[self.plan2star]
            # Ensure all inclinations are in [0, pi]
            self.I = (self.I.to(u.deg).value % 180) * u.deg

            self.O += self.TargetList.systemOmega[self.plan2star]
            # Cut longitude of the ascending nodes to [0, 2pi]
            self.O = (self.O.to(u.deg).value % 360) * u.deg
