import astropy.units as u
import numpy as np

from EXOSIMS.Prototypes.SimulatedUniverse import SimulatedUniverse


class DulzPlavchanUniverseEarthsOnly(SimulatedUniverse):
    """Simulated Universe module based on Dulz and Plavchan occurrence rates.
    attributes:
        earthPF (boolean):
            indicates whether to use Earth's phase function or not
    """

    def __init__(self, earthPF=True, guarantee_earths=False, **specs):
        self.earthPF = earthPF
        self.guarantee_earths = guarantee_earths
        SimulatedUniverse.__init__(self, **specs)

    def gen_physical_properties(self, **specs):
        """Generating universe based on Dulz and Plavchan occurrence rate tables."""

        PPop = self.PlanetPopulation
        TL = self.TargetList

        # treat eta as the rate parameter of a Poisson distribution
        targetSystems = np.random.poisson(lam=PPop.eta, size=TL.nStars)
        plan2star = []
        for j, n in enumerate(targetSystems):
            plan2star = np.hstack((plan2star, [j] * n))

        if self.guarantee_earths:
            a, e, p, Rp, plan2star = self.gen_earths(len(targetSystems), plan2star)
            self.a = a * u.AU
            self.e = np.array(e)
            self.p = np.array(p)
            self.Rp = Rp * u.R_earth
        else:
            a, e, p, Rp = PPop.gen_plan_params(len(plan2star))

            # filter to just Earth candidates
            # modify as needed NB: a has no yet been scaled by \sqrt{L}
            # so effectively everything is at 1 solar luminosity
            inds = (
                (a < 1.67 * u.AU)
                & (a > 0.95 * u.AU)
                & (Rp < 1.4 * u.R_earth)
                & (Rp > 0.8 / np.sqrt(a.value) * u.R_earth)
            )
            self.a = a[inds]
            self.e = e[inds]
            self.p = p[inds]
            self.Rp = Rp[inds]
            plan2star = plan2star[inds]

        self.plan2star = plan2star.astype(int)
        self.sInds = np.unique(self.plan2star)
        self.nPlans = len(self.plan2star)

        if PPop.scaleOrbits:
            self.a *= np.sqrt(TL.L[self.plan2star])

        # sample all of the orbital and physical parameters
        self.I, self.O, self.w = PPop.gen_angles(
            self.nPlans,
            commonSystemPlane=self.commonSystemPlane,
            commonSystemPlaneParams=self.commonSystemPlaneParams,
        )
        self.setup_system_planes()

        self.gen_M0()  # initial mean anomaly
        self.Mp = PPop.MfromRp(self.Rp)  # mass

        # Use Earth Phase Function
        if self.earthPF:
            self.phiIndex = (
                np.ones(self.nPlans, dtype=int) * 2
            )  # Used to switch select specific phase function for each planet
        else:
            self.phiIndex = np.asarray([])
        ZL = self.ZodiacalLight
        if self.commonSystemnEZ:
            # Assign the same nEZ to all planets in the system
            self.nEZ = ZL.gen_systemnEZ(TL.nStars)[self.plan2star]
        else:
            # Assign a unique nEZ to each planet
            self.nEZ = ZL.gen_systemnEZ(self.nPlans)

    def gen_earths(self, nsystems, plan2star):
        # Hacky way to guarantee that an Earth-like planet is in every system
        # a, e, p, Rp = (
        #     np.zeros(nsystems),
        #     np.zeros(nsystems),
        #     np.zeros(nsystems),
        #     np.zeros(nsystems),
        # )
        a, e, p, Rp, new_plan2star = [], [], [], [], []
        for ind in range(nsystems):
            _a, _e, _p, _Rp = self.create_earths(sum(plan2star == ind))
            a.extend(_a)
            e.extend(_e)
            p.extend(_p)
            Rp.extend(_Rp)
            new_plan2star.extend(np.ones(len(_a), dtype=int) * ind)
        return a, e, p, Rp, np.array(new_plan2star, dtype=int)

    def create_earths(self, nplans):
        _a, _e, _p, _Rp = self.PlanetPopulation.gen_plan_params(nplans)

        success = (
            (_a < 1.67 * u.AU)
            & (_a > 0.95 * u.AU)
            & (_Rp < 1.4 * u.R_earth)
            & (_Rp > 0.8 / np.sqrt(_a.value) * u.R_earth)
        )
        if np.any(success):
            success_inds = np.where(success)[0]
            return (
                _a[success_inds].to(u.AU).value,
                _e[success_inds],
                _p[success_inds],
                _Rp[success_inds].to(u.R_earth).value,
            )
        else:
            return self.create_earths(nplans)
