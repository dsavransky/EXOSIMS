# -*- coding: utf-8 -*-
from EXOSIMS.Completeness.BrownCompleteness import BrownCompleteness
import numpy as np
import os
import hashlib
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import scipy.integrate as integrate
import astropy.units as u
import pickle
from EXOSIMS.util.memoize import memoize
from tqdm import tqdm
from EXOSIMS.util._numpy_compat import copy_if_needed


class GarrettCompleteness(BrownCompleteness):
    """Analytical Completeness class

    This class contains all variables and methods necessary to perform
    Completeness Module calculations based on Garrett and Savransky 2016
    in exoplanet mission simulation.

    The completeness calculations performed by this method assume that all
    planetary parameters are independently distributed. The probability density
    functions used here are either independent or marginalized from a joint
    probability density function.

    Args:
        order_of_quadrature (int):
            The order of quadrature used in the comp_dmag function's fixed quad
            integration. Higher values will give marginal improvements in the
            comp_calc completeness values, but are slower.
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        updates (nx5 ndarray):
            Completeness values of successive observations of each star in the
            target list (initialized in gen_update)

    """

    def __init__(self, order_of_quadrature=15, **specs):

        # Set order of quadrature used in comp_dmag
        self.order_of_quadrature = int(order_of_quadrature)

        # Run BrownCompleteness init
        BrownCompleteness.__init__(self, **specs)

        self._outspec["order_of_quadrature"] = self.order_of_quadrature

    def completeness_setup(self):
        """Preform any preliminary calculations needed for this flavor of completeness

        For GarrettCompleteness this involves setting up various interpolants.
        See [Garrett2016]_ for details.
        """
        # get unitless values of population parameters
        self.amin = float(self.PlanetPopulation.arange.min().value)
        self.amax = float(self.PlanetPopulation.arange.max().value)
        self.emin = float(self.PlanetPopulation.erange.min())
        self.emax = float(self.PlanetPopulation.erange.max())
        self.pmin = float(self.PlanetPopulation.prange.min())
        self.pmax = float(self.PlanetPopulation.prange.max())
        self.Rmin = float(self.PlanetPopulation.Rprange.min().to("earthRad").value)
        self.Rmax = float(self.PlanetPopulation.Rprange.max().to("earthRad").value)
        if self.PlanetPopulation.constrainOrbits:
            self.rmin = self.amin
            self.rmax = self.amax
        else:
            self.rmin = self.amin * (1.0 - self.emax)
            self.rmax = self.amax * (1.0 + self.emax)
        self.zmin = self.pmin * self.Rmin**2
        self.zmax = self.pmax * self.Rmax**2
        # conversion factor
        self.x = float(u.earthRad.to("AU"))
        # distributions needed
        self.dist_sma = self.PlanetPopulation.dist_sma
        self.dist_eccen = self.PlanetPopulation.dist_eccen
        self.dist_eccen_con = self.PlanetPopulation.dist_eccen_from_sma
        self.dist_albedo = self.PlanetPopulation.dist_albedo
        self.dist_radius = self.PlanetPopulation.dist_radius
        # are any of a, e, p, Rp constant?
        self.aconst = self.amin == self.amax
        self.econst = self.emin == self.emax
        self.pconst = self.pmin == self.pmax
        self.Rconst = self.Rmin == self.Rmax
        # degenerate case where aconst, econst and e = 0
        assert not (
            all([self.aconst, self.econst, self.pconst, self.Rconst]) and self.emax == 0
        ), (
            "At least one parameter (out of semi-major axis, albedo, and radius) must "
            "vary when eccentricity is constant and zero."
        )
        # solve for bstar
        beta = np.linspace(0.0, np.pi, 1000) * u.rad
        Phis = self.PlanetPhysicalModel.calc_Phi(beta)
        # Interpolant for phase function which removes astropy Quantity
        self.Phi = interpolate.InterpolatedUnivariateSpline(
            beta.value, Phis, k=3, ext=1
        )
        self.Phiinv = interpolate.InterpolatedUnivariateSpline(
            Phis[::-1], beta.value[::-1], k=3, ext=1
        )
        # get numerical derivative of phase function
        dPhis = np.zeros(beta.shape)
        db = beta[1].value - beta[0].value
        dPhis[0:1] = (
            -25.0 * Phis[0:1]
            + 48.0 * Phis[1:2]
            - 36.0 * Phis[2:3]
            + 16.0 * Phis[3:4]
            - 3.0 * Phis[4:5]
        ) / (12.0 * db)
        dPhis[-2:-1] = (
            25.0 * Phis[-2:-1]
            - 48.0 * Phis[-3:-2]
            + 36.0 * Phis[-4:-3]
            - 16.0 * Phis[-5:-4]
            + 3.0 * Phis[-6:-5]
        ) / (12.0 * db)
        dPhis[2:-2] = (Phis[0:-4] - 8.0 * Phis[1:-3] + 8.0 * Phis[3:-1] - Phis[4:]) / (
            12.0 * db
        )
        self.dPhi = interpolate.InterpolatedUnivariateSpline(
            beta.value, dPhis, k=3, ext=1
        )
        # solve for bstar
        f = lambda b: 2.0 * np.sin(b) * np.cos(b) * self.Phi(b) + np.sin(
            b
        ) ** 2 * self.dPhi(b)
        self.bstar = float(optimize.root(f, np.pi / 3.0).x)
        # helpful constants
        self.cdmin1 = -2.5 * np.log10(self.pmax * (self.Rmax * self.x / self.rmin) ** 2)
        self.cdmin2 = -2.5 * np.log10(
            self.pmax
            * (self.Rmax * self.x * np.sin(self.bstar)) ** 2
            * self.Phi(self.bstar)
        )
        self.cdmin3 = -2.5 * np.log10(self.pmax * (self.Rmax * self.x / self.rmax) ** 2)
        self.cdmax = -2.5 * np.log10(self.pmin * (self.Rmin * self.x / self.rmax) ** 2)
        self.val = np.sin(self.bstar) ** 2 * self.Phi(self.bstar)
        self.d1 = -2.5 * np.log10(self.pmax * (self.Rmax * self.x / self.rmin) ** 2)
        self.d2 = -2.5 * np.log10(
            self.pmax * (self.Rmax * self.x / self.rmin) ** 2 * self.Phi(self.bstar)
        )
        self.d3 = -2.5 * np.log10(
            self.pmax * (self.Rmax * self.x / self.rmax) ** 2 * self.Phi(self.bstar)
        )
        self.d4 = -2.5 * np.log10(
            self.pmax * (self.Rmax * self.x / self.rmax) ** 2 * self.Phi(np.pi / 2.0)
        )
        self.d5 = -2.5 * np.log10(
            self.pmin * (self.Rmin * self.x / self.rmax) ** 2 * self.Phi(np.pi / 2.0)
        )
        # vectorize scalar methods
        self.rgrand2v = np.vectorize(self.rgrand2, otypes=[np.float64])
        self.f_dmagsv = np.vectorize(self.f_dmags, otypes=[np.float64])
        self.f_sdmagv = np.vectorize(self.f_sdmag, otypes=[np.float64])
        self.f_dmagv = np.vectorize(self.f_dmag, otypes=[np.float64])
        self.f_sv = np.vectorize(self.f_s, otypes=[np.float64])
        self.mindmagv = np.vectorize(self.mindmag, otypes=[np.float64])
        self.maxdmagv = np.vectorize(self.maxdmag, otypes=[np.float64])
        # inverse functions for phase angle
        b1 = np.linspace(0.0, self.bstar, 20000)
        # b < bstar
        self.binv1 = interpolate.InterpolatedUnivariateSpline(
            np.sin(b1) ** 2 * self.Phi(b1), b1, k=1, ext=1
        )
        b2 = np.linspace(self.bstar, np.pi, 20000)
        b2val = np.sin(b2) ** 2 * self.Phi(b2)
        # b > bstar
        self.binv2 = interpolate.InterpolatedUnivariateSpline(
            b2val[::-1], b2[::-1], k=1, ext=1
        )
        if self.rmin != self.rmax:
            # get pdf of r
            self.vprint("Generating pdf of orbital radius")
            r = np.linspace(self.rmin, self.rmax, 1000)
            fr = np.zeros(r.shape)
            for i in range(len(r)):
                fr[i] = self.f_r(r[i])
            self.dist_r = interpolate.InterpolatedUnivariateSpline(r, fr, k=3, ext=1)
            self.vprint("Finished pdf of orbital radius")
        if not all([self.pconst, self.Rconst]):
            # get pdf of p*R**2
            self.vprint("Generating pdf of albedo times planetary radius squared")
            z = np.linspace(self.zmin, self.zmax, 1000)
            fz = np.zeros(z.shape)
            for i in range(len(z)):
                fz[i] = self.f_z(z[i])
            self.dist_z = interpolate.InterpolatedUnivariateSpline(z, fz, k=3, ext=1)
            self.vprint("Finished pdf of albedo times planetary radius squared")

    def target_completeness(self, TL):
        """Generates completeness values for target stars

        This method is called from TargetList __init__ method.

        Args:
            TL (TargetList module):
                TargetList class object

        Returns:
            ~numpy.ndarray(float)):
                int_comp: 1D numpy array of completeness values for each target star

        """

        OS = TL.OpticalSystem
        if TL.calc_char_int_comp:
            mode = list(
                filter(lambda mode: "spec" in mode["inst"]["name"], OS.observingModes)
            )[0]
        else:
            mode = list(filter(lambda mode: mode["detectionMode"], OS.observingModes))[
                0
            ]

        # To limit the amount of computation, we want to find the most common
        # int_dMag value (typically the one the user sets in the input json since
        # int_dMag is either the user input or the intCutoff_dMag).
        vals, counts = np.unique(TL.int_dMag, return_counts=True)
        self.mode_dMag = vals[np.argwhere(counts == np.max(counts))[0][0]]
        mode_dMag_mask = TL.int_dMag == self.mode_dMag

        # important PlanetPopulation attributes
        atts = list(self.PlanetPopulation.__dict__)
        extstr = ""
        for att in sorted(atts, key=str.lower):
            if (
                not callable(getattr(self.PlanetPopulation, att))
                and att != "PlanetPhysicalModel"
            ):
                extstr += "%s: " % att + str(getattr(self.PlanetPopulation, att)) + " "
        # include mode_dMag and intCutoff_dMag
        extstr += (
            "%s: " % "mode_dMag"
            + str(self.mode_dMag)
            + f"intCutoff_dMag: {TL.intCutoff_dMag}"
            + " "
        )
        ext = hashlib.md5(extstr.encode("utf-8")).hexdigest()
        self.filename += ext
        Cpath = os.path.join(self.cachedir, self.filename + ".acomp")

        # calculate separations based on IWA
        IWA = mode["IWA"]
        OWA = mode["OWA"]
        smin = (np.tan(IWA) * TL.dist).to("AU").value
        if np.isinf(OWA):
            smax = np.array([self.rmax] * len(smin))
        else:
            smax = (np.tan(OWA) * TL.dist).to("AU").value
            smax[smax > self.rmax] = self.rmax

        int_comp = np.zeros(smin.shape)
        # calculate dMags based on maximum dMag
        if self.PlanetPopulation.scaleOrbits:
            L = np.where(TL.L > 0, TL.L, 1e-10)  # take care of zero/negative values
            smin = smin / np.sqrt(L)
            smax = smax / np.sqrt(L)
            dMag_vals = TL.int_dMag - 2.5 * np.log10(L)
            separation_mask = smin < self.rmax
            int_comp[separation_mask] = self.comp_s(
                smin[separation_mask], smax[separation_mask], dMag_vals[separation_mask]
            )
        else:
            # In this case we find where the mode dMag value is also in the
            # separation range and use the vectorized integral since they have
            # the same dMag value. Where the dMag values are not the mode we
            # must use comp_s which is slower
            dMag_vals = TL.int_dMag
            separation_mask = smin < self.rmax
            dist_s = self.genComp(Cpath, TL)
            dist_sv = np.vectorize(dist_s.integral, otypes=[np.float64])
            separation_mode_mask = separation_mask & mode_dMag_mask
            separation_not_mode_mask = separation_mask & ~mode_dMag_mask
            int_comp[separation_mode_mask] = dist_sv(
                smin[separation_mode_mask], smax[separation_mode_mask]
            )
            int_comp[separation_not_mode_mask] = self.comp_s(
                smin[separation_not_mode_mask],
                smax[separation_not_mode_mask],
                dMag_vals[separation_not_mode_mask],
            )

        # ensure that completeness values are between 0 and 1
        int_comp = np.clip(int_comp, 0.0, 1.0)

        return int_comp

    def genComp(self, Cpath, TL):
        """Generates function to get completeness values

        Args:
            Cpath (str):
                Path to pickled dictionary containing interpolant function
            TL (TargetList module):
                TargetList class object

        Returns:
            dist_s (callable(s)):
                Marginalized to self.mode_dMag probability density function for
                projected separation

        """

        if os.path.exists(Cpath):
            # dist_s interpolant already exists for parameters
            self.vprint("Loading cached completeness file from %s" % Cpath)
            try:
                with open(Cpath, "rb") as ff:
                    H = pickle.load(ff)
            except UnicodeDecodeError:
                with open(Cpath, "rb") as ff:
                    H = pickle.load(ff, encoding="latin1")
            self.vprint("Completeness loaded from cache.")
            dist_s = H["dist_s"]
        else:
            # generate dist_s interpolant and pickle it
            self.vprint('Cached completeness file not found at "%s".' % Cpath)
            self.vprint("Generating completeness.")
            self.vprint(
                "Marginalizing joint pdf of separation and dMag up to mode_dMag"
            )
            # get pdf of s up to mode_dMag
            s = np.linspace(0.0, self.rmax, 1000)
            fs = np.zeros(s.shape)
            for i in range(len(s)):
                fs[i] = self.f_s(s[i], self.mode_dMag)
            dist_s = interpolate.InterpolatedUnivariateSpline(s, fs, k=3, ext=1)
            self.vprint("Finished marginalization")
            H = {"dist_s": dist_s}
            with open(Cpath, "wb") as ff:
                pickle.dump(H, ff)
            self.vprint("Completeness data stored in %s" % Cpath)

        return dist_s

    def comp_s(self, smin, smax, dMag):
        """Calculates completeness by first integrating over dMag and then
        projected separation.

        Args:
            smin (ndarray):
                Values of minimum projected separation (AU) from instrument
            smax (ndarray):
                Value of maximum projected separation (AU) from instrument
            dMag (ndarray):
                Planet delta magnitude

        Returns:
            comp (ndarray):
                Completeness values

        """
        # cast to arrays
        smin = np.array(smin, ndmin=1, copy=copy_if_needed)
        smax = np.array(smax, ndmin=1, copy=copy_if_needed)
        dMag = np.array(dMag, ndmin=1, copy=copy_if_needed)

        comp = np.zeros(smin.shape)
        for i in tqdm(
            range(len(smin)),
            desc="Integrating pdf over dMag and separation for completeness",
        ):
            comp[i] = integrate.fixed_quad(
                self.f_sv, smin[i], smax[i], args=(dMag[i],), n=50
            )[0]
        # ensure completeness values are between 0 and 1
        comp = np.clip(comp, 0.0, 1.0)

        return comp

    @memoize
    def f_s(self, s, max_dMag):
        """Calculates probability density of projected separation marginalized
        up to max_dMag

        Args:
            s (float):
                Value of projected separation
            max_dMag (float):
                Maximum planet delta magnitude

        Returns:
            f (float):
                Probability density

        """

        if (s == 0.0) or (s == self.rmax):
            f = 0.0
        else:
            d1 = self.mindmag(s)
            d2 = self.maxdmag(s)
            if d2 > max_dMag:
                d2 = max_dMag
            if d1 > d2:
                f = 0.0
            else:
                f = integrate.fixed_quad(self.f_dmagsv, d1, d2, args=(s,), n=50)[0]

        return f

    @memoize
    def f_dmags(self, dmag, s):
        """Calculates the joint probability density of dMag and projected
        separation

        Args:
            dmag (float):
                Planet delta magnitude
            s (float):
                Value of projected separation (AU)

        Returns:
            f (float):
                Value of joint probability density

        """

        if (dmag < self.mindmag(s)) or (dmag > self.maxdmag(s)) or (s == 0.0):
            f = 0.0
        else:
            if self.rmin == self.rmax:
                b1 = np.arcsin(s / self.amax)
                b2 = np.pi - b1
                z1 = 10.0 ** (-0.4 * dmag) * (self.amax / self.x) ** 2 / self.Phi(b1)
                z2 = 10.0 ** (-0.4 * dmag) * (self.amax / self.x) ** 2 / self.Phi(b2)
                f = 0.0
                if (z1 > self.zmin) and (z1 < self.zmax):
                    f += (
                        np.sin(b1)
                        / 2.0
                        * self.dist_z(z1)
                        * z1
                        * np.log(10.0)
                        / (2.5 * self.amax * np.cos(b1))
                    )
                if (z2 > self.zmin) and (z2 < self.zmax):
                    f += (
                        np.sin(b2)
                        / 2.0
                        * self.dist_z(z2)
                        * z2
                        * np.log(10.0)
                        / (-2.5 * self.amax * np.cos(b2))
                    )
            else:
                ztest = (s / self.x) ** 2 * 10.0 ** (-0.4 * dmag) / self.val
                if self.PlanetPopulation.pfromRp:
                    f = 0.0
                    minR = self.PlanetPopulation.Rbs[:-1]
                    maxR = self.PlanetPopulation.Rbs[1:]
                    for i in range(len(minR)):
                        ptest = self.PlanetPopulation.get_p_from_Rp(
                            minR[i] * u.earthRad
                        )
                        Rtest = np.sqrt(ztest / ptest)
                        if Rtest > minR[i]:
                            if Rtest > self.Rmin:
                                Rl = Rtest
                            else:
                                Rl = self.Rmin
                        else:
                            if self.Rmin > minR[i]:
                                Rl = self.Rmin
                            else:
                                Rl = minR[i]
                        if self.Rmax > maxR[i]:
                            Ru = maxR[i]
                        else:
                            Ru = self.Rmax
                        if Rl < Ru:
                            f += integrate.fixed_quad(
                                self.f_dmagsRp, Rl, Ru, args=(dmag, s), n=200
                            )[0]
                elif ztest >= self.zmax:
                    f = 0.0
                elif self.pconst & self.Rconst:
                    f = self.f_dmagsz(self.zmin, dmag, s)
                else:
                    if ztest < self.zmin:
                        f = integrate.fixed_quad(
                            self.f_dmagsz, self.zmin, self.zmax, args=(dmag, s), n=200
                        )[0]
                    else:
                        f = integrate.fixed_quad(
                            self.f_dmagsz, ztest, self.zmax, args=(dmag, s), n=200
                        )[0]
        return f

    def f_dmagsz(self, z, dmag, s):
        """Calculates the joint probability density of albedo times planetary
        radius squared, dMag, and projected separation

        Args:
            z (ndarray):
                Values of albedo times planetary radius squared
            dmag (float):
                Planet delta magnitude
            s (float):
                Value of projected separation

        Returns:
            f (ndarray):
                Values of joint probability density

        """
        if not isinstance(z, np.ndarray):
            z = np.array(z, ndmin=1, copy=copy_if_needed)

        vals = (s / self.x) ** 2 * 10.0 ** (-0.4 * dmag) / z

        f = np.zeros(z.shape)
        fa = f[vals < self.val]
        za = z[vals < self.val]
        valsa = vals[vals < self.val]
        b1 = self.binv1(valsa)
        b2 = self.binv2(valsa)
        r1 = s / np.sin(b1)
        r2 = s / np.sin(b2)
        good1 = (r1 > self.rmin) & (r1 < self.rmax)
        good2 = (r2 > self.rmin) & (r2 < self.rmax)
        if self.pconst & self.Rconst:
            fa[good1] = (
                np.sin(b1[good1])
                / 2.0
                * self.dist_r(r1[good1])
                / np.abs(self.Jac(b1[good1]))
            )
            fa[good2] += (
                np.sin(b2[good2])
                / 2.0
                * self.dist_r(r2[good2])
                / np.abs(self.Jac(b2[good2]))
            )
        else:
            fa[good1] = (
                self.dist_z(za[good1])
                * np.sin(b1[good1])
                / 2.0
                * self.dist_r(r1[good1])
                / np.abs(self.Jac(b1[good1]))
            )
            fa[good2] += (
                self.dist_z(za[good2])
                * np.sin(b2[good2])
                / 2.0
                * self.dist_r(r2[good2])
                / np.abs(self.Jac(b2[good2]))
            )

        f[vals < self.val] = fa

        return f

    def f_dmagsRp(self, Rp, dmag, s):
        """Calculates the joint probability density of planetary radius,
        dMag, and projected separation

        Args:
            Rp (ndarray):
                Values of planetary radius
            dmag (float):
                Planet delta magnitude
            s (float):
                Value of projected separation

        Returns:
            f (ndarray):
                Values of joint probability density

        """
        if not isinstance(Rp, np.ndarray):
            Rp = np.array(Rp, ndmin=1, copy=copy_if_needed)

        vals = (
            (s / self.x) ** 2
            * 10.0 ** (-0.4 * dmag)
            / self.PlanetPopulation.get_p_from_Rp(Rp * u.earthRad)
            / Rp**2
        )

        f = np.zeros(Rp.shape)
        fa = f[vals < self.val]
        Rpa = Rp[vals < self.val]
        valsa = vals[vals < self.val]
        b1 = self.binv1(valsa)
        b2 = self.binv2(valsa)
        r1 = s / np.sin(b1)
        r2 = s / np.sin(b2)
        good1 = (r1 > self.rmin) & (r1 < self.rmax)
        good2 = (r2 > self.rmin) & (r2 < self.rmax)
        if self.pconst & self.Rconst:
            fa[good1] = (
                np.sin(b1[good1])
                / 2.0
                * self.dist_r(r1[good1])
                / np.abs(self.Jac(b1[good1]))
            )
            fa[good2] += (
                np.sin(b2[good2])
                / 2.0
                * self.dist_r(r2[good2])
                / np.abs(self.Jac(b2[good2]))
            )
        else:
            fa[good1] = (
                self.dist_radius(Rpa[good1])
                * np.sin(b1[good1])
                / 2.0
                * self.dist_r(r1[good1])
                / np.abs(self.Jac(b1[good1]))
            )
            fa[good2] += (
                self.dist_radius(Rpa[good2])
                * np.sin(b2[good2])
                / 2.0
                * self.dist_r(r2[good2])
                / np.abs(self.Jac(b2[good2]))
            )

        f[vals < self.val] = fa

        return f

    def mindmag(self, s):
        """Calculates the minimum value of dMag for projected separation

        Args:
            s (float):
                Projected separations (AU)

        Returns:
            mindmag (float):
                Minimum planet delta magnitude
        """
        if s == 0.0:
            mindmag = self.cdmin1
        elif s < self.rmin * np.sin(self.bstar):
            mindmag = self.cdmin1 - 2.5 * np.log10(self.Phi(np.arcsin(s / self.rmin)))
        elif s < self.rmax * np.sin(self.bstar):
            mindmag = self.cdmin2 + 5.0 * np.log10(s)
        elif s <= self.rmax:
            mindmag = self.cdmin3 - 2.5 * np.log10(self.Phi(np.arcsin(s / self.rmax)))
        else:
            mindmag = np.inf

        return mindmag

    def maxdmag(self, s):
        """Calculates the maximum value of dMag for projected separation

        Args:
            s (float):
                Projected separation (AU)

        Returns:
            maxdmag (float):
                Maximum planet delta magnitude

        """

        if s == 0.0:
            maxdmag = self.cdmax - 2.5 * np.log10(self.Phi(np.pi))
        elif s < self.rmax:
            maxdmag = self.cdmax - 2.5 * np.log10(
                np.abs(self.Phi(np.pi - np.arcsin(s / self.rmax)))
            )
        else:
            maxdmag = self.cdmax - 2.5 * np.log10(self.Phi(np.pi / 2.0))

        return maxdmag

    def Jac(self, b):
        """Calculates determinant of the Jacobian transformation matrix to get
        the joint probability density of dMag and s

        Args:
            b (ndarray):
                Phase angles

        Returns:
            f (ndarray):
                Determinant of Jacobian transformation matrix

        """

        f = -2.5 / (self.Phi(b) * np.log(10.0)) * self.dPhi(b) * np.sin(
            b
        ) - 5.0 / np.log(10.0) * np.cos(b)

        return f

    def rgrand1(self, e, a, r):
        """Calculates first integrand for determinining probability density of
        orbital radius

        Args:
            e (ndarray):
                Values of eccentricity
            a (float):
                Values of semi-major axis in AU
            r (float):
                Values of orbital radius in AU

        Returns:
            f (ndarray):
                Values of first integrand

        """
        if self.PlanetPopulation.constrainOrbits:
            f = 1.0 / (np.sqrt((a * e) ** 2 - (a - r) ** 2)) * self.dist_eccen_con(e, a)
        else:
            f = 1.0 / (np.sqrt((a * e) ** 2 - (a - r) ** 2)) * self.dist_eccen(e)

        return f

    def rgrand2(self, a, r):
        """Calculates second integrand for determining probability density of
        orbital radius

        Args:
            a (float):
                Value of semi-major axis in AU
            r (float):
                Value of orbital radius in AU

        Returns:
            f (float):
                Value of second integrand

        """
        emin1 = np.abs(1.0 - r / a)
        emin1 *= 1.0 + 1e-3
        if emin1 < self.emin:
            emin1 = self.emin

        if emin1 >= self.emax:
            f = 0.0
        else:
            if self.PlanetPopulation.constrainOrbits:
                if a <= 0.5 * (self.amin + self.amax):
                    elim = 1.0 - self.amin / a
                else:
                    elim = self.amax / a - 1.0
                if emin1 > elim:
                    f = 0.0
                else:
                    f = (
                        self.dist_sma(a)
                        / a
                        * integrate.fixed_quad(
                            self.rgrand1, emin1, elim, args=(a, r), n=60
                        )[0]
                    )
            else:
                f = (
                    self.dist_sma(a)
                    / a
                    * integrate.fixed_quad(
                        self.rgrand1, emin1, self.emax, args=(a, r), n=60
                    )[0]
                )

        return f

    def rgrandac(self, e, a, r):
        """Calculates integrand for determining probability density of orbital
        radius when semi-major axis is constant

        Args:
            e (ndarray):
                Values of eccentricity
            a (float):
                Value of semi-major axis in AU
            r (float):
                Value of orbital radius in AU

        Returns:
            f (ndarray):
                Value of integrand

        """
        if self.PlanetPopulation.constrainOrbits:
            f = (
                r
                / (np.pi * a * np.sqrt((a * e) ** 2 - (a - r) ** 2))
                * self.dist_eccen_con(e, a)
            )
        else:
            f = (
                r
                / (np.pi * a * np.sqrt((a * e) ** 2 - (a - r) ** 2))
                * self.dist_eccen(e)
            )

        return f

    def rgrandec(self, a, e, r):
        """Calculates integrand for determining probability density of orbital
        radius when eccentricity is constant

        Args:
            a (ndarray):
                Values of semi-major axis in AU
            e (float):
                Value of eccentricity
            r (float):
                Value of orbital radius in AU

        Returns:
            f (float):
                Value of integrand
        """

        f = r / (np.pi * a * np.sqrt((a * e) ** 2 - (a - r) ** 2)) * self.dist_sma(a)

        return f

    def f_r(self, r):
        """Calculates the probability density of orbital radius

        Args:
            r (float):
                Value of semi-major axis in AU

        Returns:
            f (float):
                Value of probability density

        """
        # takes scalar input
        if (r == self.rmin) or (r == self.rmax):
            f = 0.0
        else:
            if self.aconst & self.econst:
                if self.emin == 0.0:
                    f = self.dist_sma(r)
                else:
                    if r > self.amin * (1.0 - self.emin):
                        f = r / (
                            np.pi
                            * self.amin
                            * np.sqrt(
                                (self.amin * self.emin) ** 2 - (self.amin - r) ** 2
                            )
                        )
                    else:
                        f = 0.0
            elif self.aconst:
                etest1 = 1.0 - r / self.amin
                etest2 = r / self.amin - 1.0
                if self.emax < etest1:
                    f = 0.0
                else:
                    if r < self.amin:
                        if self.emin > etest1:
                            low = self.emin
                        else:
                            low = etest1
                    else:
                        if self.emin > etest2:
                            low = self.emin
                        else:
                            low = etest2
                    f = integrate.fixed_quad(
                        self.rgrandac, low, self.emax, args=(self.amin, r), n=60
                    )[0]
            elif self.econst:
                if self.emin == 0.0:
                    f = self.dist_sma(r)
                else:
                    atest1 = r / (1.0 - self.emin)
                    atest2 = r / (1.0 + self.emin)
                    if self.amax < atest1:
                        high = self.amax
                    else:
                        high = atest1
                    if self.amin < atest2:
                        low = atest2
                    else:
                        low = self.amin
                    f = integrate.fixed_quad(
                        self.rgrandec, low, high, args=(self.emin, r), n=60
                    )[0]
            else:
                if self.PlanetPopulation.constrainOrbits:
                    a1 = 0.5 * (self.amin + r)
                    a2 = 0.5 * (self.amax + r)
                else:
                    a1 = r / (1.0 + self.emax)
                    a2 = r / (1.0 - self.emax)
                    if a1 < self.amin:
                        a1 = self.amin
                    if a2 > self.amax:
                        a2 = self.amax
                f = (
                    r
                    / np.pi
                    * integrate.fixed_quad(self.rgrand2v, a1, a2, args=(r,), n=60)[0]
                )

        return f

    def Rgrand(self, R, z):
        """Calculates integrand for determining probability density of albedo
        times planetary radius squared

        Args:
            R (ndarray):
                Values of planetary radius
            z (float):
                Value of albedo times planetary radius squared

        Returns:
            f (ndarray):
                Values of integrand

        """

        f = self.dist_albedo(z / R**2) * self.dist_radius(R) / R**2

        return f

    def f_z(self, z):
        """Calculates probability density of albedo times planetary radius
        squared

        Args:
            z (float):
                Value of albedo times planetary radius squared

        Returns:
            f (float):
                Probability density

        """

        # takes scalar input
        if (z < self.pmin * self.Rmin**2) or (z > self.pmax * self.Rmax**2):
            f = 0.0
        else:
            if self.pconst & self.Rconst:
                f = 1.0
            elif self.pconst:
                f = (
                    1.0
                    / (2.0 * np.sqrt(self.pmin * z))
                    * self.dist_radius(np.sqrt(z / self.pmin))
                )
            elif self.Rconst:
                f = 1.0 / self.Rmin**2 * self.dist_albedo(z / self.Rmin**2)
            else:
                R1 = np.sqrt(z / self.pmax)
                R2 = np.sqrt(z / self.pmin)
                if R1 < self.Rmin:
                    R1 = self.Rmin
                if R2 > self.Rmax:
                    R2 = self.Rmax
                if R1 > R2:
                    f = 0.0
                else:
                    f = integrate.fixed_quad(self.Rgrand, R1, R2, args=(z,), n=200)[0]

        return f

    def s_bound(self, dmag, smax):
        """Calculates the bounding value of projected separation for dMag

        Args:
            dmag (float):
                Planet delta magnitude
            smax (float):
                maximum projected separation (AU)

        Returns:
            sb (float):
                boundary value of projected separation (AU)
        """

        if dmag < self.d1:
            s = 0.0
        elif (dmag > self.d1) and (dmag <= self.d2):
            s = self.rmin * np.sin(
                self.Phiinv(
                    self.rmin**2
                    * 10.0 ** (-0.4 * dmag)
                    / (self.pmax * (self.Rmax * self.x) ** 2)
                )
            )
        elif (dmag > self.d2) and (dmag <= self.d3):
            s = np.sin(self.bstar) * np.sqrt(
                self.pmax
                * (self.Rmax * self.x) ** 2
                * self.Phi(self.bstar)
                / 10.0 ** (-0.4 * dmag)
            )
        elif (dmag > self.d3) and (dmag <= self.d4):
            s = self.rmax * np.sin(
                self.Phiinv(
                    self.rmax**2
                    * 10.0 ** (-0.4 * dmag)
                    / (self.pmax * (self.Rmax * self.x) ** 2)
                )
            )
        elif (dmag > self.d4) and (dmag <= self.d5):
            s = smax
        else:
            s = self.rmax * np.sin(
                np.pi
                - self.Phiinv(
                    10.0 ** (-0.4 * dmag)
                    * self.rmax**2
                    / (self.pmin * (self.Rmin * self.x) ** 2)
                )
            )

        return s

    def f_sdmag(self, s, dmag):
        """Calculates the joint probability density of projected separation and
        dMag by flipping the order of f_dmags

        Args:
            s (float):
                Value of projected separation (AU)
            dmag (float):
                Planet delta magnitude

        Returns:
            f (float):
                Value of joint probability density

        """
        return self.f_dmags(dmag, s)

    @memoize
    def f_dmag(self, dmag, smin, smax):
        """Calculates probability density of dMag by integrating over projected
        separation

        Args:
            dmag (float):
                Planet delta magnitude
            smin (float):
                Value of minimum projected separation (AU) from instrument
            smax (float):
                Value of maximum projected separation (AU) from instrument

        Returns:
            f (float):
                Value of probability density

        """
        if dmag < self.mindmag(smin):
            f = 0.0
        else:
            su = self.s_bound(dmag, smax)
            if su > smax:
                su = smax
            if su < smin:
                f = 0.0
            else:
                f = integrate.fixed_quad(self.f_sdmagv, smin, su, args=(dmag,), n=50)[0]

        return f

    def comp_dmag(self, smin, smax, max_dMag):
        """Calculates completeness by first integrating over projected
        separation and then dMag.

        Args:
            smin (ndarray):
                Values of minimum projected separation (AU) from instrument
            smax (ndarray):
                Value of maximum projected separation (AU) from instrument
            max_dMag (float ndarray):
                Maximum planet delta magnitude

        Returns:
            comp (ndarray):
                Completeness values

        """
        # cast to arrays
        smin = np.array(smin, ndmin=1, copy=copy_if_needed)
        smax = np.array(smax, ndmin=1, copy=copy_if_needed)
        max_dMag = np.array(max_dMag, ndmin=1, copy=copy_if_needed)
        dmax = -2.5 * np.log10(
            float(
                self.PlanetPopulation.prange[0]
                * (self.PlanetPopulation.Rprange[0] / self.PlanetPopulation.rrange[1])
                ** 2
            )
            * 1e-11
        )
        max_dMag[max_dMag > dmax] = dmax

        comp = np.zeros(smin.shape)
        for i in tqdm(
            range(len(smin)),
            desc=(
                "Calculating completeness values by integrating with order of "
                "quadrature {}".format(self.order_of_quadrature)
            ),
        ):
            d1 = self.mindmag(smin[i])
            if d1 > max_dMag[i]:
                comp[i] = 0.0
            else:
                comp[i] = integrate.fixed_quad(
                    self.f_dmagv,
                    d1,
                    max_dMag[i],
                    args=(smin[i], min(smax[i], np.finfo(np.float32).max)),
                    n=self.order_of_quadrature,
                )[0]

        # ensure completeness values are between 0 and 1
        comp = np.clip(comp, 0.0, 1.0)

        return comp

    def comp_calc(self, smin, smax, dMag):
        """Calculates completeness for given minimum and maximum separations
        and dMag

        Note: this method assumes scaling orbits when scaleOrbits == True has
        already occurred for smin, smax, dMag inputs

        Args:
            smin (float ndarray):
                Minimum separation(s) in AU
            smax (float ndarray):
                Maximum separation(s) in AU
            dMag (float ndarray):
                Difference in brightness magnitude

        Returns:
            comp (float ndarray):
                Completeness value(s)

        """

        comp = self.comp_dmag(smin, smax, dMag)

        return comp

    def calc_fdmag(self, dMag, smin, smax):
        """Calculates probability density of dMag by integrating over projected
        separation

        Args:
            dMag (float ndarray):
                Planet delta magnitude(s)
            smin (float ndarray):
                Value of minimum projected separation (AU) from instrument
            smax (float ndarray):
                Value of maximum projected separation (AU) from instrument

        Returns:
            f (float):
                Value of probability density

        """

        f = self.f_dmagv(dMag, smin, smax)

        return f
