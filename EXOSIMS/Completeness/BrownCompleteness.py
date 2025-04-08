# -*- coding: utf-8 -*-
import hashlib
import os
import pickle
import time

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy import interpolate
from tqdm import tqdm

from EXOSIMS.Prototypes.Completeness import Completeness
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.eccanom import eccanom


class BrownCompleteness(Completeness):
    """Completeness class template

    This class contains all variables and methods necessary to perform
    Completeness Module calculations in exoplanet mission simulation.

    Args:
        specs:
            user specified values

    Attributes:
        Nplanets (int):
            Number of planets for initial completeness Monte Carlo simulation
        classpath (str):
            Path on disk to Brown Completeness
        filename (str):
            Name of file where completeness interpolant is stored
        updates (float nx5 ndarray):
            Completeness values of successive observations of each star in the
            target list (initialized in gen_update)

    """

    def __init__(self, Nplanets=1e8, **specs):
        # Number of planets to sample
        self.Nplanets = int(Nplanets)

        # Run Completeness prototype __init__
        Completeness.__init__(self, **specs)

        # copy Nplanets into outspec
        self._outspec["Nplanets"] = self.Nplanets

    def completeness_setup(self):
        """Preform any preliminary calculations needed for this flavor of completeness

        For BrownCompleteness this includes generating a 2D histogram of s vs. dMag for
        the planet population and creating interpolants over it.
        """

        # set up "ensemble visit photometric and obscurational completeness"
        # interpolant for initial completeness values
        # bins for interpolant
        bins = 1000
        # xedges is array of separation values for interpolant
        if self.PlanetPopulation.constrainOrbits:
            self.xedges = np.linspace(
                0.0, self.PlanetPopulation.arange[1].to_value(u.AU), bins + 1
            )
        else:
            self.xedges = np.linspace(
                0.0, self.PlanetPopulation.rrange[1].to_value(u.AU), bins + 1
            )

        # yedges is array of delta magnitude values for interpolant
        self.ymin = -2.5 * np.log10(
            float(
                self.PlanetPopulation.prange[1]
                * (self.PlanetPopulation.Rprange[1] / self.PlanetPopulation.rrange[0])
                ** 2
            )
        )
        self.ymax = -2.5 * np.log10(
            float(
                self.PlanetPopulation.prange[0]
                * (self.PlanetPopulation.Rprange[0] / self.PlanetPopulation.rrange[1])
                ** 2
            )
            * 1e-11
        )
        self.yedges = np.linspace(self.ymin, self.ymax, bins + 1)
        # number of planets for each Monte Carlo simulation
        nplan = 1e6
        # number of simulations to perform (must be integer)
        steps = int(np.floor(self.Nplanets / nplan))

        # path to 2D completeness pdf array for interpolation
        Cpath = os.path.join(self.cachedir, self.filename + ".comp")
        _Cpdf, xedges2, yedges2 = self.genC(
            Cpath,
            nplan,
            self.xedges,
            self.yedges,
            steps,
            remainder=self.Nplanets - steps * nplan,
        )

        xcent = 0.5 * (xedges2[1:] + xedges2[:-1])
        ycent = 0.5 * (yedges2[1:] + yedges2[:-1])
        xnew = np.hstack((0.0, xcent, self.PlanetPopulation.rrange[1].to_value(u.AU)))
        ynew = np.hstack((self.ymin, ycent, self.ymax))
        Cpdf = np.pad(_Cpdf, 1, mode="constant")

        # save interpolant to object
        # Cpdf is a 2D array of shape (n_dMag, n_s)
        self.Cpdf = Cpdf
        self.EVPOCpdf = interpolate.RectBivariateSpline(xnew, ynew, Cpdf.T)
        self.EVPOC = np.vectorize(self.EVPOCpdf.integral, otypes=[np.float64])
        # This calculates the cumulative sum moving up the dMag axis
        # and still has shape (n_dMag, n_s)
        ddMag = ycent[1] - ycent[0]
        self.ds = xcent[1] - xcent[0]
        # Get the cumulative sum of the pdf in dMag and multiply by the dMag
        # bin height to precalculate that part of the integral
        self.Cpdfsum = np.cumsum(_Cpdf, axis=0) * ddMag
        self.Cpdfsum_interp = interpolate.RegularGridInterpolator(
            (xcent, ycent), self.Cpdfsum.T
        )
        self.xnew = xnew
        self.ynew = ynew
        self.xcent = xcent
        self.ycent = ycent

    def generate_cache_names(self, **specs):
        """Generate unique filenames for cached products"""

        # filename for completeness interpolant stored in a pickled .comp file
        self.filename = (
            self.PlanetPopulation.__class__.__name__
            + self.PlanetPhysicalModel.__class__.__name__
            + self.__class__.__name__
            + str(self.Nplanets)
            + self.PlanetPhysicalModel.whichPlanetPhaseFunction
        )

        # filename for dynamic completeness array in a pickled .dcomp file
        self.dfilename = (
            self.PlanetPopulation.__class__.__name__
            + self.PlanetPhysicalModel.__class__.__name__
            + specs["modules"]["OpticalSystem"]
            + specs["modules"]["StarCatalog"]
            + specs["modules"]["TargetList"]
        )
        # Remove spaces from string
        self.dfilename = self.dfilename.replace(" ", "")

        atts = list(self.PlanetPopulation.__dict__)
        self.extstr = ""
        for att in sorted(atts, key=str.lower):
            if (
                not (callable(getattr(self.PlanetPopulation, att)))
                and (att != "PlanetPhysicalModel")
                and (att != "cachedir")
                and (att != "_outspec")
            ):
                self.extstr += "%s: " % att + str(getattr(self.PlanetPopulation, att))
        ext = hashlib.md5(self.extstr.encode("utf-8")).hexdigest()
        self.filename += ext
        # Remove spaces from string (in the case of prototype use)
        self.filename = self.filename.replace(" ", "")

    def target_completeness(self, TL):
        """Generates completeness values for target stars using average case
        values

        This method is called from TargetList __init__ method.

        Args:
            TL (TargetList module):
                TargetList class object

        Returns:
            float ndarray:
                Completeness values for each target star

        """

        self.vprint("Generating int_comp values")
        OS = TL.OpticalSystem
        if TL.calc_char_int_comp:
            mode = list(
                filter(lambda mode: "spec" in mode["inst"]["name"], OS.observingModes)
            )[0]
        else:
            mode = list(filter(lambda mode: mode["detectionMode"], OS.observingModes))[
                0
            ]

        IWA = mode["IWA"]
        OWA = mode["OWA"]
        smin = np.tan(IWA) * TL.dist
        if np.isinf(OWA):
            smax = np.array([self.xedges[-1]] * len(smin)) * u.AU
        else:
            smax = np.tan(OWA) * TL.dist
            smax[smax > self.PlanetPopulation.rrange[1]] = self.PlanetPopulation.rrange[
                1
            ]

        int_comp = np.zeros(smin.shape)
        if self.PlanetPopulation.scaleOrbits:
            L = np.where(TL.L > 0, TL.L, 1e-10)  # take care of zero/negative values
            smin = smin / np.sqrt(L)
            smax = smax / np.sqrt(L)
            scaled_dMag = TL.int_dMag - 2.5 * np.log10(L)
            mask = (scaled_dMag > self.ymin) & (smin < self.PlanetPopulation.rrange[1])
            int_comp[mask] = self.comp_calc(
                smin[mask].to_value(u.AU), smax[mask].to_value(u.AU), scaled_dMag[mask]
            )
        else:
            mask = smin < self.PlanetPopulation.rrange[1]
            int_comp[mask] = self.comp_calc(
                smin[mask].to_value(u.AU), smax[mask].to_value(u.AU), TL.int_dMag[mask]
            )

        int_comp[int_comp < 1e-6] = 0.0
        # ensure that completeness is between 0 and 1
        int_comp = np.clip(int_comp, 0.0, 1.0)

        return int_comp

    def gen_update(self, TL):
        """Generates dynamic completeness values for multiple visits of each
        star in the target list

        Args:
            TL (TargetList):
                TargetList class object

        """

        OS = TL.OpticalSystem
        PPop = TL.PlanetPopulation

        # get name for stored dynamic completeness updates array
        # inner and outer working angles for detection mode
        mode = list(filter(lambda mode: mode["detectionMode"], OS.observingModes))[0]
        IWA = mode["IWA"]
        OWA = mode["OWA"]
        extstr = (
            self.extstr
            + "IWA: "
            + str(IWA)
            + " OWA: "
            + str(OWA)
            + " nStars: "
            + str(TL.nStars)
        )
        ext = hashlib.md5(extstr.encode("utf-8")).hexdigest()
        self.dfilename += ext
        self.dfilename += ".dcomp"

        path = os.path.join(self.cachedir, self.dfilename)
        # if the 2D completeness update array exists as a .dcomp file load it
        if os.path.exists(path):
            self.vprint('Loading cached dynamic completeness array from "%s".' % path)
            try:
                with open(path, "rb") as ff:
                    self.updates = pickle.load(ff)
            except UnicodeDecodeError:
                with open(path, "rb") as ff:
                    self.updates = pickle.load(ff, encoding="latin1")
            self.vprint("Dynamic completeness array loaded from cache.")
        else:
            # run Monte Carlo simulation and pickle the resulting array
            self.vprint('Cached dynamic completeness array not found at "%s".' % path)
            # dynamic completeness values: rows are stars, columns are number of visits
            self.updates = np.zeros((TL.nStars, 5))
            # number of planets to simulate
            nplan = int(2e4)
            # sample quantities which do not change in time
            a, e, p, Rp = PPop.gen_plan_params(nplan)
            a = a.to_value(u.AU)
            acubed = a**3
            # sample angles
            I, O, w = PPop.gen_angles(nplan)
            I = I.to_value(u.rad)
            O = O.to_value(u.rad)
            w = w.to_value(u.rad)
            Mp = PPop.gen_mass(nplan)  # M_earth
            rmax = a * (1.0 + e)  # AU
            # sample quantity which will be updated
            M = np.random.uniform(high=2.0 * np.pi, size=nplan)
            newM = np.zeros((nplan,))
            dist_AU = TL.dist.to_value(u.AU)
            # population values
            smin = np.tan(IWA).value * dist_AU
            if np.isfinite(OWA):
                smax = np.tan(OWA).value * dist_AU
            else:
                smax = np.array(
                    [np.max(PPop.arange.to_value(u.AU)) * (1.0 + np.max(PPop.erange))]
                    * TL.nStars
                )
            _MsTrue = TL.MsTrue.to_value(u.kg)
            _Mp = Mp.to_value(u.kg)
            _G = const.G.to_value(u.AU**3 / u.day**2 / u.kg)
            mu = _G * (_Mp + _MsTrue.reshape(-1, 1))
            n_arr = np.sqrt(mu / acubed)  # in 1/day
            # normalization time equation from Brown 2015
            _dt_L_fac = (
                58
                * (TL.L / 0.83) ** (3.0 / 4.0)
                * (TL.MsTrue / (0.91 * u.M_sun)) ** (1.0 / 2.0)
            ).value  # days
            _sin_I, _cos_I = np.sin(I), np.cos(I)
            _sin_O, _cos_O = np.sin(O), np.cos(O)
            _cos_w, _sin_w = np.cos(w), np.sin(w)
            a1 = _cos_O * _cos_w - _sin_O * _sin_w * _cos_I
            a2 = _sin_O * _cos_w + _cos_O * _sin_w * _cos_I
            a3 = _sin_w * _sin_I
            A = np.hstack(
                (
                    a1.reshape(len(a1), 1),
                    a2.reshape(len(a2), 1),
                    a3.reshape(len(a3), 1),
                )
            )
            b1 = -_cos_O * _sin_w - _sin_O * _cos_w * _cos_I
            b2 = -_sin_O * _sin_w + _cos_O * _cos_w * _cos_I
            b3 = _cos_w * _sin_I
            B = np.hstack(
                (
                    b1.reshape(len(b1), 1),
                    b2.reshape(len(b2), 1),
                    b3.reshape(len(b3), 1),
                )
            )
            max_intCutoff_dMag = np.max(TL.intCutoff_dMag)
            a_one_minus_e2 = a * np.sqrt(1 - e**2)
            # fill dynamic completeness values
            for sInd in tqdm(
                range(TL.nStars), desc="Calculating dynamic completeness for each star"
            ):
                n = n_arr[sInd]
                dt = _dt_L_fac[sInd]  # days
                # remove rmax < smin
                pInds = np.where(rmax > smin[sInd])[0]
                # calculate for 5 successive observations
                for num in range(5):
                    if num == 0:
                        self.updates[sInd, num] = TL.int_comp[sInd]
                    if not pInds.any():
                        break
                    # find Eccentric anomaly
                    if num == 0:
                        E = eccanom(M[pInds], e[pInds])
                        newM[pInds] = M[pInds]
                    else:
                        E = eccanom(newM[pInds], e[pInds])

                    # Create column vectors for r1 and r2
                    r1 = (a[pInds] * (np.cos(E) - e[pInds])).reshape(-1, 1)
                    r2 = (np.sin(E) * a_one_minus_e2[pInds]).reshape(-1, 1)

                    # planet position, planet-star distance, apparent separation
                    # position vector (AU)
                    r = A[pInds] * r1 + B[pInds] * r2
                    rsq = r**2
                    d = np.sqrt(np.sum(rsq, axis=1))
                    s = np.sqrt(rsq[:, 0] + rsq[:, 1])
                    # phase angle
                    beta = np.arccos(r[:, 2] / d)
                    # phase function
                    Phi = self.PlanetPhysicalModel.calc_Phi(beta << u.rad)
                    # difference in magnitude
                    dMag = deltaMag(p[pInds], Rp[pInds], d << u.AU, Phi)

                    # Apply geometric and photometric cuts without making the
                    # expensive np.intersect1d call
                    constraints = (
                        (s > smin[sInd])
                        & (s < smax[sInd])
                        & (dMag < max_intCutoff_dMag)
                    )
                    # Remove all planets that met the constraints
                    prev_pInds_size = pInds.size
                    pInds = pInds[~constraints]
                    new_pInds_size = pInds.size
                    n_det = prev_pInds_size - new_pInds_size

                    if num == 0:
                        self.updates[sInd, num] = TL.int_comp[sInd]
                    else:
                        self.updates[sInd, num] = float(n_det) / nplan

                    # update M
                    newM[pInds] = (
                        (newM[pInds] + n[pInds] * dt) / (2 * np.pi) % 1 * 2.0 * np.pi
                    )
            # ensure that completeness values are between 0 and 1
            self.updates = np.clip(self.updates, 0.0, 1.0)
            # store dynamic completeness array as .dcomp file
            with open(path, "wb") as ff:
                pickle.dump(self.updates, ff)
            self.vprint("Dynamic completeness calculations finished")
            self.vprint("Dynamic completeness array stored in %r" % path)

    def completeness_update(self, TL, sInds, visits, dt):
        """Updates completeness value for stars previously observed by selecting
        the appropriate value from the updates array

        Args:
            TL (TargetList module):
                TargetList class object
            sInds (integer array):
                Indices of stars to update
            visits (integer array):
                Number of visits for each star
            dt (astropy Quantity array):
                Time since previous observation

        Returns:
            float ndarray:
                Completeness values for each star

        """
        # if visited more than five times, return 5th stored dynamic
        # completeness value
        visits[visits > 4] = 4
        dcomp = self.updates[sInds, visits]

        return dcomp

    def genC(self, Cpath, nplan, xedges, yedges, steps, remainder=0):
        """Gets completeness interpolant for initial completeness

        This function either loads a completeness .comp file based on specified
        Planet Population module or performs Monte Carlo simulations to get
        the 2D completeness values needed for interpolation.

        Args:
            Cpath (string):
                path to 2D completeness value array
            nplan (float):
                number of planets used in each simulation
            xedges (float ndarray):
                x edge of 2d histogram (separation)
            yedges (float ndarray):
                y edge of 2d histogram (dMag)
            steps (integer):
                number of nplan simulations to perform
            remainder (integer):
                residual number of planets to simulate

        Returns:
            float ndarray:
                2D numpy ndarray containing completeness probability density values

        """

        # if the 2D completeness pdf array exists as a .comp file load it
        if os.path.exists(Cpath):
            self.vprint('Loading cached completeness file from "%s".' % Cpath)
            try:
                with open(Cpath, "rb") as ff:
                    H = pickle.load(ff)
            except UnicodeDecodeError:
                with open(Cpath, "rb") as ff:
                    H = pickle.load(ff, encoding="latin1")
            self.vprint("Completeness loaded from cache.")
        else:
            # run Monte Carlo simulation and pickle the resulting array
            self.vprint('Cached completeness file not found at "%s".' % Cpath)

            t0, t1 = None, None  # keep track of per-iteration time
            for i in tqdm(range(steps), desc="Creating 2d completeness pdf"):
                t0, t1 = t1, time.time()
                if t0 is None:
                    delta_t_msg = ""  # no message
                else:
                    delta_t_msg = "[%.3f s/iteration]" % (t1 - t0)  # noqa: F841
                # get completeness histogram
                h, xedges, yedges = self.hist(nplan, xedges, yedges)
                if i == 0:
                    H = h
                else:
                    H += h
            if not remainder == 0:
                h, xedges, yedges = self.hist(remainder, xedges, yedges)
                if steps > 0:  # if H exists already
                    H += h
                else:  # if H does not exist
                    H = h

            H = H / (self.Nplanets * (xedges[1] - xedges[0]) * (yedges[1] - yedges[0]))

            # store 2D completeness pdf array as .comp file
            with open(Cpath, "wb") as ff:
                pickle.dump(H, ff)
            self.vprint("Monte Carlo completeness calculations finished")
            self.vprint("2D completeness array stored in %r" % Cpath)

        return H, xedges, yedges

    def hist(self, nplan, xedges, yedges):
        """Returns completeness histogram for Monte Carlo simulation

        This function uses the inherited Planet Population module.

        Args:
            nplan (float):
                number of planets used
            xedges (float ndarray):
                x edge of 2d histogram (separation)
            yedges (float ndarray):
                y edge of 2d histogram (dMag)

        Returns:
            float ndarray:
                2D numpy ndarray containing completeness frequencies

        """

        s, dMag = self.genplans(nplan)
        # get histogram
        h, yedges, xedges = np.histogram2d(
            dMag,
            s.to_value("AU"),
            bins=1000,
            range=[[yedges.min(), yedges.max()], [xedges.min(), xedges.max()]],
        )

        return h, xedges, yedges

    def genplans(self, nplan):
        """Generates planet data needed for Monte Carlo simulation

        Args:
            nplan (integer):
                Number of planets

        Returns:
            tuple:
            s (astropy Quantity array):
                Planet apparent separations in units of AU
            dMag (ndarray):
                Difference in brightness

        """

        PPop = self.PlanetPopulation

        nplan = int(nplan)

        # sample uniform distribution of mean anomaly
        M = np.random.uniform(high=2.0 * np.pi, size=nplan)
        # sample quantities
        a, e, p, Rp = PPop.gen_plan_params(nplan)
        # check if circular orbits
        if np.sum(PPop.erange) == 0:
            r = a
            e = 0.0
            E = M
        else:
            E = eccanom(M, e)
            # orbital radius
            r = a * (1.0 - e * np.cos(E))

        beta = np.arccos(1.0 - 2.0 * np.random.uniform(size=nplan))
        s = r * np.sin(beta)
        # phase function
        Phi = self.PlanetPhysicalModel.calc_Phi(beta << u.rad)
        # calculate dMag
        dMag = deltaMag(p, Rp, r, Phi)

        return s, dMag

    def comp_per_intTime(
        self, intTimes, TL, sInds, fZ, JEZ, WA, mode, C_b=None, C_sp=None, TK=None
    ):
        """Calculates completeness for integration time

        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (:ref:`TargetList`):
                TargetList object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)
            TK (:ref:`TimeKeeping`):
                TimeKeeping object (optional)

        Returns:
            flat ndarray:
                Completeness values

        """
        intTimes, sInds, fZ, JEZ, WA, smin, smax, dMag = self.comps_input_reshape(
            intTimes, TL, sInds, fZ, JEZ, WA, mode, C_b=C_b, C_sp=C_sp, TK=TK
        )

        comp = self.comp_calc(smin, smax, dMag)
        mask = smin > self.PlanetPopulation.rrange[1].to_value(u.AU)
        comp[mask] = 0.0
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
            float ndarray:
                Completeness values

        """
        smin_vals = np.clip(smin, self.xcent[0], self.xcent[-1])
        smax_vals = np.clip(smax, self.xcent[0], self.xcent[-1])
        comp = np.zeros_like(smin)
        dMag = np.clip(dMag, self.ycent[0], self.ycent[-1])
        n_steps = np.clip((smax_vals - smin_vals / self.ds), 50, 500).astype(int)
        for i, dMag_val, smin_val, smax_val, num in zip(
            range(len(smin)), dMag, smin_vals, smax_vals, n_steps
        ):
            s_grid = np.linspace(smin_val, smax_val, num=num)
            # Evaluate the cumulative function F(s, dMag_val)
            F_vals = self.Cpdfsum_interp((s_grid, dMag_val))
            # Use the trapezoidal rule to approximate the integral over s.
            ds = s_grid[1] - s_grid[0]
            comp[i] = ds * (0.5 * (F_vals[0] + F_vals[-1]) + np.sum(F_vals[1:-1]))

        # remove small values
        comp[comp < 1e-6] = 0.0

        return comp

    def dcomp_dt(
        self, intTimes, TL, sInds, fZ, JEZ, WA, mode, C_b=None, C_sp=None, TK=None
    ):
        """Calculates derivative of completeness with respect to integration time

        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)
            TK (:ref:`TimeKeeping`):
                TimeKeeping object (optional)

        Returns:
            astropy Quantity array:
                Derivative of completeness with respect to integration time
                (units 1/time)

        """
        intTimes, sInds, fZ, JEZ, WA, smin, smax, dMag = self.comps_input_reshape(
            intTimes, TL, sInds, fZ, JEZ, WA, mode, C_b=C_b, C_sp=C_sp, TK=TK
        )

        ddMag = TL.OpticalSystem.ddMag_dt(
            intTimes, TL, sInds, fZ, JEZ, WA, mode, TK=TK
        ).reshape((len(intTimes),))
        dcomp = self.calc_fdmag(dMag, smin, smax)
        mask = smin > self.PlanetPopulation.rrange[1].to_value(u.AU)
        dcomp[mask] = 0.0

        return dcomp * ddMag

    def comps_input_reshape(
        self, intTimes, TL, sInds, fZ, JEZ, WA, mode, C_b=None, C_sp=None, TK=None
    ):
        """
        Reshapes inputs for comp_per_intTime and dcomp_dt as necessary

        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface bright ness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of
                1/s (optional)
            TK (:ref:`TimeKeeping`):
                TimeKeeping object (optional)

        Returns:
            tuple:
            intTimes (astropy Quantity array):
                Integration times
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            JEZ (astropy Quantity array):
                Intensity of exo-zodiacal light in units of ph/s/m2/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            smin (ndarray):
                Minimum projected separations in AU
            smax (ndarray):
                Maximum projected separations in AU
            dMag (ndarray):
                Difference in brightness magnitude
        """

        # cast inputs to arrays and check
        intTimes = np.array(intTimes.value, ndmin=1) * intTimes.unit
        sInds = np.array(sInds, ndmin=1)
        fZ = np.array(fZ.value, ndmin=1) * fZ.unit
        JEZ = np.array(JEZ.value, ndmin=1) * JEZ.unit
        WA = np.array(WA.value, ndmin=1) * WA.unit
        assert len(intTimes) in [
            1,
            len(sInds),
        ], "intTimes must be constant or have same length as sInds"
        assert len(fZ) in [
            1,
            len(sInds),
        ], "fZ must be constant or have same length as sInds"
        assert len(JEZ) in [
            1,
            len(sInds),
        ], "JEZ must be constant or have same length as sInds"
        assert len(WA) in [
            1,
            len(sInds),
        ], "WA must be constant or have same length as sInds"
        # make constants arrays of same length as sInds if len(sInds) != 1
        if len(sInds) != 1:
            if len(intTimes) == 1:
                intTimes = np.repeat(intTimes.value, len(sInds)) * intTimes.unit
            if len(fZ) == 1:
                fZ = np.repeat(fZ.value, len(sInds)) * fZ.unit
            if len(JEZ) == 1:
                JEZ = np.repeat(JEZ.value, len(sInds)) * JEZ.unit
            if len(WA) == 1:
                WA = np.repeat(WA.value, len(sInds)) * WA.unit

        dMag = TL.OpticalSystem.calc_dMag_per_intTime(
            intTimes,
            TL,
            sInds,
            fZ,
            JEZ,
            WA,
            mode,
            C_b=C_b,
            C_sp=C_sp,
            TK=TK,
            analytic_only=True,
        ).reshape((len(intTimes),))
        # calculate separations based on IWA and OWA
        IWA = mode["IWA"]
        OWA = mode["OWA"]
        smin = (np.tan(IWA) * TL.dist[sInds]).to_value(u.AU)
        if np.isinf(OWA):
            smax = np.array(
                [self.PlanetPopulation.rrange[1].to_value(u.AU)] * len(smin)
            )
        else:
            smax = (np.tan(OWA) * TL.dist[sInds]).to_value(u.AU)
            smax[smax > self.PlanetPopulation.rrange[1].to_value(u.AU)] = (
                self.PlanetPopulation.rrange[1].to_value(u.AU)
            )
        smin[smin > smax] = smax[smin > smax]

        # take care of scaleOrbits == True
        if self.PlanetPopulation.scaleOrbits:
            L = np.where(
                TL.L[sInds] > 0, TL.L[sInds], 1e-10
            )  # take care of zero/negative values
            smin = smin / np.sqrt(L)
            smax = smax / np.sqrt(L)
            dMag -= 2.5 * np.log10(L)

        return intTimes, sInds, fZ, JEZ, WA, smin, smax, dMag

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
            float:
                Value of probability density

        """

        f = np.zeros(len(smin))
        for k, dm in enumerate(dMag):
            f[k] = interpolate.InterpolatedUnivariateSpline(
                self.xnew, self.EVPOCpdf(self.xnew, dm), ext=1
            ).integral(smin[k], smax[k])

        return f
