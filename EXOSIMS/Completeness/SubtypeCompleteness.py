# -*- coding: utf-8 -*-
import time
import numpy as np
from scipy import interpolate
import astropy.units as u
import astropy.constants as const
import os
import pickle
import hashlib
from EXOSIMS.Completeness.BrownCompleteness import BrownCompleteness
from EXOSIMS.util.eccanom import eccanom
from EXOSIMS.util.deltaMag import deltaMag
import itertools
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy.integrate import nquad


class SubtypeCompleteness(BrownCompleteness):
    """Completeness by planet subtype

    Args:
        binTypes (str):
            string specifying the kopparapuBin Types to use
        **specs:
            :ref:`sec:inputspec`

    Attributes:
        Nplanets (int):
            Number of planets for initial completeness Monte Carlo simulation
        classpath (string):
            Path on disk to Brown Completeness
        filename (str):
            Name of file where completeness interpolant is stored
        updates (float nx5 numpy.ndarray):
            Completeness values of successive observations of each star in the
            target list (initialized in gen_update)
        binTypes (str):
            string specifying the kopparapuBin Types to use

    """

    def __init__(self, binTypes="kopparapuBins_extended", **specs):

        self.binTypes = binTypes

        # Run BrownCompleteness init
        BrownCompleteness.__init__(self, **specs)

        self._outspec["binTypes"] = self.binTypes

    def completeness_setup(self):
        """Preform any preliminary calculations needed for this flavor of completeness

        For SubtypeCompleteness this generates comleteness by planet bin
        """

        # SubtypeCompleteness specific stuff
        # Generate Kopparapu Bin Ranges
        if self.binTypes == "kopparapuBins_extended":
            self.kopparapuBins_extended()

        # Overall Population Upper and Lower Limits of dmag vs s JPDF
        pmin = self.PlanetPopulation.prange[0]
        pmax = self.PlanetPopulation.prange[1]
        rmax = self.PlanetPopulation.rrange[1]
        rmin = self.PlanetPopulation.rrange[0]
        Rmax = self.PlanetPopulation.Rprange[1]
        Rmin = self.PlanetPopulation.Rprange[0]
        (
            self.dmag_limit_functions,
            self.lower_limits,
            self.upper_limits,
        ) = self.dmag_limits(
            rmin, rmax, pmax, pmin, Rmax, Rmin, self.PlanetPhysicalModel.calc_Phi
        )

        # Calculate Upper and Lower Limits of dmag vs s plot
        self.jpdf_props = dict()
        self.jpdf_props["limit_funcs"] = dict()
        self.jpdf_props["lower_limits"] = dict()
        self.jpdf_props["upper_limits"] = dict()
        for ii, j in itertools.product(
            np.arange(len(self.Rp_hi)), np.arange(len(self.L_lo[0, :]))
        ):
            pmin = self.PlanetPopulation.prange[0]
            pmax = self.PlanetPopulation.prange[1]
            emax = self.PlanetPopulation.erange[1]  # Maximum orbital Eccentricity

            # NEED TO DO ANOTHER FORMULATION for rmin and rmax
            # rmax and rmin assume a fixed sma but should vary as a function of stellar
            # luminosity. the upper and lower dmaglimits of the planet subtype bin
            # should be based on the brightest and faintest stellar luminosities
            # of the bin

            # these are the classification bin edges being used
            # ravgt_rtLstar_lo = 1./np.sqrt(self.L_lo[ii,j])
            # ravgt_rtLstar_hi = 1./np.sqrt(self.L_hi[ii,j])
            # Note at e=0, ravgt=a. Orbital Radius Limits will be based on most
            # Eccentric Orbits
            # Therefore we can say
            # rmax = ravgt_rtLstar_hi*(1. + emax) #from rp = a(1-e)
            # rmin = ravgt_rtLstar_lo*(1. - emax)

            amax = np.sqrt(
                1.0 / (self.L_hi[ii, j] * (1.0 + emax**2.0 / 2.0) ** 2.0)
            )  # This is the time-averaged orbital SMA
            amin = np.sqrt(1.0 / (self.L_lo[ii, j] * (1.0 + emax**2.0 / 2.0) ** 2.0))
            # THE ONLY OTHER THING TO DO HERE IS ACTUALLY CALCULATE THE RMAX AND RMIN
            if j == len(self.L_hi[0, :]):
                amax = self.PlanetPopulation.arange[1].value

            rmax = amax * (1.0 + emax)
            rmin = amin * (1.0 - emax)
            Rmax = self.Rp_hi[ii] * u.earthRad
            Rmin = self.Rp_lo[ii] * u.earthRad
            self.jpdf_props["limit_funcs"][ii, j] = list()
            self.jpdf_props["lower_limits"][ii, j] = list()
            self.jpdf_props["upper_limits"][ii, j] = list()
            (
                self.jpdf_props["limit_funcs"][ii, j],
                self.jpdf_props["lower_limits"][ii, j],
                self.jpdf_props["upper_limits"][ii, j],
            ) = self.dmag_limits(
                rmin * u.AU,
                rmax * u.AU,
                pmax,
                pmin,
                Rmax,
                Rmin,
                self.PlanetPhysicalModel.calc_Phi,
            )
            # funcs_tmp, lower_limits_tmp, upper_limits_tmp =
            #   self.dmag_limits(rmin,rmax,pmax,Rmax,self.PlanetPhysicalModel.calc_Phi)
            # self.jpdf_props['limit_funcs'][ii,j].append(funcs_tmp)
            # self.jpdf_props['lower_limits'][ii,j].append(lower_limits_tmp)
            # self.jpdf_props['upper_limits'][ii,j].append(upper_limits_tmp)
            # TODO replace phaseFunc with phase function for individual planet types

    def target_completeness(self, TL, subpop=-2):
        """Generates completeness values for target stars

        This method is called from TargetList __init__ method.

        Args:
            TL (TargetList module):
                TargetList class object
            subpop (int):
                planet subtype to use for calculation of int_comp
                -2 - planet population
                -1 - earthLike population
                0-N - kopparapu planet subtypes

        Returns:
            float ndarray:
                Completeness values for each target star

        """

        # set up "ensemble visit photometric and obscurational completeness"
        # interpolant for initial completeness values
        # bins for interpolant
        bins = 1000
        # xedges is array of separation values for interpolant
        if self.PlanetPopulation.constrainOrbits:
            xedges = np.linspace(
                0.0, self.PlanetPopulation.arange[1].to("AU").value, bins + 1
            )
        else:
            xedges = np.linspace(
                0.0, self.PlanetPopulation.rrange[1].to("AU").value, bins + 1
            )

        # yedges is array of delta magnitude values for interpolant
        ymin = -2.5 * np.log10(
            float(
                self.PlanetPopulation.prange[1]
                * (self.PlanetPopulation.Rprange[1] / self.PlanetPopulation.rrange[0])
                ** 2
            )
        )
        ymax = -2.5 * np.log10(
            float(
                self.PlanetPopulation.prange[0]
                * (self.PlanetPopulation.Rprange[0] / self.PlanetPopulation.rrange[1])
                ** 2
            )
            * 1e-11
        )
        yedges = np.linspace(ymin, ymax, bins + 1)
        # number of planets for each Monte Carlo simulation
        nplan = int(np.min([1e6, self.Nplanets]))
        # number of simulations to perform (must be integer)
        steps = int(self.Nplanets / nplan)

        # path to 2D completeness pdf array for interpolation
        Cpath = os.path.join(self.cachedir, self.filename + ".comp")
        Cpdf, xedges2, yedges2 = self.genSubtypeC(
            Cpath, nplan, xedges, yedges, steps, TL
        )
        Cpdf_pop = Cpdf["h"]
        Cpdf_earthLike = Cpdf["h_earthLike"]
        Cpdf_hs = Cpdf["hs"]

        xcent = 0.5 * (xedges2[1:] + xedges2[:-1])
        ycent = 0.5 * (yedges2[1:] + yedges2[:-1])
        xnew = np.hstack((0.0, xcent, self.PlanetPopulation.rrange[1].to("AU").value))
        ynew = np.hstack((ymin, ycent, ymax))
        Cpdf_pop = np.pad(Cpdf_pop, 1, mode="constant")
        Cpdf_earthLike = np.pad(Cpdf_earthLike, 1, mode="constant")
        for ii, j in itertools.product(
            np.arange(len(self.Rp_hi)), np.arange(len(self.L_lo[0, :]))
        ):  # lo
            Cpdf_hs[ii, j] = np.pad(Cpdf_hs[ii, j], 1, mode="constant")

        # save interpolant and counts to object
        self.Cpdf_pop = Cpdf_pop
        self.EVPOCpdf_pop = interpolate.RectBivariateSpline(xnew, ynew, Cpdf_pop.T)
        self.EVPOC_pop = np.vectorize(self.EVPOCpdf_pop.integral, otypes=[np.float64])
        self.count_pop = Cpdf["count"]
        self.xnew = xnew
        self.ynew = ynew
        self.Cpdf_earthLike = Cpdf_earthLike
        self.EVPOCpdf_earthLike = interpolate.RectBivariateSpline(
            xnew, ynew, Cpdf_earthLike.T
        )
        self.EVPOC_earthLike = np.vectorize(
            self.EVPOCpdf_earthLike.integral, otypes=[np.float64]
        )
        self.count_earthLike = Cpdf["count_earthLike"]
        self.Cpdf_hs = Cpdf_hs
        self.EVPOCpdf_hs = dict()
        self.EVPOC_hs = dict()
        self.count_hs = dict()
        for ii, j in itertools.product(
            np.arange(len(self.Rp_hi)), np.arange(len(self.L_lo[0, :]))
        ):  # lo
            self.EVPOCpdf_hs[ii, j] = interpolate.RectBivariateSpline(
                xnew, ynew, Cpdf_hs[ii, j].T
            )
            self.EVPOC_hs[ii, j] = np.vectorize(
                self.EVPOCpdf_hs[ii, j].integral, otypes=[np.float64]
            )
            self.count_hs[ii, j] = Cpdf["counts"][ii, j]

        # calculate separations based on IWA and OWA
        OS = TL.OpticalSystem
        if TL.calc_char_int_comp:
            mode = list(
                filter(lambda mode: "spec" in mode["inst"]["name"], self.observingModes)
            )[0]
        else:
            mode = list(filter(lambda mode: mode["detectionMode"], OS.observingModes))[
                0
            ]
        IWA = mode["IWA"]
        OWA = mode["OWA"]
        smin = np.tan(IWA) * TL.dist
        if np.isinf(OWA):
            smax = np.array([xedges[-1]] * len(smin)) * u.AU
        else:
            smax = np.tan(OWA) * TL.dist
            smax[smax > self.PlanetPopulation.rrange[1]] = self.PlanetPopulation.rrange[
                1
            ]

        # limiting planet delta magnitude for completeness
        dMagMax = max(TL.intCutoff_dMag)

        int_comp = np.zeros(smin.shape)
        if self.PlanetPopulation.scaleOrbits:
            L = np.where(TL.L > 0, TL.L, 1e-10)  # take care of zero/negative values
            smin = smin / np.sqrt(L)
            smax = smax / np.sqrt(L)
            dMagMax -= 2.5 * np.log10(L)
            mask = (dMagMax > ymin) & (smin < self.PlanetPopulation.rrange[1])
            int_comp[mask] = self.EVPOC_pop(
                smin[mask].to("AU").value, smax[mask].to("AU").value, 0.0, dMagMax[mask]
            )
        else:
            mask = smin < self.PlanetPopulation.rrange[1]
            int_comp[mask] = self.EVPOC_pop(
                smin[mask].to("AU").value, smax[mask].to("AU").value, 0.0, dMagMax
            )
        # remove small values
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

        # limiting planet delta magnitude for completeness
        dMagMax = max(TL.intCutoff_dMag)

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
            + " dMagMax: "
            + str(dMagMax)
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
            self.vprint("Beginning dynamic completeness calculations")
            # dynamic completeness values: rows are stars, columns are number of visits
            self.updates = np.zeros((TL.nStars, 5))
            # number of planets to simulate
            nplan = int(2e4)
            # sample quantities which do not change in time
            a, e, p, Rp = PPop.gen_plan_params(nplan)
            a = a.to("AU").value
            # sample angles
            I, O, w = PPop.gen_angles(nplan)
            I = I.to("rad").value
            O = O.to("rad").value
            w = w.to("rad").value
            Mp = PPop.gen_mass(nplan)  # M_earth
            rmax = a * (1.0 + e)  # AU
            # sample quantity which will be updated
            M = np.random.uniform(high=2.0 * np.pi, size=nplan)
            newM = np.zeros((nplan,))
            # population values
            smin = (np.tan(IWA) * TL.dist).to("AU").value
            if np.isfinite(OWA):
                smax = (np.tan(OWA) * TL.dist).to("AU").value
            else:
                smax = np.array(
                    [np.max(PPop.arange.to("AU").value) * (1.0 + np.max(PPop.erange))]
                    * TL.nStars
                )
            # fill dynamic completeness values
            for sInd in range(TL.nStars):
                mu = (const.G * (Mp + TL.MsTrue[sInd])).to("AU3/day2").value
                n = np.sqrt(mu / a**3)  # in 1/day
                # normalization time equation from Brown 2015
                dt = (
                    58.0
                    * (TL.L[sInd] / 0.83) ** (3.0 / 4.0)
                    * (TL.MsTrue[sInd] / (0.91 * u.M_sun)) ** (1.0 / 2.0)
                )  # days
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

                    r1 = a[pInds] * (np.cos(E) - e[pInds])
                    r1 = np.hstack(
                        (
                            r1.reshape(len(r1), 1),
                            r1.reshape(len(r1), 1),
                            r1.reshape(len(r1), 1),
                        )
                    )
                    r2 = a[pInds] * np.sin(E) * np.sqrt(1.0 - e[pInds] ** 2)
                    r2 = np.hstack(
                        (
                            r2.reshape(len(r2), 1),
                            r2.reshape(len(r2), 1),
                            r2.reshape(len(r2), 1),
                        )
                    )

                    a1 = np.cos(O[pInds]) * np.cos(w[pInds]) - np.sin(
                        O[pInds]
                    ) * np.sin(w[pInds]) * np.cos(I[pInds])
                    a2 = np.sin(O[pInds]) * np.cos(w[pInds]) + np.cos(
                        O[pInds]
                    ) * np.sin(w[pInds]) * np.cos(I[pInds])
                    a3 = np.sin(w[pInds]) * np.sin(I[pInds])
                    A = np.hstack(
                        (
                            a1.reshape(len(a1), 1),
                            a2.reshape(len(a2), 1),
                            a3.reshape(len(a3), 1),
                        )
                    )

                    b1 = -np.cos(O[pInds]) * np.sin(w[pInds]) - np.sin(
                        O[pInds]
                    ) * np.cos(w[pInds]) * np.cos(I[pInds])
                    b2 = -np.sin(O[pInds]) * np.sin(w[pInds]) + np.cos(
                        O[pInds]
                    ) * np.cos(w[pInds]) * np.cos(I[pInds])
                    b3 = np.cos(w[pInds]) * np.sin(I[pInds])
                    B = np.hstack(
                        (
                            b1.reshape(len(b1), 1),
                            b2.reshape(len(b2), 1),
                            b3.reshape(len(b3), 1),
                        )
                    )

                    # planet position, planet-star distance, apparent separation
                    r = A * r1 + B * r2  # position vector (AU)
                    d = np.linalg.norm(r, axis=1)  # planet-star distance
                    s = np.linalg.norm(r[:, 0:2], axis=1)  # apparent separation
                    beta = np.arccos(r[:, 2] / d)  # phase angle
                    Phi = self.PlanetPhysicalModel.calc_Phi(
                        beta * u.rad
                    )  # phase function
                    dMag = deltaMag(
                        p[pInds], Rp[pInds], d * u.AU, Phi
                    )  # difference in magnitude

                    toremoves = np.where((s > smin[sInd]) & (s < smax[sInd]))[0]
                    toremovedmag = np.where(dMag < dMagMax)[0]
                    toremove = np.intersect1d(toremoves, toremovedmag)

                    pInds = np.delete(pInds, toremove)

                    if num == 0:
                        self.updates[sInd, num] = TL.int_comp[sInd]
                    else:
                        self.updates[sInd, num] = float(len(toremove)) / nplan

                    # update M
                    newM[pInds] = (
                        (newM[pInds] + n[pInds] * dt) / (2 * np.pi) % 1 * 2.0 * np.pi
                    )

                if (sInd + 1) % 50 == 0:
                    self.vprint("stars: %r / %r" % (sInd + 1, TL.nStars))
            # ensure that completeness values are between 0 and 1
            self.updates = np.clip(self.updates, 0.0, 1.0)
            # store dynamic completeness array as .dcomp file
            with open(path, "wb") as ff:
                pickle.dump(self.updates, ff)
            self.vprint("Dynamic completeness calculations finished")
            self.vprint("Dynamic completeness array stored in %r" % path)

    def genSubtypeC(self, Cpath, nplan, xedges, yedges, steps, TL):
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
                number of simulations to perform
            TL (:ref:`TargetList`):
                TargetList object

        Returns:
            float ndarray:
                2D numpy ndarray containing completeness probability density values

        """
        H = dict()

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
            self.vprint("Beginning Monte Carlo completeness calculations.")

            t0, t1 = None, None  # keep track of per-iteration time
            for i in range(steps):
                t0, t1 = t1, time.time()
                if t0 is None:
                    delta_t_msg = ""  # no message
                else:
                    delta_t_msg = "[%.3f s/iteration]" % (t1 - t0)
                self.vprint(
                    "Completeness iteration: %5d / %5d %s" % (i + 1, steps, delta_t_msg)
                )
                # get completeness histogram
                (
                    hs,
                    bini,
                    binj,
                    h_earthLike,
                    h,
                    xedges,
                    yedges,
                    counts,
                    count,
                    count_earthLike,
                ) = self.SubtypeHist(nplan, xedges, yedges, TL)
                if i == 0:
                    H["h_earthLike"] = h_earthLike
                    H["h"] = h
                    H["hs"] = hs
                    H["count_earthLike"] = count_earthLike
                    H["count"] = count
                    H["counts"] = counts
                else:
                    H["h_earthLike"] += h_earthLike
                    H["h"] += h
                    H["count_earthLike"] += count_earthLike
                    H["count"] += count
                    for ii, j in itertools.product(
                        np.arange(len(self.Rp_hi)), np.arange(len(self.L_lo[0, :]))
                    ):  # lo
                        H["hs"][ii, j] += hs[ii, j]
                        H["counts"][ii, j] += counts[ii, j]

            # Not sure why this correction to  the rates was applied
            H["h_earthLike"] = H["h_earthLike"] / (
                self.Nplanets * (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])
            )
            H["h"] = H["h"] / (
                self.Nplanets * (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])
            )
            for ii, j in itertools.product(
                np.arange(len(self.Rp_hi)), np.arange(len(self.L_lo[0, :]))
            ):  # lo
                H["hs"][ii, j] = hs[ii, j] / (
                    self.Nplanets * (xedges[1] - xedges[0]) * (yedges[1] - yedges[0])
                )

            # store 2D completeness pdf array as .comp file
            with open(Cpath, "wb") as ff:
                pickle.dump(H, ff)
            self.vprint("Monte Carlo completeness calculations finished")
            self.vprint("2D completeness array stored in %r" % Cpath)

        return H, xedges, yedges

    def SubtypeHist(self, nplan, xedges, yedges, TL):
        """Returns completeness histogram for Monte Carlo simulation

        This function uses the inherited Planet Population module.

        Args:
            nplan (float):
                number of planets used
            xedges (float ndarray):
                x edge of 2d histogram (separation)
            yedges (float ndarray):
                y edge of 2d histogram (dMag)
            TL (:ref:`TargetList`):
                TargetList object

        Returns:
            tuple:
                float (numpy.ndarray):
                    2D numpy ndarray containing completeness frequencies
                hs (dict):
                    dict with index [bini,binj] containing arrays of counts
                    per Array bin
                bini (float):
                    planet size type index
                binj (float):
                    planet incident stellar flux index
                h_earthLike (float):
                    number of earth like exoplanets
                h (numpy.ndarray):
                    2D numpy array of bin counts over all dmag vs s
                xedges (numpy.ndarray):
                    array of bin edges originally input and used in histograms
                yedges (numpy.ndarray):
                    array of bin edges originally input and used in histograms
                counts (dict):
                    dict with index [bini,binj] containing total number of planets
                    in bini,binj
                count (float):
                    total number of planets in the whole populatiom
                count_earthLike (float):
                    total number of earth-like planets in the whole population

        """
        tStartSTH = time.time()
        gpStart = time.time()
        s, dMag, bini, binj, earthLike = self.genplans(nplan, TL)
        gpStop = time.time()
        self.vprint("genplans: " + str(gpStop - gpStart))

        # get histogram for whole population
        t1 = time.time()
        h, yedges, xedges = np.histogram2d(
            dMag,
            s.to("AU").value,
            bins=1000,
            range=[
                [np.nanmin(yedges), np.nanmax(yedges)],
                [np.nanmin(xedges), np.nanmax(xedges)],
            ],
        )
        count = np.sum(h)
        t2 = time.time()
        self.vprint("pop hist: " + str(t2 - t1))

        # get h_earthlike histogram for earthLike population
        t3 = time.time()
        h_earthLike, yedges, xedges = np.histogram2d(
            dMag[earthLike == 1],
            s.to("AU").value[earthLike == 1],
            bins=1000,
            range=[
                [np.nanmin(yedges), np.nanmax(yedges)],
                [np.nanmin(xedges), np.nanmax(xedges)],
            ],
        )
        count_earthLike = np.sum(h_earthLike)
        t4 = time.time()
        self.vprint("earthLike hist: " + str(t4 - t3))

        # get bini,binj
        hs = dict()
        counts = dict()
        for ii, j in itertools.product(
            np.arange(len(self.Rp_hi)), np.arange(len(self.L_lo[0, :]))
        ):  # lo
            t5 = time.time()
            hs[ii, j], yedges, xedges = np.histogram2d(
                dMag[(bini == ii) * (binj == j)],
                s.to("AU").value[(bini == ii) * (binj == j)],
                bins=1000,
                range=[
                    [np.nanmin(yedges), np.nanmax(yedges)],
                    [np.nanmin(xedges), np.nanmax(xedges)],
                ],
            )
            counts[ii, j] = np.sum(hs[ii, j])
            t6 = time.time()
            self.vprint("bin(" + str(ii) + "," + str(j) + ") hist: " + str(t6 - t5))

        tStopSTH = time.time()
        self.vprint("STH: " + str(tStopSTH - tStartSTH))

        return (
            hs,
            bini,
            binj,
            h_earthLike,
            h,
            xedges,
            yedges,
            counts,
            count,
            count_earthLike,
        )

    def genplans(self, nplan, TL):
        """Generates planet data needed for Monte Carlo simulation

        Args:
            nplan (int):
                Number of planets
            TL (:ref:`TargetList`):
                TargetList object

        Returns:
            tuple:
            s (astropy Quantity array):
                Planet apparent separations in units of AU
            dMag (ndarray):
                Difference in brightness
            bini (int):
                planet size-type: 0-rocky, 1- Super-Earths, 2- sub-Neptunes,
                3- sub-Jovians, 4- Jovians
            binj (int):
                planet incident stellar-flux: 0- hot, 1- warm, 2- cold
            earthLike (bool):
                boolean indicating whether the planet is earthLike or not earthLike

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

        beta = np.arccos(1.0 - 2.0 * np.random.uniform(size=nplan)) * u.rad
        s = r * np.sin(beta)
        # phase function
        Phi = self.PlanetPhysicalModel.calc_Phi(beta)
        # calculate dMag
        dMag = deltaMag(p, Rp, r, Phi)

        t10 = time.time()
        starInds = np.random.randint(0, TL.nStars, size=len(e))
        bini = np.zeros(len(e), dtype=int)
        binj = np.zeros(len(e), dtype=int)
        earthLike = np.zeros(len(e), dtype=int)
        bini, binj, earthLike = self.classifyPlanets(Rp, TL, starInds, a, e)
        t11 = time.time()
        self.vprint("Time Classifying Planets: " + str(t11 - t10))

        return s, dMag, bini, binj, earthLike

    def comp_per_intTime(
        self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None, TK=None
    ):
        """Calculates completeness for integration time

        Args:
            intTimes (astropy Quantity array):
                Integration times
            TL (TargetList module):
                TargetList class object
            sInds (integer ndarray):
                Integer indices of the stars of interest
            fZ (astropy Quantity array):
                Surface brightness of local zodiacal light in units of 1/arcsec2
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)
            TK (Timekeeping object):
                timekeeping object for compatability with SLSQPScheduler

        Returns:
            flat ndarray:
                Completeness values

        """
        intTimes, sInds, fZ, fEZ, WA, smin, smax, dMag = self.comps_input_reshape(
            intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=C_b, C_sp=C_sp, TK=TK
        )

        comp = self.comp_calc(smin, smax, dMag)
        mask = smin > self.PlanetPopulation.rrange[1].to("AU").value
        comp[mask] = 0.0
        # ensure completeness values are between 0 and 1
        comp = np.clip(comp, 0.0, 1.0)

        return comp

    def comp_calc(self, smin, smax, dMag, subpop=-2):
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
            subpop (int):
                planet subtype to use for calculation of int_comp
                -2 - planet population
                -1 - earthLike population
                (i,j) - kopparapu planet subtypes

        Returns:
            float ndarray:
                Completeness values

        """
        if subpop == -2:
            comp = self.EVPOC_pop(smin, smax, 0.0, dMag)
        elif subpop == -1:
            comp = self.EVPOC_earthlike(smin, smax, 0.0, dMag)
        else:
            comp = self.EVPOC_hs[subpop[0], subpop[1]](smin, smax, 0.0, dMag)
        # remove small values
        comp[comp < 1e-6] = 0.0

        return comp

    def comp_calc2(self, smin, smax, dMag_min, dMag_max, subpop=-2):
        """Calculates completeness for given minimum and maximum separations
        and dMag

        Note: this method assumes scaling orbits when scaleOrbits == True has
        already occurred for smin, smax, dMag inputs

        Args:
            smin (float ndarray):
                Minimum separation(s) in AU
            smax (float ndarray):
                Maximum separation(s) in AU
            dMag_min (float ndarray):
                Minimum difference in brightness magnitude
            dMag_max (float ndarray):
                Maximum difference in brightness magnitude
            subpop (int):
                planet subtype to use for calculation of int_comp
                -2 - planet population
                -1 - earthLike population
                (i,j) - kopparapu planet subtypes

        Returns:
            float ndarray:
                Completeness values

        """
        if subpop == -2:
            comp = self.EVPOC_pop(smin, smax, dMag_min, dMag_max)
        elif subpop == -1:
            comp = self.EVPOC_earthlike(smin, smax, dMag_min, dMag_max)
        else:
            comp = self.EVPOC_hs[subpop[0], subpop[1]](smin, smax, dMag_min, dMag_max)
        # remove small values
        comp[comp < 1e-6] = 0.0

        return comp

    def dcomp_dt(
        self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None, TK=None
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
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units of 1/s
                (optional)
            TK (Timekeeping object):
                timekeeping object for compatability with SLSQPScheduler

        Returns:
            astropy Quantity array:
                Derivative of completeness with respect to integration
                time (units 1/time)

        """
        intTimes, sInds, fZ, fEZ, WA, smin, smax, dMag = self.comps_input_reshape(
            intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=C_b, C_sp=C_sp, TK=TK
        )

        ddMag = TL.OpticalSystem.ddMag_dt(
            intTimes, TL, sInds, fZ, fEZ, WA, mode
        ).reshape((len(intTimes),))
        dcomp = self.calc_fdmag(dMag, smin, smax)
        mask = smin > self.PlanetPopulation.rrange[1].to("AU").value
        dcomp[mask] = 0.0

        return dcomp * ddMag

    def comps_input_reshape(
        self, intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=None, C_sp=None, TK=None
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
            fEZ (astropy Quantity array):
                Surface brightness of exo-zodiacal light in units of 1/arcsec2
            WA (astropy Quantity):
                Working angle of the planet of interest in units of arcsec
            mode (dict):
                Selected observing mode
            C_b (astropy Quantity array):
                Background noise electron count rate in units of 1/s (optional)
            C_sp (astropy Quantity array):
                Residual speckle spatial structure (systematic error) in units
                of 1/s (optional)
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
                fEZ (astropy Quantity array):
                    Surface brightness of exo-zodiacal light in units of 1/arcsec2
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
        fEZ = np.array(fEZ.value, ndmin=1) * fEZ.unit
        WA = np.array(WA.value, ndmin=1) * WA.unit
        assert len(intTimes) in [
            1,
            len(sInds),
        ], "intTimes must be constant or have same length as sInds"
        assert len(fZ) in [
            1,
            len(sInds),
        ], "fZ must be constant or have same length as sInds"
        assert len(fEZ) in [
            1,
            len(sInds),
        ], "fEZ must be constant or have same length as sInds"
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
            if len(fEZ) == 1:
                fEZ = np.repeat(fEZ.value, len(sInds)) * fEZ.unit
            if len(WA) == 1:
                WA = np.repeat(WA.value, len(sInds)) * WA.unit

        dMag = TL.OpticalSystem.calc_dMag_per_intTime(
            intTimes, TL, sInds, fZ, fEZ, WA, mode, C_b=C_b, C_sp=C_sp
        ).reshape((len(intTimes),))
        # calculate separations based on IWA and OWA
        IWA = mode["IWA"]
        OWA = mode["OWA"]
        smin = (np.tan(IWA) * TL.dist[sInds]).to("AU").value
        if np.isinf(OWA):
            smax = np.array(
                [self.PlanetPopulation.rrange[1].to("AU").value] * len(smin)
            )
        else:
            smax = (np.tan(OWA) * TL.dist[sInds]).to("AU").value
            smax[smax > self.PlanetPopulation.rrange[1].to("AU").value] = (
                self.PlanetPopulation.rrange[1].to("AU").value
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

        return intTimes, sInds, fZ, fEZ, WA, smin, smax, dMag

    def calc_fdmag(self, dMag, smin, smax, subpop=-2):
        """Calculates probability density of dMag by integrating over projected
        separation

        Args:
            dMag (float ndarray):
                Planet delta magnitude(s)
            smin (float ndarray):
                Value of minimum projected separation (AU) from instrument
            smax (float ndarray):
                Value of maximum projected separation (AU) from instrument
            subpop (int):
                planet subtype to use for calculation of int_comp
                -2 - planet population
                -1 - earthLike population
                (i,j) - kopparapu planet subtypes

        Returns:
            float:
                Value of probability density

        """
        if subpop == -2:
            f = np.zeros(len(smin))
            for k, dm in enumerate(dMag):
                f[k] = interpolate.InterpolatedUnivariateSpline(
                    self.xnew, self.EVPOCpdf_pop(self.xnew, dm), ext=1
                ).integral(smin[k], smax[k])
        elif subpop == -1:
            f = np.zeros(len(smin))
            for k, dm in enumerate(dMag):
                f[k] = interpolate.InterpolatedUnivariateSpline(
                    self.xnew, self.EVPOCpdf_earthLike(self.xnew, dm), ext=1
                ).integral(smin[k], smax[k])
        else:
            f = np.zeros(len(smin))
            for k, dm in enumerate(dMag):
                f[k] = interpolate.InterpolatedUnivariateSpline(
                    self.xnew,
                    self.EVPOCpdf_hs[subpop[0], subpop[1]](self.xnew, dm),
                    ext=1,
                ).integral(smin[k], smax[k])

        return f

    # MODIFY THIS TO CREATE A CLASSIFICATION FOR EACH pInd
    # WHICH IS THE CLASSIFICATION NUMBER
    def putPlanetsInBoxes(self, out, TL):
        """Classifies planets in a gen_summary out file by their hot/warm/cold and
        rocky/superearth/subneptune/subjovian/jovian bins

        Args:
            out (dict):
                a gen_summary output dict
            TL (TargetList object):
                a target list object

        Returns:
            tuple:
                aggbins (list):
                    dims [# simulations, 5x3 numpy array]
                earthLikeBins (list):
                    dims [# simulations]

        """
        aggbins = list()
        earthLikeBins = list()
        bins = np.zeros(
            (self.L_bins.shape[0] - 1, self.L_bins.shape[1] - 1)
        )  # planet type, planet temperature
        # planet types: rockey, super-Earths, sub-Neptunes, sub-Jovians, Jovians
        # planet temperatures: cold, warm, hot
        for i in np.arange(len(out["starinds"])):  # iterate over simulations
            bins = np.zeros(
                (self.L_bins.shape[0] - 1, self.L_bins.shape[1] - 1)
            )  # planet type, planet temperature
            earthLike = 0
            starinds = out["starinds"][i]  # inds of the stars
            plan_inds = out["detected"][i]  # contains the planet inds
            Rps = out["Rps"][i]
            smas = out["smas"][i]
            es = out["es"][i]
            for j in np.arange(len(plan_inds)):  # iterate over targets
                Rp = Rps[j]
                starind = int(starinds[j])
                sma = smas[j]
                ej = es[j]

                bini, binj, earthLikeBool = self.classifyPlanet(
                    Rp, TL, starind, sma, ej
                )
                if earthLikeBool:
                    earthLike += 1  # just increment count by 1

                bins[bini, binj] += 1  # just increment count by 1
                del bini
                del binj

            earthLikeBins.append(earthLike)
            aggbins.append(bins)  # aggrgate the bin count for each simulation
        return aggbins, earthLikeBins

    def classifyPlanets(self, Rp, TL, starind, sma, ej):
        """Determine Kopparapu bin of an individual planet.

        Verified with Kopparapu Extended

        Args:
            Rp (float):
                planet radius in Earth Radii
            TL (:ref:`TargetList`):
                TargetList object
            starind (ndarray(int)):
                Star indices
            sma (float):
                planet semi-major axis in AU
            ej (float):
                planet eccentricity

        Returns:
            tuple:
                bini (int):
                    planet size-type: 0-rocky, 1- Super-Earths, 2- sub-Neptunes,
                    3- sub-Jovians, 4- Jovians
                binj (int):
                    planet incident stellar-flux: 0- hot, 1- warm, 2- cold
                earthLike (bool):
                    boolean indicating whether the planet is earthLike or not earthLike

        """
        Rp = Rp.to("earthRad").value
        sma = sma.to("AU").value

        # Find Planet Rp range
        bini = np.zeros(len(ej), dtype=int) + len(
            self.Rp_hi
        )  # For each bin this is not in, subtract 1
        for ind in np.arange(len(self.Rp_hi)):
            bini -= np.asarray(Rp < self.Rp_hi[ind], dtype=int) * 1
        # TODO check to see if any self.Rp_lo violations sneak through

        # IF assigning each planet a luminosity
        # L_star = TL.L[starind] # grab star luminosity
        L_star = 1.0
        L_plan = (
            L_star / (sma * (1.0 + (ej**2.0) / 2.0)) ** 2.0 / (1.0)
        )  # adjust star luminosity by distance^2 in AU scaled to Earth Flux Units
        # Note for earth sma=1,e=0 so r=(1+(0**2)/2)=1
        # *uses true anomaly average distance

        # Find Luminosity Ranges for the Given Rp
        L_lo1 = self.L_lo[bini]  # lower bin range of luminosity
        L_lo2 = self.L_lo[bini + 1]  # lower bin range of luminosity
        # L_hi1 = self.L_hi[bini]  # upper bin range of luminosity
        # L_hi2 = self.L_hi[bini + 1]  # upper bin range of luminosity
        k1 = L_lo2 - L_lo1
        k2 = self.Rp_hi[bini] - self.Rp_lo[bini]
        k3 = Rp - self.Rp_lo[bini]
        k4 = k1 / k2[:, np.newaxis]
        L_lo = k4 * k3[:, np.newaxis] + L_lo1
        # Find Planet Stellar Flux range
        binj = np.zeros(len(ej), dtype=int) - 1
        for ind in np.arange(len(L_lo[0, :])):
            binj += np.asarray(L_plan < L_lo[:, ind]) * 1

        # NEED CITATION ON THIS #From Rhonda's definition of Earthlike
        # earthLike = False
        # if (Rp >= 0.90 and Rp <= 1.4) and (L_plan >= 0.3586 and L_plan <= 1.1080):
        #     earthLike = True
        earthLike = np.ones(len(ej), dtype=bool)
        earthLike = earthLike * (Rp >= 0.9)
        earthLike = earthLike * (Rp <= 1.4)
        earthLike = earthLike * (L_plan >= 0.3586)
        earthLike = earthLike * (L_plan <= 1.1080)

        # Limits from Kopparapu2018 pg6
        # if (Rp >= 0.5 and Rp <= 1.4)
        # if (Rp >= 0.95 and Rp <= 1.67) #conservative limits from Kopparapu2014

        return bini, binj, earthLike

    def classifyEarthlikePlanets(self, Rp, TL, starind, sma, ej):
        """Determine Kopparapu bin of an individual planet.

        Verified with Kopparapu Extended

        Args:
            Rp (float):
                planet radius in Earth Radii
            TL (object):
                EXOSIMS target list object
            starind (ndarray(int)):
                Star indices
            sma (float):
                planet semi-major axis in AU
            ej (float):
                planet eccentricity

        Returns:
            tuple:
                bini (int):
                    planet size-type: 0-Smaller than Earthlike, 1- Earthlike,
                    2- Larger than Earth-like
                binj (int):
                    planet incident stellar-flux: 0- lower than Earthlike,
                    1- flux of Earthlike, 2- higher flux than Earth-like

        """
        Rp = Rp.to("earthRad").value
        sma = sma.to("AU").value

        # IF assigning each planet a luminosity
        # L_star = TL.L[starind] # grab star luminosity
        L_star = 1.0
        L_plan = (
            L_star / (sma * (1.0 + (ej**2.0) / 2.0)) ** 2.0 / (1.0)
        )  # adjust star luminosity by distance^2 in AU scaled to Earth Flux Units

        bini = np.zeros(len(ej))
        bini[np.where(Rp < 0.9)[0]] = 0
        bini[np.where((Rp >= 0.9) * (Rp <= 1.4))[0]] = 1
        bini[np.where(Rp > 1.4)[0]] = 2

        # earthLike = np.ones(len(ej),dtype=bool)
        # earthLike = earthLike*(Rp >= 0.9)
        # earthLike = earthLike*(Rp <= 1.4)

        binj = np.zeros(len(ej))
        binj[np.where(L_plan < 0.3586)[0]] = 0
        binj[np.where((L_plan < 0.3586) * (L_plan > 1.1080))[0]] = 1
        binj[np.where(L_plan > 1.1080)[0]] = 2

        return bini, binj

    def classifyPlanet(self, Rp, TL, starind, sma, ej):
        """Determine Kopparapu bin of an individual planet

        Args:
            Rp (float):
                planet radius in Earth Radii
            TL (object):
                EXOSIMS target list object
            starind (ndarray(int)):
                Star indices
            sma (float):
                planet semi-major axis in AU
            ej (float):
                planet eccentricity

        Returns:
            tuple:
                bini (int):
                    planet size-type: 0-rocky, 1- Super-Earths, 2- sub-Neptunes,
                    3- sub-Jovians, 4- Jovians
                binj (int):
                    planet incident stellar-flux: 0- hot, 1- warm, 2- cold
                earthLike (bool):
                    boolean indicating whether the planet is earthLike or not earthLike

        """
        # Find Planet Rp range
        bini = np.where((self.Rp_lo < Rp) * (Rp < self.Rp_hi))[
            0
        ]  # index of planet size, rocky,...,jovian
        if bini.size == 0:  # correction for if planet is outside planet range
            if Rp < 0:
                bini = 0
            elif Rp > max(self.Rp_hi):
                bini = len(self.Rp_hi) - 1
        else:
            bini = bini[0]

        # IF assigning each planet a luminosity
        # L_star = TL.L[starind] # grab star luminosity
        L_star = 1.0  # Allow to be scale by stellar Luminosity
        L_plan = (
            L_star / (sma * (1.0 + (ej**2.0) / 2.0)) ** 2.0
        )  # adjust star luminosity by distance^2 in AU
        # *uses true anomaly average distance

        # Find Luminosity Ranges for the Given Rp
        L_lo1 = self.L_lo[bini]  # lower bin range of luminosity
        L_lo2 = self.L_lo[bini + 1]  # lower bin range of luminosity
        L_hi1 = self.L_hi[bini]  # upper bin range of luminosity
        L_hi2 = self.L_hi[bini + 1]  # upper bin range of luminosity

        L_lo = (L_lo2 - L_lo1) / (self.Rp_hi[bini] - self.Rp_lo[bini]) * (
            Rp - self.Rp_lo[bini]
        ) + L_lo1
        L_hi = (L_hi2 - L_hi1) / (self.Rp_hi[bini] - self.Rp_lo[bini]) * (
            Rp - self.Rp_lo[bini]
        ) + L_hi1

        binj = np.where((L_lo > L_plan) * (L_plan > L_hi))[
            0
        ]  # index of planet temp. cold,warm,hot
        if binj.size == 0:  # correction for if planet luminosity is out of bounds
            if L_plan > max(L_lo):
                binj = 0
            elif L_plan < min(L_hi):
                binj = len(L_hi) - 1
        else:
            binj = binj[0]

        # NEED CITATION ON THIS
        earthLike = False
        if (Rp >= 0.90 and Rp <= 1.4) and (L_plan >= 0.3586 and L_plan <= 1.1080):
            earthLike = True

        return bini, binj, earthLike

    def kopparapuBins_old(self):
        """A function containing the Inner 12 Kopparapu bins
        Updates the Rp_bins, Rp_lo, Rp_hi, L_bins, L_lo, and L_hi attributes
        """
        # 1: planet-radius bin-edges  [units = Earth radii]
        self.Rp_bins = np.array([0.5, 1.4, 4.0, 14.3])  # Old early 2018 had 3 bins
        # 1b: bin lo/hi edges, same size as the resulting histograms
        self.Rp_lo = self.Rp_bins[:-1]
        self.Rp_hi = self.Rp_bins[1:]

        # 2: stellar luminosity bins, in hot -> cold order
        self.L_bins = np.array(
            [
                [185, 1.5, 0.38, 0.0065],
                [185, 1.6, 0.42, 0.0065],
                [185, 1.55, 0.40, 0.0055],
            ]
        )
        # the below : selectors are correct for increasing ordering
        self.L_lo = self.L_bins[:, :-1]
        self.L_hi = self.L_bins[:, 1:]

        RpL_bin_count = self.L_bins.size - (self.Rp_bins.size - 1)  # noqa: F841

        return None

    def kopparapuBins(self):
        """A function containing the Center 15 Kopparapu bins
        Updates the Rp_bins, Rp_lo, Rp_hi, L_bins, L_lo, and L_hi attributes
        """
        # 1: planet-radius bin-edges  [units = Earth radii]
        # New (May 2018, 5 bins x 3 bins, see Kopparapu et al, arxiv:1802.09602v1,
        # Table 1 and in particular Table 3 column 1, column 2 and Fig. 2):
        self.Rp_bins = np.array([0.5, 1.0, 1.75, 3.5, 6.0, 14.3])
        # 1b: bin lo/hi edges, same size as the resulting histograms
        self.Rp_lo = self.Rp_bins[:-1]
        self.Rp_hi = self.Rp_bins[1:]

        # 2: stellar luminosity bins, in hot -> cold order
        self.L_bins = np.array(
            [
                [182, 1.0, 0.28, 0.0035],
                [187, 1.12, 0.30, 0.0030],
                [188, 1.15, 0.32, 0.0030],
                [220, 1.65, 0.45, 0.0030],
                [220, 1.65, 0.40, 0.0025],
            ]
        )
        # the below : selectors are correct for increasing ordering
        self.L_lo = self.L_bins[:, :-1]
        self.L_hi = self.L_bins[:, 1:]

        RpL_bin_count = self.L_bins.size - (self.Rp_bins.size - 1)  # noqa: F841

        return None

    def kopparapuBins_extended(self):
        """A function containing the Full 35 Kopparapu bins
        Updates the Rp_bins, Rp_lo, Rp_hi, L_bins, L_lo, L_hi, and type_names attribute
        """
        # 1: planet-radius bin-edges  [units = Earth radii]
        self.Rp_bins = np.array([0.0, 0.5, 1.0, 1.75, 3.5, 6.0, 14.3, 11.2 * 4.6])
        # 0 is the smallest theoretical planet radius
        # 4.6Rj is the size of "GQ Lupi b", the largest detected exoplanet making
        # this a nice upper bound
        # 1b: bin lo/hi edges, same size as the resulting histograms
        self.Rp_lo = self.Rp_bins[:-1]
        self.Rp_hi = self.Rp_bins[1:]

        # 2: stellar luminosity bins, in hot -> cold order
        self.L_bins = np.array(
            [
                [1000.0, 182.0, 1.0, 0.28, 0.0035, 5e-5],
                [1000.0, 182.0, 1.0, 0.28, 0.0035, 5e-5],
                [1000.0, 187.0, 1.12, 0.30, 0.0030, 5e-5],
                [1000.0, 188.0, 1.15, 0.32, 0.0030, 5e-5],
                [1000.0, 220.0, 1.65, 0.45, 0.0030, 5e-5],
                [1000.0, 220.0, 1.65, 0.40, 0.0025, 5e-5],
                [1000.0, 220.0, 1.68, 0.45, 0.0025, 5e-5],
                [1000.0, 220.0, 1.68, 0.45, 0.0025, 5e-5],
            ]
        )
        # the below : selectors are correct for increasing ordering
        self.L_lo = self.L_bins[:, :-1]
        self.L_hi = self.L_bins[:, 1:]

        RpL_bin_count = self.L_bins.size - (self.Rp_bins.size - 1)  # noqa: F841

        # Planet Subtype Names
        self.type_names = dict()
        for ii, j in itertools.product(
            np.arange(len(self.Rp_hi)), np.arange(len(self.L_lo[0, :]))
        ):
            self.type_names[ii, j] = ""  # None
        self.type_names[4 + 1, 0 + 1] = "Hot Jovians"
        self.type_names[4 + 1, 1 + 1] = "Warm Jovians"
        self.type_names[4 + 1, 2 + 1] = "Cold Jovians"
        self.type_names[3 + 1, 0 + 1] = "Hot Neptunes"
        self.type_names[3 + 1, 1 + 1] = "Warm Neptunes"
        self.type_names[3 + 1, 2 + 1] = "Cold Neptunes"
        self.type_names[2 + 1, 0 + 1] = "Hot Sub-Neptunes"
        self.type_names[2 + 1, 1 + 1] = "Warm Sub-Neptunes"
        self.type_names[2 + 1, 2 + 1] = "Cold Sub-Neptunes"
        self.type_names[1 + 1, 0 + 1] = "Hot Super Earths"
        self.type_names[1 + 1, 1 + 1] = "Warm Super Earths"
        self.type_names[1 + 1, 2 + 1] = "Cold Super Earths"
        self.type_names[0 + 1, 0 + 1] = "Hot Rocky"
        self.type_names[0 + 1, 1 + 1] = "Warm Rocky"
        self.type_names[0 + 1, 2 + 1] = "Cold Rocky"

        # Webplot digitization of the Kopparapu grid
        # webplot = np.asarray([[181.73853848906157, 0.49999999999999994],
        # [1.00182915700625, 0.49999999999999994],
        # [0.28139033938694097, 0.49999999999999994],
        # [0.00349497189402043, 0.49999999999999994],
        # [0.003004197591674162, 0.997104431643149],
        # [0.29981487455107425, 1.0025384668314574],
        # [1.1184403433978212, 1.0025384668314574],
        # [184.8087079285292, 0.997104431643149],
        # [187.99901007928253, 1.7452663629174578],
        # [1.148438018211251, 1.7452663629174578],
        # [0.3166015314382051, 1.7452663629174578],
        # [0.002999518600677973, 1.7452663629174578],
        # [0.0029937140862911983, 3.499393327383567],
        # [0.44639011070745355, 3.4804256497254378],
        # [1.6343970709211206, 3.499393327383567],
        # [217.86870468948888, 3.499393327383567],
        # [217.5425454557648, 5.993385824568572],
        # [1.6319503051177793, 5.993385824568572],
        # [0.39847121258699214, 5.993385824568572],
        # [0.002503306626150767, 5.993385824568572],
        # [0.0024972527535430267, 14.299999999999994],
        # [0.44464393306162575, 14.222490053236154],#culprit
        # [1.6899820258808993, 14.222490053236154],
        # [217.01645135159276, 14.299999999999994]])

        return None

    def dmag_limits(self, rmin, rmax, pmax, pmin, Rmax, Rmin, phaseFunc):
        """Limits of delta magnitude as a function of separation

        Limits on dmag vs s JPDF from [Garrett2016]_
        See https://github.com/dgarrett622/FuncComp/blob/master/FuncComp/util.py

        Args:
            rmin (float):
                minimum planet-star distance possible in AU
            rmax (float):
                maximum planet-star distance possible in AU
            pmax (float):
                maximum planet albedo
            pmin (float):
                minimum planet abledo
            Rmax (float):
                maximum planet radius in earthRad
            Rmin (float):
                minimum planet radius in earthRad
            phaseFunc (callable):
                with input in units of rad

        Returns:
            tuple:
                dmag_limit_functions (list):
                    list of lambda functions taking in 's' with units of AU
                lower_limits (list):
                    list of floats representing lower bounds on 's'
                upper_limits (list):
                    list of floats representing upper bounds on 's'

        """

        def betaStarFinder(beta, phaseFunc):
            """From Garrett 2016

            Args:
                beta (float or numpy.ndarray):
                    phase angle in radians
                phaseFunc (callable):
                    with input in units of rad

            Returns:
                numpy.ndarray:
                    phase function values * sin(beta)^2

            """
            return (
                -phaseFunc(np.asarray([beta]) * u.rad, np.asarray([]))
                * np.sin(beta) ** 2.0
            )

        res = minimize_scalar(
            betaStarFinder,
            args=(self.PlanetPhysicalModel.calc_Phi,),
            method="golden",
            tol=1e-4,
            bracket=(0.0, np.pi / 3.0, np.pi),
        )
        # All others show same result for betaStar
        # res2 = minimize_scalar(betaStarFinder,
        #          args=(self.PlanetPhysicalModel.calc_Phi,),
        #          method='Bounded',tol=1e-4, bounds=(0.,np.pi))
        # from scipy.optimize import minimize
        # res3 = minimize(betaStarFinder,np.pi/4.,bounds=[(0.,np.pi)],
        #                 tol=1e-4, args=(self.PlanetPhysicalModel.calc_Phi,))
        betaStar = np.abs(res["x"]) * u.rad  # in rad

        dmag_limit_functions = [
            lambda s: -2.5
            * np.log10(
                pmax
                * (Rmax / rmin).decompose() ** 2.0
                * phaseFunc(np.arcsin((s / rmin).decompose()).value, np.asarray([]))
            ),
            lambda s: -2.5
            * np.log10(
                pmax
                * (Rmax * np.sin(betaStar) / s).decompose() ** 2.0
                * phaseFunc(np.asarray([betaStar.value]), np.asarray([]))
            ),
            lambda s: -2.5
            * np.log10(
                pmax
                * (Rmax / rmax).decompose() ** 2.0
                * phaseFunc(np.arcsin((s / rmax).decompose().value), np.asarray([]))
            ),
            lambda s: -2.5
            * np.log10(
                pmin
                * (Rmin / rmax).decompose() ** 2.0
                * phaseFunc(
                    (np.pi * u.rad - np.arcsin((s / rmax).decompose())).value,
                    np.asarray([]),
                )
            ),
        ]
        lower_limits = [
            0.0 * u.AU,
            rmin * np.sin(betaStar),
            rmax * np.sin(betaStar),
            0.0 * u.AU,
        ]
        upper_limits = [rmin * np.sin(betaStar), rmax * np.sin(betaStar), rmax, rmax]

        return dmag_limit_functions, lower_limits, upper_limits

    def probDetectionIsOfType(
        self, dmag, uncertainty_dmag, separation, uncertainty_s, sub=-2
    ):
        """Calculates the probability a planet is of the given type

        Args:
            comp (completeness object):
                a completeness object with the EVPOC_hs, count_hs, count_pop,
                EVPOCpdf_pop attributes (generated by the subtype completeness module)
            dmag (float):
                the mean dmag to evaluate at
            uncertainty_dmag ():
                the uncertainty in dmag to evaluate over
            separation ():
                the mean separation to evaluate at
            uncertainty_s ():
                the uncertainty in separation to evaluate over
            sub (int):
                planet subtype to use for calculation of int_comp
                -2 - planet population
                -1 - earthLike population
                [i,j] - kopparapu planet subtypes

        Returns:
            tuple:
                prob (float):
                    a float indicating the probability a planet is both from the given
                    sub-population and the instrument probability density function.
                normProbProp (float):
                    Probability normalized by the population density joint probability
                    density function
        """
        # normal probability distribution of the telescope for detected
        # planet separation
        f_sep = norm(separation, uncertainty_s)
        # normal probability distribution of the telescope for detected planet-star
        # delta magnitude
        f_dmag = norm(dmag, uncertainty_dmag)

        if sub == -2:  # whole planet population
            normProbPop = nquad(
                lambda si, dmagi: f_sep.cdf(si)
                * f_dmag.cdf(dmagi)
                * self.EVPOCpdf_pop.ev(si, dmagi),
                ranges=[
                    (
                        separation - 3.0 * uncertainty_s,
                        separation + 3.0 * uncertainty_s,
                    ),
                    (
                        dmag - 3.0 * uncertainty_dmag * dmag,
                        dmag + 3.0 * uncertainty_dmag * dmag,
                    ),
                ],
            )[0]
            prob = normProbPop
        elif sub == -1:  # earthlike subpopulation
            normProbPop = nquad(
                lambda si, dmagi: f_sep.cdf(si)
                * f_dmag.cdf(dmagi)
                * self.EVPOCpdf_earthLike.ev(si, dmagi)
                / self.EVPOCpdf_pop.ev(si, dmagi),
                ranges=[
                    (
                        separation - 3.0 * uncertainty_s,
                        separation + 3.0 * uncertainty_s,
                    ),
                    (
                        dmag - 3.0 * uncertainty_dmag * dmag,
                        dmag + 3.0 * uncertainty_dmag * dmag,
                    ),
                ],
            )[0]
            prob = nquad(
                lambda si, dmagi: f_sep.cdf(si)
                * f_dmag.cdf(dmagi)
                * self.EVPOCpdf_earthLike.ev(si, dmagi),
                ranges=[
                    (
                        separation - 3.0 * uncertainty_s,
                        separation + 3.0 * uncertainty_s,
                    ),
                    (
                        dmag - 3.0 * uncertainty_dmag * dmag,
                        dmag + 3.0 * uncertainty_dmag * dmag,
                    ),
                ],
            )[0]
        else:  # sub is an (i,j) pair
            ii = sub[0]
            j = sub[1]
            # Calculates the probability
            normProbPop = nquad(
                lambda si, dmagi: f_sep.cdf(si)
                * f_dmag.cdf(dmagi)
                * self.EVPOCpdf_hs[ii, j].ev(si, dmagi)
                / self.EVPOCpdf_pop.ev(si, dmagi),
                ranges=[
                    (
                        separation - 3.0 * uncertainty_s,
                        separation + 3.0 * uncertainty_s,
                    ),
                    (
                        dmag - 3.0 * uncertainty_dmag * dmag,
                        dmag + 3.0 * uncertainty_dmag * dmag,
                    ),
                ],
            )[0]

            prob = nquad(
                lambda si, dmagi: f_sep.cdf(si)
                * f_dmag.cdf(dmagi)
                * self.EVPOCpdf_hs[ii, j].ev(si, dmagi),
                ranges=[
                    (
                        separation - 3.0 * uncertainty_s,
                        separation + 3.0 * uncertainty_s,
                    ),
                    (
                        dmag - 3.0 * uncertainty_dmag * dmag,
                        dmag + 3.0 * uncertainty_dmag * dmag,
                    ),
                ],
            )[0]
            # normProb = prob/(comp.count_hs/comp.count_pop)

        return prob, normProbPop
