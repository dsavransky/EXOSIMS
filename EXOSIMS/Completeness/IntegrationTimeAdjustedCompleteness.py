# -*- coding: utf-8 -*-
import numpy as np
import astropy.units as u
import astropy.constants as const
from EXOSIMS.Completeness.SubtypeCompleteness import SubtypeCompleteness
from exodetbox.projectedEllipse import integrationTimeAdjustedCompletness


class IntegrationTimeAdjustedCompleteness(SubtypeCompleteness):
    """Integration time-adjusted Completeness.  See [Keithly2021b]_

    Args:
        Nplanets (int):
            number of planets to simulate in IAC
        specs:
            user specified values

    Attributes:
        Nplanets (int):
            Number of planets for initial completeness Monte Carlo simulation
        classpath (str):
            Path on disk to Brown Completeness
        filename (str):
            Name of file where completeness interpolant is stored
        updates (float nx5 numpy.ndarray):
            Completeness values of successive observations of each star in the
            target list (initialized in gen_update)

    """

    def __init__(self, Nplanets=1e5, **specs):
        # At this point, Nplanets has been popped out of specs, put it back to
        # be picked up by BrownCompletneess init. Orig formulation allowed the
        # upstream init to set its default value of 1e8 and then overwrote it, but
        # that appears to be a null op. TODO: debug this

        specs["Nplanets"] = int(Nplanets)
        SubtypeCompleteness.__init__(self, **specs)

        # Not overloading the completeness_setup from SubtypeCompleteness because
        # we like what it does.  Just adding a few extra steps afterwards.

        # Calculate and create a set of planets
        PPop = self.PlanetPopulation
        self.inc, self.W, self.w = PPop.gen_angles(Nplanets, None)
        self.sma, self.e, self.p, self.Rp = PPop.gen_plan_params(Nplanets)
        self.inc, self.W, self.w = (
            self.inc.to("rad").value,
            self.W.to("rad").value,
            self.w.to("rad").value,
        )
        self.sma = self.sma.to("AU").value

        # Pass in as TL object?
        # starMass #set as default of 1 M_sun
        starMass = const.M_sun
        self.periods = (
            (
                2.0
                * np.pi
                * np.sqrt(
                    (self.sma * u.AU) ** 3.0 / (const.G.to("AU3 / (kg s2)") * starMass)
                )
            )
            .to("year")
            .value
        )

    def comp_calc(
        self, smin, smax, dMag, subpop=-2, tmax=0.0, starMass=const.M_sun, IACbool=False
    ):
        """Calculates completeness for given minimum and maximum separations  and dMag

        Args:
            smin (float numpy.ndarray):
                Minimum separation(s) in AU
            smax (float numpy.ndarray):
                Maximum separation(s) in AU
            dMag (float numpy.ndarray):
                Difference in brightness magnitude
            subpop (int):
                planet subtype to use for calculation of int_comp
                -2 - planet population
                -1 - earthLike population
                (i,j) - kopparapu planet subtypes
            tmax (float):
                the integration time of the observation in days
            starMass (float):
                star mass in units of M_sun
            IACbool (bool):
                Use integration time-adjusted completeness or normal Brown completeness
                If False, tmax does nothing.  Defaults False.

        Returns:
            numpy.ndarray:
                comp, SubtypeCompleteness Completeness values (Brown's method mixed with
                classification) or integration time adjusted completeness
                totalCompleteness_maxIntTimeCorrected

        ..note::

            This method assumes scaling orbits when scaleOrbits == True has already
            occurred for smin, smax, dMag inputs

        """

        if IACbool:
            self.vprint(len(self.sma))
            sma = self.sma
            e = self.e
            W = self.W
            w = self.w
            inc = self.inc
            p = self.p
            Rp = self.Rp

            # If we are assuming a constant star mass
            if starMass.size == 1:
                # Pass in as TL object?
                # starMass #set as default of 1 M_sun
                plotBool = False  # need to remove eventually
                periods = self.periods * np.sqrt(
                    const.M_sun / starMass
                )  # need to pass in

                # inputs
                s_inner = smin
                s_outer = smax
                dmag_upper = dMag
                if dmag_upper > 0.0:
                    # input tmax
                    totalCompleteness_maxIntTimeCorrected = (
                        integrationTimeAdjustedCompletness(
                            sma,
                            e,
                            W,
                            w,
                            inc,
                            p,
                            Rp,
                            starMass,
                            plotBool,
                            periods,
                            s_inner,
                            s_outer,
                            dmag_upper,
                            tmax,
                        )
                    )
                else:
                    totalCompleteness_maxIntTimeCorrected = 0

                return totalCompleteness_maxIntTimeCorrected
            else:  # sim.TargetList.MsEst >= 100
                totalCompleteness_maxIntTimeCorrected = np.zeros(len(starMass))
                # Iterate over each Star Mass Provided and calculate completeness
                # for each one
                for i in np.arange(len(starMass)):
                    # Pass in as TL object?
                    # starMass #set as default of 1 M_sun
                    plotBool = False  # need to remove eventually
                    periods = self.periods * np.sqrt(
                        const.M_sun / starMass[i]
                    )  # need to pass in

                    # inputs
                    s_inner = smin[i]
                    s_outer = smax[i]
                    dmag_upper = dMag[i]
                    if dmag_upper > 0.0:
                        # input tmax
                        totalCompleteness_maxIntTimeCorrected[i] = (
                            integrationTimeAdjustedCompletness(
                                sma,
                                e,
                                W,
                                w,
                                inc,
                                p,
                                Rp,
                                np.ones(len(sma)) * starMass[i],
                                plotBool,
                                periods,
                                s_inner,
                                s_outer,
                                dmag_upper,
                                tmax[i].value,
                            )
                        )

                    else:
                        totalCompleteness_maxIntTimeCorrected[i] = 0

                return totalCompleteness_maxIntTimeCorrected
        else:
            if subpop == -2:
                comp = self.EVPOC_pop(smin, smax, 0.0, dMag)
            elif subpop == -1:
                comp = self.EVPOC_earthlike(smin, smax, 0.0, dMag)
            else:
                comp = self.EVPOC_hs[subpop[0], subpop[1]](smin, smax, 0.0, dMag)
            # remove small values
            comp[comp < 1e-6] = 0.0

            return comp
