from EXOSIMS.SimulatedUniverse.KnownRVPlanetsUniverse import KnownRVPlanetsUniverse
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.time import Time


class KnownRVPlanetsUniverseDeterministic(KnownRVPlanetsUniverse):
    """
    Deterministic simulated universe implementation intended to work with the Known RV planet
    population and target list implementations.  Any two runs give the same universe.

    Args:
    \*\*specs:
        user specified values

    """

    def __init__(self, **specs):
        # this indirectly calls gen_physical_properties(), below
        KnownRVPlanetsUniverse.__init__(self, **specs)
        # set mean anomaly to be at quadrature
        self.M0 = 90.0 * np.ones(self.M0.shape) * u.deg
        # set inclination to 0 -- this over-rides gen_physical_properties() below
        self.I = self.I - self.I
        # re-initialize the positions
        self.init_systems()

    def gen_physical_properties(self, rvPlanetOutputFile="", **specs):
        """
        Generate the planetary systems for the current simulated universe.
        This routine populates arrays of the orbital elements and physical
        characteristics of all planets, and generates indexes that map from
        planet to parent star.

        Parameters are generated in a deterministic fashion by setting
        error/perturbation terms to zero.  So for example, instead of the planet
        mass being determined by the catalog mass plus a random perturbation
        scaled by the error bars in the catalog (as for the ordinary
        KnownRVPlanetsUniverse), the perturbation is set to zero.
        Very few inclinations are known at all, so to get some diversity,
        the inclination is set to 0 degrees for "a" planets, 10 for "b", etc.

        If given, the rvPlanetOutputFile is written with a CSV summary of
        selected planet attributes.
        """

        TL = self.TargetList
        PPop = self.PlanetPopulation

        # TL is a filtered version of PPop (filtered by, e.g., IWA/OWA, and removing NaNs).
        #   planinds is a mapping associating the entries in the attribute-vectors
        #   in self.* (like self.a, the semi-major axis) to planets in PPop
        # If pi = planinds, it is the case that:
        #    PPop.<attr>[pi(j)] corresponds to self.<attr>[j]
        # thus
        #   PPop.hostname[pi(j)] is the star corresponding to self.*[j]
        # In particular:
        #   len(planinds) = self.nPlans, and
        #   max(planinds) < len(PPop.hostname) [ = len(PPop.sma), etc.]
        planinds = np.array([])
        for j, name in enumerate(TL.Name):
            tmp = np.where(PPop.hostname == name)[0]
            planinds = np.hstack((planinds, tmp))
        # save it in case it is useful, in particular, for planet names
        self.planinds = planinds.astype(int)

        # The general approach we use is to force selected errors to zero
        # -- these errors are used to scale random perturbations
        # of planet parameters like SMA.

        # self.a is jittered using smaerr
        PPop.smaerr = np.zeros(PPop.smaerr.shape)
        # self.e is jittered using eccenerr
        PPop.eccenerr = np.zeros(PPop.eccenerr.shape)
        # It can be that selected entries in PPop.eccen were already filled in
        # by random choice, within a prior call to PlanetPop.init().  This happens
        # if the eccen was missing in the votable it was read in from.
        # We patch this injection of randomness here, after the fact.  As of 06/2016,
        # there was one missing eccen, from 55 Cnc-e.
        mask = PPop.allplanetdata["pl_orbeccen"].mask
        PPop.eccen[mask] = 0.01  # any non-random value between 0 and 1
        del mask
        # As it happens, we just need to set eccentricity to zero, irrespective of its
        # actual value in the RV Planets catalog, because that's what the test uses.
        PPop.eccen = 0.0 * PPop.eccen

        ## turmon 10/2016: currently this is just taken as 90 degrees!
        #
        # Set up a variable containing the planetary system orbital phase angle "alpha"
        # Do so via a 0/1 variable to select the phase angle by a pseudo-randomized version
        # of eccentricity (as recorded in the planet catalog), as follows:
        #  if phase_angle_selector[i] == 0:
        #    phase_angle[i] = 63.3 %[degrees] -- brightest point of Lambert phase function
        #  else:
        #    phase_angle[i] = 90 %[degrees] -- Quadrature
        # phase_angle_selector = np.floor((PPop.allplanetdata['pl_orbeccen'].data*33.0) % 2.0)
        # self.phase_angle = np.where(phase_angle_selector, 90.0, 63.3) * (np.pi/180.0)

        # self.I
        # replace Inclination, which is almost wholly unknown, using the map char_to_incl
        char_to_incl = dict(
            a=0.0, b=60.0, c=30.0, d=20.0, e=50.0, f=70.0, g=10.0, h=40.0
        )
        new_I = [
            char_to_incl.get(ch, 0.0) for ch in PPop.allplanetdata["pl_letter"].data
        ]
        PPop.allplanetdata["pl_orbincl"] = np.ma.MaskedArray(new_I)
        PPop.allplanetdata["pl_orbinclerr1"] = np.zeros(
            PPop.allplanetdata["pl_orbinclerr1"].shape
        )

        # used to randomize self.O
        PPop.allplanetdata["pl_orblpererr1"] = np.zeros(
            PPop.allplanetdata["pl_orblpererr1"].shape
        )
        # FIXME: also used for self.O: self.w, which is random from PPop.gen_w()

        # albedo is deterministic in Prototype PlanetPhysicalModel, but it is set there
        # to an Earthlike value.  We want 0.50 because that is what the test values used.
        # This line monkey-patches the function within this class instance to return 0.5.
        self.PlanetPhysicalModel.calc_albedo_from_sma = lambda a: np.array(
            [0.50] * len(a)
        )

        # Prototype PlanetPhysicalModel calculates Rp (planet radius)
        # deterministically from Mp (mass), which is itself deterministic.
        # But note, non-Prototype PlanetPhysicalModels (like FortneyMarleyCahoy)
        # may randomize Rp internally.
        PPop.radiuserr1 = PPop.radiuserr1 * 0.0
        PPop.radiuserr2 = PPop.radiuserr2 * 0.0

        # FIXME:
        # initial position and velocity ((r,v) from self.planet_pos_vel)
        # have a randomized mean anomaly.
        # also: due to this, distance and separation (d,s) are also randomized
        PPop.tpererr = PPop.tpererr * 0.0
        PPop.perioderr = PPop.perioderr * 0.0
        # the time of periastron, stored in PPop.tper, was filled in at random,
        # by __init__, for the following slots.  We fill it in deterministically.
        # Note, we cannot just replace selected values because the whole array
        # PPop.tper is immutable -- as are all astropy Time arrays.
        # 1: the new tper as a numpy array of Julian days -- start with the existing one
        tper_replacement = PPop.tper.value
        # 2: true/false indicating the randomized entries within tper
        tper_randomized = PPop.allplanetdata["pl_orbtper"].mask
        # 3: replace the randomized values (Julian day of 2016.01.01 midnight)
        tper_replacement[tper_randomized] = 2457388.5
        # 4: convert back to a Time
        PPop.tper = Time(tper_replacement, format="jd")

        # generate the planets with the zero-error metadata
        KnownRVPlanetsUniverse.gen_physical_properties(self, **specs)

        # print '*** O', self.O[0:8]
        # print '*** w', self.w[0:8]

        # write a summary to a fixed file
        if rvPlanetOutputFile:
            with open(rvPlanetOutputFile, "w") as fp:
                # header
                fp.write(
                    "# "
                    + ",".join(
                        (
                            "host",
                            "planet",
                            "axis",
                            "masse",
                            "rade",
                            "albedo",
                            "incl",
                            "eccen",
                        )
                    )
                    + "\n"
                )
                # pInx is the planet number in the SimUniverse,
                # pInx2 is the planet number in the original PPop dataset
                for pInx in range(self.nPlans):
                    R_geo = 6378136.0 * u.m  # Earth radius
                    pInx2 = self.planinds[pInx]
                    fp.write(
                        ",".join(
                            (
                                PPop.hostname[pInx2],
                                PPop.allplanetdata["pl_letter"][pInx2],
                                str(self.a[pInx].value),
                                str((self.Mp[pInx] / u.M_earth).decompose().value),
                                str((self.Rp[pInx] / R_geo).decompose().value),
                                str(self.p[pInx]),
                                str(self.I[pInx].value),
                                str(self.e[pInx]),
                            )
                        )
                        + "\n"
                    )
                fp.close()

        return
