#!/usr/bin/env python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>
# You may need to put EXOSIMS in your $PYTHONPATH, e.g.,
#   % PYTHONPATH=/path/to/exomissionsim <this_file.py>

r"""KnownRVPlanets module unit tests

Michael Turmon, JPL, Apr. 2016
"""

import os
import unittest
from EXOSIMS.PlanetPopulation.KnownRVPlanets import KnownRVPlanets
import numpy as np
import astropy.units as u
import astropy.constants as const
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams
from tests.TestSupport.Utilities import load_vo_csvfile
import copy


# First few entries in Table 6 of Traub et al., JATIS, Jan 2016: detectable RV planets
# Num,Name,AltName,Sep (arcsec),Contrast,HLC-t (days),SPC-t (days),PIAACMC-t (days),V (mag),Mass (Jup),Rad (Jup), Per (days),sma (AU)
# ended up not using this.
significant_planets = [
    (
        1,
        "55_Cnc_d",
        "55 Cnc",
        0.424,
        2.9e-09,
        0.36,
        0.43,
        0.08,
        5.96,
        3.54,
        14.1,
        4909,
        5.47,
    ),
    (
        2,
        "mu_Ara_c",
        "HD 160691",
        0.329,
        3.1e-09,
        0.10,
        0.06,
        0.02,
        5.12,
        1.89,
        14.2,
        4206,
        5.34,
    ),
    (
        3,
        "HD_217107_c",
        "HD 217107",
        0.257,
        3.1e-09,
        0.45,
        0.24,
        0.09,
        6.17,
        2.62,
        14.2,
        4270,
        5.33,
    ),
    (
        4,
        "HD_114613_b",
        "HD 114613",
        0.245,
        2.6e-09,
        0.09,
        0.05,
        0.02,
        4.85,
        0.51,
        12.9,
        3827,
        5.31,
    ),
    (
        5,
        "47_UMa_c",
        "47 UMa",
        0.243,
        5.9e-09,
        0.04,
        0.02,
        0.01,
        5.03,
        0.55,
        13.2,
        2391,
        3.57,
    ),
    (
        6,
        "HD_190360_b",
        "HD 190360",
        0.239,
        5.6e-09,
        0.10,
        0.05,
        0.02,
        5.73,
        1.54,
        14.2,
        2915,
        3.97,
    ),
    (
        7,
        "HD_154345_b",
        "HD 154345",
        0.216,
        4.9e-09,
        0.59,
        0.30,
        0.10,
        6.76,
        0.96,
        14.1,
        3342,
        4.21,
    ),
    (
        8,
        "HD_134987_c",
        "HD 134987",
        0.212,
        2.5e-09,
        1.16,
        0.57,
        0.18,
        6.47,
        0.80,
        13.9,
        5000,
        5.83,
    ),
    (
        9,
        "HD_87883_b",
        "HD 87883",
        0.188,
        6.9e-09,
        1.45,
        0.57,
        0.19,
        7.57,
        1.76,
        14.2,
        2754,
        3.58,
    ),
    (
        10,
        "upsilon_And_d",
        "ups And",
        0.179,
        1.3e-08,
        0.01,
        0.00,
        0.00,
        4.10,
        4.12,
        14.0,
        1278,
        2.52,
    ),
    (
        11,
        "HD_39091_b",
        "HD 39091",
        0.175,
        5.3e-09,
        0.13,
        0.06,
        0.02,
        5.65,
        10.09,
        11.7,
        2151,
        3.35,
    ),
    (
        12,
        "beta_Gem_b",
        "HD 62509",
        0.162,
        2.8e-08,
        0.00,
        0.00,
        0.00,
        1.15,
        2.76,
        14.2,
        590,
        1.76,
    ),
    (
        13,
        "14_Her_b",
        "14 Her",
        0.159,
        9.5e-09,
        0.25,
        0.12,
        0.03,
        6.61,
        5.22,
        13.6,
        1773,
        2.93,
    ),
    (
        14,
        "47_UMa_b",
        "47 UMa",
        0.143,
        2.0e-08,
        0.02,
        0.01,
        0.00,
        5.03,
        2.55,
        14.2,
        1078,
        2.10,
    ),
    (
        15,
        "gamma_Cep_b",
        "gam Cep",
        0.134,
        2.2e-08,
        0.00,
        0.00,
        0.00,
        3.21,
        1.52,
        14.2,
        906,
        1.98,
    ),
]

# field mapping (VOtable -> python) for quantities relevant to exoplanets
# loosely, every entry here maps a field from a CSV file (which itself is
# a string) to the string, number or astropy Quantity that we want.
# see also load_vo_csvfile for how this functions.
exoplanet_unit_map = dict(
    pl_hostname=str,
    pl_letter=str,
    pl_orbsmax=lambda x: float(x) * u.au,
    pl_orbsmaxerr1=lambda x: float(x) * u.au,
    pl_orbeccen=float,
    pl_orbeccenerr1=float,
    pl_bmasse=lambda x: (float(x) * const.M_earth),
    pl_bmasseerr1=lambda x: (float(x) * const.M_earth),
    pl_bmassprov=lambda x: (x == "Msini"),
)


class TestKnownRVPlanetsMethods(unittest.TestCase):
    r"""Test PlanetPopulation KnownRVPlanets class."""

    def setUp(self):
        # allow the chatter on stdout during object creation to be suppressed
        self.dev_null = open(os.devnull, "w")
        # The chain of init methods for this object is:
        #    KnownRVPlanets.__init__ ->
        #    KeplerLike1.__init__ ->
        #    Prototype/PlanetPopulation.__init__
        # The last init then finally does something like this:
        #    PlanPhys = get_module(specs['modules']['PlanetPhysicalModel'], 'PlanetPhysicalModel')
        #    self.PlanetPhysicalModel = PlanPhys(**specs)
        # To make this work, when we __init__ the module under test here,
        # we need to supply a "specs" that has a proper value for:
        #    [modules][PlanetPhysicalModel]
        # so that get_module can find the value.
        # Like so:
        specs = {}
        specs["modules"] = {}
        specs["modules"][
            "PlanetPhysicalModel"
        ] = " "  # so the Prototype for PhysMod will be used
        self.specs = copy.deepcopy(specs)

        # the with clause allows the chatter on stdout/stderr during
        # object creation to be suppressed
        with RedirectStreams(stdout=self.dev_null, stderr=self.dev_null):
            self.fixture = KnownRVPlanets(**specs)

    def tearDown(self):
        self.dev_null.close()
        del self.fixture

    def validate_planet_population(self, plan_pop):
        r"""Consolidate simple validation of PlanetPopulation attributes in one place."""

        self.assertEqual(plan_pop._modtype, "PlanetPopulation")
        self.assertIsInstance(plan_pop._outspec, dict)
        # check for presence of one class attribute
        self.assertIn("sma", plan_pop.__dict__)

    def test_init_trivial(self):
        r"""Test of initialization and __init__ -- trivial setup/teardown test."""
        plan_pop = self.fixture
        # ensure basic module attributes are set up
        self.validate_planet_population(plan_pop)
        # optionally, print some attributes
        if False:
            for (key, value) in plan_pop.__dict__.items():
                if key == "allplanetdata":
                    continue
                if key == "table":
                    continue
                print(key, "==>", value)

    def test_init_file_no_file(self):
        r"""Test __init__ file handling -- various non-existent input files."""
        bad_files = [
            "/dev/null",
            "/tmp/this/file/is/not/there.json",
            "/tmp/this_file_is_not_there.json",
        ]
        for bad_file in bad_files:
            with self.assertRaises(IOError):
                plan_pop = KnownRVPlanets(rvplanetfilepath=bad_file, **self.specs)

    @unittest.skip(
        "Traub et al. planet data is now out of date so it does not make sense to compare against it."
    )
    def test_init_traub_etal(self):
        r"""Test of initialization and __init__ -- compare values to Traub et al., JATIS, Mar. 2016.

        Method: Loop over the values from Traub et al., Table 6, as entered above, and find the
        corresponding planet in the EXOSIMS data.  Compare SMA (semi-major axis) and planetary mass.
        The relative error in SMA is presently allowed to be about 10%, and in mass, about 25%.
        This turned out to be an ineffective test, so it is disabled.
        """
        plan_pop = self.fixture
        # ensure basic module attributes are set up
        self.validate_planet_population(plan_pop)

        for planet_tuple in significant_planets:
            (_, name, name_alt, _, _, _, _, _, v, mass, rad, period, sma) = planet_tuple
            # units
            #  (v is a dimensionless magnitude)
            mass = mass * const.M_jup
            rad = rad * const.R_jup
            period = period * u.day
            sma = sma * u.au

            # match planets in EXOSIMS to planet in the Traub et al list
            # 1: match host star name
            index = np.array([name_alt in name1 for name1 in plan_pop.hostname])
            # 2: within this, match particular planet
            index2 = np.argmin(np.abs(plan_pop.sma[index] - sma))

            # compute relative differences
            delta_sma = ((plan_pop.sma[index][index2] - sma) / sma).decompose().value
            delta_mass = (
                ((plan_pop.mass[index][index2] - mass) / mass).decompose().value
            )

            # proportional difference in SMA
            self.assertAlmostEqual(np.abs(delta_sma), 0.0, delta=0.08)
            # proportional difference in mass
            self.assertAlmostEqual(np.abs(delta_mass), 0.0, delta=0.25)

    @unittest.skip(
        "This is comparing against a static snapshot of IPAC, which is constantly updating."
    )
    def test_init_ipac_compare(self):
        r"""Test of initialization and __init__ -- compare values to independently-extracted values.

        Method: Loop over the extracted values from EXOSIMS, and find the corresponding
        planet in the online catalog, as loaded in the first lines in this code.
        Compare values for all important parameters.  The values used by subsequent code
        are (from SimulatedUniverse/KnownRVPlanetsUniverse):
        *  (sma, smaerr) = semi-major axis & error
        *  (eccen, eccenerr) = eccentricity & error
        also present here:
           plan_pop.hostname (string)
        and, as well,
           plan_pop.allplanetdata[pl_orbitincl] = orbital inclination
        *  plan_pop.mass = mass of planet
           plan_pop.allplanetdata['pl_orblper'] = longitude of periapsis
        We check the starred values.
        """

        plan_pop = self.fixture
        # ensure basic module attributes are set up
        self.validate_planet_population(plan_pop)

        # Load canned RV planet information
        # d_ref is a dictionary, indexed by (stellar) host name,
        # containing lists of dictionaries of planet attributes.  For instance:
        #   d_ref['HD 12661']
        # contains a list of two entries (two planets), and
        #   d_ref['HD 12661'][0]['pl_orbeccen']
        # is a floating-point eccentricity for the first planet, or None if the
        # eccentricity is not known.
        comparison_file = os.path.join(resource_path(), "ipac-rv-data-planets.csv")
        d_ref = load_vo_csvfile(comparison_file, exoplanet_unit_map)
        # d_ref = load_rv_planet_comparison()

        # loop over all RV planets, looking for matches
        hosts_not_matched = 0
        for n_planet in range(len(plan_pop.hostname)):
            # specific "plan_pop" information we will check for this planet
            host_1 = plan_pop.hostname[n_planet]
            sma_1 = plan_pop.sma[n_planet]
            sma_err_1 = plan_pop.smaerr[n_planet]
            ecc_1 = plan_pop.eccen[n_planet]
            ecc_err_1 = plan_pop.eccenerr[n_planet]
            mass_1 = plan_pop.mass[n_planet]
            # reference information:
            # it's a list, because there could be several planets around that host
            self.assertIn(host_1, d_ref)
            info_refs = d_ref[host_1]
            # pick out the right planet using agreement of SMA
            info_ref = [
                d
                for d in info_refs
                if (
                    d["pl_orbsmax"] is not None
                    and np.abs(d["pl_orbsmax"] - sma_1).to(u.au).value < 1e-3
                )
            ]
            # zero matches is possible, because plan_pop.__init__ fills in a few
            # SMA values on its own if a planet is detected but SMA is not recorded by IPAC.
            if len(info_ref) != 1:
                hosts_not_matched += (
                    1  # count them: we don't allow too many unmatched planets
                )
                continue
            # assert that there was not a double-match -- should not happen
            self.assertEqual(len(info_ref), 1, msg=("Multiple matches for " + host_1))
            # take the (unique) matching object
            info_ref = info_ref[0]
            # check info_ref attributes (i.e., our reference values) against plan_pop
            #   note, all these checks are very tight in tolerance, because any difference
            #   should only arise due to float -> ascii -> float conversions
            # 1: check SMA, use relative error
            if info_ref["pl_orbsmax"] is not None:
                delta = ((info_ref["pl_orbsmax"] - sma_1) / sma_1).decompose().value
                self.assertAlmostEqual(
                    delta, 0.0, msg=("SMA difference in %s" % host_1), delta=1e-6
                )
            # 2: check SMA error, use absolute error in AU
            if info_ref["pl_orbsmaxerr1"] is not None:
                # the error can be small, so no relative error
                delta = (info_ref["pl_orbsmaxerr1"] - sma_err_1).to(u.au).value
                self.assertAlmostEqual(
                    delta, 0.0, msg=("SMA_error difference in %s" % host_1), delta=1e-6
                )
            # 3: check eccentricity, just a float
            if info_ref["pl_orbeccen"] is not None:
                self.assertAlmostEqual(
                    info_ref["pl_orbeccen"],
                    ecc_1,
                    msg=("eccentricity difference in %s" % host_1),
                    delta=1e-6,
                )
            # 4: check eccentricity error, also a float
            if info_ref["pl_orbeccenerr1"] is not None:
                self.assertAlmostEqual(
                    info_ref["pl_orbeccenerr1"],
                    ecc_err_1,
                    msg=("eccen_error difference in %s" % host_1),
                    delta=1e-6,
                )
            # 5: check mass, use relative error
            if info_ref["pl_bmasse"] is not None:
                delta = ((info_ref["pl_bmasse"] - mass_1) / mass_1).decompose().value
                self.assertAlmostEqual(
                    delta, 0.0, msg=("Mass difference in %s" % host_1), delta=1e-6
                )
        # ensure there are not too many un-matched host stars
        self.assertLess(
            hosts_not_matched,
            len(plan_pop.hostname) // 10,
            "Too many stars in PlanetPopulation are unmatched in the catalog.",
        )

    def test_gen_mass(self):
        r"""Test gen_mass method.

        Approach: Ensures the output is set, of the correct type, length, and units.
        Check the range of the returned values.  Check that, for this power law, there
        are more small than large masses (for n large).
        """

        plan_pop = self.fixture
        n = 10000
        # call the routine
        masses = plan_pop.gen_mass(n)
        # check the type
        self.assertEqual(type(masses), type(1.0 * u.kg))
        # crude check on the shape (more small than large for this power law)
        midpoint = np.mean(plan_pop.Mprange)
        self.assertGreater(
            np.count_nonzero(masses < midpoint), np.count_nonzero(masses > midpoint)
        )
        # test some illegal "n" values
        n_list_bad = [-1, "100", 22.5]
        for n in n_list_bad:
            with self.assertRaises(AssertionError):
                masses = plan_pop.gen_mass(n)

    def test_gen_radius(self):
        r"""Test gen_radius method.

        Approach: Ensures the output is set, of the correct type, length, and units.
        Check that, for this power law, there
        are more small than large masses (for n large).
        """

        plan_pop = self.fixture
        n = 10000
        # call the routine
        radii = plan_pop.gen_radius(n)
        # check the type
        self.assertEqual(type(radii), type(1.0 * u.km))
        # ensure the units are length
        self.assertEqual((radii / u.km).decompose().unit, u.dimensionless_unscaled)
        # radius > 0
        self.assertTrue(np.all(radii.value >= 0))
        # crude check on the shape (masses are a power law, so radii will also be,
        # so we require more small than large radii)
        midpoint = (np.min(radii) + np.max(radii)) * 0.5
        self.assertGreater(
            np.count_nonzero(radii < midpoint), np.count_nonzero(radii > midpoint)
        )
        # test some illegal "n" values
        # Note: as long as we're checking this, -1 should be illegal, but is passed thru
        n_list_bad = [-1, "100", 22.5]
        for n in n_list_bad:
            with self.assertRaises(AssertionError):
                radii = plan_pop.gen_radius(n)


if __name__ == "__main__":
    unittest.main()
