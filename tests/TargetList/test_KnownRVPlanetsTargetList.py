#!/usr/bin/env python
#
# Usage:
#   run from shell, or
#   % python <this_file.py>
# You may need to put EXOSIMS in your $PYTHONPATH, e.g.,
#   % PYTHONPATH=/path/to/exomissionsim <this_file.py>

r"""KnownRVPlanetsTargetList module unit tests

Michael Turmon, JPL, May 2016
"""

import sys
import os
import unittest
import warnings
import json
from collections import namedtuple
from EXOSIMS.TargetList.KnownRVPlanetsTargetList import KnownRVPlanetsTargetList
import astropy.units as u
from tests.TestSupport.Info import resource_path
from tests.TestSupport.Utilities import RedirectStreams
from tests.TestSupport.Utilities import load_vo_csvfile
from io import StringIO

# A JSON string containing KnownRVPlanets - from simplest-old.json
# The part we require is the "modules" dictionary.
# EXOSIMS also insists on instruments and starlightsuppression
# being present, so they are included.  The particular values
# within do not seem to be used.
# This string could also have been a JSON file, or a literal Python
# dictionary "specs".  We left it as a literal here because it
# makes the test code more self-contained.
ScriptLiteral = """{
  "scienceInstruments": [
    {
      "name": "imaging-EMCCD",
      "type": "imaging-EMCCD",
      "lam": 565,
      "BW": 0.10,
      "QE": 0.88,
      "CIC": 0.0013,
      "sread": 16,
      "ENF": 1.414,
      "Gem": 500
    }
  ],
  "starlightSuppressionSystems": [
    {
      "name": "internal-imaging-HLC",
      "type": "internal-imaging-HLC",
      "IWA": 0.1,
      "OWA": 0,
      "throughput": 1,
      "contrast": 1,
      "PSF": 1
    }
  ],
  "modules": {
    "PlanetPopulation": "KnownRVPlanets",
    "StarCatalog": " ",
    "OpticalSystem": " ",
    "ZodiacalLight": " ",
    "BackgroundSources": " ",
    "PlanetPhysicalModel": "FortneyMarleyCahoyMix1",
    "Observatory": "WFIRSTObservatory",
    "TimeKeeping": " ",
    "PostProcessing": " ",
    "Completeness": "Completeness",
    "TargetList": "KnownRVPlanetsTargetList",
    "SimulatedUniverse": "KnownRVPlanetsUniverse",
    "SurveySimulation": " ",
    "SurveyEnsemble": " "
  }
}"""

### BELOW IS NOT PRESENTLY USED
# These star-by-star check values are pasted from
# printed output of Matlab test routine
KnownResults = [
    {"name": "Mercury", "dist": -0.093047, "coord": [9.23773452, 0.46008903]},
    {"name": "Venus", "dist": 0.143513, "coord": [7.14057119, -0.09038863]},
]
# convenient holder for the above results
TargetInfo = namedtuple("TargetInfo", ["name", "dist", "coord"])
TargetPointTests = [TargetInfo(**d) for d in KnownResults]
### ABOVE IS NOT PRESENTLY USED

# vo-table fields relevant to stars with exoplanets
# See the "atts_mapping" dictionary within the module under test for
# a reference to which fields are included here (pl_letter is
# the only exception).
# Note: the units here must match the units within the TargetList
# attribute.  st_dist is in parsec and is tagged as such (by scaling
# by u.pc) -- because that's how TargetList does it.
# If TargetList starts tagging an attribute with units,
# the unit must be entered below.
exostar_unit_map = dict(
    pl_hostname=str,
    pl_letter=str,
    ra=lambda x: float(x) * u.deg,
    dec=lambda x: float(x) * u.deg,
    st_spstr=str,
    st_plx=lambda x: float(x) * u.mas,
    st_uj=float,
    st_bj=float,
    st_vj=float,
    st_rc=float,
    st_ic=float,
    st_j=float,
    st_h=float,
    st_k=float,
    st_dist=lambda x: float(x) * u.pc,
    st_bmvj=float,
    st_lum=lambda x: 10 ** float(x),
    st_pmra=lambda x: float(x) * u.mas / u.yr,
    st_pmdec=lambda x: float(x) * u.mas / u.yr,
    st_radv=lambda x: float(x) * u.km / u.s,
)


class TestKnownRVPlanetsTargetListMethods(unittest.TestCase):
    r"""Test TargetList.KnownRVPlanetsTargetList class."""

    dev_null = open(os.devnull, "w")

    def setUp(self):
        # print '[setup] ',
        specs = json.loads(ScriptLiteral)
        with RedirectStreams(stdout=self.dev_null):
            with warnings.catch_warnings():
                # filter out warnings about the votable RVplanets file
                warnings.filterwarnings("ignore", ".*votable")
                self.fixture = KnownRVPlanetsTargetList(**specs)

    def tearDown(self):
        del self.fixture

    def basic_validation(self, tlist):
        r"""Perform basic validation of TargetList.

        Factored out into a separate routine to avoid duplication.
        """
        self.assertEqual(tlist._modtype, "TargetList")
        self.assertEqual(type(tlist._outspec), type({}))
        # check for presence of a couple of class attributes
        self.assertIn("PlanetPopulation", tlist.__dict__)
        self.assertIn("PlanetPhysicalModel", tlist.__dict__)

    # @unittest.skip("Skipping init.")
    def test_init(self):
        r"""Test of initialization and __init__."""
        tlist = self.fixture
        self.basic_validation(tlist)

    # @unittest.skip("Skipping init.")
    def test_init_attributes(self):
        r"""Test of initialization and __init__ -- object attributes.

        Method: Ensure that the attributes special to RVTargetList are
        present and of the right length.
        """
        tlist = self.fixture
        self.basic_validation(tlist)
        # ensure the votable-derived star attributes are present
        #   these are set in __init__
        for att in tlist.atts_mapping:
            self.assertIn(att, tlist.__dict__)
            self.assertEqual(len(tlist.__dict__[att]), tlist.nStars)
        # ensure star attributes are present
        #   these attributes are set in populate_target_list
        for att in ["int_comp", "BC", "L", "MV", "coords"]:
            self.assertIn(att, tlist.__dict__)
            self.assertEqual(len(tlist.__dict__[att]), tlist.nStars)

    @unittest.skip(
        "Skipping stellar attributes check as static values are out of date."
    )
    def test_init_stellar_attributes(self):
        r"""Test of initialization and __init__ -- stellar attributes.

        Method: Comprehensive check of all target-star stored attributes
        against tabulated values, for all stars.
        TODO: Numerical checks on int_comp attribute
        """
        tlist = self.fixture
        self.basic_validation(tlist)
        # Load canned RV exo-star information
        # d_ref is a dictionary, indexed by (stellar) host name,
        # containing lists of dictionaries of star attributes.  For instance:
        #   d_ref['HD 12661']
        # contains a list of two entries (two planets for that star), and
        #   d_ref['HD 12661'][0]['st_plx']
        # is a floating-point parallax for the first planet, or None if the
        # parallax is not known.
        # Because we're checking stellar properties here, the list entries
        # refer to the same host star, so index [0] is the same as index [1],
        # etc.
        comparison_file = os.path.join(resource_path(), "ipac-rv-data-stars.csv")
        d_ref = load_vo_csvfile(comparison_file, exostar_unit_map)
        # loop over all RV host stars, looking for matching stars in the file
        hosts_not_matched = 0
        for n_host in range(len(tlist.Name)):
            # check information for the following host
            host = tlist.Name[n_host]
            if host not in d_ref:
                hosts_not_matched += 1
                continue
            # check all the attributes created in the tlist's __init__
            # the "atts_mapping" dictionary maps:
            #    tlist attribute names => votable attribute names
            for name_att, name_vo in tlist.atts_mapping.items():
                # the EXOSIMS value: tlist.$name_e[n_host]
                val_e = getattr(tlist, name_att)[n_host]
                # the validation value
                val_v = d_ref[host][0][name_vo]
                # if missing data in the validation table, skip -- it's a NaN
                if val_v is None:
                    continue
                assert isinstance(val_e, u.quantity.Quantity) == isinstance(
                    val_v, u.quantity.Quantity
                ), (
                    "Units mismatch in key %s, check atts_mapping dictionary in test code."
                    % name_att
                )
                if isinstance(val_e, u.quantity.Quantity):
                    val_e = val_e.value
                    val_v = val_v.value
                # otherwise, check it
                self.assertEqual(
                    val_e,
                    val_v,
                    msg=(
                        'Difference in star "%s", attribute "%s": %s vs. %s'
                        % (host, name_att, val_e, val_v)
                    ),
                )
            # check ra, dec, which are handled separately
            for name_att in ["ra", "dec"]:
                # the EXOSIMS value
                val_e = getattr(tlist.coords[n_host], name_att)
                # the validation value
                val_v = d_ref[host][0][name_att]
                # allow a small difference in case of truncation issues
                self.assertAlmostEqual(
                    val_e.to(u.deg).value,
                    val_v.to(u.deg).value,
                    msg=(
                        'Difference in star "%s", attribute "%s": %s vs. %s'
                        % (host, name_att, val_e, val_v)
                    ),
                    delta=1e-4,
                )
            # optionally:
            # check tlist.maxintTime, tlist.int_comp
            # noted: tlist.Binary_Cut seems not based in measurements
        # ensure there are not too many un-matched host stars
        self.assertLess(
            hosts_not_matched,
            len(tlist.Name) // 10,
            "Too many stars in TargetList are unmatched in the catalog.",
        )

    # @unittest.skip("Skipping str.")
    def test_str(self):
        r"""Test __str__ method, for full coverage."""
        tlist = self.fixture
        # replace stdout and keep a reference
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        # call __str__ method
        result = tlist.__str__()
        # examine what was printed
        contents = sys.stdout.getvalue()
        self.assertEqual(type(contents), type(""))
        self.assertIn("PlanetPhysicalModel", contents)
        self.assertIn("PlanetPopulation", contents)
        self.assertIn("Completeness", contents)
        sys.stdout.close()
        # it also returns a string, which is not necessary
        self.assertEqual(type(result), type(""))
        # put stdout back
        sys.stdout = original_stdout

    # @unittest.skip("Skipping populate_target_list.")
    def test_populate_target_list(self):
        r"""Test of populate_target_list.

        Method: None.  populate_target_list is called as part of __init__
        and is tested by the __init__ test.
        """
        pass

    # @unittest.skip("Skip filter_target_list")
    def test_filter_target_list(self):
        r"""Test filter_target_list method.

        Method:  The method under test is a pass-through because
        no filters are applied.  It is called as part of __init__,
        so is in principle tested as part of that test.  We do
        ensure the TargetList object dictionary is not altered.
        """
        tlist = self.fixture
        keys = sorted(list(tlist.__dict__))  # makes a copy
        tlist.filter_target_list({})
        self.assertEqual(tlist._modtype, "TargetList")
        # just ensure the same keys are still present
        self.assertListEqual(keys, sorted(list(tlist.__dict__)))

    def test_calc_HZ(self):
        r"""Simple test of calc_HZ method"""
        tlist = self.fixture
        assert tlist.calc_HZ_inner(0) < tlist.calc_HZ_outer(0)


if __name__ == "__main__":
    unittest.main()
